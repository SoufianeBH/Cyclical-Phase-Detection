# losses.py  — minimal, modular, MICCAI-clean
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric

class BalancedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """Balances pos/neg per-batch by averaging the available parts."""
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_0 = torch.masked_select(input, target == 0)
        input_1 = torch.masked_select(input, target != 0)
        target_0 = torch.masked_select(target, target == 0)
        target_1 = torch.masked_select(target, target != 0)

        loss_0 = super().forward(input_0, target_0) if target_0.numel() else torch.tensor(0., device=input.device)
        loss_1 = super().forward(input_1, target_1) if target_1.numel() else torch.tensor(0., device=input.device)
        n_parts = int(target_0.numel() > 0) + int(target_1.numel() > 0)
        return (loss_0 + loss_1) / max(n_parts, 1)

class TemporalEMDLoss(nn.Module):
    """
    1D EMD along time (discrete) between predicted prob curve p_pred[B,T] and labels y[B,T].
    Uses CDF L1: mean_t |cumsum(p) - cumsum(y)|.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, p_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c1 = torch.cumsum(p_pred, dim=1)
        c2 = torch.cumsum(y, dim=1)
        emd = torch.mean(torch.abs(c1 - c2), dim=1)
        return emd.mean() if self.reduction == "mean" else emd

class PhaseLoss(nn.Module):
    """
    Compare predicted & target phase vectors (cos,sin) of shape [B, T, 2].
    mode='mse': MSE on vectors;
    mode='cosine': 1 - cosine_similarity.
    """
    def __init__(self, mode: str = "mse", eps: float = 1e-8):
        super().__init__()
        assert mode in ("mse", "cosine")
        self.mode = mode
        self.eps = eps

    def forward(self, pred_phase: torch.Tensor, tgt_phase: torch.Tensor) -> torch.Tensor:
        if self.mode == "mse":
            return F.mse_loss(pred_phase, tgt_phase)
        # cosine mode
        # flatten over B*T, compute cosine similarity per vector, then 1 - mean(sim)
        p = pred_phase.reshape(-1, 2)
        t = tgt_phase.reshape(-1, 2)
        p = p / (p.norm(dim=1, keepdim=True) + self.eps)
        t = t / (t.norm(dim=1, keepdim=True) + self.eps)
        cos = (p * t).sum(dim=1)  # [B*T]
        return 1.0 - cos.mean()

class TemporalWeightedMetrics(nn.Module):
    """
    Temporal metrics using nearest neighbor assignment without thresholds.
    Each predicted frame is matched to its nearest ground truth ED frame.
    """
    
    def __init__(self, dt=0.0625, max_distance=None):
        """
        Args:
            dt: Time step between frames in seconds
            max_distance: Maximum distance (in frames) to consider for matching.
                         If None, all distances are considered.
        """
        super().__init__()
        self.dt = dt
        self.max_distance = max_distance
        
        # Accumulate weighted metrics
        self.add_state("weighted_precision_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("precision_weight_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        
        self.add_state("weighted_recall_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("recall_weight_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        
        self.add_state("temporal_distance_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_assignments", default=torch.tensor(0.0), dist_reduce_fx="sum")
        
        self.add_state("n_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update metric states using nearest neighbor assignment.
        
        Args:
            preds: Predicted logits [batch_size, sequence_length]
            target: Binary ground truth [batch_size, sequence_length]
        """
        # Convert predictions to probabilities
        pred_probs = torch.sigmoid(preds)
        
        batch_size = pred_probs.shape[0]
        
        for b in range(batch_size):
            pred_b = pred_probs[b]  # [T]
            target_b = target[b]     # [T] binary
            
            # Find ED frame indices
            ed_indices = torch.where(target_b > 0.5)[0]
            
            if len(ed_indices) == 0:
                # No ground truth ED frames - skip this sample
                continue
            
            # Calculate distance from each frame to nearest ED frame
            frame_indices = torch.arange(len(pred_b), device=pred_b.device)
            
            # Compute distance matrix [T, n_ed]
            distances = torch.abs(frame_indices.unsqueeze(1) - ed_indices.unsqueeze(0))
            
            # Find nearest ED frame for each predicted frame
            nearest_distances, nearest_ed_idx = torch.min(distances, dim=1)  # [T]
            
            # Apply max distance constraint if specified
            if self.max_distance is not None:
                valid_mask = nearest_distances <= self.max_distance
                pred_b = pred_b * valid_mask.float()
            

            distance_weights = torch.exp(-nearest_distances.float() / 10.0) 
            
            # Precision: How well do predictions align with ground truth?

            precision_contribution = pred_b * distance_weights
            self.weighted_precision_sum += precision_contribution.sum()
            self.precision_weight_sum += pred_b.sum()
            
            # Recall: How well are ground truth frames covered?
            for ed_idx in ed_indices:
                nearby_mask = torch.abs(frame_indices - ed_idx) <= (self.max_distance or len(pred_b))
                nearby_preds = pred_b * nearby_mask.float()
                nearby_distances = torch.abs(frame_indices - ed_idx) * nearby_mask.float()
                nearby_weights = torch.exp(-nearby_distances.float() / 10.0) * nearby_mask.float()

                weighted_nearby = nearby_preds * nearby_weights
                best_pred_value = weighted_nearby.max()
                
                self.weighted_recall_sum += best_pred_value
                self.recall_weight_sum += 1.0

            weighted_distances = nearest_distances.float() * pred_b
            if pred_b.sum() > 0:
                self.temporal_distance_sum += weighted_distances.sum()
                self.n_assignments += pred_b.sum()
            
            self.n_samples += 1

    def compute(self):
        """Compute final metrics"""
        eps = 1e-8
        
        # Weighted precision
        precision = self.weighted_precision_sum / (self.precision_weight_sum + eps)
        
        # Weighted recall
        recall = self.weighted_recall_sum / (self.recall_weight_sum + eps)
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        
        # Average temporal distance (in seconds)
        avg_distance_frames = self.temporal_distance_sum / (self.n_assignments + eps)
        avg_distance_seconds = avg_distance_frames * self.dt
        
        # Temporal harmonic mean
        distance_penalty = torch.exp(-avg_distance_frames / 10.0)
        temporal_harmonic_mean = f1 * distance_penalty
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_temporal_distance': avg_distance_seconds,
            'avg_distance_frames': avg_distance_frames,
            'temporal_harmonic_mean': temporal_harmonic_mean,
            'n_samples': self.n_samples
        }

# combine the losses with weights that i specify in the config

@dataclass
class LossWeights:
    bce: float = 1.0
    emd: float = 0.0
    phase: float = 0.0
    normalize: bool = False  # if True, normalize positive weights to sum to 1


class CombinedLoss(nn.Module):
    """
    Combine BCE-with-logits (balanced), Temporal EMD, and Phase loss with weights from config.

    Expected inputs to forward():
        preds: {
            "logits": Tensor [B, T]                # for BCE & EMD (we'll sigmoid inside)
            "phase":  Tensor [B, T, 2] (optional)  # for PhaseLoss if weight > 0
        }
        targets: {
            "labels": Tensor [B, T]                # binary {0,1}
            "phase":  Tensor [B, T, 2] (optional)  # for PhaseLoss if weight > 0
        }
    """
    def __init__(
        self,
        weights: LossWeights = LossWeights(),
        phase_mode: str = "mse",     # or "cosine"
        emd_reduction: str = "mean", # usually "mean"
        pos_weight: Optional[torch.Tensor] = None,  # for BCE if you want class weighting
    ):
        super().__init__()
        self.weights = weights

        # Components
        self.bce = BalancedBCEWithLogitsLoss(pos_weight=pos_weight)
        self.emd = TemporalEMDLoss(reduction=emd_reduction)
        self.phase = PhaseLoss(mode=phase_mode)

    @staticmethod
    def _normalize_weights(w: LossWeights) -> LossWeights:
        if not w.normalize:
            return w
        parts = torch.tensor([max(w.bce, 0.0), max(w.emd, 0.0), max(w.phase, 0.0)], dtype=torch.float32)
        s = float(parts.sum().item())
        if s <= 0:
            return w  # avoid divide-by-zero; keep as-is
        bce, emd, ph = [float(x / s) for x in parts]
        return LossWeights(bce=bce, emd=emd, phase=ph, normalize=True)

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Normalize weights (optionally)
        # W = self._normalize_weights(self.weights)
        W = self.weights

        logits = preds["logits"]
        labels = targets["labels"]

        # Per-loss terms
        loss_bce = self.bce(logits, labels)

        # EMD uses probabilities
        probs = torch.sigmoid(logits)
        loss_emd = self.emd(probs, labels)

        # Phase (optional)
        loss_phase = torch.tensor(0.0, device=logits.device)
        if W.phase > 0:
            if "phase" not in preds or "phase" not in targets:
                raise KeyError("Phase loss weight > 0 but 'phase' key missing in preds/targets.")
            loss_phase = self.phase(preds["phase"], targets["phase"])

        # Weighted sum
        total = W.bce * loss_bce + W.emd * loss_emd + W.phase * loss_phase

        return {
            "loss": total,
            "loss_bce": loss_bce.detach(),
            "loss_emd": loss_emd.detach(),
            "loss_phase": loss_phase.detach(),
            "w_bce": torch.tensor(W.bce, device=logits.device),
            "w_emd": torch.tensor(W.emd, device=logits.device),
            "w_phase": torch.tensor(W.phase, device=logits.device),
        }

def build_combined_loss_from_config(cfg: Dict) -> CombinedLoss:
    """
    cfg structure (example):
    {
        "LOSSES": {
            "weights": {"bce": 1.0, "emd": 0.5, "phase": 0.2, "normalize": false},
            "phase_mode": "mse",        # or "cosine"
            "emd_reduction": "mean",    # usually "mean"
            "pos_weight": 1.0           # optional scalar or list per-class (here 1D since binary)
        }
    }
    """
    lcfg = cfg.get("LOSSES", {})
    wcfg = lcfg.get("weights", {})
    weights = LossWeights(
        bce=float(wcfg.get("bce", 1.0)),
        emd=float(wcfg.get("emd", 0.0)),
        phase=float(wcfg.get("phase", 0.0)),
        normalize=bool(wcfg.get("normalize", False)),
    )

    pos_weight = lcfg.get("pos_weight", None)
    if pos_weight is not None:
        if isinstance(pos_weight, (int, float)):
            pos_weight = torch.tensor([float(pos_weight)])
        elif isinstance(pos_weight, (list, tuple)):
            pos_weight = torch.tensor([float(pos_weight[0])])
        elif not isinstance(pos_weight, torch.Tensor):
            raise TypeError("pos_weight must be a number, list/tuple, or torch.Tensor")

    return CombinedLoss(
        weights=weights,
        phase_mode=str(lcfg.get("phase_mode", "mse")),
        emd_reduction=str(lcfg.get("emd_reduction", "mean")),
        pos_weight=pos_weight,
    )