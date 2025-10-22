# import necessary libraries
import einops
import math
import sys
import torch.nn.functional as F
import torchcde
import encoder as enc

from rotary_embedding_torch import RotaryEmbedding
from torch.nn import *
from torchdiffeq import odeint
from vit_pytorch.simple_vit_1d import *
from losses import *
import encoder as enc
from losses import CombinedLoss, LossWeights


class RNNProbe(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_phase = config.get("use_phase", False)
        
        if config["layer"] == "GRU":
            layer = nn.GRU
        elif config["layer"] == "LSTM":
            layer = nn.LSTM

        self.recurrent_layer = layer(
            self.config["n_channels_in"],
            self.config["n_channels"],
            self.config["n_layers"],
            batch_first=True,
            dropout=self.config["dropout"],
            bidirectional=self.config["bidirectional"]
        )

        hidden_size = self.config["n_channels"] * (2 if self.config["bidirectional"] else 1)
        
        self.label_head = nn.Linear(hidden_size, self.config["n_classes"])
        
        if self.use_phase:
            self.phase_head = nn.Linear(hidden_size, 2)  # Predict cos/sin

    def forward(self, x):
        # x: [B, C, T] -> [B, T, C]
        x = einops.rearrange(x, "B C Z -> B Z C")
        
        x, _ = self.recurrent_layer(x)
        
        # Generate predictions
        pred_labels = self.label_head(x)  # [B, T, n_classes]
        pred_labels = einops.rearrange(pred_labels, "B Z C -> B C Z")
        
        if self.use_phase:
            pred_phase = self.phase_head(x)  # [B, T, 2] for cos/sin
            return pred_labels.squeeze(1), pred_phase
        else:
            return pred_labels.squeeze(1), None

class FullModel(nn.Module):
    """
    Paper-style model wrapper with a clear I/O contract.

    Inputs
    ------
    x : Tensor[B, C, T, H, W]

    Returns
    -------
    out : dict with
        - "logits": Tensor[B, T]           # raw logits for BCEWithLogits
        - "phase":  Optional[Tensor[B, T, 2]]  # [sin(theta), cos(theta)] if enabled, else None
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.frame_encoder = self.str_to_attr(config["frame_encoder"]["name"])(config["frame_encoder"])

        self.config["classifier"]["n_channels_in"] = config["frame_encoder"]["n_channels_out"]
        self.classifier = self.str_to_attr(config["classifier"]["name"])(config["classifier"])

        loss_cfg = config["loss"].copy()
        weights_cfg = loss_cfg.pop("weights", {})

        if isinstance(weights_cfg, dict):
            weights = LossWeights(
                bce=float(weights_cfg.get("bce", 1.0)),
                emd=float(weights_cfg.get("emd", 0.0)),
                phase=float(weights_cfg.get("phase", 0.0)),
                normalize=bool(weights_cfg.get("normalize", False)),
            )
        else:
            weights = weights_cfg 

        self.loss = CombinedLoss(
            weights=weights,
            phase_mode=str(loss_cfg.get("phase_mode", "mse")),
            emd_reduction=str(loss_cfg.get("emd_reduction", "mean")),
            pos_weight=None, 
        )


    def forward(self, x):
        x = self.frame_encoder(x)
        x = torch.mean(x, dim=(2, 3))
        pred_labels, pred_phase = self.classifier(x)
        return {"logits": pred_labels, "phase": pred_phase}

    @staticmethod
    def str_to_attr(name):

        if hasattr(sys.modules[__name__], name):
            return getattr(sys.modules[__name__], name)

        if hasattr(enc, name):
            return getattr(enc, name)
        raise AttributeError(f"{name} not found in model.py or encoder.py")
