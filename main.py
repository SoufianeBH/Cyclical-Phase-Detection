# main.py — minimal, Lightning-free runner
import argparse, json, os, time, types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from dataloader import H5DataModule, H5Dataset  
from model import FullModel                      
import plot as plot_mod    

# ----------------- small utils -----------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_device_numpy_batch(batch, device):
    # H5DataModule gives numpy; convert the keys we need
    out = {
        "data":   torch.from_numpy(batch["data"]).float().to(device),
        "ref":    torch.from_numpy(batch["ref"]).float().to(device),
        "frames": torch.from_numpy(batch["frames"]).float().to(device),
    }
    if "phase" in batch:
        out["phase"] = torch.from_numpy(batch["phase"]).float().to(device)
    return out

def next_batch(loader, it_state):
    """Get next batch from a MultiThreadedAugmenter (or any iterable).
    If exhausted, try loader.restart() and continue."""
    it = it_state.get("it")
    if it is None:
        it = iter(loader)
        it_state["it"] = it
    try:
        return next(it)
    except StopIteration:

        try:
            loader.restart()
        except Exception:
            pass
        it = iter(loader)
        it_state["it"] = it
        return next(it)

class TBShim:
    """Tiny adapter so plot.py keeps working (expects self.logger.experiment & self.current_epoch)."""
    def __init__(self, writer: SummaryWriter, epoch: int):
        self.logger = types.SimpleNamespace(experiment=writer)
        self.current_epoch = epoch

def forward_and_loss(model: nn.Module, batch: dict) -> dict:
    """
    Uses model.loss exactly as defined inside your FullModel (BCE/EMD/Phase mixing).
    FullModel.forward returns {"logits": BxT, "phase": BxTx2 or None}.
    FullModel.loss expects that dict plus target dict.
    """
    x = batch["data"]                      
    preds = model(x)                       
    targets = {"labels": batch["ref"]}
    if "phase" in batch:
        targets["phase"] = batch["phase"]
    loss_dict = model.loss(preds, targets)
    return {"preds": preds, "losses": loss_dict}

@torch.no_grad()
def validate_one_epoch(model, val_loader, device, epoch, writer, cfgM, val_steps: int = 10):
    model.eval()
    meters = {"total": 0.0, "bce": 0.0, "emd": 0.0, "phase": 0.0}
    it_state = {}

    for step in range(val_steps):
        batch_np = next_batch(val_loader, it_state)        # <- robust
        batch = to_device_numpy_batch(batch_np, device)

        out = forward_and_loss(model, batch)
        total = out["losses"]["loss"]
        meters["total"] += float(total.detach().cpu())
        meters["bce"]   += float(out["losses"]["loss_bce"].detach().cpu())
        meters["emd"]   += float(out["losses"]["loss_emd"].detach().cpu())
        meters["phase"] += float(out["losses"]["loss_phase"].detach().cpu())

        if step == 0 and (epoch % 20 == 0 or epoch == 0):
            shim = TBShim(writer, epoch)
            try:
                plot_mod.plot_outputs(shim, batch, out["preds"]["logits"])
            except Exception as e:
                print(f"[plot] skipped: {e}")

    for k in meters:
        meters[k] /= max(1, val_steps)
        writer.add_scalar(f"val/{k}", meters[k], epoch)
    return meters


def train_one_epoch(model, train_loader, optimizer, device, epoch, writer, steps_per_epoch: int = 200):
    model.train()
    meters = {"total": 0.0}
    it_state = {}  # iterator state for next_batch

    for step in range(steps_per_epoch):
        batch_np = next_batch(train_loader, it_state)
        batch = to_device_numpy_batch(batch_np, device)

        out = forward_and_loss(model, batch)
        loss = out["losses"]["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        meters["total"] += float(loss.detach().cpu())

        if step == 0 and (epoch % 20 == 0 or epoch == 0):
            # lightweight plotting
            shim = TBShim(writer, epoch)
            try:
                plot_mod.plot_inputs(shim, batch)
                plot_mod.plot_outputs(shim, batch, out["preds"]["logits"])
            except Exception as e:
                print(f"[plot] skipped: {e}")

    meters["total"] /= max(1, steps_per_epoch)
    writer.add_scalar("train/total", meters["total"], epoch)
    return meters

@torch.no_grad()
def test_sliding_window(model, x_np, seq_len: int, device, batch_size: int = 8):
    """
    x_np: numpy or tensor of shape [1, C, H, W, T].
    Returns np.ndarray[T] of averaged predictions.
    """
    if isinstance(x_np, np.ndarray):
        x = torch.from_numpy(x_np)
    else:
        x = x_np
    x = x.to(device)

    _, C, H, W, T = x.shape
    window = min(seq_len, T)
    hop = max(1, window // 8)
    n_windows = ((T - window) // hop) + 1

    overlap_counts = torch.zeros(T, device=device)
    full_preds = torch.zeros(T, device=device)

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        cur_bs = end - start
        batch_windows = torch.zeros((cur_bs, C, H, W, window), device=device)
        for i in range(cur_bs):
            w_start = (start + i) * hop
            w_end = w_start + window
            batch_windows[i] = x[0, :, :, :, w_start:w_end]

        preds = model(batch_windows)["logits"]
        preds = torch.sigmoid(preds)

        for i in range(cur_bs):
            w_start = (start + i) * hop
            w_end = w_start + window
            full_preds[w_start:w_end] += preds[i]
            overlap_counts[w_start:w_end] += 1

    mask = overlap_counts > 0
    full_preds[mask] /= overlap_counts[mask]
    return full_preds.detach().cpu().numpy()


# ----------------- entry -----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="", help="optional JSON config path")
    args = p.parse_args()

    # this will be replaces with an argparser that takes config files as well. This will make it so that we can remove the string to attribute thing that clutters the code 
    cfg = {
        "GENERAL": {"seed": 42, "name": "cpd_min"},
        "DATA": {
            "root_dir": "/home/soufiane/code/models/cpd/Roel/Dataset009_IMPACTRaw",  # expects root_dir/h5Tr/*.h5
            "batch_size": 4,
            "sample_size": [192, 128, 64],                # (H, W, T)
            "states": [1],
            "dt": 0.0625,
            "fold": 0,
            "use_phase_labels": True,
            "num_workers": 2,
            "num_cached": 1
        },
        "MODEL": {
            "name": "FullModel",
            "dt": 0.0625,
            "frame_encoder": {
                "name": "ResNetPseudo2D",  
                "n_states": 1,
                "n_channels_in": 6,
                "blocks": [2, 2, 2, 2]
            },
            "classifier": {
                "name": "RNNProbe",     
                "layer": "GRU",
                "dt": 0.0625,
                "n_channels": 128,
                "n_layers": 2,
                "dropout": 0.5,
                "bidirectional": True,
                "n_classes": 1,
                "seq_len": 256,
                "use_phase": True
            },
            "loss": {
                "name": "CombinedLoss",   
                "weights": {"bce": 1.0, "emd": 0.5, "phase": 0.2, "normalize": False},
                "phase_mode": "mse",
                "emd_reduction": "mean"
            },
            "classifier_dt": 0.0625,    
            "classifier_seq_len": 32
        },
        "OPTIM": {
            "lr": 1e-3, "weight_decay": 0.0, "epochs": 50, "gpu": 0,
            "train_steps_per_epoch": 200, "val_steps": 10
            }

    }

    if args.config:
        with open(args.config, "r") as f:
            cfg = json.load(f)

    set_seed(cfg["GENERAL"]["seed"])
    device = torch.device(f"cuda:{cfg['OPTIM']['gpu']}" if torch.cuda.is_available() else "cpu")

    dm = H5DataModule(cfg["DATA"])              
    train_iter = dm.train_dataloader()
    val_iter   = dm.val_dataloader()

    model = FullModel(cfg["MODEL"]).to(device) 

    opt = AdamW(model.parameters(), lr=cfg["OPTIM"]["lr"], weight_decay=cfg["OPTIM"]["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=cfg["OPTIM"]["epochs"])

    run_dir = Path("runs_min") / cfg["GENERAL"]["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir / "tb"))
    ckpt_last = run_dir / "last.pt"
    ckpt_best = run_dir / "best.pt"
    best_val = float("inf")

    for epoch in range(cfg["OPTIM"]["epochs"]):
        tr = train_one_epoch(model, train_iter, opt, device, epoch, writer,
                            steps_per_epoch=cfg["OPTIM"].get("train_steps_per_epoch", 200))
        va = validate_one_epoch(model, val_iter, device, epoch, writer, cfg["MODEL"],
                                val_steps=cfg["OPTIM"].get("val_steps", 10))

        try: sched.step()
        except TypeError: sched.step(epoch)

        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": opt.state_dict(), "scheduler": sched.state_dict()}, ckpt_last)
        if va["total"] < best_val:
            best_val = va["total"]
            torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_best)

        print(f"[{epoch+1:03d}/{cfg['OPTIM']['epochs']}] "
              f"train={tr['total']:.4f}  val={va['total']:.4f}  (best {best_val:.4f})")

    writer.close()

    test_ds = H5Dataset({**cfg["DATA"], "root_dir": cfg["DATA"].get("test_dir", cfg["DATA"]["root_dir"])}, mode="test")  
    if len(test_ds) > 0:
        batch = test_ds.generate_test_batch(0)
        preds = test_sliding_window(model, batch["data"], seq_len=cfg["MODEL"]["classifier_seq_len"], device=device)
        np.save(run_dir / "test_preds_sample0.npy", preds)
        print(f"Saved test preds to {run_dir/'test_preds_sample0.npy'}")


if __name__ == "__main__":
    main()
