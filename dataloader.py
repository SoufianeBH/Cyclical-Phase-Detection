# import necessary libraries
from typing import Any, Optional

# Removed unused Lightning/metrics/matplotlib/PIL/einops/io imports
import h5py
import numpy as np
import sys
import os

# Keep only what's used
import torch  # optional; not strictly needed, but often handy
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from glob import glob
from os.path import join

from pathlib import Path
from PIL import Image

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform

class H5Dataset(SlimDataLoaderBase):
    """
    Minimal, Lightning-free dataset wrapper.
    Expects a plain dict 'config' (no Hydra/OMEGACONF/etc).
    """
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.use_phase = config.get('use_phase_labels', False)

        self.id_list = self.setup()
        super(H5Dataset, self).__init__(self.id_list, config['batch_size'])

        # Keep epoch iteration behavior as-is (small & explicit)
        self.iters_per_epoch = 2
        self.current_iter = 0

        self.sample_size = config['sample_size']  # (H, W, T)
        self.states = config['states']            # e.g., [0, 1]

    def __len__(self):
        return len(self.id_list)

    def setup(self):
        id_list = sorted(glob(join(self.config['root_dir'], 'h5Tr', "*.h5")))

        # split id_list into 5 equal parts
        folds = np.array_split(id_list, 5)
        folds = [fold.tolist() for fold in folds]
        if len(folds) > 1 and len(folds[1]) > 9:
            # preserve your original deletion
            del folds[1][9]

        if self.mode == 'train':
            id_lists = [f for j, f in enumerate(folds) if j != self.config['fold']]
            return [id_ for id_list in id_lists for id_ in id_list]
        elif self.mode == 'val':
            return folds[self.config['fold']]
        else:
            # keep behavior simple; you can extend later
            return id_list

    def generate_train_batch(self):
        # reset iteration counter at epoch boundary
        if self.current_iter >= self.iters_per_epoch:
            self.current_iter = 0
            raise StopIteration
        self.current_iter += 1

        # generate batch
        indices = np.random.choice(len(self.id_list), self.batch_size, replace=True)

        H, W, T = self.sample_size
        images = np.empty((self.batch_size, len(self.states), H, W, T), dtype=np.float32)
        labels = np.empty((self.batch_size, T), dtype=np.float32)
        frames = np.empty((self.batch_size, T), dtype=np.float32)

        # ADD: Initialize phase array if using phase
        phase_labels = None
        if self.use_phase:
            phase_labels = np.empty((self.batch_size, T, 2), dtype=np.float32)

        for i, idx in enumerate(indices):
            h5_path = self.id_list[idx]
            with h5py.File(h5_path, 'r') as f:
                max_start = f['image'].shape[2] - T
                rand = np.random.randint(0, max_start) if max_start > 0 else 0

                # load
                images[i, 0] = f['image'][:, :, rand:rand + T]
                labels[i] = f['image'].attrs['label'][rand:rand + T]
                frames[i] = rand + np.arange(T)

                # ADD: Load phase if available
                if self.use_phase:
                    if 'phase' in f:
                        phase_labels[i] = f['phase'][rand:rand + T]
                    else:
                        # Generate synthetic phase if not in file
                        phases = np.linspace(0, 2 * np.pi, T, endpoint=False)
                        phase_labels[i, :, 0] = np.cos(phases)
                        phase_labels[i, :, 1] = np.sin(phases)

            if self.mode == 'train':
                if np.random.rand() < 0.5:
                    images[i, 0] = np.roll(images[i, 0], np.random.randint(32), axis=-2)
                if np.random.rand() < 0.5:
                    images[i, 0] = np.flip(images[i, 0], axis=-2)

        # Process derivatives (existing code)
        for i, state in enumerate(self.states):
            if i == 0 and state == 0:
                continue
            elif i == 0 and state == 1:
                images[:, i] = np.gradient(images[:, i], self.config["dt"], axis=-1)
            elif i == 0 and state == 2:
                images[:, i] = np.gradient(images[:, i], self.config["dt"], axis=-1)
                images[:, i] /= 10
                images[:, i] = np.gradient(images[:, i], self.config["dt"], axis=-1)
            elif i == 0 and state == 3:
                images[:, i] = np.gradient(images[:, i], self.config["dt"], axis=-1)
                images[:, i] /= 10
                images[:, i] = np.gradient(images[:, i], self.config["dt"], axis=-1)
                images[:, i] /= 10
                images[:, i] = np.gradient(images[:, i], self.config["dt"], axis=-1)
            else:
                images[:, i] = np.gradient(images[:, i - 1], self.config["dt"], axis=-1)
            images[:, i] /= 10

        batch_dict = {"data": images, "ref": labels, "frames": frames}
        if self.use_phase:
            batch_dict["phase"] = phase_labels
        return batch_dict

    def generate_test_batch(self, idx):
        h5_path = self.id_list[idx]
        with h5py.File(h5_path, 'r') as f:
            # keep original behavior: add batch & channel dims as in your code
            images = f['image'][:][None, None]
        return {"data": images}

class H5DataModule:
    """
    Minimal, Lightning-free version of your H5DataModule.
    It:
      - builds train/val/test H5Dataset instances (same split logic lives in H5Dataset.setup)
      - provides transforms (Compose) and MultiThreadedAugmenter wrappers
      - provides a transfer_batch_to_device(...) helper identical to your Lightning version
    """
    def __init__(self, config: dict):
        self.config = config
        # Reuse your H5Dataset which already handles folds & modes. :contentReference[oaicite:0]{index=0}
        self.train_dataset = H5Dataset(config, mode="train")  # uses same safe fold deletion. :contentReference[oaicite:1]{index=1}
        self.val_dataset   = H5Dataset(config, mode="val")
        # For test, use root_dir by default; set config['test_dir'] if different
        test_cfg = dict(config)
        test_cfg["root_dir"] = config.get("test_dir", config["root_dir"])
        self.test_dataset  = H5Dataset(test_cfg, mode="test")

    # ---- transforms (match your old ones) ----
    def get_train_transforms(self):
        return Compose([
            GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_sample=0.3),
            BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.3),
            GammaTransform(gamma_range=(0.8, 1.2), p_per_sample=0.3),
        ])

    def get_val_transforms(self):
        return Compose([])

    def get_test_transforms(self):
        return Compose([])

    # ---- “dataloaders” (augmenter-wrapped iterables) ----
    def train_dataloader(self):
        return MultiThreadedAugmenter(
            data_loader=self.train_dataset,
            transform=self.get_train_transforms(),
            num_processes=self.config.get('num_workers', 0),
            num_cached_per_queue=self.config.get('num_cached', 1),
            seeds=list(range(self.config.get('num_workers', 0))),
            pin_memory=True
        )

    def val_dataloader(self):
        return MultiThreadedAugmenter(
            data_loader=self.val_dataset,
            transform=self.get_val_transforms(),
            num_processes=self.config.get('num_workers', 0),
            num_cached_per_queue=self.config.get('num_cached', 1),
            seeds=list(range(self.config.get('num_workers', 0))),
            pin_memory=True
        )

    def test_dataloader(self):
        # You can also wrap with augmenter + get_test_transforms() if you want
        return self.test_dataset

    # ---- keep your batch -> torch helper (Lightning-free) ----
    @staticmethod
    def transfer_batch_to_device(batch, device):
        out = {
            "data":   torch.from_numpy(batch["data"]).to(device),
            "ref":    torch.from_numpy(batch["ref"]).to(device),
            "frames": torch.from_numpy(batch["frames"]).to(device),
        }
        if "phase" in batch:
            out["phase"] = torch.from_numpy(batch["phase"]).to(device)
        return out

# -----------------------
# Minimal shape check
# -----------------------
if __name__ == "__main__":
    cfg = {
        "root_dir": "/home/soufiane/code/models/cpd/Roel/Dataset009_IMPACTRaw",
        "batch_size": 2,
        "sample_size": [192, 128, 96],       # (H, W, T)
        "states": [0, 1],                     # first=raw, second=1st derivative
        "dt": 0.0625,
        "fold": 0,
        "use_phase_labels": True,
    }

    ds = H5Dataset(cfg, mode="train")

    batch = ds.generate_train_batch()
    data = batch["data"]
    ref = batch["ref"]
    frames = batch["frames"]
    print("Batch shapes:")
    print("  data:", data.shape, data.dtype)
    print("  ref:", ref.shape, ref.dtype)
    print("  frames:", frames.shape, frames.dtype)
    if "phase" in batch:
        print("  phase:", batch["phase"].shape, batch["phase"].dtype)

    # plot the phase labels for the first sample
    if "phase" in batch:
        import matplotlib.pyplot as plt

        phase = batch["phase"][0]  # (T, 2)
        T = phase.shape[0]
        angles = np.arctan2(phase[:, 1], phase[:, 0])
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(T), angles, marker='o')
        plt.title("Phase Angles over Time")
        plt.xlabel("Frame")
        plt.ylabel("Phase Angle (radians)")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.scatter(phase[:, 0], phase[:, 1], c=np.arange(T), cmap='viridis', marker='o')
        circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
        plt.gca().add_artist(circle)
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.title("Phase Vectors on Unit Circle")
        plt.xlabel("cos(phase)")
        plt.ylabel("sin(phase)")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar(label='Frame Index')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # save the figure
        out_dir = (Path(__file__).parent if "__file__" in globals() else
                     Path.cwd()) / "debug_frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / "phase_labels_sample0.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved phase plot to: {save_path}")

    # save a preview frame (B=0, state/channel=0, middle time frame)
    out_dir = (Path(__file__).parent if "__file__" in globals() else Path.cwd()) / "debug_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    b, c, h, w, t = data.shape
    t_idx = t // 2  # middle frame
    frame = data[0, 1, :, :, t_idx]  # (H, W) float32

    # normalize to [0,255] for viewing
    fmin, fmax = float(frame.min()), float(frame.max())
    if fmax > fmin:
        frame_vis = (255.0 * (frame - fmin) / (fmax - fmin)).astype(np.uint8)
    else:
        frame_vis = np.zeros_like(frame, dtype=np.uint8)

    img = Image.fromarray(frame_vis)
    save_path = out_dir / f"sample0_state0_t{t_idx}.png"
    img.save(save_path)
    print(f"Saved preview frame to: {save_path}")

    print("\n--- DataModuleLite quick check ---")
    cfg = {
        "root_dir": "/home/soufiane/code/models/cpd/Roel/Dataset009_IMPACTRaw",
        "batch_size": 2,
        "sample_size": [192, 128, 96],
        "states": [0, 1],
        "dt": 0.0625,
        "fold": 0,
        "use_phase_labels": True,
        "num_workers": 1,   # start simple; you can raise this later
        "num_cached": 1,
    }
    dm = H5DataModule(cfg)
    train_loader = dm.train_dataloader()

    # pull one augmented batch
    batch = next(iter(train_loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tb = H5DataModule.transfer_batch_to_device(batch, device)
    print("Train batch (torch) shapes:",
          tb["data"].shape, tb["ref"].shape, tb["frames"].shape,
          "(phase:", tb["phase"].shape if "phase" in tb else "None", ")")
