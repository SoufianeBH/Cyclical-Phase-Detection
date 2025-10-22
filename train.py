# import necessary libraries
from typing import Any, Optional

import einops
import h5py
import io
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import sys
import torch
import torch.nn.functional as F

from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from glob import glob
from os.path import getctime, join

from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
from torch.nn import *
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchmetrics.classification import BinaryAUROC, BinaryF1Score
from losses import *
from model import FullModel, LatentODE, ResAutoEncoder

class SequentialScheduler(LRScheduler):
    def __init__(self, optimizer, scheduler1, scheduler2, switch_epoch):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.switch_epoch = switch_epoch
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.switch_epoch:
            return self.scheduler1.get_last_lr()
        else:
            return self.scheduler2.get_last_lr()

    def step(self, epoch=None):
        if self.last_epoch < self.switch_epoch:
            self.scheduler1.step(epoch)
        else:
            self.scheduler2.step(epoch)
        super().step(epoch)


def configure_optimizers(self):
    if "freeze_frame_encoder" in self.c_o and self.c_o["freeze_frame_encoder"]:
        for param in self.model.frame_encoder.parameters():
            param.requires_grad = False
    optimizer = self.str_to_attr(self.c_o["name"])(self.model.parameters(), **self.c_o["optimizer"])

    if self.c_o["n_warmup"] == 0:
        scheduler = self.str_to_attr(self.c_o["lr_policy"])(optimizer, **self.c_o["scheduler"])
    else:
        scheduler1 = LambdaLR(optimizer, lr_lambda=self.warmup_lambda)
        scheduler2 = self.str_to_attr(self.c_o["lr_policy"])(optimizer, **self.c_o["scheduler"])
        scheduler = SequentialScheduler(optimizer, scheduler1, scheduler2, self.c_o["n_warmup"])
    return [optimizer], [scheduler]

class LightningGating(pl.LightningModule):
    def __init__(self, config):
        super(LightningGating, self).__init__()
        self.config = config
        self.c_d, self.c_m, self.c_o = config["DATA"], config["MODEL"], config['OPTIMIZATION']

        self.model = self.str_to_attr(self.c_m['name'])(self.c_m)
        self.metrics = [self.str_to_attr(metric)().to(f"cuda:{self.c_o['gpu']}") for metric in self.c_m['metrics']]

        if "resume_frame_encoder" in self.c_o and self.c_o["resume_frame_encoder"] is not None:
            self.load_frame_encoder()

    def load_frame_encoder(self):
        checkpoint = glob(join(self.c_o["resume_frame_encoder"], "model", "epoch=*.ckpt"))
        checkpoint = torch.load(max(checkpoint, key=getctime))

        state_dict = checkpoint["state_dict"]
        state_dict = {k.replace("model.frame_encoder.", ""): v for k, v in state_dict.items() if "frame_encoder" in k}
        self.model.frame_encoder.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['data'], batch['ref']
        hr_values = self.ref_to_hr_metrics(y)
        output = self(x)

        if self.c_m["classifier"]["do_hr"]:
            output, output_hr = output
            hr_loss = F.mse_loss(output_hr[:, 0], hr_values[:, 0]) + F.mse_loss(output_hr[:, 1], hr_values[:, 1])
            self.log(f"train/HRLoss", hr_loss, on_step=False, on_epoch=True)

        loss = self.model.loss(output, y)
        if isinstance(loss, tuple):
            # ramp down BCE loss and ramp up EMD loss until 1000 epochs
            loss_bce = loss[0] * max(0., (1 - self.current_epoch / 1000))
            loss_emd = loss[1] * min(1.0, self.current_epoch / 1000)

            if len(loss) == 2:
                loss = loss_bce + loss_emd
            else:
                loss_kld = self.c_o["lambda_kld"] * loss[2] * min(1.0, self.current_epoch / 1000)
                self.log(f"train/KLDLoss", loss[2].item(), on_step=False, on_epoch=True)

                loss = loss_bce + loss_emd + loss_kld
                output = output[0]

        if self.c_m["classifier"]["do_hr"]:
            loss += hr_loss * min(1.0, max(0.0, (self.current_epoch - 100) / 1000))

        self.log(f"train/Loss", loss, on_step=False, on_epoch=True)

        # chamfer distance
        chamfer = self.chamfer_distance(output, y)
        self.log("train/ChamferDist", chamfer, on_step=False, on_epoch=True)

        for i, metric in enumerate(self.metrics):
            m = metric(output, y)
            self.log(f"train/{self.c_m['metrics'][i]}", m, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['data'], batch['ref']
        hr_values = self.ref_to_hr_metrics(y)
        output = self(x)

        if self.c_m["classifier"]["do_hr"]:
            output, output_hr = output
            hr_loss = F.mse_loss(output_hr[:, 0], hr_values[:, 0]) + F.mse_loss(output_hr[:, 1], hr_values[:, 1])
            self.log(f"val/HRLoss", hr_loss, on_epoch=True)

        loss = self.model.loss(output, y)
        if isinstance(loss, tuple):
            # ramp down BCE loss and ramp up EMD loss until 1000 epochs
            loss_bce = loss[0] * max(0., (1 - self.current_epoch / 1000))
            loss_emd = loss[1] * min(1.0, self.current_epoch / 1000)

            if len(loss) == 2:
                loss = loss_bce + loss_emd
            else:
                loss_kld = self.c_o["lambda_kld"] * loss[2] * min(1.0, self.current_epoch / 1000)
                self.log(f"val/KLDLoss", loss[2].item(), on_epoch=True)
                loss = loss_bce + loss_emd + loss_kld
                output = output[0]

        if self.c_m["classifier"]["do_hr"]:
            loss += hr_loss * min(1.0, max(0.0, (self.current_epoch - 100) / 1000))

        self.log(f"val/Loss", loss, on_epoch=True)

        # chamfer distance
        chamfer = self.chamfer_distance(output, y)
        self.log("val/ChamferDist", chamfer, on_epoch=True)

        for i, metric in enumerate(self.metrics):
            m = metric(output, y)
            self.log(f"val/{self.c_m['metrics'][i]}", m, on_epoch=True, prog_bar=True, logger=True)

        if (self.current_epoch + 1) % 20 == 0 and batch_idx == 0:
            self.plot_outputs(batch, output)
        return loss

    def test_step(self, batch):
        x = batch['data']
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        # get windowing
        _, channels, height, width, time = x.shape
        window_size = self.c_m['classifier']['seq_len']
        if window_size > time:
            window_size = time
        n_windows = ((time - window_size) // (window_size // 8)) + 1

        # initialize arrays for predictions
        all_predictions = []
        overlap_counts = torch.zeros(time, device=self.device)
        full_predictions = torch.zeros(time, device=self.device)

        for start_idx in range(0, n_windows, self.c_d['batch_size']):
            print("Processing window", start_idx, "of", n_windows)

            # determine batch end index
            end_idx = min(start_idx + self.c_d['batch_size'], n_windows)
            current_batch_size = end_idx - start_idx

            # initialize batch tensor
            batch_windows = torch.zeros(
                (current_batch_size, channels, height, width, window_size), device=self.device)

            # fill batch with windows
            for i in range(current_batch_size):
                window_start = (start_idx + i) * (window_size // 8)
                window_end = window_start + window_size
                batch_windows[i] = x[0, :, :, :, window_start:window_end]

            predictions = self(batch_windows)
            predictions = torch.sigmoid(predictions)

            all_predictions.extend([pred.cpu().numpy() for pred in predictions])
            for i in range(current_batch_size):
                window_start = (start_idx + i) * (window_size // 8)
                window_end = window_start + window_size

                # add predictions and count overlaps
                full_predictions[window_start:window_end] += predictions[i]
                overlap_counts[window_start:window_end] += 1

        mask = overlap_counts > 0
        full_predictions[mask] /= overlap_counts[mask]

        final_predictions = full_predictions.cpu().numpy()
        return final_predictions

    def ref_to_hr_metrics(self, y):
        heart_rates = torch.zeros(y.shape[0], device=y.device)
        hr_stds = torch.zeros(y.shape[0], device=y.device)

        for b in range(y.shape[0]):
            # Get indices where value is 1 (end-diastolic frames)
            peak_indices = torch.where(y[b] == 1)[0]

            if len(peak_indices) < 2:
                # Not enough peaks to compute heart rate
                heart_rates[b] = float('nan')
                hr_stds[b] = float('nan')
                continue

            # Compute time differences between consecutive peaks
            peak_intervals = (peak_indices[1:] - peak_indices[:-1]) * self.c_m["classifier"]["dt"]

            # Convert intervals to heart rate (beats per minute)
            instantaneous_hrs = 60 / peak_intervals

            # Compute average heart rate and standard deviation
            heart_rates[b] = torch.mean(instantaneous_hrs)
            hr_stds[b] = torch.std(instantaneous_hrs)

        # normalize heart rates and standard deviations
        heart_rates /= 15
        hr_stds /= 15
        return torch.stack([heart_rates, hr_stds], dim=1)

    @staticmethod
    def str_to_attr(name):
        return getattr(sys.modules[__name__], name)

    def warmup_lambda(self, iter_):
        return iter_ / self.c_o["n_warmup"] if iter_ < self.c_o["n_warmup"] else 1


class LightningPretraining(LightningGating):
    def __init__(self, config):
        super(LightningPretraining, self).__init__(config)

    def training_step(self, batch, batch_idx):
        x, y, f = batch['data'], batch['ref'], batch['frames']
        t = f * self.c_m["dt"]

        x_recon, z, x_interp = self.model(x, t)

        # reconstruction loss
        loss = self.model.loss(x_recon, x)

        # interpolation loss
        t_interp = torch.arange(1 + self.model.interp_delta // 2,
                                x.shape[-1] - self.model.interp_delta // 2,
                                self.model.interp_delta).to(x.device)
        x_ = x[..., t_interp]
        loss_interp = self.model.loss(x_interp, x_)

        # regularization loss
        loss_reg = torch.mean(torch.sum(z**2, dim=1))

        loss_total = loss + self.c_o["lambda_interp"] * loss_interp + self.c_o["lambda_reg"] * loss_reg
        self.log(f"train/Loss", loss_total, on_step=False, on_epoch=True)
        self.log(f"train/InterpLoss", loss_interp, on_step=False, on_epoch=True)
        self.log(f"train/RegLoss", loss_reg, on_step=False, on_epoch=True)

        for i, metric in enumerate(self.metrics):
            m = metric(x_recon, x)
            self.log(f"train/{self.c_m['metrics'][i]}", m, on_step=False, on_epoch=True)
        return loss_total

    def validation_step(self, batch, batch_idx):
        x, y, f = batch['data'], batch['ref'], batch['frames']
        t = f * self.c_m["dt"]

        x_recon, z, x_interp = self.model(x, t)

        # reconstruction loss
        loss = self.model.loss(x_recon, x)

        # interpolation loss
        t_interp = torch.arange(1 + self.model.interp_delta // 2,
                                x.shape[-1] - self.model.interp_delta // 2,
                                self.model.interp_delta).to(x.device)
        x_ = x[..., t_interp]
        loss_interp = self.model.loss(x_interp, x_)

        # regularization loss
        loss_reg = torch.mean(torch.sum(z**2, dim=1))

        loss_total = loss + self.c_o["lambda_interp"] * loss_interp + self.c_o["lambda_reg"] * loss_reg
        self.log(f"val/Loss", loss_total, on_step=False, on_epoch=True)
        self.log(f"val/InterpLoss", loss_interp, on_step=False, on_epoch=True)
        self.log(f"val/RegLoss", loss_reg, on_step=False, on_epoch=True)

        for i, metric in enumerate(self.metrics):
            m = metric(x_recon, x)
            self.log(f"val/{self.c_m['metrics'][i]}", m, on_step=False, on_epoch=True)

        if (self.current_epoch + 1) % 20 == 0 and batch_idx == 0:
            x = einops.rearrange(x, 'b c h w t -> (b t) c h w')
            x_recon = einops.rearrange(x_recon, 'b c h w t -> (b t) c h w')
            self.plot_reconstructions(x, x_recon)
            self.plot_generated_samples()
        return loss_total

