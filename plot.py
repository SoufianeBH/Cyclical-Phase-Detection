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

def plot_inputs(self, batch):
    x, y = batch['data'], batch['ref']
    x = x.detach().cpu().numpy()

    fig, ax = plt.subplots(3, 5)
    for i in range(3):
        for j in range(5):
            ax[i, j].imshow(x[i, 0, :, :, 48 + j], cmap='gray', vmin=0, vmax=1)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

            if i == 0:
                ax[i, j].set_title(f"frame {48 + j}")
            if j == 0:
                ax[i, j].set_ylabel(f"sample {i}")

    # save plot to a memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    # convert to PIL Image and then to numpy array
    image = PIL.Image.open(buf)
    image = np.array(image)

    # add to tensorboard
    self.logger.experiment.add_image(f"train/batch_inputs", image.transpose(2, 0, 1), self.current_epoch)
    plt.close()

def plot_outputs(self, batch, output):
    x, y = batch['data'], batch['ref']
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    o = output.detach().cpu().numpy()

    fig, ax = plt.subplots(4, 1)
    for i in range(4):
        x_ = x[i, 0, ::2, 63, :]
        y_ = np.tile(y[i], (16, 1))
        o_ = np.tile(o[i], (16, 1))
        o_ = (o_ - o_.min()) / (o_.max() - o_.min())

        p = np.concatenate([x_, y_, o_], axis=0)
        ax[i].imshow(p, cmap='gray', vmin=0, vmax=1)
        ax[i].set_xticks([])
        ax[i].set_yticks([])

        ax[i].set_ylabel(f"batch {i}")

    # save plot to a memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    # convert to PIL Image and then to numpy array
    image = PIL.Image.open(buf)
    image = np.array(image)

    # add to tensorboard
    self.logger.experiment.add_image(f"train/batch_outputs", image.transpose(2, 0, 1), self.current_epoch)
    plt.close()

def plot_reconstructions(self, x, x_recon, num_samples=8):
    """
    Plot input images, their reconstructions, and the differences using tensorboard.

    Args:
        x: Original input images (B, 1, H, W)
        x_recon: Reconstructed images (B, 1, H, W)
        num_samples: Number of samples to plot
    """
    # generate random indices of size num_samples
    indices = torch.randperm(x.shape[0])[:num_samples]

    x_subset = x[indices].detach().cpu()
    x_recon_subset = x_recon[indices].detach().cpu()

    # Calculate difference
    diff = x_subset - x_recon_subset

    # Create a figure with subplots
    fig, axs = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    # If there's only one sample, make sure axs is 2D
    if num_samples == 1:
        axs = np.expand_dims(axs, 0)

    for i in range(num_samples):
        # Get original image
        orig_img = x_subset[i, 0].numpy()
        # Get reconstructed image
        recon_img = x_recon_subset[i, 0].numpy()
        # Get difference image
        diff_img = diff[i, 0].numpy()

        # Plot original
        axs[i, 0].imshow(orig_img, cmap='gray', vmin=0, vmax=1)
        axs[i, 0].set_title(f"Original {i + 1}")
        axs[i, 0].axis('off')

        # Plot reconstruction
        axs[i, 1].imshow(recon_img, cmap='gray', vmin=0, vmax=1)
        axs[i, 1].set_title(f"Reconstruction {i + 1}")
        axs[i, 1].axis('off')

        # Plot difference with seismic colormap
        im = axs[i, 2].imshow(diff_img, cmap='seismic', vmin=-1, vmax=1)
        axs[i, 2].set_title(f"Difference {i + 1}")
        axs[i, 2].axis('off')

    # Add colorbar for difference plots
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # Save figure to buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    # Convert to PIL Image
    image = PIL.Image.open(buf)

    # Convert PIL Image to tensor
    image_tensor = torch.tensor(np.array(image).transpose(2, 0, 1)).float() / 255.0

    # Add to tensorboard
    self.logger.experiment.add_image('Reconstructions', image_tensor, self.current_epoch)

    # Close figure to prevent memory leaks
    plt.close(fig)

def plot_generated_samples(self, num_samples=16, grid_size=(4, 4)):
    """
    Plot samples generated from latent vectors in TensorBoard.

    Args:
        num_samples: Number of samples to generate if z is None
        grid_size: Tuple (rows, cols) for arranging the grid
    """
    # Decode to get generated samples
    with torch.no_grad():
        z_recon = self.model.generate(num_samples)

    # Move to CPU for plotting
    z_recon = z_recon.detach().cpu()

    # Create figure for grid
    rows, cols = grid_size
    assert rows * cols >= num_samples, "Grid size too small for number of samples"

    fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    # Flatten axes for easy indexing if we have multiple rows and columns
    if rows > 1 or cols > 1:
        axs = axs.flatten()

    # Plot each generated sample
    for i in range(num_samples):
        img = z_recon[i, 0].numpy()

        # Handle the case of a single subplot
        if rows == 1 and cols == 1:
            ax = axs
        else:
            ax = axs[i]

        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Sample {i + 1}")
        ax.axis('off')

    # Hide any unused subplots
    if rows > 1 or cols > 1:
        for i in range(num_samples, rows * cols):
            axs[i].axis('off')

    # Save figure to buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    # Convert to PIL Image
    image = PIL.Image.open(buf)

    # Convert PIL Image to tensor
    image_tensor = torch.tensor(np.array(image).transpose(2, 0, 1)).float() / 255.0

    # Add to tensorboard
    self.logger.experiment.add_image('Generated Samples', image_tensor, self.current_epoch)

    plt.close(fig)
