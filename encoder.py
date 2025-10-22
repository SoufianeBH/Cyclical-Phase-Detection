# encoder 
import einops
import math
import sys
import torch.nn.functional as F

from torch.nn import *
from torchdiffeq import odeint
from vit_pytorch.simple_vit_1d import *
from losses import *

def time_embedding(t, embedding_dim=64, temperature=10000):
    assert embedding_dim % 2 == 0, 'embedding dimension must be even'
    t = t.unsqueeze(-1)

    # Create frequency bands
    dim_t = embedding_dim // 2
    omega = torch.arange(dim_t, device=t.device) / (dim_t - 1)
    omega = 1. / (temperature ** omega)

    while omega.dim() < t.dim():
        omega = omega.unsqueeze(0)

    # Apply to time values
    t_scaled = t * omega

    # Create sinusoidal embeddings
    pe = torch.cat([torch.sin(t_scaled), torch.cos(t_scaled)], dim=-1)
    return pe

class DownConvPseudo2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, bias=False, **kwargs),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels, out_channels, kernel_size, bias=False, **kwargs),
            nn.Conv3d(out_channels, out_channels, kernel_size, (2, 2, 1), bias=False, **kwargs),
            nn.BatchNorm3d(out_channels)
        )
        self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size, (2, 2, 1), bias=False, **kwargs)

    def forward(self, x):
        identity = self.downsample(x)

        out = self.sequential(x)
        out += identity

        return F.leaky_relu(out, 0.2)

class UpConvPseudo2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, bias=False, **kwargs),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels, out_channels, kernel_size, bias=False, **kwargs),
            nn.ConvTranspose3d(out_channels, out_channels, (4, 4, 1), (2, 2, 1), bias=False,
                               padding=kwargs["padding"], padding_mode="zeros"),
            nn.BatchNorm3d(out_channels)
        )
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, (4, 4, 1), (2, 2, 1), bias=False,
                                           padding=kwargs["padding"], padding_mode="zeros")

    def forward(self, x):
        identity = self.upsample(x)

        out = self.sequential(x)
        out += identity

        return F.leaky_relu(out, 0.2)

class DoubleConvPseudo2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, do_identity=True, **kwargs):
        super().__init__()
        self.do_identity = do_identity

        self.sequential = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, bias=False, **kwargs),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels, out_channels, kernel_size, bias=False, **kwargs),
            nn.BatchNorm3d(out_channels)
        )
        if in_channels != out_channels:
            self.identity = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        else:
            self.identity = nn.Identity()

    def forward(self, x):
        identity = self.identity(x)

        out = self.sequential(x)
        if self.do_identity:
            out += identity

        return F.leaky_relu(out, 0.2)

class ResNetPseudo2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cnn = self.make()

    def make(self):
        c = self.config["n_channels_in"]
        layers = []

        for i, block in enumerate(self.config["blocks"]):
            for j in range(block):
                if i == 0 and j == 0:
                    layers.append(DoubleConvPseudo2D(self.config["n_states"], c, (3, 3, 1),
                                                     padding=(1, 1, 0), padding_mode="replicate"))
                else:
                    layers.append(DoubleConvPseudo2D(c * (2 ** i), c * (2 ** i), (3, 3, 1),
                                                     padding=(1, 1, 0), padding_mode="replicate"))

                if j == block - 1:
                    layers.append(DownConvPseudo2D(c * (2 ** i), c * (2 ** (i + 1)), (3, 3, 1),
                                                   padding=(1, 1, 0), padding_mode="replicate"))

        self.config["n_channels_out"] = c * (2 ** (i + 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)
    
class ResAutoEncoder(nn.Module):
    def __init__(self, config):
        super(ResAutoEncoder, self).__init__()
        self.config = config
        self.latent_std = None
        self.latent_mean = None

        self.interp_delta = config["interp_delta"]
        self.n_channels_in = config["n_channels_in"]
        self.n_latent = config["n_latent"]

        self.x, self.y = config["input_shape"]
        self.encoder = self.make_encoder()
        self.decoder = self.make_decoder()

        self.to_z = nn.Linear(self.n_flat, self.n_latent)
        self.from_z = nn.Linear(self.n_latent, self.n_flat)

        self.loss = self.str_to_attr(config["loss"]["name"])()

    def make_encoder(self):
        c = self.config["n_channels_in"]
        layers = []

        for i, block in enumerate(self.config["blocks"]):
            for j in range(block):
                if i == 0 and j == 0:
                    layers.append(DoubleConvPseudo2D(self.config["n_states"], c, (3, 3, 1),
                                                     padding=(1, 1, 0), padding_mode="replicate"))
                else:
                    layers.append(DoubleConvPseudo2D(c * (2 ** i), c * (2 ** i), (3, 3, 1),
                                                     padding=(1, 1, 0), padding_mode="replicate"))

                if j == block - 1:
                    layers.append(DownConvPseudo2D(c * (2 ** i), c * (2 ** (i + 1)), (3, 3, 1),
                                                   padding=(1, 1, 0), padding_mode="replicate"))

        self.n_flat = (c * (2 ** (i + 1))) * (self.x // (2 ** (i + 1))) * (self.y // (2 ** (i + 1)))
        return nn.Sequential(*layers)

    def make_decoder(self):
        c = self.config["n_channels_in"]
        layers = []

        for i, block in enumerate(reversed(self.config["blocks"])):
            for j in range(block):
                if j == 0:
                    layers.append(UpConvPseudo2D(c * (2 ** (len(self.config["blocks"]) - i)),
                                                 c * (2 ** (len(self.config["blocks"]) - i - 1)),
                                                (3, 3, 1), padding=(1, 1, 0), padding_mode="replicate"))

                if i == len(self.config["blocks"]) - 1 and j == block - 1:
                    layers.append(DoubleConvPseudo2D(c * (2 ** (len(self.config["blocks"]) - i - 1)),
                                                     self.config["n_states"],
                                                     (3, 3, 1), padding=(1, 1, 0), padding_mode="replicate"))
                else:
                    layers.append(DoubleConvPseudo2D(c * (2 ** (len(self.config["blocks"]) - i - 1)),
                                                     c * (2 ** (len(self.config["blocks"]) - i - 1)),
                                                     (3, 3, 1), padding=(1, 1, 0), padding_mode="replicate"))

        return nn.Sequential(*layers)

    def forward(self, x, t=None):
        z = self.encoder(x)
        shape = z.shape

        z = rearrange(z, "B C X Y Z -> B Z (X Y C)")
        z = self.to_z(z)

        if t is not None:
            t = time_embedding(t, embedding_dim=self.n_latent)

            # interpolation
            z_interp = self.interpolate(z, t)

            x_interp = self.from_z(z_interp)
            x_interp = rearrange(x_interp, "B Z (X Y C) -> B C X Y Z", X=shape[2], Y=shape[3], C=shape[1])
            x_interp = self.decoder(x_interp)
            z = z + t

        x_recon = self.from_z(z)
        x_recon = rearrange(x_recon, "B Z (X Y C) -> B C X Y Z", X=shape[2], Y=shape[3], C=shape[1])
        x_recon = self.decoder(x_recon)
        if self.training:
            self.latent_std = torch.std(z - t, dim=[0, 1])
            self.latent_mean = torch.mean(z - t, dim=[0, 1])

        if t is not None:
            return x_recon, z - t, x_interp
        return x_recon, z

    def interpolate(self, z, t):
        delta = self.interp_delta
        z = z[:, torch.arange(1, z.shape[1], delta)]
        t = t[:, torch.arange(1 + delta // 2, t.shape[1] - delta // 2, delta)]

        # linearly interpolate z
        z_interp = 0.5 * (z[:, :-1] + z[:, 1:])
        return z_interp + t

    def generate(self, n_samples):
        z = torch.randn(n_samples, self.n_latent, device='cuda' if torch.cuda.is_available() else 'cpu')
        z = z * self.latent_std + self.latent_mean

        t = torch.rand(n_samples, device='cuda' if torch.cuda.is_available() else 'cpu') * 300
        t = time_embedding(t, embedding_dim=self.n_latent)

        x_recon = self.from_z(z + t)
        x_recon = rearrange(x_recon[:, None], "B Z (X Y C) -> B C X Y Z", X=12, Y=8, C=96)
        x_recon = self.decoder(x_recon)
        return x_recon[..., 0]

    @staticmethod
    def str_to_attr(name):
        return getattr(sys.modules[__name__], name)
    