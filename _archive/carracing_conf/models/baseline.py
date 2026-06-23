"""Baseline world model: non-physics-informed, black-box, 11-dim latent.

Latent has NO physical meaning. Encoder/Decoder trained as pure autoencoder,
Dynamics is a plain MLP.
"""

import torch
import torch.nn as nn
from config import FRAME_STACK

BASELINE_LATENT_DIM = 11  # same as PIWM's full state dim


class BaselineEncoder(nn.Module):
    """CNN: stacked frames (FRAME_STACK, 64, 64) -> 11-dim black-box latent."""

    def __init__(self, latent_dim=BASELINE_LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.fc(h)


class BaselineDecoder(nn.Module):
    """11-dim black-box latent -> 64x64 image. Same architecture as PIWM decoder."""

    def __init__(self, latent_dim=BASELINE_LATENT_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256 * 4 * 4),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 256, 4, 4)
        return self.deconv(h)


class BaselineDynamics(nn.Module):
    """Plain MLP: (z_t, action_t) -> delta_z -> z_{t+1}.

    No physics. Predicts residual delta, then z_next = z_t + delta.
    """

    def __init__(self, latent_dim=BASELINE_LATENT_DIM, action_dim=3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z, action):
        """z: (B, latent_dim), action: (B, 3) -> z_next: (B, latent_dim)"""
        inp = torch.cat([z, action], dim=-1)
        delta = self.net(inp)
        return z + delta
