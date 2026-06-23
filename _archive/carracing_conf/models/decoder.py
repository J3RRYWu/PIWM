import torch
import torch.nn as nn
from config import ENCODER_DIM


class PhysicsDecoder(nn.Module):
    """CNN Decoder: 9-dim observable physics -> reconstructed image (1, 64, 64).

    Input: [yaw, vx, vy, omega, w0, w1, w2, w3, steer] (no position).
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ENCODER_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256 * 4 * 4),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # -> (128, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # -> (64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # -> (32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),     # -> (1, 64, 64)
            nn.Sigmoid(),
        )

    def forward(self, z):
        """
        Args:
            z: (batch, 9) observable physics state (normalized)
        Returns:
            recon: (batch, 1, 64, 64) reconstructed grayscale image
        """
        h = self.fc(z)
        h = h.view(h.size(0), 256, 4, 4)
        return self.deconv(h)
