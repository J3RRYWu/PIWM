"""Lane-aware Decoder for PIWM-v5.

Same architecture as PhysicsDecoder, but input dim = 9 (car) + 20 (lane) = 29.
"""
import torch
import torch.nn as nn
from config import ENCODER_DIM as CAR_ENCODER_DIM
from lane_utils import LANE_DIM


class PhysicsDecoderLane(nn.Module):
    def __init__(self, in_dim=CAR_ENCODER_DIM + LANE_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 256 * 4 * 4), nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),    nn.Sigmoid(),
        )

    def forward(self, z):
        """z: (B, 29) -> (B, 1, 64, 64)"""
        h = self.fc(z).view(z.size(0), 256, 4, 4)
        return self.deconv(h)
