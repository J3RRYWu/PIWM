import torch
import torch.nn as nn
from config import FRAME_STACK, ENCODER_DIM


class PhysicsEncoder(nn.Module):
    """CNN Encoder: stacked frames (FRAME_STACK, 64, 64) -> 9-dim observable physics.

    Outputs: [yaw, vx, vy, omega, w0, w1, w2, w3, steer]
    Does NOT predict position (x, y) — position is maintained by dynamics integration.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, 4, stride=2, padding=1),   # -> (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),            # -> (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),           # -> (128, 8, 8)
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),          # -> (256, 4, 4)
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, ENCODER_DIM),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, FRAME_STACK, 64, 64) stacked grayscale frames
        Returns:
            z: (batch, 9) predicted observable physics state (normalized)
        """
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)
