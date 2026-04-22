"""CoarsePositionHeadSD: predicts (s, d) track-relative coordinates instead of (x, y).

s = arc length along centerline (normalized to [0, 1] with sin/cos encoding)
d = lateral offset (normalized)
"""

import torch
import torch.nn as nn
from config import FRAME_STACK


class CoarsePositionHeadSD(nn.Module):
    """Predicts normalized (s_sin, s_cos, d) from stacked frames.

    We encode s as (sin(2πs/L), cos(2πs/L)) so that the wrap-around at s=L
    doesn't cause a discontinuity.
    """

    def __init__(self):
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
            nn.Linear(256, 3),  # (s_sin, s_cos, d_norm)
        )

    def forward(self, frames):
        h = self.conv(frames)
        h = h.view(h.size(0), -1)
        return self.fc(h)  # (batch, 3)
