import torch
import torch.nn as nn
from config import FRAME_STACK, FULL_STATE_DIM


class CoarsePositionHead(nn.Module):
    """Predicts absolute (x, y) position from stacked frames.

    Noisy per-step but no drift. Shares CNN architecture with encoder
    but has its own weights.
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
            nn.Linear(256, 2),  # (x, y) normalized
        )

    def forward(self, frames):
        """
        Args:
            frames: (batch, FRAME_STACK, 64, 64)
        Returns:
            pos: (batch, 2) predicted (x, y) in normalized space
        """
        h = self.conv(frames)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class FusionGate(nn.Module):
    """Learns to fuse integrated position (accurate short-term, drifts)
    with coarse position head (noisy but no drift).

    gate ~ 1.0 -> trust integration (short-term)
    gate ~ 0.0 -> trust coarse head (long-term anchor)
    """

    def __init__(self):
        super().__init__()
        # Input: integrated_pos(2) + coarse_pos(2) + observable_state(9) = 13
        self.net = nn.Sequential(
            nn.Linear(13, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # one gate per (x, y)
            nn.Sigmoid(),
        )

    def forward(self, pos_integrated, pos_coarse, observable_state):
        """
        Args:
            pos_integrated: (batch, 2) position from dynamics integration
            pos_coarse: (batch, 2) position from CoarsePositionHead
            observable_state: (batch, 9) encoder output [yaw, vx, vy, ...]
        Returns:
            pos_fused: (batch, 2) fused position estimate
            gate: (batch, 2) gate values for analysis
        """
        inp = torch.cat([pos_integrated, pos_coarse, observable_state], dim=-1)
        gate = self.net(inp)  # (batch, 2) in [0, 1]
        pos_fused = gate * pos_integrated + (1 - gate) * pos_coarse
        return pos_fused, gate
