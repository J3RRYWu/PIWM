"""LSTM-based localization models (Plan A and Plan A+B).

Plan A:   CNN + LSTM → (x, y)
Plan A+B: CNN + LSTM → (s_sin, s_cos, d_norm)
"""

import torch
import torch.nn as nn
from config import FRAME_STACK


class CNNEncoder(nn.Module):
    """Shared CNN trunk: (3, 64, 64) stacked frames -> feature vector."""

    def __init__(self, feat_dim=128):
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
        self.fc = nn.Linear(256 * 4 * 4, feat_dim)

    def forward(self, x):
        """x: (B, 3, 64, 64) -> (B, feat_dim)"""
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class LSTMLocalizer(nn.Module):
    """CNN + LSTM over sequence of frame-stacks.

    Args:
        out_dim: 2 for (x, y) [Plan A], 3 for (s_sin, s_cos, d) [Plan A+B]
    """

    def __init__(self, out_dim=2, feat_dim=128, hidden_dim=256):
        super().__init__()
        self.cnn = CNNEncoder(feat_dim=feat_dim)
        self.lstm = nn.LSTM(feat_dim, hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, seq):
        """seq: (B, T, 3, 64, 64) -> (B, out_dim)

        Uses LSTM output at last timestep as prediction target.
        """
        B, T = seq.shape[:2]
        flat = seq.reshape(B * T, *seq.shape[2:])  # (B*T, 3, 64, 64)
        feats = self.cnn(flat).reshape(B, T, -1)    # (B, T, feat_dim)
        out, _ = self.lstm(feats)                    # (B, T, hidden_dim)
        last = out[:, -1, :]                         # (B, hidden_dim)
        return self.head(last)
