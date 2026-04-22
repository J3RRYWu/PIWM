"""Action-augmented LSTM Localization.

Each timestep: CNN(image) features + action vector -> LSTM -> (x,y) or (s,d)
"""

import torch
import torch.nn as nn
from config import FRAME_STACK


class CNNEncoder(nn.Module):
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
        h = self.conv(x).view(x.size(0), -1)
        return self.fc(h)


class LSTMActionLocalizer(nn.Module):
    """CNN + LSTM that fuses visual features with actions per timestep."""

    def __init__(self, out_dim=3, feat_dim=128, action_dim=3, hidden_dim=256):
        super().__init__()
        self.cnn = CNNEncoder(feat_dim=feat_dim)
        self.lstm = nn.LSTM(feat_dim + action_dim, hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, seq_frames, seq_actions):
        """
        Args:
            seq_frames: (B, T, FRAME_STACK, 64, 64)
            seq_actions: (B, T, 3)
        Returns:
            (B, out_dim)
        """
        B, T = seq_frames.shape[:2]
        flat = seq_frames.reshape(B * T, *seq_frames.shape[2:])
        feats = self.cnn(flat).reshape(B, T, -1)      # (B, T, feat_dim)
        fused = torch.cat([feats, seq_actions], dim=-1)  # (B, T, feat+3)
        out, _ = self.lstm(fused)
        return self.head(out[:, -1, :])
