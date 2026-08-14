"""LSTM and Transformer road-context encoders: the non-physical benchmarks.

The conference version evaluates against two data-driven sequence models, an LSTM
and a Transformer, "which serve as non-physical benchmarks". The journal draft
dropped both and compares only against DVBF, GOKU-net, Vid2Param and SINDYc, all of
which are physics-aware or dynamics-discovery methods. That leaves the road-context
results with no floor: nothing in the comparison answers "what does a plain
sequence model get?". These two restore it.

They are deliberately plain. Same per-frame convolutional trunk as the autoencoding
variants, so the comparison is about the temporal model and not about vision
capacity; a sequence model over the 15 frames; a linear read-out to the curvature
profile. No reconstruction, no latent partition, no interpretability structure --
that absence is the point of a non-physical benchmark.

Both take (B, FS, 64, 64) and return (B, n_offsets), so they are drop-in wherever
RoadContextEncoder is used at evaluation time.
"""
import torch
import torch.nn as nn

from lane_utils import LANE_FRAME_STACK
from models.road_perception_ae import _FrameConv


class _Trunk(nn.Module):
    """Per-frame conv trunk shared by both baselines -> (B, FS, d_model)."""

    def __init__(self, d_model):
        super().__init__()
        self.conv = _FrameConv(1)
        self.proj = nn.Linear(self.conv.out_dim, d_model)

    def forward(self, x):
        b, fs = x.shape[:2]
        h = self.conv(x.reshape(b * fs, 1, *x.shape[2:]))
        return self.proj(h).view(b, fs, -1)


class LSTMRoadEncoder(nn.Module):
    """Per-frame CNN -> LSTM over the window -> linear read-out of the last step."""

    def __init__(self, n_offsets=10, frame_stack=LANE_FRAME_STACK,
                 d_model=128, hidden=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.trunk = _Trunk(d_model)
        self.rnn = nn.LSTM(d_model, hidden, num_layers=n_layers,
                           batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_offsets)

    def forward(self, x):
        out, _ = self.rnn(self.trunk(x))
        return self.head(out[:, -1])            # last step: the road ahead of NOW


class TransformerRoadEncoder(nn.Module):
    """Per-frame CNN -> Transformer encoder over the window -> mean-pooled read-out.

    Sizing matches the conference version's physical encoder (4 heads, 512
    feedforward) so this differs from the VQ+Transformer variant only in lacking the
    quantiser, which keeps the two interpretable against each other.
    """

    def __init__(self, n_offsets=10, frame_stack=LANE_FRAME_STACK,
                 d_model=160, n_layers=3, n_heads=4, ff=512, dropout=0.1):
        super().__init__()
        self.trunk = _Trunk(d_model)
        self.pos = nn.Parameter(torch.zeros(1, frame_stack, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead=n_heads,
                                           dim_feedforward=ff, dropout=dropout,
                                           batch_first=True)
        self.tf = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_offsets)

    def forward(self, x):
        h = self.trunk(x) + self.pos[:, :x.shape[1]]
        return self.head(self.tf(h).mean(1))
