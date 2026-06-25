"""Road-context perception: front-view image stack -> upcoming curvature profile
kappa(s + [0,0.5,..,4.5] m).

This is what turns the Frenet model into a PURE world model. The dynamics used to
look kappa(s) up from the KNOWN fixed-track map, which is privileged information no
(image, action) model could have. Here kappa is instead PERCEIVED from the front
camera at t0 (one forward pass), and the 100-step rollout then indexes that already
-observed preview by how far the car has driven -- the 4.5 m preview covers the whole
rollout (mean 3.3 m / p90 4.45 m of arc length over 100 steps), so the road shape is
OBSERVED, never looked up or integrated forward.

Two training regimes (compared in train/train_kappa_perception.py):
  - frozen   : reuse the trained v6 PhysicsEncoderLane backbone (frozen), train only
               the linear kappa head on its 256-dim features.  ~2.6k params.
  - finetune : warm-start the same backbone and train it end-to-end with the head.
The kappa head predicts the profile in REAL curvature units (1/m).
"""
import torch
import torch.nn as nn

from models.encoder_lane import PhysicsEncoderLane
from lane_utils import LANE_FRAME_STACK


class RoadContextEncoder(nn.Module):
    def __init__(self, n_offsets=10, frame_stack=LANE_FRAME_STACK, freeze_backbone=False):
        super().__init__()
        self.backbone = PhysicsEncoderLane(frame_stack=frame_stack)
        self.kappa_head = nn.Linear(256, n_offsets)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def load_backbone(self, state_dict):
        """Warm-start the backbone from a trained PhysicsEncoderLane checkpoint."""
        self.backbone.load_state_dict(state_dict)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x):
        """x: (B, FRAME_STACK, 64, 64) -> (B, n_offsets) perceived kappa profile."""
        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.backbone.features(x)
        else:
            feat = self.backbone.features(x)
        return self.kappa_head(feat)
