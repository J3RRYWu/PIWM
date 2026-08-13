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
    """Perceives the road ahead in two equivalent parameterizations.

    kappa_head : kappa(s + offsets)          -- what the DYNAMICS consumes.
    shape_head : the same road as GEOMETRY   -- the local centreline sampled at
        those arc-lengths, in the road frame at the car: (X_j - offsets_j, Y_j).
        Predicting the along-track residual keeps both outputs at the same
        (centimetre) scale, and Y is the lateral offset of the road ahead.

    Why both. Position can be recovered from kappa alone, but only by
    integrating it twice (heading, then position), and that squares the error:
    measured map-free, a PERFECT kappa profile still gives 0.414 m @100 vs
    0.437 m for the perceived one -- i.e. only ~2 cm of the map-free penalty is
    perception, the rest is the double integration. The shape head reads the
    same geometry off directly, so that term is gone by construction. kappa is
    kept because the Frenet ODE genuinely needs curvature at the car, and at
    that it is accurate enough. Both come from the camera; neither touches the
    map, so the model stays a pure world model.
    """
    def __init__(self, n_offsets=10, frame_stack=LANE_FRAME_STACK, freeze_backbone=False,
                 predict_shape=False):
        super().__init__()
        self.backbone = PhysicsEncoderLane(frame_stack=frame_stack)
        self.kappa_head = nn.Linear(256, n_offsets)
        self.predict_shape = predict_shape
        # linear like kappa_head: the comparison is about the target, not capacity
        self.shape_head = nn.Linear(256, 2 * n_offsets) if predict_shape else None
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

    def features(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                return self.backbone.features(x)
        return self.backbone.features(x)

    def forward(self, x):
        """x: (B, FRAME_STACK, 64, 64) -> (B, n_offsets) perceived kappa profile."""
        return self.kappa_head(self.features(x))

    def forward_both(self, x):
        """-> (kappa (B,n_off), road points (B,n_off,2) in the local road frame).

        The shape head predicts (X - offset, Y); the caller adds the offsets
        back to get X. Requires predict_shape=True.
        """
        feat = self.features(x)
        kap = self.kappa_head(feat)
        shp = self.shape_head(feat).view(len(x), -1, 2)
        return kap, shp
