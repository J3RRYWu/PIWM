"""Lane-aware Encoder for PIWM-v5.

CNN backbone with 15-frame input (richer temporal context for waypoint
shape estimation). Two output heads:
  - car_head:  9-dim car observable [yaw, vx, vy, omega, w0..w3, delta]
  - lane_head: 20-dim lane waypoints (10 x 2) in body frame

Output concatenated to 29-dim latent obs.
"""
import torch
import torch.nn as nn
from config import ENCODER_DIM as CAR_ENCODER_DIM
from lane_utils import LANE_DIM, LANE_FRAME_STACK


class PhysicsEncoderLane(nn.Module):
    def __init__(self, car_dim=CAR_ENCODER_DIM, lane_dim=LANE_DIM,
                 frame_stack=LANE_FRAME_STACK):
        super().__init__()
        self.car_dim = car_dim    # 9
        self.lane_dim = lane_dim  # 20
        self.frame_stack = frame_stack  # 15

        self.conv = nn.Sequential(
            nn.Conv2d(frame_stack, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),          nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),         nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),        nn.ReLU(),
        )
        self.feat = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256), nn.ReLU(),
        )
        self.car_head  = nn.Linear(256, car_dim)
        self.lane_head = nn.Linear(256, lane_dim)

    def features(self, x):
        """x: (B, FRAME_STACK, 64, 64) -> (B, 256) shared penultimate features.
        Exposed so a road-context head (curvature perception) can reuse the same
        backbone; see models/road_perception.py."""
        h = self.conv(x).reshape(x.size(0), -1)
        return self.feat(h)

    def forward(self, x):
        """x: (B, FRAME_STACK, 64, 64) -> (B, 29) concat[car(9), lane(20)]"""
        h = self.features(x)
        return torch.cat([self.car_head(h), self.lane_head(h)], dim=-1)
