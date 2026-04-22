"""PIWM-v5: Lane-augmented dynamics.

Latent split:
  z_car  = z[:, 0:11]                          (11-dim car state, PIWM convention)
  z_lane = z[:, 11:31]                         (20-dim = 10 waypoints x 2)

Everything is in NORMALIZED space during dynamics. We de-normalize to do
physics, integrate, re-normalize.

Lane propagation (方案 2, per user request):
  Step 1: analytical rigid-body transform of all 10 waypoints using the car's
          body-frame motion (dx_b, dy_b, dpsi).
  Step 2: the "front-most" waypoint (was at the furthest s ahead) drifts back,
          so the actual look-ahead window shrinks each step. A small MLP
          predicts a NEW front waypoint by extrapolating from the last two.
          This lets us always maintain 10 waypoints with full look-ahead.

Body-frame convention here:
  wp column 0 = x_b (lateral, PIWM's "vx_b" direction)
  wp column 1 = y_b (forward, PIWM's "vy_b" direction)
"""

import torch
import torch.nn as nn
from config import FULL_STATE_DIM, DT, PHYSICS_STD_REL, PHYSICS_MEAN_REL
from lane_utils import N_LANE_WP, LANE_DIM, LANE_MEAN, LANE_STD
from models.dynamics_bicycle_v4 import BicycleDynamicsV4


class MLP(nn.Module):
    def __init__(self, i, o, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(i, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, o),
        )
    def forward(self, x): return self.net(x)


class LaneAugmentedV5(nn.Module):
    """11 (car) + 20 (lane) = 31-dim latent.

    Uses v4 for car dynamics, analytical rigid transform for lane, plus a
    small learned "front extension" net.
    """
    CAR_DIM  = FULL_STATE_DIM    # 11
    LANE_DIM = LANE_DIM          # 20
    TOTAL    = CAR_DIM + LANE_DIM  # 31

    def __init__(self):
        super().__init__()
        self.car_dyn = BicycleDynamicsV4()

        # Front-extension MLP: predict increment to the FURTHEST waypoint to
        # compensate slow backward drift (waypoint at s=40 slowly creeps toward
        # car as rollout proceeds). Input: car + action + last two wp.
        self.front_extend = MLP(self.CAR_DIM + 3 + 4, 2, h=32)

        # Small learned residual for all lane waypoints (normalized units)
        self.lane_residual = MLP(self.TOTAL + 3, self.LANE_DIM, h=64)

        # Normalization buffers for the lane latent (20 dims)
        self.register_buffer("lane_mean",
                             torch.tensor(LANE_MEAN, dtype=torch.float32))
        self.register_buffer("lane_std",
                             torch.tensor(LANE_STD, dtype=torch.float32))

    def _denorm_lane(self, wp_norm):
        """(..., 20) -> (..., 20) real body-frame coords."""
        return wp_norm * self.lane_std + self.lane_mean

    def _norm_lane(self, wp_real):
        return (wp_real - self.lane_mean) / self.lane_std

    def forward(self, z, action):
        """
        Args:
            z: (B, 31) normalized [car(11), lane(20)]
            action: (B, 3)
        Returns:
            z_next: (B, 31)
            car_residual: (B, 11) for logging
        """
        z_car_norm = z[:, :self.CAR_DIM]          # (B, 11)
        z_lane_norm = z[:, self.CAR_DIM:]         # (B, 20)

        # 1) Car dynamics (v4)
        z_car_next_norm, car_res = self.car_dyn(z_car_norm, action)

        # 2) Compute body-frame motion (dx_b, dy_b, dpsi) from NORMALIZED car state.
        #    We need REAL values for (vx, vy, omega, yaw) then project to body.
        #    Convention: forward = (-sin(yaw), cos(yaw)) in world = body+y.
        car_mean = self.car_dyn.mean; car_std = self.car_dyn.std
        z_car_real = z_car_norm * car_std + car_mean
        yaw = z_car_real[:, 2]
        vx_w = z_car_real[:, 3]; vy_w = z_car_real[:, 4]
        omega = z_car_real[:, 5]

        # World-frame body axes (PIWM's convention body+y = forward)
        c = torch.cos(yaw); s = torch.sin(yaw)
        # Car displacement in body frame over DT:
        # body_x = lateral = v_world · (cos, sin)      (PIWM code calls this "vx_b")
        # body_y = forward = v_world · (-sin, cos)     (PIWM code calls this "vy_b")
        dx_b = (vx_w * c + vy_w * s) * DT   # lateral displacement
        dy_b = (-vx_w * s + vy_w * c) * DT  # forward displacement
        dpsi = omega * DT

        # 3) Lane waypoints: rigid-body inverse transform
        wp_real = self._denorm_lane(z_lane_norm).reshape(-1, N_LANE_WP, 2)
        wp_shifted = wp_real.clone()
        wp_shifted[..., 0] = wp_real[..., 0] - dx_b.unsqueeze(-1)
        wp_shifted[..., 1] = wp_real[..., 1] - dy_b.unsqueeze(-1)
        # Rotate by -dpsi
        cp = torch.cos(-dpsi).unsqueeze(-1)
        sp = torch.sin(-dpsi).unsqueeze(-1)
        x_new = wp_shifted[..., 0] * cp - wp_shifted[..., 1] * sp
        y_new = wp_shifted[..., 0] * sp + wp_shifted[..., 1] * cp
        wp_rot = torch.stack([x_new, y_new], dim=-1)  # (B, 10, 2)

        # 4) "方案 2" front extension: per-step DT motion is MUCH smaller than
        #    the 5-unit sampling interval (typical DT=1/50 s × 30 m/s = 0.6 units).
        #    So we do NOT shift slots. Instead, the furthest waypoint (wp[-1],
        #    nominally at s≈40) slowly drifts backward over many rollout steps.
        #    A small MLP predicts a forward correction for wp[-1] each step,
        #    letting the look-ahead "track" forward over long rollouts.
        last_two = wp_rot[:, -2:, :].reshape(-1, 4)                 # (B, 4)
        inp_ext = torch.cat([z_car_norm, action, last_two], dim=-1) # (B, 18)
        # front_extend outputs a small per-step delta applied to wp[-1] only
        delta_front = self.front_extend(inp_ext) * DT               # (B, 2)
        wp_updated = wp_rot.clone()
        wp_updated[:, -1] = wp_rot[:, -1] + delta_front

        # 5) Small learned residual on ALL 10 wp (normalized space, scaled down)
        wp_norm_intermediate = self._norm_lane(wp_updated.reshape(-1, self.LANE_DIM))
        lane_inp = torch.cat([z_car_next_norm, wp_norm_intermediate, action], dim=-1)
        lane_res = self.lane_residual(lane_inp) * 0.1   # keep small
        wp_next_norm = wp_norm_intermediate + lane_res

        # 6) Concat + return
        z_next = torch.cat([z_car_next_norm, wp_next_norm], dim=-1)
        return z_next, car_res
