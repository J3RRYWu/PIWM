"""PIWM-v6: lane-augmented dynamics with RESAMPLE-based propagation.

Fixes v5's semantic drift: v5 kept original world points (they slid behind car
over 100 steps), v6 maintains "fixed-s ahead of car" semantics via resampling.

Algorithm (per step):
  1. Bicycle v4 updates car state.
  2. Rigid-body inverse transform of all 10 waypoints into new body frame.
  3. Resample: linearly interpolate to put each wp back at its target s-offset
     (α = dy_b / Δs where Δs = 5 units is the sampling interval).
  4. Front-most slot: linear extrapolation along tangent of last two wp.
  5. Small learned residual (scale 0.1).

All the previous v5 hyperparameters kept; only the dynamics propagation changes.
"""
import torch
import torch.nn as nn
from config import FULL_STATE_DIM, DT, PHYSICS_STD_REL, PHYSICS_MEAN_REL
from lane_utils import (N_LANE_WP, LANE_DIM, LANE_MEAN, LANE_STD, LANE_S_SAMPLES)
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


# Mean sampling interval between waypoints (Δs); used for resample alpha
# LANE_S_SAMPLES = [-5, 0, 5, 10, ..., 40]  =>  uniform Δs = 5
DS_STEP = float(LANE_S_SAMPLES[1] - LANE_S_SAMPLES[0])   # 5.0


class LaneAugmentedV6(nn.Module):
    CAR_DIM  = FULL_STATE_DIM
    LANE_DIM = LANE_DIM
    TOTAL    = CAR_DIM + LANE_DIM

    def __init__(self):
        super().__init__()
        self.car_dyn = BicycleDynamicsV4()
        self.lane_residual = MLP(self.TOTAL + 3, self.LANE_DIM, h=64)

        self.register_buffer("lane_mean",
                             torch.tensor(LANE_MEAN, dtype=torch.float32))
        self.register_buffer("lane_std",
                             torch.tensor(LANE_STD, dtype=torch.float32))

    def _denorm_lane(self, wp_norm): return wp_norm * self.lane_std + self.lane_mean
    def _norm_lane(self, wp_real):   return (wp_real - self.lane_mean) / self.lane_std

    def forward(self, z, action):
        z_car_norm = z[:, :self.CAR_DIM]
        z_lane_norm = z[:, self.CAR_DIM:]

        # 1) Car dynamics (v4)
        z_car_next_norm, car_res = self.car_dyn(z_car_norm, action)

        # 2) Body-frame motion: forward (dy_b) and lateral (dx_b), plus dpsi
        car_mean = self.car_dyn.mean; car_std = self.car_dyn.std
        z_car_real = z_car_norm * car_std + car_mean
        yaw = z_car_real[:, 2]
        vx_w = z_car_real[:, 3]; vy_w = z_car_real[:, 4]
        omega = z_car_real[:, 5]
        c = torch.cos(yaw); s = torch.sin(yaw)
        # PIWM convention: body+y = forward (= world (-sin yaw, cos yaw) direction)
        dx_b = (vx_w * c + vy_w * s) * DT   # body-x (lateral)
        dy_b = (-vx_w * s + vy_w * c) * DT  # body-y (forward)
        dpsi = omega * DT

        # 3) Rigid-body inverse transform
        wp_real = self._denorm_lane(z_lane_norm).reshape(-1, N_LANE_WP, 2)
        wp_shifted = torch.stack([
            wp_real[..., 0] - dx_b.unsqueeze(-1),
            wp_real[..., 1] - dy_b.unsqueeze(-1),
        ], dim=-1)
        cp = torch.cos(-dpsi).unsqueeze(-1)
        sp = torch.sin(-dpsi).unsqueeze(-1)
        x_new = wp_shifted[..., 0] * cp - wp_shifted[..., 1] * sp
        y_new = wp_shifted[..., 0] * sp + wp_shifted[..., 1] * cp
        wp_rigid = torch.stack([x_new, y_new], dim=-1)   # (B, 10, 2)

        # 4) RESAMPLE: each slot i should be at body-y = LANE_S_SAMPLES[i]
        #    After rigid transform, slot i is (approximately) at LANE_S_SAMPLES[i] - dy_b
        #    To get back to target, use linear interpolation between slot i and slot i+1.
        #    alpha per sample = dy_b / DS_STEP
        alpha = (dy_b / DS_STEP).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        wp_resampled = torch.zeros_like(wp_rigid)
        # Slots 0..N-2 interpolate with the next slot
        wp_resampled[:, :-1] = (1 - alpha) * wp_rigid[:, :-1] + alpha * wp_rigid[:, 1:]
        # Front slot: extrapolate along tangent of last two wp
        tangent = wp_rigid[:, -1] - wp_rigid[:, -2]          # (B, 2)
        wp_resampled[:, -1] = wp_rigid[:, -1] + alpha.squeeze(-2) * tangent

        # 5) Small learned residual (normalized space)
        wp_norm_intermediate = self._norm_lane(wp_resampled.reshape(-1, self.LANE_DIM))
        lane_inp = torch.cat([z_car_next_norm, wp_norm_intermediate, action], dim=-1)
        lane_res = self.lane_residual(lane_inp) * 0.1
        wp_next_norm = wp_norm_intermediate + lane_res

        z_next = torch.cat([z_car_next_norm, wp_next_norm], dim=-1)
        return z_next, car_res
