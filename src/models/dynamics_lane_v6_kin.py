"""PIWM-v6 with kinematic bicycle (donkeycar variant).

Identical to dynamics_lane_v6 except the inner car dynamics is
BicycleDynamicsKin (pure kinematic, no slip) instead of BicycleDynamicsV4
(dynamic, with tire forces). Justified by the donkey being a real low-speed
RC car that does not drift, and by the V4 parameter bounds (m∈[500,2000]kg,
Cf,Cr∈[5e4,5e5]N/rad) being wildly off-scale for it.

Everything else — the rigid-body inverse transform, resample step, learned
lane residual, normalization buffers — is unchanged.
"""
import torch
import torch.nn as nn
from config import FULL_STATE_DIM, DT
from lane_utils import (N_LANE_WP, LANE_DIM, LANE_MEAN, LANE_STD, LANE_S_SAMPLES)
from models.dynamics_bicycle_kin import BicycleDynamicsKin


class MLP(nn.Module):
    def __init__(self, i, o, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(i, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, o),
        )
    def forward(self, x): return self.net(x)


DS_STEP = float(LANE_S_SAMPLES[1] - LANE_S_SAMPLES[0])   # uniform spacing


class LaneAugmentedV6Kin(nn.Module):
    CAR_DIM  = FULL_STATE_DIM
    LANE_DIM = LANE_DIM
    TOTAL    = CAR_DIM + LANE_DIM

    def __init__(self):
        super().__init__()
        self.car_dyn = BicycleDynamicsKin()
        self.lane_residual = MLP(self.TOTAL + 3, self.LANE_DIM, h=64)

        self.register_buffer("lane_mean", torch.tensor(LANE_MEAN, dtype=torch.float32))
        self.register_buffer("lane_std",  torch.tensor(LANE_STD,  dtype=torch.float32))

    def _denorm_lane(self, wp_norm): return wp_norm * self.lane_std + self.lane_mean
    def _norm_lane(self, wp_real):   return (wp_real - self.lane_mean) / self.lane_std

    def forward(self, z, action):
        z_car_norm  = z[:, :self.CAR_DIM]
        z_lane_norm = z[:, self.CAR_DIM:]

        # 1) Car dynamics (kinematic bicycle).
        z_car_next_norm, car_res = self.car_dyn(z_car_norm, action)

        # 2) Body-frame motion delta (PIWM body+y = forward).
        car_mean = self.car_dyn.mean; car_std = self.car_dyn.std
        z_car_real = z_car_norm * car_std + car_mean
        yaw = z_car_real[:, 2]
        vx_w = z_car_real[:, 3]; vy_w = z_car_real[:, 4]
        omega = z_car_real[:, 5]
        c = torch.cos(yaw); s = torch.sin(yaw)
        dx_b = ( vx_w * c + vy_w * s) * DT      # lateral
        dy_b = (-vx_w * s + vy_w * c) * DT      # forward
        dpsi = omega * DT

        # 3) Rigid-body inverse transform of waypoints into new body frame.
        wp_real = self._denorm_lane(z_lane_norm).reshape(-1, N_LANE_WP, 2)
        wp_shifted = torch.stack([
            wp_real[..., 0] - dx_b.unsqueeze(-1),
            wp_real[..., 1] - dy_b.unsqueeze(-1),
        ], dim=-1)
        cp = torch.cos(-dpsi).unsqueeze(-1)
        sp = torch.sin(-dpsi).unsqueeze(-1)
        x_new = wp_shifted[..., 0] * cp - wp_shifted[..., 1] * sp
        y_new = wp_shifted[..., 0] * sp + wp_shifted[..., 1] * cp
        wp_rigid = torch.stack([x_new, y_new], dim=-1)             # (B, 10, 2)

        # 4) Resample to maintain "fixed s-offset ahead of car" semantics.
        alpha = (dy_b / DS_STEP).unsqueeze(-1).unsqueeze(-1)
        wp_resampled = torch.zeros_like(wp_rigid)
        wp_resampled[:, :-1] = (1 - alpha) * wp_rigid[:, :-1] + alpha * wp_rigid[:, 1:]
        tangent = wp_rigid[:, -1] - wp_rigid[:, -2]
        wp_resampled[:, -1] = wp_rigid[:, -1] + alpha.squeeze(-2) * tangent

        # 5) Small learned residual.
        wp_norm_intermediate = self._norm_lane(wp_resampled.reshape(-1, self.LANE_DIM))
        lane_inp = torch.cat([z_car_next_norm, wp_norm_intermediate, action], dim=-1)
        lane_res = self.lane_residual(lane_inp) * 0.1
        wp_next_norm = wp_norm_intermediate + lane_res

        z_next = torch.cat([z_car_next_norm, wp_next_norm], dim=-1)
        return z_next, car_res
