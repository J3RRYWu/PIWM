"""PIWM-donkey-v3 — closes the remaining gap to GOKU.

After v2 fixed the norm-space arithmetic, PIWM-donkey-v2 was still ~35%
worse than GOKU on single-step ω. Comparing class definitions found the
last missing piece:

  GOKU's force/wheel MLPs take the FULL 31-dim state + 3-dim action (= 34).
  PIWM-kin/kin2/donkey/v2 force/wheel MLPs took only the 11-dim car state
  + action (= 14). The car dynamics MLPs had no access to the lane shape
  ahead, so couldn't condition on upcoming curvature when predicting Δv.

This version fixes that: force_net and wheel_net see (z31, action).
Everything else (norm-space integration, no Ackermann, lane rigid+resample,
big lane residual) is unchanged.

At this point PIWM-donkey-v3's car dynamics is structurally identical to
GOKU's — the question this experiment now answers is whether PIWM's lane
resample mechanism is worth anything on top, vs GOKU's MLP-only lane
prediction.
"""
import torch
import torch.nn as nn
from config import FULL_STATE_DIM, DT, PHYSICS_STD_REL, PHYSICS_MEAN_REL
from lane_utils import (N_LANE_WP, LANE_DIM, LANE_MEAN, LANE_STD, LANE_S_SAMPLES)


class MLP(nn.Module):
    def __init__(self, i, o, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(i, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, o),
        )
    def forward(self, x): return self.net(x)


TOTAL_DIM = FULL_STATE_DIM + LANE_DIM             # 31
DS_STEP = float(LANE_S_SAMPLES[1] - LANE_S_SAMPLES[0])


class LaneAugmentedDonkeyV3(nn.Module):
    CAR_DIM  = FULL_STATE_DIM
    LANE_DIM = LANE_DIM
    TOTAL    = TOTAL_DIM

    def __init__(self):
        super().__init__()
        # Force/wheel MLPs see the FULL 31-dim state + action — same input
        # as GOKU.  h=64 (same as GOKU).
        full_inp = TOTAL_DIM + 3
        self.force_net = MLP(full_inp, 3, h=64)         # dvx, dvy, dω
        self.wheel_net = MLP(full_inp, 5, h=64)         # dw0..3, dδ

        # Lane residual still gets full 31 + action.
        self.lane_residual = MLP(full_inp, LANE_DIM, h=128)

        # Norm buffers.
        self.register_buffer("car_mean", torch.tensor(PHYSICS_MEAN_REL, dtype=torch.float32))
        self.register_buffer("car_std",  torch.tensor(PHYSICS_STD_REL,  dtype=torch.float32))
        self.register_buffer("lane_mean", torch.tensor(LANE_MEAN, dtype=torch.float32))
        self.register_buffer("lane_std",  torch.tensor(LANE_STD,  dtype=torch.float32))

        std = PHYSICS_STD_REL; mean = PHYSICS_MEAN_REL
        self.register_buffer("dt_vx_x",   torch.tensor(float(std[3] * DT / std[0])))
        self.register_buffer("dt_vy_y",   torch.tensor(float(std[4] * DT / std[1])))
        self.register_buffer("dt_om_yaw", torch.tensor(float(std[5] * DT / std[2])))
        self.register_buffer("m_vx_x",    torch.tensor(float(mean[3] * DT / std[0])))
        self.register_buffer("m_vy_y",    torch.tensor(float(mean[4] * DT / std[1])))
        self.register_buffer("m_om_yaw",  torch.tensor(float(mean[5] * DT / std[2])))

    def _denorm_lane(self, wp_norm): return wp_norm * self.lane_std + self.lane_mean
    def _norm_lane(self, wp_real):   return (wp_real - self.lane_mean) / self.lane_std

    def forward(self, z, action):
        z_car_norm  = z[:, :self.CAR_DIM]      # (B, 11) normalized
        z_lane_norm = z[:, self.CAR_DIM:]      # (B, 20) normalized

        # MLP input: full state + action.
        full_inp = torch.cat([z, action], dim=-1)

        # ----- Car dynamics (norm space, GOKU-style integration) -----
        x   = z_car_norm[:, 0]; y   = z_car_norm[:, 1]; yaw = z_car_norm[:, 2]
        vx  = z_car_norm[:, 3]; vy  = z_car_norm[:, 4]; om  = z_car_norm[:, 5]
        wheels = z_car_norm[:, 6:10]; steer = z_car_norm[:, 10]

        df = self.force_net(full_inp)
        vx_n = vx + df[:, 0]; vy_n = vy + df[:, 1]; om_n = om + df[:, 2]

        x_n   = x   + vx  * self.dt_vx_x  + self.m_vx_x
        y_n   = y   + vy  * self.dt_vy_y  + self.m_vy_y
        yaw_n = yaw + om  * self.dt_om_yaw + self.m_om_yaw

        dw = self.wheel_net(full_inp)
        wh_n = wheels + dw[:, :4]
        st_n = steer  + dw[:, 4]

        z_car_next_norm = torch.stack([
            x_n, y_n, yaw_n, vx_n, vy_n, om_n,
            wh_n[:, 0], wh_n[:, 1], wh_n[:, 2], wh_n[:, 3], st_n
        ], dim=-1)

        # ----- Lane: rigid + resample + big residual (PIWM's last prior) -----
        # Need (dx_b, dy_b, dpsi) in REAL world units for the rigid xform.
        z_car_real = z_car_norm * self.car_std + self.car_mean
        yaw_real = z_car_real[:, 2]
        vx_w = z_car_real[:, 3]; vy_w = z_car_real[:, 4]; om_w = z_car_real[:, 5]
        c = torch.cos(yaw_real); s = torch.sin(yaw_real)
        dx_b = ( vx_w * c + vy_w * s) * DT
        dy_b = (-vx_w * s + vy_w * c) * DT
        dpsi = om_w * DT

        wp_real = self._denorm_lane(z_lane_norm).reshape(-1, N_LANE_WP, 2)
        wp_shifted = torch.stack([
            wp_real[..., 0] - dx_b.unsqueeze(-1),
            wp_real[..., 1] - dy_b.unsqueeze(-1),
        ], dim=-1)
        cp = torch.cos(-dpsi).unsqueeze(-1)
        sp = torch.sin(-dpsi).unsqueeze(-1)
        x_new = wp_shifted[..., 0] * cp - wp_shifted[..., 1] * sp
        y_new = wp_shifted[..., 0] * sp + wp_shifted[..., 1] * cp
        wp_rigid = torch.stack([x_new, y_new], dim=-1)

        alpha = (dy_b / DS_STEP).unsqueeze(-1).unsqueeze(-1)
        wp_resampled = torch.zeros_like(wp_rigid)
        wp_resampled[:, :-1] = (1 - alpha) * wp_rigid[:, :-1] + alpha * wp_rigid[:, 1:]
        tangent = wp_rigid[:, -1] - wp_rigid[:, -2]
        wp_resampled[:, -1] = wp_rigid[:, -1] + alpha.squeeze(-2) * tangent

        wp_norm_inter = self._norm_lane(wp_resampled.reshape(-1, self.LANE_DIM))
        lane_inp = torch.cat([z_car_next_norm, wp_norm_inter, action], dim=-1)
        lane_res = self.lane_residual(lane_inp)
        wp_next_norm = wp_norm_inter + lane_res

        z_next = torch.cat([z_car_next_norm, wp_next_norm], dim=-1)
        return z_next, torch.zeros_like(z_car_next_norm)
