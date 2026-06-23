"""PIWM-donkey-v4 = v3 + Change-1 (damped state recurrence) + Change-2 (bounded
front-slot resample).  [This is PIWM-v6, donkey iteration 6.]

Motivation (from the 60-agent improvement analysis, verified against the K=8 /
K=32 benchmarks and per-dim diagnostics):

  The K=8 composite gap (0.152 vs GOKU 0.137) AND the K=32 collapse
  (pos=369 vs GOKU 118, yaw=44 vs 6) both live in the CAR block, not the lane
  resample.  In v3 the velocity / yaw-rate recurrence is

      vx_n = vx + df[0];  vy_n = vy + df[1];  om_n = om + df[2]

  i.e. the homogeneous part has coefficient 1.0 (no leak).  Over 100 rollout
  steps the omega residual integrates into yaw without contraction, so yaw
  drift compounds and drives the position error.  Lines x_n/y_n/yaw_n integrate
  open-loop from these, and the lane resample (wp only) cannot touch the car
  position metric at all.

CHANGE 1 — learned per-channel damping on (vx, vy, omega):

      g = sigmoid(decay)            # (3,) in (0,1)
      vx_n = g0*vx + df[0]          # spectral radius g<1  ->  bounded recurrence
      ...

  `decay` is initialised at logit(0.98) so g≈0.98: K=8 behaviour is nearly
  unchanged (only the long tail is damped), and the parameter is LEARNABLE so
  if a channel needs to retain a sustained value (e.g. steady-state omega in a
  long turn) the optimiser can push g->1 on that channel.  This is the
  correctly-targeted version of "bound the residual": we damp the STATE
  RECURRENCE, not the residual magnitude, so single-step accuracy (what K=8
  needs) is preserved.

CHANGE 2 — bounded front-slot resample (clamp-to-edge):

  v3's front waypoint used unbounded forward tangent extrapolation
      wp[-1] = rigid[-1] + alpha*(rigid[-1] - rigid[-2])
  which is the only manifold-escape operator in the lane head: if the rigid
  waypoints drift far during a long rollout, (rigid[-1]-rigid[-2]) grows and is
  re-amplified every step.  We replace it with clamp-to-edge (hold): the front
  slot keeps its rigid value, leaving it at most alpha*DS_STEP (~0.13 working
  units in-distribution) short of its target — a correction the learned
  lane_residual absorbs.  NOTE: we deliberately do NOT use the analysis's
  proposed sign-flip `rigid[-1] - alpha*tangent`; that is geometrically
  backwards (pushes the farthest lookahead point behind the car) and only
  looked good on a frozen-weight test because it shrinks the point toward the
  origin.  Clamp-to-edge is the standard, geometrically-neutral bounded form.

Everything else identical to v3.
"""
import math
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
_DECAY_INIT = math.log(0.98 / 0.02)               # logit(0.98) ≈ 3.892


class LaneAugmentedDonkeyV4(nn.Module):
    CAR_DIM  = FULL_STATE_DIM
    LANE_DIM = LANE_DIM
    TOTAL    = TOTAL_DIM

    def __init__(self, damp=True, hold=True):
        super().__init__()
        # Ablation flags so the SAME code path can reproduce a v3-equivalent
        # baseline (damp=False, hold=False) on identical data/seed/schedule.
        #   damp=True  -> Change 1 (learned damped velocity/omega recurrence)
        #   hold=True  -> Change 2 (clamp-to-edge front-slot resample)
        self.damp = damp
        self.hold = hold

        full_inp = TOTAL_DIM + 3
        self.force_net = MLP(full_inp, 3, h=64)         # dvx, dvy, dω
        self.wheel_net = MLP(full_inp, 5, h=64)         # dw0..3, dδ
        self.lane_residual = MLP(full_inp, LANE_DIM, h=128)

        # CHANGE 1: per-channel learned damping for (vx, vy, omega).
        # g = sigmoid(decay); init ≈ 0.98 so K=8 ≈ unchanged.
        self.decay = nn.Parameter(torch.full((3,), float(_DECAY_INIT)))

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
        z_car_norm  = z[:, :self.CAR_DIM]
        z_lane_norm = z[:, self.CAR_DIM:]
        full_inp = torch.cat([z, action], dim=-1)

        x   = z_car_norm[:, 0]; y   = z_car_norm[:, 1]; yaw = z_car_norm[:, 2]
        vx  = z_car_norm[:, 3]; vy  = z_car_norm[:, 4]; om  = z_car_norm[:, 5]
        wheels = z_car_norm[:, 6:10]; steer = z_car_norm[:, 10]

        # ----- CHANGE 1: damped velocity / omega recurrence -----
        df = self.force_net(full_inp)
        if self.damp:
            g = torch.sigmoid(self.decay)                # (3,) in (0,1)
            vx_n = g[0] * vx + df[:, 0]
            vy_n = g[1] * vy + df[:, 1]
            om_n = g[2] * om + df[:, 2]
        else:                                            # v3-equivalent (g=1)
            vx_n = vx + df[:, 0]
            vy_n = vy + df[:, 1]
            om_n = om + df[:, 2]

        # Position / yaw integrate open-loop from the OLD velocity/omega
        # (unchanged — using OLD om keeps lane resample in lock-step).
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

        # ----- Lane: rigid + resample + big residual -----
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
        if self.hold:
            # CHANGE 2: bounded front slot — clamp-to-edge (hold) instead of
            # unbounded forward tangent extrapolation. Removes the only
            # manifold-escape operator in the lane head.
            wp_resampled[:, -1] = wp_rigid[:, -1]
        else:                                            # v3-equivalent extrapolation
            tangent = wp_rigid[:, -1] - wp_rigid[:, -2]
            wp_resampled[:, -1] = wp_rigid[:, -1] + alpha.squeeze(-2) * tangent

        wp_norm_inter = self._norm_lane(wp_resampled.reshape(-1, self.LANE_DIM))
        lane_inp = torch.cat([z_car_next_norm, wp_norm_inter, action], dim=-1)
        lane_res = self.lane_residual(lane_inp)
        wp_next_norm = wp_norm_inter + lane_res

        z_next = torch.cat([z_car_next_norm, wp_next_norm], dim=-1)
        return z_next, torch.zeros_like(z_car_next_norm)
