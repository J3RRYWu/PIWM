"""Relaxed-kinematic bicycle — keeps Ackermann yaw-rate prior, frees vx/vy.

Motivation: a hard kinematic model `vx = -v_fwd · sin(yaw)` cannot reproduce
the small body-lateral velocity that appears in real donkey data due to
finite-differenced position, servo lag, and 22 Hz quantization. GOKU/DVBF
beat the strict kinematic PIWM on the donkey val set largely because they
let vx, vy evolve freely.

This module retains:
  - Position integration exactly (x' = vx, y' = vy)
  - Yaw integration with OLD omega (matches lane resample's dpsi)
  - Ackermann kinematic constraint on omega (yaw rate is a control variable
    in low-speed cars: it's almost entirely determined by v · tan(δ) / L)
  - One learnable wheelbase L

Frees:
  - vx, vy via a 3-dim force MLP (matches GOKU's pattern but with the
    kinematic ω still imposed)
  - Wheels, steer via the same wheel_net MLP as the strict-kin module

Same I/O signature as BicycleDynamicsKin so it drops into the v6 lane
wrapper unchanged.
"""
import torch
import torch.nn as nn
from config import FULL_STATE_DIM, DT, PHYSICS_STD_REL, PHYSICS_MEAN_REL


class MLP(nn.Module):
    def __init__(self, i, o, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(i, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, o),
        )
    def forward(self, x): return self.net(x)


def softplus_bounded(raw, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(raw)


class BicycleDynamicsKinRelaxed(nn.Module):
    def __init__(self, L_lo: float = 4.0, L_hi: float = 15.0):
        super().__init__()
        input_dim = FULL_STATE_DIM + 3

        self.raw_L = nn.Parameter(torch.tensor(0.0))
        self._L_lo, self._L_hi = float(L_lo), float(L_hi)

        # Force MLP predicts (dvx, dvy, dom_residual). The third channel is
        # added on top of the kinematic ω prediction below — small correction
        # to absorb servo lag / smoothing.
        self.force_net = MLP(input_dim, 3, h=64)
        self.wheel_net = MLP(input_dim, 5, h=64)

        self.register_buffer("mean", torch.tensor(PHYSICS_MEAN_REL, dtype=torch.float32))
        self.register_buffer("std",  torch.tensor(PHYSICS_STD_REL,  dtype=torch.float32))

    @property
    def L(self):
        return softplus_bounded(self.raw_L, self._L_lo, self._L_hi)

    def _denorm(self, z_norm): return z_norm * self.std + self.mean
    def _norm(self, z_real):   return (z_real - self.mean) / self.std

    def forward(self, z_norm, action):
        inp = torch.cat([z_norm, action], dim=-1)

        z = self._denorm(z_norm)
        x, y, yaw = z[:, 0], z[:, 1], z[:, 2]
        vx, vy, omega = z[:, 3], z[:, 4], z[:, 5]
        wheels = z[:, 6:10]
        steer = z[:, 10]

        # PIWM convention: body+y forward.
        c = torch.cos(yaw); sn = torch.sin(yaw)
        v_fwd = -vx * sn + vy * c

        # Ackermann yaw rate (kinematic prior on rotation).
        yaw_rate_kin = (v_fwd / (self.L + 1e-3)) * torch.tan(steer)

        # Learned residuals: (dvx, dvy, dom). Note dvx/dvy here are
        # WORLD-FRAME deltas like GOKU, not body-frame.
        df = self.force_net(inp)
        vx_next = vx    + df[:, 0]
        vy_next = vy    + df[:, 1]
        omega_next = yaw_rate_kin + df[:, 2]   # kin prior + small learned correction

        # Integrate position with OLD velocity; yaw with OLD omega so that
        # the lane resample's dpsi = omega * DT stays in lock-step.
        x_next   = x   + vx    * DT
        y_next   = y   + vy    * DT
        yaw_next = yaw + omega * DT

        dw = self.wheel_net(inp)
        wheels_next = wheels + dw[:, :4] * DT
        steer_next  = steer  + dw[:, 4]  * DT

        z_physics = torch.stack([
            x_next, y_next, yaw_next,
            vx_next, vy_next, omega_next,
            wheels_next[:, 0], wheels_next[:, 1],
            wheels_next[:, 2], wheels_next[:, 3],
            steer_next,
        ], dim=-1)

        z_next_norm = self._norm(z_physics)
        # `residual` returned for API compatibility with V4-based wrapper.
        # Zero here — no separate residual head.
        residual = torch.zeros_like(z_next_norm)
        return z_next_norm, residual
