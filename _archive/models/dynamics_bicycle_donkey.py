"""Donkey bicycle dynamics — Ackermann removed.

Diagnostic showed:
  - Single-step ω fit:  PIWM-kin2 0.129  vs GOKU 0.087  (PIWM 48% worse)
  - Single-step δ fit:  PIWM-kin2 0.341  vs GOKU 0.282  (PIWM 21% worse)
  - 8-step yaw MSE:     PIWM-kin2 0.171  vs GOKU 0.058  (3× worse)
  - 8-step x   MSE:     PIWM-kin2 0.147  vs GOKU 0.065  (2.3× worse)

The chain `δ_noisy → ω_kin = v·tan(δ)/L → yaw → x` blows up. δ in phys[10] is
reconstructed via `atan(ω·L/v)` from noisy ω and v, so the Ackermann roundtrip
amplifies measurement noise. Donkey hardware doesn't actually log the physical
steering angle — there is no clean signal there to trust.

This module **drops the Ackermann constraint entirely** and lets ω evolve like
any other state via a learned residual. Physics priors retained:
  - x' = vx      (exact world-frame position integration)
  - y' = vy
  - yaw' = ω    (yaw integrates with OLD ω so the lane resample
                  in dynamics_lane_v6_donkey stays in lock-step)
Learned via MLPs:
  - dvx, dvy, dω   from force_net(state, action)
  - d wheels, dδ   from wheel_net(state, action)

Functionally this is GOKU's car-part dynamics; the PIWM-specific
distinction now lives entirely in the lane resample wrapper. The question
the experiment answers: "is PIWM's lane resample worth anything once the
broken Ackermann is removed?"
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


class BicycleDynamicsDonkey(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = FULL_STATE_DIM + 3                 # 11 + 3

        self.force_net = MLP(input_dim, 3, h=64)       # dvx, dvy, dω
        self.wheel_net = MLP(input_dim, 5, h=64)       # dw0..3, dδ

        self.register_buffer("mean", torch.tensor(PHYSICS_MEAN_REL, dtype=torch.float32))
        self.register_buffer("std",  torch.tensor(PHYSICS_STD_REL,  dtype=torch.float32))

    def _denorm(self, z_norm): return z_norm * self.std + self.mean
    def _norm(self, z_real):   return (z_real - self.mean) / self.std

    def forward(self, z_norm, action):
        inp = torch.cat([z_norm, action], dim=-1)

        z = self._denorm(z_norm)
        x, y, yaw = z[:, 0], z[:, 1], z[:, 2]
        vx, vy, omega = z[:, 3], z[:, 4], z[:, 5]
        wheels = z[:, 6:10]
        steer = z[:, 10]                              # noisy reconstruction — only used
                                                       # as MLP input, never in equations.

        # Free residuals for velocity and angular velocity.
        df = self.force_net(inp)
        vx_next    = vx    + df[:, 0]
        vy_next    = vy    + df[:, 1]
        omega_next = omega + df[:, 2]

        # Physics-prior integration: position + yaw, using OLD velocity/ω
        # so the lane rigid+resample step uses the same dpsi this car step does.
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
        residual = torch.zeros_like(z_next_norm)       # API compat (no residual head)
        return z_next_norm, residual
