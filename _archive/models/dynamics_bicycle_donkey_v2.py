"""Donkey bicycle dynamics v2 — operates entirely in NORMALIZED space.

The previous `dynamics_bicycle_donkey.py` was structurally GOKU's twin but
denormalized → did real-space arithmetic → renormalized. That round-trip
shrinks the gradient on `force_net.weight` by 1/std[3] (≈11× for vx in donkey
units) because the loss is measured in normalized space. Adam can compensate
in principle, but in practice PIWM-donkey was uniformly 30-50% worse than
GOKU on single-step MSE for vx, vy, ω, wheels, δ — exactly the dims the MLP
must predict.

This version mirrors GOKU's normalization handling:
  - State stays normalized throughout.
  - Position/yaw integration uses GOKU's precomputed affine scale factors
    `dt_vx_x = std[3] * DT / std[0]` etc., so the integration is exact in
    normalized space.
  - Force/wheel MLPs output deltas directly in normalized space (same
    output magnitude regime as GOKU).

Same physics priors retained: position/yaw integration. No Ackermann.
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


class BicycleDynamicsDonkeyV2(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = FULL_STATE_DIM + 3

        self.force_net = MLP(input_dim, 3, h=64)
        self.wheel_net = MLP(input_dim, 5, h=64)

        std = PHYSICS_STD_REL; mean = PHYSICS_MEAN_REL
        # Precomputed normalized-space integrator constants (GOKU pattern).
        self.register_buffer("dt_vx_x",  torch.tensor(float(std[3] * DT / std[0])))
        self.register_buffer("dt_vy_y",  torch.tensor(float(std[4] * DT / std[1])))
        self.register_buffer("dt_om_yaw", torch.tensor(float(std[5] * DT / std[2])))
        self.register_buffer("m_vx_x",   torch.tensor(float(mean[3] * DT / std[0])))
        self.register_buffer("m_vy_y",   torch.tensor(float(mean[4] * DT / std[1])))
        self.register_buffer("m_om_yaw", torch.tensor(float(mean[5] * DT / std[2])))

        # Kept for API compatibility (lane wrapper denormalizes via these).
        self.register_buffer("mean", torch.tensor(PHYSICS_MEAN_REL, dtype=torch.float32))
        self.register_buffer("std",  torch.tensor(PHYSICS_STD_REL,  dtype=torch.float32))

    def forward(self, z_norm, action):
        inp = torch.cat([z_norm, action], dim=-1)
        x, y, yaw = z_norm[:, 0], z_norm[:, 1], z_norm[:, 2]
        vx, vy, omega = z_norm[:, 3], z_norm[:, 4], z_norm[:, 5]
        wheels = z_norm[:, 6:10]
        steer = z_norm[:, 10]

        # Force + wheel MLPs output deltas in normalized space.
        df = self.force_net(inp)
        vx_n    = vx    + df[:, 0]
        vy_n    = vy    + df[:, 1]
        omega_n = omega + df[:, 2]

        # Position / yaw integration in normalized space.
        x_n   = x   + vx    * self.dt_vx_x  + self.m_vx_x
        y_n   = y   + vy    * self.dt_vy_y  + self.m_vy_y
        yaw_n = yaw + omega * self.dt_om_yaw + self.m_om_yaw

        dw = self.wheel_net(inp)
        wheels_n = wheels + dw[:, :4]
        steer_n  = steer  + dw[:, 4]

        z_next_norm = torch.stack([
            x_n, y_n, yaw_n, vx_n, vy_n, omega_n,
            wheels_n[:, 0], wheels_n[:, 1],
            wheels_n[:, 2], wheels_n[:, 3], steer_n
        ], dim=-1)
        residual = torch.zeros_like(z_next_norm)
        return z_next_norm, residual
