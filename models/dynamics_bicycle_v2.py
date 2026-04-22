"""Bicycle Dynamics v2: residual MASKED (no modification to position x, y).

Ablation variant #1. Difference from dynamics_bicycle.py:
  - Residual network output for dims [x, y] is zeroed.
  - Position is strictly integrated from velocity via kinematics.

Rationale: the physics prior `x_{t+1} = x_t + vx*DT` is a hard constraint.
Letting the residual network modify x/y defeats the purpose of a
physics-informed model and allows short-term error "patching" that
compounds over long rollouts.
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
    def forward(self, x):
        return self.net(x)


class BicycleDynamicsV2(nn.Module):
    """Same as BicycleDynamics but residual is zeroed on [x, y] dims."""

    def __init__(self, init_L=8.0):
        super().__init__()
        input_dim = FULL_STATE_DIM + 3

        self.log_L = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init_L)))))

        self.accel_net = MLP(input_dim, 1, h=64)
        self.slip_net = MLP(input_dim, 2, h=64)
        self.yaw_rate_res = MLP(input_dim, 1, h=32)
        self.wheel_net = MLP(input_dim, 5, h=64)
        self.residual_net = MLP(input_dim, FULL_STATE_DIM, h=64)

        self.register_buffer("mean", torch.tensor(PHYSICS_MEAN_REL, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(PHYSICS_STD_REL, dtype=torch.float32))

        # Position mask: zero on indices 0, 1 (x, y); one elsewhere
        mask = torch.ones(FULL_STATE_DIM)
        mask[0] = 0.0   # x
        mask[1] = 0.0   # y
        self.register_buffer("res_mask", mask)

    @property
    def L(self):
        return torch.exp(self.log_L)

    def _denorm(self, z_norm): return z_norm * self.std + self.mean
    def _norm(self, z_real):   return (z_real - self.mean) / self.std

    def forward(self, z_norm, action):
        inp = torch.cat([z_norm, action], dim=-1)

        z = self._denorm(z_norm)
        x, y, yaw = z[:, 0], z[:, 1], z[:, 2]
        vx, vy, omega = z[:, 3], z[:, 4], z[:, 5]
        wheels = z[:, 6:10]
        steer = z[:, 10]

        c = torch.cos(yaw); s = torch.sin(yaw)
        v_forward = vx * c + vy * s
        v_lateral = -vx * s + vy * c

        L = self.L + 1e-3
        yaw_rate_bike = v_forward / L * torch.tan(steer)

        a_forward = self.accel_net(inp).squeeze(-1)
        slip = self.slip_net(inp)
        yaw_rate_dev = self.yaw_rate_res(inp).squeeze(-1)
        dw = self.wheel_net(inp)

        v_forward_next = v_forward + a_forward * DT
        vx_next = v_forward_next * c - v_lateral * s + slip[:, 0]
        vy_next = v_forward_next * s + v_lateral * c + slip[:, 1]

        omega_next = omega + (yaw_rate_bike - omega) + yaw_rate_dev

        x_next = x + vx * DT
        y_next = y + vy * DT
        yaw_next = yaw + omega * DT

        wheels_next = wheels + dw[:, :4] * DT
        steer_next = steer + dw[:, 4] * DT

        z_physics = torch.stack([
            x_next, y_next, yaw_next,
            vx_next, vy_next, omega_next,
            wheels_next[:, 0], wheels_next[:, 1],
            wheels_next[:, 2], wheels_next[:, 3],
            steer_next,
        ], dim=-1)

        z_phys_norm = self._norm(z_physics)

        # Residual correction, MASKED so position (x, y) is untouched
        residual = self.residual_net(inp) * self.res_mask
        z_next_norm = z_phys_norm + residual

        return z_next_norm, residual
