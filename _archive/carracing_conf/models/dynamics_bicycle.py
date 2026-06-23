"""Bicycle model + Residual Dynamics.

Replaces the current "position = velocity * dt" kinematic with bicycle model:
    yaw_rate  = (v_forward / L) * tan(steer)    # geometric constraint
    v_forward = |(vx, vy) projected onto yaw direction|

Learnable components (smaller than current version):
  - acceleration (1D along forward direction)
  - lateral slip (2D deviation from pure bicycle)
  - wheel speeds / steering angle delta
  - residual correction (11-dim, small regularization)

Operates in NORMALIZED (relative-coord) space via de-norm → physics → re-norm.
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


class BicycleDynamics(nn.Module):
    """Bicycle kinematics + learned residual forces.

    State (normalized, 11-dim): [x, y, yaw, vx, vy, omega, w0..w3, steer]
    Action (3-dim): [steer_cmd, gas, brake]
    """

    def __init__(self, init_L=8.0):
        super().__init__()
        input_dim = FULL_STATE_DIM + 3

        # Learnable wheelbase (in real units)
        # CarRacing track unit ~1 means about 1m; typical RC car wheelbase 2-10 units
        self.log_L = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init_L)))))

        # Learned deviations from pure bicycle
        self.accel_net = MLP(input_dim, 1, h=64)           # forward acceleration (1D)
        self.slip_net = MLP(input_dim, 2, h=64)            # lateral slip (vx, vy deviation)
        self.yaw_rate_res = MLP(input_dim, 1, h=32)        # omega deviation from bicycle
        self.wheel_net = MLP(input_dim, 5, h=64)           # wheels + steer
        self.residual_net = MLP(input_dim, FULL_STATE_DIM, h=64)  # final correction

        # Register normalization buffers (real ↔ normalized space)
        self.register_buffer("mean", torch.tensor(PHYSICS_MEAN_REL, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(PHYSICS_STD_REL, dtype=torch.float32))

    @property
    def L(self):
        return torch.exp(self.log_L)

    def _denorm(self, z_norm):
        return z_norm * self.std + self.mean

    def _norm(self, z_real):
        return (z_real - self.mean) / self.std

    def forward(self, z_norm, action):
        """
        Args:
            z_norm: (B, 11) normalized state
            action: (B, 3)
        Returns:
            z_next_norm: (B, 11)
            residual: (B, 11) residual correction magnitude (for logging)
        """
        inp = torch.cat([z_norm, action], dim=-1)

        # Denormalize to real-world quantities
        z = self._denorm(z_norm)
        x, y, yaw = z[:, 0], z[:, 1], z[:, 2]
        vx, vy, omega = z[:, 3], z[:, 4], z[:, 5]
        wheels = z[:, 6:10]
        steer = z[:, 10]

        c = torch.cos(yaw); s = torch.sin(yaw)

        # Forward speed (project world velocity onto yaw direction)
        v_forward = vx * c + vy * s
        v_lateral = -vx * s + vy * c

        # --- Bicycle geometric constraint ---
        # yaw_rate_bike = (v_forward / L) * tan(steer)
        L = self.L + 1e-3   # avoid division by zero
        yaw_rate_bike = v_forward / L * torch.tan(steer)

        # --- Learned deviations ---
        a_forward = self.accel_net(inp).squeeze(-1)        # forward accel
        slip = self.slip_net(inp)                           # (B, 2) - (dvx_slip, dvy_slip)
        yaw_rate_dev = self.yaw_rate_res(inp).squeeze(-1)   # yaw-rate residual
        dw = self.wheel_net(inp)                            # (B, 5)

        # New forward/lateral speed
        v_forward_next = v_forward + a_forward * DT
        # Rotate back to world frame, keeping lateral component + learned slip
        vx_next = v_forward_next * c - v_lateral * s + slip[:, 0]
        vy_next = v_forward_next * s + v_lateral * c + slip[:, 1]

        # Yaw evolution: bicycle prior as delta, evolve from current omega
        omega_next = omega + (yaw_rate_bike - omega) + yaw_rate_dev

        # Position integration (world frame)
        x_next = x + vx * DT
        y_next = y + vy * DT
        yaw_next = yaw + omega * DT   # use current omega (not omega_next) for integration

        # Wheels and steering
        wheels_next = wheels + dw[:, :4] * DT
        steer_next = steer + dw[:, 4] * DT

        # Assemble real-space prediction
        z_physics = torch.stack([
            x_next, y_next, yaw_next,
            vx_next, vy_next, omega_next,
            wheels_next[:, 0], wheels_next[:, 1],
            wheels_next[:, 2], wheels_next[:, 3],
            steer_next,
        ], dim=-1)

        # Back to normalized space
        z_phys_norm = self._norm(z_physics)

        # Residual correction in normalized space
        residual = self.residual_net(inp)
        z_next_norm = z_phys_norm + residual

        return z_next_norm, residual
