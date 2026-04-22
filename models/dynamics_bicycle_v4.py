"""Bicycle Dynamics v4: DYNAMIC bicycle model with learned tire stiffness.

Ablation variant #3. Replaces kinematic bicycle with a dynamic (force-based)
bicycle model commonly used in vehicle dynamics:

  alpha_f = steer - atan2(vy + L_f*omega, vx)   (front slip angle)
  alpha_r =       - atan2(vy - L_r*omega, vx)   (rear slip angle)
  F_yf = -C_f * alpha_f                          (linear tire model)
  F_yr = -C_r * alpha_r
  dvx       = a_forward - (1/m)*F_yf*sin(steer) + vy*omega
  dvy       = (1/m)*(F_yf*cos(steer) + F_yr) - vx*omega
  domega    = (L_f*F_yf*cos(steer) - L_r*F_yr) / I_z

Learnable physical parameters (positive, bounded):
  - m        (mass)
  - I_z      (yaw inertia)
  - C_f, C_r (cornering stiffness)
  - L_f, L_r (distances from COM to front/rear axle)

Plus small learned residual on non-position dims.

This captures tire slip / drift behavior that the kinematic model CANNOT,
which is expected to help on CarRacing where the vehicle frequently
operates near the tire saturation limit.
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


def softplus_bounded(raw, lo, hi):
    """Map raw in R to (lo, hi) via shifted sigmoid."""
    return lo + (hi - lo) * torch.sigmoid(raw)


class BicycleDynamicsV4(nn.Module):
    """Dynamic bicycle model + masked residual."""

    def __init__(self):
        super().__init__()
        input_dim = FULL_STATE_DIM + 3

        # Physical parameters (all learnable, bounded to plausible ranges)
        self.raw_m    = nn.Parameter(torch.tensor(0.0))   # m   ∈ [500, 2000]
        self.raw_Iz   = nn.Parameter(torch.tensor(0.0))   # I_z ∈ [500, 5000]
        self.raw_Cf   = nn.Parameter(torch.tensor(0.0))   # C_f ∈ [5e4, 5e5]
        self.raw_Cr   = nn.Parameter(torch.tensor(0.0))   # C_r ∈ [5e4, 5e5]
        self.raw_Lf   = nn.Parameter(torch.tensor(0.0))   # L_f ∈ [1, 4]
        self.raw_Lr   = nn.Parameter(torch.tensor(0.0))   # L_r ∈ [1, 4]

        # Forward accel (drive force from gas/brake), 1-dim
        self.accel_net = MLP(input_dim, 1, h=64)
        # Wheel + steer dynamics
        self.wheel_net = MLP(input_dim, 5, h=64)
        # Small residual on non-position dims
        self.residual_net = MLP(input_dim, FULL_STATE_DIM, h=64)

        self.register_buffer("mean", torch.tensor(PHYSICS_MEAN_REL, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(PHYSICS_STD_REL, dtype=torch.float32))

        mask = torch.ones(FULL_STATE_DIM)
        mask[0] = 0.0; mask[1] = 0.0
        self.register_buffer("res_mask", mask)

    # Bounded physical parameters
    @property
    def m(self):  return softplus_bounded(self.raw_m,  500.,  2000.)
    @property
    def Iz(self): return softplus_bounded(self.raw_Iz, 500.,  5000.)
    @property
    def Cf(self): return softplus_bounded(self.raw_Cf, 5e4,   5e5)
    @property
    def Cr(self): return softplus_bounded(self.raw_Cr, 5e4,   5e5)
    @property
    def Lf(self): return softplus_bounded(self.raw_Lf, 1.0,   4.0)
    @property
    def Lr(self): return softplus_bounded(self.raw_Lr, 1.0,   4.0)

    def _denorm(self, z_norm): return z_norm * self.std + self.mean
    def _norm(self, z_real):   return (z_real - self.mean) / self.std

    def forward(self, z_norm, action):
        inp = torch.cat([z_norm, action], dim=-1)

        z = self._denorm(z_norm)
        x, y, yaw = z[:, 0], z[:, 1], z[:, 2]
        vx_w, vy_w, omega = z[:, 3], z[:, 4], z[:, 5]
        wheels = z[:, 6:10]
        steer = z[:, 10]

        c = torch.cos(yaw); s = torch.sin(yaw)

        # Transform world velocity to body frame
        vx_b =  vx_w * c + vy_w * s      # forward
        vy_b = -vx_w * s + vy_w * c      # lateral

        # Slip angles (protect division by zero at low speed)
        vx_safe = torch.where(vx_b.abs() > 0.5, vx_b, torch.full_like(vx_b, 0.5) * torch.sign(vx_b + 1e-6))
        alpha_f = steer - torch.atan2(vy_b + self.Lf * omega, vx_safe)
        alpha_r =       - torch.atan2(vy_b - self.Lr * omega, vx_safe)

        F_yf = -self.Cf * alpha_f
        F_yr = -self.Cr * alpha_r

        a_forward = self.accel_net(inp).squeeze(-1)  # longitudinal accel from gas/brake

        # Body-frame accelerations
        dvx_b = a_forward - (F_yf * torch.sin(steer)) / self.m + vy_b * omega
        dvy_b = (F_yf * torch.cos(steer) + F_yr) / self.m - vx_b * omega
        domega = (self.Lf * F_yf * torch.cos(steer) - self.Lr * F_yr) / self.Iz

        # Euler integrate
        vx_b_next = vx_b + dvx_b * DT
        vy_b_next = vy_b + dvy_b * DT
        omega_next = omega + domega * DT

        # Back to world frame
        vx_next = vx_b_next * c - vy_b_next * s
        vy_next = vx_b_next * s + vy_b_next * c

        # Position integration
        x_next = x + vx_w * DT
        y_next = y + vy_w * DT
        yaw_next = yaw + omega * DT

        dw = self.wheel_net(inp)
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

        residual = self.residual_net(inp) * self.res_mask
        z_next_norm = z_phys_norm + residual

        return z_next_norm, residual
