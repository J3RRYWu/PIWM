"""Pure kinematic bicycle for low-speed real vehicles (donkeycar).

The dynamic bicycle (BicycleDynamicsV4) assumes meaningful tire slip and was
sized for ~1000 kg CarRacing-scale vehicles. A real donkey at <1 m/s shows
essentially zero lateral slip, so the only physics that matters is:

    v_fwd = -vx*sin(yaw) + vy*cos(yaw)           (forward speed,
                                                   PIWM body+y = forward)
    yaw_rate = (v_fwd / L) * tan(steer)          (kinematic Ackermann)
    vy_w_next = v_fwd_next * cos(yaw_next)
    vx_w_next = -v_fwd_next * sin(yaw_next)      (no lateral body velocity)
    x_w_next = x_w + vx_w * dt
    y_w_next = y_w + vy_w * dt
    yaw_next = yaw + yaw_rate * dt

Learnable:
  - L (wheelbase, scalar, bounded to a sensible range for the working scale)
  - accel_net(state, action) -> longitudinal acceleration
  - wheel_net(state, action) -> wheel and physical-steer rates of change
                                 (also implicitly models servo lag)
  - residual_net -> small residual on non-position dims

Same I/O signature as BicycleDynamicsV4 so it slots into LaneAugmentedV6Kin
without changes elsewhere.
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


class BicycleDynamicsKin(nn.Module):
    """Pure kinematic bicycle + small residual.

    L_BOUNDS: working-scale wheelbase bounds. For donkey at SPATIAL_SCALE=50,
    physical L=0.165 m -> 8.25 working units; we allow 4-15 for some slack.
    """

    def __init__(self, L_lo: float = 4.0, L_hi: float = 15.0):
        super().__init__()
        input_dim = FULL_STATE_DIM + 3

        self.raw_L = nn.Parameter(torch.tensor(0.0))   # bounded
        self._L_lo, self._L_hi = float(L_lo), float(L_hi)

        self.accel_net = MLP(input_dim, 1, h=64)
        self.wheel_net = MLP(input_dim, 5, h=64)
        self.residual_net = MLP(input_dim, FULL_STATE_DIM, h=64)

        self.register_buffer("mean", torch.tensor(PHYSICS_MEAN_REL, dtype=torch.float32))
        self.register_buffer("std",  torch.tensor(PHYSICS_STD_REL,  dtype=torch.float32))

        # Don't apply residual to x, y (positions are integrated exactly).
        mask = torch.ones(FULL_STATE_DIM)
        mask[0] = 0.0; mask[1] = 0.0
        self.register_buffer("res_mask", mask)

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

        c = torch.cos(yaw); sn = torch.sin(yaw)

        # PIWM convention: body+y = forward. Project world velocity.
        v_fwd = -vx * sn + vy * c                                   # (B,)

        # Longitudinal acceleration (learned).
        a_fwd = self.accel_net(inp).squeeze(-1)
        v_fwd_next = v_fwd + a_fwd * DT

        # Kinematic Ackermann yaw rate.  Snap omega to its kinematic value
        # (no rotational inertia at low speed). Yaw INTEGRATION uses the OLD
        # omega so it matches the lane-resample rotation in
        # dynamics_lane_v6_kin (which reads omega from the input state).
        # This keeps car body and lane waypoints in lock-step every step.
        yaw_rate_kin = (v_fwd / (self.L + 1e-3)) * torch.tan(steer)
        omega_next = yaw_rate_kin
        yaw_next = yaw + omega * DT
        c_n = torch.cos(yaw_next); s_n = torch.sin(yaw_next)

        # Reconstruct world velocity from forward speed and new yaw.
        vx_next = -v_fwd_next * s_n      # body+y forward, world component
        vy_next =  v_fwd_next * c_n

        # Position with current world velocity (forward Euler).
        x_next = x + vx * DT
        y_next = y + vy * DT

        # Wheel speeds + physical steer angle evolution.
        # wheel_net learns (dw0..dw3, dsteer)/dt; donkey servo's ~10-frame lag
        # is absorbed here as part of d_steer's response to action[:, 0].
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
        z_phys_norm = self._norm(z_physics)

        residual = self.residual_net(inp) * self.res_mask
        z_next_norm = z_phys_norm + residual
        return z_next_norm, residual
