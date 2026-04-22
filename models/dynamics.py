import torch
import torch.nn as nn
from config import FULL_STATE_DIM, DT, PHYSICS_STD, PHYSICS_MEAN


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PhysicsDynamics(nn.Module):
    """Physics-informed dynamics model operating on full 11-dim state.

    Position (x, y) is maintained by kinematic integration from velocity.
    Other quantities (yaw, vx, vy, omega, wheels, steer) are predicted by
    learned force/torque MLPs + residual correction.

    All operations are in NORMALIZED space.
    """

    def __init__(self):
        super().__init__()
        input_dim = FULL_STATE_DIM + 3  # z(11) + action(3) = 14

        # Learned force model: predicts changes to [vx, vy, omega] in normalized space
        self.force_net = MLP(input_dim, 3, hidden=64)

        # Learned wheel/steering dynamics: predicts changes to [w0, w1, w2, w3, steer]
        self.wheel_net = MLP(input_dim, 5, hidden=64)

        # Residual correction for all 11 dims
        self.residual_net = MLP(input_dim, FULL_STATE_DIM, hidden=64)

        # Precompute normalized-space kinematic scale factors
        std = PHYSICS_STD
        mean = PHYSICS_MEAN

        # x_norm_{t+1} = x_norm_t + vx_norm_t * (std_vx*dt/std_x) + mean_vx*dt/std_x
        self.register_buffer("dt_vx_x", torch.tensor(float(std[3] * DT / std[0])))
        self.register_buffer("dt_vy_y", torch.tensor(float(std[4] * DT / std[1])))
        self.register_buffer("dt_om_yaw", torch.tensor(float(std[5] * DT / std[2])))

        self.register_buffer("mean_vx_dt_over_std_x", torch.tensor(float(mean[3] * DT / std[0])))
        self.register_buffer("mean_vy_dt_over_std_y", torch.tensor(float(mean[4] * DT / std[1])))
        self.register_buffer("mean_om_dt_over_std_yaw", torch.tensor(float(mean[5] * DT / std[2])))

    def forward(self, z_t, action_t):
        """
        Args:
            z_t: (batch, 11) current normalized full state [x, y, yaw, vx, vy, omega, w0..w3, steer]
            action_t: (batch, 3) action [steering, gas, brake]
        Returns:
            z_next: (batch, 11) predicted next normalized full state
            residual: (batch, 11) residual correction (for regularization)
        """
        inp = torch.cat([z_t, action_t], dim=-1)  # (batch, 14)

        x, y, yaw = z_t[:, 0], z_t[:, 1], z_t[:, 2]
        vx, vy, omega = z_t[:, 3], z_t[:, 4], z_t[:, 5]
        wheels = z_t[:, 6:10]
        steer = z_t[:, 10]

        # === Kinematics: position from velocity (fixed equations) ===
        x_next = x + vx * self.dt_vx_x + self.mean_vx_dt_over_std_x
        y_next = y + vy * self.dt_vy_y + self.mean_vy_dt_over_std_y
        yaw_next = yaw + omega * self.dt_om_yaw + self.mean_om_dt_over_std_yaw

        # === Learned dynamics ===
        forces = self.force_net(inp)  # (batch, 3) -> delta [vx, vy, omega]
        vx_next = vx + forces[:, 0]
        vy_next = vy + forces[:, 1]
        omega_next = omega + forces[:, 2]

        wheel_deltas = self.wheel_net(inp)  # (batch, 5) -> delta [w0..w3, steer]
        wheels_next = wheels + wheel_deltas[:, :4]
        steer_next = steer + wheel_deltas[:, 4]

        # === Assemble physics prediction ===
        z_physics = torch.stack([
            x_next, y_next, yaw_next,
            vx_next, vy_next, omega_next,
            wheels_next[:, 0], wheels_next[:, 1], wheels_next[:, 2], wheels_next[:, 3],
            steer_next,
        ], dim=-1)

        # === Residual correction ===
        residual = self.residual_net(inp)
        z_next = z_physics + residual

        return z_next, residual
