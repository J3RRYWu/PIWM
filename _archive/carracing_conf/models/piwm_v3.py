"""V3 models: 9-dim physics + 8-dim free latent.

Encoder: image -> 17 dims (9 physics supervised + 8 free)
Decoder: 17 dims -> image
Dynamics: 19-dim state (11 physics + 8 free) + 3 action -> 19-dim next state
    Physics part: same kinematics + force MLP + residual
    Free part: MLP predicts next free dims from (full state, action)
"""

import torch
import torch.nn as nn
from config import FRAME_STACK, DT, PHYSICS_STD, PHYSICS_MEAN


ENCODER_DIM_V3 = 9 + 8    # 9 physics + 8 free
FULL_STATE_DIM_V3 = 11 + 8  # 11 physics state (x,y,...) + 8 free


class EncoderV3(nn.Module):
    def __init__(self, out_dim=ENCODER_DIM_V3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.fc(h)


class DecoderV3(nn.Module):
    def __init__(self, in_dim=ENCODER_DIM_V3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 256 * 4 * 4), nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 256, 4, 4)
        return self.deconv(h)


class MLP(nn.Module):
    def __init__(self, i, o, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))
    def forward(self, x):
        return self.net(x)


class DynamicsV3(nn.Module):
    """19-dim state (11 physics + 8 free) + 3 action -> 19-dim next state."""
    FREE_DIM = 8
    PHYS_DIM = 11

    def __init__(self):
        super().__init__()
        inp = self.PHYS_DIM + self.FREE_DIM + 3  # 22
        self.force_net = MLP(inp, 3, h=64)
        self.wheel_net = MLP(inp, 5, h=64)
        self.residual_net = MLP(inp, self.PHYS_DIM, h=64)
        self.free_net = MLP(inp, self.FREE_DIM, h=64)

        std = PHYSICS_STD; mean = PHYSICS_MEAN
        self.register_buffer("dt_vx_x", torch.tensor(float(std[3] * DT / std[0])))
        self.register_buffer("dt_vy_y", torch.tensor(float(std[4] * DT / std[1])))
        self.register_buffer("dt_om_yaw", torch.tensor(float(std[5] * DT / std[2])))
        self.register_buffer("m_vx_x", torch.tensor(float(mean[3] * DT / std[0])))
        self.register_buffer("m_vy_y", torch.tensor(float(mean[4] * DT / std[1])))
        self.register_buffer("m_om_yaw", torch.tensor(float(mean[5] * DT / std[2])))

    def forward(self, z, action):
        """z: (B, 19) = [11 physics, 8 free]  action: (B, 3)"""
        phys = z[:, :self.PHYS_DIM]
        free = z[:, self.PHYS_DIM:]
        inp = torch.cat([z, action], dim=-1)

        x, y, yaw = phys[:, 0], phys[:, 1], phys[:, 2]
        vx, vy, omega = phys[:, 3], phys[:, 4], phys[:, 5]
        wheels = phys[:, 6:10]; steer = phys[:, 10]

        x_next = x + vx * self.dt_vx_x + self.m_vx_x
        y_next = y + vy * self.dt_vy_y + self.m_vy_y
        yaw_next = yaw + omega * self.dt_om_yaw + self.m_om_yaw

        forces = self.force_net(inp)
        vx_next = vx + forces[:, 0]
        vy_next = vy + forces[:, 1]
        omega_next = omega + forces[:, 2]

        wd = self.wheel_net(inp)
        wheels_next = wheels + wd[:, :4]
        steer_next = steer + wd[:, 4]

        z_phys = torch.stack([x_next, y_next, yaw_next,
                               vx_next, vy_next, omega_next,
                               wheels_next[:, 0], wheels_next[:, 1],
                               wheels_next[:, 2], wheels_next[:, 3],
                               steer_next], dim=-1)

        residual = self.residual_net(inp)
        phys_next = z_phys + residual

        free_next = free + self.free_net(inp)

        z_next = torch.cat([phys_next, free_next], dim=-1)
        return z_next, residual
