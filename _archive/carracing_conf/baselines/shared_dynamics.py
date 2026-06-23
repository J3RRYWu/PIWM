"""Shared-encoder baseline dynamics (following Zhenjiang PIWM paper).

All baseline dynamics models operate on PIWM's 11-dim physics latent.
They replace only the Dynamics module while Encoder/Decoder are shared.

Variants:
  - DynamicsDVBF: locally-linear stochastic transition (no physics prior)
  - DynamicsGOKU: known kinematics + learned force (VAE-like, single-step)
  - DynamicsVid2Param: theta-parameterized physics (params inferred from state history)
  - (SINDYc is handled separately - it's sparse regression, not NN)
"""

import torch
import torch.nn as nn
from config import FULL_STATE_DIM, DT, PHYSICS_STD_REL, PHYSICS_MEAN_REL


class MLP(nn.Module):
    def __init__(self, i, o, h=64, n_layers=2):
        super().__init__()
        layers = [nn.Linear(i, h), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(h, h), nn.ReLU()]
        layers += [nn.Linear(h, o)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


# =====================================================================
# DVBF Dynamics
# =====================================================================
class DynamicsDVBF(nn.Module):
    """Locally-linear stochastic transition:
        z_{t+1} = A(alpha) z_t + B(alpha) a_t + C(alpha) w_t
    where alpha mixes a set of base matrices based on (z_t, a_t).

    No physics prior, pure data-driven.
    """
    def __init__(self, latent_dim=FULL_STATE_DIM, action_dim=3, w_dim=6, n_mat=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.w_dim = w_dim
        self.n_mat = n_mat

        self.A_k = nn.Parameter(torch.randn(n_mat, latent_dim, latent_dim) * 0.02)
        self.B_k = nn.Parameter(torch.randn(n_mat, latent_dim, action_dim) * 0.02)
        self.C_k = nn.Parameter(torch.randn(n_mat, latent_dim, w_dim) * 0.02)
        with torch.no_grad():
            for k in range(n_mat):
                self.A_k[k] += torch.eye(latent_dim)  # bias toward identity

        self.alpha_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64), nn.ReLU(),
            nn.Linear(64, n_mat), nn.Softmax(dim=-1),
        )
        # w is learned from current state+action (noise-free during rollout)
        self.w_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64), nn.ReLU(),
            nn.Linear(64, w_dim),
        )

    def forward(self, z, a):
        inp = torch.cat([z, a], dim=-1)
        alpha = self.alpha_net(inp)          # (B, n_mat)
        A = torch.einsum("bk,kij->bij", alpha, self.A_k)
        B = torch.einsum("bk,kij->bij", alpha, self.B_k)
        C = torch.einsum("bk,kij->bij", alpha, self.C_k)
        w = self.w_net(inp)                   # (B, w_dim)
        z_next = (torch.einsum("bij,bj->bi", A, z)
                  + torch.einsum("bij,bj->bi", B, a)
                  + torch.einsum("bij,bj->bi", C, w))
        return z_next


# =====================================================================
# GOKU Dynamics
# =====================================================================
class DynamicsGOKU(nn.Module):
    """Known kinematics (position/yaw integration) + learned force MLPs.

    Same spirit as PIWM's dynamics but WITHOUT a large residual network.
    Forces are predicted by a smaller MLP.
    """
    def __init__(self, latent_dim=FULL_STATE_DIM, action_dim=3):
        super().__init__()
        inp = latent_dim + action_dim
        self.force_net = MLP(inp, 3, h=64)   # dvx, dvy, domega
        self.wheel_net = MLP(inp, 5, h=64)   # dw0..3, dsteer

        std = PHYSICS_STD_REL; mean = PHYSICS_MEAN_REL
        self.register_buffer("dt_vx_x", torch.tensor(float(std[3] * DT / std[0])))
        self.register_buffer("dt_vy_y", torch.tensor(float(std[4] * DT / std[1])))
        self.register_buffer("dt_om_yaw", torch.tensor(float(std[5] * DT / std[2])))
        self.register_buffer("m_vx_x", torch.tensor(float(mean[3] * DT / std[0])))
        self.register_buffer("m_vy_y", torch.tensor(float(mean[4] * DT / std[1])))
        self.register_buffer("m_om_yaw", torch.tensor(float(mean[5] * DT / std[2])))

    def forward(self, z, a):
        inp = torch.cat([z, a], dim=-1)
        x, y, yaw = z[:, 0], z[:, 1], z[:, 2]
        vx, vy, omega = z[:, 3], z[:, 4], z[:, 5]
        wheels, steer = z[:, 6:10], z[:, 10]

        x_n = x + vx * self.dt_vx_x + self.m_vx_x
        y_n = y + vy * self.dt_vy_y + self.m_vy_y
        yaw_n = yaw + omega * self.dt_om_yaw + self.m_om_yaw

        df = self.force_net(inp)
        vx_n = vx + df[:, 0]; vy_n = vy + df[:, 1]; om_n = omega + df[:, 2]

        dw = self.wheel_net(inp)
        w_n = wheels + dw[:, :4]; s_n = steer + dw[:, 4]

        return torch.stack([x_n, y_n, yaw_n, vx_n, vy_n, om_n,
                            w_n[:, 0], w_n[:, 1], w_n[:, 2], w_n[:, 3], s_n], dim=-1)


# =====================================================================
# Vid2Param Dynamics
# =====================================================================
class DynamicsVid2Param(nn.Module):
    """Physical dynamics parameterized by theta, where theta is inferred from
    a short OBSERVATION history (encoder outputs, i.e. from images).

    For fair comparison with PIWM/GOKU/DVBF, theta MUST be inferred only from
    what the shared encoder sees (9-dim observable latent: yaw, vx, vy, omega,
    wheels, steer), NOT from ground-truth physics including x, y.
    """
    THETA_DIM = 8
    OBS_DIM = 9  # encoder output dim (shared with PIWM encoder)

    def __init__(self, latent_dim=FULL_STATE_DIM, action_dim=3, theta_dim=THETA_DIM,
                 obs_dim=OBS_DIM):
        super().__init__()
        # Theta inference over encoder-observation history (GRU on 9-dim obs)
        self.theta_rnn = nn.GRU(obs_dim, 64, batch_first=True)
        self.theta_head_mu = nn.Linear(64, theta_dim)
        self.theta_head_lv = nn.Linear(64, theta_dim)

        # Physics with theta
        inp = latent_dim + theta_dim + action_dim
        self.force_net = MLP(inp, 3, h=64)
        self.wheel_net = MLP(inp, 5, h=64)

        std = PHYSICS_STD_REL; mean = PHYSICS_MEAN_REL
        self.register_buffer("dt_vx_x", torch.tensor(float(std[3] * DT / std[0])))
        self.register_buffer("dt_vy_y", torch.tensor(float(std[4] * DT / std[1])))
        self.register_buffer("dt_om_yaw", torch.tensor(float(std[5] * DT / std[2])))
        self.register_buffer("m_vx_x", torch.tensor(float(mean[3] * DT / std[0])))
        self.register_buffer("m_vy_y", torch.tensor(float(mean[4] * DT / std[1])))
        self.register_buffer("m_om_yaw", torch.tensor(float(mean[5] * DT / std[2])))

    def infer_theta(self, state_history):
        """state_history: (B, T_hist, 11) -> theta (B, theta_dim)"""
        out, _ = self.theta_rnn(state_history)
        h = out[:, -1]
        mu = self.theta_head_mu(h); lv = self.theta_head_lv(h)
        std = (0.5 * lv).exp()
        theta = mu + std * torch.randn_like(std)
        return theta, mu, lv

    def step(self, z, a, theta):
        inp = torch.cat([z, theta, a], dim=-1)
        x, y, yaw = z[:, 0], z[:, 1], z[:, 2]
        vx, vy, omega = z[:, 3], z[:, 4], z[:, 5]
        wheels, steer = z[:, 6:10], z[:, 10]

        x_n = x + vx * self.dt_vx_x + self.m_vx_x
        y_n = y + vy * self.dt_vy_y + self.m_vy_y
        yaw_n = yaw + omega * self.dt_om_yaw + self.m_om_yaw

        df = self.force_net(inp)
        vx_n = vx + df[:, 0]; vy_n = vy + df[:, 1]; om_n = omega + df[:, 2]
        dw = self.wheel_net(inp)
        w_n = wheels + dw[:, :4]; s_n = steer + dw[:, 4]

        return torch.stack([x_n, y_n, yaw_n, vx_n, vy_n, om_n,
                            w_n[:, 0], w_n[:, 1], w_n[:, 2], w_n[:, 3], s_n], dim=-1)
