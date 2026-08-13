"""31-dim variants of DVBF / GOKU / Vid2Param using v5's encoder/decoder.

State layout: z[0:11]  = car (PIWM convention, normalized)
              z[11:31] = lane waypoints (10 x 2, normalized)

These baselines have NO physics prior on the lane part — everything is
learned. This isolates the value of v5's analytical lane propagation.
"""
import torch
import torch.nn as nn
from config import FULL_STATE_DIM, DT, PHYSICS_STD_REL, PHYSICS_MEAN_REL
from lane_utils import LANE_DIM


TOTAL_DIM = FULL_STATE_DIM + LANE_DIM   # 31


class MLP(nn.Module):
    def __init__(self, i, o, h=64, n_layers=2):
        super().__init__()
        layers = [nn.Linear(i, h), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(h, h), nn.ReLU()]
        layers += [nn.Linear(h, o)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


# =====================================================================
# DVBF on 31-dim
# =====================================================================
class DynamicsDVBFLane(nn.Module):
    """Locally-linear stochastic transition on 31-dim state."""
    def __init__(self, latent_dim=TOTAL_DIM, action_dim=3, w_dim=8, n_mat=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.A_k = nn.Parameter(torch.randn(n_mat, latent_dim, latent_dim) * 0.02)
        self.B_k = nn.Parameter(torch.randn(n_mat, latent_dim, action_dim) * 0.02)
        self.C_k = nn.Parameter(torch.randn(n_mat, latent_dim, w_dim) * 0.02)
        with torch.no_grad():
            for k in range(n_mat):
                self.A_k[k] += torch.eye(latent_dim)
        self.alpha_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64), nn.ReLU(),
            nn.Linear(64, n_mat), nn.Softmax(dim=-1),
        )
        self.w_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64), nn.ReLU(),
            nn.Linear(64, w_dim),
        )

    def forward(self, z, a):
        inp = torch.cat([z, a], dim=-1)
        alpha = self.alpha_net(inp)
        A = torch.einsum("bk,kij->bij", alpha, self.A_k)
        B = torch.einsum("bk,kij->bij", alpha, self.B_k)
        C = torch.einsum("bk,kij->bij", alpha, self.C_k)
        w = self.w_net(inp)
        z_next = (torch.einsum("bij,bj->bi", A, z)
                  + torch.einsum("bij,bj->bi", B, a)
                  + torch.einsum("bij,bj->bi", C, w))
        return z_next


# =====================================================================
# GOKU on 31-dim
# =====================================================================
class DynamicsGOKULane(nn.Module):
    """Known kinematics on car part + learned forces; PURELY learned for lane part.

    `theta_dim > 0` restores GOKU-net's observation pathway. In the conference
    paper (ICCPS 2026) GokuNet and Vid2Param are both *intrinsic* baselines, i.e.
    both infer their latent quantities from the observation sequence; inferring
    parameters from observations is what the "Known Unknowns" in GOKU-net refers
    to. This harness had dropped that pathway for GokuNet while keeping it for
    Vid2Param, which is the whole reason Vid2Param came out strongest.

    NOTE, and this matters when reading the result: with theta_dim > 0 this class
    becomes architecturally near-identical to DynamicsVid2ParamLane below -- same
    three MLP heads, same known-kinematics integration, same GRU over the same
    29-dim observation history. The two then differ only in initialization and
    training noise. That is a property of *this harness*, not of the two published
    methods, and any comparison between them here should be reported as such.

    theta_dim = 0 keeps the original parameter-free model, so existing
    checkpoints load unchanged.
    """
    OBS_DIM = 9 + LANE_DIM      # 29, the v5/v6 encoder output width

    def __init__(self, latent_dim=TOTAL_DIM, action_dim=3, theta_dim=0,
                 obs_dim=OBS_DIM):
        super().__init__()
        self.theta_dim = theta_dim
        if theta_dim:
            self.theta_rnn = nn.GRU(obs_dim, 64, batch_first=True)
            self.theta_head_mu = nn.Linear(64, theta_dim)
            self.theta_head_lv = nn.Linear(64, theta_dim)
        inp = latent_dim + action_dim + theta_dim
        self.force_net = MLP(inp, 3, h=64)             # dvx, dvy, domega
        self.wheel_net = MLP(inp, 5, h=64)             # dw0..3, dsteer
        self.lane_net  = MLP(inp, LANE_DIM, h=128)     # entire lane delta
        std = PHYSICS_STD_REL; mean = PHYSICS_MEAN_REL
        self.register_buffer("dt_vx_x", torch.tensor(float(std[3] * DT / std[0])))
        self.register_buffer("dt_vy_y", torch.tensor(float(std[4] * DT / std[1])))
        self.register_buffer("dt_om_yaw", torch.tensor(float(std[5] * DT / std[2])))
        self.register_buffer("m_vx_x", torch.tensor(float(mean[3] * DT / std[0])))
        self.register_buffer("m_vy_y", torch.tensor(float(mean[4] * DT / std[1])))
        self.register_buffer("m_om_yaw", torch.tensor(float(mean[5] * DT / std[2])))

    def infer_theta(self, obs_hist):
        """obs_hist: (B, T_hist, obs_dim) -> (theta, mu, logvar). Same mechanism as
        DynamicsVid2ParamLane.infer_theta, deliberately, so that enabling it tests
        the observation pathway rather than a different way of using it."""
        out, _ = self.theta_rnn(obs_hist)
        h = out[:, -1]
        mu = self.theta_head_mu(h); lv = self.theta_head_lv(h)
        theta = mu + (0.5 * lv).exp() * torch.randn_like(mu)
        return theta, mu, lv

    def step(self, z, a, theta):
        """V2P-compatible call signature so the two share a training path."""
        return self.forward(z, a, theta)

    def forward(self, z, a, theta=None):
        inp = torch.cat([z, a] if theta is None else [z, theta, a], dim=-1)
        x, y, yaw = z[:, 0], z[:, 1], z[:, 2]
        vx, vy, omega = z[:, 3], z[:, 4], z[:, 5]
        wheels, steer = z[:, 6:10], z[:, 10]
        lane = z[:, 11:31]
        x_n = x + vx * self.dt_vx_x + self.m_vx_x
        y_n = y + vy * self.dt_vy_y + self.m_vy_y
        yaw_n = yaw + omega * self.dt_om_yaw + self.m_om_yaw
        df = self.force_net(inp)
        vx_n = vx + df[:, 0]; vy_n = vy + df[:, 1]; om_n = omega + df[:, 2]
        dw = self.wheel_net(inp)
        w_n = wheels + dw[:, :4]; s_n = steer + dw[:, 4]
        # Lane: pure learned delta
        lane_n = lane + self.lane_net(inp)
        car_part = torch.stack([x_n, y_n, yaw_n, vx_n, vy_n, om_n,
                                w_n[:, 0], w_n[:, 1], w_n[:, 2], w_n[:, 3], s_n], dim=-1)
        return torch.cat([car_part, lane_n], dim=-1)


# =====================================================================
# Vid2Param on 31-dim (theta inferred from v5-encoder OBS history)
# =====================================================================
class DynamicsVid2ParamLane(nn.Module):
    THETA_DIM = 8
    OBS_DIM = 9 + LANE_DIM  # 29 (v5 encoder output)

    def __init__(self, latent_dim=TOTAL_DIM, action_dim=3, theta_dim=THETA_DIM,
                 obs_dim=OBS_DIM):
        super().__init__()
        self.theta_rnn = nn.GRU(obs_dim, 64, batch_first=True)
        self.theta_head_mu = nn.Linear(64, theta_dim)
        self.theta_head_lv = nn.Linear(64, theta_dim)
        inp = latent_dim + theta_dim + action_dim
        self.force_net = MLP(inp, 3, h=64)
        self.wheel_net = MLP(inp, 5, h=64)
        self.lane_net  = MLP(inp, LANE_DIM, h=128)
        std = PHYSICS_STD_REL; mean = PHYSICS_MEAN_REL
        self.register_buffer("dt_vx_x", torch.tensor(float(std[3] * DT / std[0])))
        self.register_buffer("dt_vy_y", torch.tensor(float(std[4] * DT / std[1])))
        self.register_buffer("dt_om_yaw", torch.tensor(float(std[5] * DT / std[2])))
        self.register_buffer("m_vx_x", torch.tensor(float(mean[3] * DT / std[0])))
        self.register_buffer("m_vy_y", torch.tensor(float(mean[4] * DT / std[1])))
        self.register_buffer("m_om_yaw", torch.tensor(float(mean[5] * DT / std[2])))

    def infer_theta(self, obs_hist):
        """obs_hist: (B, T_hist, 29) -> theta (B, theta_dim)."""
        out, _ = self.theta_rnn(obs_hist)
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
        lane = z[:, 11:31]
        x_n = x + vx * self.dt_vx_x + self.m_vx_x
        y_n = y + vy * self.dt_vy_y + self.m_vy_y
        yaw_n = yaw + omega * self.dt_om_yaw + self.m_om_yaw
        df = self.force_net(inp)
        vx_n = vx + df[:, 0]; vy_n = vy + df[:, 1]; om_n = omega + df[:, 2]
        dw = self.wheel_net(inp)
        w_n = wheels + dw[:, :4]; s_n = steer + dw[:, 4]
        lane_n = lane + self.lane_net(inp)
        car_part = torch.stack([x_n, y_n, yaw_n, vx_n, vy_n, om_n,
                                w_n[:, 0], w_n[:, 1], w_n[:, 2], w_n[:, 3], s_n], dim=-1)
        return torch.cat([car_part, lane_n], dim=-1)
