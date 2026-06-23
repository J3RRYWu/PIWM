"""Frenet world-model dynamics for the fixed donkey track.

State z = [s, d, psi_e, v, omega] in REAL units (metres / rad / rad-per-s).
Road geometry kappa(s) is a DIFFERENTIABLE lookup on the fixed track grid — it
is NOT a predicted state, so it cannot drift over a long rollout. This is the
structural fix for GOKU's win (whose 20 free lane waypoints drift and corrupt
the shared dynamics input).

Per step:
  kappa = kappa_lookup(s)                          (differentiable in s)
  den   = 1 - d*kappa            (clamped away from 0)
  s'    = s    + v*cos(psi)/den * dt               analytic
  d'    = d    + v*sin(psi)       * dt + res_d      analytic + tiny residual
  psi'  = psi  + (omega - kappa*v*cos(psi)/den)*dt + res_psi
  v'    = v    + dv_net(feat)                       learned (throttle -> accel)
  omega'= omega+ dom_net(feat)                      learned (steer -> yaw accel)

Only dv/dom and a small (res_d,res_psi) are learned; the pose kinematics is
analytic and was validated on data (lateral relation corr=0.96). Every state
dim is physically interpretable.
"""
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, i, o, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))
    def forward(self, x): return self.net(x)


class FrenetDynamics(nn.Module):
    def __init__(self, track_npz, stats_npz, dt=1.0 / 22.0):
        super().__init__()
        tr = np.load(track_npz)
        self.M = int(len(tr['kappa']))
        self.register_buffer("kappa_grid", torch.tensor(tr['kappa'], dtype=torch.float32))
        self.total_len = float(tr['total_len'])
        self.grid_ds = float(tr['grid_ds'])
        self.dt = float(dt)

        st = np.load(stats_npz)
        # state order [s,d,psi,v,omega]; s uses a fixed loss scale (0.5 m)
        sm = st['state_mean'].astype(np.float32); ss = st['state_std'].astype(np.float32)
        self.register_buffer("state_mean", torch.tensor(sm))
        self.register_buffer("state_std",  torch.tensor(ss))
        self.register_buffer("kappa_mean", torch.tensor(float(st['kappa_mean'])))
        self.register_buffer("kappa_std",  torch.tensor(float(st['kappa_std'])))

        # feature for the learned heads: [d_n, psi_n, v_n, om_n, kappa_n, action(2)]
        feat_dim = 5 + 2
        self.dv_net  = MLP(feat_dim, 1, h=64)
        self.dom_net = MLP(feat_dim, 1, h=64)
        # TIGHT residual on (res_d, res_psi) only, fixed small scale 0.1.
        # A larger / s-including residual was tested and traded away the
        # bounded long-horizon stability that is Frenet's whole advantage
        # (xy@100 0.246 -> 0.329), so we keep the prior strong.
        self.res_net = MLP(feat_dim, 2, h=64)

    def kappa_at(self, s):
        """Differentiable linear interpolation of kappa at arc-length s (B,)."""
        u = torch.remainder(s, self.total_len) / self.grid_ds        # [0, M)
        i0 = torch.floor(u)
        frac = u - i0
        i0 = i0.long() % self.M
        i1 = (i0 + 1) % self.M
        return self.kappa_grid[i0] * (1 - frac) + self.kappa_grid[i1] * frac

    def _norm_d(self, d):    return (d - self.state_mean[1]) / self.state_std[1]
    def _norm_psi(self, p):  return (p - self.state_mean[2]) / self.state_std[2]
    def _norm_v(self, v):    return (v - self.state_mean[3]) / self.state_std[3]
    def _norm_om(self, o):   return (o - self.state_mean[4]) / self.state_std[4]

    def forward(self, z, action):
        s, d, psi, v, om = z[:, 0], z[:, 1], z[:, 2], z[:, 3], z[:, 4]
        kap = self.kappa_at(s)

        den = 1.0 - d * kap
        den = torch.where(den.abs() < 0.2, torch.sign(den + 1e-6) * 0.2, den)
        cospsi = torch.cos(psi); sinpsi = torch.sin(psi)
        sdot = v * cospsi / den
        ddot = v * sinpsi
        psidot = om - kap * v * cospsi / den

        # learned increments for v, omega + small pose residual
        feat = torch.stack([self._norm_d(d), self._norm_psi(psi), self._norm_v(v),
                            self._norm_om(om), (kap - self.kappa_mean) / self.kappa_std],
                           dim=-1)
        feat = torch.cat([feat, action], dim=-1)
        dv = self.dv_net(feat).squeeze(-1)
        dom = self.dom_net(feat).squeeze(-1)
        res = self.res_net(feat) * 0.1               # (B,2): d, psi

        s_n   = s   + sdot * self.dt                 # s: pure analytic (no residual)
        d_n   = d   + ddot * self.dt + res[:, 0]
        psi_n = psi + psidot * self.dt + res[:, 1]
        v_n   = v   + dv
        om_n  = om  + dom

        return torch.stack([s_n, d_n, psi_n, v_n, om_n], dim=-1)

    # --- loss helper: per-dim normalized error (s uses fixed 0.5 m scale) ---
    def state_loss(self, z_pred, z_gt):
        scale = torch.tensor([0.5, self.state_std[1].item(), self.state_std[2].item(),
                              self.state_std[3].item(), self.state_std[4].item()],
                             device=z_pred.device)
        # wrap psi error to (-pi,pi]; wrap s error to (-L/2, L/2]
        e = z_pred - z_gt
        e0 = torch.remainder(e[:, 0] + self.total_len / 2, self.total_len) - self.total_len / 2
        e2 = torch.atan2(torch.sin(e[:, 2]), torch.cos(e[:, 2]))
        e = torch.stack([e0, e[:, 1], e2, e[:, 3], e[:, 4]], dim=-1)
        return ((e / scale) ** 2).mean(dim=0)        # (5,) per-dim
