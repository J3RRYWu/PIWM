"""Curvature feature factorial; geometric query and state normalization unchanged."""
import torch
from baselines.query_controls import QueryFrenet
from baselines.controlled_dynamics import wrap


class BranchQueryFrenet(QueryFrenet):
    def __init__(self, core, limits, response_curvature=True, pose_curvature=True):
        super().__init__(core, limits, mode='dynamic')
        self.response_curvature = response_curvature
        self.pose_curvature = pose_curvature

    def forward(self, z, action, kappa_override=None, feature_kappa=None):
        m = self.core
        s, d, psi, v, om = z.unbind(-1)
        psi = wrap(psi)
        k = kappa_override
        sd = v * psi.cos() / (1 - d * k).clamp(min=.2)
        features = torch.stack([m._norm_d(d), m._norm_psi(psi),
                                m._norm_v(v), m._norm_om(om),
                                (feature_kappa - m.kappa_mean) / m.kappa_std], -1)
        features = torch.cat([features, action], -1)
        # Zero is the normalized train-mean feature, NOT zero geometric curvature.
        masked = torch.cat([features[:, :4], torch.zeros_like(features[:, 4:5]),
                            features[:, 5:]], -1)
        response = features if self.response_curvature else masked
        pose = features if self.pose_curvature else masked
        dv = self.increment_limits[0] * torch.tanh(m.dv_net(response)[:, 0] / self.increment_limits[0])
        dw = self.increment_limits[1] * torch.tanh(m.dom_net(response)[:, 0] / self.increment_limits[1])
        r = .1 * torch.tanh(m.res_net(pose))
        return torch.stack([s + m.dt * sd, d + m.dt * v * psi.sin() + r[:, 0],
                            wrap(psi + m.dt * (om - k * sd) + r[:, 1]),
                            v + dv, om + dw], -1)
