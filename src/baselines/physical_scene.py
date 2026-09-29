"""Interpretable body response plus a rigidly transported local road curve.

All eight coefficients are effective input/output parameters, not recovered
wheelbase, tire stiffness, or actuator calibration. No road forces or pose MLP.
"""
import math
import torch
from torch import nn
from baselines.controlled_dynamics import DT, wrap


def to_vehicle(points, pose):
    """Fixed-frame points -> current vehicle frame; supports B,T,N,2."""
    d = points - pose[..., None, :2]
    c, s = pose[..., 2:3].cos(), pose[..., 2:3].sin()
    return torch.stack([c*d[..., 0]+s*d[..., 1], -s*d[..., 0]+c*d[..., 1]], -1)


def to_fixed(points, pose):
    c, s = pose[..., 2:3].cos(), pose[..., 2:3].sin()
    return torch.stack([c*points[..., 0]-s*points[..., 1], s*points[..., 0]+c*points[..., 1]], -1)+pose[..., None, :2]


class PhysicalScene(nn.Module):
    def __init__(self, throttle_reference=0.6):
        super().__init__()
        # Match the old control's initial local response approximately.
        positive = [1.6, 1.55, .1, 2.82608696, .398, .50]
        raw = [math.log(math.expm1(x)) for x in positive]
        self.raw = nn.Parameter(torch.tensor(raw+[1.6*throttle_reference, 0.]))
        self.register_buffer('throttle_reference', torch.tensor(float(throttle_reference)))

    def coefficients(self):
        ka, drag, drag2, gain, cubic, lag = torch.nn.functional.softplus(self.raw[:6]).unbind()
        return ka, drag, drag2, gain, cubic, lag+.05, self.raw[6], self.raw[7]

    def physical(self):
        names = ['drive_gain', 'linear_drag', 'quadratic_drag', 'curvature_gain',
                 'curvature_cubic', 'yaw_time_constant', 'force_at_reference', 'curvature_bias']
        return {name: float(v.detach()) for name,v in zip(names,self.coefficients())}

    def field(self, z, action):
        _, _, yaw, v, r = z.unbind(-1)
        ka, drag, drag2, gain, cubic, tau, force, bias = self.coefficients()
        steering, throttle = action.unbind(-1)
        curvature_command = gain*steering+cubic*steering**3+bias
        acceleration = ka*(throttle-self.throttle_reference)+force-drag*v-drag2*v*v.abs()
        return torch.stack([v*yaw.cos(), v*yaw.sin(), r, acceleration,
                            (v*curvature_command-r)/tau], -1)

    def rollout(self, z0, actions, profile=None, points=None, hold_last=False):
        z = torch.cat([torch.zeros_like(z0[:, :3]), z0[:, 3:]], -1)
        out = [z]
        for action in actions.unbind(1):
            # Explicit midpoint: one physical field at each internal stage.
            mid = z+.5*DT*self.field(z, action)
            z = z+DT*self.field(mid, action)
            z = torch.cat([z[:, :2], wrap(z[:, 2:3]), z[:, 3:]], -1)
            out.append(z)
        return torch.stack(out, 1)

    def scene_rollout(self, z0, actions, points):
        """Return vehicle states and the SAME finite road preview at every time.

        Input points use the inherited initial road frame; d/psi are privileged.
        No newly visible or occluded future road is invented by this transport.
        """
        anchor = torch.stack([torch.zeros_like(z0[:, 1]), z0[:, 1], z0[:, 2]], -1)
        initial_ego_points = to_vehicle(points, anchor)
        vehicle = self.rollout(z0, actions)
        current_road = to_vehicle(initial_ego_points[:, None], vehicle[..., :3])
        return dict(vehicle=vehicle, road_points=current_road,
                    initial_road_points=initial_ego_points)
