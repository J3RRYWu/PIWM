"""Matched, map-free controls for the preregistered source-held-out experiment.

Existing FrenetDynamics is reused without changing legacy runs. All readout
operations below are batched and differentiable. No surveyed centerline enters
inference. Initial road-relative state remains privileged and is disclosed.
"""
import torch
from torch import nn

DT = 1 / 22


def wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def interp_profile(profile, distance, spacing=0.5):
    u = (distance / spacing).clamp(0, profile.shape[1] - 1)
    j = u.floor().long().clamp(0, profile.shape[1] - 2)
    w = u - j
    return profile.gather(1, j[:, None])[:, 0] * (1-w) + profile.gather(1, (j+1)[:, None])[:, 0] * w


def road_pose(points, distance, d, psi, hold_last=False):
    """Catmull-Rom road readout; tangent extrapolation outside observed preview.

    hold_last exists only as a named diagnostic of the legacy endpoint clamp.
    No track length, centerline file, or ground-truth future is used.
    """
    n = points.shape[1]
    # The supplied shape is in the initial ROAD frame. Its origin tangent is
    # known to be +x (initial heading error is privileged input). Using the
    # first chord as this tangent rotates a circular road by kappa*spacing/2.
    initial_tangent = torch.zeros_like(points[:, :1])
    initial_tangent[:, :, 0] = 0.5
    tangents = torch.cat([initial_tangent,
                          (points[:, 2:]-points[:, :-2])/2,
                          points[:, -1:]-points[:, -2:-1]], 1)
    q = (distance / 0.5).clamp(0, n-1)
    j = q.floor().long().clamp(0, n-2)
    u = (q-j)[:, None]
    b = torch.arange(len(points), device=points.device)
    p0, p1, m0, m1 = points[b,j], points[b,j+1], tangents[b,j], tangents[b,j+1]
    pos = (2*u**3-3*u**2+1)*p0 + (u**3-2*u**2+u)*m0 + (-2*u**3+3*u**2)*p1 + (u**3-u**2)*m1
    tangent = (6*u**2-6*u)*p0 + (3*u**2-4*u+1)*m0 + (-6*u**2+6*u)*p1 + (3*u**2-2*u)*m1
    theta = torch.atan2(tangent[:,1], tangent[:,0])
    if not hold_last:
        excess = distance - distance.clamp(0, (n-1)*0.5)
        pos = pos + excess[:,None] * torch.stack([theta.cos(), theta.sin()], -1)
    xy = pos + d[:,None] * torch.stack([-theta.sin(), theta.cos()], -1)
    return torch.cat([xy, (theta+psi)[:,None]], -1)


def ego_transform(pose, anchor):
    dx = pose[:,0]-anchor[:,0]; dy = pose[:,1]-anchor[:,1]
    c = anchor[:,2].cos(); s = anchor[:,2].sin()
    return torch.stack([dx*c+dy*s, -dx*s+dy*c, wrap(pose[:,2]-anchor[:,2])], -1)


def frenet_rollout(model, z0, actions, profile, points, hold_last=False):
    z = torch.cat([torch.zeros_like(z0[:,:1]), z0[:,1:]], -1)
    anchor = road_pose(points, z[:,0], z[:,1], z[:,2], hold_last)
    outputs = [torch.cat([torch.zeros_like(z[:,:3]),z[:,3:]], -1)]
    for a in actions.unbind(1):
        kap = interp_profile(profile, z[:,0])
        z = model(z, a, kappa_override=kap)
        pose = road_pose(points, z[:,0], z[:,1], z[:,2], hold_last)
        outputs.append(torch.cat([ego_transform(pose, anchor), z[:,3:]], -1))
    return torch.stack(outputs,1)


class CartesianResidual(nn.Module):
    """Unicycle pose update + learned increments, with the same frozen preview.

    The no-road control has identical capacity but zeros all road information.
    Output/targets are ego-frame (x,y,heading,v,omega), in physical units.
    """
    def __init__(self, state_mean, state_std, context_mean, context_std, road=True):
        super().__init__()
        self.road = road
        self.register_buffer('motion_mean', torch.as_tensor(state_mean[3:]).float())
        self.register_buffer('motion_std', torch.as_tensor(state_std[3:]).float())
        self.register_buffer('context_mean', torch.as_tensor(context_mean).float())
        self.register_buffer('context_std', torch.as_tensor(context_std).float())
        self.net = nn.Sequential(nn.Linear(40,96),nn.ReLU(),nn.Linear(96,96),nn.ReLU(),nn.Linear(96,5))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def rollout(self,z0,actions,profile,points,hold_last=False):
        z = torch.cat([torch.zeros_like(z0[:,:3]),z0[:,3:]],-1)
        ctx = torch.cat([z0[:,1:3],profile,points.flatten(1)],-1)
        ctx = (ctx-self.context_mean)/self.context_std
        if not self.road: ctx = torch.zeros_like(ctx)
        out=[z]
        for a in actions.unbind(1):
            feat=torch.cat([z[:,:2]/2, z[:,2:3].sin(),z[:,2:3].cos(),
                            (z[:,3:]-self.motion_mean)/self.motion_std,a,ctx],-1)
            r=self.net(feat)
            v,w=z[:,3],z[:,4]
            z=torch.stack([z[:,0]+DT*v*z[:,2].cos()+0.1*r[:,0],
                           z[:,1]+DT*v*z[:,2].sin()+0.1*r[:,1],
                           z[:,2]+DT*w+0.1*r[:,2],v+r[:,3],w+r[:,4]],-1)
            out.append(z)
        return torch.stack(out,1)


class CalibratedKinematic(nn.Module):
    """Global action gains, drag, steering gain/bias and yaw-response lag.

    An action-conditioned kinematic control, not labelled as Vid2Param.
    Seven global constants are fitted from training windows only.
    """
    def __init__(self):
        super().__init__()
        self.raw=nn.Parameter(torch.tensor([0.,0.,0.,0.,0.,0.,0.]))

    def physical(self):
        a=self.raw
        return (0.1+3*a[0].sigmoid(), 0.05+3*a[1].sigmoid(),
                0.08+0.3*a[2].sigmoid(),0.1+1.1*a[3].sigmoid(),
                0.1+0.9*a[4].sigmoid(),0.3*a[5].tanh(),0.5*a[6].tanh())

    def rollout(self,z0,actions,profile,points,hold_last=False):
        z=torch.cat([torch.zeros_like(z0[:,:3]),z0[:,3:]],-1);out=[z]
        ka,kd,length,ks,tau,steer_bias,drive_bias=self.physical()
        for a in actions.unbind(1):
            v,w=z[:,3],z[:,4]
            target=v/length*torch.tan(ks*a[:,0]+steer_bias)
            z=torch.stack([z[:,0]+DT*v*z[:,2].cos(),z[:,1]+DT*v*z[:,2].sin(),
                           z[:,2]+DT*w,v+DT*(ka*a[:,1]-kd*v+drive_bias),
                           w+DT*(target-w)/tau],-1);out.append(z)
        return torch.stack(out,1)


def constant_motion(z0,actions):
    t=torch.arange(actions.shape[1]+1,device=z0.device)[None,:]*DT
    v=z0[:,3:4];w=z0[:,4:5];angle=w*t
    # torch.sinc uses pi-normalized arguments; these forms are stable at omega=0.
    x=v*t*torch.sinc(angle/torch.pi)
    y=v*t*(angle/2)*torch.sinc(angle/(2*torch.pi))**2
    return torch.stack([x,y,angle,v.expand_as(t+v),w.expand_as(t+w)],-1)


def rollout(model,z0,actions,profile,points,hold_last=False):
    if hasattr(model,'rollout'):
        return model.rollout(z0,actions,profile,points,hold_last)
    return frenet_rollout(model,z0,actions,profile,points,hold_last)
