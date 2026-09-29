"""Exploratory guarded Frenet dynamics; preserves the previous experiment.

The guard is a numerical extension outside the positive Frenet chart, not a
physical guarantee. Per-step increments are bounded, state/position errors are
not. Curvature and shape ablations are applied during BOTH fitting and scoring.
"""
import torch
from torch import nn
from baselines.controlled_dynamics import frenet_rollout, wrap


class FrenetDiagnostic(nn.Module):
    def __init__(self, core, guarded=False, road='full', limits=(.1,.1)):
        super().__init__()
        self.core=core;self.guarded=guarded;self.road=road
        self.register_buffer('increment_limits',torch.as_tensor(limits,dtype=torch.float32))

    def forward(self,z,action,kappa_override=None):
        if not self.guarded:
            return self.core(z,action,kappa_override=kappa_override)
        m=self.core;s,d,psi,v,om=z.unbind(-1);psi=wrap(psi);kap=kappa_override
        # Explicit numerical extension: do not reverse longitudinal direction
        # when a prediction has left the valid 1-d*kappa > 0 coordinate chart.
        den=(1-d*kap).clamp(min=.2)
        sd=v*psi.cos()/den
        feat=torch.stack([m._norm_d(d),m._norm_psi(psi),m._norm_v(v),
                          m._norm_om(om),(kap-m.kappa_mean)/m.kappa_std],-1)
        feat=torch.cat([feat,action],-1)
        # Preserve unit slope at zero while bounding each learned increment.
        dv=self.increment_limits[0]*torch.tanh(m.dv_net(feat)[:,0]/self.increment_limits[0])
        dom=self.increment_limits[1]*torch.tanh(m.dom_net(feat)[:,0]/self.increment_limits[1])
        res=.1*torch.tanh(m.res_net(feat))
        return torch.stack([s+m.dt*sd,d+m.dt*v*psi.sin()+res[:,0],
                            wrap(psi+m.dt*(om-kap*sd)+res[:,1]),v+dv,om+dom],-1)

    def rollout(self,z0,actions,profile,points,hold_last=False):
        if self.road!='full':profile=torch.zeros_like(profile)
        if self.road=='none':
            z0=torch.cat([z0[:,:1],torch.zeros_like(z0[:,1:3]),z0[:,3:]],-1)
            offsets=torch.arange(points.shape[1],device=points.device,dtype=points.dtype)*.5
            points=torch.stack([offsets,torch.zeros_like(offsets)],-1)[None].expand(len(z0),-1,-1)
        return frenet_rollout(self,z0,actions,profile,points,hold_last)
