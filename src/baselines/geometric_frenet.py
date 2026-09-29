"""Geometry-consistent Frenet update using the exact position-readout curve.

q is a general curve parameter, g=|P'(q)|, kappa is derived from P itself.
qdot=v*cos(psi)/(g*(1-d*kappa)); psidot=omega-kappa*g*qdot.
Positive floors are documented numerical extensions outside the valid chart.
"""
import torch
from torch import nn
from baselines.controlled_dynamics import wrap,ego_transform
from baselines.consistent_road_geometry import road_geometry


class GeometricFrenet(nn.Module):
    def __init__(self,core,limits,midpoint=False,road=True):
        super().__init__();self.core=core;self.midpoint=midpoint;self.road=road
        self.register_buffer('increment_limits',torch.as_tensor(limits,dtype=torch.float32))

    def field(self,z,action,g,kap):
        m=self.core;q,d,psi,v,om=z.unbind(-1);psi=wrap(psi)
        sd=v*psi.cos()/(g.clamp(min=.1)*(1-d*kap).clamp(min=.2))
        features=torch.stack([m._norm_d(d),m._norm_psi(psi),m._norm_v(v),m._norm_om(om),(kap-m.kappa_mean)/m.kappa_std],-1)
        features=torch.cat([features,action],-1)
        dv=self.increment_limits[0]*torch.tanh(m.dv_net(features)[:,0]/self.increment_limits[0])
        dom=self.increment_limits[1]*torch.tanh(m.dom_net(features)[:,0]/self.increment_limits[1])
        res=.1*torch.tanh(m.res_net(features))
        return torch.stack([sd,v*psi.sin()+res[:,0]/m.dt,
                            om-kap*g*sd+res[:,1]/m.dt,dv/m.dt,dom/m.dt],-1)

    def forward(self,z,action,kappa_override=None,metric_override=None,points=None):
        f=self.field(z,action,metric_override,kappa_override)
        if self.midpoint:
            mid=z+.5*self.core.dt*f
            _,_,g,k=road_geometry(points,mid[:,0]);f=self.field(mid,action,g,k)
        nxt=z+self.core.dt*f
        return torch.cat([nxt[:,:2],wrap(nxt[:,2:3]),nxt[:,3:]],-1)

    def rollout(self,z0,actions,profile,points,hold_last=False):
        if hold_last:raise ValueError('This experiment uses tangent extension only')
        z=torch.cat([torch.zeros_like(z0[:,:1]),z0[:,1:]],-1)
        if not self.road:
            z=torch.cat([z[:,:1],torch.zeros_like(z[:,1:3]),z[:,3:]],-1)
            q=torch.arange(points.shape[1],device=points.device,dtype=points.dtype)*.5
            points=torch.stack([q,torch.zeros_like(q)],-1)[None].expand(len(z),-1,-1)
        def pose(z,p,theta):
            xy=p+z[:,1:2]*torch.stack([-theta.sin(),theta.cos()],-1)
            return torch.cat([xy,(theta+z[:,2])[:,None]],-1)
        p,theta,g,k=road_geometry(points,z[:,0]);anchor=pose(z,p,theta)
        outputs=[torch.cat([torch.zeros_like(z[:,:3]),z[:,3:]],-1)]
        for action in actions.unbind(1):
            z=self(z,action,kappa_override=k,metric_override=g,points=points)
            p,theta,g,k=road_geometry(points,z[:,0])
            outputs.append(torch.cat([ego_transform(pose(z,p,theta),anchor),z[:,3:]],-1))
        return torch.stack(outputs,1)
