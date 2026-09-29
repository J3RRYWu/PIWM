"""Unit-speed, piecewise circular road from one curvature-knot representation.

The curvature within interval j is the mean of adjacent predicted knots.
Positions and tangents use the exact circular-arc integral, continuous across
boundaries; curvature may jump. Tangent extrapolation has zero curvature.
"""
import torch
from torch import nn
from baselines.controlled_dynamics import ego_transform,wrap

def arc_geometry(profile,q):
    kap=(profile[:,:-1]+profile[:,1:])*.5;length=.5
    turn=kap*length;theta0=torch.cat([torch.zeros_like(turn[:,:1]),turn.cumsum(1)[:,:-1]],1)
    middle=theta0+turn/2;scale=length*torch.sinc(turn/(2*torch.pi))
    delta=scale[...,None]*torch.stack([middle.cos(),middle.sin()],-1)
    origins=torch.cat([torch.zeros_like(delta[:,:1]),delta.cumsum(1)],1)
    clipped=q.clamp(0,kap.shape[1]*length);j=(clipped/length).floor().long().clamp(0,kap.shape[1]-1)
    index=torch.arange(len(profile),device=profile.device);local=clipped-j*length;k=kap[index,j];angle=theta0[index,j]+k*local
    half=theta0[index,j]+k*local/2
    distance=local*torch.sinc(k*local/(2*torch.pi))
    pos=origins[index,j]+distance[:,None]*torch.stack([half.cos(),half.sin()],-1)
    extra=q-clipped;pos=pos+extra[:,None]*torch.stack([angle.cos(),angle.sin()],-1)
    k=torch.where((q<0)|(q>kap.shape[1]*length),torch.zeros_like(k),k)
    return pos,angle,k

def arc_points(profile):
    kap=(profile[:,:-1]+profile[:,1:])*.5;turn=kap*.5
    theta0=torch.cat([torch.zeros_like(turn[:,:1]),turn.cumsum(1)[:,:-1]],1)
    scale=.5*torch.sinc(turn/(2*torch.pi));theta=theta0+turn/2
    delta=scale[...,None]*torch.stack([theta.cos(),theta.sin()],-1)
    return torch.cat([torch.zeros_like(delta[:,:1]),delta.cumsum(1)],1)

class ArcFrenet(nn.Module):
    def __init__(self,core,limits,curvature=True):
        super().__init__();self.core=core;self.curvature=curvature
        self.register_buffer('increment_limits',torch.as_tensor(limits,dtype=torch.float32))
    def field(self,z,action,kap):
        m=self.core;q,d,psi,v,om=z.unbind(-1);psi=wrap(psi)
        if not self.curvature:kap=torch.zeros_like(kap)
        sd=v*psi.cos()/(1-d*kap).clamp(min=.2)
        features=torch.stack([m._norm_d(d),m._norm_psi(psi),m._norm_v(v),m._norm_om(om),(kap-m.kappa_mean)/m.kappa_std],-1)
        features=torch.cat([features,action],-1)
        dv=self.increment_limits[0]*torch.tanh(m.dv_net(features)[:,0]/self.increment_limits[0])
        dw=self.increment_limits[1]*torch.tanh(m.dom_net(features)[:,0]/self.increment_limits[1]);res=.1*torch.tanh(m.res_net(features))
        return torch.stack([sd,v*psi.sin()+res[:,0]/m.dt,om-kap*sd+res[:,1]/m.dt,dv/m.dt,dw/m.dt],-1)
    def forward(self,z,action,kappa_override=None,profile=None):
        first=self.field(z,action,kappa_override);mid=z+.5*self.core.dt*first
        _,_,k=arc_geometry(profile,mid[:,0]);rate=self.field(mid,action,k);nxt=z+self.core.dt*rate
        return torch.cat([nxt[:,:2],wrap(nxt[:,2:3]),nxt[:,3:]],-1)
    def rollout(self,z0,actions,profile,points,hold_last=False):
        if hold_last:raise ValueError('Only tangent extrapolation is defined')
        z=torch.cat([torch.zeros_like(z0[:,:1]),z0[:,1:]],-1)
        def pose(z,p,th):return torch.cat([p+z[:,1:2]*torch.stack([-th.sin(),th.cos()],-1),(th+z[:,2])[:,None]],-1)
        p,th,k=arc_geometry(profile,z[:,0]);anchor=pose(z,p,th);out=[torch.cat([torch.zeros_like(z[:,:3]),z[:,3:]],-1)]
        for action in actions.unbind(1):
            used=k if self.curvature else torch.zeros_like(k)
            z=self(z,action,kappa_override=used,profile=profile);p,th,k=arc_geometry(profile,z[:,0])
            out.append(torch.cat([ego_transform(pose(z,p,th),anchor),z[:,3:]],-1))
        return torch.stack(out,1)
