"""Retrained query ablations and a bounded Cartesian control."""
import torch
from torch import nn
from baselines.controlled_dynamics import DT,wrap,road_pose,ego_transform,interp_profile
class QueryFrenet(nn.Module):
    def __init__(self,core,limits,mode='dynamic'):
        super().__init__();self.core=core;self.mode=mode;self.register_buffer('increment_limits',torch.as_tensor(limits,dtype=torch.float32))
    def forward(self,z,action,kappa_override=None,feature_kappa=None):
        m=self.core;s,d,psi,v,om=z.unbind(-1);psi=wrap(psi);k=kappa_override
        sd=v*psi.cos()/(1-d*k).clamp(min=.2)
        feat=torch.stack([m._norm_d(d),m._norm_psi(psi),m._norm_v(v),m._norm_om(om),(feature_kappa-m.kappa_mean)/m.kappa_std],-1)
        feat=torch.cat([feat,action],-1);dv=self.increment_limits[0]*torch.tanh(m.dv_net(feat)[:,0]/self.increment_limits[0]);dw=self.increment_limits[1]*torch.tanh(m.dom_net(feat)[:,0]/self.increment_limits[1]);r=.1*torch.tanh(m.res_net(feat))
        return torch.stack([s+m.dt*sd,d+m.dt*v*psi.sin()+r[:,0],wrap(psi+m.dt*(om-k*sd)+r[:,1]),v+dv,om+dw],-1)
    def rollout(self,z0,actions,profile,points,hold_last=False):
        if hold_last:raise ValueError('Only tangent extension')
        z=torch.cat([torch.zeros_like(z0[:,:1]),z0[:,1:]],-1);anchor=road_pose(points,z[:,0],z[:,1],z[:,2]);fixed=profile[:,0]
        outputs=[torch.cat([torch.zeros_like(z[:,:3]),z[:,3:]],-1)]
        for action in actions.unbind(1):
            dynamic=interp_profile(profile,z[:,0]);k=fixed if self.mode=='fixed_query' else dynamic;feature=dynamic if self.mode=='dynamic' else fixed
            z=self(z,action,kappa_override=k,feature_kappa=feature);pose=road_pose(points,z[:,0],z[:,1],z[:,2]);outputs.append(torch.cat([ego_transform(pose,anchor),z[:,3:]],-1))
        return torch.stack(outputs,1)
class GuardedCartesian(nn.Module):
    def __init__(self,core,limits):
        super().__init__();self.core=core;self.register_buffer('increment_limits',torch.as_tensor(limits,dtype=torch.float32))
    def rollout(self,z0,actions,profile,points,hold_last=False):
        m=self.core;z=torch.cat([torch.zeros_like(z0[:,:3]),z0[:,3:]],-1)
        ctx=torch.cat([z0[:,1:3],profile,points.flatten(1)],-1);ctx=(ctx-m.context_mean)/m.context_std;out=[z]
        for a in actions.unbind(1):
            feat=torch.cat([z[:,:2]/2,z[:,2:3].sin(),z[:,2:3].cos(),(z[:,3:]-m.motion_mean)/m.motion_std,a,ctx],-1);r=m.net(feat);v,w=z[:,3],z[:,4]
            dv=self.increment_limits[0]*torch.tanh(r[:,3]/self.increment_limits[0]);dw=self.increment_limits[1]*torch.tanh(r[:,4]/self.increment_limits[1]);pose=.1*torch.tanh(r[:,:3])
            z=torch.stack([z[:,0]+DT*v*z[:,2].cos()+pose[:,0],z[:,1]+DT*v*z[:,2].sin()+pose[:,1],wrap(z[:,2]+DT*w+pose[:,2]),v+dv,w+dw],-1);out.append(z)
        return torch.stack(out,1)
