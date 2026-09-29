"""Cubic road geometry and its exact parameter derivatives.

The curve parameter q has nominal metre units, but predicted knots need not be
arc-length parameterized. Its speed g=|dP/dq| must be retained in a Frenet ODE.
No map is read. Endpoint extrapolation is a straight, unit-speed tangent ray.
"""
import torch


def road_geometry(points,q):
    n=points.shape[1];spacing=.5
    m0=torch.zeros_like(points[:,:1]);m0[:,:,0]=spacing
    m=torch.cat([m0,(points[:,2:]-points[:,:-2])/2,points[:,-1:]-points[:,-2:-1]],1)
    x=(q/spacing).clamp(0,n-1);j=x.floor().long().clamp(0,n-2);u=(x-j)[:,None]
    b=torch.arange(len(points),device=points.device)
    p0,p1,t0,t1=points[b,j],points[b,j+1],m[b,j],m[b,j+1]
    p=(2*u**3-3*u**2+1)*p0+(u**3-2*u**2+u)*t0+(-2*u**3+3*u**2)*p1+(u**3-u**2)*t1
    dp=((6*u**2-6*u)*p0+(3*u**2-4*u+1)*t0+(-6*u**2+6*u)*p1+(3*u**2-2*u)*t1)/spacing
    ddp=((12*u-6)*p0+(6*u-4)*t0+(-12*u+6)*p1+(6*u-2)*t1)/(spacing**2)
    g=torch.linalg.vector_norm(dp,dim=-1)
    theta=torch.atan2(dp[:,1],dp[:,0])
    kappa=(dp[:,0]*ddp[:,1]-dp[:,1]*ddp[:,0])/g.clamp(min=1e-8)**3
    excess=q-q.clamp(0,(n-1)*spacing);outside=excess!=0
    p=p+excess[:,None]*torch.stack([theta.cos(),theta.sin()],-1)
    g=torch.where(outside,torch.ones_like(g),g)
    kappa=torch.where(outside,torch.zeros_like(kappa),kappa)
    return p,theta,g,kappa


def consistent_pose(points,q,d,psi):
    p,theta,g,kappa=road_geometry(points,q)
    p=p+d[:,None]*torch.stack([-theta.sin(),theta.cos()],-1)
    return torch.cat([p,(theta+psi)[:,None]],-1),g,kappa
