"""One smooth periodic centerline, parametrized numerically by its true arc length.

Uses the existing smoothed map only. No trajectory fitting or test-score tuning.
Projection uses the current xy only; all road labels share this curve.
"""
import numpy as np
from scipy.interpolate import CubicSpline,PchipInterpolator
from scipy.integrate import cumulative_trapezoid
from scipy.spatial import cKDTree

class ArcLengthTrack:
    def __init__(self,centers,density=64):
        centers=np.asarray(centers,dtype=np.float64)
        chord=np.linalg.norm(np.roll(centers,-1,axis=0)-centers,axis=1)
        if np.any(chord<1e-8):raise ValueError('Repeated map points')
        self.t_knots=np.r_[0,np.cumsum(chord)];self.period=self.t_knots[-1]
        self.curve=CubicSpline(self.t_knots,np.vstack([centers,centers[0]]),bc_type='periodic',axis=0)
        self.t_grid=np.linspace(0,self.period,len(centers)*density+1)
        speed=np.linalg.norm(self.curve(self.t_grid,1),axis=1)
        if speed.min()<1e-4:raise ValueError('Degenerate map tangent')
        self.s_grid=cumulative_trapezoid(speed,self.t_grid,initial=0);self.length=float(self.s_grid[-1])
        self.t_from_s=PchipInterpolator(self.s_grid,self.t_grid)
        self.s_from_t=PchipInterpolator(self.t_grid,self.s_grid)
        self.tree=cKDTree(self.curve(self.t_grid[:-1]))
    def evaluate(self,s):
        s=np.mod(np.asarray(s,dtype=np.float64),self.length);t=self.t_from_s(s)
        p=self.curve(t);v=self.curve(t,1);acc=self.curve(t,2);g=np.linalg.norm(v,axis=-1)
        heading=np.arctan2(v[...,1],v[...,0]);k=(v[...,0]*acc[...,1]-v[...,1]*acc[...,0])/g**3
        return p,heading,k
    def project(self,xy):
        xy=np.asarray(xy,dtype=np.float64);_,idx=self.tree.query(xy);t=self.t_grid[idx].copy()
        for _ in range(10):
            p=self.curve(t);v=self.curve(t,1);acc=self.curve(t,2);r=p-xy
            den=np.sum(v*v+r*acc,axis=-1)
            step=np.sum(r*v,axis=-1)/np.maximum(den,1e-8)
            t=np.mod(t-np.clip(step,-.025,.025),self.period)
        s=self.s_from_t(t);p,h,_=self.evaluate(s);normal=np.stack([-np.sin(h),np.cos(h)],-1)
        d=((xy-p)*normal).sum(-1)
        tangential=((xy-p)*np.stack([np.cos(h),np.sin(h)],-1)).sum(-1)
        return s,d,h,tangential
    def labels(self,xy,yaw,offsets):
        s,d,h,res=self.project(xy);p0=self.evaluate(s)[0]
        p,_,k=self.evaluate(s[:,None]+np.asarray(offsets)[None]);dp=p-p0[:,None]
        c=np.cos(h)[:,None];sn=np.sin(h)[:,None]
        shape=np.stack([dp[...,0]*c+dp[...,1]*sn,-dp[...,0]*sn+dp[...,1]*c],-1)
        psi=np.arctan2(np.sin(yaw-h),np.cos(yaw-h))
        return dict(s=s.astype(np.float32),d=d.astype(np.float32),psi=psi.astype(np.float32),kappa=k.astype(np.float32),shape=shape.astype(np.float32),projection_tangent_residual=res.astype(np.float32))
