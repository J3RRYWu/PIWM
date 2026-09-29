"""Independent planar dynamic-bicycle generator for a synthetic mechanism pilot.

This is a transparent toy plant, NOT a calibrated DonkeyCar or CommonRoad
implementation. Tire saturation, lateral velocity, steering lag and quadratic
drag are absent from the tested five-state predictor family. No road enters RHS.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
DT=1/22
PLANTS={
    'nominal':dict(mass=3.,inertia=.055,lf=.13,lr=.13,cf=12.,cr=14.,mu=.9,tau=.12),
    'shifted':dict(mass=3.6,inertia=.07,lf=.13,lr=.13,cf=8.,cr=10.,mu=.7,tau=.20)}

def rhs(x,u,p):
    _,_,heading,vx,vy,r,delta=x.T
    speed_floor=np.maximum(vx,.15)
    af=delta-np.arctan2(vy+p['lf']*r,speed_floor)
    ar=-np.arctan2(vy-p['lr']*r,speed_floor)
    lf,lr=p['lf'],p['lr'];mass=p['mass']
    fmaxf=p['mu']*mass*9.81*lr/(lf+lr);fmaxr=p['mu']*mass*9.81*lf/(lf+lr)
    fyf=fmaxf*np.tanh(p['cf']*af/fmaxf);fyr=fmaxr*np.tanh(p['cr']*ar/fmaxr)
    drive=2.*(u[:,1]-.32)-.48*vx-.12*vx*np.abs(vx)
    return np.stack([vx*np.cos(heading)-vy*np.sin(heading),vx*np.sin(heading)+vy*np.cos(heading),r,
        drive-fyf*np.sin(delta)/mass+r*vy,(fyf*np.cos(delta)+fyr)/mass-r*vx,
        (lf*fyf*np.cos(delta)-lr*fyr)/p['inertia'],(.55*u[:,0]-delta)/p['tau']],-1)

def simulate(initial,actions,plant='nominal',substeps=16):
    x=initial.astype(np.float64).copy();out=[x.copy()];h=DT/substeps;p=PLANTS[plant]
    for u in actions.transpose(1,0,2):
        for _ in range(substeps):
            k1=rhs(x,u,p);k2=rhs(x+h*k1/2,u,p);k3=rhs(x+h*k2/2,u,p);k4=rhs(x+h*k3,u,p)
            x=x+h*(k1+2*k2+2*k3+k4)/6
        out.append(x.copy())
    out=np.stack(out,1)
    if not np.isfinite(out).all() or np.min(out[:,:,3])<=.15:raise ValueError('Plant left declared forward-motion domain; do not filter episodes')
    return out

def road_profile(params,amplitude=0.,spacing=.001):
    # theta integrates analytic curvature; the curve is generated without any
    # vehicle targets. Perturbation has zero origin offset and zero origin slope.
    s=np.arange(0,6+spacing/2,spacing);offsets=np.arange(10)*.5
    profiles=[];points=[]
    for k0,k1,phase,freq,b0,b1 in params:
        theta=k0*s+k1/freq*(np.cos(phase)-np.cos(freq*s+phase))
        kap=k0+k1*np.sin(freq*s+phase);dk=k1*freq*np.cos(freq*s+phase)
        t=np.stack([np.cos(theta),np.sin(theta)],-1);normal=np.stack([-np.sin(theta),np.cos(theta)],-1)
        base=cumulative_trapezoid(t,s,axis=0,initial=0)
        L=4.5;w=2*np.pi/L;B=b0+b1*np.sin(w*s)
        e=amplitude*s*s/L**2*B
        ep=amplitude/L**2*(2*s*B+s*s*b1*w*np.cos(w*s))
        epp=amplitude/L**2*(2*B+4*s*b1*w*np.cos(w*s)-s*s*b1*w*w*np.sin(w*s))
        A=1-e*kap;D=ep;Ap=-ep*kap-e*dk
        first=A[:,None]*t+D[:,None]*normal
        second=(Ap-D*kap)[:,None]*t+(A*kap+epp)[:,None]*normal
        g=np.linalg.norm(first,axis=1)
        if g.min()<.2:raise ValueError('Perturbed road degeneracy, do not filter')
        curvature=(first[:,0]*second[:,1]-first[:,1]*second[:,0])/g**3
        arc=cumulative_trapezoid(g,s,initial=0);curve=base+e[:,None]*normal
        if arc[-1]<4.5:raise ValueError('Insufficient road support')
        profiles.append(np.interp(offsets,arc,curvature))
        points.append(np.stack([np.interp(offsets,arc,curve[:,j]) for j in range(2)],-1))
    return np.asarray(profiles,dtype=np.float32),np.asarray(points,dtype=np.float32)

def scenarios(seed,n):
    rng=np.random.default_rng(seed)
    params=np.stack([rng.uniform(-.45,.45,n),rng.uniform(-.25,.25,n),rng.uniform(-np.pi,np.pi,n),
        rng.uniform(.7,1.4,n),rng.choice([-1.,1.],n),rng.uniform(-.5,.5,n)],-1)
    v=rng.uniform(.50,.85,n);d=rng.uniform(-.04,.04,n);psi=rng.uniform(-.08,.08,n)
    phase=rng.uniform(-np.pi,np.pi,n);throttle_phase=rng.uniform(-np.pi,np.pi,n)
    t=np.arange(100)*DT;q=.7*t
    kap=params[:,0,None]+params[:,1,None]*np.sin(params[:,3,None]*q+params[:,2,None])
    steer=np.clip(np.arctan(.26*kap)/.55+.035*np.sin(1.8*t+phase[:,None]),-.25,.25)
    throttle=.52+.06*np.sin(.8*t+throttle_phase[:,None])
    actions=np.stack([steer,throttle],-1)
    shifted=actions.copy();shifted[:,:,1]+=.22
    initial=np.zeros((n,7));initial[:,1]=d;initial[:,2]=psi;initial[:,3]=v
    initial[:,6]=.55*steer[:,0];initial[:,5]=v/.26*np.tan(initial[:,6])
    return params,initial,{'within':actions,'outside':shifted}

def observed(hidden):
    xy=hidden[:,:,:2]-hidden[:,0:1,:2];yaw=hidden[:,0,2]
    truth=np.stack([xy[:,:,0]*np.cos(yaw[:,None])+xy[:,:,1]*np.sin(yaw[:,None]),
        -xy[:,:,0]*np.sin(yaw[:,None])+xy[:,:,1]*np.cos(yaw[:,None]),
        hidden[:,:,2]-yaw[:,None],np.linalg.norm(hidden[:,:,3:5],axis=-1),hidden[:,:,5]],-1)
    z0=np.stack([np.zeros(len(hidden)),hidden[:,0,1],yaw,truth[:,0,3],truth[:,0,4]],-1)
    return z0.astype(np.float32),truth.astype(np.float32)
