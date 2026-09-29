"""Build a versioned corrected-label cache. Never mutate the original records."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_holdout as b
from arc_length_track import ArcLengthTrack
from baselines.unit_arc_geometry import arc_points
OUT=b.ROOT/'reports/consistent_road_labels'

def checks(track):
    s=np.linspace(0,track.length,3000,endpoint=False)+.000137;eps=1e-5
    p,h,k=track.evaluate(s);pp,hp,_=track.evaluate(s+eps);pm,hm,_=track.evaluate(s-eps)
    deriv=(pp-pm)/(2*eps);speed=np.linalg.norm(deriv,axis=-1);dh=np.arctan2(np.sin(hp-hm),np.cos(hp-hm))/(2*eps)
    assert abs(speed-1).max()<1e-4,(abs(speed-1).max())
    assert abs(dh-k).max()<1e-3,(abs(dh-k).max())
    # Independent dense curvature integration versus direct curve coordinates.
    starts=np.linspace(0,track.length,21,endpoint=False);off=np.linspace(0,4.5,4501);p0,h0,_=track.evaluate(starts);pts,_,kk=track.evaluate(starts[:,None]+off)
    theta=np.cumsum((kk[:,1:]+kk[:,:-1])*.5*.001,axis=1);theta=np.c_[np.zeros(len(starts)),theta]
    tang=np.stack([np.cos(theta),np.sin(theta)],-1);integ=np.cumsum((tang[:,1:]+tang[:,:-1])*.5*.001,axis=1);integ=np.concatenate([np.zeros((len(starts),1,2)),integ],axis=1)
    delta=pts-p0[:,None];direct=np.stack([delta[...,0]*np.cos(h0[:,None])+delta[...,1]*np.sin(h0[:,None]),-delta[...,0]*np.sin(h0[:,None])+delta[...,1]*np.cos(h0[:,None])],-1)
    maxerr=float(np.linalg.norm(integ-direct,axis=-1).max());assert maxerr<.001,maxerr
    ang=np.linspace(0,2*np.pi,128,endpoint=False);circle=ArcLengthTrack(np.stack([2*np.cos(ang),2*np.sin(ang)],-1));ss=np.linspace(0,circle.length,200,endpoint=False);xy,hh,kap=circle.evaluate(ss)
    norm=np.stack([-np.sin(hh),np.cos(hh)],-1);proj,d,head,res=circle.project(xy+.1*norm)
    assert abs(circle.length-4*np.pi)<1e-5 and abs(kap-.5).max()<.001 and abs(d-.1).max()<1e-6 and abs(res).max()<1e-6
    return dict(status='PASS',max_unit_speed_error=float(abs(speed-1).max()),max_heading_derivative_curvature_error=float(abs(dh-k).max()),dense_integration_max_position_difference_m=maxerr,circle_length_error=float(abs(circle.length-4*np.pi)))

def main():
    if (OUT/'manifest.json').exists():raise RuntimeError('Labels already frozen')
    (OUT/'labels').mkdir(parents=True,exist_ok=True);meta=np.load(b.DATA/'_meta/track.npz');track=ArcLengthTrack(meta['centers']);result=checks(track);result.update(nominal_legacy_length=float(meta['total_len']),corrected_curve_length=track.length)
    old=b.read_json(b.OUT/'protocol.json');files={};maximum=0.;sufficient=0
    for name in old['files']:
        rec=b.Records([name]).records[0];labels=track.labels(rec['xy'],rec['yaw'],b.OFFSETS)
        z=rec['z'].copy();z[:,0]=labels['s'];z[:,1]=labels['d'];z[:,2]=labels['psi']
        e=float(abs(labels['projection_tangent_residual']).max());maximum=max(maximum,e)
        assert e<1e-4,(name,e);assert np.isfinite(z).all() and np.isfinite(labels['kappa']).all();assert np.max(abs(labels['shape'][:,0]))<1e-6
        target=OUT/'labels'/name;np.savez_compressed(target,z=z,kappa=labels['kappa'],shape=labels['shape'])
        files[name]=dict(sha256=b.digest(target),frames=len(z))
    result['max_projection_tangential_residual_m']=maximum
    cfg=b.read_json(b.ROOT/'reports/temporal_holdout/protocol.json');compat={}
    for part in ['train','validation']:
        kp=[];sh=[]
        for name in cfg['splits']['temporal'][part]:
            f=np.load(OUT/'labels'/name);idx=range(14,len(f['z'])-100,4);kp.extend(f['kappa'][list(idx)]);sh.extend(f['shape'][list(idx)])
        kp=np.stack(kp);sh=np.stack(sh)
        with torch.no_grad():arc=arc_points(torch.tensor(kp)).numpy()
        compat[part]=dict(windows=len(kp),coarse_arc_coordinate_RMSE=float(np.sqrt(np.mean((arc-sh)**2))),coarse_arc_endpoint_median=float(np.median(np.linalg.norm(arc[:,-1]-sh[:,-1],axis=-1))))
    manifest=dict(status='PASS',source_track_sha256=b.digest(b.DATA/'_meta/track.npz'),parent_protocol_sha256=b.digest(b.OUT/'protocol.json'),geometry='Periodic cubic spline through existing smoothed centers; chord-length parameter then dense arc integration at 64 samples per segment; PCHIP inverse arc map; current-xy projection only.',source_sha256={str(p.relative_to(b.ROOT)):b.digest(p) for p in [Path(__file__),b.ROOT/'src/arc_length_track.py']},files=files,checks=result,coarse_arc_compatibility=compat,limitations=['Existing surveyed map and historical orientation retained; not a new map/track.', 'Labels are consistent with a smooth mathematical curve, not exact physical road truth.', 'Ten 0.5-m curvature knots and piecewise-constant integration still approximate a continuously varying curve.'])
    b.write_json(OUT/'manifest.json',manifest);print(result,compat,flush=True)
if __name__=='__main__':main()
