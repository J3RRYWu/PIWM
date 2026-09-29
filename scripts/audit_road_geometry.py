"""Validation-only audit of the frozen road heads; does not train/select models."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_holdout as b
from baselines.consistent_road_geometry import road_geometry,consistent_pose
from baselines.controlled_dynamics import road_pose,interp_profile
OUT=b.ROOT/'reports/road_geometry'


def checks():
    q=torch.tensor([.2,1.7,4.4,5.,-.1],dtype=torch.float64)
    p=torch.stack([torch.arange(10,dtype=q.dtype)*.5,torch.zeros(10,dtype=q.dtype)],-1)[None].repeat(5,1,1)
    xy,theta,g,k=road_geometry(p,q)
    assert torch.allclose(g,torch.ones_like(g)) and torch.allclose(k,torch.zeros_like(k))
    assert torch.allclose(xy[:,0],q) and torch.allclose(xy[:,1],torch.zeros_like(q))
    # Synthetic radius-2 arc and exact finite-difference derivative of the readout.
    s=torch.arange(10,dtype=q.dtype)*.5;p=torch.stack([2*torch.sin(s/2),2*(1-torch.cos(s/2))],-1)[None].repeat(5,1,1)
    d=torch.full_like(q,.15);psi=torch.full_like(q,.2)
    pose,g,k=consistent_pose(p,q,d,psi)
    assert torch.allclose(pose,road_pose(p,q,d,psi),atol=1e-12)
    eps=1e-6
    plus=consistent_pose(p,q+eps,d,psi)[0];minus=consistent_pose(p,q-eps,d,psi)[0]
    deriv=(plus-minus)/(2*eps);theta=pose[:,2]-psi
    expected=torch.stack([g*(1-d*k)*theta.cos(),g*(1-d*k)*theta.sin(),g*k],-1)
    assert torch.allclose(deriv,expected,atol=1e-6), (deriv,expected)
    # Generalized Frenet vector field must map back to Cartesian vehicle velocity.
    v=torch.full_like(q,.7);omega=torch.full_like(q,.2)
    dq=v*psi.cos()/(g*(1-d*k));dd=v*psi.sin();dpsi=omega-k*g*dq
    vel=expected*dq[:,None]+torch.stack([-theta.sin(),theta.cos(),torch.zeros_like(q)],-1)*dd[:,None]
    vel[:,2]+=dpsi
    target=torch.stack([v*(theta+psi).cos(),v*(theta+psi).sin(),omega],-1)
    assert torch.allclose(vel,target,atol=1e-10)
    return 'PASS: straight curve, extrapolation, readout identity, derivative finite differences, Cartesian velocity consistency'


def summarize(points,profile):
    # Five interior samples per knot interval; avoid second-derivative knot jumps.
    offsets=np.concatenate([j*.5+np.array([.05,.15,.25,.35,.45]) for j in range(9)])
    gs=[];ks=[];ps=[]
    pts=torch.tensor(points);pro=torch.tensor(profile)
    with torch.no_grad():
        for s in offsets:
            q=torch.full((len(pts),),float(s));_,_,g,k=road_geometry(pts,q)
            gs.append(g.numpy());ks.append(k.numpy());ps.append(interp_profile(pro,q).numpy())
    g=np.concatenate(gs);k=np.concatenate(ks);p=np.concatenate(ps);valid=g>1e-6
    err=np.abs(k[valid]-p[valid]);stretch=np.abs(g-1)
    return dict(curve_samples=int(len(g)),degenerate_fraction=float(np.mean(~valid)),
                arc_speed_abs_error_median=float(np.median(stretch)),arc_speed_abs_error_p95=float(np.quantile(stretch,.95)),
                speed_outside_20percent_fraction=float(np.mean(stretch>.2)),
                curvature_head_vs_shape_abs_difference_median=float(np.median(err)),
                curvature_head_vs_shape_abs_difference_p95=float(np.quantile(err,.95)),
                nonfinite_values=int((~np.isfinite(g)).sum()+(~np.isfinite(k)).sum()))


def main():
    torch.set_num_threads(4);print(checks(),flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'validation_audit.json').exists():raise RuntimeError('Audit already exists; refusing overwrite')
    cfg=b.read_json(b.OUT/'protocol.json');result={}
    for fold,split in cfg['splits'].items():
        records=b.Records(split['validation'])
        # Use exactly the validation rollout-start windows, not test data.
        idx=[(i,t) for i,r in enumerate(records.records) for t in range(14,len(r['z'])-100,4)]
        kp=np.stack([records.records[i]['kappa'][t] for i,t in idx]);pts=np.stack([records.records[i]['shape'][t] for i,t in idx])
        result[fold+'/oracle_geometry']=summarize(pts,kp)
        for seed in cfg['seeds']:
            d=b.CK/f'{fold}_seed{seed}';b.DEVICE=b.VISION_DEVICE
            enc=b.RoadContextEncoder(predict_shape=True).to(b.DEVICE)
            enc.load_state_dict(torch.load(d/'perception.pt',map_location=b.DEVICE,weights_only=False)['model'])
            b.encode(enc,records);del enc
            k=np.stack([records.records[i]['pred_kappa'][t] for i,t in idx]);p=np.stack([records.records[i]['pred_shape'][t] for i,t in idx])
            row=summarize(p,k);row.update(windows=len(idx),curvature_RMSE=float(np.sqrt(np.mean((k-kp)**2))),shape_RMSE=float(np.sqrt(np.mean((p-pts)**2))))
            result[f'{fold}/seed{seed}']=row;print(fold,seed,row,flush=True)
    b.write_json(OUT/'validation_audit.json',dict(partition='validation only; previous source splits',checks=checks(),results=result,
        sources={str(p.relative_to(b.ROOT)):b.digest(p) for p in [Path(__file__),b.ROOT/'src/baselines/consistent_road_geometry.py']}))

if __name__=='__main__':main()
