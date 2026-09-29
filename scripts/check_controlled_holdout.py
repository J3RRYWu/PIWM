"""Protocol/geometry checks; no training or test performance is inspected."""
from pathlib import Path
import sys,tempfile
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_holdout as exp
from baselines.controlled_dynamics import road_pose,rollout,constant_motion,ego_transform


def run():
    names=sorted(p.name for p in exp.DATA.glob('*.npz'))
    sources=sorted({exp.PAT.fullmatch(n)[1] for n in names})
    for s in sources:
        split=exp.source_split(names,s)
        sets=[set(split[k]) for k in ('train','validation','test','purged')]
        assert set.union(*sets)==set(names)
        assert all(not sets[i]&sets[j] for i in range(4) for j in range(i))
        assert all(exp.PAT.fullmatch(n)[1]==s for n in split['test'])
        end=max(int(exp.PAT.fullmatch(n)[3]) for n in split['train'])
        start=min(int(exp.PAT.fullmatch(n)[2]) for n in split['validation'])
        assert start-end>=115
    # A straight road must reproduce x=s,y=d,heading=psi, including extrapolation.
    points=torch.stack([torch.arange(10)*.5,torch.zeros(10)],-1)[None].repeat(3,1,1)
    distance=torch.tensor([1.,5.,-.2]);d=torch.tensor([.1,.2,.3]);psi=torch.tensor([.2,.3,.4])
    pose=road_pose(points,distance,d,psi)
    assert torch.allclose(pose,torch.stack([distance,d,psi],-1),atol=1e-6)
    held=road_pose(points,distance,d,psi,True)
    assert held[1,0]==4.5
    # Circular-road knot positions must not acquire a spurious initial rotation.
    arc=torch.arange(10)*.5;kappa=.5
    circle=torch.stack([torch.sin(kappa*arc)/kappa,(1-torch.cos(kappa*arc))/kappa],-1)[None]
    anchor=road_pose(circle,torch.tensor([0.]),torch.tensor([0.]),torch.tensor([0.]))
    end=road_pose(circle,torch.tensor([2.]),torch.tensor([0.]),torch.tensor([0.]))
    assert abs(anchor[0,2].item())<1e-6
    assert torch.allclose(ego_transform(end,anchor)[0,:2],circle[0,4],atol=1e-6)
    split=exp.source_split(names,sources[0]);records=exp.Records(split['train'][:1])
    r=records.records[0];t=20
    assert np.allclose(exp.Records.targets(records,0,t,10)[0,:3],0)
    ds=exp.PerceptionData(records,1);x,y=ds[t-14]
    assert np.array_equal(x.numpy(),r['imgs'][t-14:t+1])
    with tempfile.TemporaryDirectory() as td:
        path=Path(td);stats=exp.fit_stats(records,path)
        for rr in records.records:
            rr['pred_kappa']=rr['kappa'].copy();rr['pred_shape']=rr['shape'].copy()
        stats=exp.context_stats(records,stats)
        windows=exp.WindowData(records,32,4)
        items=next(iter(exp.loader(windows,3)))
        z,a,truth,kp,shp=exp.to_device(items)
        for name in exp.MODELS:
            exp.seed_all(0);model=exp.make_model(name,stats,path)
            pred=rollout(model,z,a,kp,shp)
            assert pred.shape==(3,33,5) and torch.isfinite(pred).all(),name
            assert torch.allclose(pred[:,0,:3],torch.zeros_like(pred[:,0,:3]),atol=1e-6)
            loss=exp.loss_function(pred,truth,stats);loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),name
            if name=='frenet':
                before=pred.detach().clone()
                model.kappa_grid.fill_(12345);model.total_len=.001
                after=rollout(model,z,a,kp,shp)
                assert torch.allclose(before,after), 'Map contents must not affect inference'
            if name=='cartesian_no_road':
                assert torch.allclose(pred,rollout(model,z,a,kp+100,shp+100))
        z0=torch.tensor([[0.,0.,0.,1.,0.]],device=exp.DEVICE)
        a0=torch.zeros(1,22,2,device=exp.DEVICE)
        out=constant_motion(z0,a0)
        assert torch.allclose(out[0,-1,:2],torch.tensor([1.,0.],device=exp.DEVICE),atol=1e-6)
    print('PASS: source separation; temporal purge; ego truth; causal image context; straight geometry; extrapolation; gradients; map independence; no-road isolation; constant-motion analytic case.')

if __name__=='__main__':run()
