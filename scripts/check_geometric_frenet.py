"""Tests of consistent geometry, solver accuracy, and information isolation."""
from pathlib import Path
import tempfile,sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_holdout as b
from audit_road_geometry import checks
from baselines.geometric_frenet import GeometricFrenet
from baselines.controlled_dynamics import constant_motion,rollout

def main():
    print(checks());torch.set_num_threads(4);b.DEVICE='cpu';b.seed_all(0)
    names=b.read_json(b.OUT/'protocol.json')['splits']['outer0']['train'][:1];records=b.Records(names)
    with tempfile.TemporaryDirectory() as td:
        d=Path(td);stats=b.fit_stats(records,d)
        q=torch.arange(10)*.5;points=torch.stack([2*(q/2).sin(),2*(1-(q/2).cos())],-1)[None]
        z=torch.tensor([[0.,.05,.1,.5,.25]]);actions=torch.zeros(1,100,2);profile=torch.zeros(1,10)
        errors=[]
        for midpoint in [False,True]:
            core=b.make_model('frenet',stats,d);model=GeometricFrenet(core,[.1,.1],midpoint=midpoint)
            pred=rollout(model,z,actions,profile,points);truth=constant_motion(z,actions)
            error=float(torch.linalg.vector_norm(pred[0,-1,:2]-truth[0,-1,:2]).detach());errors.append(error)
            assert torch.isfinite(pred).all()
            loss=(pred[:,:33]-truth[:,:33]).square().mean();loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            assert torch.equal(pred,rollout(model,z,actions,profile+100,points))
            # Curve-derived model must ignore legacy map contents.
            model.core.kappa_grid.fill_(9999);model.core.total_len=.001
            assert torch.equal(pred,rollout(model,z,actions,profile,points))
        assert errors[1]<errors[0] and errors[1]<.01,errors
        model=GeometricFrenet(b.make_model('frenet',stats,d),[.1,.1],midpoint=True,road=False)
        a=rollout(model,z,actions,profile,points);other=z.clone();other[:,1:3]+=10
        assert torch.equal(a,rollout(model,other,actions,profile+100,points+100))
        assert torch.allclose(a[:,0,:3],torch.zeros(1,3))
    print('PASS: finite training gradients, curve-derived curvature, no-map/no-road isolation, midpoint accuracy; synthetic endpoint errors:',errors)

if __name__=='__main__':main()
