"""Analytic geometry and information-path checks, using no test records."""
from pathlib import Path
import sys,tempfile
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_holdout as b
from baselines.unit_arc_geometry import arc_geometry,arc_points,ArcFrenet
from baselines.controlled_dynamics import constant_motion,rollout

def main():
    torch.set_num_threads(4);b.DEVICE='cpu'
    q=torch.tensor([.23,1.17,3.38,4.9,-.2],dtype=torch.float64,requires_grad=True)
    profile=torch.full((5,10),.5,dtype=torch.float64,requires_grad=True)
    p,theta,k=arc_geometry(profile,q);eps=1e-6
    derivative=(arc_geometry(profile,q+eps)[0]-arc_geometry(profile,q-eps)[0])/(2*eps)
    assert torch.allclose(derivative,torch.stack([theta.cos(),theta.sin()],-1),atol=1e-8)
    assert torch.autograd.gradcheck(lambda a,z:arc_geometry(a,z)[0],(profile,q),eps=1e-6,atol=1e-5)
    for j in range(10):
        query=torch.full((5,),j*.5,dtype=profile.dtype);actual=arc_geometry(profile,query)[0]
        assert torch.allclose(actual,arc_points(profile)[:,j],atol=1e-12)
    zero=torch.zeros(2,10,requires_grad=True);arc_points(zero).sum().backward();assert torch.isfinite(zero.grad).all()
    p,th,k=arc_geometry(torch.zeros(2,10),torch.tensor([-1.,6.]));assert torch.allclose(p,torch.tensor([[-1.,0.],[6.,0.]]))
    cfg=b.read_json(b.OUT/'protocol.json');records=b.Records(cfg['splits']['outer0']['train'][:1])
    with tempfile.TemporaryDirectory() as folder:
        d=Path(folder);stats=b.fit_stats(records,d);b.seed_all(0)
        core=b.make_model('frenet',stats,d);model=ArcFrenet(core,[.1,.1]);z=torch.tensor([[0.,.05,.1,.5,.25]])
        a=torch.zeros(1,100,2);kp=torch.full((1,10),.5);points=arc_points(kp)
        pred=rollout(model,z,a,kp,points);truth=constant_motion(z,a);error=float(torch.linalg.vector_norm(pred[0,-1,:2]-truth[0,-1,:2]).detach());assert error<.001,error
        pred[:,:33].square().mean().backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        core.kappa_grid.fill_(12345);core.total_len=.001
        assert torch.equal(pred,rollout(model,z,a,kp,points+100))
    print('PASS: unit speed, analytic/autograd derivatives, knot continuity, zero-curvature gradients, tangent extension, no-map/no-independent-shape path; midpoint circle error=',error)
if __name__=='__main__':main()
