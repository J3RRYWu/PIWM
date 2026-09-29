"""Synthetic checks for the exploratory guards and road-information isolation."""
from pathlib import Path
import tempfile,sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_holdout as b
from baselines.frenet_diagnostic import FrenetDiagnostic
from baselines.controlled_dynamics import rollout,wrap

def main():
    torch.set_num_threads(4);b.DEVICE='cpu';b.seed_all(0)
    split=b.read_json(b.OUT/'protocol.json')['splits']['outer0'];r=b.Records(split['train'][:1])
    with tempfile.TemporaryDirectory() as td:
        d=Path(td);stats=b.fit_stats(r,d)
        core=b.make_model('frenet',stats,d)
        g=FrenetDiagnostic(core,True,'full',(.05,.1))
        z=torch.tensor([[0.,1.,.2,1.,.5],[0.,1.00001,.2,1.,.5]],requires_grad=True)
        a=torch.zeros(2,2);kap=torch.ones(2)
        o=g(z,a,kappa_override=kap)
        assert torch.isfinite(o).all() and abs(o[0,0]-o[1,0])<1e-5
        o.sum().backward();assert torch.isfinite(z.grad).all()
        # Large learned outputs cannot exceed explicitly documented increments.
        with torch.no_grad():
            for h in [core.dv_net,core.dom_net,core.res_net]:h.net[-1].bias.fill_(1e6)
        o=g(z.detach(),a,kappa_override=kap)
        assert (abs(o[:,3]-z[:,3])<=.050001).all()
        assert (abs(o[:,4]-z[:,4])<=.100001).all()
        assert (abs(o[:,1]-z[:,1]-core.dt*z[:,3]*z[:,2].sin())<=.100001).all()
        zp=z.detach().clone();zp[:,2]+=2*torch.pi
        assert torch.allclose(g(zp,a,kappa_override=kap),o,atol=2e-6)
        points=torch.stack([torch.arange(10)*.5,torch.zeros(10)],-1)[None].repeat(2,1,1)
        profile=torch.randn(2,10);actions=torch.zeros(2,5,2)
        shape=FrenetDiagnostic(core,True,'shape',(.05,.1))
        assert torch.equal(rollout(shape,z.detach(),actions,profile,points),rollout(shape,z.detach(),actions,profile+100,points))
        none=FrenetDiagnostic(core,True,'none',(.05,.1))
        zm=z.detach().clone();zm[:,1:3]+=100
        assert torch.equal(rollout(none,z.detach(),actions,profile,points),rollout(none,zm,actions,profile+100,points+100))
        assert torch.allclose(rollout(none,zm,actions,profile,points)[:,0,:3],torch.zeros(2,3))
    print('PASS: denominator continuity near zero; finite gradients; bounded increments; angle periodicity; shape/no-road input isolation; zero initial pose')

if __name__=='__main__':main()
