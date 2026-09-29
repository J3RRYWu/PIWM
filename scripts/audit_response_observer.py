"""Independent audit/diagnosis of the frozen history response experiment."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_response_observer as run
from baselines.controlled_dynamics import DT,wrap
from baselines.physical_scene import to_fixed
b=run.b;ex=run.ex


def audit():
    run.configure();cfg=b.read_json(run.OUT/'protocol.json');run.verify(cfg);run.completed(cfg)
    old=b.read_json(ex.OUT/'protocol.json');parent=b.read_json(b.OUT/'protocol.json');labels=b.read_json(ex.LABELS/'manifest.json')
    for name,h in old['raw_fingerprints'].items():assert b.digest(b.RAW/name)==h
    for name,v in parent['files'].items():assert b.digest(b.DATA/name)==v['sha256']
    for name,v in labels['files'].items():assert b.digest(ex.LABELS/'labels'/name)==v['sha256']
    rows={};latent_bound={};numbers={}
    for fold,split in cfg['splits'].items():
        rec=run.records(split['validation']);ds=run.HistoryData(rec,100,4)
        idx=np.unique(np.linspace(0,len(ds)-1,24,dtype=int));batch=next(iter(b.loader(torch.utils.data.Subset(ds,idx),24)))
        z,a,y,h,p=[v.double() for v in batch]
        for seed in cfg['seeds']:
            for name in run.NAMES:
                key=f'{fold}/seed{seed}/{name}';m,_=run.make(name,fold,seed);m=m.double();saved=torch.load(run.CK/f'{fold}_seed{seed}'/(name+'.pt'),weights_only=False);m.load_state_dict(saved['model']);m.eval()
                with torch.no_grad():
                    response=m.estimate_response(h,p);pred=m.forecast(z,a,response);gain,tau=m.response_parameters();rho=torch.exp(-DT/tau)
                    def integrate(substeps):
                        state=torch.cat([torch.zeros_like(z[:,:3]),z[:,3:]],-1);latent=response.clone();out=[state];dt=DT/substeps
                        decay=torch.exp(-dt/tau);half=torch.exp(-.5*dt/tau)
                        for action in a.unbind(1):
                            for _ in range(substeps):
                                extra=torch.cat([torch.zeros_like(state[:,:3]),latent],-1)
                                mid=state+.5*dt*(m.field(state,action)+extra)
                                nxt=state+dt*(m.field(mid,action)+torch.cat([torch.zeros_like(state[:,:3]),latent*half],-1))
                                state=torch.cat([nxt[:,:2],wrap(nxt[:,2:3]),nxt[:,3:]],-1);latent*=decay
                            out.append(state)
                        return torch.stack(out,1)
                    fine=integrate(8);delta=float(torch.linalg.vector_norm(pred[:,:,:2]-fine[:,:,:2],dim=-1).max())
                    assert torch.isfinite(fine).all()
                    q=rho*(1-gain);assert ((q>0)&(q<1)).all()
                    if name=='observer':
                        innovations=m.innovations(h,p);bound=gain*torch.sqrt(rho)*innovations.abs().amax(1)*(1-q**14)/(1-q)
                        assert torch.all(response.abs()<=bound+1e-9)
                        latent_bound[key]=dict(contraction=q.tolist(),gain=gain.tolist(),decay_seconds=tau.tolist(),response_bound_verified=True)
                    # Arbitrary road test exercises the scene readout, not a
                    # measurement of perception quality or new-road coverage.
                    points=torch.stack([torch.linspace(0,4.5,10,dtype=torch.float64),torch.zeros(10,dtype=torch.float64)],-1)[None].expand(len(z),-1,-1)
                    scene=m.scene_history(z,a,h,p,points)
                    reconstructed=to_fixed(scene['road_points'],scene['vehicle'][...,:3]);transport=float((reconstructed-scene['initial_road_points'][:,None]).abs().max());assert transport<1e-12
                    rows[key]=dict(validation_windows=len(idx),fine_integration_max_position_difference_m=delta,scene_inverse_error_m=transport)
                # Validation motion-state forecast metrics, kept separate from
                # main test E100 and never used for checkpoint selection.
                sq={k:np.zeros(2) for k in [1,5,25,100]};n=0
                mf=m.float()
                with torch.no_grad():
                    for zz,aa,yy,hh,pp in b.loader(ds,128):
                        pred=mf.rollout_history(zz,aa,hh,pp)
                        for step in sq:sq[step]+=((pred[:,step,3:]-yy[:,step,3:])**2).sum(0).numpy()
                        n+=len(zz)
                numbers[key]={str(step):dict(speed_RMSE=float(np.sqrt(value[0]/n)),yaw_rate_RMSE=float(np.sqrt(value[1]/n))) for step,value in sq.items()}
    b.write_json(run.OUT/'independent_audit.json',dict(status='PASS',raw_prepared_corrected_hashes=True,checks=rows,observer_contraction=latent_bound,scope='Fine integration and curve transport checked on validation windows. Observer contraction concerns latent filter only, not full vehicle stability.'))
    b.write_json(run.OUT/'validation_motion_errors.json',numbers)
    print('Independent audit PASS; max integration difference',max(v['fine_integration_max_position_difference_m'] for v in rows.values()),flush=True)


if __name__=='__main__':audit()
