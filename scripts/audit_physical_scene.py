"""Independent diagnostics; never changes the frozen physical-scene fits."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_physical_scene as run
from baselines.controlled_dynamics import DT, wrap
from baselines.physical_scene import PhysicalScene
b=run.b;ex=run.ex


def training_observations(cfg):
    report={}
    for fold,split in cfg['splits'].items():
        rec=ex.CorrectedRecords(split['train']);lags={}
        for lag in [1,5,11]:
            arrays=[]
            for r in rec.records:
                xy=r['xy'].astype(np.float64);yaw=r['yaw'].astype(np.float64);v=r['z'][:,3].astype(np.float64)
                delta=xy[lag:]-xy[:-lag]
                # Average the logged instantaneous velocity vector over the
                # SAME interval with a trapezoidal rule, not endpoint speed.
                observed=np.stack([v*np.cos(yaw),v*np.sin(yaw)],axis=-1)
                edge=.5*(observed[:-1]+observed[1:]);cs=np.concatenate([np.zeros((1,2)),np.cumsum(edge,axis=0)])
                expected=(cs[lag:]-cs[:-lag])*DT
                arrays.append((delta-expected)/(lag*DT))
            e=np.concatenate(arrays);lags[str(lag)]=dict(interval_seconds=lag*DT,velocity_vector_RMSE=float(np.sqrt(np.mean(np.sum(e*e,axis=1)))),median_vector_difference=float(np.median(np.linalg.norm(e,axis=1))))
        run.attach_placeholder(rec);ds=b.WindowData(rec,100,4);errors=[]
        for z,a,y,k,p in b.loader(ds,128):
            # FUTURE TRUE speed and heading: only an observation/kinematic
            # consistency diagnostic, never a usable forecast or baseline.
            vel=y[:,:,3:4]*torch.stack([y[:,:,2].cos(),y[:,:,2].sin()],-1)
            integrated=torch.cumsum(.5*(vel[:,1:]+vel[:,:-1])*DT,dim=1)
            errors.append(torch.linalg.vector_norm(integrated[:,-1]-y[:,-1,:2],dim=-1).numpy())
        report[fold]=dict(velocity_consistency=lags,future_truth_kinematic_reconstruction_E100=float(np.concatenate(errors).mean()),interpretation='Training only; future true heading/speed diagnostic is NOT a prediction baseline. Differences may reflect noise, timing, speed definition or lateral motion; not proof of sideslip.')
    b.write_json(run.OUT/'observation_consistency.json',report)
    print('TRAIN OBSERVATIONS',report,flush=True)


def numerical(cfg):
    run.verify(cfg);rows={}
    for fold,split in cfg['splits'].items():
        rec=ex.CorrectedRecords(split['validation']);run.attach_placeholder(rec);ds=b.WindowData(rec,100,4)
        indices=np.unique(np.linspace(0,len(ds)-1,24,dtype=int));z,a,y,k,p=next(iter(b.loader(torch.utils.data.Subset(ds,indices),24)))
        z=z.double();a=a.double()
        for seed in cfg['seeds']:
            key=f'{fold}/seed{seed}';m=PhysicalScene().double();m.load_state_dict(torch.load(run.CK/f'{fold}_seed{seed}'/'physical_scene.pt',weights_only=False)['model'])
            def integrate(substeps):
                state=torch.cat([torch.zeros_like(z[:,:3]),z[:,3:]],-1);out=[state];dt=DT/substeps
                for act in a.unbind(1):
                    for _ in range(substeps):
                        mid=state+.5*dt*m.field(state,act);state=state+dt*m.field(mid,act)
                        state=torch.cat([state[:,:2],wrap(state[:,2:3]),state[:,3:]],-1)
                    out.append(state)
                return torch.stack(out,1)
            with torch.no_grad():
                one=m.rollout(z,a);four=integrate(4);eight=integrate(8)
                rows[key]=dict(one_vs_eight_max_position_m=float(torch.linalg.vector_norm(one[:,:,:2]-eight[:,:,:2],dim=-1).max()),four_vs_eight_max_position_m=float(torch.linalg.vector_norm(four[:,:,:2]-eight[:,:,:2],dim=-1).max()))
                assert torch.isfinite(eight).all()
            if fold=='outer1':
                aa=torch.zeros(2,32,2,dtype=torch.float64);aa[:,:,1]=m.throttle_reference;zz=z[:2]
                loss=m.rollout(zz,aa).square().mean();loss.backward()
                rows[key]['constant_throttle_drive_gain_gradient']=float(m.raw.grad[0]);assert float(m.raw.grad[0])==0.
    b.write_json(run.OUT/'numerical_and_identifiability_audit.json',dict(status='PASS',rows=rows,interpretation='Validation-only integration sensitivity, plus exact zero gradient confirming unidentifiable drive gain at constant training throttle. Small discretization error is not a stability guarantee.'))
    print('NUMERICAL',rows,flush=True)


if __name__=='__main__':
    run.configure();cfg=b.read_json(run.OUT/'protocol.json')
    if sys.argv[1]=='observations':training_observations(cfg)
    else:numerical(cfg)
