"""Finite, frozen synthetic mechanism pilot: 18 fits; 216 test rows.

No camera simulation, no real-log reuse, and no new method claim. Test scenarios
are generated only after every declared fit. All variants and conditions remain.
"""
from pathlib import Path
import sys,argparse,subprocess,time,copy
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_holdout as b
import diagnose_road_branches as diagnostic
from synthetic_vehicle import scenarios,simulate,observed,road_profile,PLANTS,DT
from models.frenet_dynamics import FrenetDynamics
from baselines.controlled_dynamics import CalibratedKinematic,CartesianResidual
from baselines.query_controls import QueryFrenet,GuardedCartesian
from baselines.branch_query import BranchQueryFrenet
from baselines.geometric_frenet import GeometricFrenet
OUT=b.ROOT/'reports/synthetic_mechanism';CK=b.ROOT/'checkpoints/synthetic_mechanism'
MODELS=['kinematic','cartesian_road','cartesian_blind','guarded_full','curvature_both_off','geometry_midpoint']
NOISES={'clean':0.,'mild':.025,'strong':.10}

class Samples(torch.utils.data.Dataset):
    def __init__(self,data,horizon=100,kappa=None,shape=None):
        self.data=data;self.horizon=horizon;self.kappa=data['kappa'] if kappa is None else kappa;self.shape=data['shape'] if shape is None else shape
    def __len__(self):return len(self.data['z0'])
    def __getitem__(self,i):
        return tuple(torch.as_tensor(a,dtype=torch.float32) for a in [self.data['z0'][i],self.data['actions'][i,:self.horizon],self.data['truth'][i,:self.horizon+1],self.kappa[i],self.shape[i]])

def configure():b.DEVICE='cpu';torch.set_num_threads(4)

def mask_context(module,args):
    x=args[0];return (torch.cat([x[:,:8],torch.zeros_like(x[:,8:])],-1),)

def make(name,directory,stats,limits):
    if name=='kinematic':return CalibratedKinematic()
    if name.startswith('cartesian'):
        core=CartesianResidual(stats['state_mean'],stats['state_std'],stats['context_mean'],stats['context_std'],road=True)
        torch.nn.init.zeros_(core.net[-1].weight);torch.nn.init.zeros_(core.net[-1].bias)
        if name=='cartesian_blind':core.net.register_forward_pre_hook(mask_context)
        return GuardedCartesian(core,limits)
    core=FrenetDynamics(directory/'dummy_track.npz',directory/'stats.npz')
    for net in [core.dv_net,core.dom_net,core.res_net]:
        torch.nn.init.zeros_(net.net[-1].weight);torch.nn.init.zeros_(net.net[-1].bias)
    if name=='geometry_midpoint':return GeometricFrenet(core,limits,midpoint=True)
    if name=='curvature_both_off':return BranchQueryFrenet(core,limits,False,False)
    return QueryFrenet(core,limits,'dynamic')

def dataset(seed,n,plant='nominal',regime='within',noise=.025,substeps=16):
    params,initial,commands=scenarios(seed,n);actions=commands[regime]
    hidden=simulate(initial,actions,plant,substeps);z0,truth=observed(hidden)
    k,p=road_profile(params,noise)
    return dict(z0=z0,actions=actions.astype(np.float32),truth=truth,kappa=k,shape=p,params=params,hidden=hidden)

def stats_from(data,d):
    z=data['z0'];mean=z.mean(0);std=np.maximum(z.std(0),[1.,.05,.1,.05,.1]);mean[0]=0
    # The complete TRAIN trajectory provides motion loss scales only. No validation
    # or test state, action, curvature, or geometry fits any normalizer or guard.
    mean[3:]=data['truth'][:,:,3:].mean(axis=(0,1));std[3:]=np.maximum(data['truth'][:,:,3:].std(axis=(0,1)),[.05,.1])
    context=np.concatenate([z[:,1:3],data['kappa'],data['shape'].reshape(len(z),-1)],-1)
    stats=dict(state_mean=mean.astype(np.float32),state_std=std.astype(np.float32),context_mean=context.mean(0),context_std=context.std(0).clip(.05),
        kappa_mean=float(data['kappa'].mean()),kappa_std=max(float(data['kappa'].std()),.01))
    limits=np.maximum(.01,np.quantile(np.abs(np.diff(data['truth'][:,:,3:],axis=1)).reshape(-1,2),.995,axis=0)).tolist()
    d.mkdir(parents=True,exist_ok=True);np.savez(d/'normalizers.npz',**stats);np.savez(d/'stats.npz',**stats)
    np.savez(d/'dummy_track.npz',kappa=np.zeros(10,np.float32),total_len=5.,grid_ds=.5)
    b.write_json(d/'limits.json',dict(limits=limits,source='training only'))
    return stats,limits

def preflight():
    configure();d=OUT/'preflight';d.mkdir(parents=True,exist_ok=True)
    data=dataset(99711,16);params,initial,actions=scenarios(99711,16)
    fine=simulate(initial,actions['within'],substeps=32);coarse=data['hidden']
    delta=float(np.max(np.linalg.norm(fine[:,:,:2]-coarse[:,:,:2],axis=-1)));assert delta<1e-5
    flipped=initial.copy();flipped[:,[1,2,4,5,6]]*=-1;commands=actions['within'].copy();commands[:,:,0]*=-1
    mirrored=simulate(flipped,commands);expected=coarse.copy();expected[:,:,[1,2,4,5,6]]*=-1
    mirror_error=float(abs(mirrored-expected).max());assert mirror_error<1e-10
    straight=initial.copy();straight[:,[1,2,4,5,6]]=0;commands=actions['within'].copy();commands[:,:,0]=0
    straight_truth=simulate(straight,commands);assert np.max(abs(straight_truth[:,:,[1,2,4,5,6]]))<1e-12
    circle=np.zeros((1,6));circle[0,0]=.5;circle[0,3]=1.;kc,pc=road_profile(circle,0)
    ss=np.arange(10)*.5;exact=np.stack([np.sin(.5*ss)/.5,(1-np.cos(.5*ss))/.5],-1)
    circle_error=float(abs(pc[0]-exact).max());assert circle_error<1e-6 and np.allclose(kc,.5)
    stats,limits=stats_from(data,d);z,a,y,k,p=next(iter(b.loader(Samples(data,32),16)));checks={}
    for name in MODELS:
        b.seed_all(0);m=make(name,d,stats,limits);out=m.rollout(z,a,k,p);loss=b.loss_function(out,y,stats);loss.backward()
        assert torch.isfinite(out).all() and all(x.grad is None or torch.isfinite(x.grad).all() for x in m.parameters())
        if name in ['kinematic','cartesian_blind']:
            with torch.no_grad():assert torch.equal(out,m.rollout(z,a,k+1,p+1))
        checks[name]=dict(parameters=sum(x.numel() for x in m.parameters()),finite_backward=True)
    paths=[Path(__file__),b.ROOT/'src/synthetic_vehicle.py']
    b.write_json(OUT/'preflight.json',dict(status='PASS',rk4_refinement_position_max_m=delta,mirror_max_error=mirror_error,circle_reconstruction_max_m=circle_error,
        straight_line_identity=True,road_blind_invariance=True,checks=checks,sources={str(p.relative_to(b.ROOT)):b.digest(p) for p in paths}))
    print('Preflight PASS',delta,mirror_error,circle_error,flush=True)

def prepare():
    if (OUT/'protocol.json').exists():raise RuntimeError('Already frozen')
    pf=b.read_json(OUT/'preflight.json');assert pf['status']=='PASS'
    for n,h in pf['sources'].items():assert b.digest(b.ROOT/n)==h
    files=['scripts/run_synthetic_mechanism.py','src/synthetic_vehicle.py','scripts/run_controlled_holdout.py','scripts/diagnose_road_branches.py',
        'src/models/frenet_dynamics.py','src/baselines/controlled_dynamics.py','src/baselines/query_controls.py','src/baselines/branch_query.py','src/baselines/geometric_frenet.py','src/baselines/consistent_road_geometry.py']
    cfg=dict(status='Finite exploratory synthetic mechanism pilot, fixed before training and test generation; no real-car or camera validation.',
        train=dict(seed=99711,n=384,plant='nominal',regime='within',noise=.025),validation=dict(seed=99721,n=128,plant='nominal',regime='within',noise=.025),
        test=dict(seed=99731,n=128,plants=list(PLANTS),regimes=['within','outside'],noise_amplitudes=NOISES),
        plants=PLANTS,truth='Independent 7-state dynamic bicycle, saturating tire forces, steering lag, quadratic drag, RK4 16 substeps per 1/22 s. Toy parameters, not calibrated. No road input to plant RHS.',
        initial_state='same complete generator initial state across paired test cells; predictors receive only d/psi/speed/yaw rate, no lateral speed or steering actuator state',
        road='kappa(s)=k0+k1*sin(freq*s+phase), independently generated before trajectories. Smooth normal perturbation with zero initial offset/slope, reparameterized by its arc length. Ten 0.5 m knots; discrete representational error remains.',
        actions='Feedforward steering correlated with road kappa at nominal .7 m/s plus sinusoid, clipped [-.25,.25]. Training/within throttle [.46,.58]; outside adds .22, giving [.68,.80]. Steering and initial state unchanged by regime.',
        models=MODELS,seeds=[0,1,2],epochs=40,train_horizon=32,evaluation_horizon=100,batch=128,lr=.001,kinematic_lr=.01,
        training='Adam, cosine 40, gradient clip 1, same seed initialization/batch order; minimum validation E100; all histories retained, no hyperparameter search',
        normalization='Train only. Initial d/psi scales floor .05/.1; full training trajectories supply v/omega scales with floors .05/.1. 99.5-percentile train increments floored .01 for bounded neural models.',
        primary_model='guarded_full',primary_metric='mean E100 (m) on 128 paired independent synthetic episodes; source generation seeds disjoint',
        secondary='ADE/E25/E50; p95 endpoint; motion errors; full and midpoint chart counts; all 216 model/seed/condition rows',
        decision=dict(candidates=['curvature_both_off','geometry_midpoint'],targets=['within/strong','outside/clean'],
            go='Any one prelisted candidate on any one prelisted target reduces full E100 >=20% in BOTH plants; all three paired model seeds improve; paired-episode bootstrap lower improvement bound >0; candidate clean-within E100 <=110% of full in BOTH plants; candidate target E100 <=110% of the best of kinematic/cartesian_road/cartesian_blind in BOTH plants.',
            mechanism='For the full model, strong-minus-clean road effect at within actions OR outside-minus-within effect at clean road exceeds max(.02 m,20% of clean-within E100), lower paired bootstrap bound >0 and positive in all model seeds, in BOTH plants.',
            stop='If GO false: no automatic new architecture or parameter search; mechanism-only evidence may justify a narrower question, not a claimed successful method or acceptance.',
            disclaimer='Engineering pilot gates, not RAS standards. Bootstrap intervals are conditional exploratory simulation summaries, not real-environment confidence or multiplicity-adjusted confirmatory tests.'),
        bootstrap=dict(draws=2000,seed=99741,percentiles=[2.5,97.5],unit='episode after averaging three model seeds; same sampled episode indices across paired conditions'),
        sources={p:b.digest(b.ROOT/p) for p in files},sources_read=['https://commonroad.in.tum.de/static/media/LA17.8f9d9dc3.pdf'],
        scope='Dynamical interface only; no images, CNN fits, new DonkeyCar logs, closed-loop driving, or real-world robustness claim.')
    CK.mkdir(parents=True,exist_ok=True);b.write_json(OUT/'protocol.json',cfg);print('Frozen 18 fits; 216 test rows; no adaptive search',flush=True)

def verify(cfg):
    for p,h in cfg['sources'].items():assert b.digest(b.ROOT/p)==h,p

def generate(cfg):
    verify(cfg)
    if (OUT/'development_manifest.json').exists():raise RuntimeError('Development data already generated')
    for split in ['train','validation']:
        setting=cfg[split];data=dataset(**setting);np.savez_compressed(OUT/(split+'.npz'),**data)
    stats_from(dict(np.load(OUT/'train.npz')),OUT)
    files=['train.npz','validation.npz','stats.npz','normalizers.npz','limits.json','dummy_track.npz']
    b.write_json(OUT/'development_manifest.json',dict(protocol_sha256=b.digest(OUT/'protocol.json'),files={n:b.digest(OUT/n) for n in files}))
    print('Generated train/validation only',flush=True)

def verify_dev(cfg):
    verify(cfg);mark=b.read_json(OUT/'development_manifest.json');assert mark['protocol_sha256']==b.digest(OUT/'protocol.json')
    for n,h in mark['files'].items():assert b.digest(OUT/n)==h

def fit(cfg,seed):
    verify_dev(cfg);configure();d=CK/f'seed{seed}';d.mkdir(parents=True,exist_ok=True)
    if (d/'COMPLETE.json').exists():return
    if (OUT/'results.json').exists():raise RuntimeError('Test evaluated')
    stats=dict(np.load(OUT/'normalizers.npz'));limits=b.read_json(OUT/'limits.json')['limits']
    td=Samples(dict(np.load(OUT/'train.npz')),32);vd=Samples(dict(np.load(OUT/'validation.npz')),100)
    for name in MODELS:
        if (d/(name+'_history.json')).exists():assert len(b.read_json(d/(name+'_history.json')))==40;continue
        b.seed_all(seed);m=make(name,OUT,stats,limits);lr=cfg['kinematic_lr'] if name=='kinematic' else cfg['lr']
        opt=torch.optim.Adam(m.parameters(),lr=lr);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,40);hist=[];best=float('inf')
        for epoch in range(40):
            m.train();total=0;count=0;gnmax=0
            for z,a,y,k,p in b.loader(td,128,True):
                pred=m.rollout(z,a,k,p);loss=b.loss_function(pred,y,stats);assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1));assert np.isfinite(gn);opt.step()
                total+=float(loss.detach())*len(z);count+=len(z);gnmax=max(gnmax,gn)
            err=b.evaluate_model(m,vd);score=float(err[:,-1].mean());assert np.isfinite(score)
            row=dict(epoch=epoch+1,train_loss=total/count,val_E100=score,max_preclip_gradient=gnmax);hist.append(row)
            saved=dict(model=m.state_dict(),epoch=epoch+1,val_E100=score,parameters=sum(p.numel() for p in m.parameters()),variant=name,seed=seed,protocol_sha256=b.digest(OUT/'protocol.json'))
            if score<best:best=score;torch.save(saved,d/(name+'.pt'))
            if epoch==39:torch.save(saved,d/(name+'_last.pt'))
            sch.step();b.write_json(d/(name+'_progress.json'),hist);print('seed',seed,name,epoch+1,'val',round(score,5),flush=True)
        b.write_json(d/(name+'_history.json'),hist)
    files=[n+suffix for n in MODELS for suffix in ['.pt','_last.pt','_history.json']]
    b.write_json(d/'COMPLETE.json',dict(protocol_sha256=b.digest(OUT/'protocol.json'),files={n:b.digest(d/n) for n in files}))

def completed(cfg):
    for seed in cfg['seeds']:
        d=CK/f'seed{seed}';mark=b.read_json(d/'COMPLETE.json');assert mark['protocol_sha256']==b.digest(OUT/'protocol.json')
        for n,h in mark['files'].items():assert b.digest(d/n)==h

def evaluate(cfg):
    verify_dev(cfg);completed(cfg);configure();torch.set_num_threads(2)
    if (OUT/'results.json').exists():raise RuntimeError('Refusing test overwrite')
    params,initial,actions=scenarios(cfg['test']['seed'],cfg['test']['n']);roads={n:road_profile(params,a) for n,a in NOISES.items()}
    np.savez_compressed(OUT/'test_roads.npz',params=params,**{name+'_'+kind:value for name,pair in roads.items() for kind,value in zip(['kappa','shape'],pair)})
    tests={}
    for plant in cfg['test']['plants']:
        for regime in cfg['test']['regimes']:
            hidden=simulate(initial,actions[regime],plant);z0,truth=observed(hidden)
            data=dict(z0=z0,truth=truth,actions=actions[regime].astype(np.float32),hidden=hidden)
            tests[plant+'/'+regime]=data;np.savez_compressed(OUT/f'test_{plant}_{regime}.npz',**data)
    b.write_json(OUT/'test_manifest.json',dict(protocol_sha256=b.digest(OUT/'protocol.json'),files={p.name:b.digest(p) for p in OUT.glob('test_*.npz')}))
    stats=dict(np.load(OUT/'normalizers.npz'));limits=b.read_json(OUT/'limits.json')['limits'];arrays={};rows={}
    for seed in cfg['seeds']:
        for name in MODELS:
            m=make(name,OUT,stats,limits);m.load_state_dict(torch.load(CK/f'seed{seed}'/(name+'.pt'),weights_only=False)['model'])
            for condition,data in tests.items():
                for noise,(k,p) in roads.items():
                    values=diagnostic.evaluate(m,Samples(data,kappa=k,shape=p),name=='geometry_midpoint');err=values['position_error'];key=f'seed{seed}/{name}/{condition}/{noise}'
                    row=dict(**b.metrics(err),p95_E100=float(np.quantile(err[:,-1],.95)),speed_E100=float(values['speed_error'][:,-1].mean()),yaw_rate_E100=float(values['yaw_rate_error'][:,-1].mean()),
                        stages={key:float(value.min() if key.endswith('_min') else value.mean()) for key,value in values.items() if '/' in key})
                    rows[key]=row
                    for field,value in values.items():arrays[key+'/'+field]=value
            print('EVALUATED',seed,name,flush=True)
    np.savez_compressed(OUT/'arrays.npz',**arrays);b.write_json(OUT/'results.json',dict(protocol_sha256=b.digest(OUT/'protocol.json'),results=rows))

def jobs(cfg):
    verify_dev(cfg);active=[]
    for seed in cfg['seeds']:
        if (CK/f'seed{seed}'/'COMPLETE.json').exists():continue
        fh=(OUT/f'seed{seed}.log').open('a',encoding='utf-8')
        p=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'fit','--seed',str(seed)],cwd=b.ROOT,stdout=fh,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        active.append((p,fh,seed));print('START',seed,p.pid,flush=True)
    while active:
        for item in active[:]:
            p,fh,seed=item
            if p.poll() is not None:
                fh.close();active.remove(item);print('DONE',seed,p.returncode,flush=True)
                if p.returncode:
                    for q,h,_ in active:q.terminate();q.wait();h.close()
                    raise RuntimeError('Fit failed, inspect retained logs')
        time.sleep(2)
    evaluate(cfg)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','prepare','generate','fit','jobs','evaluate']);p.add_argument('--seed',type=int);args=p.parse_args()
    if args.stage in ['preflight','prepare']:globals()[args.stage]()
    else:
        cfg=b.read_json(OUT/'protocol.json')
        if args.stage=='fit':fit(cfg,args.seed)
        else:globals()[args.stage](cfg)
