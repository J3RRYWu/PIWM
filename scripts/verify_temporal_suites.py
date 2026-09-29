"""Recompute temporal-suite checkpoints and audit CNN statistics independently."""
from pathlib import Path
import sys,argparse,importlib
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_holdout as b

def perception_audit(ex,cfg):
    train=b.Records(cfg['splits']['temporal']['train']);val=b.Records(cfg['splits']['temporal']['validation'])
    td=b.PerceptionData(train,2);vd=b.PerceptionData(val,2)
    expected_mean=np.stack([td[i][1].numpy() for i in range(len(td))]).mean(0)
    all_targets=[]
    for r in train.records:
        shp=r['shape'].copy();shp[:,:,0]-=b.OFFSETS
        all_targets.append(np.concatenate([r['kappa'],shp.reshape(len(shp),-1)],-1))
    expected_scale=np.concatenate(all_targets).std(0).clip(.05)
    rows={}
    cls=ex.CoupledRoadEncoder if hasattr(ex,'CoupledRoadEncoder') else ex.ConditionedRoadEncoder
    for seed in cfg['seeds']:
        d=ex.CK/f'temporal_seed{seed}';saved=torch.load(d/'perception.pt',weights_only=False)
        assert np.allclose(saved['mean'],expected_mean) and np.allclose(saved['scale'],expected_scale)
        b.DEVICE=b.VISION_DEVICE;m=cls(saved['mean'],saved['scale']).to(b.DEVICE);m.load_state_dict(saved['model']);m.eval()
        total=0;n=0;scale=torch.tensor(expected_scale,device=b.DEVICE)
        with torch.no_grad():
            for x,y in b.loader(vd,64):
                k,p=m.forward_both(x.to(b.DEVICE));err=((torch.cat([k,p.flatten(1)],-1)-y.to(b.DEVICE))/scale).square().mean()
                assert torch.isfinite(err);total+=float(err)*len(x);n+=len(x)
        measured=total/n;assert np.isclose(measured,saved['val_loss'],rtol=1e-5,atol=1e-7)
        hist=b.read_json(d/'perception_history.json');best=min(hist,key=lambda x:x['val_loss']);assert best['epoch']==saved['epoch'] and len(hist)==25
        rows[f'seed{seed}']=dict(val_loss=measured,epoch=saved['epoch'],parameters=sum(p.numel() for p in m.parameters()))
        del m
    b.DEVICE='cpu';b.write_json(ex.OUT/'perception_reaudit.json',dict(status='PASS',checks=['Train-only mean and scale recomputed','All validation CNN losses recomputed from saved weights','25 epochs and validation-minimum checkpoint for every seed'],results=rows))
    print('Perception audit PASS',rows,flush=True)

def rollout_audit(ex,cfg,coupled):
    curves=np.load(ex.OUT/'curves.npz');checks={};original=b.Records(cfg['splits']['temporal']['test'])
    ex.configure(cfg)
    for seed in cfg['seeds']:
        d=ex.CK/f'temporal_seed{seed}'
        if coupled:
            rec=ex.clone_records(original);ind=ex.clone_records(original);ex.encode(rec,d.name);ex.encode(ind,d.name,True)
        else:
            rec=original;ex.c.encode(rec,d)
        b.DEVICE='cpu';stats=dict(np.load(d/'normalizers.npz'));limits=b.read_json(d/'limits.json')['limits']
        for name in ex.NAMES:
            records=ind if coupled and name=='arc_independent_encoder' else rec
            ds=b.WindowData(records,100,4);indices=np.unique(np.linspace(0,len(ds)-1,24,dtype=int));subset=torch.utils.data.Subset(ds,indices)
            m=ex.make(name,stats,d,limits);m.load_state_dict(torch.load(d/(name+'.pt'),weights_only=False)['model'])
            err=b.evaluate_model(m,subset,batch=24);key=f'temporal/seed{seed}/{name}';saved=curves[key][indices]
            delta=float(abs(err-saved).max());assert np.isfinite(err).all() and np.allclose(err,saved,rtol=2e-4,atol=2e-5),(key,delta)
            checks[key]=dict(windows=len(indices),max_absolute_recomputed_difference=delta);print(key,delta,flush=True)
    b.write_json(ex.OUT/'checkpoint_spotcheck.json',dict(status='PASS',scope='24 evenly spaced test windows per saved checkpoint; no refitting or selection',checks=checks,script_sha256=b.digest(__file__)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('suite',choices=['temporal','coupled']);ap.add_argument('--perception-only',action='store_true');args=ap.parse_args()
    ex=importlib.import_module('run_temporal_holdout' if args.suite=='temporal' else 'run_coupled_road');torch.set_num_threads(2)
    cfg=b.read_json(ex.OUT/'protocol.json');ex.verify(cfg);assert cfg['parent_sha256']==b.digest(b.OUT/'protocol.json')
    perception_audit(ex,cfg)
    if not args.perception_only:rollout_audit(ex,cfg,args.suite=='coupled')
if __name__=='__main__':main()
