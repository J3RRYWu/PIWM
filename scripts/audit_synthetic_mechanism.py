"""Independent pilot audit, paired contrasts, and predeclared go/no-go decision."""
from pathlib import Path
import argparse,sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_synthetic_mechanism as ex
b=ex.b;OUT=ex.OUT

def audit():
    cfg=b.read_json(OUT/'protocol.json');ex.verify_dev(cfg);ex.completed(cfg);ex.configure();torch.set_num_threads(2)
    tm=b.read_json(OUT/'test_manifest.json');assert tm['protocol_sha256']==b.digest(OUT/'protocol.json')
    for p,h in tm['files'].items():assert b.digest(OUT/p)==h
    tr=dict(np.load(OUT/'train.npz'));va=dict(np.load(OUT/'validation.npz'));roads=np.load(OUT/'test_roads.npz')
    # Distinct random scenario draws; paired test cells intentionally share roads.
    groups=[{tuple(row) for row in params} for params in [tr['params'],va['params'],roads['params']]]
    assert all(not groups[i]&groups[j] for i in range(3) for j in range(i+1,3))
    train_actions=tr['actions'].reshape(-1,2);lo=train_actions.min(0);hi=train_actions.max(0)
    tests={};data_checks={}
    for plant in cfg['test']['plants']:
        for regime in cfg['test']['regimes']:
            key=plant+'/'+regime;data=dict(np.load(OUT/f'test_{plant}_{regime}.npz'));tests[key]=data
            if tests:assert np.array_equal(data['z0'],next(iter(tests.values()))['z0'])
            if plant=='shifted':assert np.array_equal(data['actions'],tests['nominal/'+regime]['actions'])
            outside=np.any((data['actions']<lo-1e-6)|(data['actions']>hi+1e-6),axis=-1)
            if regime=='outside':assert np.all(data['actions'][:,:,1]>hi[1])
            data_checks[key]=dict(episodes=len(data['z0']),action_outside_train_fraction=float(outside.mean()),
                initial_speed_range=[float(data['z0'][:,3].min()),float(data['z0'][:,3].max())],
                speed_range=[float(data['truth'][:,:,3].min()),float(data['truth'][:,:,3].max())],
                lateral_speed_abs_max=float(abs(data['hidden'][:,:,4]).max()),
                endpoint_distance_over_preview_fraction=float((np.linalg.norm(data['truth'][:,-1,:2],axis=-1)>4.5).mean()))
    assert np.array_equal(tests['nominal/within']['actions'][:,:,0],tests['nominal/outside']['actions'][:,:,0])
    assert np.allclose(tests['nominal/outside']['actions'][:,:,1]-tests['nominal/within']['actions'][:,:,1],.22)
    # Geometry distortion is measured, not called a measured sensor noise SD.
    road_checks={}
    for level in ex.NOISES:
        k=roads[level+'_kappa'];p=roads[level+'_shape'];assert np.isfinite(k).all() and np.isfinite(p).all()
        assert np.array_equal(p[:,0],np.zeros_like(p[:,0]))
        road_checks[level]=dict(coordinate_rmse_m=float(np.sqrt(np.mean((p-roads['clean_shape'])**2))),
            curvature_rmse_inverse_m=float(np.sqrt(np.mean((k-roads['clean_kappa'])**2))))
    arrays=np.load(OUT/'arrays.npz');results=b.read_json(OUT/'results.json');assert len(results['results'])==216
    assert results['protocol_sha256']==b.digest(OUT/'protocol.json')
    stats=dict(np.load(OUT/'normalizers.npz'));limits=b.read_json(OUT/'limits.json')['limits'];checks={}
    # Direct train-only statistics recomputation without rewriting the saved files.
    assert np.allclose(stats['context_mean'],np.concatenate([tr['z0'][:,1:3],tr['kappa'],tr['shape'].reshape(len(tr['z0']),-1)],-1).mean(0))
    expected=np.maximum(.01,np.quantile(np.abs(np.diff(tr['truth'][:,:,3:],axis=1)).reshape(-1,2),.995,axis=0));assert np.allclose(expected,limits)
    for seed in cfg['seeds']:
        for name in ex.MODELS:
            saved=torch.load(ex.CK/f'seed{seed}'/(name+'.pt'),weights_only=False);hist=b.read_json(ex.CK/f'seed{seed}'/(name+'_history.json'))
            assert len(hist)==40;best=min(hist,key=lambda r:r['val_E100']);assert saved['epoch']==best['epoch'] and saved['val_E100']==best['val_E100']
            m=ex.make(name,OUT,stats,limits);m.load_state_dict(saved['model']);assert saved['parameters']==sum(p.numel() for p in m.parameters())
            val=b.evaluate_model(m,ex.Samples(va));assert abs(float(val[:,-1].mean())-saved['val_E100'])<2e-5
            for condition,data in tests.items():
                for level in ex.NOISES:
                    key=f'seed{seed}/{name}/{condition}/{level}';err=arrays[key+'/position_error'];row=results['results'][key]
                    assert err.shape==(128,101) and np.isfinite(err).all() and np.allclose(err[:,0],0,atol=1e-7)
                    assert b.metrics(err)=={k:row[k] for k in b.metrics(err)}
                    assert abs(float(np.quantile(err[:,-1],.95))-row['p95_E100'])<1e-10
                    ds=ex.Samples(data,kappa=roads[level+'_kappa'],shape=roads[level+'_shape']);idx=np.linspace(0,len(ds)-1,24,dtype=int)
                    measured=b.evaluate_model(m,torch.utils.data.Subset(ds,idx),batch=24);delta=float(abs(measured-err[idx]).max())
                    assert np.allclose(measured,err[idx],atol=2e-5,rtol=2e-4),(key,delta)
                    if hasattr(m,'core') and hasattr(m.core,'kappa_grid'):
                        m.core.kappa_grid.fill_(12345);m.core.total_len=.001
                        assert np.array_equal(measured,b.evaluate_model(m,torch.utils.data.Subset(ds,idx),batch=24))
                    if name in ['kinematic','cartesian_blind']:
                        assert np.array_equal(err,arrays[f'seed{seed}/{name}/{condition}/clean/position_error'])
                    checks[key]=delta
            print('AUDITED',seed,name,flush=True)
    b.write_json(OUT/'audit.json',dict(status='PASS',rows=216,fits=18,checkpoint_spotcheck_max_differences=checks,
        data_checks=data_checks,road_checks=road_checks,source_sha256=b.digest(__file__),
        scope='Internal independent recomputation, not external replication',checks=['40 epochs and validation minima','all validation scores recomputed','24 test episodes per all 216 rows','all saved metrics and tails recomputed','disjoint generated scenario draws','paired truth/initial state/action invariants','map buffer perturbation','trained road-blind models invariant to road corruption']))

def decision():
    cfg=b.read_json(OUT/'protocol.json');assert b.read_json(OUT/'audit.json')['status']=='PASS'
    arrays=np.load(OUT/'arrays.npz');rng=np.random.default_rng(cfg['bootstrap']['seed']);indices=rng.integers(0,128,(cfg['bootstrap']['draws'],128))
    def errors(name,plant,regime,level):return np.stack([arrays[f'seed{s}/{name}/{plant}/{regime}/{level}/position_error'][:,-1] for s in cfg['seeds']])
    def contrast(delta):
        per_episode=delta.mean(0);boots=per_episode[indices].mean(1)
        return dict(mean_m=float(per_episode.mean()),paired_seed_means=delta.mean(1).tolist(),ci95=np.quantile(boots,[.025,.975]).tolist())
    mechanisms={}
    for factor in ['road','action']:
        by_plant={}
        for plant in cfg['test']['plants']:
            clean=errors('guarded_full',plant,'within','clean')
            altered=errors('guarded_full',plant,'within','strong') if factor=='road' else errors('guarded_full',plant,'outside','clean')
            v=contrast(altered-clean);v['minimum_effect_m']=max(.02,.2*float(clean.mean()))
            v['pass']=v['mean_m']>v['minimum_effect_m'] and v['ci95'][0]>0 and all(x>0 for x in v['paired_seed_means'])
            by_plant[plant]=v
        mechanisms[factor]=dict(pass_both=all(v['pass'] for v in by_plant.values()),plants=by_plant)
    candidates={}
    for name in cfg['decision']['candidates']:
        for target in cfg['decision']['targets']:
            regime,level=target.split('/');by_plant={}
            for plant in cfg['test']['plants']:
                full=errors('guarded_full',plant,regime,level);candidate=errors(name,plant,regime,level)
                v=contrast(full-candidate);full_clean=float(errors('guarded_full',plant,'within','clean').mean());candidate_clean=float(errors(name,plant,'within','clean').mean())
                controls={n:float(errors(n,plant,regime,level).mean()) for n in ['kinematic','cartesian_road','cartesian_blind']}
                v.update(relative_reduction=float(1-candidate.mean()/full.mean()),candidate_target_m=float(candidate.mean()),full_target_m=float(full.mean()),
                    candidate_clean_m=candidate_clean,full_clean_m=full_clean,best_control_m=min(controls.values()),controls=controls)
                conditions=dict(reduction_at_least_20_percent=v['relative_reduction']>=.2,all_seeds_improve=all(x>0 for x in v['paired_seed_means']),
                    paired_interval_positive=v['ci95'][0]>0,clean_penalty_within_10_percent=candidate_clean<=1.1*full_clean,
                    competitive_with_controls=float(candidate.mean())<=1.1*min(controls.values()))
                v['conditions']=conditions;v['pass']=all(conditions.values());by_plant[plant]=v
            candidates[name+'/'+target]=dict(pass_both=all(v['pass'] for v in by_plant.values()),plants=by_plant)
    go=any(v['pass_both'] for v in candidates.values());mechanism=any(v['pass_both'] for v in mechanisms.values())
    label='GO' if go else 'MECHANISM_ONLY' if mechanism else 'NO_GO'
    b.write_json(OUT/'decision.json',dict(decision=label,go=go,mechanism_supported=mechanism,mechanisms=mechanisms,candidates=candidates,
        protocol_sha256=b.digest(OUT/'protocol.json'),scope=cfg['decision']['disclaimer'],script_sha256=b.digest(__file__)))
    print('FROZEN DECISION',label,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','decision','all']);args=p.parse_args()
    if args.stage=='all':audit();decision()
    else:globals()[args.stage]()
