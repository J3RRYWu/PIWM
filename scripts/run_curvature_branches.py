"""Frozen curvature-feature factorial with inherited perception and geometry.

Three new variants x three splits x three seeds = 27 fits. Prior complete
full-curvature and baseline runs are immutable references, not new fits.
"""
from pathlib import Path
import sys, argparse, copy, subprocess, time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_corrected_query as ex
from baselines.branch_query import BranchQueryFrenet
b=ex.b; c=ex.c
OUT=b.ROOT/'reports/curvature_branches'
CK=b.ROOT/'checkpoints/curvature_branches'
VARIANTS={'curvature_response_off':(False,True),'curvature_pose_off':(True,False),'curvature_both_off':(False,False)}
REFERENCES=['guarded_full','kinematic','cartesian_guarded','guarded_static_context','geometry_midpoint']

def configure():
    ex.configure(); c.make=make; c.VARIANTS={n:(True,'full',False) for n in VARIANTS}

def make(name,stats,d,limits):
    return BranchQueryFrenet(b.make_model('frenet',stats,d),limits,*VARIANTS[name])

def preflight():
    configure();OUT.mkdir(parents=True,exist_ok=True)
    cfg=b.read_json(ex.OUT/'protocol.json'); split=cfg['splits']['temporal'];rec=ex.CorrectedRecords(split['train'])
    d=ex.CK/'temporal_seed0';enc=d/'independent';ex.encode(rec,enc)
    stats=dict(np.load(enc/'normalizers.npz'));limits=b.read_json(d/'limits.json')['limits']
    ds=b.WindowData(rec,32,4);z,a,y,k,p=next(iter(b.loader(torch.utils.data.Subset(ds,list(range(16))),16)))
    reference=ex.make('guarded_full',stats,enc,limits)
    reference.load_state_dict(torch.load(d/'guarded_full.pt',weights_only=False)['model'])
    full=BranchQueryFrenet(copy.deepcopy(reference.core),limits,True,True)
    left=reference.rollout(z,a,k,p);right=full.rollout(z,a,k,p)
    assert torch.equal(left,right),float((left-right).abs().max())
    b.loss_function(left,y,stats).backward();b.loss_function(right,y,stats).backward()
    for x,v in zip(reference.parameters(),full.parameters()):assert torch.equal(x.grad,v.grad)
    checks={}
    # Nonzero learned weights ensure feature tests would detect a wrong branch mask.
    for name,flags in VARIANTS.items():
        m=BranchQueryFrenet(copy.deepcopy(reference.core),limits,*flags)
        for parameter in m.parameters():parameter.grad=None
        prediction=m.rollout(z,a,k,p);loss=b.loss_function(prediction,y,stats);loss.backward()
        assert torch.isfinite(prediction).all() and all(x.grad is None or torch.isfinite(x.grad).all() for x in m.parameters())
        geom=k[:,0];x=m(z,a[:,0],kappa_override=geom,feature_kappa=geom)
        altered=m(z,a[:,0],kappa_override=geom,feature_kappa=geom+1)
        assert torch.equal(x[:,0],altered[:,0])
        if not flags[0]:assert torch.equal(x[:,3:],altered[:,3:])
        if not flags[1]:assert torch.equal(x[:,1:3],altered[:,1:3])
        if flags[0]:assert not torch.equal(x[:,3:],altered[:,3:])
        if flags[1]:assert not torch.equal(x[:,1:3],altered[:,1:3])
        checks[name]=dict(parameters=sum(v.numel() for v in m.parameters()),finite_backward=True,
                          branch_feature_independence=True,geometric_progress_unchanged=True)
    # Both-off still uses geometric curvature: changing geometry must alter progress.
    m=BranchQueryFrenet(copy.deepcopy(reference.core),limits,False,False)
    zz=z.clone();zz[:,1]=.05;zz[:,2]=.1;zz[:,3]=.5
    aa=m(zz,a[:,0],kappa_override=torch.zeros_like(geom),feature_kappa=geom)
    bb=m(zz,a[:,0],kappa_override=torch.ones_like(geom),feature_kappa=geom)
    assert not torch.equal(aa[:,0],bb[:,0]) and not torch.equal(aa[:,2],bb[:,2])
    assert torch.equal(aa[:,3:],bb[:,3:])
    b.write_json(OUT/'preflight.json',dict(status='PASS',checks=checks,
        full_identity='bitwise 32-step outputs and all parameter gradients on 16 training windows match inherited QueryFrenet',
        model_sha256=b.digest(b.ROOT/'src/baselines/branch_query.py'),runner_sha256=b.digest(__file__)))
    print('Preflight PASS: full outputs and gradients identical; branch masks and live geometry verified',flush=True)

def prepare():
    if (OUT/'protocol.json').exists():raise RuntimeError('Already frozen')
    old=b.read_json(ex.OUT/'protocol.json');ex.verify(old)
    pf=b.read_json(OUT/'preflight.json');assert pf['status']=='PASS' and pf['runner_sha256']==b.digest(__file__) and pf['model_sha256']==b.digest(b.ROOT/'src/baselines/branch_query.py')
    labels=b.read_json(ex.LABELS/'manifest.json')
    for n,v in labels['files'].items():assert b.digest(ex.LABELS/'labels'/n)==v['sha256']
    for n,h in old['raw_fingerprints'].items():assert b.digest(b.RAW/n)==h
    paths=[Path(__file__),b.ROOT/'src/baselines/branch_query.py']
    inputs={}
    for fold in old['splits']:
        for seed in old['seeds']:
            job=f'{fold}_seed{seed}';d=ex.CK/job;mark=b.read_json(d/'COMPLETE.json')
            assert mark['protocol_sha256']==b.digest(ex.OUT/'protocol.json')
            for n,h in mark['files'].items():assert b.digest(d/n)==h
            names=['independent/perception.pt','independent/normalizers.npz','independent/stats.npz','limits.json','COMPLETE.json','evaluation_windows.json']
            names += [n+suffix for n in REFERENCES for suffix in ['.pt','_history.json']]
            inputs[job]={n:b.digest(d/n) for n in names}
    cfg=dict(status='Retrospective exploratory factorial on reused records; not new independent validation or public preregistration.',
        parent_protocol_sha256=b.digest(ex.OUT/'protocol.json'),splits=old['splits'],seeds=[0,1,2],variants=VARIANTS,references=REFERENCES,
        primary_comparison='curvature_both_off minus guarded_full chronological E100; both source-held-out directions mandatory',
        secondary_comparisons='response-off and pose-off versus full; 2x2 interaction; all paired seed and segment results; no test-score family selection',
        epochs=40,train_horizon=32,test_horizon=100,stride=4,batch=128,lr=.001,
        selection='minimum validation E100 per model, same inherited training code, Adam/cosine40/clip1; same seed initialization and shuffled batches',
        mask='normalized curvature set to zero (training mean) only in designated MLP branches; d/psi remain, so this is not fully road-independent dynamics',
        unchanged='perception weights, normalizers, split, kinematic curvature, geometry/readout, parameter count, objective and increment limits',
        evaluate_after='all 27 fits completed; no mid-run test evaluations; report all variants',
        inputs=inputs,sources={str(p.relative_to(b.ROOT)):b.digest(p) for p in paths},
        inherited_results={p:b.digest(ex.OUT/p) for p in ['curves.npz','results.json','audit.json']},
        diagnostic_protocol_sha256=b.digest(b.ROOT/'reports/road_branch_diagnosis/protocol.json'))
    OUT.mkdir(parents=True,exist_ok=True);CK.mkdir(parents=True,exist_ok=True);b.write_json(OUT/'protocol.json',cfg)
    print('Frozen 27 dynamics fits, zero new CNNs',flush=True)

def verify(cfg):
    ex.verify(b.read_json(ex.OUT/'protocol.json'))
    assert cfg['parent_protocol_sha256']==b.digest(ex.OUT/'protocol.json')
    for n,h in cfg['sources'].items():assert b.digest(b.ROOT/n)==h,n
    for job,files in cfg['inputs'].items():
        for n,h in files.items():assert b.digest(ex.CK/job/n)==h,(job,n)
    for n,h in cfg['inherited_results'].items():assert b.digest(ex.OUT/n)==h
    assert cfg['diagnostic_protocol_sha256']==b.digest(b.ROOT/'reports/road_branch_diagnosis/protocol.json')

def fit(cfg,fold,seed):
    verify(cfg);configure();d=CK/f'{fold}_seed{seed}';d.mkdir(parents=True,exist_ok=True)
    if (d/'COMPLETE.json').exists():return
    if (OUT/'results.json').exists():raise RuntimeError('Test already evaluated')
    split=cfg['splits'][fold];tr=ex.CorrectedRecords(split['train']);va=ex.CorrectedRecords(split['validation'])
    old=ex.CK/d.name;enc=old/'independent';ex.encode(tr,enc);ex.encode(va,enc)
    stats=dict(np.load(enc/'normalizers.npz'));limits=b.read_json(old/'limits.json')['limits']
    for name in VARIANTS:
        if (d/(name+'_history.json')).exists():assert len(b.read_json(d/(name+'_history.json')))==40;continue
        c.fit(name,tr,va,stats,enc,d,cfg,limits,seed)
    files=[name+suffix for name in VARIANTS for suffix in ['.pt','_last.pt','_history.json']]
    b.write_json(d/'COMPLETE.json',dict(protocol_sha256=b.digest(OUT/'protocol.json'),files={n:b.digest(d/n) for n in files}))

def completed(cfg):
    for fold in cfg['splits']:
        for seed in cfg['seeds']:
            d=CK/f'{fold}_seed{seed}';mark=b.read_json(d/'COMPLETE.json')
            assert mark['protocol_sha256']==b.digest(OUT/'protocol.json')
            for n,h in mark['files'].items():assert b.digest(d/n)==h

def evaluate(cfg):
    verify(cfg);completed(cfg);configure()
    if (OUT/'results.json').exists():raise RuntimeError('Refusing test overwrite')
    rows={};arrays={};diagnostics={};parent=b.read_json(ex.OUT/'results.json');oldarrays=np.load(ex.OUT/'curves.npz')
    for fold,split in cfg['splits'].items():
        rec=ex.CorrectedRecords(split['test'])
        for seed in cfg['seeds']:
            job=f'{fold}_seed{seed}';d=CK/job;old=ex.CK/job;enc=old/'independent';ex.encode(rec,enc)
            stats=dict(np.load(enc/'normalizers.npz'));limits=b.read_json(old/'limits.json')['limits'];ds=b.WindowData(rec,100,4)
            ws=[dict(file=rec.records[e]['name'],t=int(t)) for e,t in ds.idx]
            assert ws==b.read_json(old/'evaluation_windows.json');b.write_json(d/'evaluation_windows.json',ws)
            for name in VARIANTS:
                m=make(name,stats,enc,limits);m.load_state_dict(torch.load(d/(name+'.pt'),weights_only=False)['model'])
                monitor=c.Monitor();hook=m.register_forward_pre_hook(monitor.hook,with_kwargs=True);err=b.evaluate_model(m,ds);hook.remove()
                assert np.isfinite(err).all();key=f'{fold}/seed{seed}/{name}';rows[key]=b.metrics(err);arrays[key]=err;diagnostics[key]=monitor.result()
                print(key,rows[key],flush=True)
            for name in REFERENCES:
                key=f'{fold}/seed{seed}/{name}';rows[key]=parent['results'][key];arrays[key]=oldarrays[key];diagnostics[key]=parent['diagnostics'][key]
    np.savez_compressed(OUT/'curves.npz',**arrays)
    b.write_json(OUT/'results.json',dict(protocol_sha256=b.digest(OUT/'protocol.json'),results=rows,diagnostics=diagnostics,
        inherited_reference_names=REFERENCES))

def audit(cfg):
    verify(cfg);completed(cfg);configure();torch.set_num_threads(2)
    results=b.read_json(OUT/'results.json');arrays=np.load(OUT/'curves.npz');checks={};sub={};segments={}
    assert results['protocol_sha256']==b.digest(OUT/'protocol.json') and len(results['results'])==72
    assert set(arrays.files)==set(results['results'])
    for fold,split in cfg['splits'].items():
        rec=ex.CorrectedRecords(split['test']);expected=None
        for seed in cfg['seeds']:
            job=f'{fold}_seed{seed}';d=CK/job;old=ex.CK/job;enc=old/'independent';ex.encode(rec,enc)
            stats=dict(np.load(enc/'normalizers.npz'));limits=b.read_json(old/'limits.json')['limits'];ds=b.WindowData(rec,100,4)
            ws=b.read_json(d/'evaluation_windows.json');assert ws==b.read_json(old/'evaluation_windows.json')
            assert all(w['file'] in split['test'] for w in ws)
            if expected is not None:assert expected==ws
            expected=ws;indices=np.unique(np.linspace(0,len(ds)-1,24,dtype=int));subset=torch.utils.data.Subset(ds,indices)
            for name in list(VARIANTS)+REFERENCES:
                key=f'{fold}/seed{seed}/{name}';err=arrays[key];assert b.metrics(err)==results['results'][key]
                assert err.shape==(len(ws),101) and np.allclose(err[:,0],0,atol=1e-7)
                if name in VARIANTS:
                    saved=torch.load(d/(name+'.pt'),weights_only=False);hist=b.read_json(d/(name+'_history.json'));best=min(hist,key=lambda r:r['val_E100'])
                    assert len(hist)==40 and saved['epoch']==best['epoch'] and saved['val_E100']==best['val_E100']
                    m=make(name,stats,enc,limits);m.load_state_dict(saved['model']);assert saved['parameters']==sum(p.numel() for p in m.parameters())
                elif name=='guarded_full':
                    m=BranchQueryFrenet(b.make_model('frenet',stats,enc),limits,True,True)
                    m.load_state_dict(torch.load(old/(name+'.pt'),weights_only=False)['model'])
                else:m=None
                if m is not None:
                    measured=b.evaluate_model(m,subset,batch=24);delta=float(abs(measured-err[indices]).max())
                    assert np.allclose(measured,err[indices],atol=2e-5,rtol=2e-4),(key,delta)
                    m.core.kappa_grid.fill_(12345);m.core.total_len=.001
                    assert np.array_equal(measured,b.evaluate_model(m,subset,batch=24))
                    checks[key]=dict(windows=len(indices),max_absolute_difference_m=delta,map_buffer_independence=True)
                for source in sorted({b.PAT.fullmatch(w['file'])[1] for w in ws}):
                    mask=np.array([b.PAT.fullmatch(w['file'])[1]==source for w in ws]);sub[f'{fold}/{source}/seed{seed}/{name}']=b.metrics(err[mask])
                segments[key]={file:float(err[[w['file']==file for w in ws],-1].mean()) for file in sorted({w['file'] for w in ws})}
    b.write_json(OUT/'per_source_results.json',dict(results=sub));b.write_json(OUT/'per_segment_results.json',segments)
    b.write_json(OUT/'audit.json',dict(status='PASS',new_fits=27,reused_reference_rows=45,rows=72,checkpoint_spotchecks=checks,
        checks=['parent and source fingerprints','frozen input weights/normalizers/histories','all 40 epochs and validation minima','identical test windows','all stored metrics recomputed','all 27 checkpoints and 9 inherited full models recomputed on 24 windows','map-buffer independence','branch full implementation reproduces inherited full checkpoints']))

def report(cfg):
    rows=b.read_json(OUT/'results.json')['results'];sub=b.read_json(OUT/'per_source_results.json')['results'];seg=b.read_json(OUT/'per_segment_results.json')
    lines=['# 曲率进入学习分支的 2×2 对照','','27 个新增动力学训练；视觉编码器、几何递推、归一化、训练目标和保护规则固定。完整模型及基线复用已审计权重。三划分、三种子全部报告；回顾性研究，不是新增独立测试。','']
    comparisons={}
    for fold in cfg['splits']:
        lines += [f'## {fold}','','| 模型 | E25 | E50 | E100 | ADE |','|---|---:|---:|---:|---:|']
        for name in ['guarded_full',*VARIANTS,'kinematic','cartesian_guarded','guarded_static_context','geometry_midpoint']:
            vals=[]
            for metric in ['E25','E50','E100','ADE']:
                v=[rows[f'{fold}/seed{s}/{name}'][metric] for s in cfg['seeds']];vals.append(f'{np.mean(v):.3f} ± {np.std(v,ddof=1):.3f}')
            lines.append('| '+name+' | '+' | '.join(vals)+' |')
        for name in VARIANTS:
            differences=[rows[f'{fold}/seed{s}/{name}']['E100']-rows[f'{fold}/seed{s}/guarded_full']['E100'] for s in cfg['seeds']]
            files=seg[f'{fold}/seed0/{name}'];better=sum(np.mean([seg[f'{fold}/seed{s}/{name}'][file]-seg[f'{fold}/seed{s}/guarded_full'][file] for s in cfg['seeds']])<0 for file in files)
            comparisons[f'{fold}/{name}']=dict(paired_E100_differences=differences,mean_difference_m=float(np.mean(differences)),better_seeds=int(np.sum(np.array(differences)<0)),better_segments=better,segments=len(files))
        interaction=[rows[f'{fold}/seed{s}/guarded_full']['E100']-rows[f'{fold}/seed{s}/curvature_response_off']['E100']-rows[f'{fold}/seed{s}/curvature_pose_off']['E100']+rows[f'{fold}/seed{s}/curvature_both_off']['E100'] for s in cfg['seeds']]
        comparisons[fold+'/interaction']=dict(definition='full - response_off - pose_off + both_off; descriptive nonlinear endpoint-error interaction',per_seed=interaction,mean=float(np.mean(interaction)))
    lines+=['','## 时间划分的两个来源','','| 来源 | 模型 | E100 |','|---|---|---:|']
    for source in ['traj1_64x64','traj2_64x64']:
        for name in ['guarded_full',*VARIANTS]:
            vals=[sub[f'temporal/{source}/seed{s}/{name}']['E100'] for s in cfg['seeds']]
            lines.append(f'| {source} | {name} | {np.mean(vals):.3f} ± {np.std(vals,ddof=1):.3f} |')
    lines+=['','± 为三个训练种子的样本标准差。片段来自两个相关记录，不能当作独立采集。移除曲率后仍有 d/psi，不应称为完全道路无关模型。既有 geometry_midpoint 参考行的坐标计数仍仅覆盖全步；本轮无重训诊断另补其内部阶段。', '', '模型名称：GPT-6（Codex）。']
    b.write_json(OUT/'paired_comparisons.json',comparisons)
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def jobs(cfg):
    verify(cfg);queue=[(f,s) for f in cfg['splits'] for s in cfg['seeds'] if not (CK/f'{f}_seed{s}'/'COMPLETE.json').exists()];active=[]
    while queue or active:
        while queue and len(active)<3:
            fold,seed=queue.pop(0);fh=(OUT/f'{fold}_seed{seed}.log').open('a',encoding='utf-8')
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'fit','--fold',fold,'--seed',str(seed)],cwd=b.ROOT,stdout=fh,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active.append((p,fh,fold,seed));print('START',fold,seed,p.pid,flush=True)
        for item in active[:]:
            p,fh,fold,seed=item
            if p.poll() is not None:
                fh.close();active.remove(item);print('DONE',fold,seed,p.returncode,flush=True)
                if p.returncode:
                    for q,h,_,_ in active:q.terminate();q.wait();h.close()
                    raise RuntimeError('Fit failed; retained logs')
        time.sleep(2)
    evaluate(cfg);audit(cfg);report(cfg)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','prepare','fit','jobs','evaluate','audit','report']);ap.add_argument('--fold');ap.add_argument('--seed',type=int);args=ap.parse_args()
    if args.stage in ['preflight','prepare']:globals()[args.stage]()
    else:
        cfg=b.read_json(OUT/'protocol.json')
        if args.stage=='fit':fit(cfg,args.fold,args.seed)
        else:globals()[args.stage](cfg)
