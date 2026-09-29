"""One fixed-budget exploratory physical-scene pilot on inherited partitions."""
from pathlib import Path
import sys, argparse, subprocess, time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_corrected_query as ex
from baselines.physical_scene import PhysicalScene, to_vehicle, to_fixed
b, c = ex.b, ex.c
OUT=b.ROOT/'reports/physical_scene'; CK=b.ROOT/'checkpoints/physical_scene'
NAME='physical_scene'


def configure():
    ex.configure()
    c.VARIANTS={NAME:(False,'none',False)}
    c.make=lambda name,stats,d,limits: PhysicalScene(float(stats['throttle_reference']))


def attach_placeholder(rec):
    # Dynamics are road-blind; no oracle geometry is consumed during fitting.
    for r in rec.records:
        r['pred_kappa']=np.zeros_like(r['kappa'])
        r['pred_shape']=np.zeros_like(r['shape'])


def training_audit(rec):
    a=np.concatenate([r['actions'] for r in rec.records]).astype(np.float64)
    z=np.concatenate([r['z'] for r in rec.records]).astype(np.float64)
    u=a[:,1]; v=z[:,3]; ref=float(u.mean())
    X=np.stack([u-ref,np.ones_like(u),-v,-v*abs(v)],-1)
    singular=np.linalg.svd(X,compute_uv=False)
    yawX=np.stack([v*a[:,0],v*a[:,0]**3,v,-z[:,4]],-1)
    sy=np.linalg.svd(yawX,compute_uv=False)
    return dict(throttle_reference=ref,throttle_min=float(u.min()),throttle_max=float(u.max()),
        longitudinal_design_rank=int(np.linalg.matrix_rank(X,tol=1e-7)),longitudinal_singular_values=singular.tolist(),
        yaw_design_rank=int(np.linalg.matrix_rank(yawX,tol=1e-7)),yaw_singular_values=sy.tolist(),
        interpretation='Training-input excitation diagnostic only; full rank does not prove nonlinear dynamic identifiability. Constant throttle makes drive gain unidentifiable.')


def prepare():
    configure();OUT.mkdir(parents=True,exist_ok=True);CK.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    old=b.read_json(ex.OUT/'protocol.json');ex.verify(old)
    # Float64 transport inverse, road independence, equilibrium, gradients.
    torch.manual_seed(31);points=torch.randn(5,10,2,dtype=torch.float64);pose=torch.randn(5,3,dtype=torch.float64)
    err=float((to_fixed(to_vehicle(points,pose),pose)-points).abs().max());assert err<1e-12
    m=PhysicalScene().double();z=torch.zeros(5,5,dtype=torch.float64);z[:,3]=.6
    a=torch.zeros(5,100,2,dtype=torch.float64);a[:,:,1]=.6
    pred=m.rollout(z,a);assert torch.isfinite(pred).all();pred.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())
    assert torch.equal(pred,m.rollout(z,a,torch.randn(5,10),points))
    scene=m.scene_rollout(z,a,points);back=to_fixed(scene['road_points'],scene['vehicle'][...,:3])
    transport=float((back-scene['initial_road_points'][:,None]).detach().abs().max());assert transport<1e-12
    # Zero speed, zero reference drive has no spontaneous translation/rotation.
    with torch.no_grad():m.raw[6]=0
    zz=torch.zeros_like(z);aa=torch.zeros_like(a);aa[:,:,1]=m.throttle_reference
    assert torch.equal(m.rollout(zz,aa),torch.zeros(5,101,5,dtype=torch.float64))
    audits={}
    for fold,split in old['splits'].items():audits[fold]=training_audit(ex.CorrectedRecords(split['train']))
    b.write_json(OUT/'training_identifiability.json',audits)
    b.write_json(OUT/'preflight.json',dict(status='PASS',inverse_error=err,transport_error=transport,
        finite_100step_backward=True,road_independent_vehicle=True,stationary_equilibrium=True))
    paths=[Path(__file__),b.ROOT/'src/baselines/physical_scene.py']
    inputs={}
    for fold in old['splits']:
        for seed in old['seeds']:
            d=ex.CK/f'{fold}_seed{seed}'
            inputs[d.name]={n:b.digest(d/n) for n in ['independent/normalizers.npz','independent/perception.pt','evaluation_windows.json','kinematic.pt','guarded_full.pt']}
    cfg=dict(splits=old['splits'],seeds=[0,1,2],lr=.01,epochs=40,train_horizon=32,test_horizon=100,batch=128,stride=4,
        selection='Minimum validation E100; inherited Adam/cosine40/clip1; all 9 fits before test evaluation',
        status='Exploratory redesign on repeatedly reused data. No independent validation. Not a novelty or acceptance claim.',
        model='Eight effective body response coefficients, midpoint integration, no learned pose residual; rigid transport of finite perceived road',
        reference='Reuse 7-parameter kinematic and neural full references fitted with inherited protocol. Candidate uses identical loss, windows, epochs; kinematic learning rate .01.',
        decision='Keep as performance replacement only if temporal E100 <= full and each source-held-out E100 <= kinematic; otherwise report tradeoff, do not replace manuscript performance claims.',
        scope='No new CNN, history estimator, parameter search, or post-test refit. No tire/mass identification claim.',
        sources={str(p.relative_to(b.ROOT)):b.digest(p) for p in paths},inputs=inputs,
        training_audit_sha256=b.digest(OUT/'training_identifiability.json'),parent_sha256=b.digest(ex.OUT/'protocol.json'),
        references_sha256={n:b.digest(ex.OUT/n) for n in ['results.json','curves.npz']})
    b.write_json(OUT/'protocol.json',cfg);print('PREFLIGHT PASS; FROZEN 9 fits',flush=True)


def verify(cfg):
    ex.verify(b.read_json(ex.OUT/'protocol.json'))
    assert cfg['parent_sha256']==b.digest(ex.OUT/'protocol.json')
    for n,h in cfg['sources'].items():assert b.digest(b.ROOT/n)==h
    assert cfg['training_audit_sha256']==b.digest(OUT/'training_identifiability.json')
    for job,files in cfg['inputs'].items():
        for n,h in files.items():assert b.digest(ex.CK/job/n)==h
    for n,h in cfg['references_sha256'].items():assert b.digest(ex.OUT/n)==h


def fit(cfg,fold,seed):
    verify(cfg);configure();d=CK/f'{fold}_seed{seed}';d.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'results.json').exists()
    if (d/'COMPLETE.json').exists():return
    split=cfg['splits'][fold];tr=ex.CorrectedRecords(split['train']);va=ex.CorrectedRecords(split['validation'])
    attach_placeholder(tr);attach_placeholder(va)
    enc=ex.CK/d.name/'independent';stats=dict(np.load(enc/'normalizers.npz'))
    stats['throttle_reference']=b.read_json(OUT/'training_identifiability.json')[fold]['throttle_reference']
    c.fit(NAME,tr,va,stats,enc,d,cfg,[],seed)
    b.write_json(d/'COMPLETE.json',dict(protocol_sha256=b.digest(OUT/'protocol.json'),files={n:b.digest(d/n) for n in [NAME+'.pt',NAME+'_last.pt',NAME+'_history.json']}))


def evaluate(cfg):
    verify(cfg);configure()
    for fold in cfg['splits']:
        for seed in cfg['seeds']:
            d=CK/f'{fold}_seed{seed}';mark=b.read_json(d/'COMPLETE.json');assert mark['protocol_sha256']==b.digest(OUT/'protocol.json')
            for n,h in mark['files'].items():assert b.digest(d/n)==h
    assert not (OUT/'results.json').exists()
    arrays={};rows={};checks={};parameters={};scene_metrics={};sub={}
    oldarrays=np.load(ex.OUT/'curves.npz');oldrows=b.read_json(ex.OUT/'results.json')['results']
    for fold,split in cfg['splits'].items():
        rec=ex.CorrectedRecords(split['test']);attach_placeholder(rec);ds=b.WindowData(rec,100,4)
        windows=[dict(file=rec.records[e]['name'],t=int(t)) for e,t in ds.idx]
        for seed in cfg['seeds']:
            d=CK/f'{fold}_seed{seed}';assert windows==b.read_json(ex.CK/d.name/'evaluation_windows.json')
            saved=torch.load(d/(NAME+'.pt'),weights_only=False);hist=b.read_json(d/(NAME+'_history.json'))
            best=min(hist,key=lambda x:x['val_E100']);assert len(hist)==40 and saved['epoch']==best['epoch'] and saved['val_E100']==best['val_E100']
            m=PhysicalScene();m.load_state_dict(saved['model']);err=b.evaluate_model(m,ds)
            assert np.isfinite(err).all();key=f'{fold}/seed{seed}/{NAME}';arrays[key]=err;rows[key]=b.metrics(err);parameters[key]=m.physical()
            indices=np.unique(np.linspace(0,len(ds)-1,24,dtype=int));subset=torch.utils.data.Subset(ds,indices)
            recompute=b.evaluate_model(m,subset,batch=24);assert np.allclose(recompute,err[indices],atol=2e-5,rtol=2e-4)
            assert np.array_equal(err,b.evaluate_model(m,ds,mode='zero'))
            checks[key]=dict(recomputed_windows=len(indices),max_difference=float(abs(recompute-err[indices]).max()),road_independence=True)
            # Perceived road is used ONLY for scene readout. Transport error is
            # a pose-induced landmark discrepancy, NOT independent road accuracy.
            ex.encode(rec,ex.CK/d.name/'independent');landmarks=[];negative=[]
            with torch.no_grad():
                for z,a,y,k,p in b.loader(ds,128):
                    scene=m.scene_rollout(z,a,p);oracle_pose_road=to_vehicle(scene['initial_road_points'][:,None],y[:,:,:3])
                    landmarks.append(torch.linalg.vector_norm(scene['road_points'][:,-1]-oracle_pose_road[:,-1],dim=-1).mean(-1).numpy())
                    negative.append((scene['vehicle'][:,:,3]<0).numpy())
                    back=to_fixed(scene['road_points'],scene['vehicle'][...,:3])
                    assert torch.allclose(back,scene['initial_road_points'][:,None].expand_as(back),atol=3e-6,rtol=1e-5)
            scene_metrics[key]=dict(pose_induced_road_landmark_E100=float(np.concatenate(landmarks).mean()),negative_speed_fraction=float(np.concatenate(negative).mean()))
            for n in ['kinematic','guarded_full']:
                ref=f'{fold}/seed{seed}/{n}';arrays[ref]=oldarrays[ref];rows[ref]=oldrows[ref]
            for n in [NAME,'kinematic','guarded_full']:
                ref=f'{fold}/seed{seed}/{n}'
                for source in sorted({b.PAT.fullmatch(w['file'])[1] for w in windows}):
                    mask=np.array([b.PAT.fullmatch(w['file'])[1]==source for w in windows]);sub[f'{fold}/{source}/seed{seed}/{n}']=b.metrics(arrays[ref][mask])
            b.write_json(d/'evaluation_windows.json',windows);print(key,rows[key],flush=True)
    np.savez_compressed(OUT/'curves.npz',**arrays)
    b.write_json(OUT/'results.json',dict(protocol_sha256=b.digest(OUT/'protocol.json'),results=rows,parameters=parameters,scene=scene_metrics))
    # Every reported metric recalculated after serialization.
    stored=np.load(OUT/'curves.npz')
    for key in rows:assert b.metrics(stored[key])==rows[key]
    b.write_json(OUT/'per_source_results.json',sub)
    b.write_json(OUT/'audit.json',dict(status='PASS',new_fits=9,reused_rows=18,checks=checks,validation_minima=True,frozen_hashes=True,all_metrics_recomputed=True,scene_inverse_checked_all_test_windows=True))
    report(cfg)


def report(cfg):
    data=b.read_json(OUT/'results.json');rows=data['results'];lines=['# 显式车辆动力学与道路联合状态：有限实验','','这是更换模型结构后的探索性实验。9 个新增拟合，三个旧划分 × 三个种子；没有新采集或独立验证。','', '| 划分 | 模型 | E25 | E50 | E100 | ADE |','|---|---|---:|---:|---:|---:|']
    means={}
    for fold in cfg['splits']:
        means[fold]={}
        for name in [NAME,'kinematic','guarded_full']:
            vals=[]
            for metric in ['E25','E50','E100','ADE']:
                x=[rows[f'{fold}/seed{s}/{name}'][metric] for s in cfg['seeds']];vals.append(f'{np.mean(x):.4f} ± {np.std(x,ddof=1):.4f}')
                if metric=='E100':means[fold][name]=float(np.mean(x))
            lines.append('| '+' | '.join([fold,name,*vals])+' |')
    conditions={fold:means[fold][NAME]<=means[fold]['guarded_full' if fold=='temporal' else 'kinematic'] for fold in cfg['splits']}
    b.write_json(OUT/'decision.json',dict(replace_performance_model=all(conditions.values()),conditions=conditions,mean_E100=means))
    lines+=['','预先固定的替代条件：时间划分不差于原完整模型，两个来源留出划分均不差于标定运动学。',f'结果：{conditions}；整体替代条件={all(conditions.values())}。','', '± 是训练种子样本标准差，不能当作跨环境置信区间。所有结果保留，不按有利划分挑选。', '', '模型名称：GPT-6（Codex）。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def jobs(cfg):
    verify(cfg);queue=[(f,s) for f in cfg['splits'] for s in cfg['seeds'] if not (CK/f'{f}_seed{s}'/'COMPLETE.json').exists()];active=[]
    while queue or active:
        while queue and len(active)<3:
            f,s=queue.pop(0);fh=(OUT/f'{f}_seed{s}.log').open('a',encoding='utf-8')
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'fit','--fold',f,'--seed',str(s)],cwd=b.ROOT,stdout=fh,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));active.append((p,fh,f,s));print('START',f,s,flush=True)
        for item in active[:]:
            p,fh,f,s=item
            if p.poll() is not None:
                fh.close();active.remove(item);print('DONE',f,s,p.returncode,flush=True)
                if p.returncode:
                    for q,h,_,_ in active:q.terminate();q.wait();h.close()
                    raise RuntimeError('Fit failed; inspect logs')
        time.sleep(2)
    evaluate(cfg)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','fit','jobs','report']);ap.add_argument('--fold');ap.add_argument('--seed',type=int);args=ap.parse_args()
    if args.stage=='prepare':prepare()
    else:
        cfg=b.read_json(OUT/'protocol.json')
        if args.stage=='fit':fit(cfg,args.fold,args.seed)
        else:globals()[args.stage](cfg)

