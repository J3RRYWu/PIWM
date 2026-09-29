"""Frozen corrected-label experiment with retrained road-query controls.

All three inherited splits, all seeds, both encoder types and all controls are
retained. Evaluation waits until every prescribed fit is complete.
"""
from pathlib import Path
import sys,argparse,copy,subprocess,time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_holdout as b
import run_conditioned_dynamics as c
from models.conditioned_road_encoder import ConditionedRoadEncoder
from models.coupled_road_encoder import CoupledRoadEncoder
from baselines.query_controls import QueryFrenet,GuardedCartesian
from baselines.geometric_frenet import GeometricFrenet
from baselines.unit_arc_geometry import ArcFrenet
from baselines.frenet_diagnostic import FrenetDiagnostic
OUT=b.ROOT/'reports/corrected_query';CK=b.ROOT/'checkpoints/corrected_query';LABELS=b.ROOT/'reports/consistent_road_labels'
NAMES=['kinematic','cartesian_road','cartesian_guarded','guarded_full','guarded_fixed_query','guarded_static_context','geometry_midpoint','arc_full']
OriginalRecords=b.Records
class CorrectedRecords(OriginalRecords):
    def __init__(self,names):
        super().__init__(names)
        for r in self.records:
            lab=np.load(LABELS/'labels'/r['name']);assert np.array_equal(r['z'][:,3:],lab['z'][:,3:])
            r['z']=lab['z'].copy();r['shape']=lab['shape'].copy();r['kappa']=lab['kappa'].copy()

def clone(rec):
    obj=copy.copy(rec);obj.records=[r.copy() for r in rec.records];return obj

def make(name,stats,d,limits):
    if name in ['kinematic','cartesian_road']:return b.make_model(name,stats,d)
    if name=='cartesian_guarded':return GuardedCartesian(b.make_model('cartesian_road',stats,d),limits)
    core=b.make_model('frenet',stats,d)
    if name=='geometry_midpoint':return GeometricFrenet(core,limits,midpoint=True)
    if name=='arc_full':return ArcFrenet(core,limits)
    return QueryFrenet(core,limits,{'guarded_full':'dynamic','guarded_fixed_query':'fixed_query','guarded_static_context':'static_context'}[name])

def configure():
    b.DEVICE='cpu';torch.set_num_threads(4);c.make=make;c.VARIANTS={n:(True,'full',False) for n in NAMES}

def verify(cfg):
    for p,h in cfg['frozen_sources'].items():assert b.digest(b.ROOT/p)==h,p
    for p,h in cfg['parent_protocols'].items():assert b.digest(b.ROOT/p)==h,p
    assert b.digest(LABELS/'manifest.json')==cfg['labels_manifest_sha256']
    assert b.digest(b.DATA/'_meta/track.npz')==cfg['track_sha256']

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);CK.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise RuntimeError('Already frozen')
    old=b.read_json(b.OUT/'protocol.json');temp=b.read_json(b.ROOT/'reports/temporal_holdout/protocol.json');coupled=b.read_json(b.ROOT/'reports/coupled_road/protocol.json')
    labels=b.read_json(LABELS/'manifest.json');assert labels['status']=='PASS'
    files=dict(coupled['frozen_sources']);paths=[Path(__file__),b.ROOT/'src/arc_length_track.py',b.ROOT/'src/baselines/query_controls.py',b.ROOT/'scripts/build_consistent_road_labels.py']
    files.update({str(p.relative_to(b.ROOT)):b.digest(p) for p in paths})
    cfg=dict(status='Exploratory follow-up on reused records; fixed before these fits, not public preregistration or independent confirmation.',splits=dict(temporal=temp['splits']['temporal'],**old['splits']),seeds=[0,1,2],variants=NAMES,encoders=['independent','coupled'],perception_epochs=25,dynamics_epochs=40,train_horizon=32,test_horizon=100,batch=128,lr=.001,kinematic_lr=.01,primary_model='guarded_full',primary_metric='chronological pooled-window E100; both chronological source-specific results and both source-held-out directions mandatory',selection='each perception uses validation normalized target MSE; every dynamics checkpoint uses its own validation E100; no family is selected by test score',comparisons=['guarded_full versus guarded_fixed_query: both kinematic and neural curvature queries fixed at origin, intentionally breaking consistency away from origin','guarded_full versus guarded_static_context: correct dynamic kinematic curvature retained, neural curvature feature fixed at origin','cartesian_road versus cartesian_guarded: matched architecture with bounded speed/yaw increments, bounded pose residuals, wrapped heading','geometry_midpoint uses same independent encoder shape; arc_full uses separately trained coupled encoder'],limits='Train-only 99.5 percentile absolute one-frame speed/yaw-rate increments, lower floor .01; pose residual limits .1 per coordinate. Cartesian has both xy residuals; Frenet has lateral plus heading residual.',labels_manifest_sha256=b.digest(LABELS/'manifest.json'),frozen_sources=files,parent_protocols={p:b.digest(b.ROOT/p) for p in ['reports/controlled_holdout/protocol.json','reports/temporal_holdout/protocol.json','reports/coupled_road/protocol.json']},raw_fingerprints=temp['raw_fingerprints'],track_sha256=temp['track_sha256'],initialization='Corrected current-pose map projection for d/psi; raw speed, causal yaw rate; recorded future actions. Raw xy remains evaluation truth.',statistics='Seed sample SD, not independent-source uncertainty; overlapping windows; all prior protocols retained.')
    b.write_json(OUT/'protocol.json',cfg);print('Frozen 9 jobs, 18 new CNNs and 72 dynamics fits',flush=True)

def load_encoder(d,coupled=False):
    saved=torch.load(d/'perception.pt',map_location=b.VISION_DEVICE,weights_only=False);cls=CoupledRoadEncoder if coupled else ConditionedRoadEncoder;m=cls(saved['mean'],saved['scale']).to(b.VISION_DEVICE);m.load_state_dict(saved['model']);return m

def encode(rec,d,coupled=False):
    b.DEVICE=b.VISION_DEVICE;m=load_encoder(d,coupled);b.encode(m,rec);del m;b.DEVICE='cpu'

def perception(tr,va,d,cfg,seed,coupled):
    d.mkdir(parents=True,exist_ok=True);done=d/'PERCEPTION_COMPLETE.json'
    if done.exists():
        mark=b.read_json(done);assert mark['protocol_sha256']==b.digest(OUT/'protocol.json')
        for n,h in mark['files'].items():assert b.digest(d/n)==h
        return
    b.seed_all(seed);b.DEVICE=b.VISION_DEVICE;stats=b.fit_stats(tr,d);td=b.PerceptionData(tr,2);vd=b.PerceptionData(va,2)
    mean=np.stack([td[i][1].numpy() for i in range(len(td))]).mean(0);scale=stats['perception_scale'];cls=CoupledRoadEncoder if coupled else ConditionedRoadEncoder;m=cls(mean,scale).to(b.DEVICE);s=torch.tensor(scale,device=b.DEVICE);opt=torch.optim.Adam(m.parameters(),lr=.001);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,25);best=float('inf');hist=[]
    for epoch in range(25):
        m.train();total=0.;n=0
        for x,y in b.loader(td,64,True):
            x=x.to(b.DEVICE);y=y.to(b.DEVICE);k,p=m.forward_both(x);loss=((torch.cat([k,p.flatten(1)],-1)-y)/s).square().mean();assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),5));assert np.isfinite(gn);opt.step();total+=float(loss.detach())*len(x);n+=len(x)
        m.eval();value=0.;nv=0
        with torch.no_grad():
            for x,y in b.loader(vd,64):
                k,p=m.forward_both(x.to(b.DEVICE));value+=float(((torch.cat([k,p.flatten(1)],-1)-y.to(b.DEVICE))/s).square().mean())*len(x);nv+=len(x)
        value/=nv;hist.append(dict(epoch=epoch+1,train_loss=total/n,val_loss=value))
        if value<best:best=value;torch.save(dict(model=m.state_dict(),mean=mean,scale=scale,epoch=epoch+1,val_loss=value),d/'perception.pt')
        sch.step();b.write_json(d/'perception_history.json',hist);print(d.parent.name,d.name,'perception',epoch+1,f'{value:.6f}',flush=True)
    m.load_state_dict(torch.load(d/'perception.pt',map_location=b.DEVICE,weights_only=False)['model']);b.encode(m,tr);stats=b.context_stats(tr,stats);np.savez(d/'normalizers.npz',**stats);del m;b.DEVICE='cpu'
    b.write_json(done,dict(protocol_sha256=b.digest(OUT/'protocol.json'),files={n:b.digest(d/n) for n in ['perception.pt','normalizers.npz','stats.npz']}))

def fit(cfg,fold,seed):
    verify(cfg);configure();d=CK/f'{fold}_seed{seed}';d.mkdir(parents=True,exist_ok=True)
    if (d/'COMPLETE.json').exists():return
    if (OUT/'results.json').exists():raise RuntimeError('Test already evaluated')
    split=cfg['splits'][fold];tr=CorrectedRecords(split['train']);va=CorrectedRecords(split['validation']);tc=clone(tr);vc=clone(va)
    perception(tr,va,d/'independent',cfg,seed,False);perception(tc,vc,d/'coupled',cfg,seed,True)
    encode(tr,d/'independent');encode(va,d/'independent');encode(tc,d/'coupled',True);encode(vc,d/'coupled',True)
    delta=np.concatenate([abs(np.diff(r['z'][:,3:5],axis=0)) for r in tr.records]);limits=np.maximum(.01,np.quantile(delta,.995,axis=0)).tolist();b.write_json(d/'limits.json',dict(limits=limits,source='train only'))
    for name in NAMES:
        if (d/(name+'_history.json')).exists():assert len(b.read_json(d/(name+'_history.json')))==40;continue
        is_c=name=='arc_full';enc=d/('coupled' if is_c else 'independent');stats=dict(np.load(enc/'normalizers.npz'));local=dict(cfg,lr=.01 if name=='kinematic' else .001)
        c.fit(name,tc if is_c else tr,vc if is_c else va,stats,enc,d,local,limits,seed)
    b.write_json(d/'COMPLETE.json',dict(protocol_sha256=b.digest(OUT/'protocol.json'),files={str(p.relative_to(d)):b.digest(p) for p in [*[d/(n+'.pt') for n in NAMES],d/'limits.json',*[d/e/n for e in ['independent','coupled'] for n in ['perception.pt','normalizers.npz','stats.npz']]]}))

def preflight():
    configure();cfg=b.read_json(b.ROOT/'reports/temporal_holdout/protocol.json');tr=CorrectedRecords(cfg['splits']['temporal']['train']);d=OUT/'preflight';d.mkdir(parents=True,exist_ok=True)
    for r in tr.records:r['pred_kappa']=r['kappa'];r['pred_shape']=r['shape']
    stats=b.context_stats(tr,b.fit_stats(tr,d));limits=np.maximum(.01,np.quantile(np.concatenate([abs(np.diff(r['z'][:,3:5],axis=0)) for r in tr.records]),.995,axis=0)).tolist()
    ds=b.WindowData(tr,32,4);z,a,y,k,p=next(iter(b.loader(torch.utils.data.Subset(ds,list(range(8))),8)))
    checks={}
    for name in NAMES:
        b.seed_all(0);m=make(name,stats,d,limits);pred=m.rollout(z,a,k,p);loss=b.loss_function(pred,y,stats);loss.backward()
        assert torch.isfinite(pred).all() and all(x.grad is None or torch.isfinite(x.grad).all() for x in m.parameters())
        assert torch.allclose(pred[:,0,:3],torch.zeros_like(pred[:,0,:3]));checks[name]=dict(parameters=sum(x.numel() for x in m.parameters()),finite_backward=True)
    core=b.make_model('frenet',stats,d);new=QueryFrenet(copy.deepcopy(core),limits,'dynamic');old=FrenetDiagnostic(copy.deepcopy(core),True,'full',limits)
    with torch.no_grad():assert torch.allclose(new.rollout(z,a,k,p),old.rollout(z,a,k,p),atol=1e-6,rtol=1e-6)
    constant=k[:,:1].expand_as(k)
    with torch.no_grad():
        reference=new.rollout(z,a,constant,p)
        for mode in ['fixed_query','static_context']:
            m=QueryFrenet(copy.deepcopy(core),limits,mode);assert torch.allclose(reference,m.rollout(z,a,constant,p),atol=1e-6,rtol=1e-6)
    b.write_json(OUT/'preflight.json',dict(status='PASS',checks=checks,identities=['dynamic query exactly reproduces prior guarded implementation on the same inputs','all query modes coincide for constant curvature profiles','all variants finite 32-step forward/backward, zero initial ego positions']))
    print('Preflight PASS',checks,flush=True)

def evaluate(cfg):
    verify(cfg);configure()
    if (OUT/'results.json').exists():raise RuntimeError('Refusing test overwrite')
    for fold in cfg['splits']:
        for seed in cfg['seeds']:
            d=CK/f'{fold}_seed{seed}';mark=b.read_json(d/'COMPLETE.json');assert mark['protocol_sha256']==b.digest(OUT/'protocol.json')
            for name,h in mark['files'].items():assert b.digest(d/name)==h
    results={};curves={};diagnostics={}
    for fold,split in cfg['splits'].items():
        rec=CorrectedRecords(split['test']);rc=clone(rec)
        for seed in cfg['seeds']:
            d=CK/f'{fold}_seed{seed}';encode(rec,d/'independent');encode(rc,d/'coupled',True);limits=b.read_json(d/'limits.json')['limits']
            for name in NAMES:
                is_c=name=='arc_full';enc=d/('coupled' if is_c else 'independent');stats=dict(np.load(enc/'normalizers.npz'));ds=b.WindowData(rc if is_c else rec,100,4);m=make(name,stats,enc,limits);m.load_state_dict(torch.load(d/(name+'.pt'),weights_only=False)['model'])
                monitor=c.Monitor();hook=m.register_forward_pre_hook(monitor.hook,with_kwargs=True);err=b.evaluate_model(m,ds);hook.remove();key=f'{fold}/seed{seed}/{name}';curves[key]=err;results[key]=b.metrics(err);diagnostics[key]=monitor.result();print(key,results[key],flush=True)
            b.write_json(d/'evaluation_windows.json',[dict(file=rec.records[e]['name'],t=int(t)) for e,t in ds.idx])
    np.savez_compressed(OUT/'curves.npz',**curves);b.write_json(OUT/'results.json',dict(protocol_sha256=b.digest(OUT/'protocol.json'),results=results,diagnostics=diagnostics))

def audit(cfg):
    verify(cfg);labels=b.read_json(LABELS/'manifest.json');parent=b.read_json(b.OUT/'protocol.json')
    for name,v in labels['files'].items():assert b.digest(LABELS/'labels'/name)==v['sha256']
    for name,v in parent['files'].items():assert b.digest(b.DATA/name)==v['sha256']
    for name,h in cfg['raw_fingerprints'].items():assert b.digest(b.RAW/name)==h
    data=b.read_json(OUT/'results.json');curves=np.load(OUT/'curves.npz');assert data['protocol_sha256']==b.digest(OUT/'protocol.json');assert len(data['results'])==72 and set(curves.files)==set(data['results']);sub={}
    for fold,split in cfg['splits'].items():
        assert not set(split['train'])&set(split['validation']) and not set(split['train'])&set(split['test']) and not set(split['validation'])&set(split['test'])
        expected=None
        for seed in cfg['seeds']:
            d=CK/f'{fold}_seed{seed}';mark=b.read_json(d/'COMPLETE.json');assert mark['protocol_sha256']==data['protocol_sha256']
            for n,h in mark['files'].items():assert b.digest(d/n)==h
            ws=b.read_json(d/'evaluation_windows.json')
            if expected is not None:assert ws==expected
            expected=ws;assert all(w['file'] in split['test'] for w in ws)
            oldpath=(b.ROOT/'checkpoints/temporal_holdout' if fold=='temporal' else b.CK)/d.name/'evaluation_windows.json'
            if oldpath.exists():assert ws==b.read_json(oldpath)
            for kind in ['independent','coupled']:
                hist=b.read_json(d/kind/'perception_history.json');best=min(hist,key=lambda x:x['val_loss']);saved=torch.load(d/kind/'perception.pt',weights_only=False);assert len(hist)==25 and saved['epoch']==best['epoch'] and saved['val_loss']==best['val_loss']
            for name in NAMES:
                hist=b.read_json(d/(name+'_history.json'));best=min([r for r in hist if r['val_E100'] is not None],key=lambda x:x['val_E100']);saved=torch.load(d/(name+'.pt'),weights_only=False);assert len(hist)==40 and saved['epoch']==best['epoch'] and saved['val_E100']==best['val_E100']
                key=f'{fold}/seed{seed}/{name}';err=curves[key];assert err.shape==(len(ws),101) and np.allclose(err[:,0],0,atol=1e-7);assert b.metrics(err)==data['results'][key]
                if fold=='temporal':
                    for source in ['traj1_64x64','traj2_64x64']:
                        mask=np.array([b.PAT.fullmatch(w['file'])[1]==source for w in ws]);sub[f'{source}/seed{seed}/{name}']=b.metrics(err[mask])
    b.write_json(OUT/'per_source_results.json',dict(results=sub));b.write_json(OUT/'audit.json',dict(status='PASS',rows=72,checks=['frozen sources and parents','raw/prepared/corrected-label hashes','all seeds, 25/40 epochs, validation minima','disjoint inherited partitions','identical windows across seeds and inherited protocols where manifests exist','checkpoint/normalizer hashes','all metrics recalculated from saved errors','zero initial position errors']))

def report(cfg):
    data=b.read_json(OUT/'results.json')['results'];sub=b.read_json(OUT/'per_source_results.json')['results'];lines=['# 修正几何标签与道路查询机制对照','','此轮为旧记录上的探索性复核。协议在拟合前冻结，全部 3 个划分、3 个种子、8 个动力学模型与两类编码器保留；主模型仍是 guarded_full。E100 为主指标。','所有道路标签来自同一条弧长参数化的周期曲线；原始 xy 真值与测试窗口不变。独立／耦合编码器分别从零训练，不能把它们之间的差异全部归因于一个几何约束。','']
    for fold in list(cfg['splits'])+['traj1_64x64','traj2_64x64']:
        rows=data if fold in cfg['splits'] else sub;lines += [f'## {fold}','','| 模型 | E25 | E50 | E100 | ADE |','|---|---:|---:|---:|---:|']
        for name in NAMES:
            vals=[]
            for metric in ['E25','E50','E100','ADE']:
                v=[rows[f'{fold}/seed{s}/{name}'][metric] for s in cfg['seeds']];vals.append('nonfinite' if None in v else f'{np.mean(v):.3f} ± {np.std(v,ddof=1):.3f}')
            lines.append('| '+name+' | '+' | '.join(vals)+' |')
    lines+=['','guarded_fixed_query 同时固定运动学与神经曲率，属于故意引入几何错配的控制，不能单独证明神经网络使用局部曲率的价值。guarded_static_context 保留随进度变化的运动学曲率，仅固定神经特征，才更直接检验该问题。',
            'Cartesian 保护版保留原网络与信息，增加有界速度／角速度增量、位姿残差限幅与角度回绕。其 xy 两个残差与 Frenet 单个横向残差不是一一等价，但比无保护对照更严格。',
            '误差条为种子样本标准差，不是独立来源置信区间。全步 chart 监测不涵盖 midpoint 内部阶段；有限误差不等于坐标始终有效。已有原始数据、旧模型和不利结果均保留。','',
            '模型名称：GPT-6（Codex）；当前会话未提供更细型号标识。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def jobs(cfg):
    queue=[(f,s) for f in cfg['splits'] for s in cfg['seeds'] if not (CK/f'{f}_seed{s}'/'COMPLETE.json').exists()];active=[]
    while queue or active:
        while queue and len(active)<3:
            fold,seed=queue.pop(0);fh=(OUT/f'{fold}_seed{seed}.log').open('a',encoding='utf-8');p=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'fit','--fold',fold,'--seed',str(seed)],cwd=b.ROOT,stdout=fh,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));active.append((p,fh,fold,seed));print('START',fold,seed,p.pid,flush=True)
        for item in active[:]:
            p,fh,fold,seed=item
            if p.poll() is not None:
                fh.close();active.remove(item);print('DONE',fold,seed,p.returncode,flush=True)
                if p.returncode:
                    for q,h,_,_ in active:q.terminate();q.wait();h.close()
                    raise RuntimeError('Fit failed; inspect logs')
        time.sleep(2)
    evaluate(cfg);audit(cfg);report(cfg)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','preflight','fit','jobs','evaluate','audit','report']);ap.add_argument('--fold');ap.add_argument('--seed',type=int);args=ap.parse_args();configure()
    if args.stage=='preflight':preflight();return
    if args.stage=='prepare':preflight();prepare();return
    cfg=b.read_json(OUT/'protocol.json')
    if args.stage=='fit':fit(cfg,args.fold,args.seed)
    else:globals()[args.stage](cfg)
if __name__=='__main__':main()
