"""Conditioned-perception, explicitly exploratory experiment; original frozen runs untouched.

Uses six newly conditioned training-only CNNs, fixed before any dynamics fit. Four dynamics variants per fold/seed
are fit from scratch. All evaluation sources have been inspected previously.
No new-data generalization or independent confirmation is claimed.
"""
from pathlib import Path
import argparse,copy,json,subprocess,sys,time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_holdout as base
from baselines.frenet_diagnostic import FrenetDiagnostic
from baselines.geometric_frenet import GeometricFrenet
from models.conditioned_road_encoder import ConditionedRoadEncoder
from baselines.controlled_dynamics import rollout
ROOT=base.ROOT;OLD=base.OUT;OLDCK=base.CK
OUT=ROOT/'reports/conditioned_dynamics';CK=ROOT/'checkpoints/conditioned_dynamics'
ENC=ROOT/'checkpoints/conditioned_perception'
VARIANTS={'guarded_full':(True,'full',False),'guarded_shape':(True,'shape',False),'geometry_midpoint':(True,'full',False),'cartesian_road':(True,'full',False)}

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);CK.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise RuntimeError('Protocol exists')
    old=base.read_json(OLD/'protocol.json')
    paths=[Path(__file__),ROOT/'src/baselines/frenet_diagnostic.py',ROOT/'src/baselines/geometric_frenet.py',ROOT/'src/baselines/consistent_road_geometry.py',ROOT/'src/models/conditioned_road_encoder.py',ROOT/'scripts/run_controlled_holdout.py',
           ROOT/'src/baselines/controlled_dynamics.py',ROOT/'src/models/frenet_dynamics.py',
           ROOT/'src/models/road_perception.py',ROOT/'src/frenet_track.py',ROOT/'scripts/check_frenet_diagnostic.py']
    cfg=dict(parent_protocol_sha256=base.digest(OLD/'protocol.json'),seeds=old['seeds'],splits=old['splits'],
             encoder_sha256={f'{f}_seed{s}':base.digest(ENC/f'{f}_seed{s}'/'perception.pt') for f in old['splits'] for s in old['seeds']},
             variants=VARIANTS,epochs=40,horizon=32,batch=128,stride=4,lr=.001,
             increment_limit='max(0.01, training-only 99.5 percentile abs one-frame delta), separately v and causal omega',
             guards='positive denominator floor 0.2; wrap psi; tanh-bounded dv/domega and 0.1*tanh pose residual',
             selection='minimum finite validation E100; all 3 seeds retained',
             status='Exploratory follow-up after seeing previous test results; not a fresh independent test',
             sources={str(p.relative_to(ROOT)):base.digest(p) for p in paths},
             track_sha256=base.digest(base.DATA/'_meta/track.npz'),
             inherited_data_sha256=base.digest(OLD/'raw_data_fingerprints.json'),
             frozen_inputs={f'{fold}_seed{seed}':{name:base.digest(OLDCK/f'{fold}_seed{seed}'/name)
                 for name in ['perception.pt','normalizers.npz','stats.npz']}
                 for fold in old['splits'] for seed in old['seeds']})
    base.write_json(OUT/'protocol.json',cfg)
    print('Frozen exploratory protocol:',OUT/'protocol.json')

def verify(cfg):
    for job,h in cfg['encoder_sha256'].items():assert base.digest(ENC/job/'perception.pt')==h
    assert cfg['parent_protocol_sha256']==base.digest(OLD/'protocol.json')
    for name,h in cfg['sources'].items():assert base.digest(ROOT/name)==h,name
    assert cfg['track_sha256']==base.digest(base.DATA/'_meta/track.npz')
    for job,files in cfg['frozen_inputs'].items():
        for name,h in files.items():assert base.digest(OLDCK/job/name)==h,(job,name)

def encode(records,old):
    base.DEVICE=base.VISION_DEVICE
    saved=torch.load(ENC/old.name/'perception.pt',map_location=base.DEVICE,weights_only=False)
    model=ConditionedRoadEncoder(saved['mean'],saved['scale']).to(base.DEVICE)
    model.load_state_dict(saved['model'])
    base.encode(model,records);del model
    base.DEVICE='cpu'

def oracle_copy(records):
    out=copy.copy(records);out.records=[]
    for r in records.records:
        v=r.copy();v['pred_kappa']=r['kappa'];v['pred_shape']=r['shape'];out.records.append(v)
    return out

def make(name,stats,old,limits):
    if name=='cartesian_road':return base.make_model(name,stats,old)
    core=base.make_model('frenet',stats,old)
    if name=='geometry_midpoint':return GeometricFrenet(core,limits,midpoint=True)
    guarded,road,_=VARIANTS[name]
    return FrenetDiagnostic(core,guarded,road,limits)

class Monitor:
    def __init__(self):
        self.n=0;self.negative=0;self.near=0;self.psi=0;self.maxabs=np.zeros(5);self.denmin=float('inf')
    def hook(self,m,args,kwargs):
        z=args[0].detach();k=kwargs['kappa_override'].detach();den=1-z[:,1]*k
        self.n+=len(z);self.negative+=int((den<=0).sum());self.near+=int((den.abs()<.2).sum())
        self.psi+=int((z[:,2].abs()>torch.pi).sum());self.denmin=min(self.denmin,float(den.min()))
        self.maxabs=np.maximum(self.maxabs,z.abs().amax(0).numpy())
    def result(self):
        return dict(samples=self.n,invalid_chart_fraction=self.negative/max(1,self.n),
                    near_singular_fraction=self.near/max(1,self.n),psi_outside_pi_fraction=self.psi/max(1,self.n),
                    raw_den_min=self.denmin,max_abs_state=self.maxabs.tolist())

def fit(name,tr,va,stats,old,d,cfg,limits,seed):
    base.seed_all(seed);m=make(name,stats,old,limits)
    if VARIANTS[name][2]:tr=oracle_copy(tr);va=oracle_copy(va)
    td=base.WindowData(tr,32,4);vd=base.WindowData(va,100,4)
    opt=torch.optim.Adam(m.parameters(),lr=cfg['lr']);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,40)
    hist=[];best=float('inf')
    for ep in range(40):
        m.train();total=0;n=0;gnmax=0;clipped=0;nb=0;tm=Monitor()
        handle=m.register_forward_pre_hook(tm.hook,with_kwargs=True)
        for z,a,y,k,p in base.loader(td,128,True):
            pred=rollout(m,z,a,k,p);loss=base.loss_function(pred,y,stats)
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite train loss {d.name}/{name}/{ep+1}')
            opt.zero_grad();loss.backward()
            gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1))
            if not np.isfinite(gn):raise RuntimeError(f'Nonfinite gradient {d.name}/{name}/{ep+1}')
            opt.step();gnmax=max(gnmax,gn);clipped+=gn>1;nb+=1;total+=float(loss.detach())*len(z);n+=len(z)
        handle.remove();vm=Monitor();handle=m.register_forward_pre_hook(vm.hook,with_kwargs=True)
        errors=base.evaluate_model(m,vd);handle.remove();score=float(errors[:,100].mean())
        row=dict(epoch=ep+1,train_loss=total/n,val_E100=score,max_preclip_gradient=gnmax,
                 fraction_clipped_batches=clipped/nb,training_state=tm.result(),validation_state=vm.result())
        hist.append(row)
        checkpoint=dict(model=m.state_dict(),epoch=ep+1,val_E100=score,variant=name,limits=limits,
                        parameters=sum(p.numel() for p in m.parameters()))
        if np.isfinite(score) and score<best:
            best=score;torch.save(checkpoint,d/(name+'.pt'))
        if ep==39:torch.save(checkpoint,d/(name+'_last.pt'))
        sched.step();base.write_json(d/(name+'_progress.json'),hist)
        print(d.name,name,f'{ep+1}/40',f'train={total/n:.4f} val={score:.4f} grad={gnmax:.1f} invalid={vm.result()["invalid_chart_fraction"]:.4f}',flush=True)
    if not (d/(name+'.pt')).exists():raise RuntimeError('No finite checkpoint')
    base.write_json(d/(name+'_history.json'),hist)

def train(cfg,fold,seed):
    verify(cfg)
    if (OUT/'results.json').exists():raise RuntimeError('Follow-up evaluation already completed')
    d=CK/f'{fold}_seed{seed}';old=OLDCK/d.name;d.mkdir(parents=True,exist_ok=True)
    if (d/'COMPLETE.json').exists():return
    split=cfg['splits'][fold];tr=base.Records(split['train']);va=base.Records(split['validation'])
    stats=dict(np.load(old/'normalizers.npz'))
    delta=np.concatenate([np.abs(np.diff(r['z'][:,3:5],axis=0)) for r in tr.records])
    limits=np.maximum(.01,np.quantile(delta,.995,axis=0)).tolist()
    base.write_json(d/'limits.json',dict(limits=limits,source='train only'))
    encode(tr,old);encode(va,old)
    stats=base.context_stats(tr,stats);np.savez(d/'normalizers.npz',**stats)
    for name in VARIANTS:
        if (d/(name+'_history.json')).exists():continue
        fit(name,tr,va,stats,old,d,cfg,limits,seed)
    base.write_json(d/'COMPLETE.json',dict(protocol_sha256=base.digest(OUT/'protocol.json'),
         checkpoint_sha256={name:base.digest(d/(name+'.pt')) for name in VARIANTS}))

def evaluate(cfg):
    verify(cfg)
    if (OUT/'results.json').exists():raise RuntimeError('Refusing evaluation overwrite')
    for f in cfg['splits']:
        for s in cfg['seeds']:
            d=CK/f'{f}_seed{s}';done=base.read_json(d/'COMPLETE.json')
            assert done['protocol_sha256']==base.digest(OUT/'protocol.json')
            for n,h in done['checkpoint_sha256'].items():assert h==base.digest(d/(n+'.pt'))
    results={};curves={};diagnostics={}
    for fold,split in cfg['splits'].items():
        records=base.Records(split['test'])
        for seed in cfg['seeds']:
            d=CK/f'{fold}_seed{seed}';old=OLDCK/d.name
            stats=dict(np.load(d/'normalizers.npz'));limits=base.read_json(d/'limits.json')['limits'];encode(records,old)
            for name in VARIANTS:
                data=oracle_copy(records) if VARIANTS[name][2] else records
                ds=base.WindowData(data,100,4);m=make(name,stats,old,limits)
                saved=torch.load(d/(name+'.pt'),weights_only=False);m.load_state_dict(saved['model'])
                monitor=Monitor();hook=m.register_forward_pre_hook(monitor.hook,with_kwargs=True)
                errors=base.evaluate_model(m,ds);hook.remove()
                key=f'{fold}/seed{seed}/{name}';results[key]=base.metrics(errors);curves[key]=errors;diagnostics[key]=monitor.result()
                print(key,results[key],flush=True)
            base.write_json(d/'evaluation_windows.json',[dict(file=records.records[e]['name'],t=int(t)) for e,t in ds.idx])
    np.savez_compressed(OUT/'curves.npz',**curves)
    base.write_json(OUT/'results.json',dict(protocol_sha256=base.digest(OUT/'protocol.json'),results=results,diagnostics=diagnostics))

def audit(cfg):
    verify(cfg)
    # Verify parent data too; avoids inventing a pre-run fingerprint after evaluation.
    old=base.read_json(OLD/'protocol.json')
    for n,v in old['files'].items():assert base.digest(base.DATA/n)==v['sha256']
    for n,h in base.read_json(OLD/'raw_data_fingerprints.json').items():assert base.digest(base.RAW/n)==h
    results=base.read_json(OUT/'results.json');arrays=np.load(OUT/'curves.npz')
    assert results['protocol_sha256']==base.digest(OUT/'protocol.json')
    assert len(results['results'])==24 and set(arrays.files)==set(results['results'])
    for f,split in cfg['splits'].items():
        windows=None
        for seed in cfg['seeds']:
            d=CK/f'{f}_seed{seed}';done=base.read_json(d/'COMPLETE.json')
            assert done['protocol_sha256']==results['protocol_sha256']
            ws=base.read_json(d/'evaluation_windows.json')
            assert all(w['file'] in split['test'] for w in ws)
            if windows is not None:assert windows==ws
            windows=ws
            assert ws==base.read_json(OLDCK/d.name/'test_diagnostics.json')['windows']
            for name in VARIANTS:
                assert done['checkpoint_sha256'][name]==base.digest(d/(name+'.pt'))
                h=base.read_json(d/(name+'_history.json'));b=min((x for x in h if x['val_E100'] is not None),key=lambda x:x['val_E100'])
                c=torch.load(d/(name+'.pt'),weights_only=False)
                assert len(h)==40 and c['epoch']==b['epoch'] and c['val_E100']==b['val_E100']
                key=f'{f}/seed{seed}/{name}';e=arrays[key];assert e.shape==(len(ws),101)
                assert np.allclose(e[:,0],0,atol=1e-7)
                assert base.metrics(e)==results['results'][key]
    base.write_json(OUT/'audit.json',dict(status='PASS',rows=24,checks=['source/data/checkpoint hashes','40 epochs, validation-only selection','identical parent/follow-up windows','all saved metrics recomputed','zero initial position error']))

def report(cfg):
    res=base.read_json(OUT/'results.json')['results'];old=base.read_json(OLD/'test_results.json')['results']
    lines=['# 条件化感知器下的匹配动力学对照','','探索性诊断：沿用两条已研究过的记录、划分和六个冻结感知器。每个动力学模型从零训练。不能视为新数据确认性测试。',
           '真实状态初始化、已记录未来动作；100 步约 4.55 s。均值 ± 三个种子样本标准差，单位 m。',
           '全部道路模型使用同一个条件化感知器；guarded 为受限 Frenet 更新，geometry_midpoint 使用形状导出的曲率和参数速度，Cartesian 使用相同道路输入。纯运动学及无道路 Cartesian 的固定参考不依赖感知器。','']
    for fold in cfg['splits']:
        lines += [f'## {fold}','','| 模型 | E25 | E50 | E100 | ADE |','|---|---:|---:|---:|---:|']
        for name in list(VARIANTS)+['previous_kinematic','previous_cartesian_no_road']:
            rows=[old[f'{fold}/seed{s}/{name.removeprefix("previous_")}/normal'] if name.startswith('previous_') else res[f'{fold}/seed{s}/{name}'] for s in cfg['seeds']]
            cells=[]
            for metric in ['E25','E50','E100','ADE']:
                v=[x[metric] for x in rows];cells.append('nonfinite' if None in v else f'{np.mean(v):.3f} ± {np.std(v,ddof=1):.3f}')
            lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines+=['','## 训练稳定性','','invalid 表示验证 rollout 状态的原始 1-dκ≤0 比例。表中为最后 epoch；最后 checkpoint 不用于主测试。梯度为整个训练过程、裁剪前的最大范数。','',
            '| 方向/种子 | 模型 | 最优 epoch | 最优 val E100 | 最后 val E100 | 最大梯度 | 最后 invalid |','|---|---|---:|---:|---:|---:|---:|']
    for f in cfg['splits']:
        for s in cfg['seeds']:
            for name in VARIANTS:
                h=base.read_json(CK/f'{f}_seed{s}'/(name+'_history.json'));b=min((x for x in h if x['val_E100'] is not None),key=lambda x:x['val_E100'])
                lines.append(f'| {f}/{s} | {name} | {b["epoch"]} | {b["val_E100"]:.3f} | {h[-1]["val_E100"]:.3f} | {max(x["max_preclip_gradient"] for x in h):.1f} | {h[-1]["validation_state"]["invalid_chart_fraction"]:.2%} |')
    lines+=['','## 可追溯性','','协议与实现指纹见 protocol.json；逐种子指标和状态诊断见 results.json；逐窗口曲线见 curves.npz；审计见 audit.json。',
            'guarded_shape 从训练开始将独立曲率输入清零；geometry_midpoint 使用形状导出的曲率；Cartesian 没有 Frenet 分母，其 samples=0 的 invalid 统计不适用。',
            '本轮只重训动力学，视觉编码器及其监督固定。因此 shape 消融没有隔离“感知器训练时使用曲率标签”这个辅助监督因素。',
            '逐步增量有限不代表位置或长期误差有界。坐标域外的正分母扩展仅为数值措施，必须结合 invalid 比例解释。','',
            '模型名称：GPT-6（Codex）；当前会话未提供更细型号标识。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def jobs(cfg):
    queue=[(f,s) for f in cfg['splits'] for s in cfg['seeds'] if not (CK/f'{f}_seed{s}'/'COMPLETE.json').exists()];active=[]
    while queue or active:
        while queue and len(active)<3:
            f,s=queue.pop(0);fh=(OUT/f'{f}_seed{s}.log').open('a',encoding='utf-8')
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'train','--outer',f,'--seed',str(s)],cwd=ROOT,stdout=fh,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active.append((p,fh,f,s));print('START',f,s,p.pid,flush=True)
        for item in active[:]:
            p,fh,f,s=item
            if p.poll() is not None:
                fh.close();active.remove(item);print('DONE',f,s,p.returncode,flush=True)
                if p.returncode:
                    for q,log,_,_ in active:q.terminate();q.wait();log.close()
                    raise RuntimeError('Training failed; inspect logs')
        time.sleep(2)
    evaluate(cfg);audit(cfg);report(cfg)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','jobs','evaluate','audit','report'])
    parser.add_argument('--outer');parser.add_argument('--seed',type=int);args=parser.parse_args();torch.set_num_threads(4);base.DEVICE='cpu'
    if args.stage=='prepare':prepare();return
    cfg=base.read_json(OUT/'protocol.json')
    if args.stage=='train':train(cfg,args.outer,args.seed)
    elif args.stage=='jobs':jobs(cfg)
    else:globals()[args.stage](cfg)

if __name__=='__main__':main()
