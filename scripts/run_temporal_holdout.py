"""Exploratory, purged chronological holdout within both existing records.

This tests same-track forecasting with both recorded operating regimes represented
in development. It does not replace source-held-out stress tests or create fresh
independent data. Freeze before fitting; retain all variants and three seeds.
"""
from pathlib import Path
import argparse,sys,math,json,subprocess,time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_holdout as b
import run_conditioned_dynamics as c
from models.conditioned_road_encoder import ConditionedRoadEncoder
from baselines.controlled_dynamics import rollout
OUT=b.ROOT/'reports/temporal_holdout';CK=b.ROOT/'checkpoints/temporal_holdout'
NAMES=['kinematic','cartesian_no_road','cartesian_road','guarded_full','guarded_shape','geometry_midpoint']
OLD_MAKE=c.make

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);CK.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise RuntimeError('Protocol exists')
    parent=b.read_json(b.OUT/'protocol.json');groups={}
    for name in parent['files']:
        match=b.PAT.fullmatch(name);groups.setdefault(match[1],[]).append((int(match[2]),int(match[3]),name))
    merged={k:[] for k in ['train','validation','test','purged']};detail={}
    for source,items in sorted(groups.items()):
        items.sort();n=len(items);nv=nt=math.ceil(.2*n);tr=items[:-nv-nt];va=items[-nv-nt:-nt];te=items[-nt:]
        kept_tr=[x for x in tr if x[1]+115<=va[0][0]];kept_va=[x for x in va if x[1]+115<=te[0][0]]
        excluded=[x for x in tr+va if x not in kept_tr+kept_va]
        parts=dict(train=kept_tr,validation=kept_va,test=te,purged=excluded)
        detail[source]={k:[x[2] for x in v] for k,v in parts.items()}
        for key in merged:merged[key]+=[x[2] for x in parts[key]]
    paths=[Path(__file__),b.ROOT/'scripts/run_controlled_holdout.py',b.ROOT/'scripts/run_conditioned_dynamics.py',
           b.ROOT/'src/models/conditioned_road_encoder.py',b.ROOT/'src/models/road_perception.py',b.ROOT/'src/models/encoder_lane.py',
           b.ROOT/'src/models/frenet_dynamics.py',b.ROOT/'src/baselines/controlled_dynamics.py',b.ROOT/'src/baselines/frenet_diagnostic.py',
           b.ROOT/'src/baselines/geometric_frenet.py',b.ROOT/'src/baselines/consistent_road_geometry.py',b.ROOT/'src/frenet_track.py']
    cfg=dict(status='exploratory protocol after prior source-held-out analyses; not independent confirmation',
        rationale='one source has constant throttle and a lower speed distribution; distinguish temporal within-track performance from source/regime transfer',
        split_rule='per source, ordered segments: first remaining ~60% train, next ceil(20%) validation, last ceil(20%) test; purge end segments to leave >=115 frames between adjacent partitions',
        splits={'temporal':merged},source_partitions=detail,seeds=[0,1,2],variants=NAMES,
        perception_epochs=25,dynamics_epochs=40,train_horizon=32,test_horizon=100,batch=128,perception_batch=64,
        lr=.001,kinematic_lr=.01,primary_metric='pooled-window E100; also report both original sources separately, all seeds retained',
        checkpoint_selection='perception: validation normalized supervised MSE; dynamics: pooled validation E100',
        primary_model='guarded_full; matched controls and ablations all retained; no selection of the best family by final errors',
        frozen_sources={str(p.relative_to(b.ROOT)):b.digest(p) for p in paths},parent_sha256=b.digest(b.OUT/'protocol.json'),
        raw_fingerprints=b.read_json(b.OUT/'raw_data_fingerprints.json'),track_sha256=b.digest(b.DATA/'_meta/track.npz'))
    b.write_json(OUT/'protocol.json',cfg);print({s:{k:len(v) for k,v in p.items()} for s,p in detail.items()},flush=True)

def verify(cfg):
    assert cfg['parent_sha256']==b.digest(b.OUT/'protocol.json')
    for p,h in cfg['frozen_sources'].items():assert b.digest(b.ROOT/p)==h,p
    assert b.digest(b.DATA/'_meta/track.npz')==cfg['track_sha256']

def perception(cfg,seed):
    verify(cfg);d=CK/f'temporal_seed{seed}';d.mkdir(parents=True,exist_ok=True)
    if (d/'PERCEPTION_COMPLETE.json').exists():return
    if (OUT/'results.json').exists():raise RuntimeError('Evaluation already completed')
    split=cfg['splits']['temporal'];tr=b.Records(split['train']);va=b.Records(split['validation']);b.seed_all(seed);b.DEVICE=b.VISION_DEVICE
    stats=b.fit_stats(tr,d);td=b.PerceptionData(tr,2);vd=b.PerceptionData(va,2)
    mean=np.stack([td[i][1].numpy() for i in range(len(td))]).mean(0);scale=stats['perception_scale'];scale_t=torch.tensor(scale,device=b.DEVICE)
    m=ConditionedRoadEncoder(mean,scale).to(b.DEVICE);opt=torch.optim.Adam(m.parameters(),lr=.001);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,25)
    best=float('inf');history=[]
    for epoch in range(25):
        m.train();total=0;n=0
        for x,y in b.loader(td,64,True):
            x=x.to(b.DEVICE);y=y.to(b.DEVICE);k,p=m.forward_both(x);loss=((torch.cat([k,p.flatten(1)],-1)-y)/scale_t).square().mean()
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),5));assert np.isfinite(gn);opt.step();total+=float(loss.detach())*len(x);n+=len(x)
        m.eval();value=0;nv=0
        with torch.no_grad():
            for x,y in b.loader(vd,64):
                x=x.to(b.DEVICE);y=y.to(b.DEVICE);k,p=m.forward_both(x);value+=float(((torch.cat([k,p.flatten(1)],-1)-y)/scale_t).square().mean())*len(x);nv+=len(x)
        value/=nv;history.append(dict(epoch=epoch+1,train_loss=total/n,val_loss=value))
        if value<best:
            best=value;torch.save(dict(model=m.state_dict(),mean=mean,scale=scale,epoch=epoch+1,val_loss=value),d/'perception.pt')
        sch.step();b.write_json(d/'perception_history.json',history);print(d.name,'perception',epoch+1,value,flush=True)
    m.load_state_dict(torch.load(d/'perception.pt',map_location=b.DEVICE,weights_only=False)['model']);b.encode(m,tr);stats=b.context_stats(tr,stats)
    np.savez(d/'normalizers.npz',**stats)
    delta=np.concatenate([np.abs(np.diff(r['z'][:,3:5],axis=0)) for r in tr.records]);limits=np.maximum(.01,np.quantile(delta,.995,axis=0)).tolist()
    b.write_json(d/'limits.json',dict(limits=limits,source='train only'))
    b.write_json(d/'PERCEPTION_COMPLETE.json',dict(protocol_sha256=b.digest(OUT/'protocol.json'),checkpoint_sha256=b.digest(d/'perception.pt'),normalizers_sha256=b.digest(d/'normalizers.npz')))

def make(name,stats,d,limits):
    if name in ['kinematic','cartesian_no_road','cartesian_road']:return b.make_model(name,stats,d)
    return OLD_MAKE(name,stats,d,limits)

def configure(cfg):
    c.OUT=OUT;c.CK=CK;c.OLDCK=CK;c.ENC=CK;c.VARIANTS={n:(True,'shape' if n=='guarded_shape' else 'full',False) for n in NAMES};c.make=make;c.verify=verify

def fit(cfg,seed):
    verify(cfg);configure(cfg);d=CK/f'temporal_seed{seed}'
    if (d/'COMPLETE.json').exists():return
    if (OUT/'results.json').exists():raise RuntimeError('Evaluation already completed')
    done=b.read_json(d/'PERCEPTION_COMPLETE.json');assert done['protocol_sha256']==b.digest(OUT/'protocol.json');assert done['checkpoint_sha256']==b.digest(d/'perception.pt');assert done['normalizers_sha256']==b.digest(d/'normalizers.npz')
    split=cfg['splits']['temporal'];tr=b.Records(split['train']);va=b.Records(split['validation']);c.encode(tr,d);c.encode(va,d);b.DEVICE='cpu'
    stats=dict(np.load(d/'normalizers.npz'));limits=b.read_json(d/'limits.json')['limits']
    for name in NAMES:
        if (d/(name+'_history.json')).exists():continue
        local=dict(cfg,lr=cfg['kinematic_lr'] if name=='kinematic' else cfg['lr'])
        c.fit(name,tr,va,stats,d,d,local,limits,seed)
    b.write_json(d/'COMPLETE.json',dict(protocol_sha256=b.digest(OUT/'protocol.json'),checkpoint_sha256={n:b.digest(d/(n+'.pt')) for n in NAMES+['perception']}))
def evaluate(cfg):
    configure(cfg);c.evaluate(cfg)

def audit(cfg):
    verify(cfg);parent=b.read_json(b.OUT/'protocol.json')
    for n,v in parent['files'].items():assert b.digest(b.DATA/n)==v['sha256']
    for n,h in cfg['raw_fingerprints'].items():assert b.digest(b.RAW/n)==h
    for source,p in cfg['source_partitions'].items():
        groups=[set(p[k]) for k in ['train','validation','test','purged']]
        assert all(not groups[i]&groups[j] for i in range(4) for j in range(i))
        for a,z in [('train','validation'),('validation','test')]:
            assert max(int(b.PAT.fullmatch(n)[3]) for n in p[a])+115<=min(int(b.PAT.fullmatch(n)[2]) for n in p[z])
    data=b.read_json(OUT/'results.json');curves=np.load(OUT/'curves.npz');assert len(data['results'])==18
    assert data['protocol_sha256']==b.digest(OUT/'protocol.json')
    expected=None;per_source={}
    for seed in cfg['seeds']:
        d=CK/f'temporal_seed{seed}';ws=b.read_json(d/'evaluation_windows.json')
        if expected is not None:assert expected==ws
        expected=ws;assert all(x['file'] in cfg['splits']['temporal']['test'] for x in ws)
        ph=b.read_json(d/'perception_history.json');pc=torch.load(d/'perception.pt',weights_only=False);pb=min(ph,key=lambda x:x['val_loss']);assert len(ph)==25 and pc['epoch']==pb['epoch'] and pc['val_loss']==pb['val_loss']
        done=b.read_json(d/'COMPLETE.json');assert done['protocol_sha256']==data['protocol_sha256']
        for n,h in done['checkpoint_sha256'].items():assert b.digest(d/(n+'.pt'))==h
        for name in NAMES:
            hist=b.read_json(d/(name+'_history.json'));best=min((x for x in hist if x['val_E100'] is not None),key=lambda x:x['val_E100']);saved=torch.load(d/(name+'.pt'),weights_only=False)
            assert len(hist)==40 and saved['epoch']==best['epoch'] and saved['val_E100']==best['val_E100']
            key=f'temporal/seed{seed}/{name}';e=curves[key];assert e.shape==(len(ws),101) and np.allclose(e[:,0],0,atol=1e-7);assert b.metrics(e)==data['results'][key]
            for source in cfg['source_partitions']:
                mask=np.array([b.PAT.fullmatch(w['file'])[1]==source for w in ws]);per_source[f'{source}/seed{seed}/{name}']=b.metrics(e[mask])
    b.write_json(OUT/'per_source_results.json',dict(results=per_source))
    b.write_json(OUT/'audit.json',dict(status='PASS',rows=18,checks=['raw and prepared data fingerprints','temporal partition disjointness and >=115-frame gaps','25/40-epoch budgets and validation-only selection','all seed/checkpoint hashes','identical windows across models and seeds','metrics recomputed for pooled and source-specific windows','zero initial errors']))

def report(cfg):
    res=b.read_json(OUT/'results.json')['results'];sub=b.read_json(OUT/'per_source_results.json')['results']
    lines=['# 两来源内部的时间留出对照','','这是新增的探索性、同赛道／已覆盖工况预测实验。两条记录都曾参与历史开发，不是新的独立数据验证。跨来源留出压力测试仍保留，不能用本表替代或隐藏其结果。',
           '所有方法共享划分、窗口、初始化和已记录未来动作。CNN 均从零训练；感知 25 epochs，动力学 40 epochs，验证选模。主模型预先固定为 guarded_full，主指标为全部窗口平均 E100；同时公开分来源结果。','']
    for source in ['temporal']+list(cfg['source_partitions']):
        source_rows=res if source=='temporal' else sub
        lines += [f'## {source}','','| 模型 | E25 | E50 | E100 | ADE |','|---|---:|---:|---:|---:|']
        for name in NAMES:
            cells=[]
            for metric in ['E25','E50','E100','ADE']:
                v=[source_rows[f'{source}/seed{s}/{name}'][metric] for s in cfg['seeds']]
                cells.append('nonfinite' if None in v else f'{np.mean(v):.3f} ± {np.std(v,ddof=1):.3f}')
            lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines+=['','均值 ± 三个训练种子样本标准差，不是独立采集 session 的置信区间。训练片段覆盖了两个原始记录中的工况，不能据此声称泛化到未见速度、车辆或赛道。','',
            '协议见 protocol.json，完整指标与逐窗口曲线见 results.json / curves.npz；全部原始来源分项见 per_source_results.json；审计见 audit.json。','',
            '模型名称：GPT-6（Codex）；当前会话未提供更细型号标识。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def jobs(cfg,stage):
    marker='PERCEPTION_COMPLETE.json' if stage=='perception' else 'COMPLETE.json'
    queue=[s for s in cfg['seeds'] if not (CK/f'temporal_seed{s}'/marker).exists()];active=[];capacity=2 if stage=='perception' else 3
    while queue or active:
        while queue and len(active)<capacity:
            seed=queue.pop(0);fh=(OUT/f'{stage}_seed{seed}.log').open('a',encoding='utf-8')
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),stage,'--seed',str(seed)],cwd=b.ROOT,stdout=fh,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));active.append((p,fh,seed));print('START',stage,seed,p.pid,flush=True)
        for item in active[:]:
            p,fh,seed=item
            if p.poll() is not None:
                fh.close();active.remove(item);print('DONE',stage,seed,p.returncode,flush=True)
                if p.returncode:
                    for q,f,_ in active:q.terminate();q.wait();f.close()
                    raise RuntimeError('Job failed; inspect log')
        time.sleep(2)
    if stage=='fit':evaluate(cfg);audit(cfg);report(cfg)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','perception','perceptions','fit','jobs','evaluate','audit','report']);parser.add_argument('--seed',type=int);args=parser.parse_args();torch.set_num_threads(4);b.DEVICE='cpu'
    if args.stage=='prepare':prepare();return
    cfg=b.read_json(OUT/'protocol.json')
    if args.stage in ['perception','fit']:globals()[args.stage](cfg,args.seed)
    elif args.stage=='perceptions':jobs(cfg,'perception')
    elif args.stage=='jobs':jobs(cfg,'fit')
    else:globals()[args.stage](cfg)
if __name__=='__main__':main()
