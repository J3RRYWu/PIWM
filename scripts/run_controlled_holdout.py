"""Frozen source-held-out experiments; run from piwm/ with the project venv.

Stages: prepare -> train -> evaluate -> report. Evaluation is refused until all
predeclared checkpoints exist. A completed test result cannot be overwritten.
No legacy checkpoint or all-data normalization is reused.
"""
from pathlib import Path
import argparse, hashlib, json, math, re, sys, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from models.frenet_dynamics import FrenetDynamics
from models.road_perception import RoadContextEncoder
from baselines.controlled_dynamics import rollout, CartesianResidual, CalibratedKinematic, constant_motion, wrap
from frenet_track import local_road_points

DATA=ROOT.parent/'Data_Donkeycar_frenet'
RAW=ROOT.parent/'Data_Donkeycar'
OUT=ROOT/'reports'/'controlled_holdout'
CK=ROOT/'checkpoints'/'controlled_holdout'
VISION_DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
DEVICE=VISION_DEVICE
OFFSETS=np.arange(10,dtype=np.float32)*0.5
MODELS=('kinematic','cartesian_no_road','cartesian_road','frenet')
PAT=re.compile(r'(.+)_seg\d+_(\d+)_(\d+)\.npz$')


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def safe(value):
        if isinstance(value,dict): return {k:safe(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [safe(v) for v in value]
        if isinstance(value,float) and not math.isfinite(value): return None
        return value
    path.write_text(json.dumps(safe(obj),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def seed_all(seed):
    np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False

def source_split(names,test_source,purge=115):
    groups={}
    for name in names:
        m=PAT.fullmatch(name)
        if m is None: raise ValueError(name)
        groups.setdefault(m[1],[]).append((int(m[2]),int(m[3]),name))
    if len(groups)!=2: raise ValueError('Protocol requires exactly two raw sources')
    development=[s for s in groups if s!=test_source][0]
    ordered=sorted(groups[development]);nv=max(1,math.ceil(.2*len(ordered)))
    val=ordered[-nv:];cut=val[0][0]
    train=[x for x in ordered[:-nv] if x[1]+purge<=cut]
    omitted=[x for x in ordered[:-nv] if x not in train]
    return dict(train=[x[2] for x in train],validation=[x[2] for x in val],
                test=[x[2] for x in sorted(groups[test_source])],purged=[x[2] for x in omitted],
                test_source=test_source,development_source=development)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);CK.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists(): raise RuntimeError('Protocol already frozen; refusing overwrite')
    names=sorted(p.name for p in DATA.glob('*.npz'))
    sources=sorted({PAT.fullmatch(n)[1] for n in names})
    manifest={n:dict(bytes=(DATA/n).stat().st_size,sha256=digest(DATA/n)) for n in names}
    protocol=dict(version=1,seeds=[0,1,2],context=15,train_horizon=32,test_horizon=100,
        train_stride=4,evaluation_stride=4,perception_stride=2,perception_epochs=25,
        dynamics_epochs=40,batch=128,perception_batch=64,purge_frames=115,
        models=list(MODELS),preview_modes=['normal','zero','train_mean','permuted'],
        primary_metric='mean endpoint Euclidean error at 100 steps, metres; raw pose truth',
        secondary_metrics=['E25','E50','ADE1:100','per-source and per-seed results'],
        initialization='ground-truth initial v, causal omega, d, heading error; no future samples',
        extrapolation='curvature hold-last; map-free shape tangent extension (legacy hold-last diagnostic)',
        selection='perception: validation normalized supervised MSE; dynamics: validation E100 only',
        statistics='all normalizers fit train only; fixed surveyed map only creates labels and initial road-relative state',
        limitations=['Retrospective reanalysis: both raw records were previously explored in model development.',
          'Two raw files on the same track; no claim of independent sessions or unseen-road generalization.',
          'Ground-truth initial state and recorded future actions: offline action-conditioned prediction.',
          'New matched controls and matched training objective, not a reproduction of published baseline implementations.'],
        splits={f'outer{i}':source_split(names,s) for i,s in enumerate(sources)},files=manifest)
    write_json(path,protocol)
    write_json(OUT/'source_snapshot.json',{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'src/baselines/controlled_dynamics.py',ROOT/'src/models/frenet_dynamics.py',ROOT/'src/models/road_perception.py']})
    print(json.dumps({k:{r:len(v[r]) for r in ['train','validation','test','purged']} for k,v in protocol['splits'].items()},indent=2),flush=True)


class Records:
    def __init__(self,names):
        tr=np.load(DATA/'_meta'/'track.npz')
        self.track={k:tr[k] for k in tr.files};self.records=[]
        self.L=float(tr['total_len']);ds=float(tr['grid_ds']);cen=tr['centers'];hd=tr['heading'];M=len(cen)
        raw_cache={}
        for name in names:
            m=PAT.fullmatch(name);source=m[1];a,b=int(m[2]),int(m[3])
            if source not in raw_cache:
                with np.load(RAW/(source+'.npz'),allow_pickle=False) as raw:
                    raw_cache[source]=raw['state'].copy()
            raw=raw_cache[source][a:b].copy()
            with np.load(DATA/name) as f:
                z=f['state'].copy();imgs=f['imgs'].copy();actions=f['action'].copy();kp=f['kappa_profile'].copy()
            assert len(raw)==len(z),name
            # Recompute initialization/targets without centered temporal smoothing.
            yaw=np.unwrap(np.deg2rad(raw[:,2])).astype(np.float32)
            dy=np.r_[0,np.diff(yaw)]*22
            omega=np.array([dy[max(1,t-4):t+1].mean() if t else 0 for t in range(len(dy))],np.float32)
            idx=(np.mod(z[:,0],self.L)/ds).astype(int)%M
            normals=np.stack([-np.sin(hd[idx]),np.cos(hd[idx])],-1)
            z[:,1]=((raw[:,:2]-cen[idx])*normals).sum(-1)
            z[:,2]=np.arctan2(np.sin(yaw-hd[idx]),np.cos(yaw-hd[idx]))
            z[:,3]=raw[:,3];z[:,4]=omega
            shape=local_road_points(cen,hd,self.L,ds,z[:,0],OFFSETS)
            self.records.append(dict(name=name,z=z,xy=raw[:,:2],yaw=yaw,imgs=imgs,
                                     actions=actions,kappa=kp,shape=shape))

    def targets(self,ep,t,k):
        r=self.records[ep];xy=r['xy'][t:t+k+1]-r['xy'][t];th=r['yaw'][t]
        loc=np.stack([xy[:,0]*np.cos(th)+xy[:,1]*np.sin(th),-xy[:,0]*np.sin(th)+xy[:,1]*np.cos(th)],-1)
        yaw=r['yaw'][t:t+k+1]-th
        return np.concatenate([loc,yaw[:,None],r['z'][t:t+k+1,3:]],-1).astype(np.float32)


class PerceptionData(Dataset):
    def __init__(self,records,stride):
        self.r=records.records
        self.idx=[(i,t) for i,r in enumerate(self.r) for t in range(14,len(r['z']),stride)]
    def __len__(self):return len(self.idx)
    def __getitem__(self,i):
        e,t=self.idx[i];r=self.r[e]
        shape=r['shape'][t].copy();shape[:,0]-=OFFSETS
        return (torch.from_numpy(r['imgs'][t-14:t+1]).float(),
                torch.from_numpy(np.r_[r['kappa'][t],shape.flatten()]).float())


class WindowData(Dataset):
    def __init__(self,records,horizon,stride):
        self.data=records;self.k=horizon
        self.idx=[(i,t) for i,r in enumerate(records.records) for t in range(14,len(r['z'])-horizon,stride)]
        if not self.idx:raise ValueError('No valid rollout windows')
    def __len__(self):return len(self.idx)
    def __getitem__(self,i):
        e,t=self.idx[i];r=self.data.records[e]
        return (torch.from_numpy(r['z'][t]).float(),torch.from_numpy(r['actions'][t:t+self.k]).float(),
                torch.from_numpy(self.data.targets(e,t,self.k)),torch.from_numpy(r['pred_kappa'][t]).float(),
                torch.from_numpy(r['pred_shape'][t]).float())


def loader(ds,batch,shuffle=False):
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=0,pin_memory=DEVICE=='cuda')

def to_device(batch):return [x.to(DEVICE,non_blocking=True) for x in batch]

def fit_stats(train,directory):
    Z=np.concatenate([r['z'] for r in train.records]);kp=np.concatenate([r['kappa'] for r in train.records])
    means=Z.mean(0);std=Z.std(0).clip(.01);means[0]=0;std[0]=1
    np.savez(directory/'stats.npz',state_mean=means,state_std=std,kappa_mean=kp.mean(),kappa_std=max(kp.std(),.01))
    targets=[]
    for r in train.records:
        s=r['shape'].copy();s[:,:,0]-=OFFSETS
        targets.append(np.concatenate([r['kappa'],s.reshape(len(s),-1)],-1))
    a=np.concatenate(targets)
    return dict(state_mean=means,state_std=std,perception_scale=a.std(0).clip(.05),
                mean_kappa=kp.mean(0),mean_shape=np.concatenate([r['shape'] for r in train.records]).mean(0))


def encode(model,records):
    model.eval()
    with torch.no_grad():
        for r in records.records:
            n=len(r['z']);r['pred_kappa']=np.zeros((n,10),np.float32);r['pred_shape']=np.zeros((n,10,2),np.float32)
            for lo in range(14,n,128):
                ts=list(range(lo,min(lo+128,n)))
                x=torch.tensor(np.stack([r['imgs'][t-14:t+1] for t in ts]),device=DEVICE)
                kap,shp=model.forward_both(x);shp[:,:,0]+=torch.tensor(OFFSETS,device=DEVICE)
                r['pred_kappa'][ts]=kap.cpu().numpy();r['pred_shape'][ts]=shp.cpu().numpy()


def train_perception(train,val,stats,cfg,directory):
    model=RoadContextEncoder(predict_shape=True).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['perception_epochs'])
    scale=torch.tensor(stats['perception_scale'],device=DEVICE)
    tr=loader(PerceptionData(train,cfg['perception_stride']),cfg['perception_batch'],True)
    va=loader(PerceptionData(val,cfg['perception_stride']),cfg['perception_batch'])
    best=float('inf');history=[]
    for ep in range(cfg['perception_epochs']):
        model.train();total=0;n=0
        for batch in tr:
            x,y=to_device(batch);a,b=model.forward_both(x);pred=torch.cat([a,b.flatten(1)],-1)
            loss=((pred-y)/scale).square().mean();opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=loss.item()*len(x);n+=len(x)
        model.eval();score=0;nv=0
        with torch.no_grad():
            for batch in va:
                x,y=to_device(batch);a,b=model.forward_both(x)
                score+=(((torch.cat([a,b.flatten(1)],-1)-y)/scale).square().mean()).item()*len(x);nv+=len(x)
        score/=nv;history.append(dict(epoch=ep+1,train_loss=total/n,val_loss=score))
        if score<best:
            best=score;torch.save(dict(model=model.state_dict(),epoch=ep+1,val_loss=score),directory/'perception.pt')
        schedule.step()
        print(f'{directory.name} perception {ep+1}/{cfg["perception_epochs"]} train={total/n:.4f} val={score:.4f}',flush=True)
    write_json(directory/'perception_history.json',history)
    model.load_state_dict(torch.load(directory/'perception.pt',map_location=DEVICE,weights_only=False)['model'])
    return model


def context_stats(train,stats):
    c=np.concatenate([np.concatenate([r['z'][14:,1:3],r['pred_kappa'][14:],r['pred_shape'][14:].reshape(-1,20)],-1) for r in train.records])
    stats['context_mean']=c.mean(0);stats['context_std']=c.std(0).clip(.05)
    return stats


def make_model(name,stats,directory):
    if name=='frenet':
        m=FrenetDynamics(DATA/'_meta'/'track.npz',directory/'stats.npz')
        # All neural controls start at their analytic no-residual update.
        for h in [m.dv_net,m.dom_net,m.res_net]:
            torch.nn.init.zeros_(h.net[-1].weight);torch.nn.init.zeros_(h.net[-1].bias)
    elif name=='kinematic':m=CalibratedKinematic()
    else:m=CartesianResidual(stats['state_mean'],stats['state_std'],stats['context_mean'],stats['context_std'],road=name=='cartesian_road')
    return m.to(DEVICE)


def loss_function(pred,truth,stats):
    err=pred[:,1:]-truth[:,1:]
    err=torch.cat([err[:,:,:2],wrap(err[:,:,2:3]),err[:,:,3:]],-1)
    scales=torch.tensor([.5,.5,.5,stats['state_std'][3],stats['state_std'][4]],device=DEVICE)
    return (err/scales).square().mean()


@torch.no_grad()
def evaluate_model(model,ds,batch=128,mode='normal',stats=None,hold_last=False):
    model.eval();errors=[]
    # Deterministic cross-window permutation independent of targets and metrics.
    donor=np.random.default_rng(714).permutation(len(ds))
    start=0
    for items in loader(ds,batch):
        z,a,truth,kp,shp=to_device(items)
        if mode=='zero':
            kp=torch.zeros_like(kp);shp=torch.stack([torch.tensor(OFFSETS,device=DEVICE),torch.zeros(10,device=DEVICE)],-1)[None].expand(len(z),-1,-1)
        elif mode=='train_mean':
            kp=torch.tensor(stats['mean_kappa'],device=DEVICE)[None].expand(len(z),-1)
            shp=torch.tensor(stats['mean_shape'],device=DEVICE)[None].expand(len(z),-1,-1)
        elif mode=='permuted':
            chosen=[ds[int(i)] for i in donor[start:start+len(z)]]
            kp=torch.stack([x[3] for x in chosen]).to(DEVICE);shp=torch.stack([x[4] for x in chosen]).to(DEVICE)
        elif mode in ('kappa_zero','kappa_mean','kappa_permuted'):
            if mode=='kappa_zero':kp=torch.zeros_like(kp)
            elif mode=='kappa_mean':kp=torch.tensor(stats['mean_kappa'],device=DEVICE)[None].expand(len(z),-1)
            else:kp=torch.stack([ds[int(i)][3] for i in donor[start:start+len(z)]]).to(DEVICE)
        pred=rollout(model,z,a,kp,shp,hold_last)
        err=torch.linalg.vector_norm(pred[:,:,:2]-truth[:,:,:2],dim=-1)
        errors.append(err.cpu().numpy());start+=len(z)
    return np.concatenate(errors)


def train_dynamics(name,train,val,stats,cfg,directory):
    model=make_model(name,stats,directory)
    tr=WindowData(train,cfg['train_horizon'],cfg['train_stride']);va=WindowData(val,100,cfg['evaluation_stride'])
    opt=torch.optim.Adam(model.parameters(),lr=.01 if name=='kinematic' else 1e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['dynamics_epochs'])
    best=float('inf');hist=[]
    for ep in range(cfg['dynamics_epochs']):
        model.train();tot=0;n=0
        for items in loader(tr,cfg['batch'],True):
            z,a,truth,kp,shp=to_device(items)
            loss=loss_function(rollout(model,z,a,kp,shp),truth,stats)
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite training loss: {name}, epoch {ep+1}')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            tot+=loss.item()*len(z);n+=len(z)
        errs=evaluate_model(model,va);score=float(errs[:,100].mean())
        hist.append(dict(epoch=ep+1,train_loss=tot/n,val_E100=score))
        if np.isfinite(score) and score<best:
            best=score;torch.save(dict(model=model.state_dict(),epoch=ep+1,val_E100=score,model_name=name,
                                      parameters=sum(p.numel() for p in model.parameters())),directory/(name+'.pt'))
        sch.step()
        print(f'{directory.name} {name} {ep+1}/{cfg["dynamics_epochs"]} loss={tot/n:.4f} val100={score:.4f}',flush=True)
    write_json(directory/(name+'_history.json'),hist)
    if not (directory/(name+'.pt')).exists():raise RuntimeError('No finite validation checkpoint')


def train(cfg):
    global DEVICE
    if (OUT/'test_results.json').exists():raise RuntimeError('Test already opened; frozen run cannot be retrained')
    for fold,split in cfg['splits'].items():
        # No test records are loaded, normalized or encoded in this stage.
        tr=Records(split['train']);va=Records(split['validation'])
        for seed in cfg['seeds']:
            DEVICE=VISION_DEVICE
            directory=CK/f'{fold}_seed{seed}';directory.mkdir(parents=True,exist_ok=True)
            done=directory/'TRAINING_COMPLETE.json'
            if done.exists():
                print(f'Skipping completed {directory.name}',flush=True);continue
            seed_all(seed)
            stats=fit_stats(tr,directory)
            if (directory/'perception.pt').exists() and (directory/'perception_history.json').exists():
                perception=RoadContextEncoder(predict_shape=True).to(DEVICE)
                perception.load_state_dict(torch.load(directory/'perception.pt',map_location=DEVICE,weights_only=False)['model'])
            else:perception=train_perception(tr,va,stats,cfg,directory)
            encode(perception,tr);encode(perception,va)
            stats=context_stats(tr,stats);np.savez(directory/'normalizers.npz',**stats)
            del perception
            DEVICE='cpu'
            for j,name in enumerate(MODELS):
                seed_all(seed)
                if (directory/(name+'_history.json')).exists() and (directory/(name+'.pt')).exists():continue
                train_dynamics(name,tr,va,stats,cfg,directory)
            write_json(done,dict(protocol_sha256=digest(OUT/'protocol.json'),seed=seed,fold=fold,
                                 checkpoint_sha256={n:digest(directory/(n+'.pt')) for n in ['perception',*MODELS]}))
    full=read_json(OUT/'protocol.json')
    if all((CK/f'{f}_seed{s}'/'TRAINING_COMPLETE.json').exists() for f in full['splits'] for s in full['seeds']):
        write_json(OUT/'TRAINING_COMPLETE.json',dict(protocol_sha256=digest(OUT/'protocol.json'),completed=time.strftime('%Y-%m-%d %H:%M:%S')))


def metrics(e):
    good=np.isfinite(e).all(1)
    if not good.all():return dict(n=len(e),nonfinite_windows=int((~good).sum()),E25=None,E50=None,E100=None,ADE=None)
    return dict(n=len(e),nonfinite_windows=0,E25=float(e[:,25].mean()),E50=float(e[:,50].mean()),
                E100=float(e[:,100].mean()),ADE=float(e[:,1:].mean()))


def test(cfg):
    global DEVICE
    if (OUT/'test_results.json').exists():raise RuntimeError('Final test results exist; refusing overwrite')
    if not (OUT/'TRAINING_COMPLETE.json').exists():raise RuntimeError('All training must finish before any test evaluation')
    for fold in cfg['splits']:
        for seed in cfg['seeds']:
            d=CK/f'{fold}_seed{seed}'
            done=read_json(d/'TRAINING_COMPLETE.json')
            assert done['protocol_sha256']==digest(OUT/'protocol.json')
            for n,h in done['checkpoint_sha256'].items(): assert digest(d/(n+'.pt'))==h
    results={};curves={}
    for fold,split in cfg['splits'].items():
        te=Records(split['test'])
        for seed in cfg['seeds']:
            DEVICE=VISION_DEVICE
            directory=CK/f'{fold}_seed{seed}'
            done=read_json(directory/'TRAINING_COMPLETE.json')
            assert done['protocol_sha256']==digest(OUT/'protocol.json')
            for name,sha in done['checkpoint_sha256'].items():assert sha==digest(directory/(name+'.pt'))
            stats=dict(np.load(directory/'normalizers.npz'))
            enc=RoadContextEncoder(predict_shape=True).to(DEVICE)
            enc.load_state_dict(torch.load(directory/'perception.pt',map_location=DEVICE,weights_only=False)['model'])
            encode(enc,te);ds=WindowData(te,100,cfg['evaluation_stride'])
            del enc
            DEVICE='cpu'
            for name in MODELS:
                model=make_model(name,stats,directory)
                model.load_state_dict(torch.load(directory/(name+'.pt'),map_location=DEVICE,weights_only=False)['model'])
                modes=cfg['preview_modes'] if name in ('frenet','cartesian_road') else ['normal']
                if name=='frenet':modes=modes+['kappa_zero','kappa_mean','kappa_permuted','legacy_hold_last']
                for mode in modes:
                    errors=evaluate_model(model,ds,mode='normal' if mode=='legacy_hold_last' else mode,
                                          stats=stats,hold_last=mode=='legacy_hold_last')
                    key=f'{fold}/seed{seed}/{name}/{mode}';results[key]=metrics(errors);curves[key]=errors
                    print(key,results[key],flush=True)
            ce=[]
            with torch.no_grad():
                for items in loader(ds,128):
                    z,a,truth,kp,shp=to_device(items)
                    ce.append(torch.linalg.vector_norm(constant_motion(z,a)[:,:,:2]-truth[:,:,:2],dim=-1).cpu().numpy())
            e=np.concatenate(ce);key=f'{fold}/seed{seed}/constant_motion/normal';results[key]=metrics(e);curves[key]=e
            # Audit oracle initialization, preview coverage, and perception performance.
            kerr=[];serr=[];beyond=[]
            for ep,t in ds.idx:
                r=te.records[ep]
                kerr.append((r['pred_kappa'][t]-r['kappa'][t])**2)
                serr.append((r['pred_shape'][t]-r['shape'][t])**2)
                s=r['z'][t:t+101,0];su=np.unwrap(s*2*np.pi/te.L)*te.L/(2*np.pi)
                beyond.append(float(su[-1]-su[0])>4.5)
            write_json(directory/'test_diagnostics.json',dict(curvature_RMSE=float(np.sqrt(np.mean(kerr))),
                    shape_RMSE=float(np.sqrt(np.mean(serr))),fraction_beyond_preview=float(np.mean(beyond)),
                    windows=[dict(file=te.records[e]['name'],t=int(t)) for e,t in ds.idx]))
    np.savez_compressed(OUT/'test_curves.npz',**curves)
    write_json(OUT/'test_results.json',dict(protocol_sha256=digest(OUT/'protocol.json'),results=results))


def report(cfg):
    results=read_json(OUT/'test_results.json')['results']
    lines=['# 独立划分与匹配基线：补充实验报告','','这是已有两条原始记录上的回顾性重分析，不是新采集的未见赛道测试。',
        '所有模型从零训练；仅用内层验证集选模型，训练完成后才打开外层测试评估。初始化仍使用真实状态，未来动作使用记录值。',
        '测试位置真值直接来自原始 pose，预测使用无地图输出；新数字不能直接替换旧表的地图辅助／投影真值数字。','',
        '## 固定协议','',f'- 训练种子：{cfg["seeds"]}；每方向分别训练。',
        '- 两条原始轨迹互为外层测试；训练来源末尾 20% 的片段用于验证，边界至少间隔 115 帧。',
        '- CNN 曲率／形状预测器训练 25 epochs；四种动力学模型均训练 40 epochs、32 步 rollout。',
        '- 动力学统一使用原始 ego 坐标下的多步位置、航向、速度、角速度损失；统一按验证 E100 选 checkpoint。',
        '- 所有标准化、平均曲率／形状仅由训练数据拟合；不加载旧模型。',
        '- 主方法在预览外沿末端切线延伸；另列旧版位置 hold-last 诊断，避免边界截断隐藏长时域误差。','',
        '| 外层 | 训练片段 | 验证片段 | 测试片段 | 边界剔除 | 测试原始记录 |','|---|---:|---:|---:|---:|---|']
    for f,s in cfg['splits'].items():lines.append(f'| {f} | {len(s["train"])} | {len(s["validation"])} | {len(s["test"])} | {len(s["purged"])} | {s["test_source"]} |')
    for fold in cfg['splits']:
        lines+=['',f'## {fold} 测试结果','','同一原始记录的种子均值 ± 种子样本标准差；不是置信区间。单位 m。','',
                '| 模型／预览 | E25 | E50 | E100 | ADE 1—100 |','|---|---:|---:|---:|---:|']
        rows=sorted(set('/'.join(k.split('/')[2:]) for k in results if k.startswith(fold+'/')))
        for row in rows:
            vals=[results[f'{fold}/seed{s}/{row}'] for s in cfg['seeds']]
            cells=[]
            for metric in ['E25','E50','E100','ADE']:
                a=[v[metric] for v in vals]
                cells.append('nonfinite' if None in a else f'{np.mean(a):.3f} ± {np.std(a,ddof=1):.3f}')
            lines.append('| '+row+' | '+' | '.join(cells)+' |')
    lines+=['','## 解释边界','',
        '- calibrated kinematic 是全局参数的动作响应模型；constant_motion 忽略未来动作，仅作诊断。',
        '- Cartesian road 与 Frenet 使用相同冻结感知器、相同初始道路状态、相同动作及监督目标。',
        '- Cartesian no-road 与 road 版本参数量相同，但 no-road 把所有道路信息清零。',
        '- normal/zero/train_mean/permuted 同时干预曲率和道路形状；kappa_* 只干预曲率，保留形状。',
        '- 干预是测试时移除信息的诊断，不等于重新训练过的最优无道路模型；后者由 Cartesian no-road 提供。',
        '- 两个外层方向及三个 seed 不能当作六次独立采集，不能据此证明跨赛道泛化。',
        '- 原记录此前参与过方法开发，因此本次隔离修复了当前拟合和选择流程，但仍需新数据作确认性验证。','',
        '## 可追溯文件','',
        '- `protocol.json`：在训练前冻结的划分、预算、消融与数据指纹。',
        '- `source_snapshot.json`：实验实现指纹。',
        '- `test_results.json` / `test_curves.npz`：每方向、每 seed、每模型的指标与逐窗口曲线。',
        '- `checkpoints/controlled_holdout/`：模型、归一化、训练历史和 checkpoint 指纹。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(OUT/'RESULTS.md',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','evaluate','report','all'])
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--outer',choices=['outer0','outer1']);parser.add_argument('--seed',type=int,choices=[0,1,2])
    args=parser.parse_args();torch.set_num_threads(args.threads)
    print('Device:',DEVICE,torch.cuda.get_device_name(0) if DEVICE=='cuda' else '',flush=True)
    if args.stage=='prepare':prepare();return
    cfg=read_json(OUT/'protocol.json')
    if args.outer is not None or args.seed is not None:
        if args.stage!='train' or args.outer is None or args.seed is None:
            raise ValueError('Single jobs require stage train and both --outer/--seed')
        cfg['splits']={args.outer:cfg['splits'][args.outer]};cfg['seeds']=[args.seed]
    if args.stage in ('train','all'):train(cfg)
    if args.stage in ('evaluate','all'):test(cfg)
    if args.stage in ('report','all'):report(cfg)

if __name__=='__main__':main()
