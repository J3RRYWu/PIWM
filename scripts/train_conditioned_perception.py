"""Train all six conditioned encoders; select using validation only.

No held-out-source data are loaded or scored by this script. Fixes were chosen
from original training/validation evidence of a nearly constant encoder.
"""
from pathlib import Path
import argparse,sys,subprocess,time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_holdout as b
from models.conditioned_road_encoder import ConditionedRoadEncoder
OUT=b.ROOT/'reports/conditioned_perception';CK=b.ROOT/'checkpoints/conditioned_perception'


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);CK.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise RuntimeError('Protocol already exists')
    old=b.read_json(b.OUT/'protocol.json')
    paths=[Path(__file__),b.ROOT/'src/models/conditioned_road_encoder.py',b.ROOT/'scripts/run_controlled_holdout.py',b.ROOT/'src/models/road_perception.py',b.ROOT/'src/models/encoder_lane.py']
    b.write_json(OUT/'protocol.json',dict(parent_sha256=b.digest(b.OUT/'protocol.json'),splits=old['splits'],seeds=old['seeds'],epochs=25,lr=.001,batch=64,stride=2,
        changes='standardized physical outputs: train mean+scale*raw; replace ReLU with LeakyReLU(.01); same width, depth, Adam schedule and budget',
        scope='development/validation only, all six encoders; no selection based on outer-source evaluation',
        source_sha256={str(p.relative_to(b.ROOT)):b.digest(p) for p in paths}))


def verify(cfg):
    assert cfg['parent_sha256']==b.digest(b.OUT/'protocol.json')
    for p,h in cfg['source_sha256'].items():assert b.digest(b.ROOT/p)==h


def train(cfg,fold,seed):
    verify(cfg);d=CK/f'{fold}_seed{seed}';d.mkdir(parents=True,exist_ok=True)
    if (d/'COMPLETE.json').exists():return
    b.seed_all(seed);tr=b.Records(cfg['splits'][fold]['train']);va=b.Records(cfg['splits'][fold]['validation'])
    stats=b.fit_stats(tr,d);td=b.PerceptionData(tr,2);vd=b.PerceptionData(va,2)
    target=np.stack([td[i][1].numpy() for i in range(len(td))]);mean=target.mean(0);scale=stats['perception_scale']
    np.savez(d/'perception_stats.npz',mean=mean,scale=scale)
    model=ConditionedRoadEncoder(mean,scale).to(b.DEVICE);opt=torch.optim.Adam(model.parameters(),lr=.001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,25)
    scale_t=torch.tensor(scale,device=b.DEVICE);hist=[];best=float('inf')
    for ep in range(25):
        model.train();tot=0;n=0;gradmax=0
        for x,y in b.loader(td,64,True):
            x=x.to(b.DEVICE);y=y.to(b.DEVICE);kap,shape=model.forward_both(x)
            loss=((torch.cat([kap,shape.flatten(1)],-1)-y)/scale_t).square().mean()
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5));assert np.isfinite(gn)
            opt.step();tot+=float(loss.detach())*len(x);n+=len(x);gradmax=max(gradmax,gn)
        model.eval();score=0;nv=0;predictions=[];truth=[]
        with torch.no_grad():
            for x,y in b.loader(vd,64):
                x=x.to(b.DEVICE);y=y.to(b.DEVICE);kap,shape=model.forward_both(x);p=torch.cat([kap,shape.flatten(1)],-1)
                score+=float(((p-y)/scale_t).square().mean())*len(x);nv+=len(x);predictions.append(p.cpu().numpy());truth.append(y.cpu().numpy())
        pred=np.concatenate(predictions);gt=np.concatenate(truth);score/=nv
        row=dict(epoch=ep+1,train_loss=tot/n,val_loss=score,max_gradient=gradmax,
            val_curvature_RMSE=float(np.sqrt(np.mean((pred[:,:10]-gt[:,:10])**2))),val_shape_RMSE=float(np.sqrt(np.mean((pred[:,10:]-gt[:,10:])**2))),
            val_mean_curvature_output_std=float(pred[:,:10].std(0).mean()))
        hist.append(row)
        if score<best:
            best=score;torch.save(dict(model=model.state_dict(),epoch=ep+1,val_loss=score,mean=mean,scale=scale),d/'perception.pt')
        sched.step();b.write_json(d/'progress.json',hist)
        print(d.name,f'{ep+1}/25',row,flush=True)
    b.write_json(d/'perception_history.json',hist)
    b.write_json(d/'COMPLETE.json',dict(protocol_sha256=b.digest(OUT/'protocol.json'),checkpoint_sha256=b.digest(d/'perception.pt')))


def report(cfg):
    verify(cfg);lines=['# 感知器数值条件化：开发／验证检查','','所有六个感知器统一从零重训，未加载外层评价来源。选择最低验证归一化 MSE 的 epoch。',
        'ReLU 改为 LeakyReLU，网络输出在训练均值／标准差坐标下参数化，再恢复物理单位；是组合数值修正，非单因素消融。','',
        '| 方向/种子 | 原验证损失 | 新验证损失 | 新曲率 RMSE | 新形状 RMSE | 新曲率输出 std |','|---|---:|---:|---:|---:|---:|']
    audit={}
    for f in cfg['splits']:
        for seed in cfg['seeds']:
            job=f'{f}_seed{seed}';d=CK/job;done=b.read_json(d/'COMPLETE.json');assert done['protocol_sha256']==b.digest(OUT/'protocol.json')
            assert done['checkpoint_sha256']==b.digest(d/'perception.pt')
            h=b.read_json(d/'perception_history.json');best=min(h,key=lambda x:x['val_loss']);saved=torch.load(d/'perception.pt',map_location='cpu',weights_only=False);assert best['epoch']==saved['epoch'] and len(h)==25
            old=min(b.read_json(b.CK/job/'perception_history.json'),key=lambda x:x['val_loss'])
            audit[job]=dict(original=old,conditioned=best)
            lines.append(f'| {job} | {old["val_loss"]:.4f} | {best["val_loss"]:.4f} | {best["val_curvature_RMSE"]:.4f} | {best["val_shape_RMSE"]:.4f} | {best["val_mean_curvature_output_std"]:.4f} |')
    b.write_json(OUT/'validation_comparison.json',audit)
    lines+=['','这些是验证指标，不是无偏泛化估计；感知改善是否带来轨迹收益需要另行固定动力学对照。','', '模型名称：GPT-6（Codex）；当前会话未提供更细型号标识。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def jobs(cfg):
    queue=[(f,s) for f in cfg['splits'] for s in cfg['seeds'] if not (CK/f'{f}_seed{s}'/'COMPLETE.json').exists()];active=[]
    while queue or active:
        while queue and len(active)<2:
            f,s=queue.pop(0);fh=(OUT/f'{f}_seed{s}.log').open('a',encoding='utf-8')
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'train','--outer',f,'--seed',str(s)],cwd=b.ROOT,stdout=fh,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));active.append((p,fh,f,s));print('START',f,s,p.pid,flush=True)
        for item in active[:]:
            p,fh,f,s=item
            if p.poll() is not None:
                fh.close();active.remove(item);print('DONE',f,s,p.returncode,flush=True)
                if p.returncode:
                    for q,log,_,_ in active:q.terminate();q.wait();log.close()
                    raise RuntimeError('Training failed; inspect logs')
        time.sleep(2)
    report(cfg)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','jobs','report']);parser.add_argument('--outer');parser.add_argument('--seed',type=int);args=parser.parse_args();torch.set_num_threads(4)
    if args.stage=='prepare':prepare();return
    cfg=b.read_json(OUT/'protocol.json')
    if args.stage=='train':train(cfg,args.outer,args.seed)
    else:globals()[args.stage](cfg)

if __name__=='__main__':main()
