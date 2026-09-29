"""Exploratory unit-arc coupling on the frozen temporal split; retain all controls."""
from pathlib import Path
import argparse,copy,sys,subprocess,time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_holdout as b
import run_conditioned_dynamics as c
import run_temporal_holdout as temporal
from models.conditioned_road_encoder import ConditionedRoadEncoder
from models.coupled_road_encoder import CoupledRoadEncoder
from baselines.unit_arc_geometry import ArcFrenet,arc_points
OUT=b.ROOT/'reports/coupled_road';CK=b.ROOT/'checkpoints/coupled_road'
PARENT_OUT=temporal.OUT;PARENT_CK=temporal.CK
NAMES=['arc_full','arc_shape','arc_independent_encoder','cartesian_road']

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);CK.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise RuntimeError('Protocol exists')
    cfg=b.read_json(PARENT_OUT/'protocol.json')
    paths=[Path(__file__),b.ROOT/'scripts/run_temporal_holdout.py',b.ROOT/'scripts/run_conditioned_dynamics.py',b.ROOT/'scripts/run_controlled_holdout.py',
           b.ROOT/'src/models/coupled_road_encoder.py',b.ROOT/'src/baselines/unit_arc_geometry.py',b.ROOT/'scripts/check_unit_arc_geometry.py']
    cfg.update(status='exploratory method study on reused records, frozen before fitting; no independent confirmation',
        rationale='one curvature-knot representation generates an exact unit-speed piecewise-arc shape; joint curvature/shape supervision removes independent shape-head inconsistency',
        variants=NAMES,primary_model='arc_full',temporal_parent_sha256=b.digest(PARENT_OUT/'protocol.json'),
        independent_encoders={f'temporal_seed{s}':b.digest(PARENT_CK/f'temporal_seed{s}'/'perception.pt') for s in cfg['seeds']},
        curve='interval curvature = mean of neighboring predicted knots; exact circular-arc integral; tangent extension with curvature zero; explicit midpoint Frenet with positive denominator floor .2',
        comparison='arc_independent_encoder uses previous frozen independent-head CNN curvature with same arc readout/dynamics; arc_shape clears curvature in dynamics but retains curved readout; Cartesian shares coupled CNN inputs')
    cfg['frozen_sources'].update({str(p.relative_to(b.ROOT)):b.digest(p) for p in paths})
    b.write_json(OUT/'protocol.json',cfg);print('Frozen coupled-road protocol',flush=True)

def verify(cfg):
    assert cfg['temporal_parent_sha256']==b.digest(PARENT_OUT/'protocol.json')
    for p,h in cfg['frozen_sources'].items():assert b.digest(b.ROOT/p)==h,p
    for job,h in cfg['independent_encoders'].items():assert b.digest(PARENT_CK/job/'perception.pt')==h
    assert cfg['track_sha256']==b.digest(b.DATA/'_meta/track.npz')

def configure(cfg):
    temporal.OUT=OUT;temporal.CK=CK;temporal.ConditionedRoadEncoder=CoupledRoadEncoder;temporal.verify=verify
    c.VARIANTS={n:(True,'full',False) for n in NAMES};c.make=make

def clone_records(records):
    out=copy.copy(records);out.records=[r.copy() for r in records.records];return out

def encode(records,job,independent=False):
    b.DEVICE=b.VISION_DEVICE;directory=(PARENT_CK if independent else CK)/job
    saved=torch.load(directory/'perception.pt',map_location=b.DEVICE,weights_only=False)
    cls=ConditionedRoadEncoder if independent else CoupledRoadEncoder
    model=cls(saved['mean'],saved['scale']).to(b.DEVICE);model.load_state_dict(saved['model']);b.encode(model,records)
    if independent:
        for r in records.records:
            # Only the old curvature pathway is used; the independent shape head
            # is replaced by the same analytic arc decoder as the coupled model.
            with torch.no_grad():r['pred_shape']=arc_points(torch.tensor(r['pred_kappa'])).numpy()
    del model;b.DEVICE='cpu'

def make(name,stats,d,limits):
    if name=='cartesian_road':return b.make_model(name,stats,d)
    return ArcFrenet(b.make_model('frenet',stats,d),limits,curvature=name!='arc_shape')

def perception(cfg,seed):
    configure(cfg);temporal.perception(cfg,seed)

def fit(cfg,seed):
    verify(cfg);configure(cfg);d=CK/f'temporal_seed{seed}'
    if (d/'COMPLETE.json').exists():return
    if (OUT/'results.json').exists():raise RuntimeError('Evaluation already completed')
    done=b.read_json(d/'PERCEPTION_COMPLETE.json');assert done['protocol_sha256']==b.digest(OUT/'protocol.json')
    assert done['checkpoint_sha256']==b.digest(d/'perception.pt') and done['normalizers_sha256']==b.digest(d/'normalizers.npz')
    split=cfg['splits']['temporal'];tr=b.Records(split['train']);va=b.Records(split['validation'])
    ind_tr=clone_records(tr);ind_va=clone_records(va);encode(tr,d.name);encode(va,d.name);encode(ind_tr,d.name,True);encode(ind_va,d.name,True)
    stats=dict(np.load(d/'normalizers.npz'));limits=b.read_json(d/'limits.json')['limits']
    for name in NAMES:
        if (d/(name+'_history.json')).exists():continue
        train,val=(ind_tr,ind_va) if name=='arc_independent_encoder' else (tr,va)
        c.fit(name,train,val,stats,d,d,cfg,limits,seed)
    b.write_json(d/'COMPLETE.json',dict(protocol_sha256=b.digest(OUT/'protocol.json'),checkpoint_sha256={n:b.digest(d/(n+'.pt')) for n in NAMES+['perception']}))

def evaluate(cfg):
    verify(cfg);configure(cfg)
    if (OUT/'results.json').exists():raise RuntimeError('Refusing evaluation overwrite')
    for seed in cfg['seeds']:
        d=CK/f'temporal_seed{seed}';done=b.read_json(d/'COMPLETE.json');assert done['protocol_sha256']==b.digest(OUT/'protocol.json')
        for n,h in done['checkpoint_sha256'].items():assert b.digest(d/(n+'.pt'))==h
    original=b.Records(cfg['splits']['temporal']['test']);results={};curves={};diagnostics={}
    for seed in cfg['seeds']:
        d=CK/f'temporal_seed{seed}';records=clone_records(original);ind=clone_records(original);encode(records,d.name);encode(ind,d.name,True)
        stats=dict(np.load(d/'normalizers.npz'));limits=b.read_json(d/'limits.json')['limits']
        for name in NAMES:
            selected=ind if name=='arc_independent_encoder' else records;ds=b.WindowData(selected,100,4);m=make(name,stats,d,limits)
            saved=torch.load(d/(name+'.pt'),weights_only=False);m.load_state_dict(saved['model']);monitor=c.Monitor();hook=m.register_forward_pre_hook(monitor.hook,with_kwargs=True)
            errors=b.evaluate_model(m,ds);hook.remove();key=f'temporal/seed{seed}/{name}';results[key]=b.metrics(errors);curves[key]=errors;diagnostics[key]=monitor.result();print(key,results[key],flush=True)
        b.write_json(d/'evaluation_windows.json',[dict(file=records.records[e]['name'],t=int(t)) for e,t in ds.idx])
    np.savez_compressed(OUT/'curves.npz',**curves);b.write_json(OUT/'results.json',dict(protocol_sha256=b.digest(OUT/'protocol.json'),results=results,diagnostics=diagnostics))
def audit(cfg):
    verify(cfg);parent=b.read_json(b.OUT/'protocol.json')
    for n,v in parent['files'].items():assert b.digest(b.DATA/n)==v['sha256']
    for n,h in cfg['raw_fingerprints'].items():assert b.digest(b.RAW/n)==h
    data=b.read_json(OUT/'results.json');curves=np.load(OUT/'curves.npz');assert len(data['results'])==len(NAMES)*3 and set(data['results'])==set(curves.files)
    assert data['protocol_sha256']==b.digest(OUT/'protocol.json');expected=None;sub={}
    for seed in cfg['seeds']:
        d=CK/f'temporal_seed{seed}';ws=b.read_json(d/'evaluation_windows.json');done=b.read_json(d/'COMPLETE.json')
        assert done['protocol_sha256']==data['protocol_sha256']
        if expected is not None:assert expected==ws
        expected=ws;assert all(w['file'] in cfg['splits']['temporal']['test'] for w in ws)
        for n,h in done['checkpoint_sha256'].items():assert b.digest(d/(n+'.pt'))==h
        ph=b.read_json(d/'perception_history.json');pc=torch.load(d/'perception.pt',weights_only=False);pb=min(ph,key=lambda x:x['val_loss']);assert len(ph)==25 and pc['epoch']==pb['epoch'] and pc['val_loss']==pb['val_loss']
        for name in NAMES:
            hist=b.read_json(d/(name+'_history.json'));best=min((x for x in hist if x['val_E100'] is not None),key=lambda x:x['val_E100']);saved=torch.load(d/(name+'.pt'),weights_only=False)
            assert len(hist)==40 and saved['epoch']==best['epoch'] and saved['val_E100']==best['val_E100']
            key=f'temporal/seed{seed}/{name}';e=curves[key];assert e.shape==(len(ws),101) and np.allclose(e[:,0],0,atol=1e-7);assert b.metrics(e)==data['results'][key]
            for source in cfg['source_partitions']:
                mask=np.array([b.PAT.fullmatch(w['file'])[1]==source for w in ws]);sub[f'{source}/seed{seed}/{name}']=b.metrics(e[mask])
    if (PARENT_CK/'temporal_seed0/evaluation_windows.json').exists():assert expected==b.read_json(PARENT_CK/'temporal_seed0/evaluation_windows.json')
    b.write_json(OUT/'per_source_results.json',dict(results=sub))
    b.write_json(OUT/'audit.json',dict(status='PASS',rows=len(NAMES)*3,checks=['frozen temporal split and implementation hashes','all raw/prepared data hashes','all seeds retained','25/40 epochs, validation-only checkpoint selection','identical test windows','all pooled/per-source metrics recomputed','zero initial errors']))

def report(cfg):
    all_rows=b.read_json(OUT/'results.json')['results'];sub=b.read_json(OUT/'per_source_results.json')['results']
    lines=['# 单一曲率表示与单位弧长道路解码','','探索性方法实验，沿用冻结的两来源时间划分。全新耦合 CNN 从零训练 25 epochs；每个动力学从零训练 40 epochs；3 seeds 全部保留。主模型预先固定为 arc_full，主指标为 pooled E100。',
           '道路曲率在每个 0.5 m 区间内为相邻预测节点的均值，位置与航向通过解析圆弧积分生成。因此位置导数为单位切向量，动力学查询曲率与读出曲线一致（区间节点处曲率允许跳变）。预览外使用零曲率的切线延伸。',
           'arc_independent_encoder 使用上一轮独立双头 CNN 的曲率，配相同圆弧解码和动力学；它隔离联合感知约束相对于只更改解码的效果。arc_shape 仅在动力学中将曲率清零，曲线读出仍然弯曲，不能称为完全没有曲率信息。Cartesian 接收同一个耦合 CNN 的曲率及形状。','']
    for source in ['temporal']+list(cfg['source_partitions']):
        data=all_rows if source=='temporal' else sub;lines += [f'## {source}','','| 模型 | E25 | E50 | E100 | ADE |','|---|---:|---:|---:|---:|']
        for name in NAMES:
            cells=[]
            for m in ['E25','E50','E100','ADE']:
                v=[data[f'{source}/seed{s}/{name}'][m] for s in cfg['seeds']];cells.append('nonfinite' if None in v else f'{np.mean(v):.3f} ± {np.std(v,ddof=1):.3f}')
            lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines+=['','方差统计为训练种子样本标准差，不是独立记录的置信区间。简单标定运动学参照 temporal_holdout 的相同窗口结果；新编码器不会改变该控制的输入。',
            '这个时间划分仍包含可能的工况变化：训练油门只有约 0.567 和 0.633 两档。所有记录此前已被研究，不能称为新的独立验证。',
            '数值保护沿用正分母下限 0.2、角度回绕和训练分位数增量限制；单位弧长不保证 Frenet 图有效，也不保证轨迹误差有界。chart 监测只记录完整时间步，并不包含每个 midpoint 内部阶段；arc_shape 的零曲率分母统计不代表真实曲线图始终有效。','',
            '模型名称：GPT-6（Codex）；当前会话未提供更细型号标识。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def jobs(cfg,stage):
    marker='PERCEPTION_COMPLETE.json' if stage=='perception' else 'COMPLETE.json';queue=[s for s in cfg['seeds'] if not (CK/f'temporal_seed{s}'/marker).exists()];active=[];capacity=2 if stage=='perception' else 3
    while queue or active:
        while queue and len(active)<capacity:
            seed=queue.pop(0);fh=(OUT/f'{stage}_seed{seed}.log').open('a',encoding='utf-8');p=subprocess.Popen([sys.executable,'-u',str(Path(__file__)) ,stage,'--seed',str(seed)],cwd=b.ROOT,stdout=fh,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));active.append((p,fh,seed));print('START',stage,seed,p.pid,flush=True)
        for item in active[:]:
            p,fh,seed=item
            if p.poll() is not None:
                fh.close();active.remove(item);print('DONE',stage,seed,p.returncode,flush=True)
                if p.returncode:
                    for q,f,_ in active:q.terminate();q.wait();f.close()
                    raise RuntimeError('Job failed; inspect logs')
        time.sleep(2)
    if stage=='fit':evaluate(cfg);audit(cfg);report(cfg)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','perception','perceptions','fit','jobs','evaluate','audit','report']);ap.add_argument('--seed',type=int);args=ap.parse_args();torch.set_num_threads(4);b.DEVICE='cpu'
    if args.stage=='prepare':prepare();return
    cfg=b.read_json(OUT/'protocol.json')
    if args.stage in ['perception','fit']:globals()[args.stage](cfg,args.seed)
    elif args.stage=='perceptions':jobs(cfg,'perception')
    elif args.stage=='jobs':jobs(cfg,'fit')
    else:globals()[args.stage](cfg)
if __name__=='__main__':main()
