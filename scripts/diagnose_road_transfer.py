"""Frozen extension of no-refit diagnostics to inherited test partitions.
Runs only after all 27 branch fits; bins come from train-only diagnostics.
"""
from pathlib import Path
import sys,argparse
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import diagnose_road_branches as diag
import run_curvature_branches as branch
ex=diag.ex;b=diag.b
OUT=b.ROOT/'reports/road_branch_transfer'

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise RuntimeError('Already frozen')
    b.write_json(OUT/'protocol.json',dict(status='Retrospective test-partition diagnostic; no tuning or retraining',
        branch_protocol_sha256=b.digest(branch.OUT/'protocol.json'),
        diagnostic_protocol_sha256=b.digest(diag.OUT/'protocol.json'),
        sources={str(Path(__file__).relative_to(b.ROOT)):b.digest(__file__)},
        train_diagnostics={f'{f}_seed{s}':b.digest(diag.OUT/f'{f}_seed{s}'/'summary.json') for f in ['temporal','outer0','outer1'] for s in [0,1,2]},
        models=diag.MODELS,modes=['predicted','exact'],grouping='training thirds for initial speed, predicted curvature/shape RMSE; action support within vs any outside train coordinate ranges',
        timing='after all 27 branch fits; no new independent dataset; exact preview is a privileged intervention, not causal identification'))

def run():
    cfg=b.read_json(OUT/'protocol.json');bc=b.read_json(branch.OUT/'protocol.json');branch.verify(bc);branch.completed(bc)
    assert cfg['branch_protocol_sha256']==b.digest(branch.OUT/'protocol.json') and cfg['diagnostic_protocol_sha256']==b.digest(diag.OUT/'protocol.json')
    for n,h in cfg['sources'].items():assert b.digest(b.ROOT/n)==h
    for job,h in cfg['train_diagnostics'].items():assert b.digest(diag.OUT/job/'summary.json')==h
    ex.configure();torch.set_num_threads(2);allrows={};audits={};old=np.load(ex.OUT/'curves.npz')
    for fold,split in bc['splits'].items():
        rec=ex.CorrectedRecords(split['test'])
        for seed in bc['seeds']:
            job=f'{fold}_seed{seed}';d=ex.CK/job;enc=d/'independent';dest=OUT/job;dest.mkdir(exist_ok=True)
            if (dest/'COMPLETE.json').exists():
                mark=b.read_json(dest/'COMPLETE.json');assert mark['protocol_sha256']==b.digest(OUT/'protocol.json')
                for n,h in mark['files'].items():assert b.digest(dest/n)==h
                allrows[job]=b.read_json(dest/'summary.json');audits[job]=b.read_json(dest/'checks.json');continue
            ex.encode(rec,enc);ds=b.WindowData(rec,100,4);training=b.read_json(diag.OUT/job/'summary.json')
            cov=diag.covariates(rec,ds,np.asarray(training['train_action_support']));bins=training['train_bins']
            stats=dict(np.load(enc/'normalizers.npz'));limits=b.read_json(d/'limits.json')['limits'];arrays={};rows={};checks={}
            for k,v in cov.items():arrays['covariates/'+k]=v
            ws=[dict(file=rec.records[e]['name'],t=int(t)) for e,t in ds.idx];assert ws==b.read_json(d/'evaluation_windows.json')
            for name in cfg['models']:
                m=ex.make(name,stats,enc,limits);m.load_state_dict(torch.load(d/(name+'.pt'),weights_only=False)['model'])
                for mode,records in [('predicted',rec),('exact',ex.c.oracle_copy(rec))]:
                    data=b.WindowData(records,100,4);values=diag.evaluate(m,data,name=='geometry_midpoint');key=name+'/'+mode;err=values['position_error']
                    row=dict(**b.metrics(err),speed_E100=float(values['speed_error'][:,-1].mean()),yaw_rate_E100=float(values['yaw_rate_error'][:,-1].mean()),
                        shape_rmse_mean=float(cov['shape_rmse'].mean()),curvature_rmse_mean=float(cov['curvature_rmse'].mean()),
                        any_action_outside_train_fraction=float((cov['action_outside_support']>0).mean()),
                        stages={k:float(v.min() if k.endswith('_min') else v.mean()) for k,v in values.items() if '/' in k},groups={})
                    for feature in ['initial_speed','curvature_rmse','shape_rmse']:
                        ids=np.searchsorted(bins[feature],cov[feature],side='right')
                        row['groups'][feature]=[dict(n=int((ids==i).sum()),E100=float(err[ids==i,-1].mean()) if (ids==i).any() else None) for i in range(3)]
                    for event in ['den_guard','den_invalid','preview_outside','action_outside_support']:
                        mask=cov[event]>0 if event=='action_outside_support' else np.logical_or.reduce([v>0 for k,v in values.items() if k.endswith('/'+event)])
                        row['groups'][event]=dict(affected_n=int(mask.sum()),affected_E100=float(err[mask,-1].mean()) if mask.any() else None,
                            unaffected_n=int((~mask).sum()),unaffected_E100=float(err[~mask,-1].mean()) if (~mask).any() else None)
                    rows[key]=row
                    for k,v in values.items():arrays[key+'/'+k]=v
                    if mode=='predicted':
                        delta=float(abs(err-old[f'{fold}/seed{seed}/{name}']).max());assert np.allclose(err,old[f'{fold}/seed{seed}/{name}'],atol=2e-5,rtol=2e-4)
                    else:
                        indices=np.unique(np.linspace(0,len(data)-1,24,dtype=int));measured=b.evaluate_model(m,torch.utils.data.Subset(data,indices),batch=24)
                        delta=float(abs(measured-err[indices]).max());assert np.allclose(measured,err[indices],atol=2e-5,rtol=2e-4)
                    checks[key]=dict(max_absolute_difference_m=delta,scope='all inherited windows' if mode=='predicted' else '24 independent uninstrumented recomputations')
                    print(job,key,row['E100'],flush=True)
            np.savez_compressed(dest/'arrays.npz',**arrays);b.write_json(dest/'windows.json',ws);b.write_json(dest/'summary.json',rows);b.write_json(dest/'checks.json',checks)
            b.write_json(dest/'COMPLETE.json',dict(protocol_sha256=b.digest(OUT/'protocol.json'),files={p:b.digest(dest/p) for p in ['arrays.npz','windows.json','summary.json','checks.json']}))
            allrows[job]=rows;audits[job]=checks
    b.write_json(OUT/'summary.json',allrows);b.write_json(OUT/'audit.json',dict(status='PASS',rows=36,checks=audits,protocol_sha256=b.digest(OUT/'protocol.json')))
    lines=['# 来源转移与时间测试的无重训诊断','','旧记录上的回顾性诊断；全体 27 个分支模型训练结束后执行。分组边界来自训练分区。真值道路替换是特权诊断，并不证明因果归因。','',
           '| 划分 | 模型 | 预测道路 E100 | 真值道路 E100 |','|---|---|---:|---:|']
    for fold in bc['splits']:
        for model in cfg['models']:
            cells=[]
            for mode in cfg['modes']:
                vals=[allrows[f'{fold}_seed{s}'][model+'/'+mode]['E100'] for s in bc['seeds']];cells.append(f'{np.mean(vals):.3f} ± {np.std(vals,ddof=1):.3f}')
            lines.append(f'| {fold} | {model} | '+' | '.join(cells)+' |')
    lines+=['','逐窗口速度／角速度误差、道路误差、训练动作范围、分组和完整积分阶段诊断见各 job 数组与汇总。所有情景保留。','', '模型名称：GPT-6（Codex）。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);args=p.parse_args();globals()[args.stage]()
