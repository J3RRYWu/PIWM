"""No-refit train/validation diagnosis; all geometric midpoint stages counted.

Exact-road replacement is a privileged input intervention, not a deployable result
or a causal identification. Training quantiles determine descriptive bins.
"""
from pathlib import Path
import argparse, types, sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_corrected_query as ex
b = ex.b
OUT = b.ROOT / 'reports/road_branch_diagnosis'
MODELS = ['guarded_full', 'geometry_midpoint']

class StageMonitor:
    def __init__(self, n):
        self.n = n
        self.data = {}
    def record(self, stage, z, k, g):
        den = 1 - z[:, 1] * k
        x = self.data.setdefault(stage, dict(count=0, den_min=np.full(self.n, np.inf),
            g_min=np.full(self.n, np.inf), den_guard=np.zeros(self.n),
            den_invalid=np.zeros(self.n), g_guard=np.zeros(self.n),
            g_degenerate=np.zeros(self.n), preview_outside=np.zeros(self.n)))
        x['count'] += 1
        x['den_min'] = np.minimum(x['den_min'], den.detach().numpy())
        x['g_min'] = np.minimum(x['g_min'], g.detach().numpy())
        for key, condition in [('den_guard', den < .2), ('den_invalid', den <= 0),
                               ('g_guard', g < .1), ('g_degenerate', g <= 1e-8),
                               ('preview_outside', (z[:, 0] < 0) | (z[:, 0] > 4.5))]:
            x[key] += condition.detach().numpy()
    def arrays(self):
        return {stage + '/' + key: (value if key.endswith('_min') else value / x['count'])
                for stage, x in self.data.items() for key, value in x.items() if key != 'count'}

@torch.no_grad()
def evaluate(m, ds, geometric):
    m.eval()
    accumulated = {}
    for z, a, truth, k, p in b.loader(ds, 128):
        monitor = StageMonitor(len(z))
        if geometric:
            original = m.field
            calls = [0]
            def field(self, state, action, g, kap):
                monitor.record('start' if calls[0] % 2 == 0 else 'midpoint', state, kap, g)
                calls[0] += 1
                return original(state, action, g, kap)
            m.field = types.MethodType(field, m)
        else:
            def hook(model, args, kwargs):
                monitor.record('start', args[0], kwargs['kappa_override'], torch.ones_like(args[0][:, 0]))
            handle = m.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            pred = m.rollout(z, a, k, p)
        finally:
            if geometric:
                m.field = original
            else:
                handle.remove()
        assert torch.isfinite(pred).all()
        values = dict(position_error=torch.linalg.vector_norm(pred[:, :, :2] - truth[:, :, :2], dim=-1).numpy(),
                      speed_error=(pred[:, :, 3] - truth[:, :, 3]).abs().numpy(),
                      yaw_rate_error=(pred[:, :, 4] - truth[:, :, 4]).abs().numpy(), **monitor.arrays())
        for key, value in values.items(): accumulated.setdefault(key, []).append(value)
    return {key: np.concatenate(value) for key, value in accumulated.items()}

def covariates(rec, ds, support):
    result = {key: [] for key in ['initial_speed', 'action_outside_support', 'curvature_rmse', 'shape_rmse']}
    for e, t in ds.idx:
        r = rec.records[e]
        result['initial_speed'].append(float(r['z'][t, 3]))
        actions = r['actions'][t:t+100]
        result['action_outside_support'].append(float(np.any((actions < support[0]-1e-6) | (actions > support[1]+1e-6), axis=1).mean()))
        result['curvature_rmse'].append(float(np.sqrt(np.mean((r['pred_kappa'][t]-r['kappa'][t])**2))))
        result['shape_rmse'].append(float(np.sqrt(np.mean((r['pred_shape'][t]-r['shape'][t])**2))))
    return {key: np.asarray(value) for key, value in result.items()}

def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'protocol.json').exists(): raise RuntimeError('Already frozen')
    cfg = b.read_json(ex.OUT/'protocol.json'); ex.verify(cfg)
    paths = [Path(__file__)]
    b.write_json(OUT/'protocol.json', dict(parent_sha256=b.digest(ex.OUT/'protocol.json'),
        sources={str(p.relative_to(b.ROOT)): b.digest(p) for p in paths}, splits=cfg['splits'], seeds=cfg['seeds'],
        partitions=['train', 'validation'], models=MODELS, previews=['predicted', 'exact'],
        train_quantiles=[1/3, 2/3], support='per-action-coordinate train min/max; tolerance 1e-6',
        boundaries='den_guard<0.2; positive-chart violation den<=0; g_guard<0.1; g_degenerate<=1e-8; preview q outside [0,4.5]',
        status='Retrospective no-refit diagnostics; no test partition loaded; no causal decomposition.'))

def run():
    cfg = b.read_json(OUT/'protocol.json'); ex.verify(b.read_json(ex.OUT/'protocol.json'))
    assert cfg['parent_sha256'] == b.digest(ex.OUT/'protocol.json')
    for p, h in cfg['sources'].items(): assert b.digest(b.ROOT/p) == h
    ex.configure(); torch.set_num_threads(2)
    summary = {}
    for fold, split in cfg['splits'].items():
        tr = ex.CorrectedRecords(split['train']); va = ex.CorrectedRecords(split['validation'])
        actions = np.concatenate([r['actions'] for r in tr.records]); support = np.stack([actions.min(0), actions.max(0)])
        for seed in cfg['seeds']:
            job = f'{fold}_seed{seed}'; d = ex.CK/job; dest = OUT/job; dest.mkdir(exist_ok=True)
            if (dest/'COMPLETE.json').exists():
                mark = b.read_json(dest/'COMPLETE.json')
                assert mark['protocol_sha256'] == b.digest(OUT/'protocol.json')
                for p,h in mark['files'].items(): assert b.digest(dest/p) == h
                summary[job] = b.read_json(dest/'summary.json'); continue
            mark = b.read_json(d/'COMPLETE.json')
            for p,h in mark['files'].items(): assert b.digest(d/p) == h
            enc = d/'independent'; stats = dict(np.load(enc/'normalizers.npz')); limits = b.read_json(d/'limits.json')['limits']
            rows = {}; bins = {}; saved_arrays = {}; windows = {}
            for part, rec in [('train',tr),('validation',va)]:
                ex.encode(rec, enc); ds = b.WindowData(rec,100,4); cov = covariates(rec, ds, support)
                windows[part] = [dict(file=rec.records[e]['name'],t=int(t)) for e,t in ds.idx]
                if part == 'train': bins = {key: np.quantile(value,[1/3,2/3]).tolist() for key,value in cov.items()}
                for key,value in cov.items(): saved_arrays[f'{part}/covariates/{key}'] = value
                exact = ex.c.oracle_copy(rec)
                for name in MODELS:
                    m = ex.make(name,stats,enc,limits)
                    m.load_state_dict(torch.load(d/(name+'.pt'),weights_only=False)['model'])
                    for mode, records in [('predicted',rec),('exact',exact)]:
                        values = evaluate(m,b.WindowData(records,100,4),name=='geometry_midpoint')
                        key = f'{part}/{name}/{mode}'; err = values['position_error']
                        row = dict(**b.metrics(err), speed_E100=float(values['speed_error'][:,-1].mean()),
                                   yaw_rate_E100=float(values['yaw_rate_error'][:,-1].mean()),
                                   stages={k:float(v.min() if k.endswith('_min') else v.mean()) for k,v in values.items() if '/' in k}, bins={})
                        for feature,v in cov.items():
                            ix = np.searchsorted(bins[feature],v,side='right')
                            row['bins'][feature] = [dict(n=int((ix==i).sum()), E100=float(err[ix==i,-1].mean()) if (ix==i).any() else None) for i in range(3)]
                        for event in ['den_guard','den_invalid','preview_outside']:
                            mask = np.logical_or.reduce([v>0 for k,v in values.items() if k.endswith('/'+event)])
                            row[event+'_windows'] = dict(n=int(mask.sum()), E100=float(err[mask,-1].mean()) if mask.any() else None,
                                                         unaffected_n=int((~mask).sum()), unaffected_E100=float(err[~mask,-1].mean()) if (~mask).any() else None)
                        rows[key]=row
                        for k,v in values.items(): saved_arrays[key+'/'+k]=v
                        print(job,key,'E100',row['E100'],flush=True)
            np.savez_compressed(dest/'arrays.npz',**saved_arrays)
            b.write_json(dest/'windows.json',windows)
            b.write_json(dest/'summary.json',dict(rows=rows,train_bins=bins,train_action_support=support.tolist()))
            b.write_json(dest/'COMPLETE.json',dict(protocol_sha256=b.digest(OUT/'protocol.json'),files={p:b.digest(dest/p) for p in ['arrays.npz','windows.json','summary.json']}))
            summary[job]=b.read_json(dest/'summary.json')
    b.write_json(OUT/'summary.json',summary)
    report(summary)

def report(summary):
    lines=['# 道路分支修改前的无重训诊断','','仅训练／验证分区；全部三种子。真值预览替换是特权诊断，不能当作部署结果或因果归因。','',
           '| 划分 | 分区 | 模型 | 预测预览 E100 | 真值预览 E100 |','|---|---|---|---:|---:|']
    cfg=b.read_json(OUT/'protocol.json')
    for fold in cfg['splits']:
        for part in cfg['partitions']:
            for model in MODELS:
                values=[]
                for mode in ['predicted','exact']:
                    vals=[summary[f'{fold}_seed{s}']['rows'][f'{part}/{model}/{mode}']['E100'] for s in cfg['seeds']]
                    values.append(f'{np.mean(vals):.3f} ± {np.std(vals,ddof=1):.3f}')
                lines.append(f'| {fold} | {part} | {model} | '+ ' | '.join(values)+' |')
    lines+=['','每窗口位置、速度与角速度误差、训练分位分组、动作支持范围、初始及 midpoint 阶段监测均保存在各 job 的 arrays.npz / summary.json。分母触发保护不等于坐标已失效；局部正分母也不保证全局投影唯一。', '', '模型名称：GPT-6（Codex）。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run']);args=ap.parse_args()
    globals()[args.stage]()
