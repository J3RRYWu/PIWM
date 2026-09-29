"""Frozen initialization sensitivity scenarios; no refits or test-based tuning."""
from pathlib import Path
import sys,argparse
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_corrected_query as ex
b=ex.b
OUT=b.ROOT/'reports/corrected_initialization'
NAMES=['kinematic','cartesian_guarded','guarded_full']

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise RuntimeError('Already frozen')
    assert not (ex.OUT/'results.json').exists(),'Freeze before the corrected test results'
    cfg=b.read_json(ex.OUT/'protocol.json');ex.verify(cfg)
    scenarios={'clean':[0.,0.,0.,0.]}
    for level,sigma in [('small',[.02,float(np.deg2rad(2)),.02,float(np.deg2rad(2))]),('larger',[.05,float(np.deg2rad(5)),.05,float(np.deg2rad(5))])]:
        for index,name in enumerate(['d','psi','v','omega']):
            row=[0.,0.,0.,0.];row[index]=sigma[index];scenarios[level+'_'+name]=row
        scenarios[level+'_joint']=sigma
    b.write_json(OUT/'protocol.json',dict(status='Exploratory sensitivity scenarios fixed before corrected-query test evaluation; no measured sensor model or fresh independent data.',parent_sha256=b.digest(ex.OUT/'protocol.json'),script_sha256=b.digest(__file__),fold='temporal',models=NAMES,seeds=cfg['seeds'],noise_seeds=[92801,92802,92803],scenarios=scenarios,units=['m','radian','m/s','radian/s'],distribution='Independent zero-mean Gaussian initial d/psi/v/omega error; one shared standard-normal array per noise seed across models, training seeds and scenarios; no clipping. Images, preview, future actions, targets and ego reference unchanged.',semantics='psi is road-relative heading estimation error. This does not rotate the true ego evaluation frame or simulate global-pose localization error. It changes the road-relative input and the models own anchored readout consistently.',scope='Only the frozen temporal split. Report both sources. No retraining, checkpoint reselection, family selection or inferred physical sensor tolerance.',statistics='Average 3 noise realizations within each training seed, then mean and sample SD across 3 training seeds. Neither windows nor noise realizations are independent driving sessions.',diagnostics='Nonfinite windows, endpoint error >2 m (descriptive threshold, not safety failure), and full-step Frenet denominator <=0 where available.'))
    print('Initialization protocol frozen: 3 models, 3 seeds, 10 noisy scenarios x 3 noise draws plus clean',flush=True)

def verify(cfg):
    assert b.digest(__file__)==cfg['script_sha256']
    assert b.digest(ex.OUT/'protocol.json')==cfg['parent_sha256']
    ex.verify(b.read_json(ex.OUT/'protocol.json'))

@torch.no_grad()
def evaluate(cfg):
    verify(cfg);ex.configure();torch.set_num_threads(2)
    assert b.read_json(ex.OUT/'audit.json')['status']=='PASS'
    if (OUT/'results.json').exists():raise RuntimeError('Refusing result overwrite')
    parent=b.read_json(ex.OUT/'protocol.json');rec=ex.CorrectedRecords(parent['splits']['temporal']['test']);curves={};results={};diagnostics={};original=np.load(ex.OUT/'curves.npz');window_manifest=None;noises=None;checkpoint_hashes={}
    for seed in cfg['seeds']:
        d=ex.CK/f'temporal_seed{seed}';mark=b.read_json(d/'COMPLETE.json')
        for n,h in mark['files'].items():assert b.digest(d/n)==h
        ex.encode(rec,d/'independent');stats=dict(np.load(d/'independent/normalizers.npz'));limits=b.read_json(d/'limits.json')['limits'];ds=b.WindowData(rec,100,4)
        windows=[dict(file=rec.records[e]['name'],t=int(t)) for e,t in ds.idx];assert windows==b.read_json(d/'evaluation_windows.json')
        if window_manifest is None:
            window_manifest=windows;noises={str(n):np.random.default_rng(n).standard_normal((len(ds),4)).astype(np.float32) for n in cfg['noise_seeds']}
        assert windows==window_manifest
        for name in cfg['models']:
            m=ex.make(name,stats,d/'independent',limits);m.load_state_dict(torch.load(d/(name+'.pt'),weights_only=False)['model']);m.eval();checkpoint_hashes[f'seed{seed}/{name}']=b.digest(d/(name+'.pt'))
            for scenario,sigma in cfg['scenarios'].items():
                for noise_seed in ([0] if scenario=='clean' else cfg['noise_seeds']):
                    delta=np.zeros((len(ds),4),np.float32) if scenario=='clean' else noises[str(noise_seed)]*np.asarray(sigma,np.float32)
                    monitor=ex.c.Monitor();hook=m.register_forward_pre_hook(monitor.hook,with_kwargs=True);err=[];start=0
                    for z,a,y,k,p in b.loader(ds,128):
                        z=z.clone();z[:,1:]+=torch.tensor(delta[start:start+len(z)]);z[:,2]=torch.atan2(z[:,2].sin(),z[:,2].cos())
                        pred=m.rollout(z,a,k,p);err.append(torch.linalg.vector_norm(pred[:,:,:2]-y[:,:,:2],dim=-1).numpy());start+=len(z)
                    hook.remove();errors=np.concatenate(err);key=f'seed{seed}/{name}/{scenario}/noise{noise_seed}'
                    if scenario=='clean':assert np.allclose(errors,original[f'temporal/seed{seed}/{name}'],rtol=2e-4,atol=2e-5)
                    assert np.allclose(errors[:,0],0,atol=1e-7)
                    curves[key]=errors;results[key]=b.metrics(errors);row=monitor.result();row['endpoint_over_2m_fraction']=float((errors[:,100]>2).mean());diagnostics[key]=row
                print('DONE',seed,name,scenario,flush=True)
    np.savez_compressed(OUT/'curves.npz',**curves);np.savez_compressed(OUT/'standard_normal_draws.npz',**noises);b.write_json(OUT/'evaluation_windows.json',window_manifest);b.write_json(OUT/'results.json',dict(protocol_sha256=b.digest(OUT/'protocol.json'),checkpoint_sha256=checkpoint_hashes,results=results,diagnostics=diagnostics))
    audit(cfg);report(cfg)

def audit(cfg):
    verify(cfg);obj=b.read_json(OUT/'results.json');assert obj['protocol_sha256']==b.digest(OUT/'protocol.json');curves=np.load(OUT/'curves.npz');draws=np.load(OUT/'standard_normal_draws.npz');windows=b.read_json(OUT/'evaluation_windows.json');sub={}
    assert len(obj['results'])==279 and set(curves.files)==set(obj['results'])
    for n in cfg['noise_seeds']:assert np.array_equal(draws[str(n)],np.random.default_rng(n).standard_normal((len(windows),4)).astype(np.float32))
    for key,h in obj['checkpoint_sha256'].items():
        seed,name=key.split('/');assert b.digest(ex.CK/f'temporal_{seed}'/(name+'.pt'))==h
    for key,row in obj['results'].items():
        e=curves[key];assert e.shape==(len(windows),101) and np.allclose(e[:,0],0,atol=1e-7) and b.metrics(e)==row
        for source in ['traj1_64x64','traj2_64x64']:
            mask=np.array([b.PAT.fullmatch(w['file'])[1]==source for w in windows]);sub[source+'/'+key]=b.metrics(e[mask])
    b.write_json(OUT/'per_source_results.json',dict(results=sub));b.write_json(OUT/'audit.json',dict(status='PASS',rows=279,checks=['unchanged frozen parent, code and checkpoints','all declared scenarios, models and seeds','shared reproducible noise draws','zero initial xy error','clean predictions reproduce parent','all metrics recalculated, both sources retained']))

def report(cfg):
    data=b.read_json(OUT/'results.json');src=b.read_json(OUT/'per_source_results.json')['results'];summaries={};lines=['# 初始状态敏感性复核','','只评价时间留出，不重训或挑选 checkpoint。误差以真实初始 ego 坐标评估；psi 是道路相对航向输入误差，并非全局航向定位误差模拟。道路图像、预览和未来动作保持不变。',
    'small 的标准差依次为 d=0.02 m、psi=2°、v=0.02 m/s、omega=2°/s；larger 为 0.05 m、5°、0.05 m/s、5°/s。分别和联合施加独立零均值高斯扰动。这些是预定情景，不是测得的传感器噪声，也不代表可保证的容错范围。',
    '每个训练种子先平均三个噪声实现，再计算三个训练种子的均值 ± 样本标准差。它不是独立采集数据的置信区间。','']
    for source in ['pooled','traj1_64x64','traj2_64x64']:
        lines += [f'## {source}','','| 情景 | 运动学 E100 | 有界 Cartesian E100 | guarded full E100 |','|---|---:|---:|---:|']
        for scenario in cfg['scenarios']:
            cells=[]
            for model in cfg['models']:
                row={};noise=[0] if scenario=='clean' else cfg['noise_seeds']
                for metric in ['E100','ADE']:
                    by_seed=[]
                    for seed in cfg['seeds']:
                        values=[(data['results'] if source=='pooled' else src)[('' if source=='pooled' else source+'/')+f'seed{seed}/{model}/{scenario}/noise{n}'][metric] for n in noise]
                        by_seed.append(None if None in values else float(np.mean(values)))
                    row[metric]=dict(mean=None if None in by_seed else float(np.mean(by_seed)),sd=None if None in by_seed else float(np.std(by_seed,ddof=1)),by_seed=by_seed)
                summaries[f'{source}/{scenario}/{model}']=row;v=row['E100'];cells.append('nonfinite' if v['mean'] is None else f"{v['mean']:.3f} ± {v['sd']:.3f}")
            lines.append('| '+scenario+' | '+' | '.join(cells)+' |')
    lines+=['','完整 E100/ADE、非有限窗口、超过 2 m 的终点误差比例及可用的全步分母监测见 results.json、summary.json；2 m 仅是描述性阈值，不是安全失效定义。所有情景均保留。','','模型名称：GPT-6（Codex）。']
    b.write_json(OUT/'summary.json',summaries);(OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','evaluate','audit','report']);args=ap.parse_args()
    if args.stage=='prepare':prepare()
    else:globals()[args.stage](b.read_json(OUT/'protocol.json'))
if __name__=='__main__':main()
