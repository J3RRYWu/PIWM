"""Independent evidence checks for the corrected-query suite; no training changes."""
from pathlib import Path
import sys,argparse
import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_corrected_query as ex
b=ex.b
from arc_length_track import ArcLengthTrack

def inputs(cfg):
    ex.verify(cfg)
    track=ArcLengthTrack(np.load(b.DATA/'_meta/track.npz')['centers'])
    starts=np.linspace(0,track.length,100,endpoint=False);convergence={}
    for step in [.5,.25,.1,.05,.01]:
        off=np.arange(0,4.5+step/2,step);pos,h,k=track.evaluate(starts[:,None]+off)
        theta=cumulative_trapezoid(k,off,axis=1,initial=0)
        rec=cumulative_trapezoid(np.stack([np.cos(theta),np.sin(theta)],-1),off,axis=1,initial=0)
        delta=pos-pos[:,:1];ang=h[:,0:1]
        target=np.stack([delta[...,0]*np.cos(ang)+delta[...,1]*np.sin(ang),-delta[...,0]*np.sin(ang)+delta[...,1]*np.cos(ang)],-1)
        convergence[str(step)]=dict(coordinate_RMSE_m=float(np.sqrt(np.mean((rec-target)**2))),endpoint_median_m=float(np.median(np.linalg.norm(rec[:,-1]-target[:,-1],axis=-1))))
    assert convergence['0.01']['coordinate_RMSE_m']<.001
    split_rows={};parent=b.read_json(b.OUT/'protocol.json')
    for fold,split in cfg['splits'].items():
        sets=[set(split[k]) for k in ['train','validation','test','purged']]
        assert set.union(*sets)==set(parent['files'])
        assert all(not sets[i]&sets[j] for i in range(4) for j in range(i))
        rows={}
        for source in ['traj1_64x64','traj2_64x64']:
            by={key:sorted([tuple(map(int,b.PAT.fullmatch(n).group(2,3))) for n in split[key] if b.PAT.fullmatch(n)[1]==source]) for key in ['train','validation','test']}
            gap={}
            for left,right in [('train','validation'),('validation','test')]:
                if by[left] and by[right]:
                    distance=min(x[0] for x in by[right])-max(x[1] for x in by[left]);assert distance>=115
                    gap[left+'_'+right]=distance
            if fold!='temporal' and by['test']:assert not by['train'] and not by['validation']
            rows[source]=dict(counts={k:len(v) for k,v in by.items()},gaps_frames=gap)
        split_rows[fold]=rows
    out=dict(status='PASS',scope='Map discretization and raw frame partitions; no test performance.',protocol_sha256=b.digest(ex.OUT/'protocol.json'),map_integration_convergence=convergence,partitions=split_rows,interpretation='Continuous labels are geometrically consistent; ten pointwise curvature knots cannot accurately reproduce the sharp continuously varying curve. Coarse decoder mismatch remains. No smoothing, knot count or training change was selected from these diagnostics.',script_sha256=b.digest(__file__))
    b.write_json(ex.OUT/'input_reaudit.json',out)
    lines=['# 修正标签：训练前诊断与实验边界','','连续曲线通过单位速度、切向导数、密集曲率积分及投影检查。稀疏曲率表示仍有离散误差；连续标签一致并不意味着 10 个曲率点能精确重建道路。','','| 积分间隔 (m) | 地图坐标 RMSE (m) | 端点误差中位数 (m) |','|---|---:|---:|']
    for step,row in convergence.items():lines.append(f"| {step} | {row['coordinate_RMSE_m']:.6f} | {row['endpoint_median_m']:.6f} |")
    lines+=['','本表在地图上均匀取 100 个起点，以梯形积分计算，属于数值收敛诊断；不是测试轨迹预测成绩，也不是训练中的分段圆弧解码器成绩。实际解码器的训练／验证兼容性见 consistent_road_labels/manifest.json。',
            '保留已冻结的 0.5 m 表示和所有模型，不按此诊断修改训练设置。此轮检验标签修正、查询机制和有界 Cartesian 对照，不能预先将几何耦合称为性能改进。',
            '三划分与原始帧隔离逐项核对通过；来源转移结果与全部种子必须报告。独立与耦合视觉模型分别训练，二者之差不能归为单一动态模块的作用。',
            'guarded_fixed_query 同时改变几何递推和神经特征，因此只作为故意错配的控制；guarded_static_context 才隔离神经特征是否随进度查询。',
            '后续推断只限同赛道、记录动作条件下的离线预测；没有新采集数据、闭环实验或真实传感器噪声测量。','','模型名称：GPT-6（Codex）。']
    (ex.OUT/'PREFIT_DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Input reaudit PASS',convergence,flush=True)

def perception(cfg):
    results={}
    for fold,split in cfg['splits'].items():
        tr=ex.CorrectedRecords(split['train']);va=ex.CorrectedRecords(split['validation']);td=b.PerceptionData(tr,2);vd=b.PerceptionData(va,2)
        mean=np.stack([td[i][1].numpy() for i in range(len(td))]).mean(0)
        all_targets=[]
        for r in tr.records:
            sh=r['shape'].copy();sh[:,:,0]-=b.OFFSETS;all_targets.append(np.concatenate([r['kappa'],sh.reshape(len(sh),-1)],-1))
        scale=np.concatenate(all_targets).std(0).clip(.05)
        z=np.concatenate([r['z'] for r in tr.records]);sm=z.mean(0);ss=z.std(0).clip(.01);sm[0]=0;ss[0]=1
        kp=np.concatenate([r['kappa'] for r in tr.records]);limits=np.maximum(.01,np.quantile(np.concatenate([abs(np.diff(r['z'][:,3:5],axis=0)) for r in tr.records]),.995,axis=0))
        for seed in cfg['seeds']:
            job=ex.CK/f'{fold}_seed{seed}';assert np.allclose(limits,b.read_json(job/'limits.json')['limits'])
            for kind in cfg['encoders']:
                d=job/kind;saved=torch.load(d/'perception.pt',weights_only=False)
                assert np.array_equal(mean,saved['mean']) and np.array_equal(scale,saved['scale'])
                stats=np.load(d/'stats.npz');assert np.array_equal(sm,stats['state_mean']) and np.array_equal(ss,stats['state_std']);assert np.isclose(kp.mean(),stats['kappa_mean']) and np.isclose(max(kp.std(),.01),stats['kappa_std'])
                b.DEVICE=b.VISION_DEVICE;m=ex.load_encoder(d,kind=='coupled');m.eval();total=0.;n=0;pred=[];sc=torch.tensor(scale,device=b.DEVICE)
                with torch.no_grad():
                    for x,y in b.loader(vd,64):
                        k,p=m.forward_both(x.to(b.DEVICE));prediction=torch.cat([k,p.flatten(1)],-1);err=((prediction-y.to(b.DEVICE))/sc).square().mean();assert torch.isfinite(err)
                        total+=float(err)*len(x);n+=len(x);pred.append(prediction.cpu().numpy())
                measured=total/n;assert np.isclose(measured,saved['val_loss'],rtol=1e-5,atol=1e-7)
                b.encode(m,tr);context=np.concatenate([np.concatenate([r['z'][14:,1:3],r['pred_kappa'][14:],r['pred_shape'][14:].reshape(-1,20)],-1) for r in tr.records]);norm=np.load(d/'normalizers.npz')
                assert np.allclose(context.mean(0),norm['context_mean'],rtol=1e-5,atol=1e-6) and np.allclose(context.std(0).clip(.05),norm['context_std'],rtol=1e-5,atol=1e-6)
                key=f'{fold}/seed{seed}/{kind}';results[key]=dict(validation_loss=measured,epoch=saved['epoch'],curvature_std=float(np.concatenate(pred)[:,:10].std(0).mean()),parameters=sum(p.numel() for p in m.parameters()),checkpoint_sha256=b.digest(d/'perception.pt'));print(key,results[key],flush=True);del m
    b.DEVICE='cpu';b.write_json(ex.OUT/'perception_reaudit.json',dict(status='PASS',scope='All training means/scales/context statistics and 18 validation perception losses recalculated; no refitting.',results=results,script_sha256=b.digest(__file__)))

def checkpoints(cfg):
    curves=np.load(ex.OUT/'curves.npz');rows={};ex.configure();torch.set_num_threads(2)
    for fold,split in cfg['splits'].items():
        rec=ex.CorrectedRecords(split['test']);rc=ex.clone(rec)
        for seed in cfg['seeds']:
            d=ex.CK/f'{fold}_seed{seed}';ex.encode(rec,d/'independent');ex.encode(rc,d/'coupled',True);limits=b.read_json(d/'limits.json')['limits']
            for name in ex.NAMES:
                is_c=name=='arc_full';enc=d/('coupled' if is_c else 'independent');stats=dict(np.load(enc/'normalizers.npz'));ds=b.WindowData(rc if is_c else rec,100,4);indices=np.unique(np.linspace(0,len(ds)-1,24,dtype=int));subset=torch.utils.data.Subset(ds,indices)
                m=ex.make(name,stats,enc,limits);saved=torch.load(d/(name+'.pt'),weights_only=False);m.load_state_dict(saved['model']);assert saved['parameters']==sum(p.numel() for p in m.parameters())
                err=b.evaluate_model(m,subset,batch=24);key=f'{fold}/seed{seed}/{name}';delta=float(abs(err-curves[key][indices]).max());assert np.isfinite(err).all() and np.allclose(err,curves[key][indices],rtol=2e-4,atol=2e-5),(key,delta)
                if hasattr(m,'core') and hasattr(m.core,'kappa_grid'):
                    m.core.kappa_grid.fill_(12345);m.core.total_len=.001;changed=b.evaluate_model(m,subset,batch=24);assert np.array_equal(err,changed),key
                rows[key]=dict(windows=len(indices),max_absolute_difference_m=delta,parameters=saved['parameters'],map_buffer_independence=True if hasattr(m,'core') and hasattr(m.core,'kappa_grid') else 'not applicable');print(key,delta,flush=True)
    b.write_json(ex.OUT/'checkpoint_spotcheck.json',dict(status='PASS',scope='24 evenly spaced windows per all 72 checkpoints, plus map-buffer perturbation; independent recomputation by same agent, not external replication.',results=rows,script_sha256=b.digest(__file__)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['inputs','perception','checkpoints','all']);args=ap.parse_args();ex.configure();torch.set_num_threads(2);cfg=b.read_json(ex.OUT/'protocol.json');ex.verify(cfg)
    for stage in (['inputs','perception','checkpoints'] if args.stage=='all' else [args.stage]):globals()[stage](cfg)
if __name__=='__main__':main()
