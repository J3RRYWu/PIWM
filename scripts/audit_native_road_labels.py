"""Audit legacy map parameterization and label compatibility without test windows."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_holdout as b
from baselines.unit_arc_geometry import arc_points

def summary(x):
    x=np.asarray(x)
    assert np.isfinite(x).all()
    return dict(n=int(x.size),mean=float(x.mean()),median=float(np.median(x)),p95=float(np.quantile(x,.95)),max=float(x.max()))

def main():
    torch.set_num_threads(2)
    cfg=b.read_json(b.ROOT/'reports/temporal_holdout/protocol.json')
    tr=np.load(b.DATA/'_meta/track.npz');cen=tr['centers'].astype(np.float64);ds=float(tr['grid_ds']);h=tr['heading'].astype(np.float64)
    derivative=(np.roll(cen,-1,axis=0)-np.roll(cen,1,axis=0))/(2*ds)
    g=np.linalg.norm(derivative,axis=1)
    angle=np.arctan2(derivative[:,1],derivative[:,0])
    def dh(theta):
        diff=np.roll(theta,-1)-np.roll(theta,1)
        return np.arctan2(np.sin(diff),np.cos(diff))/(2*ds)
    kgeom=dh(angle)/g.clip(1e-8)
    kh=dh(h)/g.clip(1e-8)
    out=dict(scope='Persisted map metadata plus temporal train/validation windows only; no training or test-window use.',track_sha256=b.digest(b.DATA/'_meta/track.npz'),protocol_sha256=b.digest(b.ROOT/'reports/temporal_holdout/protocol.json'),map=dict(points=len(cen),nominal_length=float(tr['total_len']),polygon_length=float(np.linalg.norm(np.roll(cen,-1,axis=0)-cen,axis=1).sum()),native_speed=summary(g),native_speed_abs_error=summary(abs(g-1)),native_speed_outside_20percent_fraction=float(np.mean(abs(g-1)>.2)),stored_curvature_vs_center_derivative=summary(abs(tr['kappa']-kgeom)),stored_curvature_vs_heading_derivative=summary(abs(tr['kappa']-kh))),partitions={})
    for part in ['train','validation']:
        rec=b.Records(cfg['splits']['temporal'][part]);idx=[(i,t) for i,r in enumerate(rec.records) for t in range(14,len(r['z'])-100,4)]
        kp=np.stack([rec.records[i]['kappa'][t] for i,t in idx]);p=np.stack([rec.records[i]['shape'][t] for i,t in idx])
        with torch.no_grad():recon=arc_points(torch.tensor(kp)).numpy()
        err=np.linalg.norm(recon-p,axis=-1)
        out['partitions'][part]=dict(windows=len(idx),arc_reconstruction_of_stored_curvature_vs_stored_shape=summary(err),endpoint_shape_difference=summary(err[:,-1]),coordinate_RMSE=float(np.sqrt(np.mean((recon-p)**2))))
    out['interpretation']=['The persisted grid parameter is nominal distance before smoothing, not exactly arc length of the final centerline.', 'Curvature is additionally smoothed separately from centerline/heading; identities need not hold exactly.', 'Arc reconstruction mismatch also includes 0.5-m curvature discretization and piecewise-constant approximation; this audit does not isolate all causes.', 'A constrained unit-speed encoder trained against both legacy targets faces representation mismatch. This does not prove that label mismatch caused any test result.', 'Existing frozen experiments and all results are retained; no labels or checkpoints are modified.']
    dest=b.ROOT/'reports/road_geometry';b.write_json(dest/'native_label_audit.json',out)
    lines=['# 道路标签几何一致性审计','','范围：持久化地图与时间划分的训练／验证窗口；不使用测试窗口，不修改标签。','',f"名义周长 {out['map']['nominal_length']:.3f} m；平滑后折线长度 {out['map']['polygon_length']:.3f} m。",f"原生 0.05 m 网格的 |dC/ds|-1 绝对偏差中位数 {np.median(abs(g-1)):.4f}，95 分位 {np.quantile(abs(g-1),.95):.4f}。",'','| 划分 | 窗口 | 曲率积分与原形状坐标 RMSE (m) | 4.5 m 端点差中位数 (m) |','|---|---:|---:|---:|']
    for name,row in out['partitions'].items():lines.append(f"| {name} | {row['windows']} | {row['coordinate_RMSE']:.4f} | {row['endpoint_shape_difference']['median']:.4f} |")
    lines+=['','上述差异包含中心线平滑后没有重新弧长参数化、曲率单独平滑、0.5 m 离散和分段常曲率近似的共同作用，不能把全部差异归因于某一项。它说明旧标签不精确满足单一单位弧长曲线约束，不能证明某项测试误差由此造成。已有实验全部保留。','','模型名称：GPT-6（Codex）。']
    (dest/'NATIVE_LABEL_AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(out)
if __name__=='__main__':main()
