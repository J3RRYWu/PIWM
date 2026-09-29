"""Summarize paired segments and complete temporal figures without model selection."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def analyze(suite,primary):
    d=R/'reports'/suite;assert read(d/'audit.json')['status']=='PASS'
    c=np.load(d/'curves.npz');names=read(d/'protocol.json')['variants'];w=read(R/'checkpoints'/suite/'temporal_seed0/evaluation_windows.json')
    files=sorted({x['file'] for x in w});base=np.load(R/'reports/temporal_holdout/curves.npz')
    averaged={n:np.stack([c[f'temporal/seed{s}/{n}'] for s in range(3)]).mean(0) for n in names}
    averaged['kinematic']=np.stack([base[f'temporal/seed{s}/kinematic'] for s in range(3)]).mean(0)
    rows=[]
    for name in files:
        mask=np.array([x['file']==name for x in w]);rows.append(dict(file=name,windows=int(mask.sum()),E100={k:float(v[mask,-1].mean()) for k,v in averaged.items()}))
    comparisons={}
    for control in averaged:
        if control==primary:continue
        diff=np.array([row['E100'][primary]-row['E100'][control] for row in rows])
        comparisons[control]=dict(segments_improved=int((diff<0).sum()),segments_total=len(rows),equal_segment_mean_difference=float(diff.mean()),worst_segment_difference=float(diff.max()),pooled_relative_reduction=float(1-averaged[primary][:,-1].mean()/averaged[control][:,-1].mean()))
    obj=dict(scope='Descriptive paired comparison, first averaging training seeds then each original test segment. Segments/windows are correlated; no p-values or independent-session confidence intervals.',primary=primary,segments=rows,comparisons=comparisons)
    (d/'paired_segments.json').write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
    lines=['# 配对片段与数值状态复核','','先对三个种子的逐窗口误差取平均，再按原始测试片段聚合。片段和窗口存在相关性，下表不是独立样本显著性检验。','','| 对照 | 主模型改善的片段数 | 等片段平均差（主模型−对照，m） | 全窗口 E100 相对降低 |','|---|---:|---:|---:|']
    for name,row in comparisons.items():lines.append(f"| {name} | {row['segments_improved']}/{row['segments_total']} | {row['equal_segment_mean_difference']:.3f} | {100*row['pooled_relative_reduction']:.1f}% |")
    diag=read(d/'results.json')['diagnostics'];lines+=['','数值监测只覆盖完整时间步，midpoint 内部阶段不在其中。shape-only 把动力学曲率清零，其分母为 1 不代表真实曲线图没有问题。Cartesian/kinematic 未挂相同 Frenet 监测，samples=0 是不适用。','', '| 模型 | 三种子最大无效图比例 | 三种子最大近奇异比例 |','|---|---:|---:|']
    for name in names:
        rr=[diag[f'temporal/seed{s}/{name}'] for s in range(3)]
        if not any(r['samples'] for r in rr):continue
        lines.append(f"| {name} | {100*max(r['invalid_chart_fraction'] for r in rr):.4f}% | {100*max(r['near_singular_fraction'] for r in rr):.4f}% |")
    lines+=['','模型名称：GPT-6（Codex）。'];(d/'PAIRED_AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(suite,comparisons)

def figure():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,1,figsize=(7.2,7.4),layout='constrained');x=np.arange(101)/22
    defs=[('temporal_holdout','Independent curvature and shape heads',[('guarded_full','Guarded, full'),('guarded_shape','Guarded, shape only'),('geometry_midpoint','Consistent, midpoint'),('cartesian_road','Cartesian, road'),('cartesian_no_road','Cartesian, no road'),('kinematic','Calibrated kinematics')]),('coupled_road','Single curvature representation and arc decoder',[('arc_full','Coupled arc, full'),('arc_shape','Coupled arc, shape only'),('arc_independent_encoder','Arc, independent encoder'),('cartesian_road','Cartesian, coupled road'),('kinematic','Calibrated kinematics')])]
    colors=['#4338ca','#059669','#c2410c','#0369a1','#9ca3af','#111827']
    for ax,(suite,title,entries) in zip(axes,defs):
        curves=np.load(R/'reports'/suite/'curves.npz');base=np.load(R/'reports/temporal_holdout/curves.npz')
        for j,(name,label) in enumerate(entries):
            store=base if name=='kinematic' else curves;color='#111827' if name=='kinematic' else colors[j]
            ys=np.stack([store[f'temporal/seed{s}/{name}'].mean(0) for s in range(3)]);mean=ys.mean(0);sd=ys.std(0,ddof=1)
            ax.plot(x,mean,color=color,label=label,ls='--' if name=='kinematic' else '-',lw=1.8);ax.fill_between(x,np.maximum(0,mean-sd),mean+sd,color=color,alpha=.10,linewidth=0)
        ax.set(title=title,xlabel='Forecast time (s)',ylabel='Mean position error (m)',xlim=(0,100/22),ylim=(0,None));ax.legend(fontsize=9.5,frameon=False,loc='upper left');ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    fig.savefig(R/'reports/coupled_road/temporal_comparison.png',dpi=160);fig.savefig(R/'Jounral_PIWM/imgs/fig_temporal_comparison.pdf');plt.close(fig)

if __name__=='__main__':
    analyze('temporal_holdout','guarded_full')
    if (R/'reports/coupled_road/audit.json').exists():analyze('coupled_road','arc_full');figure()
