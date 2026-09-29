"""Plot frozen test arrays; does not select/retrain any model."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/controlled_holdout'

def main():
    cfg=json.loads((OUT/'protocol.json').read_text())
    arrays=np.load(OUT/'test_curves.npz')
    folds=list(cfg['splits']);seeds=cfg['seeds']
    colors={'frenet':'#4338CA','cartesian_road':'#C2410C','cartesian_no_road':'#059669','kinematic':'#64748B','constant_motion':'#A855F7'}
    labels={'frenet':'Frenet (map-free)','cartesian_road':'Cartesian + road','cartesian_no_road':'Cartesian, no road','kinematic':'Calibrated kinematic','constant_motion':'Constant motion'}
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,fold in enumerate(folds):
        ax=axes[0,col]
        for model in colors:
            curves=np.stack([arrays[f'{fold}/seed{s}/{model}/normal'].mean(0) for s in seeds])
            if not np.isfinite(curves).all():continue
            mean=curves.mean(0);std=curves.std(0,ddof=1)
            ax.plot(np.arange(101)/22,mean,label=labels[model],color=colors[model],lw=2)
            ax.fill_between(np.arange(101)/22,np.maximum(0,mean-std),mean+std,color=colors[model],alpha=.12)
        ax.set(title=f'Test: {cfg["splits"][fold]["test_source"]}',xlabel='Prediction horizon (s)',ylabel='Mean position error (m)')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax=axes[1,col]
        modes=['normal','kappa_zero','kappa_mean','kappa_permuted','zero','train_mean','permuted','legacy_hold_last']
        tick=['Normal','Zero curvature','Mean curvature','Shuffled curvature','Straight road','Mean road','Shuffled road','Endpoint clamp']
        for i,mode in enumerate(modes):
            vals=np.array([arrays[f'{fold}/seed{s}/frenet/{mode}'][:,100].mean() for s in seeds])
            if np.isfinite(vals).all():
                ax.barh(i,vals.mean(),color='#4338CA' if mode=='normal' else '#94A3B8',height=.65)
                ax.scatter(vals,np.full(len(vals),i),c='black',s=15,zorder=3)
            else:ax.text(.05,i,'nonfinite',va='center')
        ax.set_yticks(range(len(tick)),tick);ax.invert_yaxis();ax.set_xlabel('Endpoint error at 100 steps (m)');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Source-held-out diagnostic: fixed protocol, 3 seeds per direction',fontsize=13)
    fig.savefig(OUT/'controlled_holdout.png',dpi=180);plt.close(fig)
    print(OUT/'controlled_holdout.png')

if __name__=='__main__':main()
