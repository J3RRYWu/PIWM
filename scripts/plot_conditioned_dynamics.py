"""Plot all prescribed conditioned-dynamics variants plus fixed controls."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];D=R/'reports/conditioned_dynamics'
assert json.loads((D/'audit.json').read_text())['status']=='PASS'
a=np.load(D/'curves.npz');b=np.load(R/'reports/controlled_holdout/test_curves.npz')
names=[('guarded_full','Guarded, full','#4338ca'),('guarded_shape','Guarded, shape only','#059669'),('geometry_midpoint','Consistent, midpoint','#c2410c'),('cartesian_road','Cartesian, road','#0369a1'),('kinematic','Calibrated kinematics','#111827'),('cartesian_no_road','Cartesian, no road','#9ca3af')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(7.2,3.6),layout='constrained')
for ax,fold,title in zip(axes,['outer0','outer1'],['A: first recording held out','B: second recording held out']):
    for name,label,color in names:
        old=name in ['kinematic','cartesian_no_road'];store=b if old else a;suffix='/normal' if old else ''
        curves=np.stack([store[f'{fold}/seed{s}/{name}{suffix}'].mean(0) for s in range(3)])
        assert np.isfinite(curves).all()
        mean=curves.mean(0);sd=curves.std(0,ddof=1);x=np.arange(101)/22
        ax.plot(x,mean,label=label,color=color,lw=1.7,ls='--' if old else '-')
        ax.fill_between(x,np.maximum(0,mean-sd),mean+sd,color=color,alpha=.09,linewidth=0)
    ax.set(title=title,xlabel='Forecast time (s)',ylabel='Mean position error (m)',xlim=(0,100/22),ylim=(0,None))
    ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
axes[0].legend(fontsize=9.5,loc='upper left',frameon=False)
fig.savefig(D/'conditioned_comparison.png',dpi=190)
fig.savefig(R/'Jounral_PIWM/imgs/fig_conditioned_comparison.pdf')
plt.close(fig)
print('Saved complete six-model comparison; shading is seed sample SD, not a confidence interval.')
