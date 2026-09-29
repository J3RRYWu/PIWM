"""Plot completed exploratory results without training or selecting models."""
from pathlib import Path
import sys,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_frenet_diagnostic as ex

def main():
    cfg=ex.base.read_json(ex.OUT/'protocol.json');res=ex.base.read_json(ex.OUT/'results.json')['results']
    colors={'legacy_full':'#9a3412','guarded_full':'#4338ca','legacy_shape':'#d97706','guarded_shape':'#059669'}
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for i,f in enumerate(cfg['splits']):
        for j,s in enumerate(cfg['seeds']):
            ax=axes[i,j]
            for n,c in colors.items():
                h=ex.base.read_json(ex.CK/f'{f}_seed{s}'/(n+'_history.json'))
                v=np.array([np.nan if x['val_E100'] is None else x['val_E100'] for x in h])
                ax.plot(np.arange(1,41),v,label=n,color=c,lw=1.6)
                best=np.nanargmin(v);ax.scatter(best+1,v[best],color=c,s=22)
            ax.set(title=f'{f}, seed {s}',xlabel='Epoch',ylabel='Validation E100 (m)',yscale='log');ax.grid(alpha=.2)
            if i==0 and j==0:ax.legend(fontsize=8)
    fig.suptitle('Validation histories: dots mark selected checkpoints')
    fig.savefig(ex.OUT/'training_stability.png',dpi=170);fig.savefig(ex.OUT/'training_stability.pdf');plt.close(fig)
    old=ex.base.read_json(ex.OLD/'test_results.json')['results'];names=list(ex.VARIANTS)+['kinematic','cartesian_no_road']
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,f in zip(axes,cfg['splits']):
        for i,n in enumerate(names):
            values=np.array([res[f'{f}/seed{s}/{n}']['E100'] if n in ex.VARIANTS else old[f'{f}/seed{s}/{n}/normal']['E100'] for s in cfg['seeds']],dtype=float)
            if not np.isfinite(values).all():ax.text(0,i,'nonfinite');continue
            ax.barh(i,values.mean(),color='#4338ca' if n=='guarded_full' else '#94a3b8',height=.65)
            ax.scatter(values,np.full(3,i),c='black',s=18,zorder=3)
        ax.set_yticks(range(len(names)),names);ax.invert_yaxis();ax.set(title=f'Previously examined source: {f}',xlabel='E100 (m), lower is better');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Exploratory retraining: mean and individual training seeds')
    fig.savefig(ex.OUT/'retrained_ablations.png',dpi=170);fig.savefig(ex.OUT/'retrained_ablations.pdf');plt.close(fig)
    print('Saved diagnostic figures.')

if __name__=='__main__':main()
