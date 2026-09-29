"""Clean small multiples from the same five-fold cache as the article table.
Bands preserve the fold min/max; only one PIWM band is drawn per panel.
The reference in every panel is Vid2Param trained with clean labels.
"""
from pathlib import Path
import os,sys,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from fig_main_cv import _ensure_latex
from paper_style import apply as apply_style, FIG_W,COL
CURVES=ROOT/'reports/matrix/folds_table_symmetric_curves.npz'

def main():
    apply_style(usetex=_ensure_latex());z=np.load(CURVES)
    folds=sorted({int(k.split('_',1)[0][1:]) for k in z.files});assert len(folds)==5
    fig,axes=plt.subplots(1,3,figsize=(FIG_W,2.7),sharex=True,sharey=True)
    fig.subplots_adjust(left=.105,right=.98,bottom=.215,top=.72,wspace=.16)
    ref=np.stack([z[f'f{f}_V2P'] for f in folds]).mean(0)
    for ax,row,delta in zip(axes,['ours-a','ours-a_d5','ours-a_d10'],['0','5\\%','10\\%']):
        C=np.stack([z[f'f{f}_{row}'] for f in folds]);assert np.isfinite(C).all()
        steps=np.arange(C.shape[1])
        ax.fill_between(steps,C.min(0),C.max(0),color=COL['Frenet'],alpha=.16,lw=0)
        ax.plot(steps,C.mean(0),color=COL['Frenet'],lw=1.8,label='PIWM-Frenet')
        ax.plot(steps,ref,color=COL['V2P'],ls='--',lw=1.4,label='Vid2Param (clean labels)')
        ax.set_title('$\\delta='+delta+'$',fontsize=10,pad=8)
        ax.set_xlim(0,100);ax.set_ylim(0,.78);ax.set_xticks([0,50,100])
        ax.tick_params(labelsize=8.5)
        print(row,C[:,-1].mean(),C[:,-1].std(ddof=1))
    axes[0].set_ylabel('Position error (m)')
    fig.text(.54,.055,'Rollout step',ha='center',fontsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(),ncol=2,frameon=False,loc='upper center',bbox_to_anchor=(.54,.99),fontsize=9)
    for ext in ['pdf','png']:fig.savefig(ROOT/'figures'/('fig_delta_cv.'+ext),bbox_inches=None)
    shutil.copyfile(ROOT/'figures/fig_delta_cv.pdf',ROOT/'Jounral_PIWM/imgs/fig_delta_cv.pdf')
    plt.close(fig)
if __name__=='__main__':main()
