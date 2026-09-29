"""Conference results, reconstructed layout from exact embedded plot images.
Run prepare_conference_assets.py first. No numeric data are estimated from pixels.
Axes use the original published limits, including unequal CartPole panel ranges.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from paper_style import apply,FIG_W
from fig_main_cv import _ensure_latex
SRC=ROOT/'_archive/conference_source/extracted';OUT=ROOT/'Jounral_PIWM/imgs'
COLORS=['#9476c4','#c82328','#3b88bc','#ed8b18','#855d50','#46a047']

def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches=None)
    fig.savefig(ROOT/'tmp/pdfs/after'/(name+'.png'),dpi=180,bbox_inches=None)
    plt.close(fig)

def results(name,source,ids,limits):
    fig,axes=plt.subplots(2,3,figsize=(FIG_W,4.9))
    fig.subplots_adjust(left=.085,right=.985,bottom=.085,top=.785,wspace=.28,hspace=.93)
    for r in range(2):
        y=.975 if r==0 else .480
        fig.text(.085,y,'('+('a' if r==0 else 'b')+') '+('Extrinsic' if r==0 else 'Intrinsic'),fontsize=10,va='top')
        labels=['PIWM (discrete)','PIWM (continuous)', 'DVBF' if r==0 else 'GOKU-net',
                'SINDYc' if r==0 else 'Vid2Param','LSTM','Transformer']
        handles=[Line2D([],[],color=c,lw=1.8,label=l) for c,l in zip(COLORS,labels)]
        fig.legend(handles=handles,ncol=3,loc='upper left',bbox_to_anchor=(.07,y-.033),
                   fontsize=8.3,columnspacing=1.1,handlelength=1.5,frameon=False)
        for j in range(3):
            ax=axes[r,j];hi=limits[r][j]
            raw=Image.open(SRC/f'{source}_Image{ids[r][j]}.jpg')
            ax.imshow(raw,extent=(0,30,0,hi),aspect='auto',interpolation='none',zorder=1)
            ax.set_xlim(0,30);ax.set_ylim(0,hi);ax.set_xticks([0,10,20,30])
            ax.set_yticks([0,1,2] if hi==2 else ([0,2,hi] if hi>=3.5 else [0,1.5,3]))
            ax.set_title(r'$\delta='+['0','5\%','10\%'][j]+'$',pad=5,fontsize=9)
            ax.grid(False);ax.tick_params(labelsize=8,pad=2,length=2)
            if j==0:ax.set_ylabel('State RMSE',fontsize=9)
            if j==1:ax.set_xlabel('Prediction horizon (steps)',fontsize=9,labelpad=3)
    save(fig,name)

def frames():
    # A PDF inset is a contiguous figure raster. Display full frame rectangles
    # by viewport clipping; no enhancement, retouching, or synthesized pixels.
    pred=Image.open(SRC/'pred_Image22.jpg')
    cart=Image.open(SRC/'cartviz_Image24.jpg') # original ground truth, k=15
    viewports=[(135,82,277,208),(1042,82,1182,205)]
    fig,axes=plt.subplots(1,3,figsize=(FIG_W,1.55))
    fig.subplots_adjust(left=.012,right=.99,bottom=.05,top=.78,wspace=.10)
    for ax,title in zip(axes,['(a) CartPole','(b) Lunar Lander','(c) DonkeyCar simulator']):
        ax.set_title(title,fontsize=9,pad=6);ax.axis('off')
    axes[0].imshow(cart,interpolation='none')
    for ax,box in [(axes[1],viewports[1]),(axes[2],viewports[0])]:
        ax.imshow(pred,interpolation='none');ax.set_xlim(box[0],box[2]);ax.set_ylim(box[3],box[1]);ax.set_aspect('equal')
    save(fig,'fig_conference_environments')
    # Two environments, all five original comparison rows; retain k=5,15,30.
    # Intermediate columns are omitted only to keep the printed frames legible.
    fig=plt.figure(figsize=(FIG_W,4.25))
    starts=[.19,.61]; cw=.119; ch=.135
    row_bounds=[(82,208),(225,347),(366,493),(515,635),(652,772)]
    cols=[0,2,5]
    for env,(base_x,title) in enumerate(zip(starts,['DonkeyCar simulator','Lunar Lander'])):
        fig.text(base_x+.183,.985,title,ha='center',va='top',fontsize=9)
        for j,k in enumerate([5,15,30]):fig.text(base_x+j*.13+cw/2,.922,f'$k={k}$',ha='center',fontsize=8.5)
        for row,(y0,y1) in enumerate(row_bounds):
            for j,col in enumerate(cols):
                ax=fig.add_axes([base_x+j*.13,.735-row*.15,cw,ch])
                ax.imshow(pred,interpolation='none')
                if env==0: x0=135+col*(856/6); x1=135+(col+1)*(856/6)
                else: x0=1042+col*(840/6);x1=1042+(col+1)*(840/6)
                ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.axis('off')
    for row,label in enumerate(['Ground truth','PIWM','Standard\nworld model','GOKU-net','Vid2Param']):
        fig.text(.173,.8025-row*.15,label,ha='right',va='center',fontsize=8.5)
    save(fig,'fig_conference_rollouts')

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    (ROOT/'tmp/pdfs/after').mkdir(parents=True,exist_ok=True)
    apply(usetex=_ensure_latex())
    results('fig_conference_donkey','resultsx2',[[28,29,30],[33,32,31]],[[2]*3,[2]*3])
    results('fig_conference_lunar','resultsx2',[[25,26,27],[24,23,22]],[[2]*3,[2]*3])
    results('fig_conference_cartpole','cartresults',[[26,27,28],[23,24,25]],[[3.5,3.5,4],[3,3.5,4]])
    frames()
