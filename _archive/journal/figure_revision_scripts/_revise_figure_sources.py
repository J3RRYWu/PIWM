from pathlib import Path
p=Path('scripts/fig_main_cv.py');s=p.read_text(encoding='utf-8')
start=s.index('    fig, ax = plt.subplots(')
end=s.index('\n\nif __name__',start)
s=s[:start]+'''    fig = plt.figure(figsize=(FIG_W, 4.65))
    ax = fig.add_axes([.11, .47, .86, .35])
    endax = fig.add_axes([.35, .085, .62, .235])
    ax.set_title('(a) Mean prediction error', loc='left', pad=9)
    endpoint_labels, records = [], []
    for row, col, lbl, lw, ls in SERIES:
        if any(f"f{f}_{row}" not in z.files for f in folds):
            raise SystemExit(f"Incomplete folds for {row}: {a.curves}")
        C = np.stack([z[f"f{f}_{row}"] for f in folds])
        assert np.isfinite(C).all() and C.shape[0] == 5
        steps = np.arange(C.shape[1])
        label = {'ours-a': 'PIWM-Frenet', 'ours-c': 'PIWM (map-free)'}.get(row, lbl)
        ax.plot(steps, C.mean(0), color=col, lw=lw, ls=ls, label=label)
        endpoint_labels.append(label)
        records.append((col,C[:, -1]))
        print(f"{row}: @100 {C[:, -1].mean():.6f} +/- {C[:, -1].std(ddof=1):.6f}")
    for i,(col,values) in enumerate(records):
        y=len(records)-1-i
        endax.plot([values.min(),values.max()],[y,y],color=col,lw=1.4)
        endax.scatter(values, y+np.linspace(-.13,.13,len(values)), color=col, s=13, alpha=.75, zorder=3)
        endax.scatter([values.mean()],[y],marker='D',s=30,facecolor=col,edgecolor='white',linewidth=.5,zorder=4)
    endax.set_yticks(range(len(records)),endpoint_labels[::-1])
    endax.set_ylim(-.55,len(records)-.45)
    endax.set_xlim(left=0)
    endax.set_xlabel('Position error at step 100 (m)')
    endax.tick_params(axis='y',length=0,pad=7)
    endax.spines['left'].set_visible(False)
    endax.grid(axis='y',visible=False)
    fig.text(.11,.372,'(b) Across-fold spread at step 100',fontsize=10)
    ax.set_xlabel('Rollout step')
    ax.set_ylabel('Position error (m)')
    ax.set_xlim(0,100);ax.set_ylim(0,a.ymax if a.ymax else None)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.52,.995),
               ncol=3,frameon=False,columnspacing=1.1,handlelength=2,fontsize=8.5)
    _os.makedirs("figures",exist_ok=True)
    for ext in ('pdf','png'):
        fig.savefig(f"{OUT}.{ext}",bbox_inches=None)
    shutil.copyfile(f"{OUT}.pdf",_os.path.join('Jounral_PIWM','imgs','fig_main_cv.pdf'))
    plt.close(fig)
''' +s[end:]
s=s.replace('min-max, chosen over a standard','min-max, chosen over a standard')
p.write_text(s,encoding='utf-8')
