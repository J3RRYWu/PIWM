"""Five-fold manuscript figure from the symmetric-selection evaluation cache.
Top: mean error trajectories. Bottom: individual fold endpoints, their mean
(diamond), and full min-max range. No rollouts or statistics are re-estimated.
Run from piwm/: .venv/Scripts/python.exe scripts/fig_main_cv.py
"""
import os as _os, sys as _sys
_SRC = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
_sys.path.insert(0, _SRC)

import argparse
import shutil
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from paper_style import apply as apply_style, FIG_W, COL, LBL

# The article's tables use the SYMMETRIC checkpoint selection (HANDOFF §0.12),
# so that cache is the default here; the val-loss-selected cache stays reachable
# via --curves for diagnostics only.
CURVES = _os.path.join("reports", "matrix", "folds_table_symmetric_curves.npz")
OUT = _os.path.join("figures", "fig_main_cv")

# row key in the npz -> (colour, label, linewidth, linestyle)
# goku_obs is deliberately absent: its checkpoints are bit-identical to V2P's
# (HANDOFF §0.11), so plotting it as a separate baseline would contradict the
# article, whose Table 1 carries no such row.
SERIES = [
    ("DVBF",     COL["DVBF"],     LBL["DVBF"],                 1.6, "--"),
    ("GOKU",     COL["GOKU"],     "GOKU-net",                  1.6, "--"),
    ("V2P",      COL["V2P"],      LBL["V2P"],                  1.6, "--"),
    ("ours-a",   COL["Frenet"],   LBL["Frenet"],               2.4, "-"),
    ("ours-c",   COL["FrenetOr"], "PIWM-Frenet (map-free)",    1.4, ":"),
]


def _ensure_latex():
    """MiKTeX is installed on this machine but deliberately not on PATH (see
    CLAUDE.md), so usetex silently fails unless we go looking for it."""
    if shutil.which("latex"):
        return True
    guess = _os.path.join(_os.environ.get("LOCALAPPDATA", ""),
                          "Programs", "MiKTeX", "miktex", "bin", "x64")
    if _os.path.isdir(guess):
        _os.environ["PATH"] = guess + _os.pathsep + _os.environ.get("PATH", "")
        if shutil.which("latex"):
            print(f"prepended MiKTeX to PATH: {guess}")
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", default=CURVES)
    ap.add_argument("--no-tex", action="store_true", dest="no_tex")
    ap.add_argument("--ymax", type=float, default=None,
                    help="clip the y axis; without it the axis fits every fold band")
    a = ap.parse_args()

    usetex = (not a.no_tex) and _ensure_latex()
    if not usetex:
        print("! LaTeX not reachable -- falling back to a metric-similar serif. "
              "The figure will be self-consistent but not identical to the body font.")
    apply_style(usetex=usetex)

    z = np.load(a.curves)
    folds = sorted({int(k.split("_", 1)[0][1:]) for k in z.files})
    print(f"{a.curves}: folds {folds}")

    fig = plt.figure(figsize=(FIG_W, 4.65))
    ax = fig.add_axes([.11, .50, .86, .32])
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


if __name__ == "__main__":
    main()
