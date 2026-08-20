"""fig_main, redrawn from the 5-fold run instead of one checkpoint per model.

WHY A SECOND FIGURE RATHER THAN AN EDIT TO paper_figures.py
`src/paper_figures.py` recomputes its curves from a single checkpoint per model on
the legacy hold-out. That is exactly what the 2026-08 audit found to be untrustworthy:
both the ours row and the baseline rows in the paper turned out to be lucky draws.
This one plots what the 5-fold run measured -- so its subject is no longer "which
curve is lower" but "how much does each curve move when the split changes", which
is where our actual advantage is (std over folds 0.045 m against Vid2Param's 0.142
and DVBF's 0.131). The BAND is the fold-to-fold min-max, chosen over a standard
error because SE over windows shrinks with window count and would hide exactly what
is being shown; the legend reports the std over folds, matching the tables.

Curves come from the cache written by scripts/eval_folds_table.py -- no rollouts are
recomputed here, so this figure and the CV tables cannot drift apart. Pass
`--curves reports/matrix/folds_table_symmetric_curves.npz` for the symmetric-selection
version, which is what the article reports.

SINDYc is deliberately absent: its cached curve was measured on the legacy split,
and dropping a legacy-split curve into a CV figure is the provenance mistake this
whole round has been cleaning up. paper_figures.py keeps it for the single-split figure.

    <py311> scripts/fig_main_cv.py            # -> figures/fig_main_cv.{pdf,png}
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

    fig, ax = plt.subplots(figsize=(FIG_W, 3.15))
    steps, top = None, 0.0
    for row, col, lbl, lw, ls in SERIES:
        got = [z[f"f{f}_{row}"] for f in folds if f"f{f}_{row}" in z.files]
        if not got:
            print(f"  (skip {row}: not in the cache)")
            continue
        C = np.stack(got)                      # (n_folds, K+1)
        steps = np.arange(C.shape[1])
        mu, lo, hi = C.mean(0), C.min(0), C.max(0)
        half = C[:, -1].std(ddof=1)
        ax.plot(steps, mu, color=col, lw=lw, ls=ls,
                label=f"{lbl} ({mu[-1]:.2f} $\\pm$ {half:.2f} m)")
        ax.fill_between(steps, lo, hi, color=col, alpha=0.13, lw=0)
        top = max(top, float(hi.max()))
        print(f"  {row:<9} n={len(got)}  @100 {mu[-1]:.3f} +/- {half:.3f}  "
              f"fold range [{C[:, -1].min():.3f}, {C[:, -1].max():.3f}]")

    if steps is None:
        raise SystemExit(f"none of the expected rows are in {a.curves}; "
                         "run scripts/eval_folds_table.py first")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("position error (m)")
    ax.set_xlim(0, steps[-1])
    ax.set_ylim(0, a.ymax if a.ymax else top * 1.04)
    ax.legend(loc="upper left")
    _os.makedirs("figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}")
    plt.close(fig)
    print(f"saved -> {OUT}.pdf / .png"
          + (f"  (y clipped at {a.ymax})" if a.ymax else ""))
    print("band = min-max across folds; NOT copied into Jounral_PIWM/imgs/ "
          "-- do that when the results section is rewritten.")


if __name__ == "__main__":
    main()
