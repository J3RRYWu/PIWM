"""fig_delta, redrawn from the 5-fold curves instead of the retired single split.

The previous fig_delta_supervision.pdf was produced from the dyn_k16 checkpoint
family whose numbers did not reproduce, and its caption cited the old baseline
figure; both contradict the cross-validated Table 3. This one reads the SAME cache
the table is built from (folds_table_symmetric_curves.npz), so figure and table
cannot disagree.

Content: our model at delta = 0 / 5% / 10% (fold-mean lines, min-max bands), with
the strongest baseline at its CLEAN-label best as the reference line -- the honest
comparison, since the baselines only get worse under noise.

    <py311> scripts/fig_delta_cv.py        # -> figures/fig_delta_cv.{pdf,png}
"""
import os as _os, sys as _sys
_SRC = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
_sys.path.insert(0, _SRC)

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from fig_main_cv import _ensure_latex          # same MiKTeX-not-on-PATH dance
from paper_style import apply as apply_style, FIG_W, COL, LBL

CURVES = _os.path.join("reports", "matrix", "folds_table_symmetric_curves.npz")
OUT = _os.path.join("figures", "fig_delta_cv")

SERIES = [("ours-a",     r"$\delta = 0$",    1.00),
          ("ours-a_d5",  r"$\delta = 5\%$",  0.65),
          ("ours-a_d10", r"$\delta = 10\%$", 0.35)]


def main():
    apply_style(usetex=_ensure_latex())
    z = np.load(CURVES)
    folds = sorted({int(k.split("_", 1)[0][1:]) for k in z.files})

    fig, ax = plt.subplots(figsize=(FIG_W, 3.0))
    base = np.array(mpl.colors.to_rgb(COL["Frenet"]))
    steps = None
    for row, lbl, shade in SERIES:
        C = np.stack([z[f"f{f}_{row}"] for f in folds])
        steps = np.arange(C.shape[1])
        col = tuple(base * shade + (1 - shade) * 0.82)   # lighter = noisier
        half = (C[:, -1].max() - C[:, -1].min()) / 2
        ax.plot(steps, C.mean(0), color=col, lw=2.0,
                label=f"{lbl} ({C.mean(0)[-1]:.2f} $\\pm$ {half:.2f} m)")
        ax.fill_between(steps, C.min(0), C.max(0), color=col, alpha=0.14, lw=0)
        print(f"{row}: @100 {C.mean(0)[-1]:.3f} +/- {half:.3f}")

    V = np.stack([z[f"f{f}_V2P"] for f in folds]).mean(0)
    ax.plot(steps, V, color=COL["V2P"], lw=1.4, ls="--",
            label=f"{LBL['V2P']}, clean labels ({V[-1]:.2f} m)")

    ax.set_xlabel("rollout step"); ax.set_ylabel("position error (m)")
    ax.set_xlim(0, steps[-1]); ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    _os.makedirs("figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}")
    print(f"saved -> {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
