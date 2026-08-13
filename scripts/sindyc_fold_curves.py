"""SINDYc's per-fold error curves -- the fourth baseline the k-fold sweep was missing.

The paper compares against four baselines (DVBF, GOKU-net, Vid2Param, SINDYc) but the
5-fold run covered only the first three; SINDYc's curve was still the legacy-split
cache in figures/_sindyc_curve.npy, which is not commensurate with the CV tables.

Two things force this into its own script rather than into eval_folds_table.py:
  * pysindy cannot share a process with the CUDA torch build (segfault), so this runs
    in .venv-sindy -- see piwm/README.md.
  * SINDYc genuinely diverges here. Once the state overflows, sklearn's predict()
    RAISES on the non-finite input instead of returning it, so the rollout has to be
    frozen at the blow-up and the error clamped, exactly as paper_figures.sindyc_curve
    does. The clamp is a reporting convention, not a measurement: past it the number
    means "diverged", and the paper must say so rather than quoting a value.

Windows are built identically to eval_folds_table.py (same folds, same stride, same
metre conversion) so the resulting row is comparable cell-for-cell.

    <venv-sindy> scripts/sindyc_fold_curves.py      # -> reports/matrix/sindyc_curves.npz
"""
import os as _os, sys as _sys
_SRC = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
_sys.path.insert(0, _SRC)
_sys.path.insert(0, _os.path.join(_SRC, "train"))
import donkey_config; donkey_config.patch_globals()

import argparse
import pickle
import time
import numpy as np

from config import DATA_DIR, PHYSICS_MEAN_REL, PHYSICS_STD_REL
from lane_utils import LANE_MEAN, LANE_STD, LANE_DIM, LANE_FRAME_STACK as FS
from donkey_config import DONKEY_SPATIAL_SCALE as SCALE
from relative_coords import to_relative_np
from train_piwm_lane_v5 import SeqLaneDataset
from folds import fold_split

K = 100
STRIDE = 4
CLAMP_M = 100.0          # the reporting convention for a diverged rollout
OUT = _os.path.join("reports", "matrix", "sindyc_curves.npz")


def build_state31(phys, wp_world, t0, T):
    seg = phys[t0:t0 + T]
    car = (to_relative_np(seg, ref_idx=0) - PHYSICS_MEAN_REL) / PHYSICS_STD_REL
    pos0, yaw0 = seg[0, :2], seg[0, 2]
    c, s = np.cos(yaw0), np.sin(yaw0)
    dd = wp_world[t0:t0 + T] - pos0
    x = dd[..., 0] * c + dd[..., 1] * s
    y = -dd[..., 0] * s + dd[..., 1] * c
    lane = (np.stack([x, y], -1).reshape(T, LANE_DIM) - LANE_MEAN) / LANE_STD
    return np.concatenate([car, lane], -1).astype(np.float32)


def rollout(model, s31, acts):
    """(K+1,) metre error for one window, frozen and clamped once it diverges."""
    z = s31[0:1].copy().astype(np.float64)
    errs = [0.0]
    dead = False
    for k in range(K):
        if not dead:
            try:
                z = np.asarray(model.predict(z, u=acts[k:k + 1]), dtype=np.float64)
            except Exception:
                dead = True
            if not np.isfinite(z).all() or np.abs(z).max() > 1e4:
                dead = True
        if dead:
            errs.append(CLAMP_M)
        else:
            e = np.linalg.norm((z[0, :2] - s31[k + 1, :2]) * PHYSICS_STD_REL[:2] / SCALE)
            errs.append(min(float(e), CLAMP_M))
    return np.array(errs), dead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--nfolds", type=int, default=5)
    ap.add_argument("--stride", type=int, default=STRIDE)
    a = ap.parse_args()

    t0 = time.time()
    base = SeqLaneDataset(DATA_DIR, seq_len=2)
    out = {}
    for f in a.folds:
        ck = f"checkpoints/sindyc_lane_donkey_f{f}/model.pkl"
        if not _os.path.exists(ck):
            print(f"fold {f}: {ck} missing -- skipped"); continue
        with open(ck, "rb") as fh:
            model = pickle.load(fh)
        _, val_eps = fold_split(len(base.imgs_list), f, a.nfolds)
        curves, ndead = [], 0
        for ep in sorted(val_eps):
            phys = base.phys_list[ep]; acts31 = base.acts_list[ep]
            wpw = base.wp_world_list[ep]
            T = len(phys)
            if T < K + 1 or T < FS:
                continue
            for t in range(FS - 1, T - K - 1, a.stride):
                t = int(t)
                s31 = build_state31(phys, wpw, t, K + 1)
                e, dead = rollout(model, s31, acts31[t:t + K])
                curves.append(e); ndead += int(dead)
        if not curves:
            print(f"fold {f}: no windows"); continue
        C = np.array(curves)
        out[f"f{f}_SINDYc"] = C.mean(0)
        print(f"fold {f}: {len(C)} windows, {ndead} diverged "
              f"({100*ndead/len(C):.0f}%)  mean @25={C[:,25].mean():.2f} "
              f"@50={C[:,50].mean():.2f} @100={C[:,100].mean():.2f} m "
              f"({time.time()-t0:.0f}s)", flush=True)

    if not out:
        print("nothing computed"); return
    _os.makedirs(_os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, **out)
    at100 = [v[100] for v in out.values()]
    print(f"\nacross folds @100: mean {np.mean(at100):.1f} m "
          f"(clamped at {CLAMP_M:.0f}); report as DIVERGES, not as a value")
    print(f"saved -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
