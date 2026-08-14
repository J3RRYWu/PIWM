"""Initial-state noise robustness, per fold.

The published version of this study (src/noise_robustness.py, figures/fig_noise)
was computed from the retired single-split checkpoints, so its numbers cannot sit
beside the cross-validated tables. This recomputes it on the per-fold checkpoints
using the same evaluator conventions as scripts/eval_folds_table.py.

The perturbation is applied to the state the rollout STARTS from, in units of each
state variable's own standard deviation, with several draws per window. It probes a
different failure mode from the delta label noise of Table 3 and the two must not be
conflated: this one is a test-time perturbation, that one corrupts training labels.

    <py311> scripts/eval_noise_cv.py        # -> reports/matrix/noise_cv.md
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)

import argparse
import glob
import time
import numpy as np
import torch

import eval_seeds_table as est
from folds import fold_split

from config import DATA_DIR, PHYSICS_STD_REL
from lane_utils import LANE_FRAME_STACK as FS
from donkey_config import DONKEY_SPATIAL_SCALE as SCALE
from train_piwm_lane_v5 import SeqLaneDataset
from baselines.shared_dynamics_lane import (DynamicsGOKULane, DynamicsDVBFLane,
                                            DynamicsVid2ParamLane)
from models.encoder_lane import PhysicsEncoderLane
from utils import load_checkpoint

K = est.K
OFFSETS = est.OFFSETS
SIGMAS = [0.0, 0.25, 0.5, 1.0, 1.5]
N_DRAW = 10
BASELINES = [("DVBF", DynamicsDVBFLane), ("GOKU", DynamicsGOKULane),
             ("V2P", DynamicsVid2ParamLane)]
REPORT = _os.path.join("reports", "matrix", "noise_cv.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--nfolds", type=int, default=5)
    ap.add_argument("--stride", type=int, default=8, help="coarser than the tables; "
                    "this study needs many draws per window, not maximum precision")
    a = ap.parse_args()

    t0 = time.time()
    lines = []
    def log(s=""):
        print(s, flush=True); lines.append(s)

    base = SeqLaneDataset(DATA_DIR, seq_len=2)
    fr_files = sorted([f for f in glob.glob(_os.path.join(est.FRENET_DIR, "*.npz"))
                       if "_meta" not in f])
    enc = PhysicsEncoderLane()
    enc.load_state_dict(load_checkpoint("checkpoints/piwm_lane_v6_donkey/ae.tar")["encoder"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False

    per_fold = {}
    for f in a.folds:
        dyn_ck = f"checkpoints/cv/dyn_f{f}.tar"
        kap_ck = f"checkpoints/cv/kappa_f{f}.tar"
        if not (_os.path.exists(dyn_ck) and _os.path.exists(kap_ck)):
            log(f"fold {f}: checkpoints missing -- skipped"); continue
        fr = est._load_frenet(dyn_ck)
        kenc = est._load_road(kap_ck)
        bl = {}
        for v, cls in BASELINES:
            p = f"checkpoints/{v.lower()}_lane_donkey_f{f}/best.tar"
            if _os.path.exists(p):
                m = cls(); m.load_state_dict(load_checkpoint(p)["model"]); m.eval()
                bl[v] = m
        # per-variable scale for the perturbation, from the fold's own training data
        std = fr.state_std.numpy().copy()
        std[0] = 0.5                       # arc length: fixed 0.5 m rather than track length
        rng = np.random.default_rng(f)

        _, val_eps = fold_split(len(base.imgs_list), f, a.nfolds)
        acc = {("ours", s): [] for s in SIGMAS}
        acc.update({(v, s): [] for v in bl for s in SIGMAS})
        for ep in sorted(val_eps):
            phys = base.phys_list[ep]; acts31 = base.acts_list[ep]
            wpw = base.wp_world_list[ep]; imgs = base.imgs_list[ep]
            frd = np.load(fr_files[ep])
            fst, fac, fim = frd["state"], frd["action"], frd["imgs"].astype(np.float32)
            T = min(len(phys), len(fst))
            if T < K + 1 or T < FS:
                continue
            for t in range(FS - 1, T - K - 1, a.stride):
                t = int(t)
                s31 = est.build_state31(phys, wpw, t, K + 1)
                acts_f = torch.tensor(fac[t:t + K])
                acts_b = [torch.tensor(acts31[t + k]).unsqueeze(0) for k in range(K)]
                gt = fst[t:t + K + 1]
                gt_xy = est.sd2xy_nn(gt[:, 0], gt[:, 1])
                with torch.no_grad():
                    stack = torch.tensor(fim[t - FS + 1:t + 1],
                                         dtype=torch.float32).unsqueeze(0)
                    prof = torch.tensor(kenc(stack)[0].numpy().astype(np.float32))
                    obs = enc(torch.tensor(
                        np.stack([imgs[t - FS + 1 + j] for j in range(FS)], 0)[None],
                        dtype=torch.float32)).unsqueeze(1)
                    # posterior mean; the only randomness in this study should be
                    # the initial-state perturbation being measured
                    theta = {v: bl[v].infer_theta(obs)[1] for v in bl if v == "V2P"}
                    for sig in SIGMAS:
                        eo, eb = [], {v: [] for v in bl}
                        for _ in range(N_DRAW if sig > 0 else 1):
                            z0 = fst[t] + rng.standard_normal(5).astype(np.float32) * std * sig
                            p = fr.rollout_perceived(torch.tensor(z0), acts_f,
                                                     prof, OFFSETS).numpy()
                            eo.append(np.linalg.norm(
                                est.sd2xy_nn(p[:, 0], p[:, 1])[K] - gt_xy[K]))
                            zb0 = s31[0] + rng.standard_normal(31).astype(np.float32) * sig
                            for v, m in bl.items():
                                z = torch.tensor(zb0).unsqueeze(0)
                                for k in range(K):
                                    if v == "V2P":
                                        z = m.step(z, acts_b[k], theta["V2P"])
                                    else:
                                        out = m(z, acts_b[k])
                                        z = out[0] if isinstance(out, tuple) else out
                                eb[v].append(np.linalg.norm(
                                    (z[0, :2].numpy() - s31[K, :2])
                                    * PHYSICS_STD_REL[:2] / SCALE))
                        acc[("ours", sig)].append(np.mean(eo))
                        for v in bl:
                            acc[(v, sig)].append(np.mean(eb[v]))
        per_fold[f] = {k: float(np.mean(v)) for k, v in acc.items() if v}
        log(f"fold {f}: done ({time.time()-t0:.0f}s)")

    got = sorted(per_fold)
    if not got:
        log("nothing evaluated"); return
    models = ["ours"] + [v for v, _ in BASELINES]
    log("\n" + "=" * 74)
    log(f"INITIAL-STATE NOISE  E_xy(100) in m, mean +/- std over {len(got)} folds")
    log("=" * 74)
    log(f"{'model':<10}" + "".join(f"{'sigma=' + str(s):>15}" for s in SIGMAS))
    log("-" * 74)
    for m in models:
        cells = []
        for s in SIGMAS:
            x = np.array([per_fold[f][(m, s)] for f in got if (m, s) in per_fold[f]])
            cells.append("--" if not len(x)
                         else f"{x.mean():.3f}+/-{x.std(ddof=1):.3f}")
        log(f"{m:<10}" + "".join(f"{c:>15}" for c in cells))

    log("\nrelative degradation from sigma=0 to sigma=1.5:")
    for m in models:
        a0 = np.mean([per_fold[f][(m, 0.0)] for f in got if (m, 0.0) in per_fold[f]])
        a1 = np.mean([per_fold[f][(m, 1.5)] for f in got if (m, 1.5) in per_fold[f]])
        log(f"  {m:<8} {a0:.3f} -> {a1:.3f} m  ({a1/a0:.2f}x)")

    _os.makedirs(_os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("# initial-state noise, 5-fold\n\n```\n" + "\n".join(lines) + "\n```\n")
    print(f"\nsaved -> {REPORT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
