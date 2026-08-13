"""5-fold CV tables: main comparison and the delta-noise sweep, with error bars.

WHAT IS DIFFERENT FROM `eval_seeds_table.py`
That script varies the TRAINING seed on ONE fixed split, so every model walks the
same 201 windows and contrasts are paired window-by-window. Here the SPLIT itself
varies, which is the whole point (the seed study cannot see split variance), but
it means:

  * windows differ between folds -- so nothing is paired ACROSS folds. Pairing is
    only valid WITHIN a fold, where every model does walk the same windows.
  * the aggregate is therefore: mean per fold first, then mean +/- spread over the
    5 fold-level means. Pooling raw windows across folds would weight the bigger
    folds more and mix two different sources of variance.
  * each fold holds out 1/5 of the episodes (~14-15) versus the legacy 10% (7), so
    per-fold window counts are roughly double the 201 the single-split tables use.
    Fold numbers are NOT comparable cell-by-cell with those tables.

The encoders are per-FOLD, not per-delta: delta is weak-supervision noise on the
dynamics' state labels and never touches the (image -> curvature) training pair.
So dyn_f2_d10 is evaluated with kappa_f2 / shape_f2.

Runs on whatever has finished -- missing checkpoints are reported and skipped, so
this is usable as a live preview while scripts/run_matrix.py is still going.

    <py311> scripts/eval_folds_table.py                 # every fold that exists
    <py311> scripts/eval_folds_table.py --folds 0 1     # just these
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)

import argparse
import json
import time
import numpy as np
import torch

# The window construction, the sd->xy conventions and the model loaders are
# already pinned by the seed-study evaluator (whose ours-(a) row reproduces the
# 08-08 harness exactly). Reuse them verbatim so the two tables stay commensurate.
import eval_seeds_table as est
from folds import fold_split, describe as describe_split

from config import DATA_DIR, PHYSICS_STD_REL
from lane_utils import LANE_FRAME_STACK as FS
from donkey_config import DONKEY_SPATIAL_SCALE as SCALE
from train_piwm_lane_v5 import SeqLaneDataset
from baselines.shared_dynamics_lane import (DynamicsGOKULane, DynamicsDVBFLane,
                                            DynamicsVid2ParamLane)
from models.encoder_lane import PhysicsEncoderLane
from utils import load_checkpoint

K, STRIDE = est.K, est.STRIDE
OFFSETS = est.OFFSETS
DELTAS = [0.0, 0.05, 0.10]
BASELINES = [("DVBF", DynamicsDVBFLane), ("GOKU", DynamicsGOKULane),
             ("V2P", DynamicsVid2ParamLane)]
REPORT = _os.path.join("reports", "matrix", "folds_table.md")


def dtag(d):
    return "" if d == 0 else f"_d{int(round(d * 100))}"


def _exists(p):
    return _os.path.exists(p)


def eval_one_fold(base, fr_files, fold, nfolds, deltas, log, eps_override=None):
    """Every model of one fold, measured on that fold's held-out episodes.

    `eps_override` restricts the evaluation to a subset of them (used by the
    leak audit, which needs the same models measured on two disjoint subsets).

    Returns {row: (n, curve_mean(101,), per_window[:, (25,50,100)])} or None if the
    fold has no usable checkpoints yet.
    """
    _, val_eps = fold_split(len(base.imgs_list), fold, nfolds)
    if eps_override is not None:
        val_eps = set(eps_override)
    log(f"fold {fold}: {describe_split(len(base.imgs_list), fold, nfolds)}")

    kap_ck = f"checkpoints/cv/kappa_f{fold}.tar"
    shp_ck = f"checkpoints/cv/shape_f{fold}.tar"
    kenc = est._load_road(kap_ck) if _exists(kap_ck) else None
    senc = est._load_road(shp_ck, predict_shape=True) if _exists(shp_ck) else None
    if kenc is None:
        log(f"  ! {kap_ck} missing -- ours rows skipped for this fold")

    enc = PhysicsEncoderLane()
    enc.load_state_dict(load_checkpoint("checkpoints/piwm_lane_v6_donkey/ae.tar")["encoder"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False

    dyn, bl = {}, {}
    for d in deltas:
        p = f"checkpoints/cv/dyn_f{fold}{dtag(d)}.tar"
        if _exists(p):
            dyn[d] = est._load_frenet(p)
        else:
            log(f"  ! missing {p}")
        for v, cls in BASELINES:
            p = f"checkpoints/{v.lower()}_lane_donkey_f{fold}{dtag(d)}/best.tar"
            if _exists(p):
                m = cls(); m.load_state_dict(load_checkpoint(p)["model"]); m.eval()
                bl[(v, d)] = m
            else:
                log(f"  ! missing {p}")
    if not dyn and not bl:
        return None

    rows = ([f"{v}{dtag(d)}" for v, _ in BASELINES for d in deltas]
            + [f"ours-a{dtag(d)}" for d in deltas] + [f"ours-c{dtag(d)}" for d in deltas])
    res = {r: [] for r in rows}

    for ep in sorted(val_eps):
        phys = base.phys_list[ep]; acts31 = base.acts_list[ep]
        wpw = base.wp_world_list[ep]; imgs = base.imgs_list[ep]
        frd = np.load(fr_files[ep])
        fst, fac, fim = frd["state"], frd["action"], frd["imgs"].astype(np.float32)
        T = min(len(phys), len(fst))
        if T < K + 1 or T < FS:
            continue
        for t0 in range(FS - 1, T - K - 1, STRIDE):
            t0 = int(t0)
            s31 = est.build_state31(phys, wpw, t0, K + 1)
            acts_b = [torch.tensor(acts31[t0 + k]).unsqueeze(0) for k in range(K)]
            gt = fst[t0:t0 + K + 1]
            z0 = torch.tensor(fst[t0]); acts_f = torch.tensor(fac[t0:t0 + K])
            s0 = float(fst[t0, 0])
            gt_xy_nn = est.sd2xy_nn(gt[:, 0], gt[:, 1])
            gt_loc = est.to_local(est.sd2xy(gt[:, 0], gt[:, 1]), s0)
            stack = torch.tensor(fim[t0 - FS + 1:t0 + 1], dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                # the encoders are per-fold, so their forward is done ONCE per window
                prof = kenc(stack)[0].numpy().astype(np.float32) if kenc is not None else None
                if senc is not None:
                    kap_s, shp_s = senc.forward_both(stack)
                    kap_s = kap_s[0].numpy().astype(np.float32)
                    shp_s = shp_s[0].numpy().astype(np.float32)
                    shp_s[:, 0] += OFFSETS              # residual -> absolute X
                    shp_t = torch.tensor(shp_s)
                obs = enc(torch.tensor(
                    np.stack([imgs[t0 - FS + 1 + j] for j in range(FS)], 0)[None],
                    dtype=torch.float32)).unsqueeze(1)

                for (v, _), d in [((v, c), d) for v, c in BASELINES for d in deltas]:
                    m = bl.get((v, d))
                    if m is None:
                        continue
                    z = torch.tensor(s31[0]).unsqueeze(0); xy = [s31[0, :2]]
                    theta = m.infer_theta(obs)[0] if v == "V2P" else None
                    for k in range(K):
                        if v == "V2P":
                            z = m.step(z, acts_b[k], theta)
                        else:
                            out = m(z, acts_b[k])
                            z = out[0] if isinstance(out, tuple) else out
                        xy.append(z[0, :2].numpy())
                    res[f"{v}{dtag(d)}"].append(np.linalg.norm(
                        (np.array(xy) - s31[:, :2]) * PHYSICS_STD_REL[:2] / SCALE, axis=-1))

                for d, fr in dyn.items():
                    if kenc is not None:
                        f = fr.rollout_perceived(z0, acts_f, torch.tensor(prof), OFFSETS).numpy()
                        res[f"ours-a{dtag(d)}"].append(np.linalg.norm(
                            est.sd2xy_nn(f[:, 0], f[:, 1]) - gt_xy_nn, axis=-1))
                    if senc is not None:
                        p = fr.rollout_perceived_shape(z0, acts_f, torch.tensor(kap_s),
                                                       shp_t, OFFSETS).numpy()
                        res[f"ours-c{dtag(d)}"].append(np.linalg.norm(p[:, :2] - gt_loc, axis=-1))

    out = {}
    counts = set()
    for r, v in res.items():
        if not v:
            continue
        E = np.array(v)
        counts.add(len(E))
        out[r] = (len(E), E.mean(0), E[:, [25, 50, 100]])
    if len(counts) > 1:
        raise RuntimeError(f"fold {fold}: rows walked different window counts {counts} "
                           "-- within-fold pairing would be invalid")
    log(f"  {len(out)} rows x {counts.pop() if counts else 0} windows")
    return out


def leak_audit(a):
    """Does the shared v6 encoder actually buy V2P anything on episodes it trained on?

    Raw V2P error on the two subsets is not the answer -- the subsets are different
    episodes with different difficulty. The control is GOKU/DVBF/ours, which use no
    image encoder at all: if V2P's ADVANTAGE OVER THEM is the same on seen and unseen
    episodes, the exposure is buying nothing and the leak is empirically inert.
    """
    lines = []
    def log(s=""):
        print(s, flush=True); lines.append(s)

    base = SeqLaneDataset(DATA_DIR, seq_len=2)
    import glob
    fr_files = sorted([f for f in glob.glob(_os.path.join(est.FRENET_DIR, "*.npz"))
                       if "_meta" not in f])
    n_eps = len(base.imgs_list)
    _, legacy_val = fold_split(n_eps)          # exactly what v6 never trained on
    _, f0 = fold_split(n_eps, fold=0, nfolds=a.nfolds)
    clean, seen = sorted(f0 & legacy_val), sorted(f0 - legacy_val)
    log("v6 LEAK AUDIT (fold 0)")
    log(f"  clean (v6 never saw): {clean}")
    log(f"  seen  (v6 trained on): {seen}")
    if not clean or not seen:
        log("  ! fold 0 does not split -- audit impossible"); return

    subsets = {}
    for tag, eps in [("clean", clean), ("seen", seen)]:
        log(f"\n-- subset {tag} --")
        subsets[tag] = eval_one_fold(base, fr_files, 0, a.nfolds, [0.0], log,
                                     eps_override=eps)

    log("\n" + "=" * 70)
    log("E_xy@100 (m) by v6 exposure, fold 0, delta=0")
    log("=" * 70)
    log(f"{'model':<16} {'clean':>10} {'seen':>10} {'seen-clean':>12}   uses v6?")
    log("-" * 70)
    means = {}
    for row, uses in [("V2P", "YES (theta)"), ("GOKU", "no"), ("DVBF", "no"),
                      ("ours-a", "no"), ("ours-c", "no")]:
        if any(row not in subsets[t] for t in ("clean", "seen")):
            continue
        c = subsets["clean"][row][2][:, 2].mean()
        s = subsets["seen"][row][2][:, 2].mean()
        means[row] = (c, s)
        log(f"{row:<16} {c:>10.3f} {s:>10.3f} {s - c:>+12.3f}   {uses}")

    log("\nthe actual test -- V2P's edge over each encoder-free model, per subset:")
    log("(if the leak helps V2P, its edge must be LARGER on 'seen')")
    log("-" * 70)
    verdict = []
    for row in ("GOKU", "DVBF", "ours-a"):
        if row not in means or "V2P" not in means:
            continue
        ec = means["V2P"][0] - means[row][0]
        es = means["V2P"][1] - means[row][1]
        verdict.append(es - ec)
        log(f"  V2P - {row:<7} clean {ec:+.3f}   seen {es:+.3f}   "
            f"shift {es - ec:+.3f}m {'(favours V2P)' if es < ec else '(against V2P)'}")
    if verdict:
        m = float(np.mean(verdict))
        log(f"\n  mean shift {m:+.3f}m -- "
            + ("exposure buys V2P nothing measurable; the leak is inert"
               if m > -0.02 else
               f"exposure is worth ~{-m:.3f}m to V2P; retrain the encoder per fold"))

    _os.makedirs(_os.path.dirname(REPORT), exist_ok=True)
    p = REPORT.replace(".md", "_leak_audit.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("# v6 encoder leak audit\n\n```\n" + "\n".join(lines) + "\n```\n")
    print(f"\nsaved -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--nfolds", type=int, default=5)
    ap.add_argument("--deltas", type=float, nargs="+", default=DELTAS)
    ap.add_argument("--leak-audit", action="store_true", dest="leak_audit",
                    help="quantify the shared v6 encoder's leakage into V2P. Only V2P "
                         "uses that encoder (for theta); DVBF/GOKU/ours do not, so they "
                         "act as a difficulty control. Folds are contiguous blocks of the "
                         "same permutation and the legacy hold-out is its first block, so "
                         "fold 0's validation set splits exactly into 7 episodes the v6 "
                         "encoder never saw and 7 it trained on -- same fold, same models, "
                         "only exposure differs.")
    a = ap.parse_args()
    if a.leak_audit:
        return leak_audit(a)

    t0 = time.time()
    lines = []
    def log(s=""):
        print(s, flush=True); lines.append(s)

    base = SeqLaneDataset(DATA_DIR, seq_len=2)
    import glob
    fr_files = sorted([f for f in glob.glob(_os.path.join(est.FRENET_DIR, "*.npz"))
                       if "_meta" not in f])
    folds = a.folds if a.folds is not None else list(range(a.nfolds))

    per_fold = {}
    for f in folds:
        r = eval_one_fold(base, fr_files, f, a.nfolds, a.deltas, log)
        if r:
            per_fold[f] = r
        else:
            log(f"fold {f}: nothing trained yet, skipped")
    if not per_fold:
        log("no fold has usable checkpoints yet."); return
    got = sorted(per_fold)
    log(f"\nfolds with results: {got}")

    def cell(row, col):
        """mean +/- half-range of the per-fold means, over folds that have `row`."""
        v = [per_fold[f][row][2][:, col].mean() for f in got if row in per_fold[f]]
        if not v:
            return None, 0
        v = np.array(v)
        return (v.mean(), (v.max() - v.min()) / 2 if len(v) > 1 else 0.0), len(v)

    # ---- main table (delta = 0) ---------------------------------------
    log("\n" + "=" * 74)
    log(f"MAIN TABLE  E_xy (m), mean +/- half-range over {len(got)} folds")
    log("=" * 74)
    log(f"{'model':<18} {'@25':>16} {'@50':>16} {'@100':>16} {'k':>3}")
    log("-" * 74)
    for row, label in [("DVBF", "DVBF"), ("GOKU", "GokuNet"), ("V2P", "Vid2Param"),
                       ("ours-a", "ours (map)"), ("ours-c", "ours (map-free)")]:
        cells, n = [], 0
        for j, c in enumerate((0, 1, 2)):
            (m_s), n = cell(row, c)
            cells.append("--" if m_s is None else f"{m_s[0]:.3f}+/-{m_s[1]:.3f}")
        if n:
            log(f"{label:<18} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16} {n:>3}")

    # ---- delta sweep --------------------------------------------------
    log("\n" + "=" * 74)
    log("DELTA WEAK-SUPERVISION NOISE  E_xy@100 (m), mean +/- half-range over folds")
    log("(the single-split table is non-monotonic -- 5% worse than 10% -- and the")
    log(" paper attributes that to single-run variance; these error bars test it)")
    log("=" * 74)
    log(f"{'model':<18} " + " ".join(f"{'d=' + str(int(d*100)) + '%':>16}" for d in a.deltas))
    log("-" * 74)
    for row, label in [("ours-a", "ours (map)"), ("ours-c", "ours (map-free)"),
                       ("V2P", "Vid2Param"), ("GOKU", "GokuNet"), ("DVBF", "DVBF")]:
        cells = []
        for d in a.deltas:
            (m_s), n = cell(f"{row}{dtag(d)}", 2)
            cells.append("--" if m_s is None else f"{m_s[0]:.3f}+/-{m_s[1]:.3f}")
        log(f"{label:<18} " + " ".join(f"{c:>16}" for c in cells))

    # ---- within-fold paired contrasts ---------------------------------
    log("\n" + "=" * 74)
    log("PAIRED within each fold: ours-(a) minus baseline, E_xy@100, delta=0")
    log("(pairing is only valid inside a fold -- different folds are different windows)")
    log("=" * 74)
    for v, _ in BASELINES:
        gaps, wins = [], 0
        for f in got:
            if "ours-a" not in per_fold[f] or v not in per_fold[f]:
                continue
            g = (per_fold[f]["ours-a"][2][:, 2] - per_fold[f][v][2][:, 2]).mean()
            gaps.append(g); wins += int(g < 0)
        if gaps:
            g = np.array(gaps)
            log(f"  vs {v:<5} mean {g.mean():+.3f}m  per fold "
                f"[{', '.join(f'{x:+.3f}' for x in g)}]  ours better in {wins}/{len(g)}")

    # ---- map-free cost -------------------------------------------------
    log("\npaired (a) vs (c): cost of dropping the known centreline, delta=0")
    gaps = [(per_fold[f]["ours-c"][2][:, 2] - per_fold[f]["ours-a"][2][:, 2]).mean()
            for f in got if "ours-c" in per_fold[f] and "ours-a" in per_fold[f]]
    if gaps:
        log(f"  mean {np.mean(gaps):+.3f}m  per fold [{', '.join(f'{x:+.3f}' for x in gaps)}]")

    _os.makedirs(_os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("# 5-fold CV tables\n\n```\n" + "\n".join(lines) + "\n```\n")
    curves = {f"f{f}_{r}": per_fold[f][r][1] for f in got for r in per_fold[f]}
    np.savez_compressed(REPORT.replace(".md", "_curves.npz"), **curves)
    print(f"\nsaved -> {REPORT}\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
