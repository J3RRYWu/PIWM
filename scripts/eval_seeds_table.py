"""Main table with error bars: every model retrained on 3 seeds, one evaluator.

WHY THIS EXISTS
`src/eval_frenet_vs_baselines.py` reports ONE checkpoint per model, and the
2026-08-08 round showed the ours row (0.267 m) is not reproducible -- controlled
retrains scatter over 0.412 +/- 0.080, an interval wider than the 0.116 m gap to
the strongest baseline. So the baselines were retrained on the same three seeds
(reports/repeats/baselines_rerun.log) and this script measures ALL of them with
the SAME evaluator the paper uses, so the table can be reported as mean +/- range.

Conventions are copied verbatim from `eval_frenet_vs_baselines.py` (same split,
same 201 windows in the same order, legacy nearest-neighbour sd2xy) so the ours-(a)
row must reproduce reports/repeats/ac_paired_3seeds.md -- that agreement is the
harness's own self-check and is asserted at the end.

Rows
  DVBF / GOKU / V2P     checkpoints/{v}_lane_donkey_s{i}/best.tar
  ours-(a) map          _repeat/dyn_s{i} + _repeat/kappa_s{i}, (s,d) -> xy via the map
  ours-(c) map-free     _repeat/dyn_s{i} + _repeat/shape_s{i}, pose read off the shape head
  paper                 frenet/dyn_k16 + frenet/kappa_scratch (the un-reproducible row)

Because every row walks the same windows in the same order the seed-wise
contrasts are PAIRED; the ours-vs-baseline gap is bootstrapped over windows.

    <py311> scripts/eval_seeds_table.py            # from piwm/
"""
import os as _os, sys as _sys
_SRC = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
_sys.path.insert(0, _SRC)
_sys.path.insert(0, _os.path.join(_SRC, "train"))
import donkey_config; donkey_config.patch_globals()

import argparse
import glob
import json
import time
import numpy as np
import torch

from config import DATA_DIR, PHYSICS_MEAN_REL, PHYSICS_STD_REL
from lane_utils import LANE_MEAN, LANE_STD, LANE_DIM, LANE_FRAME_STACK as FS
from donkey_config import DONKEY_SPATIAL_SCALE as SCALE
from relative_coords import to_relative_np
from donkey_dataset import split_segments
from train_piwm_lane_v5 import SeqLaneDataset
from baselines.shared_dynamics_lane import (DynamicsGOKULane, DynamicsDVBFLane,
                                            DynamicsVid2ParamLane)
from models.frenet_dynamics import FrenetDynamics
from models.encoder_lane import PhysicsEncoderLane
from models.road_perception import RoadContextEncoder
from frenet_track import local_road_points
from utils import load_checkpoint

K = 100
STRIDE = 4
SEEDS = (0, 1, 2)
META = _os.path.join(_SRC, "..", "..", "Data_Donkeycar_frenet", "_meta")
FRENET_DIR = _os.path.join(_SRC, "..", "..", "Data_Donkeycar_frenet")
REPORT = _os.path.join("reports", "repeats", "main_table_3seeds.md")
CURVES = _os.path.join("reports", "repeats", "main_table_3seeds_curves.npz")

_tr = np.load(_os.path.join(META, "track.npz"))
_cen, _nrm, _hd = _tr["centers"], _tr["normals"], _tr["heading"]
_L, _ds, _M = float(_tr["total_len"]), float(_tr["grid_ds"]), len(_cen)
OFFSETS = np.load(_os.path.join(META, "stats.npz"))["kappa_offsets"].astype(np.float32)


def _idx(s):
    return (np.mod(s, _L) / _ds).astype(int) % _M


def sd2xy_nn(s, d):
    """The paper evaluator's convention (nearest grid point)."""
    i = _idx(s)
    return _cen[i] + d[..., None] * _nrm[i]


def _cen_lin(s):
    u = np.mod(s, _L) / _ds
    i0 = np.floor(u).astype(int) % _M
    w = (u - np.floor(u))[..., None]
    return _cen[i0] * (1 - w) + _cen[(i0 + 1) % _M] * w


def sd2xy(s, d):
    return _cen_lin(s) + d[..., None] * _nrm[_idx(s)]


def to_local(xy, s0):
    """World xy -> road frame at s0: the frame the map-free rollout reconstructs in."""
    th = _hd[_idx(np.array([s0]))[0]]
    c, sn = np.cos(th), np.sin(th)
    dd = xy - _cen_lin(np.array([s0]))[0]
    return np.stack([dd[..., 0] * c + dd[..., 1] * sn,
                     -dd[..., 0] * sn + dd[..., 1] * c], -1)


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


def _load_frenet(ckpt):
    fr = FrenetDynamics(_os.path.join(META, "track.npz"), _os.path.join(META, "stats.npz"))
    fr.load_state_dict(load_checkpoint(ckpt)["dynamics"]); fr.eval()
    return fr


def _load_road(ckpt, predict_shape=False):
    enc = RoadContextEncoder(n_offsets=len(OFFSETS), freeze_backbone=True,
                             predict_shape=predict_shape)
    enc.load_state_dict(load_checkpoint(ckpt)["model"]); enc.eval()
    return enc


def main(stride=STRIDE, seeds=SEEDS):
    t_start = time.time()
    base = SeqLaneDataset(DATA_DIR, seq_len=2)
    _, val_eps = split_segments(base, val_frac=0.10, seed=0)
    fr_files = sorted([f for f in glob.glob(_os.path.join(FRENET_DIR, "*.npz")) if "_meta" not in f])
    print(f"val segments: {sorted(val_eps)}")

    # ---- models -------------------------------------------------------
    # the v6 image encoder feeding V2P's theta is NOT reseeded: it is shared by
    # every V2P seed, so the spread below is the dynamics' training noise only.
    enc = PhysicsEncoderLane()
    enc.load_state_dict(load_checkpoint("checkpoints/piwm_lane_v6_donkey/ae.tar")["encoder"]); enc.eval()
    for p in enc.parameters(): p.requires_grad = False

    bl = {}          # (variant, seed) -> module
    for v, cls in [("DVBF", DynamicsDVBFLane), ("GOKU", DynamicsGOKULane),
                   ("V2P", DynamicsVid2ParamLane)]:
        for i in seeds:
            m = cls()
            m.load_state_dict(load_checkpoint(
                f"checkpoints/{v.lower()}_lane_donkey_s{i}/best.tar")["model"])
            m.eval(); bl[(v, i)] = m

    dyn = {i: _load_frenet(f"checkpoints/_repeat/dyn_s{i}.tar") for i in seeds}
    kap = {i: _load_road(f"checkpoints/_repeat/kappa_s{i}.tar") for i in seeds}
    shp = {i: _load_road(f"checkpoints/_repeat/shape_s{i}.tar", predict_shape=True) for i in seeds}
    dyn_paper = _load_frenet("checkpoints/frenet/dyn_k16.tar")
    kap_paper = _load_road("checkpoints/frenet/kappa_scratch.tar")

    rows = ([f"DVBF_s{i}" for i in seeds] + [f"GOKU_s{i}" for i in seeds]
            + [f"V2P_s{i}" for i in seeds] + [f"ours-a_s{i}" for i in seeds]
            + [f"ours-c_s{i}" for i in seeds] + [f"ours-c*_s{i}" for i in seeds]
            + ["paper(dyn_k16)"])
    res = {r: [] for r in rows}

    # ---- one pass over the windows, every model measured on each -------
    nwin = 0
    for ep in sorted(val_eps):
        phys = base.phys_list[ep]; acts31 = base.acts_list[ep]
        wpw = base.wp_world_list[ep]; imgs = base.imgs_list[ep]
        frd = np.load(fr_files[ep])
        fst, fac, fim = frd["state"], frd["action"], frd["imgs"].astype(np.float32)
        T = min(len(phys), len(fst))
        if T < K + 1 or T < FS: continue
        for t0 in range(FS - 1, T - K - 1, stride):
            t0 = int(t0); nwin += 1
            s31 = build_state31(phys, wpw, t0, K + 1)
            acts_b = [torch.tensor(acts31[t0 + k]).unsqueeze(0) for k in range(K)]
            gt = fst[t0:t0 + K + 1]
            z0 = torch.tensor(fst[t0]); acts_f = torch.tensor(fac[t0:t0 + K])
            s0 = float(fst[t0, 0])
            gt_xy_nn = sd2xy_nn(gt[:, 0], gt[:, 1])
            gt_loc = to_local(sd2xy(gt[:, 0], gt[:, 1]), s0)
            stack = torch.tensor(fim[t0 - FS + 1:t0 + 1], dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                # --- baselines: GT 31-dim init, error back in real metres ---
                for v in ("DVBF", "GOKU"):
                    for i in seeds:
                        z = torch.tensor(s31[0]).unsqueeze(0); xy = [s31[0, :2]]
                        for k in range(K):
                            out = bl[(v, i)](z, acts_b[k])
                            z = out[0] if isinstance(out, tuple) else out
                            xy.append(z[0, :2].numpy())
                        res[f"{v}_s{i}"].append(np.linalg.norm(
                            (np.array(xy) - s31[:, :2]) * PHYSICS_STD_REL[:2] / SCALE, axis=-1))
                # V2P: theta inferred from the 15-frame stack ending at t0
                stack0 = np.stack([imgs[t0 - FS + 1 + j] for j in range(FS)], 0)[None]
                obs = enc(torch.tensor(stack0, dtype=torch.float32)).unsqueeze(1)
                for i in seeds:
                    # posterior mean, so the reported cells are deterministic
                    _, theta, _ = bl[("V2P", i)].infer_theta(obs)
                    z = torch.tensor(s31[0]).unsqueeze(0); xy = [s31[0, :2]]
                    for k in range(K):
                        z = bl[("V2P", i)].step(z, acts_b[k], theta)
                        xy.append(z[0, :2].numpy())
                    res[f"V2P_s{i}"].append(np.linalg.norm(
                        (np.array(xy) - s31[:, :2]) * PHYSICS_STD_REL[:2] / SCALE, axis=-1))

                # --- ours (a): perceived kappa, position via the KNOWN centreline ---
                prof_i = {}
                for i in seeds:
                    prof = kap[i](stack)[0].numpy().astype(np.float32)
                    prof_i[i] = prof
                    f = dyn[i].rollout_perceived(z0, acts_f, torch.tensor(prof), OFFSETS).numpy()
                    res[f"ours-a_s{i}"].append(np.linalg.norm(
                        sd2xy_nn(f[:, 0], f[:, 1]) - gt_xy_nn, axis=-1))
                # --- ours (c): map-free, road READ off the shape head ---
                # (c)  = geometry AND curvature from the joint shape model
                # (c*) = geometry from the shape head, curvature still from the
                #        kappa-only model, so a weaker joint kappa head cannot be
                #        mistaken for the shape readout failing. Both camera-only.
                for i in seeds:
                    kap_s, shp_s = shp[i].forward_both(stack)
                    kap_s = kap_s[0].numpy().astype(np.float32)
                    shp_s = shp_s[0].numpy().astype(np.float32)
                    shp_s[:, 0] += OFFSETS                 # residual -> absolute X
                    sp = torch.tensor(shp_s)
                    for tag, kp in [("c", kap_s), ("c*", prof_i[i])]:
                        p = dyn[i].rollout_perceived_shape(z0, acts_f, torch.tensor(kp),
                                                           sp, OFFSETS).numpy()
                        res[f"ours-{tag}_s{i}"].append(np.linalg.norm(p[:, :2] - gt_loc, axis=-1))
                # --- the row the paper currently reports ---
                prof = kap_paper(stack)[0].numpy().astype(np.float32)
                f = dyn_paper.rollout_perceived(z0, acts_f, torch.tensor(prof), OFFSETS).numpy()
                res["paper(dyn_k16)"].append(np.linalg.norm(
                    sd2xy_nn(f[:, 0], f[:, 1]) - gt_xy_nn, axis=-1))
        print(f"  ep {ep}: {nwin} windows cumulative ({time.time()-t_start:.0f}s)", flush=True)

    E = {r: np.array(v) for r, v in res.items() if v}
    np.savez_compressed(CURVES, **E)

    # ---- per-seed table -----------------------------------------------
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out(f"n windows = {nwin}   (val split seed=0, stride {stride}, {K}-step rollout, GT init)")
    out()
    out(f"{'row':<16} {'xy@25':>8} {'xy@50':>8} {'xy@100':>9}")
    out("-" * 44)
    for r in rows:
        if r not in E: continue
        e = E[r]
        out(f"{r:<16} {e[:,25].mean():>7.3f}m {e[:,50].mean():>7.3f}m {e[:,100].mean():>8.3f}m")

    # ---- mean +/- range across seeds -----------------------------------
    def agg(prefix):
        v = [E[f"{prefix}_s{i}"] for i in seeds if f"{prefix}_s{i}" in E]
        return np.array([[e[:, k].mean() for k in (25, 50, 100)] for e in v])

    out()
    out("mean +/- std over 3 seeds (E_xy, m)")
    out(f"{'model':<16} {'@25':>16} {'@50':>16} {'@100':>16}")
    out("-" * 68)
    summary = {}
    for prefix, label in [("DVBF", "DVBF"), ("GOKU", "GokuNet"), ("V2P", "Vid2Param"),
                          ("ours-a", "ours (map)"), ("ours-c", "ours (map-free)"),
                          ("ours-c*", "ours (mf, k*)")]:
        a = agg(prefix)
        summary[prefix] = a.tolist()
        cells = [f"{a[:,j].mean():.3f}+/-{(a[:,j].max()-a[:,j].min())/2:.3f}" for j in range(3)]
        out(f"{label:<16} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16}")
    p = E["paper(dyn_k16)"]
    out(f"{'paper row':<16} {p[:,25].mean():>15.3f} {p[:,50].mean():>15.3f} {p[:,100].mean():>15.3f}")

    # ---- paired contrasts, ours vs the strongest baseline ---------------
    out()
    out("paired gap on xy@100, ours-(a) minus baseline, per seed "
        "(negative = ours better; 95% CI, 10k-resample bootstrap over windows)")
    out("-" * 78)
    rng = np.random.default_rng(0)
    for v in ("V2P", "GOKU", "DVBF"):
        for i in seeds:
            da = E[f"ours-a_s{i}"][:, 100] - E[f"{v}_s{i}"][:, 100]
            idx = rng.integers(0, len(da), (10000, len(da)))
            lo, hi = np.percentile(da[idx].mean(1), [2.5, 97.5])
            out(f"  ours-a_s{i} - {v}_s{i:<3} = {da.mean():+.3f}m  [{lo:+.3f},{hi:+.3f}]")

    out()
    out("paired (a) vs (c): what dropping the known centreline costs")
    for tag in ("c", "c*"):
        for i in seeds:
            d = (E[f"ours-{tag}_s{i}"][:, 100] - E[f"ours-a_s{i}"][:, 100]).mean()
            out(f"  ({tag}) seed {i}: {d:+.3f}m")

    # ---- self-check against the 08-08 run -------------------------------
    ref_a = {0: 0.383, 1: 0.506, 2: 0.346}
    out()
    out("self-check vs reports/repeats/ac_paired_3seeds.md (ours-a @100)")
    ok = True
    for i in seeds:
        got = E[f"ours-a_s{i}"][:, 100].mean()
        d = abs(got - ref_a[i])
        ok &= d < 0.002
        out(f"  seed {i}: got {got:.3f}  expected {ref_a[i]:.3f}  {'OK' if d < 0.002 else 'MISMATCH'}")
    out(f"  -> evaluator {'matches' if ok else 'DIVERGES FROM'} the 08-08 harness")

    _os.makedirs(_os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("# ä¸»è¡¨ 3 seeds (mean +/- range)\n\n```\n" + "\n".join(lines) + "\n```\n")
    with open(REPORT.replace(".md", ".json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nsaved -> {REPORT}\nsaved -> {CURVES}\n({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=STRIDE)
    a = ap.parse_args()
    main(a.stride)
