"""How much does dropping the known centreline actually cost, and does reading the
road shape instead of integrating curvature buy it back?

Every row is the SAME 100-step rollout of the SAME trained Frenet dynamics on the
SAME val windows; only how the (s,d,psi) state is turned into a position differs.

  map / perceived-k    (s,d) -> xy via the KNOWN centreline  [what the paper reports]
  map-free / integral  theta=int k, C=int(cos,sin), then place the car   (rollout_perceived_local)
  map-free / shape     read C off the perceived road-shape head          (rollout_perceived_shape)

Each map-free row is also run with the ORACLE road taken from the map, which
separates "perception is imperfect" from "the reconstruction itself loses
information". GOKU/V2P are printed for scale: they are already map-free.

    <py311> scripts/eval_mapfree.py
"""
import os as _os, sys as _sys
_SRC = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
_sys.path.insert(0, _SRC)
_sys.path.insert(0, _os.path.join(_SRC, "train"))
import donkey_config; donkey_config.patch_globals()

import argparse
import glob
import numpy as np
import torch

from config import DATA_DIR, PHYSICS_MEAN_REL, PHYSICS_STD_REL
from lane_utils import LANE_MEAN, LANE_STD, LANE_DIM, LANE_FRAME_STACK as FS
from donkey_config import DONKEY_SPATIAL_SCALE as SCALE
from relative_coords import to_relative_np
from donkey_dataset import split_segments
from train_piwm_lane_v5 import SeqLaneDataset
from baselines.shared_dynamics_lane import DynamicsGOKULane, DynamicsVid2ParamLane
from models.frenet_dynamics import FrenetDynamics
from models.encoder_lane import PhysicsEncoderLane
from models.road_perception import RoadContextEncoder
from frenet_track import local_road_points
from utils import load_checkpoint

K = 100
STRIDE = 4
META = _os.path.join(_SRC, "..", "..", "Data_Donkeycar_frenet", "_meta")
FRENET_DIR = _os.path.join(_SRC, "..", "..", "Data_Donkeycar_frenet")

_tr = np.load(_os.path.join(META, "track.npz"))
_cen, _nrm, _hd = _tr["centers"], _tr["normals"], _tr["heading"]
_L, _ds, _M = float(_tr["total_len"]), float(_tr["grid_ds"]), len(_cen)
OFFSETS = np.load(_os.path.join(META, "stats.npz"))["kappa_offsets"].astype(np.float32)
KAPPA_GRID = _tr["kappa"]


def _idx(s):
    return (np.mod(s, _L) / _ds).astype(int) % _M


def _cen_lin(s):
    """Centreline at arc-length s, linearly interpolated (matches the shape target)."""
    u = np.mod(s, _L) / _ds
    i0 = np.floor(u).astype(int) % _M
    w = (u - np.floor(u))[..., None]
    return _cen[i0] * (1 - w) + _cen[(i0 + 1) % _M] * w


def sd2xy_nn(s, d):                    # legacy convention used by the paper's eval
    i = _idx(s)
    return _cen[i] + d[..., None] * _nrm[i]


def sd2xy(s, d):
    return _cen_lin(s) + d[..., None] * _nrm[_idx(s)]


def to_local(xy, s0):
    """World xy -> the road frame at s0 (origin C(s0), x along the tangent there):
    the frame both map-free rollouts reconstruct in."""
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


def main(shape_ckpt="checkpoints/frenet/road_shape.tar", stride=STRIDE):
    base = SeqLaneDataset(DATA_DIR, seq_len=2)
    _, val_eps = split_segments(base, val_frac=0.10, seed=0)
    fr_files = sorted([f for f in glob.glob(_os.path.join(FRENET_DIR, "*.npz")) if "_meta" not in f])
    print(f"val segments: {sorted(val_eps)}")

    fr = FrenetDynamics(_os.path.join(META, "track.npz"), _os.path.join(META, "stats.npz"))
    fr.load_state_dict(load_checkpoint("checkpoints/frenet/dyn_k16.tar")["dynamics"]); fr.eval()
    kenc = RoadContextEncoder(n_offsets=len(OFFSETS), freeze_backbone=True)
    kenc.load_state_dict(load_checkpoint("checkpoints/frenet/kappa_scratch.tar")["model"]); kenc.eval()

    senc = None
    sh_ck = shape_ckpt
    if _os.path.exists(sh_ck):
        senc = RoadContextEncoder(n_offsets=len(OFFSETS), freeze_backbone=True, predict_shape=True)
        senc.load_state_dict(load_checkpoint(sh_ck)["model"]); senc.eval()
    else:
        print(f"! {sh_ck} missing -- shape rows skipped (train it first)")

    goku = DynamicsGOKULane(); goku.load_state_dict(load_checkpoint("checkpoints/goku_lane_donkey_longK/best.tar")["model"]); goku.eval()
    v2p = DynamicsVid2ParamLane(); v2p.load_state_dict(load_checkpoint("checkpoints/v2p_lane_donkey_longK/best.tar")["model"]); v2p.eval()
    enc = PhysicsEncoderLane(); enc.load_state_dict(load_checkpoint("checkpoints/piwm_lane_v6_donkey/ae.tar")["encoder"]); enc.eval()
    for p in enc.parameters(): p.requires_grad = False

    rows = ["map/perc-k (legacy xy)", "map/perc-k", "map/true-k",
            "free-int/perc-k", "free-int/true-k",
            "free-shape/perc", "free-shape/perc(k*)", "free-shape/true", "GOKU", "V2P"]
    res = {r: [] for r in rows}

    for ep in sorted(val_eps):
        phys = base.phys_list[ep]; acts31 = base.acts_list[ep]
        wpw = base.wp_world_list[ep]; imgs = base.imgs_list[ep]
        frd = np.load(fr_files[ep])
        fst, fac, fim = frd["state"], frd["action"], frd["imgs"].astype(np.float32)
        kap_true_all = frd["kappa_profile"]
        T = min(len(phys), len(fst))
        if T < K + 1 or T < FS: continue
        for t0 in range(FS - 1, T - K - 1, stride):
            t0 = int(t0)
            gt = fst[t0:t0 + K + 1]
            z0 = torch.tensor(fst[t0]); acts = torch.tensor(fac[t0:t0 + K])
            s0 = float(fst[t0, 0])
            gt_xy = sd2xy(gt[:, 0], gt[:, 1])
            gt_loc = to_local(gt_xy, s0)

            with torch.no_grad():
                stack = torch.tensor(fim[t0 - FS + 1:t0 + 1], dtype=torch.float32).unsqueeze(0)
                kap_p = kenc(stack)[0].numpy().astype(np.float32)
                if senc is not None:
                    kap_s, shp_s = senc.forward_both(stack)
                    kap_s = kap_s[0].numpy().astype(np.float32)
                    shp_s = shp_s[0].numpy().astype(np.float32)
                    shp_s[:, 0] += OFFSETS                    # residual -> absolute X
            kap_t = kap_true_all[t0].astype(np.float32)
            shp_t = local_road_points(_cen, _hd, _L, _ds, np.array([s0]), OFFSETS)[0]

            # --- (1) known centreline: (s,d) -> xy through the map ---
            for name, kp in [("map/perc-k", kap_p), ("map/true-k", kap_t)]:
                f = fr.rollout_perceived(z0, acts, torch.tensor(kp), OFFSETS).numpy()
                res[name].append(np.linalg.norm(sd2xy(f[:, 0], f[:, 1]) - gt_xy, axis=-1))
                if name == "map/perc-k":
                    res["map/perc-k (legacy xy)"].append(np.linalg.norm(
                        sd2xy_nn(f[:, 0], f[:, 1]) - sd2xy_nn(gt[:, 0], gt[:, 1]), axis=-1))
            # --- (2) map-free, road rebuilt by integrating curvature twice ---
            for name, kp in [("free-int/perc-k", kap_p), ("free-int/true-k", kap_t)]:
                p = fr.rollout_perceived_local(z0, acts, torch.tensor(kp), OFFSETS).numpy()
                res[name].append(np.linalg.norm(p[:, :2] - gt_loc, axis=-1))
            # --- (3) map-free, road READ from the shape head (no integration) ---
            if senc is not None:
                # (k*) = geometry from the shape head but curvature from the original
                # kappa-only model, so a weaker joint kappa head cannot be mistaken
                # for the shape head failing. Both are still camera-only.
                for name, kp, sp in [("free-shape/perc", kap_s, shp_s),
                                     ("free-shape/perc(k*)", kap_p, shp_s),
                                     ("free-shape/true", kap_t, shp_t)]:
                    p = fr.rollout_perceived_shape(z0, acts, torch.tensor(kp),
                                                   torch.tensor(sp), OFFSETS).numpy()
                    res[name].append(np.linalg.norm(p[:, :2] - gt_loc, axis=-1))

            # --- baselines (already map-free), same windows ---
            s31 = build_state31(phys, wpw, t0, K + 1)
            with torch.no_grad():
                stack0 = np.stack([imgs[t0 - FS + 1 + j] for j in range(FS)], 0)[None]
                theta, _, _ = v2p.infer_theta(enc(torch.tensor(stack0, dtype=torch.float32)).unsqueeze(1))
                for name, step in [("GOKU", lambda z, a: goku(z, a)),
                                   ("V2P", lambda z, a: v2p.step(z, a, theta))]:
                    z = torch.tensor(s31[0]).unsqueeze(0); xy = [s31[0, :2]]
                    for k in range(K):
                        z = step(z, torch.tensor(acts31[t0 + k]).unsqueeze(0))
                        xy.append(z[0, :2].numpy())
                    res[name].append(np.linalg.norm(
                        (np.array(xy) - s31[:, :2]) * PHYSICS_STD_REL[:2] / SCALE, axis=-1))

    print(f"\n{'variant':<24} {'n':>5} {'xy@25':>8} {'xy@50':>8} {'xy@100':>9}")
    print("-" * 58)
    for r in rows:
        if not res[r]: continue
        E = np.array(res[r])
        print(f"{r:<24} {len(E):>5} {E[:,25].mean():>7.3f}m {E[:,50].mean():>7.3f}m "
              f"{E[:,100].mean():>8.3f}m")
    # every row is the same windows in the same order, so the contrasts are PAIRED
    print(f"\npaired contrasts on xy@100 (n={len(res['free-shape/perc'])}, "
          f"95% CI by 10k-resample bootstrap over windows)")
    print("-" * 66)
    rng = np.random.default_rng(0)
    pairs = [("free-int/perc-k", "free-shape/perc", "reading shape vs integrating kappa"),
             ("free-shape/perc", "map/perc-k", "what the map is still worth"),
             ("free-shape/perc", "V2P", "map-free Frenet vs the best baseline"),
             ("free-shape/true", "map/perc-k", "readout formulation, perception aside")]
    for a, b, what in pairs:
        if not res[a] or not res[b]: continue
        da = np.array(res[a])[:, 100] - np.array(res[b])[:, 100]
        idx = rng.integers(0, len(da), (10000, len(da)))
        lo, hi = np.percentile(da[idx].mean(1), [2.5, 97.5])
        print(f"  {a:>19} - {b:<16} = {da.mean():+.3f}m  [{lo:+.3f},{hi:+.3f}]  {what}")

    print("\nmap/*        = position via the KNOWN centreline (privileged)")
    print("free-int/*   = map-free, road from integrating kappa twice")
    print("free-shape/* = map-free, road read from the perceived shape head")
    print("*/true-k, */true = the road handed over from the map = perception-free upper bound")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shape-ckpt", default="checkpoints/frenet/road_shape.tar")
    p.add_argument("--stride", type=int, default=STRIDE)
    a = p.parse_args()
    main(a.shape_ckpt, a.stride)
