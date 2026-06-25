"""Gold-standard fair comparison: Frenet vs GOKU/DVBF, ALL from GT init,
100-step rollout, error in REAL METRES on the SAME val windows.

- Frenet: GT [s,d,psi,v,omega] init -> rollout -> (s,d)->xy metres.
- GOKU/DVBF: GT 31-dim normalized rel state init -> rollout -> (x,y)_norm;
  metre error = ||Δ(x,y)_norm * PHYSICS_STD_REL[:2] / 50||.
Both isolate DYNAMICS quality (perfect init), same segments, same metric.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "train"))
import donkey_config; donkey_config.patch_globals()

import glob
import numpy as np
import torch

from config import DATA_DIR, PHYSICS_MEAN_REL, PHYSICS_STD_REL
from lane_utils import LANE_MEAN, LANE_STD, LANE_DIM
from donkey_config import DONKEY_SPATIAL_SCALE as SCALE
from relative_coords import to_relative_np
from donkey_dataset import split_segments
from train_piwm_lane_v5 import SeqLaneDataset
from baselines.shared_dynamics_lane import (DynamicsGOKULane, DynamicsDVBFLane,
                                            DynamicsVid2ParamLane)
from models.frenet_dynamics import FrenetDynamics
from models.encoder_lane import PhysicsEncoderLane
from models.road_perception import RoadContextEncoder
from lane_utils import LANE_FRAME_STACK
from config import ENCODER_DIM as CAR_ENCODER_DIM
from utils import load_checkpoint

DEVICE = torch.device("cpu")
K = 100
META = _os.path.join(_os.path.dirname(__file__), "..", "..", "Data_Donkeycar_frenet", "_meta")
FRENET_DIR = _os.path.join(_os.path.dirname(__file__), "..", "..", "Data_Donkeycar_frenet")

# track for sd->xy
_tr = np.load(_os.path.join(META, "track.npz"))
_cen, _nrm = _tr["centers"], _tr["normals"]
_L, _ds, _M = float(_tr["total_len"]), float(_tr["grid_ds"]), len(_tr["centers"])
def sd2xy(s, d):
    i = (np.mod(s, _L) / _ds).astype(int) % _M
    return _cen[i] + d[..., None] * _nrm[i]

OFFSETS = np.load(_os.path.join(META, "stats.npz"))["kappa_offsets"].astype(np.float32)
# SINDYc diverges; its curve is read from a CPU-precomputed cache rather than run
# in-process — pysindy alongside torch segfaults. Regenerate via eval_stability_log.py.
SINDYC_CACHE = _os.path.join(_os.path.dirname(__file__), "..", "figures", "_sindyc_curve.npy")


def build_state31(phys, wp_world, t0, T):
    seg = phys[t0:t0 + T]
    rel = to_relative_np(seg, ref_idx=0)
    car = (rel - PHYSICS_MEAN_REL) / PHYSICS_STD_REL
    pos0 = seg[0, :2]; yaw0 = seg[0, 2]
    c, s = np.cos(yaw0), np.sin(yaw0)
    dd = wp_world[t0:t0 + T] - pos0
    x = dd[..., 0] * c + dd[..., 1] * s
    y = -dd[..., 0] * s + dd[..., 1] * c
    lane = (np.stack([x, y], -1).reshape(T, LANE_DIM) - LANE_MEAN) / LANE_STD
    return np.concatenate([car, lane], -1).astype(np.float32)


@torch.no_grad()
def rollout_baseline(model, z0, acts):
    z = torch.tensor(z0).unsqueeze(0)
    xy = [z0[:2]]
    for k in range(K):
        out = model(z, torch.tensor(acts[k]).unsqueeze(0))
        z = out[0] if isinstance(out, tuple) else out
        xy.append(z[0, :2].numpy())
    return np.array(xy)        # (K+1,2) normalized rel


@torch.no_grad()
def rollout_frenet(dyn, z0, acts):
    z = torch.tensor(z0).unsqueeze(0)
    out = [z0.copy()]
    for k in range(K):
        z = dyn(z, torch.tensor(acts[k]).unsqueeze(0))
        out.append(z[0].numpy())
    return np.array(out)       # (K+1,5)


def main():
    base = SeqLaneDataset(DATA_DIR, seq_len=2)
    _, val_eps = split_segments(base, val_frac=0.10, seed=0)
    fr_files = sorted([f for f in glob.glob(_os.path.join(FRENET_DIR, "*.npz")) if "_meta" not in f])
    print(f"val segments: {sorted(val_eps)}")

    goku = DynamicsGOKULane(); goku.load_state_dict(load_checkpoint("checkpoints/goku_lane_donkey_longK/best.tar")["model"]); goku.eval()
    dvbf = DynamicsDVBFLane(); dvbf.load_state_dict(load_checkpoint("checkpoints/dvbf_lane_donkey_longK/best.tar")["model"]); dvbf.eval()
    v2p = DynamicsVid2ParamLane(); v2p.load_state_dict(load_checkpoint("checkpoints/v2p_lane_donkey_longK/best.tar")["model"]); v2p.eval()
    enc = PhysicsEncoderLane(); enc.load_state_dict(load_checkpoint("checkpoints/piwm_lane_v6_donkey/ae.tar")["encoder"]); enc.eval()
    for p in enc.parameters(): p.requires_grad = False
    fr = FrenetDynamics(_os.path.join(META, "track.npz"), _os.path.join(META, "stats.npz"))
    fr.load_state_dict(load_checkpoint("checkpoints/frenet/dyn_k16.tar")["dynamics"]); fr.eval()
    # native road-context perception (kappa from the front camera, NOT the map) -> pure WM
    kenc = RoadContextEncoder(n_offsets=len(OFFSETS), freeze_backbone=True)
    kenc.load_state_dict(load_checkpoint("checkpoints/frenet/kappa_scratch.tar")["model"]); kenc.eval()

    FS = LANE_FRAME_STACK
    std_xy = PHYSICS_STD_REL[:2]                 # working units
    res = {"Frenet-perc": [], "Frenet-oracle": [], "GOKU": [], "DVBF": [], "V2P": []}
    rng = np.random.default_rng(1)
    for ep in sorted(val_eps):
        phys = base.phys_list[ep]; acts31 = base.acts_list[ep]; wpw = base.wp_world_list[ep]
        imgs = base.imgs_list[ep]
        frd = np.load(fr_files[ep]); fst = frd["state"]; fac = frd["action"]; fim = frd["imgs"].astype(np.float32)
        T = min(len(phys), len(fst))
        if T < K + 1 or T < FS: continue
        cands = np.arange(FS - 1, T - K - 1, 4)   # dense stride for low-variance estimate
        if len(cands) == 0: continue
        for t0 in cands:
            t0 = int(t0)
            s31 = build_state31(phys, wpw, t0, K + 1)
            for name, m in [("GOKU", goku), ("DVBF", dvbf)]:
                xy_n = rollout_baseline(m, s31[0], acts31[t0:t0 + K])
                e = np.linalg.norm((xy_n - s31[:, :2]) * std_xy / SCALE, axis=-1)
                res[name].append(e)
            # V2P: theta from encoder over the 15-frame stack ending at t0; GT z0 state
            with torch.no_grad():
                stack0 = np.stack([imgs[t0 - FS + 1 + j] for j in range(FS)], 0)[None]
                obs = enc(torch.tensor(stack0, dtype=torch.float32)).unsqueeze(1)
                theta, _, _ = v2p.infer_theta(obs)
                z = torch.tensor(s31[0]).unsqueeze(0); xy = [s31[0, :2]]
                for k in range(K):
                    z = v2p.step(z, torch.tensor(acts31[t0 + k]).unsqueeze(0), theta)
                    xy.append(z[0, :2].numpy())
                e = np.linalg.norm((np.array(xy) - s31[:, :2]) * std_xy / SCALE, axis=-1)
                res["V2P"].append(e)
            gt = fst[t0:t0 + K + 1]
            # (a) Frenet-oracle: kappa(s) from the KNOWN map every step (privileged ablation)
            fp = rollout_frenet(fr, fst[t0], fac[t0:t0 + K])
            res["Frenet-oracle"].append(
                np.linalg.norm(sd2xy(fp[:, 0], fp[:, 1]) - sd2xy(gt[:, 0], gt[:, 1]), axis=-1))
            # (b) Frenet-perc: kappa PERCEIVED from the t0 front-cam stack (pure world model)
            with torch.no_grad():
                stack = torch.tensor(fim[t0 - FS + 1:t0 + 1], dtype=torch.float32).unsqueeze(0)
                prof = kenc(stack)[0].numpy().astype(np.float32)
            fpp = fr.rollout_perceived(torch.tensor(fst[t0]), torch.tensor(fac[t0:t0 + K]),
                                       torch.tensor(prof), OFFSETS).numpy()
            res["Frenet-perc"].append(
                np.linalg.norm(sd2xy(fpp[:, 0], fpp[:, 1]) - sd2xy(gt[:, 0], gt[:, 1]), axis=-1))

    print(f"\n{'model':<14} {'n':>4} {'xy@25':>8} {'xy@50':>8} {'xy@100 med':>11} {'xy@100 mean':>12}")
    print("-" * 62)
    for name in ["Frenet-perc", "Frenet-oracle", "GOKU", "V2P", "DVBF"]:
        E = np.array(res[name])
        print(f"{name:<14} {len(E):>4} {E[:,25].mean():>7.3f}m {E[:,50].mean():>7.3f}m "
              f"{np.median(E[:,100]):>10.3f}m {E[:,100].mean():>11.3f}m")
    sc_curve = np.load(SINDYC_CACHE) if _os.path.exists(SINDYC_CACHE) else None   # cached mean (K+1,)
    if sc_curve is not None:
        print(f"{'SINDYc':<14} {'--':>4} {'diverges':>8} {'diverges':>8} "
              f"{'~1e17':>10} {'(clip 100m)':>11}")
    print("\n(all from GT init, same val windows, real metres — lower=better)")
    print("Frenet-perc = pure WM (kappa from camera); Frenet-oracle = privileged map lookup")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sty = {"Frenet-perc": ("#5B2C9B", 3.0, "-"), "Frenet-oracle": ("#9B7FC9", 2.0, ":"),
           "GOKU": ("#9DDA9D", 2.0, "--"), "V2P": ("#F2A24C", 2.0, "--"),
           "DVBF": ("#F08AB1", 2.0, "--"), "SINDYc": ("#C0392B", 1.8, ":")}
    steps = np.arange(K + 1)
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    ymax = 0.0
    for name in ["Frenet-perc", "Frenet-oracle", "GOKU", "V2P", "DVBF"]:
        E = np.array(res[name]); m = E.mean(0); sd = E.std(0) / np.sqrt(len(E))
        c, lw, ls = sty[name]
        ax.plot(steps, m, color=c, lw=lw, ls=ls, label=f"{name} (@100={m[100]:.3f}m)")
        ax.fill_between(steps, m - sd, m + sd, color=c, alpha=0.15)
        ymax = max(ymax, float((m + sd).max()))
    # SINDYc (from CPU-cached curve) plotted, but the y-axis is capped to the BOUNDED
    # models' max so SINDYc shoots off the top instead of squashing every curve flat.
    ax.set_ylim(0, ymax * 1.10)
    if sc_curve is not None:
        c, lw, ls = sty["SINDYc"]
        ax.plot(steps, sc_curve, color=c, lw=lw, ls=ls, label="SINDYc (diverges →∞)")
        ax.text(K * 0.04, ymax * 1.06, "SINDYc ↑ diverges (off scale)", color=c,
                fontsize=10, va="top")
    ax.set_title("xy error vs rollout step (mean ± SE, GT init)", fontsize=13)
    ax.set_xlabel("rollout step", fontsize=12); ax.set_ylabel("position error (m)", fontsize=12)
    ax.grid(alpha=.3); ax.legend(fontsize=11, loc="upper left"); ax.set_xlim(0, K)
    _os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig_frenet_vs_baselines.png", dpi=140, bbox_inches="tight")
    print("saved -> figures/fig_frenet_vs_baselines.png")


if __name__ == "__main__":
    main()
