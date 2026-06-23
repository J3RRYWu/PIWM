"""Rollout-stability figure on a LOG axis — the only view that can show SINDYc
(which diverges to ~1e17) alongside the bounded learned/structured models.

All 5 models, GT init, 100-step rollout, position error in metres, log-y.
SINDYc is rolled out with a magnitude guard (clip at 1e2 m = "diverged") so its
pysindy C-extension never sees the inf that would abort the process.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "train"))
import donkey_config; donkey_config.patch_globals()

import glob, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA_DIR, PHYSICS_STD_REL
from lane_utils import LANE_FRAME_STACK as FS
from donkey_config import DONKEY_SPATIAL_SCALE as SCALE
from donkey_dataset import split_segments
from train_piwm_lane_v5 import SeqLaneDataset
from baselines.shared_dynamics_lane import (DynamicsGOKULane, DynamicsVid2ParamLane,
                                            DynamicsDVBFLane)
from models.frenet_dynamics import FrenetDynamics
from models.encoder_lane import PhysicsEncoderLane
from utils import load_checkpoint
from eval_frenet_vs_baselines import build_state31, sd2xy, META, FRENET_DIR, K

std_xy = PHYSICS_STD_REL[:2]
CLIP = 1e2          # display ceiling for diverged SINDYc (true blowup ~1e17)


@torch.no_grad()
def roll_nn(m, z0, acts, theta=None):
    z = torch.tensor(z0, dtype=torch.float32).unsqueeze(0); xy = [z0[:2]]
    for k in range(K):
        z = m.step(z, torch.tensor(acts[k]).unsqueeze(0), theta) if theta is not None else m(z, torch.tensor(acts[k]).unsqueeze(0))
        xy.append(z[0, :2].numpy())
    return np.array(xy)


def roll_sindyc(sindy, s31, acts):
    """Guarded SINDYc rollout -> per-step position error (m); clipped at CLIP once
    it diverges (prevents the inf that crashes pysindy)."""
    z = s31[0:1].copy().astype(np.float64)
    errs = [0.0]; dead = False
    for k in range(K):
        if not dead:
            try:
                z = np.asarray(sindy.predict(z, u=acts[k:k+1]), dtype=np.float64)
            except Exception:
                dead = True
            if not np.isfinite(z).all() or np.abs(z).max() > 1e4:
                dead = True
        if dead:
            errs.append(CLIP)
        else:
            errs.append(min(np.linalg.norm((z[0, :2] - s31[k+1, :2]) * std_xy / SCALE), CLIP))
    return np.array(errs)


def main():
    base = SeqLaneDataset(DATA_DIR, seq_len=2)
    _, val_eps = split_segments(base, val_frac=0.10, seed=0)
    fr_files = sorted([f for f in glob.glob(_os.path.join(FRENET_DIR, "*.npz")) if "_meta" not in f])

    goku = DynamicsGOKULane(); goku.load_state_dict(load_checkpoint("checkpoints/goku_lane_donkey_longK/best.tar")["model"]); goku.eval()
    v2p = DynamicsVid2ParamLane(); v2p.load_state_dict(load_checkpoint("checkpoints/v2p_lane_donkey_longK/best.tar")["model"]); v2p.eval()
    dvbf = DynamicsDVBFLane(); dvbf.load_state_dict(load_checkpoint("checkpoints/dvbf_lane_donkey_longK/best.tar")["model"]); dvbf.eval()
    enc = PhysicsEncoderLane(); enc.load_state_dict(load_checkpoint("checkpoints/piwm_lane_v6_donkey/ae.tar")["encoder"]); enc.eval()
    for p in enc.parameters(): p.requires_grad = False
    fr = FrenetDynamics(_os.path.join(META, "track.npz"), _os.path.join(META, "stats.npz"))
    fr.load_state_dict(load_checkpoint("checkpoints/frenet/dyn_k16.tar")["dynamics"]); fr.eval()
    sindy = pickle.load(open("checkpoints/sindyc_lane_donkey/model.pkl", "rb"))

    res = {m: [] for m in ["Frenet", "GOKU", "V2P", "DVBF", "SINDYc"]}
    for ep in sorted(val_eps):
        phys = base.phys_list[ep]; acts31 = base.acts_list[ep]; wpw = base.wp_world_list[ep]
        imgs = base.imgs_list[ep]
        fst = np.load(fr_files[ep])["state"]; fac = np.load(fr_files[ep])["action"]
        T = min(len(phys), len(fst), len(imgs))
        for t0 in range(FS - 1, T - K - 1, 8):
            t0 = int(t0)
            s31 = build_state31(phys, wpw, t0, K + 1)
            stack0 = np.stack([imgs[t0 - FS + 1 + j] for j in range(FS)], 0)[None]
            with torch.no_grad():
                theta, _, _ = v2p.infer_theta(enc(torch.tensor(stack0, dtype=torch.float32)).unsqueeze(1))
            gt_xy = sd2xy(fst[t0:t0+K+1, 0], fst[t0:t0+K+1, 1])
            with torch.no_grad():
                zf = torch.tensor(fst[t0], dtype=torch.float32).unsqueeze(0); F = [fst[t0]]
                for k in range(K):
                    zf = fr(zf, torch.tensor(fac[t0+k]).unsqueeze(0)); F.append(zf[0].numpy())
                F = np.array(F)
            res["Frenet"].append(np.linalg.norm(sd2xy(F[:, 0], F[:, 1]) - gt_xy, axis=-1))
            for nm, m, th in [("GOKU", goku, None), ("V2P", v2p, theta), ("DVBF", dvbf, None)]:
                xy = roll_nn(m, s31[0], acts31[t0:t0+K], th)
                res[nm].append(np.linalg.norm((xy - s31[:, :2]) * std_xy / SCALE, axis=-1))
            res["SINDYc"].append(roll_sindyc(sindy, s31, acts31[t0:t0+K]))

    sty = {"Frenet": ("#5B2C9B", 3.0, "-"), "GOKU": ("#7FBF7F", 2.0, "--"),
           "V2P": ("#F2A24C", 2.0, "--"), "DVBF": ("#F08AB1", 2.0, "--"),
           "SINDYc": ("#C44", 2.2, ":")}
    steps = np.arange(K + 1)
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    print(f"\n{'model':<8} {'@100 (m)':>10}")
    for nm in ["Frenet", "GOKU", "V2P", "DVBF", "SINDYc"]:
        m = np.array(res[nm]).mean(0); c, lw, ls = sty[nm]
        tag = f"{nm} (@100={m[100]:.3f}m)" if nm != "SINDYc" else f"{nm} (diverges → 1e17)"
        ax.plot(steps, np.maximum(m, 1e-3), color=c, lw=lw, ls=ls, label=tag)
        print(f"  {nm:<8} {m[100]:>9.3f}")
    ax.set_yscale("log"); ax.set_xlim(0, K)
    ax.axhspan(CLIP*0.9, CLIP*5, color="#C44", alpha=0.06)
    ax.text(2, CLIP*1.4, "SINDYc clipped (true ~1e17)", color="#C44", fontsize=9)
    ax.set_xlabel("rollout step", fontsize=12); ax.set_ylabel("position error (m, log)", fontsize=12)
    ax.set_title("Rollout stability (log scale) — SINDYc diverges, others bounded", fontsize=12)
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=10, loc="center right")
    _os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig_stability_log.png", dpi=140, bbox_inches="tight")
    print("\nsaved figures/fig_stability_log.png")


if __name__ == "__main__":
    main()
