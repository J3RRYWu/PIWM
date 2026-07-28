"""Conference δ weak-supervision noise experiment, applied to the LATEST model
(perceived-κ Frenet world model). Mirrors the conference Fig 3: prediction error
vs horizon across supervision-noise levels δ ∈ {0, 5%, 10%}.

The Frenet dynamics is trained under biased-uniform δ noise on the state labels
(train_frenet.py --delta {0.05,0.10}); here each is rolled out with PERCEIVED κ
(scratch encoder) on the val windows, plotting xy error vs rollout step.
Run from piwm/ with the default python (CPU; no pysindy/cuda needed)."""
import os, sys, glob
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models.frenet_dynamics import FrenetDynamics
from models.road_perception import RoadContextEncoder
from lane_utils import LANE_FRAME_STACK as FS
from utils import load_checkpoint

K = 100
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
FRENET_DIR = os.path.join(ROOT, "Data_Donkeycar_frenet")
META = os.path.join(FRENET_DIR, "_meta")
_tr = np.load(os.path.join(META, "track.npz"))
_cen, _nrm = _tr["centers"], _tr["normals"]
_L, _ds, _M = float(_tr["total_len"]), float(_tr["grid_ds"]), len(_cen)


def sd2xy(s, d):
    i = (np.mod(s, _L) / _ds).astype(int) % _M
    return _cen[i] + d[..., None] * _nrm[i]


def val_split(files, val_frac=0.10, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(files))
    nval = max(1, int(round(len(files) * val_frac)))
    return set(perm[:nval].tolist())


def load_dyn(path):
    d = FrenetDynamics(os.path.join(META, "track.npz"), os.path.join(META, "stats.npz"))
    d.load_state_dict(load_checkpoint(path)["dynamics"]); d.eval()
    return d


def main():
    offsets = np.load(os.path.join(META, "stats.npz"))["kappa_offsets"].astype(np.float32)
    kenc = RoadContextEncoder(n_offsets=len(offsets), freeze_backbone=True)
    kenc.load_state_dict(load_checkpoint("checkpoints/frenet/kappa_scratch.tar")["model"]); kenc.eval()
    # delta=0 is the clean dyn_k16; 5%/10% are trained under conference weak-supervision noise
    levels = [("0%", "checkpoints/frenet/dyn_k16.tar"),
              ("5%", "checkpoints/frenet/dyn_k16_delta5.tar"),
              ("10%", "checkpoints/frenet/dyn_k16_delta10.tar")]
    dyns = {tag: load_dyn(p) for tag, p in levels}

    files = sorted(f for f in glob.glob(os.path.join(FRENET_DIR, "*.npz")) if "_meta" not in f)
    val_eps = val_split(files)
    print(f"val segs: {sorted(val_eps)}")
    res = {tag: [] for tag, _ in levels}
    for ep in sorted(val_eps):
        d = np.load(files[ep]); st = d["state"]; ac = d["action"]; im = d["imgs"].astype(np.float32)
        T = min(len(st), len(ac) + 1, len(im))
        for t0 in range(FS - 1, T - K - 1, 4):
            z0 = st[t0].astype(np.float32); acts = ac[t0:t0 + K].astype(np.float32)
            gt = st[t0:t0 + K + 1]
            with torch.no_grad():
                prof = kenc(torch.tensor(im[t0 - FS + 1:t0 + 1]).unsqueeze(0))[0].numpy().astype(np.float32)
            for tag, dyn in dyns.items():
                F = dyn.rollout_perceived(torch.tensor(z0), torch.tensor(acts),
                                          torch.tensor(prof), offsets).numpy()
                res[tag].append(np.linalg.norm(sd2xy(F[:, 0], F[:, 1]) - sd2xy(gt[:, 0], gt[:, 1]), axis=-1))

    sty = {"0%": ("#3B33A0", 2.6, "-"), "5%": ("#E8820C", 2.0, "--"), "10%": ("#C0392B", 2.0, ":")}
    steps = np.arange(K + 1)
    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    print(f"\n{'delta':<6} {'xy@50':>8} {'xy@100':>8}")
    for tag, _ in levels:
        E = np.array(res[tag]); m = E.mean(0); se = E.std(0) / np.sqrt(len(E))
        c, lw, ls = sty[tag]
        ax.plot(steps, m, color=c, lw=lw, ls=ls, label=f"delta={tag} (@100={m[100]:.3f} m)")
        ax.fill_between(steps, m - se, m + se, color=c, alpha=0.15, lw=0)
        print(f"{tag:<6} {m[50]:>7.3f}m {m[100]:>7.3f}m")
    ax.set_xlabel("rollout step"); ax.set_ylabel("position error (m)")
    ax.set_xlim(0, K); ax.set_ylim(bottom=0)
    ax.set_title("Frenet (perceived $\\kappa$) under $\\delta$ weak-supervision noise", fontsize=10)
    ax.grid(alpha=.3); ax.legend(loc="upper left", fontsize=9)
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig_delta_supervision.png", dpi=150, bbox_inches="tight")
    fig.savefig("figures/fig_delta_supervision.pdf", bbox_inches="tight")
    print("\nsaved figures/fig_delta_supervision.{png,pdf}")


if __name__ == "__main__":
    main()
