"""Conference Fig 3 replication on the LATEST models: prediction error vs rollout
horizon for Frenet-perc + GOKU/V2P/DVBF, across weak-supervision noise δ ∈ {0,5%,10%}.

Each model is trained at each δ (train_frenet.py / train_baselines_donkey.py --delta);
Frenet is rolled out with PERCEIVED κ (scratch encoder). One panel per δ, same val
windows + GT init. Run from piwm/ with the default python (CPU; baselines load fine).
"""
import os, sys, glob
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "train"))
import donkey_config; donkey_config.patch_globals()

from config import DATA_DIR, PHYSICS_STD_REL
from lane_utils import LANE_FRAME_STACK as FS
from donkey_config import DONKEY_SPATIAL_SCALE as SCALE
from donkey_dataset import split_segments
from train_piwm_lane_v5 import SeqLaneDataset
from baselines.shared_dynamics_lane import (DynamicsGOKULane, DynamicsVid2ParamLane,
                                            DynamicsDVBFLane)
from models.frenet_dynamics import FrenetDynamics
from models.encoder_lane import PhysicsEncoderLane
from models.road_perception import RoadContextEncoder
from utils import load_checkpoint
from eval_frenet_vs_baselines import build_state31, sd2xy, rollout_baseline, OFFSETS, META, FRENET_DIR, K

STD_XY = PHYSICS_STD_REL[:2]
LEVELS = [("δ=0%", ""), ("δ=5%", "_delta5"), ("δ=10%", "_delta10")]
COL = {"Frenet": "#3B33A0", "GOKU": "#2E9E4F", "V2P": "#E8820C", "DVBF": "#C0457B"}


def bck(name, sfx):  # baseline checkpoint: clean=_longK, noisy=_delta5/_delta10
    d = f"{name}_lane_donkey" + ("_longK" if sfx == "" else sfx)
    return f"checkpoints/{d}/best.tar"


def frck(sfx):       # frenet dynamics: clean=dyn_k16, noisy=dyn_k16_delta{5,10}
    return "checkpoints/frenet/dyn_k16.tar" if sfx == "" else f"checkpoints/frenet/dyn_k16{sfx}.tar"


def eval_level(sfx, base, fr_files, val_eps, enc, kenc):
    goku = DynamicsGOKULane(); goku.load_state_dict(load_checkpoint(bck("goku", sfx))["model"]); goku.eval()
    dvbf = DynamicsDVBFLane(); dvbf.load_state_dict(load_checkpoint(bck("dvbf", sfx))["model"]); dvbf.eval()
    v2p = DynamicsVid2ParamLane(); v2p.load_state_dict(load_checkpoint(bck("v2p", sfx))["model"]); v2p.eval()
    fr = FrenetDynamics(os.path.join(META, "track.npz"), os.path.join(META, "stats.npz"))
    fr.load_state_dict(load_checkpoint(frck(sfx))["dynamics"]); fr.eval()

    res = {m: [] for m in ["Frenet", "GOKU", "V2P", "DVBF"]}
    for ep in sorted(val_eps):
        phys = base.phys_list[ep]; acts31 = base.acts_list[ep]; wpw = base.wp_world_list[ep]
        imgs = base.imgs_list[ep]
        frd = np.load(fr_files[ep]); fst = frd["state"]; fac = frd["action"]; fim = frd["imgs"].astype(np.float32)
        T = min(len(phys), len(fst), len(fim))
        for t0 in range(FS - 1, T - K - 1, 4):
            t0 = int(t0)
            s31 = build_state31(phys, wpw, t0, K + 1)
            gt_sd = fst[t0:t0 + K + 1]
            gt_xy = sd2xy(gt_sd[:, 0], gt_sd[:, 1])
            # GOKU / DVBF: GT 31-dim init, rollout
            for nm, m in [("GOKU", goku), ("DVBF", dvbf)]:
                xy_n = rollout_baseline(m, s31[0], acts31[t0:t0 + K])
                res[nm].append(np.linalg.norm((xy_n - s31[:, :2]) * STD_XY / SCALE, axis=-1))
            # V2P: theta from clean encoder over the 15-frame stack, then rollout
            with torch.no_grad():
                stack0 = np.stack([imgs[t0 - FS + 1 + j] for j in range(FS)], 0)[None]
                theta, _, _ = v2p.infer_theta(enc(torch.tensor(stack0, dtype=torch.float32)).unsqueeze(1))
                z = torch.tensor(s31[0]).unsqueeze(0); xy = [s31[0, :2]]
                for k in range(K):
                    z = v2p.step(z, torch.tensor(acts31[t0 + k]).unsqueeze(0), theta); xy.append(z[0, :2].numpy())
                res["V2P"].append(np.linalg.norm((np.array(xy) - s31[:, :2]) * STD_XY / SCALE, axis=-1))
                # Frenet: perceived kappa from the t0 front-cam stack
                prof = kenc(torch.tensor(fim[t0 - FS + 1:t0 + 1]).unsqueeze(0))[0].numpy().astype(np.float32)
            F = fr.rollout_perceived(torch.tensor(fst[t0]), torch.tensor(fac[t0:t0 + K]),
                                     torch.tensor(prof), OFFSETS).numpy()
            res["Frenet"].append(np.linalg.norm(sd2xy(F[:, 0], F[:, 1]) - gt_xy, axis=-1))
    return {m: np.array(v) for m, v in res.items()}


def main():
    base = SeqLaneDataset(DATA_DIR, seq_len=2)
    _, val_eps = split_segments(base, val_frac=0.10, seed=0)
    fr_files = sorted([f for f in glob.glob(os.path.join(FRENET_DIR, "*.npz")) if "_meta" not in f])
    enc = PhysicsEncoderLane(); enc.load_state_dict(load_checkpoint("checkpoints/piwm_lane_v6_donkey/ae.tar")["encoder"]); enc.eval()
    for p in enc.parameters(): p.requires_grad = False
    kenc = RoadContextEncoder(n_offsets=len(OFFSETS), freeze_backbone=True)
    kenc.load_state_dict(load_checkpoint("checkpoints/frenet/kappa_scratch.tar")["model"]); kenc.eval()

    steps = np.arange(K + 1)
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.1), sharey=True, constrained_layout=True)
    print(f"{'level':<8} {'Frenet':>8} {'GOKU':>8} {'V2P':>8} {'DVBF':>8}   (xy@100 m)")
    for ax, (title, sfx) in zip(axs, LEVELS):
        R = eval_level(sfx, base, fr_files, val_eps, enc, kenc)
        row = []
        for nm in ["Frenet", "GOKU", "V2P", "DVBF"]:
            E = R[nm]; m = E.mean(0); se = E.std(0) / np.sqrt(len(E))
            lw = 2.6 if nm == "Frenet" else 1.6; ls = "-" if nm == "Frenet" else "--"
            ax.plot(steps, m, color=COL[nm], lw=lw, ls=ls, label=f"{nm} ({m[100]:.2f})")
            ax.fill_between(steps, m - se, m + se, color=COL[nm], alpha=0.15, lw=0)
            row.append(m[100])
        ax.set_title(title); ax.set_xlabel("rollout step"); ax.set_xlim(0, K); ax.set_ylim(bottom=0)
        ax.grid(alpha=.3); ax.legend(loc="upper left", fontsize=8)
        print(f"{title:<8} " + " ".join(f"{x:>7.3f}m" for x in row))
    axs[0].set_ylabel("position error (m)")
    fig.suptitle("Prediction error vs horizon under δ weak-supervision noise (perceived-κ Frenet vs baselines)",
                 fontsize=11)
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig_delta_all_models.png", dpi=150, bbox_inches="tight")
    fig.savefig("figures/fig_delta_all_models.pdf", bbox_inches="tight")
    print("\nsaved figures/fig_delta_all_models.{png,pdf}")


if __name__ == "__main__":
    main()
