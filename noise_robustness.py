"""Noise-robustness study (journal extension, DonkeyCar case) — BATCHED + GPU.

Two experiments, 100-step rollout, position error in metres, N noise draws per
(window, level) AVERAGED with +/-SE bands:

  A) MATCHED initial-state noise across models — inject sigma standard-deviations
     of Gaussian noise into the initial state, sweep sigma; Frenet vs GOKU/V2P/DVBF.
  B) PER-PHYSICAL-QUANTITY sensitivity (Frenet only) — physical-unit noise into ONE
     quantity at a time (CTE d, heading psi_e, speed v, yaw-rate omega, localization s).

Speed: all N draws for all windows are rolled out as ONE big batch (per level),
so the per-step Python loop runs once instead of (windows x draws x steps) times.
Run with the CUDA build:  <py311>\python.exe noise_robustness.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "train"))
import donkey_config; donkey_config.patch_globals()

import glob
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA_DIR, PHYSICS_STD_REL, DEVICE
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

print("DEVICE =", DEVICE)
rng = np.random.default_rng(0)
std_xy = PHYSICS_STD_REL[:2]
N_DRAW = 10


@torch.no_grad()
def roll_frenet_xyerr(fr, z0, acts, gt_xy):
    """z0 (B,5), acts (B,K,2), gt_xy (B,K+1,2) -> (B,) xy error @K (metres)."""
    z = torch.tensor(z0, dtype=torch.float32, device=DEVICE)
    a = torch.tensor(acts, dtype=torch.float32, device=DEVICE)
    sd = [z[:, :2].cpu().numpy()]
    for k in range(K):
        z = fr(z, a[:, k])
        sd.append(z[:, :2].cpu().numpy())
    sd = np.stack(sd, 1)                                  # (B,K+1,2) = (s,d)
    xy = sd2xy(sd[..., 0], sd[..., 1])                    # (B,K+1,2)
    return np.linalg.norm(xy[:, K] - gt_xy[:, K], axis=-1)


@torch.no_grad()
def roll_base_xyerr(m, z0, acts, gt_norm, theta=None):
    """z0 (B,31), acts (B,K,2), gt_norm (B,K+1,2) -> (B,) xy error @K (metres)."""
    z = torch.tensor(z0, dtype=torch.float32, device=DEVICE)
    a = torch.tensor(acts, dtype=torch.float32, device=DEVICE)
    for k in range(K):
        z = m.step(z, a[:, k], theta) if theta is not None else m(z, a[:, k])
    xy = z[:, :2].cpu().numpy()
    return np.linalg.norm((xy - gt_norm[:, K]) * std_xy / SCALE, axis=-1)


def main():
    base = SeqLaneDataset(DATA_DIR, seq_len=2)
    _, val_eps = split_segments(base, val_frac=0.10, seed=0)
    fr_files = sorted([f for f in glob.glob(_os.path.join(FRENET_DIR, "*.npz")) if "_meta" not in f])

    goku = DynamicsGOKULane().to(DEVICE); goku.load_state_dict(load_checkpoint("checkpoints/goku_lane_donkey_longK/best.tar")["model"]); goku.eval()
    v2p = DynamicsVid2ParamLane().to(DEVICE); v2p.load_state_dict(load_checkpoint("checkpoints/v2p_lane_donkey_longK/best.tar")["model"]); v2p.eval()
    dvbf = DynamicsDVBFLane().to(DEVICE); dvbf.load_state_dict(load_checkpoint("checkpoints/dvbf_lane_donkey_longK/best.tar")["model"]); dvbf.eval()
    enc = PhysicsEncoderLane().to(DEVICE); enc.load_state_dict(load_checkpoint("checkpoints/piwm_lane_v6_donkey/ae.tar")["encoder"]); enc.eval()
    for p in enc.parameters(): p.requires_grad = False
    fr = FrenetDynamics(_os.path.join(META, "track.npz"), _os.path.join(META, "stats.npz")).to(DEVICE)
    fr.load_state_dict(load_checkpoint("checkpoints/frenet/dyn_k16.tar")["dynamics"]); fr.eval()
    fr_std = fr.state_std.cpu().numpy().copy(); fr_std[0] = 0.5

    # ---- precompute per-window arrays ----
    W_s31, W_fst, W_fac, W_act31, W_gtxy, W_theta = [], [], [], [], [], []
    for ep in sorted(val_eps):
        phys = base.phys_list[ep]; acts31 = base.acts_list[ep]; wpw = base.wp_world_list[ep]
        imgs = base.imgs_list[ep]
        fst = np.load(fr_files[ep])["state"]; fac = np.load(fr_files[ep])["action"]
        T = min(len(phys), len(fst), len(imgs))
        for t0 in range(FS - 1, T - K - 1, 10):
            t0 = int(t0)
            W_s31.append(build_state31(phys, wpw, t0, K + 1))
            W_fst.append(fst[t0].copy()); W_fac.append(fac[t0:t0 + K].copy())
            W_act31.append(acts31[t0:t0 + K].copy())
            W_gtxy.append(sd2xy(fst[t0:t0+K+1, 0], fst[t0:t0+K+1, 1]))
            stack0 = np.stack([imgs[t0 - FS + 1 + j] for j in range(FS)], 0)[None]
            with torch.no_grad():
                th, _, _ = v2p.infer_theta(enc(torch.tensor(stack0, dtype=torch.float32, device=DEVICE)).unsqueeze(1))
            W_theta.append(th[0].cpu().numpy())
    Wn = len(W_s31); print(f"{Wn} windows, N={N_DRAW} draws")
    s31 = np.stack(W_s31); fst0 = np.stack(W_fst); fac = np.stack(W_fac)
    act31 = np.stack(W_act31); gtxy = np.stack(W_gtxy); gtN = s31[:, :, :2]
    theta_np = np.stack(W_theta)

    def tile(x): return np.repeat(x, N_DRAW, axis=0)      # (Wn*N, ...)
    fac_b = tile(fac); act31_b = tile(act31); gtxy_b = tile(gtxy); gtN_b = tile(gtN)
    theta_b = torch.tensor(tile(theta_np), dtype=torch.float32, device=DEVICE)

    def mse(per_win):  # per_win (Wn,) -> mean, se
        return per_win.mean(), per_win.std() / np.sqrt(len(per_win))

    # ---------- Experiment A ----------
    sigmas = [0.0, 0.25, 0.5, 1.0, 1.5]
    MODELS_A = ["Frenet", "GOKU", "V2P", "DVBF"]
    A = {m: {} for m in MODELS_A}
    for sig in sigmas:
        eF = rng.standard_normal((Wn * N_DRAW, 5)).astype(np.float32) * fr_std * sig
        z0f = tile(fst0) + eF
        errF = roll_frenet_xyerr(fr, z0f, fac_b, gtxy_b).reshape(Wn, N_DRAW).mean(1)
        A["Frenet"][sig] = errF
        e31 = rng.standard_normal((Wn * N_DRAW, 31)).astype(np.float32) * sig
        z0b = tile(s31[:, 0]) + e31
        for nm, m, th in [("GOKU", goku, None), ("V2P", v2p, theta_b), ("DVBF", dvbf, None)]:
            err = roll_base_xyerr(m, z0b, act31_b, gtN_b, th).reshape(Wn, N_DRAW).mean(1)
            A[nm][sig] = err

    print(f"\nExperiment A — pos@100 (m), N={N_DRAW} draws averaged (mean±SE over {Wn} windows):")
    print(f"  {'sigma':>6} " + " ".join(f"{m:>14}" for m in MODELS_A))
    for s in sigmas:
        print(f"  {s:>6.2f} " + " ".join(f"{mse(A[m][s])[0]:>7.3f}±{mse(A[m][s])[1]:.3f}" for m in MODELS_A))

    # ---------- Experiment B ----------
    dims = {"localization s": (0, [0, 0.02, 0.05, 0.10, 0.20]),
            "CTE d":          (1, [0, 0.02, 0.05, 0.10, 0.20]),
            "heading psi_e":  (2, [0, 0.05, 0.10, 0.20, 0.35]),
            "speed v":        (3, [0, 0.05, 0.10, 0.20, 0.40]),
            "yaw-rate omega": (4, [0, 0.1, 0.25, 0.5, 1.0])}
    B = {q: [] for q in dims}; Bse = {q: [] for q in dims}
    for q, (di, mags) in dims.items():
        for mag in mags:
            z0f = tile(fst0).copy()
            z0f[:, di] += rng.standard_normal(Wn * N_DRAW).astype(np.float32) * mag
            err = roll_frenet_xyerr(fr, z0f, fac_b, gtxy_b).reshape(Wn, N_DRAW).mean(1)
            mu, se = mse(err); B[q].append(mu); Bse[q].append(se)

    print("\nExperiment B — Frenet pos@100 (m) vs single-quantity physical noise:")
    for q, (di, mags) in dims.items():
        print(f"  {q:<16} " + " ".join(f"{m:.3f}" for m in B[q]))

    # ---------- figures ----------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    cset = {"Frenet": "#5B2C9B", "GOKU": "#7FBF7F", "V2P": "#F2A24C", "DVBF": "#F08AB1"}
    for m in MODELS_A:
        mu = np.array([mse(A[m][s])[0] for s in sigmas]); se = np.array([mse(A[m][s])[1] for s in sigmas])
        lw = 3.0 if m == "Frenet" else 2.0
        ax[0].plot(sigmas, mu, "-o", color=cset[m], lw=lw, label=m)
        ax[0].fill_between(sigmas, mu - se, mu + se, color=cset[m], alpha=0.18)
    ax[0].set_xlabel("init-state noise  σ  (std)"); ax[0].set_ylabel("pos error @100 (m)")
    ax[0].set_title(f"(a) Matched init noise — cross-model (N={N_DRAW}±SE)"); ax[0].grid(alpha=.3); ax[0].legend()
    qcol = plt.cm.viridis(np.linspace(0, 0.85, len(dims)))
    for (q, (di, mags)), c in zip(dims.items(), qcol):
        mu = np.array(B[q]); se = np.array(Bse[q])
        ax[1].plot(mags, mu, "-o", color=c, lw=2.2, label=q)
        ax[1].fill_between(mags, mu - se, mu + se, color=c, alpha=0.18)
    ax[1].set_xlabel("physical noise std (m / rad / m·s⁻¹)"); ax[1].set_ylabel("pos error @100 (m)")
    ax[1].set_title("(b) Frenet per-physical-quantity sensitivity"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=9)
    fig.suptitle("Noise robustness — DonkeyCar, 100-step rollout", fontweight="bold")
    _os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig_noise_robustness.png", dpi=140, bbox_inches="tight")
    print("\nsaved figures/fig_noise_robustness.png")


if __name__ == "__main__":
    main()
