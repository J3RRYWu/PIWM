"""Publication-quality figure generator (Elsevier elsarticle style).

Recomputes the three paper figures (GPU-batched) and renders them with a clean,
consistent style, exporting vector PDF + PNG:
  fig_main.pdf       — 100-step rollout, Frenet vs GOKU/V2P/DVBF (mean ± SE)
  fig_stability.pdf  — log-scale rollout error incl. SINDYc (divergence)
  fig_noise.pdf      — (a) matched-noise robustness  (b) per-physical-quantity

Run with the CUDA build:  <py311>\python.exe paper_figures.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "train"))
import donkey_config; donkey_config.patch_globals()

import glob, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib as mpl; mpl.use("Agg")
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

# ---------------- publication style ----------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.8, "lines.markersize": 4,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})
# consistent colourblind-safe palette
COL = {"Frenet": "#3B33A0", "GOKU": "#2E9E4F", "V2P": "#E8820C",
       "DVBF": "#C0457B", "SINDYc": "#C0392B"}
LBL = {"Frenet": "PIWM-Frenet (ours)", "GOKU": "GOKU", "V2P": "Vid2Param",
       "DVBF": "DVBF", "SINDYc": "SINDYc"}
std_xy = PHYSICS_STD_REL[:2]
rng = np.random.default_rng(0)
_os.makedirs("figures", exist_ok=True)


@torch.no_grad()
def frenet_curve(fr, z0, fac, gtxy):
    z = torch.tensor(z0, dtype=torch.float32, device=DEVICE)
    a = torch.tensor(fac, dtype=torch.float32, device=DEVICE)
    sd = [z[:, :2].cpu().numpy()]
    for k in range(K):
        z = fr(z, a[:, k]); sd.append(z[:, :2].cpu().numpy())
    sd = np.stack(sd, 1); xy = sd2xy(sd[..., 0], sd[..., 1])
    return np.linalg.norm(xy - gtxy, axis=-1)          # (B,K+1) metres


@torch.no_grad()
def base_curve(m, z0, act, gtN, theta=None):
    z = torch.tensor(z0, dtype=torch.float32, device=DEVICE)
    a = torch.tensor(act, dtype=torch.float32, device=DEVICE)
    e = [np.linalg.norm((z[:, :2].cpu().numpy() - gtN[:, 0]) * std_xy / SCALE, axis=-1)]
    for k in range(K):
        z = m.step(z, a[:, k], theta) if theta is not None else m(z, a[:, k])
        e.append(np.linalg.norm((z[:, :2].cpu().numpy() - gtN[:, k + 1]) * std_xy / SCALE, axis=-1))
    return np.stack(e, 1)                                # (B,K+1)


def sindyc_curve(sindy, s31, act):
    out = []
    for i in range(len(s31)):
        z = s31[i, 0:1].copy().astype(np.float64); errs = [0.0]; dead = False
        for k in range(K):
            if not dead:
                try: z = np.asarray(sindy.predict(z, u=act[i, k:k+1]), dtype=np.float64)
                except Exception: dead = True
                if not np.isfinite(z).all() or np.abs(z).max() > 1e4: dead = True
            errs.append(1e2 if dead else min(np.linalg.norm((z[0, :2] - s31[i, k+1, :2]) * std_xy / SCALE), 1e2))
        out.append(errs)
    return np.array(out)


def ms(x):  # (B,K+1)->mean,SE over B
    return x.mean(0), x.std(0) / np.sqrt(len(x))


def main():
    print("DEVICE =", DEVICE)
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
    # SINDYc curve is loaded from a CPU-precomputed cache (figures/_sindyc_curve.npy)
    # to avoid importing pysindy alongside torch+cuda, which segfaults on py311.
    # Regenerate the cache with:  python src/eval_stability_log.py   (CPU / py3.13)
    fr_std = fr.state_std.cpu().numpy().copy(); fr_std[0] = 0.5

    s31, fst0, fac, act31, gtxy, theta = [], [], [], [], [], []
    for ep in sorted(val_eps):
        phys = base.phys_list[ep]; a31 = base.acts_list[ep]; wpw = base.wp_world_list[ep]; imgs = base.imgs_list[ep]
        fs = np.load(fr_files[ep])["state"]; fc = np.load(fr_files[ep])["action"]
        T = min(len(phys), len(fs), len(imgs))
        for t0 in range(FS - 1, T - K - 1, 4):
            t0 = int(t0)
            s31.append(build_state31(phys, wpw, t0, K + 1)); fst0.append(fs[t0].copy())
            fac.append(fc[t0:t0+K].copy()); act31.append(a31[t0:t0+K].copy())
            gtxy.append(sd2xy(fs[t0:t0+K+1, 0], fs[t0:t0+K+1, 1]))
            st = np.stack([imgs[t0-FS+1+j] for j in range(FS)], 0)[None]
            with torch.no_grad():
                th, _, _ = v2p.infer_theta(enc(torch.tensor(st, dtype=torch.float32, device=DEVICE)).unsqueeze(1))
            theta.append(th[0].cpu().numpy())
    s31 = np.stack(s31); fst0 = np.stack(fst0); fac = np.stack(fac)
    act31 = np.stack(act31); gtxy = np.stack(gtxy); theta = np.stack(theta); gtN = s31[:, :, :2]
    th_t = torch.tensor(theta, dtype=torch.float32, device=DEVICE)
    Wn = len(s31); steps = np.arange(K + 1); print(f"{Wn} windows")

    # ===== Fig 1: main comparison (linear, mean ± SE) =====
    curves = {"Frenet": frenet_curve(fr, fst0, fac, gtxy),
              "GOKU": base_curve(goku, s31[:, 0], act31, gtN),
              "V2P": base_curve(v2p, s31[:, 0], act31, gtN, th_t),
              "DVBF": base_curve(dvbf, s31[:, 0], act31, gtN)}
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for nm in ["Frenet", "V2P", "GOKU", "DVBF"]:
        mu, se = ms(curves[nm]); lw = 2.4 if nm == "Frenet" else 1.6
        ls = "-" if nm == "Frenet" else "--"
        ax.plot(steps, mu, color=COL[nm], lw=lw, ls=ls, label=f"{LBL[nm]} ({mu[K]:.2f} m)")
        ax.fill_between(steps, mu - se, mu + se, color=COL[nm], alpha=0.15, lw=0)
    ax.set_xlabel("rollout step"); ax.set_ylabel("position error (m)"); ax.set_xlim(0, K); ax.set_ylim(bottom=0)
    ax.legend(loc="upper left"); fig.savefig("figures/fig_main.pdf"); fig.savefig("figures/fig_main.png"); plt.close(fig)

    # ===== Fig 2: stability log (incl SINDYc) =====
    sc_path = "figures/_sindyc_curve.npy"
    sc_mean = np.load(sc_path) if _os.path.exists(sc_path) else None      # precomputed mean curve (K+1,)
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    order2 = ["SINDYc", "DVBF", "GOKU", "V2P", "Frenet"] if sc_mean is not None else ["DVBF", "GOKU", "V2P", "Frenet"]
    for nm in order2:
        m = sc_mean if nm == "SINDYc" else curves[nm].mean(0)
        lw = 2.4 if nm == "Frenet" else 1.6; ls = "-" if nm == "Frenet" else (":" if nm == "SINDYc" else "--")
        tag = LBL[nm] + (" (diverges)" if nm == "SINDYc" else "")
        ax.plot(steps, np.maximum(m, 1e-3), color=COL[nm], lw=lw, ls=ls, label=tag)
    ax.set_yscale("log"); ax.set_xlim(0, K); ax.set_xlabel("rollout step"); ax.set_ylabel("position error (m, log)")
    ax.legend(loc="lower right", ncol=1); fig.savefig("figures/fig_stability.pdf"); fig.savefig("figures/fig_stability.png"); plt.close(fig)

    # ===== Fig 3: noise robustness (a)+(b) =====
    N = 10
    def tile(x): return np.repeat(x, N, 0)
    fac_b, act_b, gtxy_b, gtN_b = tile(fac), tile(act31), tile(gtxy), tile(gtN)
    th_b = torch.tensor(tile(theta), dtype=torch.float32, device=DEVICE)
    sigmas = [0.0, 0.25, 0.5, 1.0, 1.5]; MA = ["Frenet", "V2P", "GOKU", "DVBF"]
    A = {m: ([], []) for m in MA}
    for sig in sigmas:
        zf = tile(fst0) + rng.standard_normal((Wn*N, 5)).astype(np.float32) * fr_std * sig
        eF = frenet_curve(fr, zf, fac_b, gtxy_b)[:, K].reshape(Wn, N).mean(1)
        zb = tile(s31[:, 0]) + rng.standard_normal((Wn*N, 31)).astype(np.float32) * sig
        vals = {"Frenet": eF}
        for nm, m, t in [("GOKU", goku, None), ("V2P", v2p, th_b), ("DVBF", dvbf, None)]:
            vals[nm] = base_curve(m, zb, act_b, gtN_b, t)[:, K].reshape(Wn, N).mean(1)
        for nm in MA: A[nm][0].append(vals[nm].mean()); A[nm][1].append(vals[nm].std()/np.sqrt(Wn))
    dims = {"localization $s$": (0, [0, .02, .05, .1, .2]), "CTE $d$": (1, [0, .02, .05, .1, .2]),
            "heading $\\psi_e$": (2, [0, .05, .1, .2, .35]), "speed $v$": (3, [0, .05, .1, .2, .4]),
            "yaw-rate $\\omega$": (4, [0, .1, .25, .5, 1.0])}
    Bv = {}
    for q, (di, mags) in dims.items():
        mu, se = [], []
        for mg in mags:
            zf = tile(fst0).copy(); zf[:, di] += rng.standard_normal(Wn*N).astype(np.float32) * mg
            e = frenet_curve(fr, zf, fac_b, gtxy_b)[:, K].reshape(Wn, N).mean(1)
            mu.append(e.mean()); se.append(e.std()/np.sqrt(Wn))
        Bv[q] = (mags, np.array(mu), np.array(se))

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for nm in MA:
        mu = np.array(A[nm][0]); se = np.array(A[nm][1]); lw = 2.4 if nm == "Frenet" else 1.6
        axs[0].plot(sigmas, mu, "-o", color=COL[nm], lw=lw, label=LBL[nm])
        axs[0].fill_between(sigmas, mu-se, mu+se, color=COL[nm], alpha=0.15, lw=0)
    axs[0].set_xlabel("init-state noise  $\\sigma$  (std)"); axs[0].set_ylabel("position error @100 (m)")
    axs[0].set_title("(a) cross-model robustness"); axs[0].legend(loc="upper left")
    qcol = plt.cm.viridis(np.linspace(0.05, 0.85, len(dims)))
    for (q, (mags, mu, se)), c in zip(Bv.items(), qcol):
        axs[1].plot(mags, mu, "-o", color=c, lw=1.8, label=q)
        axs[1].fill_between(mags, mu-se, mu+se, color=c, alpha=0.15, lw=0)
    axs[1].set_xlabel("physical noise std (m / rad / m·s$^{-1}$)"); axs[1].set_ylabel("position error @100 (m)")
    axs[1].set_title("(b) per-quantity sensitivity (Frenet)"); axs[1].legend(loc="upper left", fontsize=7.5)
    fig.savefig("figures/fig_noise.pdf"); fig.savefig("figures/fig_noise.png"); plt.close(fig)

    print("saved figures/fig_main.{pdf,png}  fig_stability.{pdf,png}  fig_noise.{pdf,png}")


if __name__ == "__main__":
    main()
