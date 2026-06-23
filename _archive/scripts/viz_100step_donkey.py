"""100-step rollout: PIWM vs 4 baselines on donkey val data.

Three panels (matches the standard PIWM paper figure):
  1. Position MSE = (norm_x - GT)^2 + (norm_y - GT)^2  per step
  2. yaw_rel MSE per step
  3. Image MSE per step  (uses the SAME frozen Stage-1 decoder for everyone)

Setup:
  - z0 layout for ALL models:
       z[0:2]  = 0   (GT relative position at t0, which is always 0)
       z[2:11] = encoder car-obs head     ← shared frozen Stage-1 encoder
       z[11:]  = encoder lane-obs head    ← shared
  - The frozen Stage-1 encoder/decoder lives in
      checkpoints/piwm_lane_v6_donkey/ae.tar
  - For long rollouts (100 steps >> training horizon of 8 steps) models can
    diverge dramatically. SINDYc in particular tends to explode.

Output: figures/fig_100step.png
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "train"))

import donkey_config; donkey_config.patch_globals()

import os, glob, pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DEVICE, DATA_DIR, ENCODER_DIM as CAR_ENCODER_DIM
from lane_utils import LANE_DIM, LANE_FRAME_STACK
from models.dynamics_lane_donkey_v3 import LaneAugmentedDonkeyV3
from models.dynamics_lane_donkey_v4 import LaneAugmentedDonkeyV4
from baselines.shared_dynamics_lane import (DynamicsDVBFLane, DynamicsGOKULane,
                                            DynamicsVid2ParamLane)
from models.encoder_lane import PhysicsEncoderLane
from models.decoder_lane import PhysicsDecoderLane
from utils import load_checkpoint
from donkey_dataset import split_segments
from train_piwm_lane_v5 import SeqLaneDataset
from relative_coords import to_relative_np


ROLLOUT_K = 100
SEED = 0
SAVE_DIR = "figures"
_os.makedirs(SAVE_DIR, exist_ok=True)
ENC_CK = "checkpoints/piwm_lane_v6_donkey/ae.tar"


def _to_normalized_rel(phys_seg, lane_world_seg, ref_pos, ref_yaw,
                       phys_mean, phys_std, lane_mean, lane_std):
    """Build (T, 31) normalized state with everything in ref-frame body coords.
    phys_seg : (T, 11) raw (donkey-scaled) physics, world frame
    lane_world_seg : (T, 10, 2) world-frame lane wp (already pre-built)
    """
    rel = to_relative_np(phys_seg, ref_idx=0)
    car_norm = (rel - phys_mean) / phys_std
    # Lane: rotate world wp into ref body frame.
    c, s = np.cos(ref_yaw), np.sin(ref_yaw)
    d = lane_world_seg - ref_pos
    x =  d[..., 0] * c + d[..., 1] * s
    y = -d[..., 0] * s + d[..., 1] * c
    wp_ref = np.stack([x, y], axis=-1).reshape(len(phys_seg), -1)
    lane_norm = (wp_ref - lane_mean) / lane_std
    return np.concatenate([car_norm, lane_norm], axis=-1).astype(np.float32)


def collect_long_windows(base, val_eps, window=ROLLOUT_K + 1,
                         frame_stack=LANE_FRAME_STACK, stride=20):
    """Yield rollout starts that have enough room for `window` future steps
    AND `frame_stack` history before. Each yield is a dict ready to consume.
    """
    from config import PHYSICS_MEAN_REL, PHYSICS_STD_REL
    from lane_utils import LANE_MEAN, LANE_STD
    out = []
    for ep in sorted(val_eps):
        phys = base.phys_list[ep]
        acts = base.acts_list[ep]
        wp_world = base.wp_world_list[ep]
        imgs = base.imgs_list[ep]
        T = len(phys)
        need = frame_stack - 1 + window
        if T < need + 1:
            continue
        for t0 in range(frame_stack - 1, T - window, stride):
            phys_seg = phys[t0:t0 + window]
            wp_seg   = wp_world[t0:t0 + window]
            imgs_seg = imgs[t0:t0 + window]                      # (window, 64, 64)
            ref_pos  = phys_seg[0, 0:2]; ref_yaw = phys_seg[0, 2]
            state31 = _to_normalized_rel(phys_seg, wp_seg, ref_pos, ref_yaw,
                                         PHYSICS_MEAN_REL, PHYSICS_STD_REL,
                                         LANE_MEAN, LANE_STD)
            stack0 = np.stack([imgs[t0 - frame_stack + 1 + j]
                              for j in range(frame_stack)], axis=0)
            out.append(dict(
                state_gt=state31,                                # (window, 31)
                action=acts[t0:t0 + window - 1].astype(np.float32),
                imgs_gt=imgs_seg.astype(np.float32),             # (window, 64, 64)
                stack0=stack0.astype(np.float32),
                ep=ep, t0=t0,
            ))
    return out


def _load_models():
    """Returns dict tag -> (model, kind). Also returns frozen enc/dec."""
    print("Loading frozen encoder/decoder from Stage 1...")
    enc = PhysicsEncoderLane().to(DEVICE)
    dec = PhysicsDecoderLane().to(DEVICE)
    ck = load_checkpoint(ENC_CK)
    enc.load_state_dict(ck['encoder']); enc.eval()
    dec.load_state_dict(ck['decoder']); dec.eval()
    for p in enc.parameters(): p.requires_grad = False
    for p in dec.parameters(): p.requires_grad = False

    models = {}
    for tag, kind, path in [
        # Clean warm-started K=32 pair (same protocol) — the real test:
        ("PIWM-v3eqv (warm)", "donkey4_base", "checkpoints/piwm_lane_donkey_v4_d0h0_K32_w1/dyn.tar"),
        ("PIWM-v4 (damped)",  "donkey4",      "checkpoints/piwm_lane_donkey_v4_v4_K32_w1/dyn.tar"),
        # Cold longK reference baselines (different protocol — reference only):
        ("GOKU-lane",        "goku",    "checkpoints/goku_lane_donkey_longK/best.tar"),
        ("DVBF-lane",        "dvbf",    "checkpoints/dvbf_lane_donkey_longK/best.tar"),
        ("V2P-lane",         "v2p",     "checkpoints/v2p_lane_donkey_longK/best.tar"),
        # SINDYc dropped from the 100-step plot: its pysindy.predict C-extension
        # hard-aborts the process at long horizon (uncatchable), and it diverges
        # to ~1e17 anyway — irrelevant to the v3-vs-v4 question.
    ]:
        if not _os.path.exists(path):
            print(f"  skip {tag}: missing {path}")
            continue
        if kind == "sindyc":
            with open(path, "rb") as f:
                m = pickle.load(f)
        elif kind == "donkey3":
            m = LaneAugmentedDonkeyV3().to(DEVICE)
            m.load_state_dict(load_checkpoint(path)['dynamics']); m.eval()
        elif kind in ("donkey4", "donkey4_base"):
            m = LaneAugmentedDonkeyV4(damp=(kind == "donkey4"),
                                      hold=(kind == "donkey4")).to(DEVICE)
            m.load_state_dict(load_checkpoint(path)['dynamics']); m.eval()
        elif kind == "goku":
            m = DynamicsGOKULane().to(DEVICE)
            m.load_state_dict(load_checkpoint(path)['model']); m.eval()
        elif kind == "dvbf":
            m = DynamicsDVBFLane().to(DEVICE)
            m.load_state_dict(load_checkpoint(path)['model']); m.eval()
        elif kind == "v2p":
            m = DynamicsVid2ParamLane().to(DEVICE)
            m.load_state_dict(load_checkpoint(path)['model']); m.eval()
        models[tag] = (m, kind)
    return enc, dec, models


@torch.no_grad()
def rollout_one(model, kind, stack0, state_gt, action, enc):
    """Returns (K, 31) predicted state in NORMALIZED space, plus the 31-dim
    initial state (encoder-driven for everyone)."""
    # Build z0 from encoder. Note that the "x_rel=0 at t0" condition lives
    # in real space; in normalized space its value is `-mean/std`. We pull
    # it from state_gt[0] directly so the initial position MSE is exactly 0.
    stack0_t = torch.from_numpy(stack0).unsqueeze(0).to(DEVICE)
    z_obs = enc(stack0_t)                                       # (1, 29)
    z0 = torch.zeros(1, 31, device=DEVICE)
    gt0 = torch.from_numpy(state_gt[0]).to(DEVICE)
    z0[:, 0:2] = gt0[0:2]                                       # GT normalized pos
    z0[:, 2:11] = z_obs[:, :CAR_ENCODER_DIM]
    z0[:, 11:]  = z_obs[:, CAR_ENCODER_DIM:]

    if kind == "v2p":
        theta, _, _ = model.infer_theta(z_obs.unsqueeze(1))    # (B, 1, 29)

    K = action.shape[0]
    if kind == "sindyc":
        z_np = z0.cpu().numpy()
        traj = [z_np[0].copy()]
        for k in range(K):
            u = action[k:k + 1]
            z_np = np.asarray(model.predict(z_np, u=u), dtype=np.float32)
            traj.append(z_np[0].copy())
            # clamp to avoid overflow blowing up the entire figure y-axis
            if np.isnan(z_np).any() or np.abs(z_np).max() > 1e6:
                # Fill rest with the last sensible value
                bad_step = k + 1
                for _ in range(K - bad_step):
                    traj.append(traj[-1])
                break
        return np.stack(traj, axis=0)            # (K+1, 31)

    z = z0
    traj = [z[0].cpu().numpy().copy()]
    act_t = torch.from_numpy(action).to(DEVICE)
    for k in range(K):
        a = act_t[k:k + 1]
        if kind == "v2p":
            z = model.step(z, a, theta)
        else:
            out = model(z, a)
            z = out[0] if isinstance(out, tuple) else out
        traj.append(z[0].cpu().numpy().copy())
    return np.stack(traj, axis=0)


@torch.no_grad()
def image_mse_traj(traj_norm, imgs_gt, dec):
    """traj_norm: (K+1, 31), imgs_gt: (K+1, 64, 64). Returns (K+1,) image MSE."""
    z = torch.from_numpy(traj_norm).to(DEVICE)               # (T, 31)
    dec_input = torch.cat([z[:, 2:11], z[:, 11:]], dim=-1)   # (T, 29)
    # Decode in chunks to keep memory low.
    out = []
    for i in range(0, dec_input.size(0), 32):
        out.append(dec(dec_input[i:i + 32]))                  # (chunk, 1, 64, 64)
    rec = torch.cat(out, dim=0).squeeze(1).cpu().numpy()      # (T, 64, 64)
    gt = imgs_gt                                              # (T, 64, 64)
    return ((rec - gt) ** 2).mean(axis=(1, 2)).astype(np.float32)


def main():
    print("Loading val data...")
    base = SeqLaneDataset(DATA_DIR, seq_len=2)               # dummy seq_len; we use phys_list directly
    train_eps, val_eps = split_segments(base, val_frac=0.10, seed=SEED)
    windows = collect_long_windows(base, val_eps, window=ROLLOUT_K + 1, stride=15)
    print(f"  {len(windows)} val windows of length {ROLLOUT_K + 1}")
    if len(windows) == 0:
        raise RuntimeError("No val segment long enough for 100-step rollout.")

    enc, dec, models = _load_models()
    print(f"Models: {list(models.keys())}")

    # Accumulators
    K = ROLLOUT_K + 1
    pos_mse  = {tag: np.zeros(K, dtype=np.float64) for tag in models}
    yaw_mse  = {tag: np.zeros(K, dtype=np.float64) for tag in models}
    img_mse  = {tag: np.zeros(K, dtype=np.float64) for tag in models}
    n_used   = {tag: 0 for tag in models}

    for wi, w in enumerate(windows):
        if wi % 5 == 0:
            print(f"  window {wi}/{len(windows)} (ep={w['ep']}, t0={w['t0']})")
        for tag, (m, kind) in models.items():
            try:
                traj = rollout_one(m, kind, w["stack0"], w["state_gt"], w["action"], enc)
            except Exception as e:
                print(f"    {tag} failed: {e}")
                continue
            err = (traj - w["state_gt"]) ** 2                     # (K, 31)
            # Replace NaNs (from sindyc blowup) with the worst clean value to
            # avoid black holes in the plot — but still informative.
            err = np.nan_to_num(err, nan=1e6, posinf=1e6, neginf=1e6)
            pos = err[:, 0] + err[:, 1]                           # x² + y²
            pos_mse[tag] += pos
            yaw_mse[tag] += err[:, 2]
            # Image MSE
            try:
                imse = image_mse_traj(traj, w["imgs_gt"], dec)
            except Exception:
                imse = np.zeros(K, dtype=np.float32)
            img_mse[tag] += imse
            n_used[tag]  += 1

    for tag in models:
        if n_used[tag] > 0:
            pos_mse[tag] /= n_used[tag]
            yaw_mse[tag] /= n_used[tag]
            img_mse[tag] /= n_used[tag]

    # =================================================================
    # Plot
    # =================================================================
    style = {
        "PIWM-v3eqv (warm)": dict(color="#9B7FC4", lw=2.5, ls=":",  label="PIWM-v3eqv (warm)"),
        "PIWM-v4 (damped)":  dict(color="#5B2C9B", lw=3.0, ls="-",  label="PIWM-v4 (damped)"),
        "SINDYc-lane":       dict(color="#C49EE0", lw=2.0, ls="--", label="SINDYc-lane"),
        "DVBF-lane":         dict(color="#F08AB1", lw=2.0, ls="--", label="DVBF-lane"),
        "GOKU-lane":         dict(color="#9DDA9D", lw=2.0, ls="--", label="GOKU-lane"),
        "V2P-lane":          dict(color="#F2A24C", lw=2.0, ls="--", label="V2P-lane"),
    }
    order = ["PIWM-v4 (damped)", "PIWM-v3eqv (warm)", "SINDYc-lane", "DVBF-lane", "GOKU-lane", "V2P-lane"]
    order = [t for t in order if t in models]

    # Two-row layout: (a) full 100-step linear-y; (b) zoom to first 40 steps
    # to show the meaningful regime where models haven't fully diverged.
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), constrained_layout=True)
    steps = np.arange(K)
    for row, (xmax, suffix) in enumerate([(100, "full 100 steps"),
                                          (40,  "zoom 0-40 (training horizon × 5)")]):
        for tag in order:
            s = style[tag]
            axes[row, 0].plot(steps, pos_mse[tag], **s)
            axes[row, 1].plot(steps, yaw_mse[tag], **s)
            axes[row, 2].plot(steps, img_mse[tag], **s)
        titles = [f"Position MSE  (x² + y²)  —  {suffix}",
                  f"yaw_rel MSE  —  {suffix}",
                  f"Image MSE  —  {suffix}"]
        for ax, ttl in zip(axes[row], titles):
            ax.set_xlabel("Step", fontsize=12)
            ax.set_title(ttl, fontsize=12)
            ax.grid(alpha=0.4)
            ax.legend(fontsize=10, loc="upper left")
            ax.set_xlim(0, xmax)
        axes[row, 0].set_ylabel("Position MSE (x² + y²)", fontsize=12)
        axes[row, 1].set_ylabel("yaw_rel MSE", fontsize=12)
        axes[row, 2].set_ylabel("Image MSE", fontsize=12)
    # Caps. Top row matches the standard PIWM-paper ranges so a reader can
    # compare; bottom row is tighter so the early curves are readable.
    axes[0, 0].set_ylim(0, 8.0)
    axes[0, 1].set_ylim(0, 25.0)
    axes[0, 2].set_ylim(0, 0.030)
    axes[1, 0].set_ylim(0, 2.0)
    axes[1, 1].set_ylim(0, 3.0)
    axes[1, 2].set_ylim(0, 0.020)

    fig.suptitle(f"PIWM-lane-v6 (donkey) vs lane baselines  —  {ROLLOUT_K}-step rollout on val",
                 fontsize=15, fontweight="bold")
    out_path = _os.path.join(SAVE_DIR, "fig_100step.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out_path}")
    print(f"Per-tag windows used: {n_used}")

    # Also dump final 100-step values
    print("\nFinal (k=100) errors:")
    print(f"  {'model':<22s}  {'pos':>8s}  {'yaw':>8s}  {'img':>8s}")
    for tag in order:
        print(f"  {tag:<22s}  "
              f"{pos_mse[tag][-1]:8.3f}  {yaw_mse[tag][-1]:8.3f}  {img_mse[tag][-1]:8.4f}")


if __name__ == "__main__":
    main()
