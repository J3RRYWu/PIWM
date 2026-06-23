"""Unified evaluation: PIWM V2_rel vs SINDYc / GOKU / DVBF / Vid2Param.

Metrics vs rollout step:
  1. Image MSE
  2. State MSE (x_rel, y_rel, yaw_rel separately + average)

All methods operate on the same test trajectories with relative coordinates.
"""

# --- auto-added: make repo root importable when run as a script ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end shim ---


import os, sys, glob, pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (DEVICE, DATA_DIR, PHYSICS_MEAN_REL, PHYSICS_STD_REL,
                    FRAME_STACK, ENCODER_STATE_INDICES)
from relative_coords import to_relative_np
from models.encoder import PhysicsEncoder
from models.decoder import PhysicsDecoder as PIWMDecoder
from models.dynamics import PhysicsDynamics
from models.dynamics_bicycle import BicycleDynamics
from baselines.goku import GOKU, ENCODER_SEQ as GOKU_ENC_SEQ
from baselines.dvbf import DVBF, ENCODER_SEQ as DVBF_ENC_SEQ
from baselines.vid2param import Vid2Param, ENCODER_SEQ as V2P_ENC_SEQ
from utils import load_checkpoint

MAX_STEPS = 50  # rollout horizon
NUM_EPISODES = 20  # test episodes


def load_test_episodes(n=NUM_EPISODES):
    files = sorted(glob.glob(f"{DATA_DIR}/*.npz"))
    episodes = []
    for f in files[:n]:
        try:
            d = np.load(f, allow_pickle=True)
            imgs = d["imgs"].astype(np.float32)
            if imgs.max() > 1.0: imgs /= 255.0
            pos = d["position"].astype(np.float32)
            yaw = d["yaw"].astype(np.float32)
            vel = d["velocity"].astype(np.float32)
            omega = d["angular_velocity"].astype(np.float32)
            wheel = d["wheel_omega"].astype(np.float32)
            steer = d["steering_angle"].astype(np.float32)
            acts = d["action"].astype(np.float32)
            phys = np.column_stack([pos, yaw, vel, omega, wheel, steer])
            # Need enough frames for encoder seq (max 4) + frame stack + rollout
            n_frames = len(imgs)
            if n_frames < FRAME_STACK + 4 + MAX_STEPS: continue
            episodes.append({'imgs': imgs, 'phys': phys, 'acts': acts, 'n': n_frames})
        except: pass
    print(f"Loaded {len(episodes)} test episodes")
    return episodes


def patch_dyn_rel(dyn):
    """Apply REL normalization to PhysicsDynamics buffers."""
    from config import DT
    std = PHYSICS_STD_REL; mean = PHYSICS_MEAN_REL
    dyn.dt_vx_x = torch.tensor(float(std[3] * DT / std[0])).to(dyn.dt_vx_x)
    dyn.dt_vy_y = torch.tensor(float(std[4] * DT / std[1])).to(dyn.dt_vy_y)
    dyn.dt_om_yaw = torch.tensor(float(std[5] * DT / std[2])).to(dyn.dt_om_yaw)
    dyn.mean_vx_dt_over_std_x = torch.tensor(float(mean[3] * DT / std[0])).to(dyn.mean_vx_dt_over_std_x)
    dyn.mean_vy_dt_over_std_y = torch.tensor(float(mean[4] * DT / std[1])).to(dyn.mean_vy_dt_over_std_y)
    dyn.mean_om_dt_over_std_yaw = torch.tensor(float(mean[5] * DT / std[2])).to(dyn.mean_om_dt_over_std_yaw)
    return dyn


# ============================================================
# PIWM V2_rel (linear kinematic + residual)
# ============================================================
def eval_piwm(episodes, ckpt_path="checkpoints/piwm_v2_rel/best.tar", use_bicycle=False):
    ck = load_checkpoint(ckpt_path)
    enc = PhysicsEncoder().to(DEVICE); enc.load_state_dict(ck['encoder']); enc.eval()
    dec = PIWMDecoder().to(DEVICE); dec.load_state_dict(ck['decoder']); dec.eval()
    if use_bicycle:
        dyn = BicycleDynamics().to(DEVICE)
    else:
        dyn = PhysicsDynamics().to(DEVICE)
        patch_dyn_rel(dyn)
    dyn.load_state_dict(ck['dynamics']); dyn.eval()

    img_mses = [[] for _ in range(MAX_STEPS + 1)]
    state_mses = [[] for _ in range(MAX_STEPS + 1)]  # per-step MSE for (x, y, yaw)

    for ep in episodes:
        imgs = ep['imgs']; phys = ep['phys']; acts = ep['acts']
        t0 = FRAME_STACK - 1
        stack0 = np.stack([imgs[t0 - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
        rel_phys = to_relative_np(phys[t0:t0 + MAX_STEPS + 1], ref_idx=0)
        rel_phys_norm = (rel_phys - PHYSICS_MEAN_REL) / PHYSICS_STD_REL

        with torch.no_grad():
            z_enc = enc(torch.tensor(stack0, dtype=torch.float32).unsqueeze(0).to(DEVICE))[0]
            z = torch.zeros(1, 11, device=DEVICE)
            z[0, 0:2] = torch.tensor(rel_phys_norm[0, 0:2], device=DEVICE)
            z[0, 2:11] = z_enc

            recon = dec(z[:, 2:11]).cpu().numpy()[0, 0]
            img_mses[0].append(((recon - imgs[t0]) ** 2).mean())
            state_pred = z[0].cpu().numpy()
            state_mses[0].append((state_pred - rel_phys_norm[0]) ** 2)

            for k in range(1, MAX_STEPS + 1):
                a = torch.tensor(acts[t0 + k - 1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                z, _ = dyn(z, a)
                recon = dec(z[:, 2:11]).cpu().numpy()[0, 0]
                img_mses[k].append(((recon - imgs[t0 + k]) ** 2).mean())
                state_pred = z[0].cpu().numpy()
                state_mses[k].append((state_pred - rel_phys_norm[k]) ** 2)

    return img_mses, state_mses


# ============================================================
# SINDYc (+ external decoder for image MSE)
# ============================================================
def eval_sindyc(episodes):
    with open("checkpoints/sindyc/model.pkl", "rb") as f:
        model = pickle.load(f)
    # Use PIWM's decoder for image reconstruction
    dec = PIWMDecoder().to(DEVICE)
    dec.load_state_dict(load_checkpoint("checkpoints/piwm_v2_rel/best.tar")['decoder'])
    dec.eval()

    img_mses = [[] for _ in range(MAX_STEPS + 1)]
    state_mses = [[] for _ in range(MAX_STEPS + 1)]

    for ep in episodes:
        imgs = ep['imgs']; phys = ep['phys']; acts = ep['acts']
        t0 = FRAME_STACK - 1
        rel_phys = to_relative_np(phys[t0:t0 + MAX_STEPS + 1], ref_idx=0)
        rel_phys_norm = (rel_phys - PHYSICS_MEAN_REL) / PHYSICS_STD_REL

        # SINDYc rollout
        z = rel_phys_norm[0].copy()
        for k in range(MAX_STEPS + 1):
            if k > 0:
                u = np.atleast_2d(acts[t0 + k - 1])
                try:
                    z = model.predict(z.reshape(1, -1), u=u).squeeze(0)
                except Exception:
                    pass  # keep previous z on error

            # Image via PIWM decoder
            z_torch = torch.tensor(z[ENCODER_STATE_INDICES], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                recon = dec(z_torch).cpu().numpy()[0, 0]
            img_mses[k].append(((recon - imgs[t0 + k]) ** 2).mean())
            state_mses[k].append((z - rel_phys_norm[k]) ** 2)

    return img_mses, state_mses


# ============================================================
# GOKU
# ============================================================
def eval_goku(episodes):
    model = GOKU().to(DEVICE)
    model.load_state_dict(load_checkpoint("checkpoints/goku/best.tar")['model'])
    model.eval()

    img_mses = [[] for _ in range(MAX_STEPS + 1)]
    state_mses = [[] for _ in range(MAX_STEPS + 1)]

    for ep in episodes:
        imgs = ep['imgs']; phys = ep['phys']; acts = ep['acts']
        t0 = FRAME_STACK - 1 + GOKU_ENC_SEQ - 1
        if t0 + MAX_STEPS >= ep['n']: continue

        # Encoder sequence
        enc_seq = []
        for k in range(GOKU_ENC_SEQ):
            center = t0 - GOKU_ENC_SEQ + 1 + k
            stack = np.stack([imgs[center - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
            enc_seq.append(stack)
        enc_seq = np.stack(enc_seq, axis=0)
        fut_acts = acts[t0:t0 + MAX_STEPS]
        rel_phys = to_relative_np(phys[t0:t0 + MAX_STEPS + 1], ref_idx=0)
        rel_phys_norm = (rel_phys - PHYSICS_MEAN_REL) / PHYSICS_STD_REL

        with torch.no_grad():
            enc_seq_t = torch.tensor(enc_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            acts_t = torch.tensor(fut_acts, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            recons, states, _, _ = model(enc_seq_t, acts_t, MAX_STEPS)
            recons = recons.cpu().numpy()[0, :, 0]  # (T+1, 64, 64)
            states = states.cpu().numpy()[0]  # (T+1, 11)

        for k in range(MAX_STEPS + 1):
            img_mses[k].append(((recons[k] - imgs[t0 + k]) ** 2).mean())
            state_mses[k].append((states[k] - rel_phys_norm[k]) ** 2)

    return img_mses, state_mses


# ============================================================
# DVBF
# ============================================================
def eval_dvbf(episodes):
    model = DVBF().to(DEVICE)
    model.load_state_dict(load_checkpoint("checkpoints/dvbf/best.tar")['model'])
    model.eval()

    img_mses = [[] for _ in range(MAX_STEPS + 1)]
    state_mses = [[] for _ in range(MAX_STEPS + 1)]

    for ep in episodes:
        imgs = ep['imgs']; phys = ep['phys']; acts = ep['acts']
        t0 = FRAME_STACK - 1 + DVBF_ENC_SEQ - 1
        if t0 + MAX_STEPS >= ep['n']: continue

        enc_seq = []
        for k in range(DVBF_ENC_SEQ):
            center = t0 - DVBF_ENC_SEQ + 1 + k
            stack = np.stack([imgs[center - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
            enc_seq.append(stack)
        enc_seq = np.stack(enc_seq, axis=0)
        fut_acts = acts[t0:t0 + MAX_STEPS]
        rel_phys = to_relative_np(phys[t0:t0 + MAX_STEPS + 1], ref_idx=0)
        rel_phys_norm = (rel_phys - PHYSICS_MEAN_REL) / PHYSICS_STD_REL

        with torch.no_grad():
            enc_seq_t = torch.tensor(enc_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            acts_t = torch.tensor(fut_acts, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            z0, _, _ = model.infer_z0(enc_seq_t)
            zs = model.rollout(z0, acts_t, MAX_STEPS)  # (1, T+1, 11)
            recons = model.decoder(zs.reshape(-1, zs.size(-1))).reshape(1, MAX_STEPS + 1, 1, 64, 64)
            recons = recons.cpu().numpy()[0, :, 0]
            states = zs.cpu().numpy()[0]

        for k in range(MAX_STEPS + 1):
            img_mses[k].append(((recons[k] - imgs[t0 + k]) ** 2).mean())
            state_mses[k].append((states[k] - rel_phys_norm[k]) ** 2)

    return img_mses, state_mses


# ============================================================
# Vid2Param
# ============================================================
def eval_v2p(episodes):
    model = Vid2Param().to(DEVICE)
    model.load_state_dict(load_checkpoint("checkpoints/vid2param/best.tar")['model'])
    model.eval()

    img_mses = [[] for _ in range(MAX_STEPS + 1)]
    state_mses = [[] for _ in range(MAX_STEPS + 1)]

    for ep in episodes:
        imgs = ep['imgs']; phys = ep['phys']; acts = ep['acts']
        t0 = FRAME_STACK - 1 + V2P_ENC_SEQ - 1
        if t0 + MAX_STEPS >= ep['n']: continue

        enc_seq = []
        for k in range(V2P_ENC_SEQ):
            center = t0 - V2P_ENC_SEQ + 1 + k
            stack = np.stack([imgs[center - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
            enc_seq.append(stack)
        enc_seq = np.stack(enc_seq, axis=0)
        fut_acts = acts[t0:t0 + MAX_STEPS]
        rel_phys = to_relative_np(phys[t0:t0 + MAX_STEPS + 1], ref_idx=0)
        rel_phys_norm = (rel_phys - PHYSICS_MEAN_REL) / PHYSICS_STD_REL

        with torch.no_grad():
            enc_seq_t = torch.tensor(enc_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            acts_t = torch.tensor(fut_acts, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            recons, states, _, _ = model(enc_seq_t, acts_t, MAX_STEPS)
            recons = recons.cpu().numpy()[0, :, 0]
            states = states.cpu().numpy()[0]

        for k in range(MAX_STEPS + 1):
            img_mses[k].append(((recons[k] - imgs[t0 + k]) ** 2).mean())
            state_mses[k].append((states[k] - rel_phys_norm[k]) ** 2)

    return img_mses, state_mses


# ============================================================
# Plotting
# ============================================================
def plot_image_mse(all_results, save_path):
    colors = {
        'PIWM-linear': '#0050dc',
        'PIWM-bicycle': '#00aaff',
        'SINDYc': '#aa00cc',
        'GOKU-net': '#6bba55',
        'Vid2Param': '#ff8c00',
        'DVBF': '#dc2828',
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, (img_mses, _) in all_results.items():
        steps, means, stds = [], [], []
        for k in range(len(img_mses)):
            if len(img_mses[k]) >= 3:
                arr = np.array(img_mses[k])
                steps.append(k); means.append(arr.mean()); stds.append(arr.std())
        steps = np.array(steps); means = np.array(means); stds = np.array(stds)
        c = colors.get(name, 'gray')
        ax.plot(steps, means, color=c, linewidth=2.2, label=name)
        ax.fill_between(steps, np.maximum(0, means - stds), means + stds, color=c, alpha=0.12)

    ax.set_xlabel('Prediction Horizon (steps)', fontsize=12)
    ax.set_ylabel('Image MSE (per pixel)', fontsize=12)
    ax.set_title('Image MSE vs Rollout Step  (relative coordinates)', fontsize=13)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_state_mse(all_results, save_path):
    colors = {
        'PIWM-linear': '#0050dc',
        'PIWM-bicycle': '#00aaff',
        'SINDYc': '#aa00cc',
        'GOKU-net': '#6bba55',
        'Vid2Param': '#ff8c00',
        'DVBF': '#dc2828',
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles = ['x_rel MSE', 'y_rel MSE', 'yaw_rel MSE']
    dim_idx = [0, 1, 2]

    for ax, title, idx in zip(axes, titles, dim_idx):
        for name, (_, state_mses) in all_results.items():
            steps, means = [], []
            for k in range(len(state_mses)):
                if len(state_mses[k]) >= 3:
                    arr = np.array(state_mses[k])  # (N, 11)
                    steps.append(k); means.append(arr[:, idx].mean())
            c = colors.get(name, 'gray')
            ax.plot(steps, means, color=c, linewidth=2.2, label=name)

        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle('State MSE (normalized) vs Rollout Step', fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"Saved {save_path}")


def print_summary(all_results):
    print(f"\n{'Step':>5s} " + " ".join([f"{n:>18s}" for n in all_results.keys()]))
    print("-" * (5 + 20 * len(all_results)))
    for k in [0, 5, 10, 20, 30, 50]:
        row = f"{k:>5d} "
        for name, (img_mses, _) in all_results.items():
            if k < len(img_mses) and img_mses[k]:
                row += f" {np.mean(img_mses[k]):>18.6f}"
            else:
                row += f" {'--':>18s}"
        print(row)


if __name__ == "__main__":
    os.makedirs("vis", exist_ok=True)
    episodes = load_test_episodes()
    all_results = {}

    print("Evaluating PIWM (linear kinematic)...")
    all_results['PIWM-linear'] = eval_piwm(episodes)
    try:
        print("Evaluating PIWM (bicycle)...")
        all_results['PIWM-bicycle'] = eval_piwm(episodes,
            ckpt_path="checkpoints/piwm_bike/best.tar", use_bicycle=True)
    except FileNotFoundError as e:
        print(f"Skipping bicycle ({e})")
    print("Evaluating SINDYc...")
    all_results['SINDYc'] = eval_sindyc(episodes)
    try:
        print("Evaluating GOKU-net...")
        all_results['GOKU-net'] = eval_goku(episodes)
    except FileNotFoundError as e:
        print(f"Skipping GOKU ({e})")
    try:
        print("Evaluating DVBF...")
        all_results['DVBF'] = eval_dvbf(episodes)
    except FileNotFoundError as e:
        print(f"Skipping DVBF ({e})")
    try:
        print("Evaluating Vid2Param...")
        all_results['Vid2Param'] = eval_v2p(episodes)
    except FileNotFoundError as e:
        print(f"Skipping V2P ({e})")

    plot_image_mse(all_results, "vis/rel_image_mse.png")
    plot_state_mse(all_results, "vis/rel_state_mse.png")
    print_summary(all_results)
