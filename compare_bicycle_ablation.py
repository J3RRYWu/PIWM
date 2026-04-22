"""Ablation comparison: PIWM-bicycle variants on shared encoder/decoder.

Loads each variant's trained Dynamics (best.tar) and evaluates on the same
20-episode test set used in compare_all_shared.py. Plots state MSE (x_rel,
y_rel, yaw_rel) and image MSE to show the effect of each modification.

Variants compared:
  bike-v1  : original (residual affects all 11 dims, free L)        [baseline]
  bike-v2  : + residual masked on x, y
  bike-v3  : v2 + L constrained to [2, 8]
  bike-v4  : dynamic bicycle (tire forces) + masked residual
"""

import os, glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (DEVICE, DATA_DIR, PHYSICS_MEAN_REL, PHYSICS_STD_REL, FRAME_STACK)
from relative_coords import to_relative_np
from models.encoder import PhysicsEncoder
from models.decoder import PhysicsDecoder as PIWMDecoder
from models.dynamics_bicycle import BicycleDynamics                # v1
from models.dynamics_bicycle_v2 import BicycleDynamicsV2            # v2
from models.dynamics_bicycle_v3 import BicycleDynamicsV3            # v3
from models.dynamics_bicycle_v4 import BicycleDynamicsV4            # v4
from utils import load_checkpoint

MAX_STEPS = 100
NUM_EPISODES = 20


VARIANTS = [
    ("bike-v1-orig",    BicycleDynamics,   "checkpoints/piwm_bike/best.tar",    '#00aaff'),
    ("bike-v2-mask",    BicycleDynamicsV2, "checkpoints/piwm_bike_v2/best.tar", '#1f8fff'),
    ("bike-v3-Lcap",    BicycleDynamicsV3, "checkpoints/piwm_bike_v3/best.tar", '#004499'),
    ("bike-v4-dyn",     BicycleDynamicsV4, "checkpoints/piwm_bike_v4/best.tar", '#8c00dc'),
]


def load_shared_ae():
    ck = load_checkpoint("checkpoints/piwm_v2_rel/best.tar")
    enc = PhysicsEncoder().to(DEVICE); enc.load_state_dict(ck['encoder']); enc.eval()
    dec = PIWMDecoder().to(DEVICE); dec.load_state_dict(ck['decoder']); dec.eval()
    return enc, dec


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
            if len(imgs) < FRAME_STACK + MAX_STEPS + 1: continue
            episodes.append({'imgs': imgs, 'phys': phys, 'acts': acts})
        except: pass
    print(f"Loaded {len(episodes)} episodes")
    return episodes


def eval_one(dyn, episodes, enc, dec):
    img_mses = [[] for _ in range(MAX_STEPS + 1)]
    state_mses = [[] for _ in range(MAX_STEPS + 1)]
    for ep in episodes:
        imgs = ep['imgs']; phys = ep['phys']; acts = ep['acts']
        t0 = FRAME_STACK - 1
        if t0 + MAX_STEPS >= len(imgs): continue
        stack0 = np.stack([imgs[t0 - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
        rel_phys = to_relative_np(phys[t0:t0 + MAX_STEPS + 1], ref_idx=0)
        rel_norm = (rel_phys - PHYSICS_MEAN_REL) / PHYSICS_STD_REL

        with torch.no_grad():
            z_enc = enc(torch.tensor(stack0, dtype=torch.float32).unsqueeze(0).to(DEVICE))[0]
            z = torch.zeros(1, 11, device=DEVICE)
            z[0, 0:2] = torch.tensor(rel_norm[0, 0:2], device=DEVICE)
            z[0, 2:11] = z_enc
            recon = dec(z[:, 2:11]).cpu().numpy()[0, 0]
            img_mses[0].append(((recon - imgs[t0]) ** 2).mean())
            state_mses[0].append((z[0].cpu().numpy() - rel_norm[0]) ** 2)
            for k in range(1, MAX_STEPS + 1):
                a = torch.tensor(acts[t0 + k - 1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                z, _ = dyn(z, a)
                recon = dec(z[:, 2:11]).cpu().numpy()[0, 0]
                img_mses[k].append(((recon - imgs[t0 + k]) ** 2).mean())
                state_mses[k].append((z[0].cpu().numpy() - rel_norm[k]) ** 2)
    return img_mses, state_mses


def plot_state(all_results, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ['x_rel', 'y_rel', 'yaw_rel']
    dims = [0, 1, 2]
    for ax, title, idx in zip(axes, titles, dims):
        for name, (_, state_mses, color) in all_results.items():
            steps, means = [], []
            for k in range(len(state_mses)):
                if len(state_mses[k]) >= 3:
                    arr = np.array(state_mses[k])
                    steps.append(k); means.append(arr[:, idx].mean())
            ax.plot(steps, means, color=color, linewidth=2.2, label=name)
        ax.set_xlabel('Step'); ax.set_ylabel(f'{title} MSE')
        ax.set_title(title); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        all_vals = []
        for name2, (_, sm2, _) in all_results.items():
            for k in range(len(sm2)):
                if len(sm2[k]) >= 3:
                    all_vals.append(np.array(sm2[k])[:, idx].mean())
        if all_vals:
            y_top = np.percentile(all_vals, 95) * 1.2
            ax.set_ylim(bottom=0, top=max(y_top, 0.01))
    fig.suptitle('Ablation: PIWM-bicycle variants (State MSE)', fontsize=13)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"Saved {save_path}")


def plot_image(all_results, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, (img_mses, _, color) in all_results.items():
        steps, means = [], []
        for k in range(len(img_mses)):
            if len(img_mses[k]) >= 3:
                steps.append(k); means.append(np.mean(img_mses[k]))
        ax.plot(steps, means, color=color, linewidth=2.2, label=name)
    ax.set_xlabel('Step'); ax.set_ylabel('Image MSE (per pixel)')
    ax.set_title('Ablation: PIWM-bicycle variants (Image MSE)')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0, top=0.025)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    os.makedirs("vis", exist_ok=True)
    episodes = load_test_episodes()
    enc, dec = load_shared_ae()

    all_results = {}
    for name, cls, ckpt_path, color in VARIANTS:
        if not os.path.exists(ckpt_path):
            print(f"Skip {name}: checkpoint not found ({ckpt_path})")
            continue
        print(f"[{name}] evaluating from {ckpt_path} ...")
        dyn = cls().to(DEVICE)
        ck = load_checkpoint(ckpt_path)
        dyn.load_state_dict(ck['dynamics'])
        dyn.eval()
        img_mses, state_mses = eval_one(dyn, episodes, enc, dec)
        all_results[name] = (img_mses, state_mses, color)

        # Report learned physical params
        if hasattr(dyn, 'L'): print(f"    L={float(dyn.L):.3f}")
        if hasattr(dyn, 'm'):
            print(f"    m={float(dyn.m):.1f}  Iz={float(dyn.Iz):.1f}"
                  f"  Cf={float(dyn.Cf):.0f}  Cr={float(dyn.Cr):.0f}"
                  f"  Lf={float(dyn.Lf):.2f}  Lr={float(dyn.Lr):.2f}")

    plot_state(all_results, "vis/ablation_bicycle_state_mse.png")
    plot_image(all_results, "vis/ablation_bicycle_image_mse.png")

    # Summary
    print(f"\n{'Step':>5s} " + " ".join([f"{n:>15s}" for n in all_results.keys()]))
    for k in [0, 5, 10, 20, 30, 50, 75, 100]:
        row = f"{k:>5d} "
        for name, (img, _, _) in all_results.items():
            row += f" {np.mean(img[k]) if k < len(img) and img[k] else -1:>15.6f}"
        print(row)
