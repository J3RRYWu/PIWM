"""Compare PIWM vs Baseline world model: Image MSE vs rollout step."""

# --- auto-added: make repo root importable when run as a script ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end shim ---


import numpy as np
import torch
import glob
import os
from PIL import Image, ImageDraw

from config import (DEVICE, DATA_DIR, PHYSICS_MEAN, PHYSICS_STD, FRAME_STACK,
                    DYN_SAVE_DIR, AE_SAVE_DIR, ENCODER_STATE_INDICES)
from models.dynamics import PhysicsDynamics
from models.decoder import PhysicsDecoder as PIWMDecoder
from models.baseline import BaselineEncoder, BaselineDecoder, BaselineDynamics
from utils import load_checkpoint

MAX_STEPS = 100


def eval_piwm():
    """PIWM: GT state -> Dynamics rollout -> Decoder."""
    dyn = PhysicsDynamics().to(DEVICE)
    dyn.load_state_dict(load_checkpoint(os.path.join(DYN_SAVE_DIR, "best.tar"))["dynamics"])
    dyn.eval()
    dec = PIWMDecoder().to(DEVICE)
    dec.load_state_dict(load_checkpoint(os.path.join(AE_SAVE_DIR, "best.tar"))["decoder"])
    dec.eval()

    files = sorted(glob.glob(f"{DATA_DIR}/*.npz"))
    mses = {k: [] for k in range(MAX_STEPS + 1)}

    for f in files:
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
            actions = d["action"].astype(np.float32)
            physics = np.column_stack([pos, yaw, vel, omega, wheel, steer])

            T = min(MAX_STEPS + 1, len(imgs), len(actions) + 1)
            if T < 10: continue

            init_norm = (physics[0] - PHYSICS_MEAN) / PHYSICS_STD
            z = torch.tensor(init_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                enc_dims = z[:, ENCODER_STATE_INDICES]
                recon = dec(enc_dims).cpu().numpy()[0, 0]
                mses[0].append(((recon - imgs[0]) ** 2).mean())
                for k in range(1, T):
                    a = torch.tensor(actions[k-1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    z, _ = dyn(z, a)
                    enc_dims = z[:, ENCODER_STATE_INDICES]
                    recon = dec(enc_dims).cpu().numpy()[0, 0]
                    mses[k].append(((recon - imgs[k]) ** 2).mean())
        except: pass

    return mses


def eval_baseline():
    """Baseline: Encoder(image_0) -> Dynamics rollout -> Decoder."""
    enc = BaselineEncoder().to(DEVICE)
    dec = BaselineDecoder().to(DEVICE)
    dyn = BaselineDynamics().to(DEVICE)
    ae_ck = load_checkpoint("checkpoints/baseline_ae/best.tar")
    dy_ck = load_checkpoint("checkpoints/baseline_dynamics/best.tar")
    enc.load_state_dict(ae_ck["encoder"])
    dec.load_state_dict(ae_ck["decoder"])
    dyn.load_state_dict(dy_ck["dynamics"])
    enc.eval(); dec.eval(); dyn.eval()

    files = sorted(glob.glob(f"{DATA_DIR}/*.npz"))
    mses = {k: [] for k in range(MAX_STEPS + 1)}

    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            imgs = d["imgs"].astype(np.float32)
            if imgs.max() > 1.0: imgs /= 255.0
            actions = d["action"].astype(np.float32)
            T = min(MAX_STEPS + 1, len(imgs), len(actions) + 1)
            if T < FRAME_STACK + 1: continue

            # Start from image_0 (actually image at t=FRAME_STACK-1 for proper frame stack)
            t0 = FRAME_STACK - 1
            stack0 = np.stack([imgs[t0 - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)

            with torch.no_grad():
                z = enc(torch.tensor(stack0, dtype=torch.float32).unsqueeze(0).to(DEVICE))
                recon = dec(z).cpu().numpy()[0, 0]
                mses[0].append(((recon - imgs[t0]) ** 2).mean())

                for k in range(1, T - t0):
                    a = torch.tensor(actions[t0 + k - 1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    z = dyn(z, a)
                    recon = dec(z).cpu().numpy()[0, 0]
                    mses[k].append(((recon - imgs[t0 + k]) ** 2).mean())
        except Exception as e:
            print(f"Skip {f}: {e}")

    return mses


def plot_comparison(piwm_mses, base_mses, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Collect all steps where both models have data
    steps = []
    piwm_mean, piwm_std = [], []
    base_mean, base_std = [], []
    for k in range(MAX_STEPS + 1):
        if len(piwm_mses[k]) >= 5 and len(base_mses[k]) >= 5:
            steps.append(k)
            piwm_mean.append(np.mean(piwm_mses[k]))
            piwm_std.append(np.std(piwm_mses[k]))
            base_mean.append(np.mean(base_mses[k]))
            base_std.append(np.std(base_mses[k]))

    steps = np.array(steps)
    piwm_mean = np.array(piwm_mean); piwm_std = np.array(piwm_std)
    base_mean = np.array(base_mean); base_std = np.array(base_std)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, piwm_mean, color='#0050dc', linewidth=2.5, label='PIWM (physics)')
    ax.fill_between(steps, np.maximum(0, piwm_mean - piwm_std), piwm_mean + piwm_std,
                    color='#0050dc', alpha=0.2)
    ax.plot(steps, base_mean, color='#dc2828', linewidth=2.5, label='Baseline (black-box)')
    ax.fill_between(steps, np.maximum(0, base_mean - base_std), base_mean + base_std,
                    color='#dc2828', alpha=0.2)
    ax.set_xlabel('Prediction Horizon (steps)', fontsize=12)
    ax.set_ylabel('Image MSE (per pixel)', fontsize=12)
    ax.set_title('PIWM vs Baseline Black-Box World Model', fontsize=13)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max(steps)])
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved {save_path}")

    # Print table
    print(f"\n{'Step':>5s} {'PIWM':>12s} {'Baseline':>12s} {'Ratio':>8s}")
    print("-" * 42)
    for k in [0, 1, 5, 10, 20, 30, 50, 75, 100]:
        if piwm_mses[k] and base_mses[k]:
            pm = np.mean(piwm_mses[k])
            bm = np.mean(base_mses[k])
            print(f"{k:>5d} {pm:>12.6f} {bm:>12.6f} {bm/pm:>7.2f}x")


if __name__ == "__main__":
    os.makedirs("vis", exist_ok=True)
    print("Evaluating PIWM...")
    piwm_mses = eval_piwm()
    print("Evaluating Baseline...")
    base_mses = eval_baseline()

    # Debug: check data density
    print("\nData density check:")
    for k in [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]:
        print(f"  step {k}: PIWM={len(piwm_mses[k])}, Baseline={len(base_mses[k])}")

    plot_comparison(piwm_mses, base_mses, "vis/mse_comparison.png")
