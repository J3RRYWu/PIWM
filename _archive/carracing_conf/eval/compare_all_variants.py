"""Full ablation: PIWM V0/V1/V2/V3 + Baseline on image MSE vs rollout step."""

# --- auto-added: make repo root importable when run as a script ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end shim ---


import os, glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (DEVICE, DATA_DIR, PHYSICS_MEAN, PHYSICS_STD, FRAME_STACK,
                    DYN_SAVE_DIR, AE_SAVE_DIR, ENCODER_STATE_INDICES)
from models.encoder import PhysicsEncoder
from models.decoder import PhysicsDecoder as PIWMDecoder
from models.dynamics import PhysicsDynamics
from models.baseline import BaselineEncoder, BaselineDecoder, BaselineDynamics
from models.piwm_v3 import EncoderV3, DecoderV3, DynamicsV3
from utils import load_checkpoint

MAX_STEPS = 100


def eval_piwm_v0():
    """V0: Original, GT state init, physics dynamics, 9-dim decoder."""
    dyn = PhysicsDynamics().to(DEVICE)
    dyn.load_state_dict(load_checkpoint(os.path.join(DYN_SAVE_DIR, "best.tar"))["dynamics"])
    dyn.eval()
    dec = PIWMDecoder().to(DEVICE)
    dec.load_state_dict(load_checkpoint(os.path.join(AE_SAVE_DIR, "best.tar"))["decoder"])
    dec.eval()
    return _eval_piwm_generic(dyn, None, dec, 'v0')


def eval_piwm_v1():
    """V1: Multistep dynamics + V0 AE."""
    dyn = PhysicsDynamics().to(DEVICE)
    dyn.load_state_dict(load_checkpoint("checkpoints/dynamics_v1/best.tar")["dynamics"])
    dyn.eval()
    dec = PIWMDecoder().to(DEVICE)
    dec.load_state_dict(load_checkpoint(os.path.join(AE_SAVE_DIR, "best.tar"))["decoder"])
    dec.eval()
    return _eval_piwm_generic(dyn, None, dec, 'v1')


def eval_piwm_v2():
    """V2: End-to-end fine-tuned (encoder, decoder, dynamics all updated)."""
    ck = load_checkpoint("checkpoints/piwm_v2/best.tar")
    enc = PhysicsEncoder().to(DEVICE)
    dec = PIWMDecoder().to(DEVICE)
    dyn = PhysicsDynamics().to(DEVICE)
    enc.load_state_dict(ck['encoder']); enc.eval()
    dec.load_state_dict(ck['decoder']); dec.eval()
    dyn.load_state_dict(ck['dynamics']); dyn.eval()
    return _eval_piwm_generic(dyn, enc, dec, 'v2')


def _eval_piwm_generic(dyn, enc, dec, tag):
    """Rollout from GT state (or encoder output), decode each step, compare to actual image."""
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

            # Start at t0 = FRAME_STACK - 1 (so encoder has valid input)
            t0 = FRAME_STACK - 1
            init_norm = (physics[t0] - PHYSICS_MEAN) / PHYSICS_STD

            if enc is not None:
                # Use encoder for 9 dims, GT for x, y
                stack0 = np.stack([imgs[t0 - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
                with torch.no_grad():
                    z_enc = enc(torch.tensor(stack0, dtype=torch.float32).unsqueeze(0).to(DEVICE))[0]
                z = torch.zeros(1, 11, device=DEVICE)
                z[0, 0:2] = torch.tensor(init_norm[0:2], dtype=torch.float32, device=DEVICE)
                z[0, 2:11] = z_enc
            else:
                z = torch.tensor(init_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                recon = dec(z[:, 2:11]).cpu().numpy()[0, 0]
                mses[0].append(((recon - imgs[t0]) ** 2).mean())
                for k in range(1, T - t0):
                    a = torch.tensor(actions[t0 + k - 1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    z, _ = dyn(z, a)
                    recon = dec(z[:, 2:11]).cpu().numpy()[0, 0]
                    mses[k].append(((recon - imgs[t0 + k]) ** 2).mean())
        except Exception as e:
            pass
    return mses


def eval_piwm_v3():
    """V3: Free latent dims (9 physics + 8 free)."""
    ck = load_checkpoint("checkpoints/piwm_v3/best.tar")
    enc = EncoderV3().to(DEVICE)
    dec = DecoderV3().to(DEVICE)
    dyn = DynamicsV3().to(DEVICE)
    enc.load_state_dict(ck['encoder']); enc.eval()
    dec.load_state_dict(ck['decoder']); dec.eval()
    dyn.load_state_dict(ck['dynamics']); dyn.eval()

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
            t0 = FRAME_STACK - 1
            init_norm = (physics[t0] - PHYSICS_MEAN) / PHYSICS_STD

            stack0 = np.stack([imgs[t0 - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)

            with torch.no_grad():
                z_enc = enc(torch.tensor(stack0, dtype=torch.float32).unsqueeze(0).to(DEVICE))[0]
                physics_pred = z_enc[:9]; free_pred = z_enc[9:]

                z = torch.zeros(1, 19, device=DEVICE)
                z[0, 0:2] = torch.tensor(init_norm[0:2], dtype=torch.float32, device=DEVICE)
                z[0, 2:11] = physics_pred
                z[0, 11:19] = free_pred

                recon = dec(torch.cat([z[0, 2:11], z[0, 11:19]]).unsqueeze(0)).cpu().numpy()[0, 0]
                mses[0].append(((recon - imgs[t0]) ** 2).mean())

                for k in range(1, T - t0):
                    a = torch.tensor(actions[t0 + k - 1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    z, _ = dyn(z, a)
                    recon = dec(torch.cat([z[0, 2:11], z[0, 11:19]]).unsqueeze(0)).cpu().numpy()[0, 0]
                    mses[k].append(((recon - imgs[t0 + k]) ** 2).mean())
        except Exception as e:
            pass
    return mses


def eval_baseline():
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
        except: pass
    return mses


def plot_all(all_mses, save_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {
        'V0 (original)': '#888888',
        'V1 (multistep)': '#6bba55',
        'V2 (multistep+e2e)': '#0050dc',
        'V3 (free latent)': '#aa00cc',
        'Baseline (black-box)': '#dc2828',
    }
    for name, mses in all_mses.items():
        steps, means, stds = [], [], []
        for k in range(MAX_STEPS + 1):
            if len(mses[k]) >= 5:
                steps.append(k); means.append(np.mean(mses[k])); stds.append(np.std(mses[k]))
        steps = np.array(steps); means = np.array(means); stds = np.array(stds)
        color = colors.get(name, 'black')
        ax.plot(steps, means, color=color, linewidth=2.3, label=name)
        ax.fill_between(steps, np.maximum(0, means - stds), means + stds, color=color, alpha=0.12)

    ax.set_xlabel('Prediction Horizon (steps)', fontsize=12)
    ax.set_ylabel('Image MSE (per pixel)', fontsize=12)
    ax.set_title('PIWM Ablation: Image MSE vs Rollout Step', fontsize=13)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, MAX_STEPS])
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    os.makedirs("vis", exist_ok=True)
    print("Evaluating V0...");         v0 = eval_piwm_v0()
    print("Evaluating V1...");         v1 = eval_piwm_v1()
    print("Evaluating V2...");         v2 = eval_piwm_v2()
    print("Evaluating V3...");         v3 = eval_piwm_v3()
    print("Evaluating Baseline...");   base = eval_baseline()

    all_mses = {
        'V0 (original)': v0,
        'V1 (multistep)': v1,
        'V2 (multistep+e2e)': v2,
        'V3 (free latent)': v3,
        'Baseline (black-box)': base,
    }
    plot_all(all_mses, "vis/ablation_all.png")

    # Print table
    print(f"\n{'Step':>5s}  " + "  ".join([f"{n:>20s}" for n in all_mses.keys()]))
    print("-" * (5 + 23 * len(all_mses)))
    for k in [0, 1, 5, 10, 20, 30, 50, 75, 100]:
        row = f"{k:>5d}  "
        for name, m in all_mses.items():
            if m[k]:
                row += f"{np.mean(m[k]):>20.6f}  "
            else:
                row += f"{'--':>20s}  "
        print(row)
