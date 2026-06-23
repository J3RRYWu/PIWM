"""Final comparison: all methods use the SAME PIWM Encoder/Decoder, differ only in Dynamics.

This matches Zhenjiang's PIWM paper comparison protocol.

Methods:
  - PIWM-linear  (our V2_rel Dynamics)
  - PIWM-bicycle (our bicycle+residual Dynamics)
  - SINDYc       (sparse regression)
  - GOKU         (kinematics + learned force, lighter than PIWM)
  - DVBF         (locally-linear stochastic)
  - Vid2Param    (theta-parameterized physics)

Shared Encoder/Decoder from checkpoints/piwm_v2_rel/best.tar.
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

from config import (DEVICE, DATA_DIR, PHYSICS_MEAN_REL, PHYSICS_STD_REL, FRAME_STACK)
from relative_coords import to_relative_np
from models.encoder import PhysicsEncoder
from models.decoder import PhysicsDecoder as PIWMDecoder
from models.dynamics import PhysicsDynamics
from models.dynamics_bicycle import BicycleDynamics
from baselines.shared_dynamics import DynamicsDVBF, DynamicsGOKU, DynamicsVid2Param
from utils import load_checkpoint

MAX_STEPS = 100
NUM_EPISODES = 20
THETA_HIST = 4


def patch_dyn_rel(dyn):
    from config import DT
    std = PHYSICS_STD_REL; mean = PHYSICS_MEAN_REL
    dyn.dt_vx_x = torch.tensor(float(std[3] * DT / std[0])).to(dyn.dt_vx_x)
    dyn.dt_vy_y = torch.tensor(float(std[4] * DT / std[1])).to(dyn.dt_vy_y)
    dyn.dt_om_yaw = torch.tensor(float(std[5] * DT / std[2])).to(dyn.dt_om_yaw)
    dyn.mean_vx_dt_over_std_x = torch.tensor(float(mean[3] * DT / std[0])).to(dyn.mean_vx_dt_over_std_x)
    dyn.mean_vy_dt_over_std_y = torch.tensor(float(mean[4] * DT / std[1])).to(dyn.mean_vy_dt_over_std_y)
    dyn.mean_om_dt_over_std_yaw = torch.tensor(float(mean[5] * DT / std[2])).to(dyn.mean_om_dt_over_std_yaw)
    return dyn


def load_shared_ae():
    """Load PIWM's encoder/decoder."""
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
            if len(imgs) < FRAME_STACK + THETA_HIST + MAX_STEPS + 1: continue
            episodes.append({'imgs': imgs, 'phys': phys, 'acts': acts})
        except: pass
    print(f"Loaded {len(episodes)} episodes")
    return episodes


def eval_shared(episodes, enc, dec, dyn_fn, needs_theta=False, theta_fn=None):
    """Generic shared-encoder rollout evaluator.

    dyn_fn(z, a): returns z_next (and optionally theta-dependent).
    """
    img_mses = [[] for _ in range(MAX_STEPS + 1)]
    state_mses = [[] for _ in range(MAX_STEPS + 1)]

    for ep in episodes:
        imgs = ep['imgs']; phys = ep['phys']; acts = ep['acts']
        t0 = FRAME_STACK - 1 + (THETA_HIST - 1 if needs_theta else 0)
        if t0 + MAX_STEPS >= len(imgs): continue

        # Encoder uses the stack ending at t0
        stack0 = np.stack([imgs[t0 - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
        rel_phys = to_relative_np(phys[t0:t0 + MAX_STEPS + 1], ref_idx=0)
        rel_norm = (rel_phys - PHYSICS_MEAN_REL) / PHYSICS_STD_REL

        with torch.no_grad():
            # Encoder gives 9-dim; combine with GT (relative) x, y for full 11-dim
            z_enc = enc(torch.tensor(stack0, dtype=torch.float32).unsqueeze(0).to(DEVICE))[0]
            z = torch.zeros(1, 11, device=DEVICE)
            z[0, 0:2] = torch.tensor(rel_norm[0, 0:2], device=DEVICE)
            z[0, 2:11] = z_enc

            # Optional theta inference for Vid2Param
            # FAIR: theta is inferred from IMAGES (via shared encoder), not GT physics.
            theta = None
            if needs_theta:
                hist_stacks = []
                for k in range(THETA_HIST):
                    center = t0 - THETA_HIST + 1 + k
                    st = np.stack([imgs[center - FRAME_STACK + 1 + j]
                                   for j in range(FRAME_STACK)], axis=0)
                    hist_stacks.append(st)
                hist_stacks = np.stack(hist_stacks, axis=0)   # (T_hist, FS, 64, 64)
                hist_t = torch.tensor(hist_stacks, dtype=torch.float32).to(DEVICE)
                hist_obs = enc(hist_t).unsqueeze(0)           # (1, T_hist, 9)
                theta, _, _ = theta_fn(hist_obs)

            # Step 0: reconstruct current image
            recon = dec(z[:, 2:11]).cpu().numpy()[0, 0]
            img_mses[0].append(((recon - imgs[t0]) ** 2).mean())
            state_mses[0].append((z[0].cpu().numpy() - rel_norm[0]) ** 2)

            for k in range(1, MAX_STEPS + 1):
                a = torch.tensor(acts[t0 + k - 1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                if needs_theta:
                    z = dyn_fn(z, a, theta)
                else:
                    result = dyn_fn(z, a)
                    z = result[0] if isinstance(result, tuple) else result
                recon = dec(z[:, 2:11]).cpu().numpy()[0, 0]
                img_mses[k].append(((recon - imgs[t0 + k]) ** 2).mean())
                state_mses[k].append((z[0].cpu().numpy() - rel_norm[k]) ** 2)

    return img_mses, state_mses


def eval_piwm_linear(episodes, enc, dec):
    dyn = PhysicsDynamics().to(DEVICE)
    dyn.load_state_dict(load_checkpoint("checkpoints/piwm_v2_rel/best.tar")['dynamics'])
    patch_dyn_rel(dyn); dyn.eval()
    return eval_shared(episodes, enc, dec, lambda z, a: dyn(z, a))


def eval_piwm_bicycle(episodes, enc, dec):
    dyn = BicycleDynamics().to(DEVICE)
    dyn.load_state_dict(load_checkpoint("checkpoints/piwm_bike/best.tar")['dynamics'])
    dyn.eval()
    return eval_shared(episodes, enc, dec, lambda z, a: dyn(z, a))


def eval_dvbf(episodes, enc, dec):
    dyn = DynamicsDVBF().to(DEVICE)
    dyn.load_state_dict(load_checkpoint("checkpoints/shared_dvbf/best.tar")['model'])
    dyn.eval()
    return eval_shared(episodes, enc, dec, lambda z, a: dyn(z, a))


def eval_goku(episodes, enc, dec):
    dyn = DynamicsGOKU().to(DEVICE)
    dyn.load_state_dict(load_checkpoint("checkpoints/shared_goku/best.tar")['model'])
    dyn.eval()
    return eval_shared(episodes, enc, dec, lambda z, a: dyn(z, a))


def eval_v2p(episodes, enc, dec):
    dyn = DynamicsVid2Param().to(DEVICE)
    dyn.load_state_dict(load_checkpoint("checkpoints/shared_v2p/best.tar")['model'])
    dyn.eval()
    return eval_shared(episodes, enc, dec,
                       dyn_fn=lambda z, a, t: dyn.step(z, a, t),
                       needs_theta=True, theta_fn=dyn.infer_theta)


def eval_sindyc(episodes, enc, dec):
    with open("checkpoints/sindyc/model.pkl", "rb") as f:
        model = pickle.load(f)

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
            z = np.zeros(11, dtype=np.float32)
            z[0:2] = rel_norm[0, 0:2]
            z[2:11] = z_enc.cpu().numpy()

            recon = dec(torch.tensor(z[2:11], dtype=torch.float32).unsqueeze(0).to(DEVICE)).cpu().numpy()[0, 0]
            img_mses[0].append(((recon - imgs[t0]) ** 2).mean())
            state_mses[0].append((z - rel_norm[0]) ** 2)

            for k in range(1, MAX_STEPS + 1):
                u = np.atleast_2d(acts[t0 + k - 1])
                try:
                    z = model.predict(z.reshape(1, -1), u=u).squeeze(0).astype(np.float32)
                except Exception:
                    pass
                recon = dec(torch.tensor(z[2:11], dtype=torch.float32).unsqueeze(0).to(DEVICE)).cpu().numpy()[0, 0]
                img_mses[k].append(((recon - imgs[t0 + k]) ** 2).mean())
                state_mses[k].append((z - rel_norm[k]) ** 2)
    return img_mses, state_mses


def plot_image_mse(all_results, save_path):
    colors = {
        'PIWM-linear': '#0050dc', 'PIWM-bicycle': '#00aaff',
        'SINDYc': '#aa00cc', 'GOKU-net': '#6bba55',
        'Vid2Param': '#ff8c00', 'DVBF': '#dc2828',
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, (img_mses, _) in all_results.items():
        steps, means, stds = [], [], []
        for k in range(len(img_mses)):
            if len(img_mses[k]) >= 3:
                arr = np.array(img_mses[k])
                steps.append(k); means.append(arr.mean()); stds.append(arr.std())
        c = colors.get(name, 'gray')
        ax.plot(steps, means, color=c, linewidth=2.2, label=name)
        ax.fill_between(steps, np.maximum(0, np.array(means)-np.array(stds)),
                        np.array(means)+np.array(stds), color=c, alpha=0.1)
    ax.set_xlabel('Prediction Horizon (steps)', fontsize=12)
    ax.set_ylabel('Image MSE (per pixel)', fontsize=12)
    ax.set_title('Image MSE vs Rollout Step (shared encoder/decoder)', fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0, top=0.025)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"Saved {save_path}")


def plot_state_mse(all_results, save_path):
    colors = {
        'PIWM-linear': '#0050dc', 'PIWM-bicycle': '#00aaff',
        'SINDYc': '#aa00cc', 'GOKU-net': '#6bba55',
        'Vid2Param': '#ff8c00', 'DVBF': '#dc2828',
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    dim_idx = [0, 1, 2]
    titles = ['x_rel', 'y_rel', 'yaw_rel']
    for ax, title, idx in zip(axes, titles, dim_idx):
        for name, (_, state_mses) in all_results.items():
            steps, means = [], []
            for k in range(len(state_mses)):
                if len(state_mses[k]) >= 3:
                    arr = np.array(state_mses[k])
                    steps.append(k); means.append(arr[:, idx].mean())
            c = colors.get(name, 'gray')
            ax.plot(steps, means, color=c, linewidth=2.2, label=name)
        ax.set_xlabel('Step'); ax.set_ylabel(f'{title} MSE')
        ax.set_title(title); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        # Auto-limit y to 95th percentile across methods to avoid outlier blowup
        all_vals = []
        for name2, (_, sm2) in all_results.items():
            for k in range(len(sm2)):
                if len(sm2[k]) >= 3:
                    all_vals.append(np.array(sm2[k])[:, idx].mean())
        if all_vals:
            y_top = np.percentile(all_vals, 95) * 1.2
            ax.set_ylim(bottom=0, top=max(y_top, 0.01))
    fig.suptitle('State MSE vs Rollout Step (normalized, shared encoder)', fontsize=13)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    os.makedirs("vis", exist_ok=True)
    episodes = load_test_episodes()
    enc, dec = load_shared_ae()

    all_results = {}
    print("[1/6] PIWM-linear...")
    all_results['PIWM-linear'] = eval_piwm_linear(episodes, enc, dec)
    try:
        print("[2/6] PIWM-bicycle...")
        all_results['PIWM-bicycle'] = eval_piwm_bicycle(episodes, enc, dec)
    except FileNotFoundError as e: print(f"Skip bicycle: {e}")
    print("[3/6] SINDYc...")
    all_results['SINDYc'] = eval_sindyc(episodes, enc, dec)
    try:
        print("[4/6] GOKU (shared)...")
        all_results['GOKU-net'] = eval_goku(episodes, enc, dec)
    except FileNotFoundError as e: print(f"Skip GOKU: {e}")
    try:
        print("[5/6] DVBF (shared)...")
        all_results['DVBF'] = eval_dvbf(episodes, enc, dec)
    except FileNotFoundError as e: print(f"Skip DVBF: {e}")
    try:
        print("[6/6] Vid2Param (shared)...")
        all_results['Vid2Param'] = eval_v2p(episodes, enc, dec)
    except FileNotFoundError as e: print(f"Skip V2P: {e}")

    plot_image_mse(all_results, "vis/shared_image_mse.png")
    plot_state_mse(all_results, "vis/shared_state_mse.png")

    # Summary
    print(f"\n{'Step':>5s} " + " ".join([f"{n:>15s}" for n in all_results.keys()]))
    for k in [0, 5, 10, 20, 30, 50]:
        row = f"{k:>5d} "
        for name, (img, _) in all_results.items():
            row += f" {np.mean(img[k]) if k < len(img) and img[k] else -1:>15.6f}"
        print(row)
