"""Compare v5 (lane-augmented) vs v4 (dynamic bicycle) on shared eval protocol.

v5 uses its own encoder/decoder (29-dim, from ae.tar) because encoder
architecture differs from the 9-dim version used by v4.

Rollouts: 100 steps on 20 test episodes, measures state MSE (x, y, yaw),
image MSE, and v5-only lane MSE.
"""

import os, glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (DEVICE, DATA_DIR, PHYSICS_MEAN_REL, PHYSICS_STD_REL, FRAME_STACK,
                    ENCODER_DIM as CAR_ENCODER_DIM)
from lane_utils import LANE_DIM, LANE_MEAN, LANE_STD, N_LANE_WP
from relative_coords import to_relative_np
from models.encoder import PhysicsEncoder
from models.decoder import PhysicsDecoder
from models.dynamics_bicycle_v4 import BicycleDynamicsV4
from models.encoder_lane import PhysicsEncoderLane
from models.decoder_lane import PhysicsDecoderLane
from models.dynamics_lane_v5 import LaneAugmentedV5
from utils import load_checkpoint

MAX_STEPS = 100
NUM_EPISODES = 20


def load_test_episodes(n=NUM_EPISODES):
    files = sorted(glob.glob(f"{DATA_DIR}/*.npz"))
    files = [f for f in files if ".lane." not in f]
    episodes = []
    for f in files[:n]:
        lane_f = f.replace(".npz", ".lane.npz")
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

            wp_body = np.load(lane_f)["lane_wp_body"].astype(np.float32) if os.path.exists(lane_f) else None
            if wp_body is not None:
                cy = np.cos(yaw); sy = np.sin(yaw)
                wp_world = np.zeros_like(wp_body)
                wp_world[..., 0] = wp_body[..., 0] * cy[:, None] - wp_body[..., 1] * sy[:, None] + pos[:, 0:1]
                wp_world[..., 1] = wp_body[..., 0] * sy[:, None] + wp_body[..., 1] * cy[:, None] + pos[:, 1:2]
            else:
                wp_world = None

            if len(imgs) < FRAME_STACK + MAX_STEPS + 1: continue
            episodes.append({'imgs': imgs, 'phys': phys, 'acts': acts, 'wp_world': wp_world})
        except Exception as e:
            print(f"  skip {f}: {e}")
    print(f"Loaded {len(episodes)} episodes")
    return episodes


def world_to_ref_body(wp_world, pos0, yaw0):
    rel = wp_world - pos0
    c = np.cos(yaw0); s = np.sin(yaw0)
    x =  rel[..., 0] * c + rel[..., 1] * s
    y = -rel[..., 0] * s + rel[..., 1] * c
    return np.stack([x, y], axis=-1).astype(np.float32)


def eval_v4(episodes):
    print("[v4] loading...")
    ck_ae = load_checkpoint("checkpoints/piwm_v2_rel/best.tar")
    enc = PhysicsEncoder().to(DEVICE); enc.load_state_dict(ck_ae['encoder']); enc.eval()
    dec = PhysicsDecoder().to(DEVICE); dec.load_state_dict(ck_ae['decoder']); dec.eval()
    dyn = BicycleDynamicsV4().to(DEVICE)
    dyn.load_state_dict(load_checkpoint("checkpoints/piwm_bike_v4/best.tar")['dynamics'])
    dyn.eval()

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
            r0 = dec(z[:, 2:11]).cpu().numpy()[0, 0]
            img_mses[0].append(((r0 - imgs[t0]) ** 2).mean())
            state_mses[0].append((z[0].cpu().numpy() - rel_norm[0]) ** 2)
            for k in range(1, MAX_STEPS + 1):
                a = torch.tensor(acts[t0 + k - 1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                z, _ = dyn(z, a)
                rec = dec(z[:, 2:11]).cpu().numpy()[0, 0]
                img_mses[k].append(((rec - imgs[t0 + k]) ** 2).mean())
                state_mses[k].append((z[0].cpu().numpy() - rel_norm[k]) ** 2)
    return img_mses, state_mses, None


def eval_v5(episodes):
    print("[v5] loading (stage2 ae + stage2 dyn)...")
    enc = PhysicsEncoderLane().to(DEVICE)
    dec = PhysicsDecoderLane().to(DEVICE)
    dyn = LaneAugmentedV5().to(DEVICE)
    ae = load_checkpoint("checkpoints/piwm_lane_v5/ae.tar")
    enc.load_state_dict(ae['encoder']); dec.load_state_dict(ae['decoder'])
    dy = load_checkpoint("checkpoints/piwm_lane_v5/dyn.tar")
    dyn.load_state_dict(dy['dynamics'])
    enc.eval(); dec.eval(); dyn.eval()

    img_mses = [[] for _ in range(MAX_STEPS + 1)]
    state_mses = [[] for _ in range(MAX_STEPS + 1)]
    lane_mses = [[] for _ in range(MAX_STEPS + 1)]
    for ep in episodes:
        imgs = ep['imgs']; phys = ep['phys']; acts = ep['acts']; wp_world = ep['wp_world']
        if wp_world is None: continue
        t0 = FRAME_STACK - 1
        if t0 + MAX_STEPS >= len(imgs): continue
        stack0 = np.stack([imgs[t0 - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
        rel_phys = to_relative_np(phys[t0:t0 + MAX_STEPS + 1], ref_idx=0)
        rel_norm = (rel_phys - PHYSICS_MEAN_REL) / PHYSICS_STD_REL
        # GT lane in ref body frame
        pos0 = phys[t0, 0:2]; yaw0 = phys[t0, 2]
        wp_ref = world_to_ref_body(wp_world[t0:t0 + MAX_STEPS + 1], pos0, yaw0)
        wp_ref_norm = (wp_ref.reshape(-1, LANE_DIM) - LANE_MEAN) / LANE_STD

        with torch.no_grad():
            z_obs = enc(torch.tensor(stack0, dtype=torch.float32).unsqueeze(0).to(DEVICE))[0]
            z = torch.zeros(1, 31, device=DEVICE)
            z[0, 0:2] = torch.tensor(rel_norm[0, 0:2], device=DEVICE)
            z[0, 2:11] = z_obs[:CAR_ENCODER_DIM]
            z[0, 11:]  = z_obs[CAR_ENCODER_DIM:]
            r0 = dec(torch.cat([z[:, 2:11], z[:, 11:]], dim=-1)).cpu().numpy()[0, 0]
            img_mses[0].append(((r0 - imgs[t0]) ** 2).mean())
            state_mses[0].append((z[0, :11].cpu().numpy() - rel_norm[0]) ** 2)
            lane_mses[0].append((z[0, 11:].cpu().numpy() - wp_ref_norm[0]) ** 2)
            for k in range(1, MAX_STEPS + 1):
                a = torch.tensor(acts[t0 + k - 1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                z, _ = dyn(z, a)
                rec = dec(torch.cat([z[:, 2:11], z[:, 11:]], dim=-1)).cpu().numpy()[0, 0]
                img_mses[k].append(((rec - imgs[t0 + k]) ** 2).mean())
                state_mses[k].append((z[0, :11].cpu().numpy() - rel_norm[k]) ** 2)
                lane_mses[k].append((z[0, 11:].cpu().numpy() - wp_ref_norm[k]) ** 2)
    return img_mses, state_mses, lane_mses


def plot(all_results, save_path):
    """3-panel: state MSE (x, y, yaw) + Image MSE + Lane MSE (if available)."""
    has_lane = any(r[2] is not None for r in all_results.values())
    n_panels = 5 if has_lane else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 4.5, 4.5))
    colors = {'v4': '#00a050', 'v5-stage2': '#0080ff'}

    titles = ['x_rel', 'y_rel', 'yaw_rel']
    for ax, title, idx in zip(axes[:3], titles, [0, 1, 2]):
        for name, (img, sm, lm) in all_results.items():
            steps, means = [], []
            for k in range(len(sm)):
                if len(sm[k]) >= 3:
                    steps.append(k); means.append(np.array(sm[k])[:, idx].mean())
            ax.plot(steps, means, color=colors.get(name, 'gray'),
                    linewidth=2.2, label=name)
        ax.set_xlabel('Step'); ax.set_ylabel(f'{title} MSE')
        ax.set_title(f'State MSE: {title}'); ax.grid(True, alpha=0.3); ax.legend()
        vals = []
        for n2, (_, sm2, _) in all_results.items():
            for k in range(len(sm2)):
                if len(sm2[k]) >= 3:
                    vals.append(np.array(sm2[k])[:, idx].mean())
        if vals: ax.set_ylim(0, np.percentile(vals, 95) * 1.3)

    # Image MSE
    ax = axes[3]
    for name, (img, _, _) in all_results.items():
        steps, means = [], []
        for k in range(len(img)):
            if len(img[k]) >= 3:
                steps.append(k); means.append(np.mean(img[k]))
        ax.plot(steps, means, color=colors.get(name, 'gray'), linewidth=2.2, label=name)
    ax.set_xlabel('Step'); ax.set_ylabel('Image MSE')
    ax.set_title('Image MSE'); ax.grid(True, alpha=0.3); ax.legend()
    ax.set_ylim(0, 0.025)

    # Lane MSE (v5 only)
    if has_lane:
        ax = axes[4]
        for name, (_, _, lm) in all_results.items():
            if lm is None: continue
            steps, means = [], []
            for k in range(len(lm)):
                if len(lm[k]) >= 3:
                    steps.append(k); means.append(np.array(lm[k]).mean())
            ax.plot(steps, means, color=colors.get(name, 'gray'),
                    linewidth=2.2, label=name)
        ax.set_xlabel('Step'); ax.set_ylabel('Lane MSE (normalized, all 20 dims)')
        ax.set_title('Lane MSE (v5 only)'); ax.grid(True, alpha=0.3); ax.legend()

    fig.suptitle('v4 vs v5: 100-step rollout on 20 episodes', fontsize=13)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    os.makedirs("vis", exist_ok=True)
    episodes = load_test_episodes()
    res = {}
    res['v4']        = eval_v4(episodes)
    res['v5-stage2'] = eval_v5(episodes)
    plot(res, "vis/v5_vs_v4.png")

    # Summary
    print(f"\n{'Step':>5s} " + " ".join([f"{n:>15s}" for n in res.keys()]))
    for dim_name, dim_idx in [("x_rel", 0), ("y_rel", 1), ("yaw_rel", 2)]:
        print(f"\n--- {dim_name} state MSE ---")
        for k in [0, 10, 20, 50, 75, 100]:
            row = f"{k:>5d} "
            for name, (_, sm, _) in res.items():
                if k < len(sm) and sm[k]:
                    row += f" {np.array(sm[k])[:, dim_idx].mean():>15.4f}"
                else:
                    row += f" {'-':>15s}"
            print(row)

    print(f"\n--- Image MSE ---")
    for k in [0, 10, 20, 50, 75, 100]:
        row = f"{k:>5d} "
        for name, (img, _, _) in res.items():
            if k < len(img) and img[k]:
                row += f" {np.mean(img[k]):>15.5f}"
            else:
                row += f" {'-':>15s}"
        print(row)

    if res['v5-stage2'][2] is not None:
        print(f"\n--- Lane MSE (v5 only, normalized) ---")
        for k in [0, 10, 20, 50, 75, 100]:
            arr = res['v5-stage2'][2]
            if k < len(arr) and arr[k]:
                print(f"  step {k}: {np.array(arr[k]).mean():.5f}")
