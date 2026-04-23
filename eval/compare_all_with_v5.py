"""Final all-method comparison including v5 (lane-augmented).

Compares on shared 100-step rollout, 20 episodes:
  - PIWM-linear, PIWM-bicycle (v1), PIWM-bicycle-v4 (dynamic), PIWM-lane-v5
  - SINDYc, GOKU-net, DVBF, Vid2Param (fair)

v5 uses its own encoder/decoder (29-dim). All other methods share the
3-frame PhysicsEncoder (9-dim) from piwm_v2_rel/best.tar.

Outputs:
  vis/all_with_v5_state_mse.png
  vis/all_with_v5_image_mse.png
  vis/all_with_v5_lane_mse.png    (v5 only)
"""

# --- auto-added: make repo root importable when run as a script ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end shim ---


import os, glob, pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (DEVICE, DATA_DIR, PHYSICS_MEAN_REL, PHYSICS_STD_REL, FRAME_STACK,
                    ENCODER_DIM as CAR_ENCODER_DIM)
from lane_utils import LANE_DIM, LANE_MEAN, LANE_STD, LANE_FRAME_STACK
from baselines.shared_dynamics_lane import (DynamicsDVBFLane, DynamicsGOKULane,
                                             DynamicsVid2ParamLane)
from relative_coords import to_relative_np
from models.encoder import PhysicsEncoder
from models.decoder import PhysicsDecoder
from models.dynamics import PhysicsDynamics
from models.dynamics_bicycle import BicycleDynamics
from models.dynamics_bicycle_v4 import BicycleDynamicsV4
from models.encoder_lane import PhysicsEncoderLane
from models.decoder_lane import PhysicsDecoderLane
from models.dynamics_lane_v5 import LaneAugmentedV5
from models.dynamics_lane_v6 import LaneAugmentedV6
from baselines.shared_dynamics import DynamicsDVBF, DynamicsGOKU, DynamicsVid2Param
from utils import load_checkpoint

MAX_STEPS = 100
NUM_EPISODES = 20
THETA_HIST = 4

COLORS = {
    'PIWM-linear':     '#0050dc',
    'PIWM-bicycle':    '#00aaff',
    'PIWM-bicycle-v4': '#00a050',
    'PIWM-lane-v5':    '#e63946',     # red, the new hero
    'SINDYc':          '#aa00cc',
    'GOKU-net':        '#6bba55',
    'DVBF':            '#dc2828',
    'Vid2Param':       '#ff8c00',
    # Lane-augmented baselines (15-frame)
    'DVBF-lane':       '#ff88aa',
    'GOKU-lane':       '#aaff88',
    'V2P-lane':        '#ffaa55',
    'SINDYc-lane':     '#cc88dd',
    'PIWM-lane-v6':    '#5500aa',     # purple: resample-based
}
LW = {'PIWM-lane-v5': 3.0, 'PIWM-bicycle-v4': 2.6, 'PIWM-lane-v6': 3.2}
LS = {'DVBF-lane': '--', 'GOKU-lane': '--', 'V2P-lane': '--', 'SINDYc-lane': '--'}


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
    ck = load_checkpoint("checkpoints/piwm_v2_rel/best.tar")
    enc = PhysicsEncoder().to(DEVICE); enc.load_state_dict(ck['encoder']); enc.eval()
    dec = PhysicsDecoder().to(DEVICE); dec.load_state_dict(ck['decoder']); dec.eval()
    return enc, dec


def load_v5_ae():
    """Try best.tar first (post-stage-3), fall back to ae.tar (stage 1)."""
    enc = PhysicsEncoderLane().to(DEVICE)
    dec = PhysicsDecoderLane().to(DEVICE)
    if os.path.exists("checkpoints/piwm_lane_v5/best.tar"):
        ck = load_checkpoint("checkpoints/piwm_lane_v5/best.tar")
        print("  v5: using best.tar (post-stage-3)")
    else:
        ck = load_checkpoint("checkpoints/piwm_lane_v5/ae.tar")
        print("  v5: using ae.tar (stage-1 only)")
    enc.load_state_dict(ck['encoder']); dec.load_state_dict(ck['decoder'])
    enc.eval(); dec.eval()
    return enc, dec


def load_v5_dyn():
    """Use best.tar dynamics if available, else dyn.tar."""
    dyn = LaneAugmentedV5().to(DEVICE)
    if os.path.exists("checkpoints/piwm_lane_v5/best.tar"):
        ck = load_checkpoint("checkpoints/piwm_lane_v5/best.tar")
        if 'dynamics' in ck:
            dyn.load_state_dict(ck['dynamics'])
            print("  v5 dyn: from best.tar")
            dyn.eval(); return dyn
    ck = load_checkpoint("checkpoints/piwm_lane_v5/dyn.tar")
    dyn.load_state_dict(ck['dynamics'])
    print("  v5 dyn: from dyn.tar (stage-2 only)")
    dyn.eval(); return dyn


def load_episodes(n=NUM_EPISODES):
    files = sorted(glob.glob(f"{DATA_DIR}/*.npz"))
    files = [f for f in files if ".lane." not in f]
    eps = []
    for f in files[:n]:
        try:
            d = np.load(f, allow_pickle=True)
            imgs = d["imgs"].astype(np.float32)
            if imgs.max() > 1.0: imgs /= 255.0
            pos = d["position"].astype(np.float32); yaw = d["yaw"].astype(np.float32)
            vel = d["velocity"].astype(np.float32); omega = d["angular_velocity"].astype(np.float32)
            wheel = d["wheel_omega"].astype(np.float32); steer = d["steering_angle"].astype(np.float32)
            acts = d["action"].astype(np.float32)
            phys = np.column_stack([pos, yaw, vel, omega, wheel, steer])

            lane_f = f.replace(".npz", ".lane.npz")
            wp_world = None
            if os.path.exists(lane_f):
                wp_body = np.load(lane_f)["lane_wp_body"].astype(np.float32)
                cy = np.cos(yaw); sy = np.sin(yaw)
                wp_world = np.zeros_like(wp_body)
                wp_world[..., 0] = wp_body[..., 0] * cy[:, None] - wp_body[..., 1] * sy[:, None] + pos[:, 0:1]
                wp_world[..., 1] = wp_body[..., 0] * sy[:, None] + wp_body[..., 1] * cy[:, None] + pos[:, 1:2]

            if len(imgs) < FRAME_STACK + THETA_HIST + MAX_STEPS + 1: continue
            eps.append({'imgs': imgs, 'phys': phys, 'acts': acts, 'wp_world': wp_world})
        except Exception as e:
            print(f"  skip {f}: {e}")
    print(f"Loaded {len(eps)} episodes")
    return eps


def world_to_ref_body(wp_world, pos0, yaw0):
    rel = wp_world - pos0
    c = np.cos(yaw0); s = np.sin(yaw0)
    x =  rel[..., 0] * c + rel[..., 1] * s
    y = -rel[..., 0] * s + rel[..., 1] * c
    return np.stack([x, y], axis=-1).astype(np.float32)


def eval_shared(eps, enc, dec, dyn_fn, needs_theta=False, theta_fn=None):
    img_mses = [[] for _ in range(MAX_STEPS + 1)]
    state_mses = [[] for _ in range(MAX_STEPS + 1)]
    for ep in eps:
        imgs = ep['imgs']; phys = ep['phys']; acts = ep['acts']
        t0 = FRAME_STACK - 1 + (THETA_HIST - 1 if needs_theta else 0)
        if t0 + MAX_STEPS >= len(imgs): continue
        stack0 = np.stack([imgs[t0 - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
        rel_phys = to_relative_np(phys[t0:t0 + MAX_STEPS + 1], ref_idx=0)
        rel_norm = (rel_phys - PHYSICS_MEAN_REL) / PHYSICS_STD_REL
        with torch.no_grad():
            z_enc = enc(torch.tensor(stack0, dtype=torch.float32).unsqueeze(0).to(DEVICE))[0]
            z = torch.zeros(1, 11, device=DEVICE)
            z[0, 0:2] = torch.tensor(rel_norm[0, 0:2], device=DEVICE)
            z[0, 2:11] = z_enc
            theta = None
            if needs_theta:
                hist_stacks = []
                for k in range(THETA_HIST):
                    center = t0 - THETA_HIST + 1 + k
                    st = np.stack([imgs[center - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
                    hist_stacks.append(st)
                hist_stacks = np.stack(hist_stacks, axis=0)
                hist_t = torch.tensor(hist_stacks, dtype=torch.float32).to(DEVICE)
                hist_obs = enc(hist_t).unsqueeze(0)
                theta, _, _ = theta_fn(hist_obs)
            r0 = dec(z[:, 2:11]).cpu().numpy()[0, 0]
            img_mses[0].append(((r0 - imgs[t0]) ** 2).mean())
            state_mses[0].append((z[0].cpu().numpy() - rel_norm[0]) ** 2)
            for k in range(1, MAX_STEPS + 1):
                a = torch.tensor(acts[t0 + k - 1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                if needs_theta:
                    z = dyn_fn(z, a, theta)
                else:
                    result = dyn_fn(z, a)
                    z = result[0] if isinstance(result, tuple) else result
                rec = dec(z[:, 2:11]).cpu().numpy()[0, 0]
                img_mses[k].append(((rec - imgs[t0 + k]) ** 2).mean())
                state_mses[k].append((z[0].cpu().numpy() - rel_norm[k]) ** 2)
    return img_mses, state_mses


def eval_sindyc(eps, enc, dec):
    with open("checkpoints/sindyc/model.pkl", "rb") as f:
        model = pickle.load(f)
    img_mses = [[] for _ in range(MAX_STEPS + 1)]
    state_mses = [[] for _ in range(MAX_STEPS + 1)]
    for ep in eps:
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
            r0 = dec(torch.tensor(z[2:11], dtype=torch.float32).unsqueeze(0).to(DEVICE)).cpu().numpy()[0, 0]
            img_mses[0].append(((r0 - imgs[t0]) ** 2).mean())
            state_mses[0].append((z - rel_norm[0]) ** 2)
            for k in range(1, MAX_STEPS + 1):
                u = np.atleast_2d(acts[t0 + k - 1])
                try:
                    z = model.predict(z.reshape(1, -1), u=u).squeeze(0).astype(np.float32)
                except Exception: pass
                rec = dec(torch.tensor(z[2:11], dtype=torch.float32).unsqueeze(0).to(DEVICE)).cpu().numpy()[0, 0]
                img_mses[k].append(((rec - imgs[t0 + k]) ** 2).mean())
                state_mses[k].append((z - rel_norm[k]) ** 2)
    return img_mses, state_mses


def eval_lane_baseline(eps, enc, dec, dyn, fs=LANE_FRAME_STACK, needs_theta=False, theta_fn=None):
    """Evaluate lane-augmented baseline (DVBF-lane / GOKU-lane / V2P-lane).
    Uses v5's encoder (29-dim), dynamics operate on 31-dim state.
    Returns (img_mses, state_mses, lane_mses) all over 100 steps.
    """
    img_mses = [[] for _ in range(MAX_STEPS + 1)]
    state_mses = [[] for _ in range(MAX_STEPS + 1)]
    lane_mses = [[] for _ in range(MAX_STEPS + 1)]
    for ep in eps:
        imgs = ep['imgs']; phys = ep['phys']; acts = ep['acts']; wp_world = ep['wp_world']
        if wp_world is None: continue
        t0 = fs - 1 + (THETA_HIST - 1 if needs_theta else 0)
        if t0 + MAX_STEPS >= len(imgs): continue
        stack0 = np.stack([imgs[t0 - fs + 1 + j] for j in range(fs)], axis=0)
        rel_phys = to_relative_np(phys[t0:t0 + MAX_STEPS + 1], ref_idx=0)
        rel_norm = (rel_phys - PHYSICS_MEAN_REL) / PHYSICS_STD_REL
        pos0 = phys[t0, 0:2]; yaw0 = phys[t0, 2]
        wp_ref = world_to_ref_body(wp_world[t0:t0 + MAX_STEPS + 1], pos0, yaw0)
        wp_norm = (wp_ref.reshape(-1, LANE_DIM) - LANE_MEAN) / LANE_STD
        with torch.no_grad():
            z_obs = enc(torch.tensor(stack0, dtype=torch.float32).unsqueeze(0).to(DEVICE))[0]
            z = torch.zeros(1, 31, device=DEVICE)
            z[0, 0:2] = torch.tensor(rel_norm[0, 0:2], device=DEVICE)
            z[0, 2:11] = z_obs[:CAR_ENCODER_DIM]
            z[0, 11:]  = z_obs[CAR_ENCODER_DIM:]
            theta = None
            if needs_theta:
                hist_stacks = []
                for k in range(THETA_HIST):
                    center = t0 - THETA_HIST + 1 + k
                    st = np.stack([imgs[center - fs + 1 + j] for j in range(fs)], axis=0)
                    hist_stacks.append(st)
                hist_stacks = np.stack(hist_stacks, axis=0)
                hist_t = torch.tensor(hist_stacks, dtype=torch.float32).to(DEVICE)
                hist_obs = enc(hist_t).unsqueeze(0)
                theta, _, _ = theta_fn(hist_obs)
            r0 = dec(torch.cat([z[:, 2:11], z[:, 11:]], dim=-1)).cpu().numpy()[0, 0]
            img_mses[0].append(((r0 - imgs[t0]) ** 2).mean())
            state_mses[0].append((z[0, :11].cpu().numpy() - rel_norm[0]) ** 2)
            lane_mses[0].append((z[0, 11:].cpu().numpy() - wp_norm[0]) ** 2)
            for k in range(1, MAX_STEPS + 1):
                a = torch.tensor(acts[t0 + k - 1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                if needs_theta:
                    z = dyn.step(z, a, theta)
                else:
                    z = dyn(z, a)
                rec = dec(torch.cat([z[:, 2:11], z[:, 11:]], dim=-1)).cpu().numpy()[0, 0]
                img_mses[k].append(((rec - imgs[t0 + k]) ** 2).mean())
                state_mses[k].append((z[0, :11].cpu().numpy() - rel_norm[k]) ** 2)
                lane_mses[k].append((z[0, 11:].cpu().numpy() - wp_norm[k]) ** 2)
    return img_mses, state_mses, lane_mses


def eval_sindyc_lane(eps, enc, dec, ckpt="checkpoints/sindyc_lane/model.pkl",
                     fs=LANE_FRAME_STACK):
    """SINDYc-lane: 31-dim pysindy on (car || lane). Inits from v5 encoder."""
    with open(ckpt, "rb") as f:
        model = pickle.load(f)
    img_mses = [[] for _ in range(MAX_STEPS + 1)]
    state_mses = [[] for _ in range(MAX_STEPS + 1)]
    lane_mses = [[] for _ in range(MAX_STEPS + 1)]
    for ep in eps:
        imgs = ep['imgs']; phys = ep['phys']; acts = ep['acts']; wp_world = ep['wp_world']
        if wp_world is None: continue
        t0 = fs - 1
        if t0 + MAX_STEPS >= len(imgs): continue
        stack0 = np.stack([imgs[t0 - fs + 1 + j] for j in range(fs)], axis=0)
        rel_phys = to_relative_np(phys[t0:t0 + MAX_STEPS + 1], ref_idx=0)
        rel_norm = (rel_phys - PHYSICS_MEAN_REL) / PHYSICS_STD_REL
        pos0 = phys[t0, 0:2]; yaw0 = phys[t0, 2]
        wp_ref = world_to_ref_body(wp_world[t0:t0 + MAX_STEPS + 1], pos0, yaw0)
        wp_norm = (wp_ref.reshape(-1, LANE_DIM) - LANE_MEAN) / LANE_STD
        with torch.no_grad():
            z_obs = enc(torch.tensor(stack0, dtype=torch.float32).unsqueeze(0).to(DEVICE))[0]
            z = np.zeros(31, dtype=np.float32)
            z[0:2]  = rel_norm[0, 0:2]
            z[2:11] = z_obs[:CAR_ENCODER_DIM].cpu().numpy()
            z[11:]  = z_obs[CAR_ENCODER_DIM:].cpu().numpy()
            r0 = dec(torch.cat([
                torch.tensor(z[2:11], dtype=torch.float32).unsqueeze(0).to(DEVICE),
                torch.tensor(z[11:], dtype=torch.float32).unsqueeze(0).to(DEVICE),
            ], dim=-1)).cpu().numpy()[0, 0]
            img_mses[0].append(((r0 - imgs[t0]) ** 2).mean())
            state_mses[0].append((z[:11] - rel_norm[0]) ** 2)
            lane_mses[0].append((z[11:] - wp_norm[0]) ** 2)
            for k in range(1, MAX_STEPS + 1):
                u = np.atleast_2d(acts[t0 + k - 1])
                try:
                    z = model.predict(z.reshape(1, -1), u=u).squeeze(0).astype(np.float32)
                except Exception: pass
                rec = dec(torch.cat([
                    torch.tensor(z[2:11], dtype=torch.float32).unsqueeze(0).to(DEVICE),
                    torch.tensor(z[11:], dtype=torch.float32).unsqueeze(0).to(DEVICE),
                ], dim=-1)).cpu().numpy()[0, 0]
                img_mses[k].append(((rec - imgs[t0 + k]) ** 2).mean())
                state_mses[k].append((z[:11] - rel_norm[k]) ** 2)
                lane_mses[k].append((z[11:] - wp_norm[k]) ** 2)
    return img_mses, state_mses, lane_mses


def eval_v5(eps, enc, dec, dyn, fs=LANE_FRAME_STACK):
    img_mses = [[] for _ in range(MAX_STEPS + 1)]
    state_mses = [[] for _ in range(MAX_STEPS + 1)]
    lane_mses = [[] for _ in range(MAX_STEPS + 1)]
    for ep in eps:
        imgs = ep['imgs']; phys = ep['phys']; acts = ep['acts']; wp_world = ep['wp_world']
        if wp_world is None: continue
        t0 = fs - 1
        if t0 + MAX_STEPS >= len(imgs): continue
        stack0 = np.stack([imgs[t0 - fs + 1 + j] for j in range(fs)], axis=0)
        rel_phys = to_relative_np(phys[t0:t0 + MAX_STEPS + 1], ref_idx=0)
        rel_norm = (rel_phys - PHYSICS_MEAN_REL) / PHYSICS_STD_REL
        pos0 = phys[t0, 0:2]; yaw0 = phys[t0, 2]
        wp_ref = world_to_ref_body(wp_world[t0:t0 + MAX_STEPS + 1], pos0, yaw0)
        wp_norm = (wp_ref.reshape(-1, LANE_DIM) - LANE_MEAN) / LANE_STD
        with torch.no_grad():
            z_obs = enc(torch.tensor(stack0, dtype=torch.float32).unsqueeze(0).to(DEVICE))[0]
            z = torch.zeros(1, 31, device=DEVICE)
            z[0, 0:2] = torch.tensor(rel_norm[0, 0:2], device=DEVICE)
            z[0, 2:11] = z_obs[:CAR_ENCODER_DIM]
            z[0, 11:]  = z_obs[CAR_ENCODER_DIM:]
            r0 = dec(torch.cat([z[:, 2:11], z[:, 11:]], dim=-1)).cpu().numpy()[0, 0]
            img_mses[0].append(((r0 - imgs[t0]) ** 2).mean())
            state_mses[0].append((z[0, :11].cpu().numpy() - rel_norm[0]) ** 2)
            lane_mses[0].append((z[0, 11:].cpu().numpy() - wp_norm[0]) ** 2)
            for k in range(1, MAX_STEPS + 1):
                a = torch.tensor(acts[t0 + k - 1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                z, _ = dyn(z, a)
                rec = dec(torch.cat([z[:, 2:11], z[:, 11:]], dim=-1)).cpu().numpy()[0, 0]
                img_mses[k].append(((rec - imgs[t0 + k]) ** 2).mean())
                state_mses[k].append((z[0, :11].cpu().numpy() - rel_norm[k]) ** 2)
                lane_mses[k].append((z[0, 11:].cpu().numpy() - wp_norm[k]) ** 2)
    return img_mses, state_mses, lane_mses


def plot_state(results, save):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, title, idx in zip(axes, ['x_rel', 'y_rel', 'yaw_rel'], [0, 1, 2]):
        for name, sm in results.items():
            steps, means = [], []
            for k in range(len(sm)):
                if len(sm[k]) >= 3:
                    steps.append(k); means.append(np.array(sm[k])[:, idx].mean())
            ax.plot(steps, means, color=COLORS.get(name, 'gray'),
                    linewidth=LW.get(name, 2.0),
                    linestyle=LS.get(name, '-'), label=name)
        ax.set_xlabel('Step'); ax.set_ylabel(f'{title} MSE')
        ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
        vals = []
        for sm in results.values():
            for k in range(len(sm)):
                if len(sm[k]) >= 3:
                    vals.append(np.array(sm[k])[:, idx].mean())
        vals = [v for v in vals if np.isfinite(v)]
        if vals: ax.set_ylim(0, np.percentile(vals, 95) * 1.3)
    fig.suptitle('State MSE: all baselines + PIWM-lane-v5', fontsize=13)
    fig.tight_layout(); fig.savefig(save, dpi=150); plt.close(fig)
    print(f"Saved {save}")


def plot_image(results, save):
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, img in results.items():
        steps, means = [], []
        for k in range(len(img)):
            if len(img[k]) >= 3:
                steps.append(k); means.append(np.mean(img[k]))
        ax.plot(steps, means, color=COLORS.get(name, 'gray'),
                linewidth=LW.get(name, 2.0),
                linestyle=LS.get(name, '-'), label=name)
    ax.set_xlabel('Step'); ax.set_ylabel('Image MSE')
    ax.set_title('Image MSE: all baselines + PIWM-lane-v5')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
    ax.set_ylim(0, 0.030)
    fig.tight_layout(); fig.savefig(save, dpi=150); plt.close(fig)
    print(f"Saved {save}")


def plot_lane_multi(lane_results, save):
    """lane_results: dict name -> lane_mses list (None if method has no lane)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, lm in lane_results.items():
        if lm is None: continue
        steps, means = [], []
        for k in range(len(lm)):
            if len(lm[k]) >= 3:
                steps.append(k); means.append(np.array(lm[k]).mean())
        ax.plot(steps, means, color=COLORS.get(name, 'gray'),
                linewidth=LW.get(name, 2.2),
                linestyle=LS.get(name, '-'), label=name)
    ax.set_xlabel('Step'); ax.set_ylabel('Lane Waypoint MSE (normalized)')
    ax.set_title('Lane Waypoint MSE: lane-augmented methods')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(save, dpi=150); plt.close(fig)
    print(f"Saved {save}")


if __name__ == "__main__":
    os.makedirs("vis", exist_ok=True)
    eps = load_episodes()
    enc, dec = load_shared_ae()

    state_results, image_results = {}, {}

    print("[1] PIWM-linear...")
    dyn = PhysicsDynamics().to(DEVICE)
    dyn.load_state_dict(load_checkpoint("checkpoints/piwm_v2_rel/best.tar")['dynamics'])
    patch_dyn_rel(dyn); dyn.eval()
    img, sm = eval_shared(eps, enc, dec, lambda z, a: dyn(z, a))
    image_results['PIWM-linear'] = img; state_results['PIWM-linear'] = sm

    print("[2] PIWM-bicycle (v1)...")
    dyn_b = BicycleDynamics().to(DEVICE)
    dyn_b.load_state_dict(load_checkpoint("checkpoints/piwm_bike/best.tar")['dynamics'])
    dyn_b.eval()
    img, sm = eval_shared(eps, enc, dec, lambda z, a: dyn_b(z, a))
    image_results['PIWM-bicycle'] = img; state_results['PIWM-bicycle'] = sm

    print("[3] PIWM-bicycle-v4...")
    dyn4 = BicycleDynamicsV4().to(DEVICE)
    dyn4.load_state_dict(load_checkpoint("checkpoints/piwm_bike_v4/best.tar")['dynamics'])
    dyn4.eval()
    img, sm = eval_shared(eps, enc, dec, lambda z, a: dyn4(z, a))
    image_results['PIWM-bicycle-v4'] = img; state_results['PIWM-bicycle-v4'] = sm

    print("[4] SINDYc...")
    img, sm = eval_sindyc(eps, enc, dec)
    image_results['SINDYc'] = img; state_results['SINDYc'] = sm

    print("[5] GOKU-net...")
    dyn_g = DynamicsGOKU().to(DEVICE)
    dyn_g.load_state_dict(load_checkpoint("checkpoints/shared_goku/best.tar")['model'])
    dyn_g.eval()
    img, sm = eval_shared(eps, enc, dec, lambda z, a: dyn_g(z, a))
    image_results['GOKU-net'] = img; state_results['GOKU-net'] = sm

    print("[6] DVBF...")
    dyn_d = DynamicsDVBF().to(DEVICE)
    dyn_d.load_state_dict(load_checkpoint("checkpoints/shared_dvbf/best.tar")['model'])
    dyn_d.eval()
    img, sm = eval_shared(eps, enc, dec, lambda z, a: dyn_d(z, a))
    image_results['DVBF'] = img; state_results['DVBF'] = sm

    print("[7] Vid2Param (fair)...")
    dyn_v = DynamicsVid2Param().to(DEVICE)
    dyn_v.load_state_dict(load_checkpoint("checkpoints/shared_v2p/best.tar")['model'])
    dyn_v.eval()
    img, sm = eval_shared(eps, enc, dec,
                          dyn_fn=lambda z, a, t: dyn_v.step(z, a, t),
                          needs_theta=True, theta_fn=dyn_v.infer_theta)
    image_results['Vid2Param'] = img; state_results['Vid2Param'] = sm

    print("[8] PIWM-lane-v5...")
    enc5, dec5 = load_v5_ae()
    dyn5 = load_v5_dyn()
    img, sm, lm_v5 = eval_v5(eps, enc5, dec5, dyn5)
    image_results['PIWM-lane-v5'] = img; state_results['PIWM-lane-v5'] = sm

    print("[9] PIWM-lane-v6 (resample-based)...")
    enc6 = PhysicsEncoderLane().to(DEVICE)
    dec6 = PhysicsDecoderLane().to(DEVICE)
    ck6 = load_checkpoint("checkpoints/piwm_lane_v6/best.tar")
    enc6.load_state_dict(ck6['encoder']); dec6.load_state_dict(ck6['decoder'])
    enc6.eval(); dec6.eval()
    dyn6 = LaneAugmentedV6().to(DEVICE)
    dyn6.load_state_dict(ck6['dynamics']); dyn6.eval()
    img, sm, lm_v6 = eval_v5(eps, enc6, dec6, dyn6)   # eval_v5 is generic
    image_results['PIWM-lane-v6'] = img; state_results['PIWM-lane-v6'] = sm

    # --- Lane-augmented baselines (15-frame, v5 encoder) ---
    lane_results = {'PIWM-lane-v5': lm_v5, 'PIWM-lane-v6': lm_v6}

    sindyc_lane_ckpt = "checkpoints/sindyc_lane/model.pkl"
    if os.path.exists(sindyc_lane_ckpt):
        print("[*] SINDYc-lane...")
        img_s, sm_s, lm_s = eval_sindyc_lane(eps, enc5, dec5, ckpt=sindyc_lane_ckpt)
        image_results['SINDYc-lane'] = img_s
        state_results['SINDYc-lane'] = sm_s
        lane_results['SINDYc-lane'] = lm_s
    else:
        print(f"[skip] SINDYc-lane: {sindyc_lane_ckpt} not found")
        lane_results['SINDYc-lane'] = None

    for name, cls, ckpt_path, needs_theta in [
        ('DVBF-lane', DynamicsDVBFLane,       "checkpoints/shared_dvbf_lane/best.tar", False),
        ('GOKU-lane', DynamicsGOKULane,       "checkpoints/shared_goku_lane/best.tar", False),
        ('V2P-lane',  DynamicsVid2ParamLane,  "checkpoints/shared_v2p_lane/best.tar",  True),
    ]:
        if not os.path.exists(ckpt_path):
            print(f"[skip] {name}: {ckpt_path} not found")
            lane_results[name] = None
            continue
        print(f"[*] {name}...")
        dyn_b = cls().to(DEVICE)
        dyn_b.load_state_dict(load_checkpoint(ckpt_path)['model'])
        dyn_b.eval()
        theta_fn = dyn_b.infer_theta if needs_theta else None
        img_b, sm_b, lm_b = eval_lane_baseline(
            eps, enc5, dec5, dyn_b, needs_theta=needs_theta, theta_fn=theta_fn,
        )
        image_results[name] = img_b; state_results[name] = sm_b
        lane_results[name] = lm_b

    plot_state(state_results, "vis/all_with_v5_state_mse.png")
    plot_image(image_results, "vis/all_with_v5_image_mse.png")
    plot_lane_multi(lane_results, "vis/all_with_v5_lane_mse.png")

    # Numeric summary table
    print(f"\n{'':>5s} " + " ".join([f"{n:>16s}" for n in state_results.keys()]))
    for dim_name, dim_idx in [("x_rel", 0), ("y_rel", 1), ("yaw_rel", 2)]:
        print(f"\n--- {dim_name} state MSE ---")
        for k in [0, 10, 20, 50, 75, 100]:
            row = f"{k:>5d} "
            for sm in state_results.values():
                if k < len(sm) and sm[k]:
                    row += f" {np.array(sm[k])[:, dim_idx].mean():>16.4f}"
                else:
                    row += f" {'-':>16s}"
            print(row)

    print(f"\n--- Image MSE ---")
    for k in [0, 10, 20, 50, 75, 100]:
        row = f"{k:>5d} "
        for img in image_results.values():
            if k < len(img) and img[k]:
                row += f" {np.mean(img[k]):>16.5f}"
            else:
                row += f" {'-':>16s}"
        print(row)

    print(f"\n--- Lane MSE (lane-augmented methods) ---")
    lane_names = [n for n, lm in lane_results.items() if lm is not None]
    header = f"{'Step':>5s} " + " ".join([f"{n:>14s}" for n in lane_names])
    print(header)
    for k in [0, 10, 20, 50, 75, 100]:
        row = f"{k:>5d} "
        for n in lane_names:
            lm_n = lane_results[n]
            if k < len(lm_n) and lm_n[k]:
                row += f" {np.array(lm_n[k]).mean():>14.4f}"
            else:
                row += f" {'-':>14s}"
        print(row)
