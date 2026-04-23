"""Final state-MSE comparison: all baselines + PIWM-bicycle-v4 (dynamic).

Methods:
  - PIWM-linear           (our V2_rel Dynamics)
  - PIWM-bicycle          (v1 original kinematic bicycle)
  - PIWM-bicycle-v4       (dynamic bicycle + masked residual, our improved variant)
  - SINDYc
  - GOKU-net
  - DVBF
  - Vid2Param             (fair: theta inferred from encoded images)

Outputs:
  vis/final_state_mse.png   (x_rel, y_rel, yaw_rel)
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

from config import (DEVICE, DATA_DIR, PHYSICS_MEAN_REL, PHYSICS_STD_REL, FRAME_STACK)
from relative_coords import to_relative_np
from models.encoder import PhysicsEncoder
from models.decoder import PhysicsDecoder as PIWMDecoder
from models.dynamics import PhysicsDynamics
from models.dynamics_bicycle import BicycleDynamics
from models.dynamics_bicycle_v4 import BicycleDynamicsV4
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
    state_mses = [[] for _ in range(MAX_STEPS + 1)]
    for ep in episodes:
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
                    st = np.stack([imgs[center - FRAME_STACK + 1 + j]
                                   for j in range(FRAME_STACK)], axis=0)
                    hist_stacks.append(st)
                hist_stacks = np.stack(hist_stacks, axis=0)
                hist_t = torch.tensor(hist_stacks, dtype=torch.float32).to(DEVICE)
                hist_obs = enc(hist_t).unsqueeze(0)
                theta, _, _ = theta_fn(hist_obs)

            state_mses[0].append((z[0].cpu().numpy() - rel_norm[0]) ** 2)
            for k in range(1, MAX_STEPS + 1):
                a = torch.tensor(acts[t0 + k - 1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                if needs_theta:
                    z = dyn_fn(z, a, theta)
                else:
                    result = dyn_fn(z, a)
                    z = result[0] if isinstance(result, tuple) else result
                state_mses[k].append((z[0].cpu().numpy() - rel_norm[k]) ** 2)
    return state_mses


def eval_sindyc(episodes, enc, dec):
    with open("checkpoints/sindyc/model.pkl", "rb") as f:
        model = pickle.load(f)
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
            state_mses[0].append((z - rel_norm[0]) ** 2)
            for k in range(1, MAX_STEPS + 1):
                u = np.atleast_2d(acts[t0 + k - 1])
                try:
                    z = model.predict(z.reshape(1, -1), u=u).squeeze(0).astype(np.float32)
                except Exception:
                    pass
                state_mses[k].append((z - rel_norm[k]) ** 2)
    return state_mses


def plot_final(all_results, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ['x_rel', 'y_rel', 'yaw_rel']
    dims = [0, 1, 2]
    for ax, title, idx in zip(axes, titles, dims):
        for name, (state_mses, color, ls, lw) in all_results.items():
            steps, means = [], []
            for k in range(len(state_mses)):
                if len(state_mses[k]) >= 3:
                    arr = np.array(state_mses[k])
                    steps.append(k); means.append(arr[:, idx].mean())
            ax.plot(steps, means, color=color, linewidth=lw, linestyle=ls, label=name)
        ax.set_xlabel('Prediction Horizon (steps)', fontsize=12)
        ax.set_ylabel(f'{title} MSE', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
        all_vals = []
        for _, (sm2, _, _, _) in all_results.items():
            for k in range(len(sm2)):
                if len(sm2[k]) >= 3:
                    all_vals.append(np.array(sm2[k])[:, idx].mean())
        if all_vals:
            y_top = np.percentile(all_vals, 95) * 1.2
            ax.set_ylim(bottom=0, top=max(y_top, 0.01))
    fig.suptitle('State MSE vs Rollout Step  (all baselines + PIWM-bicycle-v4)', fontsize=13)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    os.makedirs("vis", exist_ok=True)
    episodes = load_test_episodes()
    enc, dec = load_shared_ae()

    # (color, linestyle, linewidth)
    all_results = {}

    print("[1] PIWM-linear...")
    dyn = PhysicsDynamics().to(DEVICE)
    dyn.load_state_dict(load_checkpoint("checkpoints/piwm_v2_rel/best.tar")['dynamics'])
    patch_dyn_rel(dyn); dyn.eval()
    all_results['PIWM-linear'] = (
        eval_shared(episodes, enc, dec, lambda z, a: dyn(z, a)),
        '#0050dc', '-', 2.2,
    )

    print("[2] PIWM-bicycle (v1 orig)...")
    dyn_b = BicycleDynamics().to(DEVICE)
    dyn_b.load_state_dict(load_checkpoint("checkpoints/piwm_bike/best.tar")['dynamics'])
    dyn_b.eval()
    all_results['PIWM-bicycle'] = (
        eval_shared(episodes, enc, dec, lambda z, a: dyn_b(z, a)),
        '#00aaff', '-', 2.2,
    )

    print("[3] PIWM-bicycle-v4 (dynamic)...")
    dyn_v4 = BicycleDynamicsV4().to(DEVICE)
    dyn_v4.load_state_dict(load_checkpoint("checkpoints/piwm_bike_v4/best.tar")['dynamics'])
    dyn_v4.eval()
    all_results['PIWM-bicycle-v4'] = (
        eval_shared(episodes, enc, dec, lambda z, a: dyn_v4(z, a)),
        '#00a050', '-', 2.8,   # bright green, thicker
    )

    print("[4] SINDYc...")
    all_results['SINDYc'] = (
        eval_sindyc(episodes, enc, dec),
        '#aa00cc', '-', 2.0,
    )

    print("[5] GOKU-net...")
    dyn_g = DynamicsGOKU().to(DEVICE)
    dyn_g.load_state_dict(load_checkpoint("checkpoints/shared_goku/best.tar")['model'])
    dyn_g.eval()
    all_results['GOKU-net'] = (
        eval_shared(episodes, enc, dec, lambda z, a: dyn_g(z, a)),
        '#6bba55', '--', 2.0,
    )

    print("[6] DVBF...")
    dyn_d = DynamicsDVBF().to(DEVICE)
    dyn_d.load_state_dict(load_checkpoint("checkpoints/shared_dvbf/best.tar")['model'])
    dyn_d.eval()
    all_results['DVBF'] = (
        eval_shared(episodes, enc, dec, lambda z, a: dyn_d(z, a)),
        '#dc2828', '-', 2.0,
    )

    print("[7] Vid2Param (fair)...")
    dyn_v = DynamicsVid2Param().to(DEVICE)
    dyn_v.load_state_dict(load_checkpoint("checkpoints/shared_v2p/best.tar")['model'])
    dyn_v.eval()
    all_results['Vid2Param'] = (
        eval_shared(episodes, enc, dec,
                    dyn_fn=lambda z, a, t: dyn_v.step(z, a, t),
                    needs_theta=True, theta_fn=dyn_v.infer_theta),
        '#ff8c00', '-', 2.0,
    )

    plot_final(all_results, "vis/final_state_mse.png")

    # Numeric summary
    print(f"\n{'Step':>5s} " + " ".join([f"{n:>16s}" for n in all_results.keys()]))
    for dim_name, dim_idx in [("x_rel", 0), ("y_rel", 1), ("yaw_rel", 2)]:
        print(f"\n--- {dim_name} ---")
        for k in [0, 10, 20, 50, 75, 100]:
            row = f"{k:>5d} "
            for name, (sm, _, _, _) in all_results.items():
                if k < len(sm) and sm[k]:
                    val = np.array(sm[k])[:, dim_idx].mean()
                    row += f" {val:>16.4f}"
                else:
                    row += f" {'-':>16s}"
            print(row)
