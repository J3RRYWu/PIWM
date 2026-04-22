"""Lane-only comparison: position MSE (x+y combined) + yaw + image + lane.

Only lane-augmented methods:
  PIWM-lane-v5, PIWM-lane-v6, DVBF-lane, GOKU-lane, V2P-lane

Position MSE = x_rel MSE + y_rel MSE  (sum of squared errors in x and y).
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from compare_all_with_v5 import (
    load_episodes, eval_v5, eval_lane_baseline,
    MAX_STEPS, DEVICE, COLORS, LW, LS,
)
from models.encoder_lane import PhysicsEncoderLane
from models.decoder_lane import PhysicsDecoderLane
from models.dynamics_lane_v5 import LaneAugmentedV5
from models.dynamics_lane_v6 import LaneAugmentedV6
from baselines.shared_dynamics_lane import (
    DynamicsDVBFLane, DynamicsGOKULane, DynamicsVid2ParamLane,
)
from utils import load_checkpoint


def _curve(results, save, ylabel, title, key='state', idx=None, ylim_top=None):
    """Generic curve plot: results name -> list-of-list of values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, series in results.items():
        steps, means = [], []
        for k in range(len(series)):
            if len(series[k]) >= 3:
                arr = np.array(series[k])
                if idx is not None:
                    means.append(arr[:, idx].mean())
                elif arr.ndim == 2:
                    # state MSE sum over specified dims
                    means.append(arr.mean() if key != 'pos' else (arr[:, 0] + arr[:, 1]).mean())
                else:
                    means.append(arr.mean())
                steps.append(k)
        ax.plot(steps, means,
                color=COLORS.get(name, 'gray'),
                linewidth=LW.get(name, 2.2),
                linestyle=LS.get(name, '-'), label=name)
    ax.set_xlabel('Step'); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend(fontsize=11)
    if ylim_top is not None:
        ax.set_ylim(0, ylim_top)
    else:
        # Auto via 95th percentile of means
        vals = []
        for series in results.values():
            for k in range(len(series)):
                if len(series[k]) >= 3:
                    arr = np.array(series[k])
                    if idx is not None:
                        vals.append(arr[:, idx].mean())
                    elif arr.ndim == 2:
                        vals.append((arr[:, 0] + arr[:, 1]).mean() if key == 'pos' else arr.mean())
                    else:
                        vals.append(arr.mean())
        if vals:
            ax.set_ylim(0, np.percentile(vals, 95) * 1.3)
    fig.tight_layout(); fig.savefig(save, dpi=150); plt.close(fig)
    print(f"Saved {save}")


if __name__ == "__main__":
    os.makedirs("vis", exist_ok=True)
    eps = load_episodes()

    # Shared encoder/decoder from v5 ae.tar (all lane methods use this)
    v5_ae = load_checkpoint("checkpoints/piwm_lane_v5/best.tar")
    enc5 = PhysicsEncoderLane().to(DEVICE); enc5.load_state_dict(v5_ae['encoder']); enc5.eval()
    dec5 = PhysicsDecoderLane().to(DEVICE); dec5.load_state_dict(v5_ae['decoder']); dec5.eval()

    v6_ck = load_checkpoint("checkpoints/piwm_lane_v6/best.tar")
    enc6 = PhysicsEncoderLane().to(DEVICE); enc6.load_state_dict(v6_ck['encoder']); enc6.eval()
    dec6 = PhysicsDecoderLane().to(DEVICE); dec6.load_state_dict(v6_ck['decoder']); dec6.eval()

    state_results, image_results, lane_results = {}, {}, {}

    print("[1] v5")
    dyn5 = LaneAugmentedV5().to(DEVICE)
    dyn5.load_state_dict(v5_ae['dynamics']); dyn5.eval()
    img, sm, lm = eval_v5(eps, enc5, dec5, dyn5)
    state_results['PIWM-lane-v5'] = sm; image_results['PIWM-lane-v5'] = img
    lane_results['PIWM-lane-v5'] = lm

    print("[2] v6")
    dyn6 = LaneAugmentedV6().to(DEVICE)
    dyn6.load_state_dict(v6_ck['dynamics']); dyn6.eval()
    img, sm, lm = eval_v5(eps, enc6, dec6, dyn6)
    state_results['PIWM-lane-v6'] = sm; image_results['PIWM-lane-v6'] = img
    lane_results['PIWM-lane-v6'] = lm

    for name, cls, ckpt_path, needs_theta in [
        ('DVBF-lane', DynamicsDVBFLane, "checkpoints/shared_dvbf_lane/best.tar", False),
        ('GOKU-lane', DynamicsGOKULane, "checkpoints/shared_goku_lane/best.tar", False),
        ('V2P-lane',  DynamicsVid2ParamLane, "checkpoints/shared_v2p_lane/best.tar", True),
    ]:
        print(f"[*] {name}")
        b = cls().to(DEVICE)
        b.load_state_dict(load_checkpoint(ckpt_path)['model']); b.eval()
        theta_fn = b.infer_theta if needs_theta else None
        img, sm, lm = eval_lane_baseline(eps, enc5, dec5, b,
                                          needs_theta=needs_theta, theta_fn=theta_fn)
        state_results[name] = sm; image_results[name] = img
        lane_results[name] = lm

    # ----- 4-panel figure -----
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # Panel 1: Position MSE (x + y)
    ax = axes[0]
    for name, sm in state_results.items():
        steps, means = [], []
        for k in range(len(sm)):
            if len(sm[k]) >= 3:
                arr = np.array(sm[k])
                means.append((arr[:, 0] + arr[:, 1]).mean())
                steps.append(k)
        ax.plot(steps, means, color=COLORS.get(name, 'gray'),
                linewidth=LW.get(name, 2.2),
                linestyle=LS.get(name, '-'), label=name)
    ax.set_xlabel('Prediction Horizon (steps)')
    ax.set_ylabel('Position MSE  ($x^2 + y^2$, normalized)')
    ax.set_title('Position MSE')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
    vals = []
    for sm in state_results.values():
        for k in range(len(sm)):
            if len(sm[k]) >= 3:
                arr = np.array(sm[k])
                vals.append((arr[:, 0] + arr[:, 1]).mean())
    if vals: ax.set_ylim(0, np.percentile(vals, 95) * 1.3)

    # Panel 2: yaw MSE
    ax = axes[1]
    for name, sm in state_results.items():
        steps, means = [], []
        for k in range(len(sm)):
            if len(sm[k]) >= 3:
                means.append(np.array(sm[k])[:, 2].mean()); steps.append(k)
        ax.plot(steps, means, color=COLORS.get(name, 'gray'),
                linewidth=LW.get(name, 2.2),
                linestyle=LS.get(name, '-'), label=name)
    ax.set_xlabel('Step'); ax.set_ylabel('yaw_rel MSE')
    ax.set_title('Yaw MSE')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
    vals = []
    for sm in state_results.values():
        for k in range(len(sm)):
            if len(sm[k]) >= 3:
                vals.append(np.array(sm[k])[:, 2].mean())
    if vals: ax.set_ylim(0, np.percentile(vals, 95) * 1.3)

    # Panel 3: Image MSE
    ax = axes[2]
    for name, img in image_results.items():
        steps, means = [], []
        for k in range(len(img)):
            if len(img[k]) >= 3:
                means.append(np.mean(img[k])); steps.append(k)
        ax.plot(steps, means, color=COLORS.get(name, 'gray'),
                linewidth=LW.get(name, 2.2),
                linestyle=LS.get(name, '-'), label=name)
    ax.set_xlabel('Step'); ax.set_ylabel('Image MSE (per pixel)')
    ax.set_title('Image MSE')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
    ax.set_ylim(0, 0.03)

    # Panel 4: Lane MSE
    ax = axes[3]
    for name, lm in lane_results.items():
        if lm is None: continue
        steps, means = [], []
        for k in range(len(lm)):
            if len(lm[k]) >= 3:
                means.append(np.array(lm[k]).mean()); steps.append(k)
        ax.plot(steps, means, color=COLORS.get(name, 'gray'),
                linewidth=LW.get(name, 2.2),
                linestyle=LS.get(name, '-'), label=name)
    ax.set_xlabel('Step'); ax.set_ylabel('Lane Waypoint MSE (normalized)')
    ax.set_title('Lane Waypoint MSE')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=10)

    fig.suptitle('Lane-augmented methods: v5 / v6 / DVBF-lane / GOKU-lane / V2P-lane',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig("vis/lane_only_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved vis/lane_only_comparison.png")

    # Numeric summary
    print(f"\n{'Step':>5s} " + " ".join([f"{n:>14s}" for n in state_results.keys()]))
    print("--- Position MSE (x^2 + y^2, normalized) ---")
    for k in [0, 10, 20, 50, 75, 100]:
        row = f"{k:>5d} "
        for name, sm in state_results.items():
            if k < len(sm) and sm[k]:
                arr = np.array(sm[k]); v = (arr[:, 0] + arr[:, 1]).mean()
                row += f" {v:>14.4f}"
            else:
                row += f" {'-':>14s}"
        print(row)
    print("--- Yaw MSE ---")
    for k in [0, 10, 20, 50, 75, 100]:
        row = f"{k:>5d} "
        for name, sm in state_results.items():
            if k < len(sm) and sm[k]:
                v = np.array(sm[k])[:, 2].mean(); row += f" {v:>14.4f}"
            else:
                row += f" {'-':>14s}"
        print(row)
    print("--- Image MSE ---")
    for k in [0, 10, 20, 50, 75, 100]:
        row = f"{k:>5d} "
        for name, img in image_results.items():
            if k < len(img) and img[k]:
                row += f" {np.mean(img[k]):>14.5f}"
            else:
                row += f" {'-':>14s}"
        print(row)
    print("--- Lane MSE ---")
    for k in [0, 10, 20, 50, 75, 100]:
        row = f"{k:>5d} "
        for name, lm in lane_results.items():
            if lm is None: row += f" {'-':>14s}"; continue
            if k < len(lm) and lm[k]:
                row += f" {np.array(lm[k]).mean():>14.4f}"
            else:
                row += f" {'-':>14s}"
        print(row)
