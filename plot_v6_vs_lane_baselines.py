"""V6-only vs lane baselines (DVBF-lane / SINDYc-lane / GOKU-lane / V2P-lane).

Produces: vis/v6_vs_lane_baselines.png  (4 panels: x_rel, y_rel, yaw_rel, image MSE)
          vis/v6_vs_lane_baselines_laneMSE.png  (lane waypoint MSE)
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from compare_all_with_v5 import (
    load_episodes, eval_v5, eval_lane_baseline, eval_sindyc_lane,
    DEVICE, COLORS, LW, LS,
)
from models.encoder_lane import PhysicsEncoderLane
from models.decoder_lane import PhysicsDecoderLane
from models.dynamics_lane_v6 import LaneAugmentedV6
from baselines.shared_dynamics_lane import DynamicsDVBFLane, DynamicsGOKULane, DynamicsVid2ParamLane
from utils import load_checkpoint


def _series(container, idx=None):
    """idx: int for single dim, list/tuple to sum MSE across those dims, None to mean-all."""
    steps, means = [], []
    for k in range(len(container)):
        if len(container[k]) < 3: continue
        arr = np.array(container[k])
        if idx is None:
            m = arr.mean()
        elif isinstance(idx, (list, tuple)):
            m = arr[:, list(idx)].sum(axis=-1).mean()
        else:
            m = arr[:, idx].mean()
        if np.isfinite(m):
            steps.append(k); means.append(m)
    return steps, means


def plot_state_image(state_results, image_results, save, title):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = [('Position MSE (x²+y²)', [0, 1]), ('yaw_rel MSE', 2)]
    for ax, (tl, idx) in zip(axes[:2], panels):
        for name, sm in state_results.items():
            s, m = _series(sm, idx=idx)
            ax.plot(s, m, color=COLORS.get(name, 'gray'),
                    linewidth=LW.get(name, 2.0),
                    linestyle=LS.get(name, '-'), label=name)
        ax.set_xlabel('Step'); ax.set_ylabel(tl); ax.set_title(tl)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
        vals = [m for sm in state_results.values() for m in _series(sm, idx=idx)[1]]
        if vals: ax.set_ylim(0, np.percentile(vals, 95) * 1.3)
    ax = axes[2]
    for name, img in image_results.items():
        s, m = _series(img)
        ax.plot(s, m, color=COLORS.get(name, 'gray'),
                linewidth=LW.get(name, 2.0),
                linestyle=LS.get(name, '-'), label=name)
    ax.set_xlabel('Step'); ax.set_ylabel('Image MSE'); ax.set_title('Image MSE')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9); ax.set_ylim(0, 0.03)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(); fig.savefig(save, dpi=150); plt.close(fig)
    print(f"Saved {save}")


def plot_lane(lane_results, save, title):
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, lm in lane_results.items():
        s, m = _series(lm)
        ax.plot(s, m, color=COLORS.get(name, 'gray'),
                linewidth=LW.get(name, 2.2),
                linestyle=LS.get(name, '-'), label=name)
    ax.set_xlabel('Step'); ax.set_ylabel('Lane waypoint MSE (normalized)')
    ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
    vals = [m for lm in lane_results.values() for m in _series(lm)[1]]
    if vals: ax.set_ylim(0, np.percentile(vals, 95) * 1.3)
    fig.tight_layout(); fig.savefig(save, dpi=150); plt.close(fig)
    print(f"Saved {save}")


if __name__ == "__main__":
    os.makedirs("vis", exist_ok=True)
    eps = load_episodes()

    # v5 encoder/decoder is shared by all 31-dim lane baselines for init
    v5_ae = load_checkpoint("checkpoints/piwm_lane_v5/best.tar")
    enc5 = PhysicsEncoderLane().to(DEVICE); enc5.load_state_dict(v5_ae['encoder']); enc5.eval()
    dec5 = PhysicsDecoderLane().to(DEVICE); dec5.load_state_dict(v5_ae['decoder']); dec5.eval()

    # v6 uses its own encoder/decoder
    v6_ck = load_checkpoint("checkpoints/piwm_lane_v6/best.tar")
    enc6 = PhysicsEncoderLane().to(DEVICE); enc6.load_state_dict(v6_ck['encoder']); enc6.eval()
    dec6 = PhysicsDecoderLane().to(DEVICE); dec6.load_state_dict(v6_ck['decoder']); dec6.eval()

    state_results, image_results, lane_results = {}, {}, {}

    print("[1] PIWM-lane-v6")
    dyn6 = LaneAugmentedV6().to(DEVICE); dyn6.load_state_dict(v6_ck['dynamics']); dyn6.eval()
    img, sm, lm = eval_v5(eps, enc6, dec6, dyn6)
    state_results['PIWM-lane-v6'] = sm; image_results['PIWM-lane-v6'] = img
    lane_results['PIWM-lane-v6'] = lm

    print("[2] SINDYc-lane")
    img, sm, lm = eval_sindyc_lane(eps, enc5, dec5)
    state_results['SINDYc-lane'] = sm; image_results['SINDYc-lane'] = img
    lane_results['SINDYc-lane'] = lm

    for name, cls, path, needs_theta in [
        ('DVBF-lane', DynamicsDVBFLane, "checkpoints/shared_dvbf_lane/best.tar", False),
        ('GOKU-lane', DynamicsGOKULane, "checkpoints/shared_goku_lane/best.tar", False),
        ('V2P-lane',  DynamicsVid2ParamLane, "checkpoints/shared_v2p_lane/best.tar", True),
    ]:
        print(f"[*] {name}")
        b = cls().to(DEVICE); b.load_state_dict(load_checkpoint(path)['model']); b.eval()
        theta_fn = b.infer_theta if needs_theta else None
        img, sm, lm = eval_lane_baseline(eps, enc5, dec5, b,
                                          needs_theta=needs_theta, theta_fn=theta_fn)
        state_results[name] = sm; image_results[name] = img; lane_results[name] = lm

    plot_state_image(state_results, image_results,
                     "vis/v6_vs_lane_baselines.png",
                     "PIWM-lane-v6 vs lane baselines")
    plot_lane(lane_results,
              "vis/v6_vs_lane_baselines_laneMSE.png",
              "Lane waypoint MSE: v6 vs lane baselines")

    # Numeric table
    names = list(state_results.keys())
    print("\n" + " " * 6 + " ".join(f"{n:>14s}" for n in names))
    dims = [("position", [0, 1]), ("yaw_rel", 2)]
    for dim_name, idx in dims:
        print(f"\n--- {dim_name} ---")
        for k in [10, 20, 50, 75, 100]:
            row = f"{k:>5d} "
            for n in names:
                sm = state_results[n]
                if k < len(sm) and sm[k]:
                    arr = np.array(sm[k])
                    v = arr[:, list(idx)].sum(axis=-1).mean() if isinstance(idx, list) else arr[:, idx].mean()
                else:
                    v = float('nan')
                row += f" {v:>14.4f}"
            print(row)
    print("\n--- Image MSE ---")
    for k in [10, 20, 50, 75, 100]:
        row = f"{k:>5d} "
        for n in names:
            img = image_results[n]
            v = np.mean(img[k]) if k < len(img) and img[k] else float('nan')
            row += f" {v:>14.5f}"
        print(row)
    print("\n--- Lane MSE ---")
    for k in [10, 20, 50, 75, 100]:
        row = f"{k:>5d} "
        for n in names:
            lm = lane_results[n]
            v = np.array(lm[k]).mean() if k < len(lm) and lm[k] else float('nan')
            row += f" {v:>14.4f}"
        print(row)
