"""Visualize PIWM vs baselines on donkey val data.

Generates four figures:
  1. fig_bars.png   — composite/car/lane MSE bar chart
  2. fig_per_step.png — per-step car MSE growth curves for all models
  3. fig_rollout_trajectory.png — bird's-eye view of GT vs predicted car
     trajectories over an 8-step rollout, on the actual donkey map
  4. fig_lane_overlay.png — GT vs predicted lane waypoints at t=0 and t=K
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "train"))

import donkey_config; donkey_config.patch_globals()

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from config import DEVICE, DATA_DIR, PHYSICS_MEAN_REL, PHYSICS_STD_REL
from lane_utils import LANE_MEAN, LANE_STD
from donkey_config import DONKEY_SPATIAL_SCALE
from models.dynamics_lane_v6_kin   import LaneAugmentedV6Kin
from models.dynamics_lane_v6_kin2  import LaneAugmentedV6Kin2
from models.dynamics_lane_donkey   import LaneAugmentedDonkey
from models.dynamics_lane_donkey_v2 import LaneAugmentedDonkeyV2
from models.dynamics_lane_donkey_v3 import LaneAugmentedDonkeyV3
from baselines.shared_dynamics_lane import (DynamicsDVBFLane, DynamicsGOKULane,
                                            DynamicsVid2ParamLane)
from models.encoder_lane import PhysicsEncoderLane
from utils import load_checkpoint
from donkey_dataset import split_segments
from train_piwm_lane_v5 import SeqLaneDataset
from donkey_track import DonkeyCenterline


ROLLOUT_K = 8
SEED = 0
SAVE_DIR = "figures"
_os.makedirs(SAVE_DIR, exist_ok=True)


# --- helpers ---
def load_model(kind, path):
    if   kind == "kin":     m = LaneAugmentedV6Kin()
    elif kind == "kin2":    m = LaneAugmentedV6Kin2()
    elif kind == "donkey":  m = LaneAugmentedDonkey()
    elif kind == "donkey2": m = LaneAugmentedDonkeyV2()
    elif kind == "donkey3": m = LaneAugmentedDonkeyV3()
    elif kind == "goku":    m = DynamicsGOKULane()
    elif kind == "dvbf":    m = DynamicsDVBFLane()
    elif kind == "v2p":     m = DynamicsVid2ParamLane()
    else: raise ValueError(kind)
    state = load_checkpoint(path)
    sd = state.get("dynamics", state.get("model"))
    m.load_state_dict(sd)
    m.to(DEVICE).eval()
    return m


def step_model(m, kind):
    is_piwm = kind in ("kin", "kin2", "donkey", "donkey2", "donkey3")
    def fn(z, a):
        out = m(z, a)
        return out[0] if isinstance(out, tuple) else out
    return fn


MODELS = [
    ("GOKU",          "goku",    "checkpoints/goku_lane_donkey/best.tar",          "tab:green"),
    ("DVBF",          "dvbf",    "checkpoints/dvbf_lane_donkey/best.tar",          "tab:orange"),
    ("V2P",           "v2p",     "checkpoints/v2p_lane_donkey/best.tar",           "tab:red"),
    ("PIWM-kin",      "kin",     "checkpoints/piwm_lane_v6_donkey/dyn.tar",        "tab:purple"),
    ("PIWM-kin2",     "kin2",    "checkpoints/piwm_lane_v6_kin2_donkey/dyn.tar",   "tab:brown"),
    ("PIWM-v3",       "donkey3", "checkpoints/piwm_lane_donkey_v3/dyn.tar",        "tab:blue"),
]


def build_val_loader():
    base = SeqLaneDataset(DATA_DIR, seq_len=ROLLOUT_K + 1)
    _, val_eps = split_segments(base, val_frac=0.10, seed=SEED)
    val_idx = [i for i, (ep, _) in enumerate(base.indices) if ep in val_eps]
    return DataLoader(Subset(base, val_idx), batch_size=128, shuffle=False), base, val_eps


# --- per-step error ---
@torch.no_grad()
def per_step_errors(step_fn, loader, theta_provider=None):
    car  = np.zeros(ROLLOUT_K)
    lane = np.zeros(ROLLOUT_K)
    nb = 0
    for s0, fi, fp, fa, wn in loader:
        s0 = s0.to(DEVICE); fp = fp.to(DEVICE); fa = fa.to(DEVICE); wn = wn.to(DEVICE)
        if theta_provider is not None:
            theta = theta_provider(s0)
        z = torch.cat([fp[:, 0], wn[:, 0]], dim=-1)
        for k in range(ROLLOUT_K):
            if theta_provider is not None:
                z = step_fn(z, fa[:, k], theta)
            else:
                z = step_fn(z, fa[:, k])
            gt = torch.cat([fp[:, k + 1], wn[:, k + 1]], dim=-1)
            car[k]  += ((z[:, :11] - gt[:, :11])  ** 2).mean().item()
            lane[k] += ((z[:, 11:] - gt[:, 11:]) ** 2).mean().item()
        nb += 1
    return car / nb, lane / nb


def main():
    val_loader, base, val_eps = build_val_loader()

    # --- v2p needs encoder + theta provider ---
    enc = PhysicsEncoderLane().to(DEVICE)
    ENC_CK = "checkpoints/piwm_lane_v6_donkey/ae.tar"
    enc.load_state_dict(load_checkpoint(ENC_CK)["encoder"]); enc.eval()
    for p in enc.parameters(): p.requires_grad = False

    print("Evaluating per-step errors...")
    results = {}
    for tag, kind, path, color in MODELS:
        if not _os.path.exists(path):
            print(f"  skip {tag}: missing {path}")
            continue
        m = load_model(kind, path)
        if kind == "v2p":
            def _theta(stack0, _m=m):
                with torch.no_grad(): obs = enc(stack0).unsqueeze(1)
                t, _, _ = _m.infer_theta(obs); return t
            step = lambda z, a, theta: m.step(z, a, theta)
            car, lane = per_step_errors(step, val_loader, theta_provider=_theta)
        else:
            step = step_model(m, kind)
            car, lane = per_step_errors(step, val_loader)
        comp = car.mean() + 2.0 * lane.mean()
        results[tag] = dict(car=car, lane=lane, color=color, composite=comp,
                            car_mean=car.mean(), lane_mean=lane.mean(),
                            model=m, kind=kind)
        print(f"  {tag}: car={car.mean():.4f}  lane={lane.mean():.4f}  comp={comp:.4f}")

    # =============================================================
    # Fig 1: bar chart of composite, car, lane
    # =============================================================
    order = sorted(results.keys(), key=lambda k: results[k]["composite"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    x = np.arange(len(order))
    cars  = [results[k]["car_mean"]  for k in order]
    lanes = [results[k]["lane_mean"] for k in order]
    comps = [results[k]["composite"] for k in order]
    colors = [results[k]["color"]    for k in order]
    for axi, vals, ttl, ylab in zip(
            ax, [cars, lanes, comps],
            ["car MSE (k=1..8 mean)", "lane MSE (k=1..8 mean)",
             "composite = car + 2 · lane"],
            ["MSE (norm.)"] * 3):
        bars = axi.bar(x, vals, color=colors)
        axi.set_xticks(x); axi.set_xticklabels(order, rotation=30, ha="right")
        axi.set_title(ttl); axi.set_ylabel(ylab); axi.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            axi.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=8)
    fig.suptitle("Donkey val (segment holdout): 8-step rollout MSE per model",
                 fontweight="bold")
    fig.savefig(_os.path.join(SAVE_DIR, "fig_bars.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # =============================================================
    # Fig 2: per-step error growth curves
    # =============================================================
    fig, ax = plt.subplots(1, 2, figsize=(18, 6.5), constrained_layout=True)
    steps = np.arange(1, ROLLOUT_K + 1)
    for tag in order:
        r = results[tag]
        ax[0].plot(steps, r["car"],  marker="o", color=r["color"],
                   label=f"{tag}  (mean={r['car_mean']:.3f})", lw=2.5, markersize=8)
        ax[1].plot(steps, r["lane"], marker="o", color=r["color"],
                   label=f"{tag}  (mean={r['lane_mean']:.3f})", lw=2.5, markersize=8)
    for axi, ttl, ylab in zip(ax,
                              ["Car  MSE  per rollout step",
                               "Lane MSE  per rollout step"],
                              ["MSE (norm.)"] * 2):
        axi.set_xlabel("rollout step k", fontsize=12)
        axi.set_ylabel(ylab, fontsize=12)
        axi.set_title(ttl, fontsize=13)
        axi.grid(alpha=0.3); axi.legend(fontsize=11, loc="upper left")
        axi.tick_params(labelsize=11)
    fig.suptitle("Error accumulation across 8-step rollout", fontweight="bold")
    fig.savefig(_os.path.join(SAVE_DIR, "fig_per_step.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # =============================================================
    # Fig 3: bird's-eye rollout on the donkey track for a few samples
    # =============================================================
    # We pick 6 validation windows, do GT rollout vs each model's prediction,
    # transform back to world coordinates, and plot on the centerline.
    print("Building world-frame trajectories for visualization...")

    # Build the centerline visualisation from the first segment's pos.
    raw_dir = _os.path.join(_os.path.dirname(__file__), "..", "Data_Donkeycar")
    import glob
    raw_files = sorted(glob.glob(_os.path.join(raw_dir, "*.npz")))
    raw0 = np.load(raw_files[0], allow_pickle=True)
    track = DonkeyCenterline(raw0["map_world"])
    centerline_world = track.centers * DONKEY_SPATIAL_SCALE   # to working units

    # Pick val sequences with relatively large motion (so the 8-step rollout
    # is visually meaningful — many donkey segments are slow/stationary).
    val_subset_idx = [i for i, (ep, _) in enumerate(base.indices) if ep in val_eps]
    motion_scores = []
    for i in val_subset_idx:
        ep, t0 = base.indices[i]
        seg = base.phys_list[ep][t0:t0 + ROLLOUT_K + 1, 0:2]
        motion_scores.append(np.linalg.norm(seg[-1] - seg[0]))
    motion_scores = np.array(motion_scores)
    # Take top 50% by motion, then sample 6 from those for variety.
    median = np.median(motion_scores)
    eligible = [val_subset_idx[i] for i in range(len(val_subset_idx))
                if motion_scores[i] >= median]
    rng = np.random.default_rng(SEED)
    pick = rng.choice(eligible, size=6, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    for ax_i, idx in enumerate(pick):
        ep, t0 = base.indices[idx]
        phys_seg = base.phys_list[ep][t0:t0 + ROLLOUT_K + 1]   # (K+1, 11)  raw working units
        ref_pos = phys_seg[0, 0:2]; ref_yaw = phys_seg[0, 2]
        ref_c = np.cos(ref_yaw); ref_s = np.sin(ref_yaw)

        # Run each model in normalized space, then convert predicted relative
        # (x,y) back to world for plotting.
        s0, fi, fp, fa, wn = base[idx]
        s0   = s0.unsqueeze(0).to(DEVICE)
        fp_b = fp.unsqueeze(0).to(DEVICE)
        fa_b = fa.unsqueeze(0).to(DEVICE)
        wn_b = wn.unsqueeze(0).to(DEVICE)

        # Pre-compute theta for V2P.
        with torch.no_grad():
            v2p_theta = None
            if "V2P" in results:
                v2p_theta_obs = enc(s0).unsqueeze(1)
                v2p_theta, _, _ = results["V2P"]["model"].infer_theta(v2p_theta_obs)

        ax = axes.flat[ax_i]
        # GT trajectory (relative xy in normalized space → unnorm → world rotate)
        gt_xy_norm = fp[:, 0:2].numpy()                       # (K+1, 2) norm rel
        gt_xy_rel  = gt_xy_norm * PHYSICS_STD_REL[0:2] + PHYSICS_MEAN_REL[0:2]
        gt_world = np.stack([
            ref_pos[0] + ref_c * gt_xy_rel[:, 0] - ref_s * gt_xy_rel[:, 1],
            ref_pos[1] + ref_s * gt_xy_rel[:, 0] + ref_c * gt_xy_rel[:, 1],
        ], axis=-1)

        # Plot centerline (we'll zoom in around the GT trajectory so the
        # short donkey rollout — ~2m at ×50 scale = ~100 working units — is
        # actually visible; full track is huge).
        ax.plot(centerline_world[:, 0], centerline_world[:, 1],
                color="lightgray", lw=1.0, label="lane centerline")
        ax.plot(gt_world[:, 0], gt_world[:, 1], "k-o", lw=2.5, markersize=7,
                label="GT", zorder=10)

        # Each model
        for tag in ["GOKU", "PIWM-v3", "PIWM-kin"]:
            if tag not in results: continue
            r = results[tag]
            m = r["model"]; kind = r["kind"]
            with torch.no_grad():
                z = torch.cat([fp_b[:, 0], wn_b[:, 0]], dim=-1)
                preds = [fp_b[:, 0, 0:2].cpu().numpy()[0]]   # rel-norm xy
                for k in range(ROLLOUT_K):
                    if kind == "v2p":
                        z = m.step(z, fa_b[:, k], v2p_theta)
                    else:
                        out = m(z, fa_b[:, k])
                        z = out[0] if isinstance(out, tuple) else out
                    preds.append(z[0, 0:2].cpu().numpy())
                preds = np.stack(preds, axis=0)              # (K+1, 2) norm rel
            preds_rel = preds * PHYSICS_STD_REL[0:2] + PHYSICS_MEAN_REL[0:2]
            preds_world = np.stack([
                ref_pos[0] + ref_c * preds_rel[:, 0] - ref_s * preds_rel[:, 1],
                ref_pos[1] + ref_s * preds_rel[:, 0] + ref_c * preds_rel[:, 1],
            ], axis=-1)
            ax.plot(preds_world[:, 0], preds_world[:, 1], "-x",
                    color=r["color"], lw=2.0, markersize=7, label=tag, alpha=0.85)

        # Zoom tightly on the trajectory so the per-step prediction differences
        # are visible. Pad proportional to motion size.
        all_x = gt_world[:, 0]
        all_y = gt_world[:, 1]
        span = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), 5.0)
        pad = span * 1.2
        cx, cy = (all_x.max() + all_x.min()) / 2, (all_y.max() + all_y.min()) / 2
        ax.set_xlim(cx - pad, cx + pad)
        ax.set_ylim(cy - pad, cy + pad)
        ax.set_xlabel("x  (1 working unit = 2 cm)")
        ax.set_ylabel("y")
        motion_m = span / DONKEY_SPATIAL_SCALE
        ax.set_title(f"val sample idx={idx}  (8-step motion ≈ {motion_m:.2f} m)",
                     fontsize=10)
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")

    fig.suptitle("8-step rollout — GT vs predicted car trajectories on donkey track",
                 fontweight="bold")
    fig.savefig(_os.path.join(SAVE_DIR, "fig_rollout_trajectory.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # =============================================================
    # Fig 4: lane waypoints at t=0 and t=K for one sample
    # =============================================================
    idx = pick[0]
    ep, t0 = base.indices[idx]
    s0, fi, fp, fa, wn = base[idx]
    s0_b = s0.unsqueeze(0).to(DEVICE)
    fp_b = fp.unsqueeze(0).to(DEVICE)
    fa_b = fa.unsqueeze(0).to(DEVICE)
    wn_b = wn.unsqueeze(0).to(DEVICE)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for axi, k_show, ttl in zip(axes, [1, ROLLOUT_K], ["k=1 (1 step ahead)",
                                                       f"k={ROLLOUT_K} (final step)"]):
        # GT lane wp at k_show, in normalized → unnormalized body frame
        gt_lane_n = wn[k_show].numpy()                       # (20,)
        gt_lane = gt_lane_n * LANE_STD + LANE_MEAN
        gt_lane = gt_lane.reshape(10, 2)
        axi.plot(gt_lane[:, 0], gt_lane[:, 1], "k-o",
                 lw=2.5, markersize=8, label="GT", zorder=10)

        for tag in ["GOKU", "PIWM-v3", "PIWM-kin"]:
            if tag not in results: continue
            r = results[tag]; m = r["model"]; kind = r["kind"]
            with torch.no_grad():
                z = torch.cat([fp_b[:, 0], wn_b[:, 0]], dim=-1)
                for k in range(k_show):
                    if kind == "v2p":
                        z = m.step(z, fa_b[:, k], v2p_theta)
                    else:
                        out = m(z, fa_b[:, k])
                        z = out[0] if isinstance(out, tuple) else out
                pred_lane_n = z[0, 11:].cpu().numpy()
            pred_lane = pred_lane_n * LANE_STD + LANE_MEAN
            pred_lane = pred_lane.reshape(10, 2)
            axi.plot(pred_lane[:, 0], pred_lane[:, 1], "--x",
                     color=r["color"], lw=1.8, markersize=8, label=tag, alpha=0.85)
        axi.axhline(0, color="gray", lw=0.5); axi.axvline(0, color="gray", lw=0.5)
        axi.scatter([0], [0], marker="^", s=120, color="red",
                    zorder=11, label="car @ that step")
        axi.set_xlabel("body-x (lateral, working units)")
        axi.set_ylabel("body-y (forward, working units)")
        axi.set_title(ttl); axi.set_aspect("equal"); axi.grid(alpha=0.3)
        axi.legend(fontsize=9, loc="best")
    fig.suptitle("Lane waypoint prediction (body-frame, val sample)", fontweight="bold")
    fig.savefig(_os.path.join(SAVE_DIR, "fig_lane_overlay.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved figures to: {SAVE_DIR}/")
    for f in ["fig_bars.png", "fig_per_step.png",
              "fig_rollout_trajectory.png", "fig_lane_overlay.png"]:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
