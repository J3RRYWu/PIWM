"""Diagnostic: where exactly does PIWM lose to GOKU on donkey?

Decomposes the per-dim, per-step error so we can attribute the car_MSE gap to
specific physics dimensions (yaw, vx, vy, omega, etc.). Also reports per-dim
single-step error vs 8-step accumulated error to separate model-fit from
stability issues.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "train"))

import donkey_config; donkey_config.patch_globals()

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import DEVICE, DATA_DIR
from models.dynamics_lane_v6_kin import LaneAugmentedV6Kin
from models.dynamics_lane_v6_kin2 import LaneAugmentedV6Kin2
from models.dynamics_lane_donkey import LaneAugmentedDonkey
from models.dynamics_lane_donkey_v2 import LaneAugmentedDonkeyV2
from models.dynamics_lane_donkey_v3 import LaneAugmentedDonkeyV3
from baselines.shared_dynamics_lane import DynamicsGOKULane, DynamicsDVBFLane
from donkey_dataset import split_segments
from train_piwm_lane_v5 import SeqLaneDataset
from utils import load_checkpoint


ROLLOUT_K = 8
CAR_DIMS = ["x", "y", "yaw", "vx", "vy", "ω", "w0", "w1", "w2", "w3", "δ"]


def build_val_loader():
    base = SeqLaneDataset(DATA_DIR, seq_len=ROLLOUT_K + 1)
    _, val_eps = split_segments(base, val_frac=0.10, seed=0)
    val_idx = [i for i, (ep, _) in enumerate(base.indices) if ep in val_eps]
    return DataLoader(Subset(base, val_idx), batch_size=128, shuffle=False), base


@torch.no_grad()
def per_dim_errors(step_fn, val_loader):
    """Return (K, 11) per-step per-car-dim MSE and (K, 20) per-lane-dim MSE."""
    car_err  = torch.zeros(ROLLOUT_K, 11)
    lane_err = torch.zeros(ROLLOUT_K, 20)
    nb = 0
    for stack0, fi, fp, fa, wn in val_loader:
        stack0 = stack0.to(DEVICE); fp = fp.to(DEVICE); fa = fa.to(DEVICE); wn = wn.to(DEVICE)
        z = torch.cat([fp[:, 0], wn[:, 0]], dim=-1)
        for k in range(ROLLOUT_K):
            z = step_fn(z, fa[:, k], stack0)
            gt = torch.cat([fp[:, k + 1], wn[:, k + 1]], dim=-1)
            err = ((z - gt) ** 2).mean(dim=0)
            car_err[k]  += err[:11].cpu()
            lane_err[k] += err[11:].cpu()
        nb += 1
    return (car_err / nb).numpy(), (lane_err / nb).numpy()


def load_model(kind, path):
    if kind == "kin":
        m = LaneAugmentedV6Kin().to(DEVICE)
    elif kind == "kin2":
        m = LaneAugmentedV6Kin2().to(DEVICE)
    elif kind == "donkey":
        m = LaneAugmentedDonkey().to(DEVICE)
    elif kind == "donkey2":
        m = LaneAugmentedDonkeyV2().to(DEVICE)
    elif kind == "donkey3":
        m = LaneAugmentedDonkeyV3().to(DEVICE)
    elif kind == "goku":
        m = DynamicsGOKULane().to(DEVICE)
    elif kind == "dvbf":
        m = DynamicsDVBFLane().to(DEVICE)
    else:
        raise ValueError(kind)
    ck = load_checkpoint(path)
    state = ck.get('dynamics', ck.get('model'))
    m.load_state_dict(state)
    m.eval()
    return m


def step_lane_v6(model):
    def fn(z, a, _img):
        out, _ = model(z, a); return out
    return fn


def step_baseline(model):
    def fn(z, a, _img):
        return model(z, a)
    return fn


def main():
    val_loader, base = build_val_loader()

    runs = []
    for tag, kind, path in [
        ("PIWM-kin (s2)", "kin",    "checkpoints/piwm_lane_v6_donkey/dyn.tar"),
        ("PIWM-kin2",     "kin2",   "checkpoints/piwm_lane_v6_kin2_donkey/dyn.tar"),
        ("PIWM-donkey",   "donkey",  "checkpoints/piwm_lane_donkey/dyn.tar"),
        ("PIWM-donkey-v2","donkey2", "checkpoints/piwm_lane_donkey_v2/dyn.tar"),
        ("PIWM-donkey-v3","donkey3", "checkpoints/piwm_lane_donkey_v3/dyn.tar"),
        ("GOKU",          "goku",   "checkpoints/goku_lane_donkey/best.tar"),
        ("DVBF",          "dvbf",   "checkpoints/dvbf_lane_donkey/best.tar"),
    ]:
        if not os.path.exists(path):
            print(f"skip {tag}: {path} missing")
            continue
        m = load_model(kind, path)
        step = step_lane_v6(m) if kind in ("kin", "kin2", "donkey", "donkey2", "donkey3") else step_baseline(m)
        car_err, lane_err = per_dim_errors(step, val_loader)
        runs.append((tag, car_err, lane_err, m))

    # =================================================================
    # Table 1: single-step (k=1) car MSE per dim
    # =================================================================
    print("\n" + "=" * 88)
    print("Single-step car MSE by dim  (k=1)  — model-fit quality")
    print("=" * 88)
    print(f"  {'model':<16s}  " + "  ".join(f"{n:>6s}" for n in CAR_DIMS))
    for tag, car_err, _, _ in runs:
        row = "  ".join(f"{car_err[0, d]:6.4f}" for d in range(11))
        print(f"  {tag:<16s}  {row}")

    # =================================================================
    # Table 2: 8-step mean car MSE per dim
    # =================================================================
    print("\n" + "=" * 88)
    print("8-step mean car MSE by dim — model + accumulation")
    print("=" * 88)
    print(f"  {'model':<16s}  " + "  ".join(f"{n:>6s}" for n in CAR_DIMS))
    for tag, car_err, _, _ in runs:
        m = car_err.mean(axis=0)
        row = "  ".join(f"{m[d]:6.4f}" for d in range(11))
        print(f"  {tag:<16s}  {row}")

    # =================================================================
    # Table 3: error accumulation factor (k=8 / k=1) per dim
    # =================================================================
    print("\n" + "=" * 88)
    print("Error growth factor (k=8 / k=1) per dim — pure accumulation")
    print("=" * 88)
    print(f"  {'model':<16s}  " + "  ".join(f"{n:>6s}" for n in CAR_DIMS))
    for tag, car_err, _, _ in runs:
        ratio = car_err[-1] / np.maximum(car_err[0], 1e-9)
        row = "  ".join(f"{ratio[d]:6.2f}" for d in range(11))
        print(f"  {tag:<16s}  {row}")

    # =================================================================
    # Per-step time-series for the worst dims
    # =================================================================
    print("\n" + "=" * 88)
    print("Per-step car MSE for the 5 worst dims (PIWM-kin2 minus GOKU)")
    print("=" * 88)
    if any(t[0].startswith("PIWM-kin2") for t in runs) and any(t[0].startswith("GOKU") for t in runs):
        pk2 = [r for r in runs if r[0] == "PIWM-kin2"][0]
        gk  = [r for r in runs if r[0] == "GOKU"     ][0]
        gap = pk2[1].mean(axis=0) - gk[1].mean(axis=0)
        worst = np.argsort(gap)[::-1][:5]
        for d in worst:
            print(f"\n  dim={CAR_DIMS[d]:<6s}  gap(8-step mean) = {gap[d]:+.4f}")
            print(f"    {'k':<3s}" + "".join(f"  k={k+1}  " for k in range(ROLLOUT_K)))
            for tag, car_err, _, _ in runs:
                row = "".join(f" {car_err[k, d]:6.4f}" for k in range(ROLLOUT_K))
                print(f"    {tag:<16s}{row}")

    # =================================================================
    # Hyperparameters learned by PIWM models
    # =================================================================
    print("\n" + "=" * 88)
    print("Learned physical parameters")
    print("=" * 88)
    for tag, _, _, m in runs:
        if hasattr(m, "car_dyn") and hasattr(m.car_dyn, "L"):
            L_work = m.car_dyn.L.item()
            L_phys = L_work / 50.0   # SPATIAL_SCALE
            print(f"  {tag:<16s}  L = {L_work:.3f} working units  ({L_phys:.4f} m physical, true=0.165m)")


if __name__ == "__main__":
    main()
