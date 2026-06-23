"""Fair side-by-side comparison of PIWM-v6-kin vs 4 baselines on donkey val.

All models are evaluated identically:
  - SAME val segments (segment-level split, seed=0)
  - SAME normalization (donkey patched constants)
  - SAME initial state z0 = GT phys || GT lane wp
  - SAME rollout horizon K = 8 steps
  - SAME metric:  car_RMSE in normalized space, lane_RMSE in normalized space
                  composite = car + 2.0 * lane  (loss used during training)

Run:
    python eval_compare_donkey.py
"""

# --- repo root + train/ on sys.path ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)
_sys.path.insert(0, _os.path.join(_HERE, "train"))
# --- end shim ---

import donkey_config
donkey_config.patch_globals()

import os, pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import DEVICE, DATA_DIR
from models.dynamics_lane_v6_kin import LaneAugmentedV6Kin
from models.dynamics_lane_v6_kin2 import LaneAugmentedV6Kin2
from models.dynamics_lane_donkey import LaneAugmentedDonkey
from models.dynamics_lane_donkey_v2 import LaneAugmentedDonkeyV2
from models.dynamics_lane_donkey_v3 import LaneAugmentedDonkeyV3
from models.dynamics_lane_donkey_v4 import LaneAugmentedDonkeyV4
from models.encoder_lane import PhysicsEncoderLane
from baselines.shared_dynamics_lane import (DynamicsDVBFLane, DynamicsGOKULane,
                                            DynamicsVid2ParamLane)
from utils import load_checkpoint
from donkey_dataset import (split_segments, collect_state31_windows)
from train_piwm_lane_v5 import SeqLaneDataset


ROLLOUT_K = 8
SEED = 0
LAMBDA_LANE = 2.0
ENCODER_CK = "checkpoints/piwm_lane_v6_donkey/ae.tar"


def build_val_loader(batch_size=128):
    """Same split logic as `make_donkey_loaders` but val-only, no flip."""
    base = SeqLaneDataset(DATA_DIR, seq_len=ROLLOUT_K + 1)
    train_eps, val_eps = split_segments(base, val_frac=0.10, seed=SEED)
    val_idx = [i for i, (ep, _) in enumerate(base.indices) if ep in val_eps]
    val_ds = Subset(base, val_idx)
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                      drop_last=False), base, val_eps


def _step_metrics(z, gt):
    """Compute per-batch car and lane MSEs (already in normalized space)."""
    car  = ((z[:, :11]  - gt[:, :11])  ** 2).mean().item()
    lane = ((z[:, 11:]  - gt[:, 11:]) ** 2).mean().item()
    return car, lane


# ----------------------------------------------------------------------
# Per-model evaluators
# ----------------------------------------------------------------------
@torch.no_grad()
def eval_nn_dynamics(model, val_loader, theta_provider=None):
    """Generic eval: rollout K steps from GT z0, accumulate car/lane MSE.

    `theta_provider` optionally returns (theta,) given (stack0,) for V2P.
    Returns: dict with `car_8step`, `lane_8step`, `composite_8step`,
             per-step car / lane RMSE arrays of length K.
    """
    model.eval()
    car_per_step  = np.zeros(ROLLOUT_K, dtype=np.float64)
    lane_per_step = np.zeros(ROLLOUT_K, dtype=np.float64)
    nb = 0
    for stack0, fi, fp, fa, wn in val_loader:
        stack0 = stack0.to(DEVICE); fp = fp.to(DEVICE)
        fa = fa.to(DEVICE); wn = wn.to(DEVICE)
        if theta_provider is not None:
            theta = theta_provider(stack0)
        z = torch.cat([fp[:, 0], wn[:, 0]], dim=-1)
        for k in range(ROLLOUT_K):
            if theta_provider is None:
                out = model(z, fa[:, k])
                z = out[0] if isinstance(out, tuple) else out
            else:
                z = model.step(z, fa[:, k], theta)
            gt = torch.cat([fp[:, k + 1], wn[:, k + 1]], dim=-1)
            c, l = _step_metrics(z, gt)
            car_per_step[k]  += c
            lane_per_step[k] += l
        nb += 1
    car_per_step  /= nb
    lane_per_step /= nb
    car_mean  = float(car_per_step.mean())
    lane_mean = float(lane_per_step.mean())
    return {
        "car_8step":  car_mean,
        "lane_8step": lane_mean,
        "composite":  car_mean + LAMBDA_LANE * lane_mean,
        "car_per_step":  car_per_step.tolist(),
        "lane_per_step": lane_per_step.tolist(),
    }


def eval_sindyc(model_path, base, val_eps):
    """SINDYc is numpy-based; eval by extracting val windows directly."""
    with open(model_path, "rb") as f:
        sindy = pickle.load(f)
    windows = collect_state31_windows(
        base, val_eps, window=ROLLOUT_K + 1, stride=2, with_flip=False)
    car_per_step  = np.zeros(ROLLOUT_K, dtype=np.float64)
    lane_per_step = np.zeros(ROLLOUT_K, dtype=np.float64)
    nb = 0
    for state31, action3 in windows:
        z = state31[0:1].copy()
        for k in range(ROLLOUT_K):
            u = action3[k:k + 1]
            z = np.asarray(sindy.predict(z, u=u), dtype=np.float32)
            err = (z.flatten() - state31[k + 1]) ** 2
            car_per_step[k]  += err[:11].mean()
            lane_per_step[k] += err[11:].mean()
        nb += 1
    car_per_step  /= nb
    lane_per_step /= nb
    car_mean  = float(car_per_step.mean())
    lane_mean = float(lane_per_step.mean())
    return {
        "car_8step":  car_mean,
        "lane_8step": lane_mean,
        "composite":  car_mean + LAMBDA_LANE * lane_mean,
        "car_per_step":  car_per_step.tolist(),
        "lane_per_step": lane_per_step.tolist(),
    }


# ----------------------------------------------------------------------
# Run all evaluations
# ----------------------------------------------------------------------
def main():
    print("Loading val data (segment-level split, seed=0)...")
    val_loader, base, val_eps = build_val_loader(batch_size=128)
    print(f"  val segments: {len(val_eps)}; val samples: {sum(1 for ep, _ in base.indices if ep in val_eps)}")

    results = {}

    # ---- PIWM-v6-kin — load BOTH Stage 2 (dyn.tar, pure-dynamics) and
    #      Stage 3 (best.tar, E2E-finetuned) for comparison. Baselines are
    #      pure-dynamics, so Stage 2's `dyn.tar` is the apples-to-apples one.
    for tag, ck_path, factory in [
            ("PIWM(stage2)",  "checkpoints/piwm_lane_v6_donkey/dyn.tar",  LaneAugmentedV6Kin),
            ("PIWM(stage3)",  "checkpoints/piwm_lane_v6_donkey/best.tar", LaneAugmentedV6Kin),
            ("PIWM-kin2",     "checkpoints/piwm_lane_v6_kin2_donkey/dyn.tar", LaneAugmentedV6Kin2),
            ("PIWM-donkey",   "checkpoints/piwm_lane_donkey/dyn.tar",    LaneAugmentedDonkey),
            ("PIWM-donkey-v2","checkpoints/piwm_lane_donkey_v2/dyn.tar", LaneAugmentedDonkeyV2),
            ("PIWM-donkey-v3","checkpoints/piwm_lane_donkey_v3/dyn.tar", LaneAugmentedDonkeyV3),
            # ---- clean K=8 matrix on CURRENT 71-seg data (identical split/seed) ----
            ("PIWM-v3eqv(base)",   "checkpoints/piwm_lane_donkey_v4_d0h0_K8_w1/dyn.tar", lambda: LaneAugmentedDonkeyV4(damp=False, hold=False)),
            ("PIWM-v4(damp+hold)", "checkpoints/piwm_lane_donkey_v4_v4_K8_w1/dyn.tar",   lambda: LaneAugmentedDonkeyV4(damp=True,  hold=True)),
            ("PIWM-v4+omega",      "checkpoints/piwm_lane_donkey_v4_v4_K8_w2/dyn.tar",   lambda: LaneAugmentedDonkeyV4(damp=True,  hold=True)),
    ]:
        if os.path.exists(ck_path):
            dyn = factory().to(DEVICE)
            dyn.load_state_dict(load_checkpoint(ck_path)['dynamics'])
            results[tag] = eval_nn_dynamics(dyn, val_loader)
        else:
            print(f"  skip {tag}: {ck_path} not found")

    # ---- DVBF ----
    dvbf_ck = "checkpoints/dvbf_lane_donkey/best.tar"
    if os.path.exists(dvbf_ck):
        m = DynamicsDVBFLane().to(DEVICE)
        m.load_state_dict(load_checkpoint(dvbf_ck)['model'])
        results["DVBF-lane"] = eval_nn_dynamics(m, val_loader)
    else:
        print(f"  skip DVBF: {dvbf_ck} not found")

    # ---- GOKU ----
    goku_ck = "checkpoints/goku_lane_donkey/best.tar"
    if os.path.exists(goku_ck):
        m = DynamicsGOKULane().to(DEVICE)
        m.load_state_dict(load_checkpoint(goku_ck)['model'])
        results["GOKU-lane"] = eval_nn_dynamics(m, val_loader)
    else:
        print(f"  skip GOKU: {goku_ck} not found")

    # ---- V2P (needs frozen encoder for theta) ----
    v2p_ck = "checkpoints/v2p_lane_donkey/best.tar"
    if os.path.exists(v2p_ck) and os.path.exists(ENCODER_CK):
        enc = PhysicsEncoderLane().to(DEVICE)
        enc.load_state_dict(load_checkpoint(ENCODER_CK)['encoder'])
        enc.eval()
        for p in enc.parameters(): p.requires_grad = False
        m = DynamicsVid2ParamLane().to(DEVICE)
        m.load_state_dict(load_checkpoint(v2p_ck)['model'])

        def _theta_provider(stack0):
            with torch.no_grad():
                obs = enc(stack0).unsqueeze(1)
            theta, _, _ = m.infer_theta(obs)
            return theta

        results["V2P-lane"] = eval_nn_dynamics(m, val_loader, theta_provider=_theta_provider)
    else:
        print(f"  skip V2P: ckpt or encoder missing")

    # ---- SINDYc ----
    sindyc_ck = "checkpoints/sindyc_lane_donkey/model.pkl"
    if os.path.exists(sindyc_ck):
        results["SINDYc-lane"] = eval_sindyc(sindyc_ck, base, val_eps)
    else:
        print(f"  skip SINDYc: {sindyc_ck} not found")

    # ---- Print comparison table ----
    print()
    print("=" * 78)
    print(f"  8-step rollout on val segments (z0 = GT). Lower is better.")
    print("=" * 78)
    print(f"  {'model':<14s}  {'car_MSE':>10s}  {'lane_MSE':>10s}  {'composite':>10s}")
    print("  " + "-" * 50)
    order = sorted(results.items(), key=lambda kv: kv[1]["composite"])
    for name, r in order:
        print(f"  {name:<14s}  {r['car_8step']:10.5f}  {r['lane_8step']:10.5f}  {r['composite']:10.5f}")

    # ---- Per-step error growth ----
    print()
    print("=" * 78)
    print("  Per-step car MSE (k=1..8)")
    print("=" * 78)
    header = "  " + " " * 14 + "  " + "".join(f"  k={k+1:<2d}  " for k in range(ROLLOUT_K))
    print(header)
    for name, r in order:
        cols = "".join(f" {v:7.4f}" for v in r["car_per_step"])
        print(f"  {name:<14s} {cols}")

    print()
    print("=" * 78)
    print("  Per-step lane MSE (k=1..8)")
    print("=" * 78)
    print(header)
    for name, r in order:
        cols = "".join(f" {v:7.4f}" for v in r["lane_per_step"])
        print(f"  {name:<14s} {cols}")

    # Persist for later inspection.
    out_path = "checkpoints/donkey_comparison.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved raw results -> {out_path}")


if __name__ == "__main__":
    main()
