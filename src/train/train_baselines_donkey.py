"""Donkey-adapted baseline trainers — DVBF / GOKU / V2P / SINDYc.

All four baselines operate on the same 31-dim state PIWM-v6-kin works on
(11 car + 20 lane waypoints), use the SAME donkey constants (DT, stats,
forward-only lane samples) via `donkey_config.patch_globals()`, the SAME
segment-level train/val split via `make_donkey_loaders`, and the SAME
training schedule (60 epochs, Adam 1e-3, ReduceLROnPlateau, grad clip 5.0,
horizontal-flip augmentation).

Loss is also aligned with PIWM Stage 2 donkey:
    loss = MSE(car) + LAMBDA_LANE * MSE(lane)        (LAMBDA_LANE = 2.0)
The baselines DO NOT get the residual regularization term (they have no
residual head).

All baselines train with z0 = GT phys + GT lane wp (dynamics-only setup
that matches PIWM Stage 2). V2P additionally uses PIWM Stage-1 encoder
(frozen, loaded from `checkpoints/piwm_lane_v6_donkey/ae.tar`) for its
theta inference — that's intrinsic to V2P, can't be removed.

Run:
    python train/train_baselines_donkey.py --variant dvbf
    python train/train_baselines_donkey.py --variant goku
    python train/train_baselines_donkey.py --variant v2p
    python train/train_baselines_donkey.py --variant sindyc
    python train/train_baselines_donkey.py --variant all
"""

# --- repo root on sys.path ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end shim ---

import donkey_config
donkey_config.patch_globals()

import os, argparse, pickle, time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import DEVICE, DATA_DIR, ENCODER_DIM as CAR_ENCODER_DIM
from lane_utils import LANE_DIM
from baselines.shared_dynamics_lane import (DynamicsDVBFLane, DynamicsGOKULane,
                                            DynamicsVid2ParamLane)
from models.encoder_lane import PhysicsEncoderLane
from utils import save_checkpoint, load_checkpoint
from donkey_dataset import (make_donkey_loaders, split_segments,
                            collect_state31_windows)
from train_piwm_lane_v5 import SeqLaneDataset

# ----------------------------------------------------------------------
# Shared hyperparams — kept identical to PIWM Stage 2 donkey for fairness
# ----------------------------------------------------------------------
ROLLOUT_K  = 8
EPOCHS     = 60
BATCH_NN   = 128
BATCH_V2P  = 64
LR         = 1e-3
LAMBDA_LANE = 2.0
GRAD_CLIP  = 5.0
SEED       = 0

ENCODER_CK = "checkpoints/piwm_lane_v6_donkey/ae.tar"


def _split_loss(z, gt):
    """Match PIWM Stage 2 donkey: car MSE + LAMBDA_LANE * lane MSE.

    Returns (loss_total, loss_car_only, loss_lane_only) for monitoring.
    """
    car  = ((z[:, :11]  - gt[:, :11])  ** 2).mean()
    lane = ((z[:, 11:]  - gt[:, 11:]) ** 2).mean()
    return car + LAMBDA_LANE * lane, car.detach(), lane.detach()


# ----------------------------------------------------------------------
# NN baseline boilerplate (DVBF, GOKU)
# ----------------------------------------------------------------------
def _train_nn_baseline(model_factory, name, save_path,
                       batch_size=BATCH_NN):
    print("=" * 60); print(f"Train {name} [donkey]"); print("=" * 60)
    tr_loader, val_loader, _ = make_donkey_loaders(
        DATA_DIR, seq_len=ROLLOUT_K + 1, batch_size=batch_size,
        val_frac=0.10, flip_aug=True, seed=SEED)

    model = model_factory().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  param count: {n_params:,}")

    for epoch in range(EPOCHS):
        model.train()
        t_loss, t_car, t_lane, nb = 0.0, 0.0, 0.0, 0
        for s0, fi, fp, fa, wn in tqdm(tr_loader, desc=f"{name} {epoch+1}/{EPOCHS}"):
            fp = fp.to(DEVICE); fa = fa.to(DEVICE); wn = wn.to(DEVICE)
            z = torch.cat([fp[:, 0], wn[:, 0]], dim=-1)             # (B, 31)
            l_total, l_car, l_lane = 0, 0, 0
            for k in range(ROLLOUT_K):
                z = model(z, fa[:, k])
                gt = torch.cat([fp[:, k + 1], wn[:, k + 1]], dim=-1)
                lt, lc, ll = _split_loss(z, gt)
                l_total = l_total + lt
                l_car   = l_car   + lc
                l_lane  = l_lane  + ll
            l_total /= ROLLOUT_K; l_car /= ROLLOUT_K; l_lane /= ROLLOUT_K
            opt.zero_grad(); l_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            t_loss += l_total.item(); t_car += l_car.item(); t_lane += l_lane.item(); nb += 1
        t_loss /= nb; t_car /= nb; t_lane /= nb

        model.eval()
        v_loss, v_car, v_lane, vn = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for s0, fi, fp, fa, wn in val_loader:
                fp = fp.to(DEVICE); fa = fa.to(DEVICE); wn = wn.to(DEVICE)
                z = torch.cat([fp[:, 0], wn[:, 0]], dim=-1)
                l_total, l_car, l_lane = 0, 0, 0
                for k in range(ROLLOUT_K):
                    z = model(z, fa[:, k])
                    gt = torch.cat([fp[:, k + 1], wn[:, k + 1]], dim=-1)
                    lt, lc, ll = _split_loss(z, gt)
                    l_total = l_total + lt; l_car = l_car + lc; l_lane = l_lane + ll
                v_loss += (l_total/ROLLOUT_K).item()
                v_car  += (l_car  /ROLLOUT_K).item()
                v_lane += (l_lane /ROLLOUT_K).item()
                vn += 1
        v_loss /= vn; v_car /= vn; v_lane /= vn
        sch.step(v_loss)
        print(f"  Train tot={t_loss:.5f} car={t_car:.5f} lane={t_lane:.5f}  "
              f"Val tot={v_loss:.5f} car={v_car:.5f} lane={v_lane:.5f}")
        if v_loss < best:
            best = v_loss
            save_checkpoint({'model': model.state_dict(),
                             'val_loss': v_loss,
                             'val_car': v_car, 'val_lane': v_lane},
                            save_path)
    print(f"{name} best val_total: {best:.5f}")


def train_dvbf():
    _train_nn_baseline(DynamicsDVBFLane, "DVBF-lane",
                       "checkpoints/dvbf_lane_donkey/best.tar")


def train_goku():
    _train_nn_baseline(DynamicsGOKULane, "GOKU-lane",
                       "checkpoints/goku_lane_donkey/best.tar")


# ----------------------------------------------------------------------
# V2P-lane: needs frozen donkey encoder for theta inference
# ----------------------------------------------------------------------
def train_v2p():
    print("=" * 60); print("Train V2P-lane [donkey]"); print("=" * 60)
    if not os.path.exists(ENCODER_CK):
        raise FileNotFoundError(
            f"Encoder checkpoint not found: {ENCODER_CK}. "
            "Run PIWM Stage 1 first (train_piwm_lane_v6_donkey.py --stage ae).")

    tr_loader, val_loader, _ = make_donkey_loaders(
        DATA_DIR, seq_len=ROLLOUT_K + 1, batch_size=BATCH_V2P,
        val_frac=0.10, flip_aug=True, seed=SEED)

    enc = PhysicsEncoderLane().to(DEVICE)
    ck = load_checkpoint(ENCODER_CK)
    enc.load_state_dict(ck['encoder']); enc.eval()
    for p in enc.parameters(): p.requires_grad = False
    print(f"  frozen encoder from {ENCODER_CK}  (val_loss={ck.get('val_loss','?')})")

    model = DynamicsVid2ParamLane().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  param count: {n_params:,} (dynamics+theta heads, encoder frozen)")

    def _theta_from_stack(stack0):
        # PIWM and V2P see the same image window (15-frame stack). For theta
        # inference we feed a single (B, 1, 29) sequence into the GRU.
        with torch.no_grad():
            obs = enc(stack0)                          # (B, 29)
        return obs.unsqueeze(1)                        # (B, 1, 29)

    for epoch in range(EPOCHS):
        model.train()
        t_loss, t_car, t_lane, t_kl, nb = 0.0, 0.0, 0.0, 0.0, 0
        for s0, fi, fp, fa, wn in tqdm(tr_loader, desc=f"V2P {epoch+1}/{EPOCHS}"):
            s0 = s0.to(DEVICE); fp = fp.to(DEVICE); fa = fa.to(DEVICE); wn = wn.to(DEVICE)
            obs_hist = _theta_from_stack(s0)
            theta, mu, lv = model.infer_theta(obs_hist)
            z = torch.cat([fp[:, 0], wn[:, 0]], dim=-1)
            l_car, l_lane, l_total = 0, 0, 0
            for k in range(ROLLOUT_K):
                z = model.step(z, fa[:, k], theta)
                gt = torch.cat([fp[:, k + 1], wn[:, k + 1]], dim=-1)
                lt, lc, ll = _split_loss(z, gt)
                l_total = l_total + lt; l_car = l_car + lc; l_lane = l_lane + ll
            l_total /= ROLLOUT_K; l_car /= ROLLOUT_K; l_lane /= ROLLOUT_K
            kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
            loss = l_total + 1e-3 * kl
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            t_loss += l_total.item(); t_car += l_car.item(); t_lane += l_lane.item()
            t_kl += kl.item(); nb += 1
        t_loss /= nb; t_car /= nb; t_lane /= nb; t_kl /= nb

        model.eval()
        v_loss, v_car, v_lane, vn = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for s0, fi, fp, fa, wn in val_loader:
                s0 = s0.to(DEVICE); fp = fp.to(DEVICE); fa = fa.to(DEVICE); wn = wn.to(DEVICE)
                obs_hist = _theta_from_stack(s0)
                theta, _, _ = model.infer_theta(obs_hist)
                z = torch.cat([fp[:, 0], wn[:, 0]], dim=-1)
                l_car, l_lane, l_total = 0, 0, 0
                for k in range(ROLLOUT_K):
                    z = model.step(z, fa[:, k], theta)
                    gt = torch.cat([fp[:, k + 1], wn[:, k + 1]], dim=-1)
                    lt, lc, ll = _split_loss(z, gt)
                    l_total = l_total + lt; l_car = l_car + lc; l_lane = l_lane + ll
                v_loss += (l_total/ROLLOUT_K).item()
                v_car  += (l_car  /ROLLOUT_K).item()
                v_lane += (l_lane /ROLLOUT_K).item()
                vn += 1
        v_loss /= vn; v_car /= vn; v_lane /= vn
        sch.step(v_loss)
        print(f"  Train tot={t_loss:.5f} car={t_car:.5f} lane={t_lane:.5f} kl={t_kl:.4f}  "
              f"Val tot={v_loss:.5f} car={v_car:.5f} lane={v_lane:.5f}")
        if v_loss < best:
            best = v_loss
            save_checkpoint({'model': model.state_dict(),
                             'val_loss': v_loss,
                             'val_car': v_car, 'val_lane': v_lane},
                            "checkpoints/v2p_lane_donkey/best.tar")
    print(f"V2P-lane best val_total: {best:.5f}")


# ----------------------------------------------------------------------
# SINDYc-lane: PySINDy on 31-dim state + 3-dim action
# ----------------------------------------------------------------------
def fit_sindyc(degree=2, threshold=0.05, alpha=0.01,
               window=50, stride=25):
    print("=" * 60); print("Fit SINDYc-lane [donkey]"); print("=" * 60)
    import pysindy as ps

    base = SeqLaneDataset(DATA_DIR, seq_len=ROLLOUT_K + 1)
    train_eps, val_eps = split_segments(base, val_frac=0.10, seed=SEED)
    print(f"  segments: {len(train_eps)} train / {len(val_eps)} val")

    # Fit on train windows (orig + flipped).
    train_data = collect_state31_windows(
        base, train_eps, window=window, stride=stride, with_flip=True)
    print(f"  train windows (incl. flips): {len(train_data)}")
    if not train_data:
        raise RuntimeError("No SINDYc train windows generated.")

    states_list = [s for s, _ in train_data]
    acts_list   = [a for _, a in train_data]

    state_names = (
        ["x", "y", "yaw", "vx", "vy", "omega", "w0", "w1", "w2", "w3", "steer"]
        + [f"{ax}{i}" for i in range(10) for ax in ("lx", "ly")]
    )
    act_names = ["u_steer", "u_throttle", "u_brake"]

    lib = ps.PolynomialLibrary(degree=degree, include_bias=True)
    optimizer = ps.STLSQ(threshold=threshold, alpha=alpha)
    # PySINDy 2.x: DiscreteSINDy is its own class; feature_names goes to fit().
    model = ps.DiscreteSINDy(optimizer=optimizer, feature_library=lib)
    t0 = time.time()
    model.fit(states_list, t=1, u=acts_list,
              feature_names=state_names + act_names)
    print(f"  fit time: {time.time()-t0:.1f}s")

    os.makedirs("checkpoints/sindyc_lane_donkey", exist_ok=True)
    save_path = "checkpoints/sindyc_lane_donkey/model.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  saved -> {save_path}")

    # Eval: 8-step rollout starting from many positions in val segments.
    print("  eval on val segments...")
    eval_data = collect_state31_windows(
        base, val_eps, window=ROLLOUT_K + 1, stride=2, with_flip=False)
    print(f"  val windows: {len(eval_data)}")

    car_se, lane_se, n = 0.0, 0.0, 0
    for state31, action3 in eval_data:
        z = state31[0:1].copy()
        z_pred = [np.asarray(z, dtype=np.float32).flatten()]
        for k in range(ROLLOUT_K):
            u = action3[k:k + 1]
            z = np.asarray(model.predict(z, u=u), dtype=np.float32)
            z_pred.append(z.flatten())
        z_pred = np.stack(z_pred, axis=0)                         # (K+1, 31)
        err = (z_pred[1:] - state31[1:]) ** 2                     # (K, 31)
        car_se  += err[:, :11].mean()
        lane_se += err[:, 11:].mean()
        n += 1
    v_car = car_se / n
    v_lane = lane_se / n
    v_tot = v_car + LAMBDA_LANE * v_lane
    print(f"  SINDYc val: tot={v_tot:.5f} car={v_car:.5f} lane={v_lane:.5f}")

    with open("checkpoints/sindyc_lane_donkey/metrics.txt", "w") as f:
        f.write(f"val_total={v_tot:.6f}\nval_car={v_car:.6f}\nval_lane={v_lane:.6f}\n"
                f"degree={degree} threshold={threshold} alpha={alpha} "
                f"window={window} stride={stride}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="all",
                   choices=["dvbf", "goku", "v2p", "sindyc", "all"])
    p.add_argument("--degree",    type=int,   default=2)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--alpha",     type=float, default=0.01)
    args = p.parse_args()

    os.makedirs("checkpoints/dvbf_lane_donkey", exist_ok=True)
    os.makedirs("checkpoints/goku_lane_donkey", exist_ok=True)
    os.makedirs("checkpoints/v2p_lane_donkey",  exist_ok=True)

    if args.variant in ("dvbf",   "all"): train_dvbf()
    if args.variant in ("goku",   "all"): train_goku()
    if args.variant in ("v2p",    "all"): train_v2p()
    if args.variant in ("sindyc", "all"): fit_sindyc(degree=args.degree,
                                                     threshold=args.threshold,
                                                     alpha=args.alpha)
