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

from config import (DEVICE, DATA_DIR, ENCODER_DIM as CAR_ENCODER_DIM,
                    PHYSICS_MEAN_REL, PHYSICS_STD_REL)
from lane_utils import LANE_DIM, LANE_MEAN, LANE_STD, LANE_FRAME_STACK
from relative_coords import to_relative_np
from baselines.shared_dynamics_lane import (DynamicsDVBFLane, DynamicsGOKULane,
                                            DynamicsVid2ParamLane)
from models.encoder_lane import PhysicsEncoderLane
from utils import save_checkpoint, load_checkpoint
from donkey_dataset import (make_donkey_loaders, split_segments,
                            collect_state31_windows, _phys_flip_sign,
                            _phys_flip_offset, _lane_flip_sign, _lane_flip_offset,
                            _ACTION_FLIP_SIGN)
from train_piwm_lane_v5 import SeqLaneDataset

# ----------------------------------------------------------------------
# Shared hyperparams — kept identical to PIWM Stage 2 donkey for fairness
# ----------------------------------------------------------------------
ROLLOUT_K  = 8                      # set from --K in __main__ (longK uses 32)
EPOCHS     = 60
BATCH_NN   = 128
BATCH_V2P  = 64
LR         = 1e-3
LAMBDA_LANE = 2.0
GRAD_CLIP  = 5.0
SEED       = 0
SUFFIX     = ""                     # checkpoint dir suffix (e.g. "_longK"), set in __main__
SUP_DELTA  = 0.0                    # conference δ weak-supervision noise level, set in __main__

ENCODER_CK = "checkpoints/piwm_lane_v6_donkey/ae.tar"


def _split_loss(z, gt):
    """Match PIWM Stage 2 donkey: car MSE + LAMBDA_LANE * lane MSE.

    Returns (loss_total, loss_car_only, loss_lane_only) for monitoring.
    """
    car  = ((z[:, :11]  - gt[:, :11])  ** 2).mean()
    lane = ((z[:, 11:]  - gt[:, 11:]) ** 2).mean()
    return car + LAMBDA_LANE * lane, car.detach(), lane.detach()


# ----------------------------------------------------------------------
# Precompute windows as GPU-resident tensors (NO images for DVBF/GOKU; V2P
# precomputes the FROZEN encoder's obs once). The 31-dim state windows are
# tiny (~100 MB) so they live on-GPU and minibatches are pure index_select —
# this removes the image-laden DataLoader that profiled at 2.4x the rollout.
# ----------------------------------------------------------------------
def _precompute_nn(base, ep_set):
    """(Z, A) on DEVICE. Z (N,K+1,31) normalized state31, A (N,K+1,3) actions."""
    data = collect_state31_windows(base, ep_set, window=ROLLOUT_K + 1, stride=1, with_flip=True)
    Z = torch.from_numpy(np.stack([s for s, _ in data])).to(DEVICE)
    A = torch.from_numpy(np.stack([a for _, a in data])).to(DEVICE)
    return Z, A


def _precompute_v2p(base, ep_set, enc):
    """(Z, A, OBS) on DEVICE for V2P. OBS = frozen-encoder output of each window's
    15-frame stack, computed ONCE (orig + h-flip), so training never touches images.
    Stacks are built per-episode with a sliding-window VIEW (vectorized) rather than a
    per-window Python np.stack loop -- the loop dominated runtime on ~27k windows."""
    FS = LANE_FRAME_STACK; W = ROLLOUT_K + 1
    psgn, poff = _phys_flip_sign().numpy(), _phys_flip_offset().numpy()
    lsgn, loff = _lane_flip_sign().numpy(), _lane_flip_offset().numpy()
    asgn = _ACTION_FLIP_SIGN.numpy()
    Zs, As, OBS = [], [], []
    n_done = 0
    for ep in ep_set:
        phys = base.phys_list[ep]; acts = base.acts_list[ep]
        wpw = base.wp_world_list[ep]; imgs = base.imgs_list[ep]
        T = len(phys)
        t0s = np.arange(FS - 1, T - W + 1)
        if len(t0s) == 0:
            continue
        # state31 (orig + flip) per window -- cheap numpy, order [t0a_orig,t0a_flip,t0b_orig,...]
        for t0 in t0s:
            seg = phys[t0:t0 + W]; rel = to_relative_np(seg, 0)
            car = (rel - PHYSICS_MEAN_REL) / PHYSICS_STD_REL
            p0 = seg[0, :2]; y0 = seg[0, 2]; c, s = np.cos(y0), np.sin(y0)
            d = wpw[t0:t0 + W] - p0
            x = d[..., 0] * c + d[..., 1] * s; y = -d[..., 0] * s + d[..., 1] * c
            lane = (np.stack([x, y], -1).reshape(W, LANE_DIM) - LANE_MEAN) / LANE_STD
            s31 = np.concatenate([car, lane], -1).astype(np.float32)
            a = acts[t0:t0 + W].astype(np.float32)
            Zs.append(s31); As.append(a)
            s31f = s31.copy(); s31f[:, :11] = s31[:, :11] * psgn + poff
            s31f[:, 11:] = s31[:, 11:] * lsgn + loff
            Zs.append(s31f); As.append((a * asgn).astype(np.float32))
        # vectorized 15-frame stacks: stack for window t0 = imgs[t0-FS+1 : t0+1]
        sw = np.lib.stride_tricks.sliding_window_view(imgs, FS, axis=0)        # (T-FS+1,64,64,FS)
        stacks = np.moveaxis(sw, -1, 1)[t0s - (FS - 1)].astype(np.float32)     # (n,FS,64,64)
        with torch.no_grad():
            for j in range(0, len(stacks), 256):
                ch = stacks[j:j + 256]
                ob = enc(torch.from_numpy(ch).to(DEVICE))
                obf = enc(torch.from_numpy(ch[:, :, :, ::-1].copy()).to(DEVICE))
                OBS.append(torch.stack([ob, obf], 1).reshape(-1, ob.shape[-1]).cpu())  # orig,flip,...
        n_done += len(t0s)
        print(f"    [v2p precompute] {n_done} windows", flush=True)
    OBS = torch.cat(OBS, 0).to(DEVICE)
    Z = torch.from_numpy(np.stack(Zs)).to(DEVICE); A = torch.from_numpy(np.stack(As)).to(DEVICE)
    return Z, A, OBS


# ----------------------------------------------------------------------
# NN baseline boilerplate (DVBF, GOKU)
# ----------------------------------------------------------------------
_BASE = None
def _get_base():
    """SeqLaneDataset is built once and shared across DVBF/GOKU/V2P."""
    global _BASE
    if _BASE is None:
        _BASE = SeqLaneDataset(DATA_DIR, seq_len=ROLLOUT_K + 1)
    return _BASE


def _delta_noise(Z, half):
    """Conference δ weak-supervision noise on the 31-dim state labels (biased uniform,
    Sec. 4.1): mu_tilde = x + Δ (Δ~Unif[-half,half]); label = mean of 50 samples
    ~Unif[mu_tilde±half]. half = 0.5*δ*|X_i| per dim. Applied to train init+targets only."""
    b, t, d = Z.shape
    bias = (torch.rand(b, t, d, device=Z.device) * 2 - 1) * half
    resid = ((torch.rand(b, t, d, 50, device=Z.device) * 2 - 1) * half[:, None]).mean(-1)
    return Z + bias + resid


def _half_width(Ztr):
    """Per-dim 0.5*δ*|X_i| from the data range (zeros when δ=0)."""
    return 0.5 * SUP_DELTA * (Ztr.amax((0, 1)) - Ztr.amin((0, 1)))


def _train_nn_baseline(model_factory, name, save_path, batch_size=None):
    batch_size = BATCH_NN if batch_size is None else batch_size   # read global at call time
    print("=" * 60); print(f"Train {name} [donkey]"); print("=" * 60)
    base = _get_base()
    train_eps, val_eps = split_segments(base, val_frac=0.10, seed=SEED)
    Ztr, Atr = _precompute_nn(base, train_eps)
    Zval, Aval = _precompute_nn(base, val_eps)
    N = Ztr.shape[0]
    half = _half_width(Ztr)                      # δ weak-supervision noise half-widths (31,)

    model = model_factory().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')
    print(f"  param count: {sum(p.numel() for p in model.parameters()):,}  "
          f"windows: {N} train / {Zval.shape[0]} val (image-free, on {DEVICE}, batch={batch_size}, "
          f"delta={SUP_DELTA})")

    def _rollout(Z, A):
        z = Z[:, 0]; l_total = l_car = l_lane = 0.0
        for k in range(ROLLOUT_K):
            z = model(z, A[:, k])
            lt, lc, ll = _split_loss(z, Z[:, k + 1])
            l_total = l_total + lt; l_car = l_car + lc; l_lane = l_lane + ll
        return l_total / ROLLOUT_K, l_car / ROLLOUT_K, l_lane / ROLLOUT_K

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(N, device=DEVICE)
        t_loss, nb = 0.0, 0
        for i in range(0, N - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            Zb = _delta_noise(Ztr[idx], half) if SUP_DELTA > 0 else Ztr[idx]
            lt, lc, ll = _rollout(Zb, Atr[idx])
            opt.zero_grad(); lt.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); opt.step()
            t_loss += lt.item(); nb += 1
        t_loss /= nb

        model.eval()
        v_loss, v_car, v_lane, vn = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for i in range(0, Zval.shape[0], batch_size):
                lt, lc, ll = _rollout(Zval[i:i + batch_size], Aval[i:i + batch_size])
                v_loss += lt.item(); v_car += lc.item(); v_lane += ll.item(); vn += 1
        v_loss /= vn; v_car /= vn; v_lane /= vn
        sch.step(v_loss)
        print(f"  ep{epoch+1}/{EPOCHS}  Train tot={t_loss:.5f}  "
              f"Val tot={v_loss:.5f} car={v_car:.5f} lane={v_lane:.5f}")
        if v_loss < best:
            best = v_loss
            save_checkpoint({'model': model.state_dict(), 'val_loss': v_loss,
                             'val_car': v_car, 'val_lane': v_lane}, save_path)
    print(f"{name} best val_total: {best:.5f}")


def train_dvbf():
    _train_nn_baseline(DynamicsDVBFLane, "DVBF-lane",
                       f"checkpoints/dvbf_lane_donkey{SUFFIX}/best.tar")


def train_goku():
    _train_nn_baseline(DynamicsGOKULane, "GOKU-lane",
                       f"checkpoints/goku_lane_donkey{SUFFIX}/best.tar")


# ----------------------------------------------------------------------
# V2P-lane: needs frozen donkey encoder for theta inference
# ----------------------------------------------------------------------
def train_v2p():
    print("=" * 60); print("Train V2P-lane [donkey]"); print("=" * 60)
    if not os.path.exists(ENCODER_CK):
        raise FileNotFoundError(
            f"Encoder checkpoint not found: {ENCODER_CK}. "
            "Run PIWM Stage 1 first (train_piwm_lane_v6_donkey.py --stage ae).")

    base = _get_base()
    train_eps, val_eps = split_segments(base, val_frac=0.10, seed=SEED)

    enc = PhysicsEncoderLane().to(DEVICE)
    ck = load_checkpoint(ENCODER_CK)
    enc.load_state_dict(ck['encoder']); enc.eval()
    for p in enc.parameters(): p.requires_grad = False
    print(f"  frozen encoder from {ENCODER_CK}; precomputing obs (one pass) ...")
    Ztr, Atr, Otr = _precompute_v2p(base, train_eps, enc)
    Zval, Aval, Oval = _precompute_v2p(base, val_eps, enc)
    N = Ztr.shape[0]
    half = _half_width(Ztr)                      # δ weak-supervision noise half-widths (31,)

    model = DynamicsVid2ParamLane().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')
    print(f"  param count: {sum(p.numel() for p in model.parameters()):,}  "
          f"windows: {N} train / {Zval.shape[0]} val (obs precomputed, batch={BATCH_V2P})")

    def _rollout(Z, A, O):
        theta, mu, lv = model.infer_theta(O.unsqueeze(1))          # (B,1,29)
        z = Z[:, 0]; l_total = l_car = l_lane = 0.0
        for k in range(ROLLOUT_K):
            z = model.step(z, A[:, k], theta)
            lt, lc, ll = _split_loss(z, Z[:, k + 1])
            l_total = l_total + lt; l_car = l_car + lc; l_lane = l_lane + ll
        kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
        return l_total / ROLLOUT_K, l_car / ROLLOUT_K, l_lane / ROLLOUT_K, kl

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(N, device=DEVICE)
        t_loss, t_kl, nb = 0.0, 0.0, 0
        for i in range(0, N - BATCH_V2P + 1, BATCH_V2P):
            idx = perm[i:i + BATCH_V2P]
            Zb = _delta_noise(Ztr[idx], half) if SUP_DELTA > 0 else Ztr[idx]
            lt, lc, ll, kl = _rollout(Zb, Atr[idx], Otr[idx])
            loss = lt + 1e-3 * kl
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); opt.step()
            t_loss += lt.item(); t_kl += kl.item(); nb += 1
        t_loss /= nb; t_kl /= nb

        model.eval()
        v_loss, v_car, v_lane, vn = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for i in range(0, Zval.shape[0], BATCH_V2P):
                lt, lc, ll, kl = _rollout(Zval[i:i + BATCH_V2P], Aval[i:i + BATCH_V2P], Oval[i:i + BATCH_V2P])
                v_loss += lt.item(); v_car += lc.item(); v_lane += ll.item(); vn += 1
        v_loss /= vn; v_car /= vn; v_lane /= vn
        sch.step(v_loss)
        print(f"  ep{epoch+1}/{EPOCHS}  Train tot={t_loss:.5f} kl={t_kl:.4f}  "
              f"Val tot={v_loss:.5f} car={v_car:.5f} lane={v_lane:.5f}")
        if v_loss < best:
            best = v_loss
            save_checkpoint({'model': model.state_dict(), 'val_loss': v_loss,
                             'val_car': v_car, 'val_lane': v_lane},
                            f"checkpoints/v2p_lane_donkey{SUFFIX}/best.tar")
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
    p.add_argument("--K", type=int, default=8, help="training rollout horizon (longK uses 32)")
    p.add_argument("--suffix", default="", help="checkpoint dir suffix, e.g. _longK")
    p.add_argument("--batch", type=int, default=BATCH_NN, help="NN-baseline batch (DVBF/GOKU)")
    p.add_argument("--batch_v2p", type=int, default=BATCH_V2P, help="V2P batch")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="training epochs")
    p.add_argument("--delta", type=float, default=0.0,
                   help="conference δ weak-supervision noise on 31-dim state labels (e.g. 0.05)")
    p.add_argument("--degree",    type=int,   default=2)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--alpha",     type=float, default=0.01)
    args = p.parse_args()
    ROLLOUT_K = args.K
    SUFFIX = args.suffix
    BATCH_NN = args.batch
    BATCH_V2P = args.batch_v2p
    EPOCHS = args.epochs
    SUP_DELTA = args.delta
    print(f"[baselines] ROLLOUT_K={ROLLOUT_K}  SUFFIX='{SUFFIX}'  BATCH_NN={BATCH_NN}  "
          f"BATCH_V2P={BATCH_V2P}  EPOCHS={EPOCHS}  delta={SUP_DELTA}")

    for v in ("dvbf", "goku", "v2p"):
        os.makedirs(f"checkpoints/{v}_lane_donkey{SUFFIX}", exist_ok=True)

    if args.variant in ("dvbf",   "all"): train_dvbf()
    if args.variant in ("goku",   "all"): train_goku()
    if args.variant in ("v2p",    "all"): train_v2p()
    if args.variant in ("sindyc", "all"): fit_sindyc(degree=args.degree,
                                                     threshold=args.threshold,
                                                     alpha=args.alpha)
