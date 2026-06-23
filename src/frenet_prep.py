"""Build the Frenet-state dataset for the donkey world model.

Per frame, state z = [s, d, psi_e, v, omega] (5 interpretable dims):
  s     absolute arc-length on the fixed track (m, mod L)   -- for kappa lookup
  d     signed lateral offset from centerline (m)           -- == cte signal
  psi_e heading error (rad)                                 -- car-fwd vs track-tangent
  v     forward speed (m/s)                                 -- logged encoder
  omega yaw rate (rad/s)                                    -- smoothed finite-diff yaw

Also stored per frame:
  kappa_profile  kappa(s + [0,0.5,..,4.5] m)  (10 values)   -- road shape ahead (decoder)
  img            R-channel 64x64 in [0,1]
  action         [steer_cmd, throttle] (2)

Glitch filtering + clean-segment splitting reuse donkey_prep's logic. Spatial
quantities stay in REAL metres (no x50 scale — the Frenet equations are
scale-natural and the residual MLPs are tiny).

Output: Data_Donkeycar_frenet/<seg>.npz + _meta/stats.npz
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import os, glob
import numpy as np

from frenet_track import FrenetTrack
from donkey_prep import (_bad_frame_mask, _contiguous_segments, rgb_to_gray,
                         DONKEY_DT)

RAW_DIR  = _os.path.join(_os.path.dirname(__file__), "..", "..", "Data_Donkeycar")
OUT_DIR  = _os.path.join(_os.path.dirname(__file__), "..", "..", "Data_Donkeycar_frenet")
STATS    = _os.path.join(OUT_DIR, "_meta", "stats.npz")

KAPPA_OFFSETS = np.arange(0, 10, dtype=np.float32) * 0.5    # [0,0.5,..,4.5] m ahead
MIN_SEG = 15 + 32 + 1 + 5                                    # frame-stack + K=32 room
LANE_FRAME_STACK = 15


def smooth(x, w=5):
    x = np.asarray(x, np.float32)
    k = np.ones(w, np.float32) / w
    if x.ndim == 1:
        return np.convolve(x, k, mode='same')
    return np.stack([np.convolve(x[:, j], k, 'same') for j in range(x.shape[1])], 1)


def build():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(_os.path.dirname(STATS), exist_ok=True)
    for old in glob.glob(_os.path.join(OUT_DIR, "*.npz")):
        os.remove(old)

    files = sorted(glob.glob(_os.path.join(RAW_DIR, "*.npz")))
    # shared fixed track, oriented from the first file's trajectory
    raw0 = np.load(files[0], allow_pickle=True)
    st0 = raw0['state'][:].astype(np.float32); m0 = ~np.isnan(st0).any(axis=1)
    pos0 = st0[m0, :2]; yaw0 = np.unwrap(st0[m0, 2] * np.pi / 180) - np.pi / 2
    # 5cm grid + win9 smoothing is the tuned config. A finer 2cm grid was
    # tested and HURT long-horizon (0.246 -> 0.458 m@100) — finer kappa makes
    # the analytic propagation more sensitive to per-step noise. Keep 5cm.
    track = FrenetTrack(raw0['map_world'], grid_ds=0.05, smooth_win=9,
                        orient_xy=pos0, orient_yaw=yaw0)
    print(f"FrenetTrack: M={track.M}, L={track.total_len:.2f}m, flipped={track.flipped}")
    # persist the track grid so the model/eval use an identical kappa(s)
    np.savez(_os.path.join(_os.path.dirname(STATS), "track.npz"),
             centers=track.centers, heading=track.heading, kappa=track.kappa,
             normals=track.normals, total_len=track.total_len, grid_ds=track.grid_ds)

    all_state, all_kap = [], []
    kept = total = 0
    for f in files:
        base = _os.path.splitext(_os.path.basename(f))[0]
        raw = np.load(f, allow_pickle=True)
        T = len(raw['frame']); total += T
        bad = _bad_frame_mask(raw['state'], raw['cte'], raw['interpolated'])
        segs = _contiguous_segments(bad, MIN_SEG)
        print(f"[{base}] T={T} bad={bad.sum()} segs={len(segs)}")
        for k, (i, j) in enumerate(segs):
            sl = slice(i, j)
            state = raw['state'][sl].astype(np.float32)
            # fill any residual nans by edge (segments are already clean)
            for c in range(4):
                col = state[:, c]
                if np.isnan(col).any():
                    idx = np.where(~np.isnan(col), np.arange(len(col)), 0)
                    np.maximum.accumulate(idx, out=idx); col = col[idx]
                    state[:, c] = col
            pos = state[:, :2]
            yaw = (np.unwrap(state[:, 2] * np.pi / 180) - np.pi / 2).astype(np.float32)
            v = state[:, 3].astype(np.float32)
            omega = smooth(np.gradient(yaw) / DONKEY_DT, 5).astype(np.float32)

            s, d = track.xy_to_sd(pos)
            psi = track.psi_e(s, yaw)
            # light smoothing of the noisy lookup-derived signals (removing it
            # was tested and badly hurt: noisy targets -> 0.143m@25, 0.509m@100)
            d = smooth(d, 3); psi = smooth(psi, 3)
            kapprof = track.curvature_profile(s, KAPPA_OFFSETS)     # (T,10)

            z = np.stack([s, d, psi, v, omega], axis=-1).astype(np.float32)  # (T,5)
            acts = raw['action'][sl][:, :2].astype(np.float32)
            imgs = rgb_to_gray(raw['frame'][sl])

            name = f"{base}_seg{k:02d}_{i:05d}_{j:05d}"
            np.savez_compressed(_os.path.join(OUT_DIR, f"{name}.npz"),
                                state=z, kappa_profile=kapprof, imgs=imgs, action=acts)
            kept += (j - i)
            # stats: use relative s within window-ish — collect d,psi,v,omega only
            all_state.append(z[:, 1:])      # d,psi,v,omega (s handled separately)
            all_kap.append(kapprof)

    A = np.concatenate(all_state, 0)
    mean = np.concatenate([[0.0], A.mean(0)]).astype(np.float32)    # s mean=0 (relative)
    std = np.concatenate([[1.0], A.std(0)]).astype(np.float32)
    std[std < 1e-6] = 1.0
    kap = np.concatenate(all_kap, 0)
    np.savez(STATS, state_mean=mean, state_std=std,
             kappa_mean=kap.mean(), kappa_std=max(kap.std(), 1e-6),
             kappa_offsets=KAPPA_OFFSETS)
    print(f"\nkept {kept}/{total} ({100*kept/total:.1f}%)")
    names = ['s', 'd', 'psi_e', 'v', 'omega']
    print("state mean:", dict(zip(names, np.round(mean, 4).tolist())))
    print("state std :", dict(zip(names, np.round(std, 4).tolist())))
    print(f"kappa: mean={kap.mean():.3f} std={kap.std():.3f}")


if __name__ == "__main__":
    build()
