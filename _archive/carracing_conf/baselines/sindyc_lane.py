"""SINDYc-lane baseline: 31-dim state (11 car + 20 lane) + 3-dim control.

Mirrors baselines/sindyc.py for fair comparison with DVBF-lane / GOKU-lane /
V2P-lane: same SeqLaneStateDataset conventions — car in relative body frame,
lane waypoints expressed in the reference-body frame at t0, both normalized
with PHYSICS_MEAN_REL / PHYSICS_STD_REL and LANE_MEAN / LANE_STD.

NOTE on library size: degree-2 polynomial on 34 variables (31 state + 3 action)
yields ~630 features. STLSQ with threshold=0.05 keeps it sparse. If fit time is
a problem, drop to degree=1 via --degree 1.
"""

import os, glob, sys, argparse
import numpy as np
import pickle
import pysindy as ps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PHYSICS_MEAN_REL, PHYSICS_STD_REL, DT
from lane_utils import LANE_DIM, LANE_MEAN, LANE_STD, N_LANE_WP
from relative_coords import to_relative_np


SAVE_PATH = "checkpoints/sindyc_lane/model.pkl"
SEQ_LEN = 50
MAX_EPISODES = 40


def _lane_wp_ref_body(phys, wp_world, t0, seq_len):
    """Express world-frame waypoints in the body frame at time t0.

    phys: (N, 11) raw (not relative, not normalized) — we only need pos & yaw.
    wp_world: (N, 10, 2) centerline waypoints in world frame.
    Returns (seq_len, 10, 2) in the ref-body frame — matches SeqLaneStateDataset.
    """
    wp_seg = wp_world[t0:t0 + seq_len]
    pos0 = phys[t0, 0:2]; yaw0 = phys[t0, 2]
    rel = wp_seg - pos0
    c, s = np.cos(yaw0), np.sin(yaw0)
    x =  rel[..., 0] * c + rel[..., 1] * s
    y = -rel[..., 0] * s + rel[..., 1] * c
    return np.stack([x, y], axis=-1).astype(np.float32)


def load_sequences(data_dir, seq_len=SEQ_LEN, max_episodes=MAX_EPISODES):
    """Load lane-augmented 31-dim trajectories."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    files = [f for f in files if ".lane." not in f][:max_episodes]
    seq_states, seq_acts = [], []

    for f in files:
        lane_f = f.replace(".npz", ".lane.npz")
        if not os.path.exists(lane_f): continue
        try:
            d = np.load(f, allow_pickle=True)
            pos   = d["position"].astype(np.float32)
            yaw   = d["yaw"].astype(np.float32)
            vel   = d["velocity"].astype(np.float32)
            omega = d["angular_velocity"].astype(np.float32)
            wheel = d["wheel_omega"].astype(np.float32)
            steer = d["steering_angle"].astype(np.float32)
            acts  = d["action"].astype(np.float32)
            phys  = np.column_stack([pos, yaw, vel, omega, wheel, steer])
            wp_body_self = np.load(lane_f)["lane_wp_body"].astype(np.float32)
            # Reconstruct world-frame waypoints (the stored ones are in the
            # self-body frame of each step; we need them in the frame at t0).
            cy, sy = np.cos(yaw), np.sin(yaw)
            wp_world = np.zeros_like(wp_body_self)
            wp_world[..., 0] = wp_body_self[..., 0] * cy[:, None] - wp_body_self[..., 1] * sy[:, None] + pos[:, 0:1]
            wp_world[..., 1] = wp_body_self[..., 0] * sy[:, None] + wp_body_self[..., 1] * cy[:, None] + pos[:, 1:2]

            n = len(phys)
            if n < seq_len + 1: continue
            for start in range(0, n - seq_len, seq_len // 2):
                seg_phys = phys[start:start + seq_len]
                seg_acts = acts[start:start + seq_len]
                rel = to_relative_np(seg_phys, ref_idx=0)
                car_norm = (rel - PHYSICS_MEAN_REL) / PHYSICS_STD_REL           # (S, 11)
                wp_ref = _lane_wp_ref_body(phys, wp_world, start, seq_len)       # (S, 10, 2)
                lane_norm = (wp_ref.reshape(seq_len, LANE_DIM) - LANE_MEAN) / LANE_STD
                state31 = np.concatenate([car_norm, lane_norm], axis=-1)        # (S, 31)
                seq_states.append(state31.astype(np.float32))
                seq_acts.append(seg_acts)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    return seq_states, seq_acts


def fit_sindyc_lane(degree=2, threshold=0.05, alpha=0.01):
    print("=" * 60)
    print(f"Fitting SINDYc-lane on 31-dim state (poly degree {degree})")
    print("=" * 60)
    seq_states, seq_acts = load_sequences(DATA_DIR)
    print(f"Loaded {len(seq_states)} sequences")
    if not seq_states:
        raise RuntimeError("No sequences loaded — check DATA_DIR and .lane.npz files.")

    lib = ps.ConcatLibrary([
        ps.PolynomialLibrary(degree=degree, include_bias=True),
    ])
    optimizer = ps.STLSQ(threshold=threshold, alpha=alpha)
    state_names = (
        ["x", "y", "yaw", "vx", "vy", "omega", "w0", "w1", "w2", "w3", "steer"]
        + [f"{ax}{i}" for i in range(N_LANE_WP) for ax in ("lx", "ly")]
    )
    act_names = ["u_steer", "u_gas", "u_brake"]
    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=lib,
        feature_names=state_names + act_names,
        discrete_time=True,
    )

    model.fit(seq_states, u=seq_acts, t=DT, multiple_trajectories=True)
    print("\nLearned model (showing first 11 dims):")
    try:
        model.print()
    except Exception as e:
        print(f"(print failed: {e})")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved SINDYc-lane model to {SAVE_PATH}")


def rollout_sindyc_lane(model, z0_norm, action_seq):
    """Rollout SINDYc-lane from z0 (31-dim, normalized) given actions.

    Args:
        model: fitted pysindy SINDy model (31-dim state, 3-dim control).
        z0_norm: (31,) initial state.
        action_seq: (T, 3) actions.

    Returns:
        (T+1, 31) state rollout, normalized.
    """
    states = [z0_norm]
    z = z0_norm.copy()
    for t in range(len(action_seq)):
        u = action_seq[t:t+1]
        z_next = model.predict(z.reshape(1, -1), u=u).squeeze(0)
        states.append(z_next.astype(np.float32))
        z = z_next
    return np.array(states)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--degree", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.01)
    args = p.parse_args()
    fit_sindyc_lane(degree=args.degree, threshold=args.threshold, alpha=args.alpha)
