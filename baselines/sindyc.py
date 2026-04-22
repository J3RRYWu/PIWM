"""SINDYc baseline: Sparse Identification of Nonlinear Dynamics with Control.

Fits sparse regression models:  dz/dt = Theta(z, u) * Xi
where Theta is a library of candidate nonlinear features (polynomials, trig)
and Xi is a sparse coefficient matrix.

Operates on relative-coordinate state (11-dim) + action (3-dim).
No image processing — the state is provided as ground truth.

For image MSE: we decode predicted state via a frozen PIWM decoder (9-dim).
"""

import os, glob, sys
import numpy as np
import pickle
import pysindy as ps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PHYSICS_MEAN_REL, PHYSICS_STD_REL, DT
from relative_coords import to_relative_np


SAVE_PATH = "checkpoints/sindyc/model.pkl"


def load_sequences(data_dir, seq_len=50, max_episodes=40):
    """Load multiple rollout sequences in RELATIVE coords."""
    files = sorted(glob.glob(f"{data_dir}/*.npz"))[:max_episodes]
    sequences_state = []
    sequences_action = []

    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            pos = d["position"].astype(np.float32)
            yaw = d["yaw"].astype(np.float32)
            vel = d["velocity"].astype(np.float32)
            omega = d["angular_velocity"].astype(np.float32)
            wheel = d["wheel_omega"].astype(np.float32)
            steer = d["steering_angle"].astype(np.float32)
            acts = d["action"].astype(np.float32)
            physics = np.column_stack([pos, yaw, vel, omega, wheel, steer])
            n = len(physics)
            if n < seq_len + 1: continue

            # Cut into windows of seq_len each, converted to relative
            for start in range(0, n - seq_len, seq_len // 2):  # overlap stride
                seg_phys = physics[start:start + seq_len]
                seg_acts = acts[start:start + seq_len]  # pysindy wants same length as state
                rel = to_relative_np(seg_phys)
                sequences_state.append(rel)
                sequences_action.append(seg_acts)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    return sequences_state, sequences_action


def fit_sindyc():
    print("=" * 60)
    print("Fitting SINDYc on relative-coord state")
    print("=" * 60)
    seq_states, seq_acts = load_sequences(DATA_DIR, seq_len=50, max_episodes=40)
    print(f"Loaded {len(seq_states)} sequences")

    # Normalize states (match PIWM normalization)
    seq_states_norm = [(s - PHYSICS_MEAN_REL) / PHYSICS_STD_REL for s in seq_states]

    # Feature library: polynomial (degree 2) + some trig features on yaw/steer
    lib = ps.ConcatLibrary([
        ps.PolynomialLibrary(degree=2, include_bias=True),
    ])
    optimizer = ps.STLSQ(threshold=0.05, alpha=0.01)
    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=lib,
        feature_names=[
            "x", "y", "yaw", "vx", "vy", "omega",
            "w0", "w1", "w2", "w3", "steer",
            "u_steer", "u_gas", "u_brake",
        ],
        discrete_time=True,  # discrete-time: z_{t+1} = f(z_t, u_t)
    )

    # Fit: multiple trajectories with different lengths
    model.fit(seq_states_norm, u=seq_acts, t=DT, multiple_trajectories=True)
    print("\nLearned model:")
    try:
        model.print()
    except Exception as e:
        print(f"(print failed: {e})")

    # Save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved SINDYc model to {SAVE_PATH}")


def rollout_sindyc(model, z0_norm, action_seq):
    """Rollout SINDYc from z0 given action sequence.

    Args:
        model: fitted pysindy SINDy model
        z0_norm: (11,) initial state, normalized
        action_seq: (T, 3) action sequence

    Returns:
        (T+1, 11) normalized state rollout
    """
    states = [z0_norm]
    z = z0_norm.copy()
    for t in range(len(action_seq)):
        u = action_seq[t:t+1]
        # pysindy discrete-time predict: z_{t+1} = model.predict(z_t, u_t)
        z_next = model.predict(z.reshape(1, -1), u=u).squeeze(0)
        states.append(z_next)
        z = z_next
    return np.array(states)


if __name__ == "__main__":
    fit_sindyc()
