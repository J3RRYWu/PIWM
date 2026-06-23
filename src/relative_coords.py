"""Relative coordinate transforms for PIWM.

For each rollout window, we set the initial pose (x0, y0, yaw0) as the origin
of a local body frame. The state is then expressed relative to this frame,
which makes the problem translation+rotation invariant.

Equivalent to "car starts at origin facing +X direction" for every rollout.
"""

import numpy as np
import torch


# Indices into 11-dim physics state
IDX_X, IDX_Y, IDX_YAW = 0, 1, 2
IDX_VX, IDX_VY = 3, 4
IDX_OMEGA = 5
IDX_WHEELS = slice(6, 10)
IDX_STEER = 10


def to_relative_np(physics, ref_idx=0):
    """Convert absolute 11-dim physics sequence to relative (numpy).

    Args:
        physics: (T, 11) or (B, T, 11) array [x, y, yaw, vx, vy, omega, w0..w3, steer]
        ref_idx: index of reference frame (default 0, i.e. first frame)

    Returns:
        rel_physics: same shape as input
    """
    squeeze_batch = False
    if physics.ndim == 2:
        physics = physics[None]  # (1, T, 11)
        squeeze_batch = True

    rel = physics.copy().astype(np.float32)
    B, T, _ = rel.shape

    x0 = rel[:, ref_idx, IDX_X]      # (B,)
    y0 = rel[:, ref_idx, IDX_Y]
    yaw0 = rel[:, ref_idx, IDX_YAW]

    c, s = np.cos(yaw0), np.sin(yaw0)  # (B,)

    # Translate + rotate position
    dx = physics[:, :, IDX_X] - x0[:, None]  # (B, T)
    dy = physics[:, :, IDX_Y] - y0[:, None]
    rel[:, :, IDX_X] = c[:, None] * dx + s[:, None] * dy
    rel[:, :, IDX_Y] = -s[:, None] * dx + c[:, None] * dy

    # Rotate yaw (scalar angle offset)
    rel[:, :, IDX_YAW] = physics[:, :, IDX_YAW] - yaw0[:, None]

    # Rotate velocity to initial body frame
    vx = physics[:, :, IDX_VX]
    vy = physics[:, :, IDX_VY]
    rel[:, :, IDX_VX] = c[:, None] * vx + s[:, None] * vy
    rel[:, :, IDX_VY] = -s[:, None] * vx + c[:, None] * vy

    # omega, wheels, steer: unchanged (intrinsic quantities)

    if squeeze_batch:
        rel = rel[0]
    return rel


def to_relative_torch(physics, ref_idx=0):
    """Torch version of to_relative_np. Supports gradients."""
    squeeze_batch = False
    if physics.ndim == 2:
        physics = physics.unsqueeze(0)
        squeeze_batch = True

    rel = physics.clone()
    x0 = rel[:, ref_idx, IDX_X]
    y0 = rel[:, ref_idx, IDX_Y]
    yaw0 = rel[:, ref_idx, IDX_YAW]

    c = torch.cos(yaw0)
    s = torch.sin(yaw0)

    dx = physics[:, :, IDX_X] - x0.unsqueeze(-1)
    dy = physics[:, :, IDX_Y] - y0.unsqueeze(-1)
    rel[:, :, IDX_X] = c.unsqueeze(-1) * dx + s.unsqueeze(-1) * dy
    rel[:, :, IDX_Y] = -s.unsqueeze(-1) * dx + c.unsqueeze(-1) * dy
    rel[:, :, IDX_YAW] = physics[:, :, IDX_YAW] - yaw0.unsqueeze(-1)

    vx = physics[:, :, IDX_VX]
    vy = physics[:, :, IDX_VY]
    rel[:, :, IDX_VX] = c.unsqueeze(-1) * vx + s.unsqueeze(-1) * vy
    rel[:, :, IDX_VY] = -s.unsqueeze(-1) * vx + c.unsqueeze(-1) * vy

    if squeeze_batch:
        rel = rel.squeeze(0)
    return rel


def from_relative_np(rel_physics, ref_pose):
    """Inverse transform: relative → absolute.

    Args:
        rel_physics: (T, 11) relative state
        ref_pose: (3,) [x0, y0, yaw0] reference origin
    Returns:
        absolute (T, 11)
    """
    x0, y0, yaw0 = ref_pose
    c, s = np.cos(yaw0), np.sin(yaw0)
    abs_phys = rel_physics.copy().astype(np.float32)

    # Rotate position back (inverse of rotate by -yaw0, which is rotate by yaw0)
    rx, ry = rel_physics[:, IDX_X], rel_physics[:, IDX_Y]
    abs_phys[:, IDX_X] = c * rx - s * ry + x0
    abs_phys[:, IDX_Y] = s * rx + c * ry + y0
    abs_phys[:, IDX_YAW] = rel_physics[:, IDX_YAW] + yaw0

    rvx, rvy = rel_physics[:, IDX_VX], rel_physics[:, IDX_VY]
    abs_phys[:, IDX_VX] = c * rvx - s * rvy
    abs_phys[:, IDX_VY] = s * rvx + c * rvy

    return abs_phys


def compute_relative_stats(data_dir, seq_len=100, n_samples=20000, seed=0):
    """Sample many (t0, seq_len) rollout windows from episodes, convert to relative,
    and compute mean/std of each dim.
    """
    import glob
    rng = np.random.RandomState(seed)
    files = sorted(glob.glob(f"{data_dir}/*.npz"))

    all_rel = []
    attempts = 0
    while len(all_rel) * seq_len < n_samples * seq_len and attempts < n_samples * 3:
        f = files[rng.randint(len(files))]
        try:
            d = np.load(f, allow_pickle=True)
            pos = d["position"].astype(np.float32)
            yaw = d["yaw"].astype(np.float32)
            vel = d["velocity"].astype(np.float32)
            omega = d["angular_velocity"].astype(np.float32)
            wheel = d["wheel_omega"].astype(np.float32)
            steer = d["steering_angle"].astype(np.float32)
            physics = np.column_stack([pos, yaw, vel, omega, wheel, steer])
            n = len(physics)
            if n < seq_len + 1:
                attempts += 1
                continue
            t0 = rng.randint(n - seq_len)
            seg = physics[t0:t0 + seq_len]
            rel = to_relative_np(seg)
            all_rel.append(rel)
            if len(all_rel) >= n_samples:
                break
        except Exception:
            pass
        attempts += 1

    all_rel = np.stack(all_rel, axis=0)  # (N, T, 11)
    flat = all_rel.reshape(-1, 11)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    return mean, std


if __name__ == "__main__":
    # Quick verification of transforms (roundtrip)
    np.random.seed(0)
    fake = np.zeros((5, 11), dtype=np.float32)
    # simulate trajectory
    fake[:, 0] = [100, 102, 104, 106, 108]  # x drifting
    fake[:, 1] = [50, 51, 52, 53, 54]        # y drifting
    fake[:, 2] = [np.pi / 4, 0.8, 0.85, 0.9, 0.95]  # yaw
    fake[:, 3] = 10.0  # vx
    fake[:, 4] = 5.0   # vy
    rel = to_relative_np(fake)
    print("Original (first frame):", fake[0])
    print("Relative (first frame, should be zeros for x,y,yaw):", rel[0])
    print("Relative (last frame):", rel[-1])
    back = from_relative_np(rel, fake[0, :3])
    print("Roundtrip max err:", np.abs(back - fake).max())

    # Compute stats on real data
    import os
    DATA = "E:/Desktop/PIWM/hsai-predictor-main/hsai-predictor-main/RacingCar/data/train/controller_5/"
    if os.path.exists(DATA):
        print("\nComputing relative stats from", DATA)
        mean, std = compute_relative_stats(DATA, seq_len=100, n_samples=5000)
        names = ["x_rel", "y_rel", "yaw_rel", "vx_rel", "vy_rel",
                 "omega", "w0", "w1", "w2", "w3", "steer"]
        print(f"{'dim':<10s} {'mean':>10s} {'std':>10s}")
        for n, m, s in zip(names, mean, std):
            print(f"{n:<10s} {m:>10.4f} {s:>10.4f}")
