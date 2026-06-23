"""Convert raw donkeycar npz trajectories into the v5/v6 dataset format.

Highlights of this version (post field-by-field audit of the raw data):
  * Uses LOGGED speed (state[:, 3]) instead of finite-diff, since the donkey
    records wheel-encoder speed which is denoiser and lag-free. Verified
    corr=0.72 with finite-diff and matches mean speed magnitude.
  * Recovers PHYSICAL steering angle (radians) from omega/v kinematics, not
    the asymmetric normalised servo command in action[:, 0]. The command is
    kept in action[:, 0] for the model to read.
  * GLITCH FILTERING:
      - NaN in state cols 2/3 or cte (boundary frames).
      - interpolated == 1 (donkey filled in missing frames; unreliable).
      - position step > 0.2 m (teleport/respawn; observed in traj2 around
        t=3614-3616 with concurrent state[:,3]=4.66 m/s spike).
    Glitches are dilated ±3 frames; each trajectory is then split into
    CLEAN CONTIGUOUS SEGMENTS and each segment is written as its own
    pseudo-episode so SeqLaneDataset never crosses a glitch.
  * yaw is degrees wrapped to [-180, 180]: unwrap THEN shift by -π/2 so
    PIWM's body+y = forward convention holds.
  * Spatial quantities upscaled by DONKEY_SPATIAL_SCALE (×50) to match the
    parameter bounds of the kinematic bicycle model.
"""
# --- auto-added: make repo root importable when run as a script ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
# --- end shim ---

import os, glob
import numpy as np

from donkey_config import (DONKEY_RAW_DIR, DONKEY_DATA_DIR, DONKEY_DT,
                           DONKEY_LANE_S_SAMPLES, DONKEY_LANE_S_SAMPLES_PHYS,
                           DONKEY_SPATIAL_SCALE, DONKEY_STATS_FILE,
                           DONKEY_WHEELBASE_PHYS, DONKEY_STEER_PHYS_MAX)
from donkey_track import DonkeyCenterline
from relative_coords import to_relative_np


N_LANE_WP = 10
LANE_DIM = 20
ROLLOUT_K = 8
LANE_FRAME_STACK = 15
# Need room for FRAME_STACK history + the longest rollout horizon used
# anywhere downstream. We train at K=8 originally and also retrain at K=32
# for long-horizon comparisons, so size for the larger.
MIN_SEGMENT_LEN = LANE_FRAME_STACK + 32 + 1 + 5         # = 53
POS_JUMP_M = 0.2                     # raw donkey metres; >0.2m/frame is teleport
GLITCH_DILATE = 3                    # also drop frames within this many of a glitch


# ----------------------------------------------------------------------
# Image conversion
# ----------------------------------------------------------------------
def rgb_to_gray(rgb_chw):
    """(T, 3, 64, 64) uint8 -> (T, 64, 64) float32 in [0,1].

    Uses the RED channel directly instead of BT.601 luma. The donkey track's
    lane markings are red on a grey background: luma weights R at only 0.299
    so the lane line ends up at value 76 and the grey ground at ~130
    (contrast 0.21). The raw R channel gives the lane line value 255 vs grey
    ground 130 (contrast 0.50) — a 2.4x improvement at zero cost, and color
    cues unrelated to the lane (walls, posters) are filtered out.
    """
    return (rgb_chw[:, 0].astype(np.float32) / 255.0).astype(np.float32)


# ----------------------------------------------------------------------
# Glitch detection / segment splitting
# ----------------------------------------------------------------------
def _bad_frame_mask(state, cte, interpolated):
    """Return (T,) bool; True = frame is unusable."""
    T = len(state)
    bad = np.zeros(T, dtype=bool)
    bad |= np.isnan(state).any(axis=1)
    bad |= np.isnan(cte)
    bad |= (interpolated.astype(np.int32) == 1)

    pos = state[:, :2]
    valid = ~np.isnan(pos).any(axis=1)
    steps = np.zeros(T, dtype=np.float32)
    if valid.sum() >= 2:
        # We only compare consecutive valid frames; pos-jumps that span a NaN
        # gap are already excluded by the NaN mask above.
        d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        steps[1:] = d
    bad |= (steps > POS_JUMP_M)

    if GLITCH_DILATE > 0:
        # Morphological dilation along the time axis.
        kernel = np.ones(2 * GLITCH_DILATE + 1, dtype=np.int32)
        bad_int = bad.astype(np.int32)
        bad_dilated = np.convolve(bad_int, kernel, mode='same') > 0
        bad = bad_dilated
    return bad


def _contiguous_segments(bad_mask, min_len):
    """Yield (start, end) inclusive index pairs for contiguous False runs."""
    T = len(bad_mask)
    i = 0
    segs = []
    while i < T:
        if bad_mask[i]:
            i += 1; continue
        j = i
        while j < T and not bad_mask[j]:
            j += 1
        # [i, j)
        if (j - i) >= min_len:
            segs.append((i, j))
        i = j
    return segs


# ----------------------------------------------------------------------
# Per-segment physics construction
# ----------------------------------------------------------------------
def build_segment_physics(raw, seg_slice):
    """Build all per-frame quantities for one clean segment.

    Returns dict with keys: imgs, position, yaw, velocity, angular_velocity,
    wheel_omega, steering_angle, action  (already in raw donkey units, NOT
    yet upscaled by DONKEY_SPATIAL_SCALE).
    """
    s = seg_slice
    state    = raw["state"][s].astype(np.float32)
    action   = raw["action"][s].astype(np.float32)
    frame_ch = raw["frame"][s]

    pos      = state[:, :2]
    yaw_deg  = state[:, 2]
    v_logged = state[:, 3]                                # m/s (wheel encoder)

    # Donkey yaw=0 means heading world+X (standard math convention). Shift by
    # -pi/2 so PIWM's "body+y = forward" convention holds.
    yaw_rad = (np.unwrap(yaw_deg * np.pi / 180.0) - np.pi / 2).astype(np.float32)

    # Angular velocity: central difference on the (now-clean) unwrapped yaw.
    omega = np.gradient(yaw_rad) / DONKEY_DT

    # World-frame velocity reconstructed from logged forward speed + yaw.
    # PIWM forward = (-sin(yaw), cos(yaw)) in world.
    c = np.cos(yaw_rad); sn = np.sin(yaw_rad)
    vx = -v_logged * sn
    vy =  v_logged * c
    velocity = np.stack([vx, vy], axis=-1).astype(np.float32)

    # All four wheel speeds = forward speed (donkey logs only one).
    wheel_omega = np.tile(v_logged[:, None], (1, 4)).astype(np.float32)

    # Physical steering angle (rad) from kinematics.  Low-speed clamp.
    v_eps = 0.05
    v_safe = np.maximum(np.abs(v_logged), v_eps)
    sign = np.sign(v_logged); sign[sign == 0] = 1.0
    steer_phys = np.arctan(omega * DONKEY_WHEELBASE_PHYS / (sign * v_safe))
    steer_phys = np.clip(steer_phys,
                         -DONKEY_STEER_PHYS_MAX, DONKEY_STEER_PHYS_MAX).astype(np.float32)
    steer_phys[np.abs(v_logged) < v_eps] = 0.0

    # 3-dim action: (steer_cmd, throttle, brake=0).
    steer_cmd = action[:, 0].astype(np.float32)
    throttle  = action[:, 1].astype(np.float32)
    action3 = np.stack([steer_cmd, throttle, np.zeros_like(steer_cmd)], axis=-1)

    imgs = rgb_to_gray(frame_ch)

    return dict(
        imgs=imgs,
        position=pos,
        yaw=yaw_rad,
        velocity=velocity,
        angular_velocity=omega.astype(np.float32),
        wheel_omega=wheel_omega,
        steering_angle=steer_phys,
        action=action3,
        v_logged=v_logged,
    )


# ----------------------------------------------------------------------
# Lane waypoint extraction
# ----------------------------------------------------------------------
def detect_centerline_direction(track, pos, yaw):
    """+1 if centerline arclength s increases with car forward motion, else -1."""
    s_car, _ = track.xy_to_sd(pos)
    s_unw = np.unwrap(s_car * (2 * np.pi / track.total_len)) * (track.total_len / (2 * np.pi))
    ds = np.gradient(s_unw)
    c, sn = np.cos(yaw), np.sin(yaw)
    vel = np.gradient(pos, axis=0)
    fwd = -vel[:, 0] * sn + vel[:, 1] * c
    return 1.0 if np.mean(ds * fwd) > 0 else -1.0


def extract_lane_body(track, pos, yaw, direction=1.0):
    """For every frame, query centerline at car_s + direction*DONKEY_LANE_S_SAMPLES_PHYS,
    transform into body frame, then scale up by DONKEY_SPATIAL_SCALE.
    """
    T = len(pos)
    s_car, _ = track.xy_to_sd(pos)
    out = np.zeros((T, N_LANE_WP, 2), dtype=np.float32)
    samples_phys = DONKEY_LANE_S_SAMPLES_PHYS * direction
    for t in range(T):
        s_query = s_car[t] + samples_phys
        d_query = np.zeros_like(samples_phys)
        wp_world = track.sd_to_xy(s_query, d_query)              # (10, 2)
        rel = wp_world - pos[t]
        c, sn = np.cos(yaw[t]), np.sin(yaw[t])
        x_b =  rel[:, 0] * c + rel[:, 1] * sn
        y_b = -rel[:, 0] * sn + rel[:, 1] * c
        out[t] = np.stack([x_b, y_b], axis=-1)
    return (out * DONKEY_SPATIAL_SCALE).astype(np.float32)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    os.makedirs(DONKEY_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DONKEY_STATS_FILE), exist_ok=True)
    # Clean any stale processed files so we don't mix old segments in.
    for old in glob.glob(os.path.join(DONKEY_DATA_DIR, "*.npz")):
        os.remove(old)

    files = sorted(glob.glob(os.path.join(DONKEY_RAW_DIR, "*.npz")))
    if not files:
        raise FileNotFoundError(f"no donkey npz files in {DONKEY_RAW_DIR}")

    raw0 = np.load(files[0], allow_pickle=True)
    print(f"Building centerline from map_world ({raw0['map_world'].shape[0]} pts)...")
    track = DonkeyCenterline(raw0["map_world"])
    print(f"  track total_len = {track.total_len:.3f} m, n_segs = {track.n}")

    all_phys_segs = []
    all_lane_wp = []
    total_clean_frames = 0
    total_raw_frames   = 0

    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        print(f"\n[{base}] loading ...")
        raw = np.load(f, allow_pickle=True)
        T = len(raw["frame"])
        total_raw_frames += T

        bad = _bad_frame_mask(raw["state"], raw["cte"], raw["interpolated"])
        segs = _contiguous_segments(bad, MIN_SEGMENT_LEN)
        clean = T - bad.sum()
        print(f"  T={T}, bad={bad.sum()} ({bad.mean()*100:.1f}%), "
              f"{len(segs)} clean segment(s) >= {MIN_SEGMENT_LEN} frames "
              f"covering {sum(j-i for i,j in segs)} frames")

        # Detect direction once using the largest clean segment.
        i0, j0 = max(segs, key=lambda ij: ij[1] - ij[0])
        pos_d = raw["state"][i0:j0, :2].astype(np.float32)
        yaw_d = np.unwrap(raw["state"][i0:j0, 2].astype(np.float32)
                          * np.pi / 180.0) - np.pi / 2
        direction = detect_centerline_direction(track, pos_d, yaw_d)
        print(f"  centerline direction relative to car motion: {direction:+.0f}")

        # Process each clean segment as its own pseudo-episode.
        for k, (i, j) in enumerate(segs):
            sl = slice(i, j)
            seg = build_segment_physics(raw, sl)
            wp_body = extract_lane_body(track, seg["position"], seg["yaw"], direction)

            # Upscale spatial quantities.
            pos_s   = seg["position"]    * DONKEY_SPATIAL_SCALE
            vel_s   = seg["velocity"]    * DONKEY_SPATIAL_SCALE
            wheel_s = seg["wheel_omega"] * DONKEY_SPATIAL_SCALE
            phys = np.column_stack([
                pos_s, seg["yaw"], vel_s,
                seg["angular_velocity"],
                wheel_s, seg["steering_angle"]
            ]).astype(np.float32)
            assert phys.shape[1] == 11

            name = f"{base}_seg{k:02d}_{i:05d}_{j:05d}"
            out_main = os.path.join(DONKEY_DATA_DIR, f"{name}.npz")
            out_lane = os.path.join(DONKEY_DATA_DIR, f"{name}.lane.npz")
            np.savez_compressed(
                out_main,
                imgs=seg["imgs"],
                position=pos_s,
                yaw=seg["yaw"],
                velocity=vel_s,
                angular_velocity=seg["angular_velocity"],
                wheel_omega=wheel_s,
                steering_angle=seg["steering_angle"],
                action=seg["action"],
            )
            np.savez_compressed(out_lane, lane_wp_body=wp_body)
            total_clean_frames += (j - i)

            # Sample rollout windows for relative-coords stats.
            seq_len = ROLLOUT_K + 1
            stride = max(1, seq_len // 2)
            for t0 in range(LANE_FRAME_STACK - 1, len(phys) - seq_len, stride):
                rel = to_relative_np(phys[t0:t0 + seq_len], ref_idx=0)
                all_phys_segs.append(rel)
            all_lane_wp.append(wp_body)

    print(f"\nTotal: kept {total_clean_frames}/{total_raw_frames} frames "
          f"({100*total_clean_frames/total_raw_frames:.1f}%)")

    # Stats.
    print("\nComputing normalization stats...")
    all_phys = np.stack(all_phys_segs, axis=0).reshape(-1, 11)
    phys_mean = all_phys.mean(axis=0).astype(np.float32)
    phys_std  = all_phys.std(axis=0).astype(np.float32)
    phys_std[phys_std < 1e-6] = 1.0

    all_wp = np.concatenate(all_lane_wp, axis=0).reshape(-1, LANE_DIM)
    lane_mean = all_wp.mean(axis=0).astype(np.float32)
    lane_std  = all_wp.std(axis=0).astype(np.float32)
    lane_std[lane_std < 1e-6] = 1.0

    np.savez(
        DONKEY_STATS_FILE,
        physics_mean_rel=phys_mean,
        physics_std_rel=phys_std,
        lane_mean=lane_mean,
        lane_std=lane_std,
        lane_s_samples=DONKEY_LANE_S_SAMPLES,
    )
    print(f"  saved -> {DONKEY_STATS_FILE}")
    names = ["x", "y", "yaw", "vx", "vy", "omega", "w0", "w1", "w2", "w3", "steer"]
    print(f"  PHYSICS_MEAN_REL = {dict(zip(names, np.round(phys_mean, 4).tolist()))}")
    print(f"  PHYSICS_STD_REL  = {dict(zip(names, np.round(phys_std,  4).tolist()))}")
    print(f"  LANE_MEAN (s_phys={DONKEY_LANE_S_SAMPLES_PHYS.tolist()}, scaled units):")
    print(lane_mean.reshape(N_LANE_WP, 2))
    print(f"  LANE_STD:")
    print(lane_std.reshape(N_LANE_WP, 2))


if __name__ == "__main__":
    main()
