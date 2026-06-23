"""Lane waypoint extraction in body frame.

IMPORTANT — CarRacing body-frame convention (consistent with rest of PIWM):
  - forward direction  = body +y  (i.e. yaw=0 means car points along world +y)
  - lateral direction  = body +x  (left-positive depending on yaw sign)
  - Verified by: velocity of moving car aligns with (-sin(yaw), cos(yaw))
  - Also see: PHYSICS_MEAN_REL has vy_rel mean ≈ +40 (forward) and
    vx_rel mean ≈ -14 (tiny lateral), confirming body+y is forward.

Each waypoint is stored as (x_b, y_b) in that frame:
  - y_b is along-track (forward/backward)
  - x_b is cross-track (lateral offset)
"""

import numpy as np
from track_utils import TrackCoords


# Arc-length samples relative to car's position on the track centerline.
# Negative = behind the car, positive = ahead. s_body values are in world units.
LANE_S_SAMPLES = np.array(
    [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0],
    dtype=np.float32,
)
N_LANE_WP = len(LANE_S_SAMPLES)   # 10
LANE_DIM = N_LANE_WP * 2          # 20 (x_b, y_b per waypoint)

# Frame-stack for the lane-aware encoder (independent of config.FRAME_STACK
# which the existing 3-frame baselines still use). 15 frames give the encoder
# more temporal context — important for noisy real sensors and for resolving
# lane shape disambiguation.
LANE_FRAME_STACK = 15


def extract_lane_waypoints_world(track: TrackCoords, car_pos, s_samples=LANE_S_SAMPLES):
    """Query track centerline at arc-lengths (car_s + s_samples), return world coords.

    Args:
        track: TrackCoords for this episode
        car_pos: (2,) world (x, y)
        s_samples: (N,) arc-length offsets from car position
    Returns:
        wp_world: (N, 2) world-frame waypoints
    """
    s_car, _ = track.xy_to_sd(np.asarray(car_pos, dtype=np.float32))
    s_query = s_car + s_samples
    d_query = np.zeros_like(s_samples)
    return track.sd_to_xy(s_query, d_query)   # (N, 2)


def world_to_body(points, car_pos, car_yaw):
    """Transform world-frame points into body frame.

    Body frame: x-axis = car forward (along yaw), y-axis = car left.
    Consistent with dynamics_bicycle_v4.py body frame.

    Args:
        points: (N, 2) world-frame
        car_pos: (2,) world-frame
        car_yaw: scalar
    Returns:
        (N, 2) body-frame
    """
    rel = np.asarray(points) - np.asarray(car_pos)
    c, s = np.cos(car_yaw), np.sin(car_yaw)
    # v_body = R(-yaw) v_world
    x_b =  rel[..., 0] * c + rel[..., 1] * s
    y_b = -rel[..., 0] * s + rel[..., 1] * c
    return np.stack([x_b, y_b], axis=-1).astype(np.float32)


def extract_lane_waypoints_body(track: TrackCoords, car_pos, car_yaw,
                                 s_samples=LANE_S_SAMPLES):
    """Full pipeline: car pose + track -> (N, 2) body-frame waypoints."""
    wp_world = extract_lane_waypoints_world(track, car_pos, s_samples)
    return world_to_body(wp_world, car_pos, car_yaw)


# ---- Normalization stats (measured from controller_5 training data, 19000 frames) ----
# Ordering: [wp0_x, wp0_y, wp1_x, wp1_y, ..., wp9_x, wp9_y]  (20 dims)
# Recall: y_b is along-track (forward direction), x_b is lateral.
LANE_MEAN = np.array([
    -0.30,  -4.00,   # s=-5
    -0.35,  -0.08,   # s= 0
    -0.41,   3.82,   # s= 5
    -0.57,   7.67,   # s=10
    -0.88,  11.40,   # s=15
    -1.10,  15.01,   # s=20
    -1.39,  18.42,   # s=25
    -1.86,  21.59,   # s=30
    -1.92,  24.77,   # s=35
    -2.02,  27.74,   # s=40
], dtype=np.float32)

LANE_STD = np.array([
    7.60, 6.11,
    7.09, 5.84,
    7.53, 6.25,
    7.86, 7.07,
    7.92, 8.14,
    8.73, 9.51,
    9.59, 11.09,
    10.24, 12.92,
    10.93, 14.64,
    11.55, 16.44,
], dtype=np.float32)


def compute_lane_stats(wp_all):
    """Compute mean/std over (M, 10, 2) waypoint data; flatten to (20,)."""
    flat = wp_all.reshape(-1, LANE_DIM)
    return flat.mean(axis=0), flat.std(axis=0)
