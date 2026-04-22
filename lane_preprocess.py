"""Preprocess CarRacing episodes: extract per-frame lane waypoints in body frame.

For each .npz in the dataset, loads the track (from 'map' field), and for every
frame's (position, yaw), samples N waypoints along the centerline at fixed
arc-lengths. Saves as 'lane_wp_body' inside a sibling .lane.npz file (so we
don't risk corrupting original data).

Output layout:
    DATA_DIR/episode_XXXX.npz           (original, unchanged)
    DATA_DIR/episode_XXXX.lane.npz      (new, just 'lane_wp_body' key)

Loaded lazily by the new dataset class.
"""

import os, glob, sys
import numpy as np
from tqdm import tqdm

from track_utils import TrackCoords
from lane_utils import extract_lane_waypoints_body, LANE_S_SAMPLES, N_LANE_WP, LANE_DIM


def preprocess_one(npz_path, out_path, s_samples=LANE_S_SAMPLES):
    d = np.load(npz_path, allow_pickle=True)
    road_poly = d['map']
    positions = d['position'].astype(np.float32)   # (T, 2)
    yaws      = d['yaw'].astype(np.float32)        # (T,)

    track = TrackCoords(road_poly)

    T = len(positions)
    wp_all = np.zeros((T, N_LANE_WP, 2), dtype=np.float32)
    for t in range(T):
        wp_all[t] = extract_lane_waypoints_body(track, positions[t], yaws[t], s_samples)

    np.savez_compressed(out_path, lane_wp_body=wp_all, s_samples=s_samples)


def main(data_dir, out_dir=None):
    if out_dir is None:
        out_dir = data_dir
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    # Skip already-processed
    files = [f for f in files if not f.endswith(".lane.npz")]
    print(f"Preprocessing {len(files)} episodes from {data_dir}")

    all_wp = []
    for f in tqdm(files, desc="extract lane"):
        base = os.path.splitext(os.path.basename(f))[0]
        out  = os.path.join(out_dir, base + ".lane.npz")
        if os.path.exists(out):
            continue
        try:
            preprocess_one(f, out)
            # Sample for stats
            if len(all_wp) < 20:
                d = np.load(out)
                all_wp.append(d['lane_wp_body'])
        except Exception as e:
            print(f"  Failed {base}: {e}")

    # Report normalization stats
    if all_wp:
        stacked = np.concatenate(all_wp, axis=0)   # (M, 10, 2)
        flat = stacked.reshape(-1, LANE_DIM)
        m = flat.mean(axis=0); s = flat.std(axis=0)
        print(f"\nLane waypoint stats over {stacked.shape[0]} frames:")
        print(f"  mean = {m}")
        print(f"  std  = {s}")
        print(f"\nSuggested LANE_MEAN/STD in lane_utils.py:")
        print(f"  mean_xy_pairs: {m.reshape(N_LANE_WP, 2).tolist()}")
        print(f"  std_xy_pairs:  {s.reshape(N_LANE_WP, 2).tolist()}")


if __name__ == "__main__":
    DATA_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/train/controller_5/"
    main(DATA_DIR)
