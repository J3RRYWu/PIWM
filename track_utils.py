"""Utilities for converting between (x, y) and track-relative (s, d) coordinates.

s = arc length along track centerline (0 to track_length, wraps around)
d = lateral offset (perpendicular distance from centerline, left negative / right positive)

Usage:
    track = TrackCoords(road_poly)
    s, d = track.xy_to_sd(x, y)
    x, y = track.sd_to_xy(s, d)
"""

import numpy as np


class TrackCoords:
    """Converts between (x, y) and (s, d) for a fixed track."""

    def __init__(self, road_poly):
        """
        Args:
            road_poly: list of [polygon, color] pairs from npz file.
                       Each polygon is a list of 4 (x, y) tuples.
        """
        # Extract centerline from tile centers
        centers = []
        for item in road_poly:
            poly = item[0]  # list of 4 (x, y)
            cx = sum(p[0] for p in poly) / len(poly)
            cy = sum(p[1] for p in poly) / len(poly)
            centers.append((cx, cy))

        self.centers = np.array(centers, dtype=np.float32)  # (N, 2)
        self.n = len(centers)

        # Compute cumulative arc length at each center
        diffs = np.diff(np.vstack([self.centers, self.centers[0:1]]), axis=0)
        seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
        self.cumlen = np.concatenate([[0], np.cumsum(seg_lens[:-1])])  # s at each center
        self.total_len = seg_lens.sum()
        self.seg_lens = seg_lens  # length of each segment (centers[i] to centers[i+1])

        # Segment directions (tangent)
        self.tangents = diffs / (seg_lens[:, None] + 1e-9)  # (N, 2)
        # Normals (perpendicular, left-hand rotate 90 deg)
        self.normals = np.stack([-self.tangents[:, 1], self.tangents[:, 0]], axis=-1)

    def xy_to_sd(self, xy):
        """Convert (x, y) to (s, d). Works on single point or batch.

        Args:
            xy: (2,) or (N, 2) array
        Returns:
            s: scalar or (N,) array
            d: scalar or (N,) array
        """
        single = (xy.ndim == 1)
        if single:
            xy = xy[None, :]

        # For each query point, find nearest segment (center[i] -> center[i+1])
        # Segment start = centers[i], direction = tangents[i], length = seg_lens[i]
        starts = self.centers  # (N_seg, 2)
        tangents = self.tangents  # (N_seg, 2)
        seg_lens = self.seg_lens  # (N_seg,)

        # Vector from segment start to query point: (N_query, N_seg, 2)
        rel = xy[:, None, :] - starts[None, :, :]
        # Projection onto tangent: (N_query, N_seg)
        t = (rel * tangents[None, :, :]).sum(axis=-1)
        # Clamp to [0, seg_len]
        t_clamped = np.clip(t, 0, seg_lens[None, :])
        # Closest point on segment: (N_query, N_seg, 2)
        closest = starts[None, :, :] + t_clamped[:, :, None] * tangents[None, :, :]
        # Distance to closest point: (N_query, N_seg)
        dist = np.sqrt(((xy[:, None, :] - closest) ** 2).sum(axis=-1))
        # Which segment is nearest: (N_query,)
        seg_idx = np.argmin(dist, axis=1)

        # Compute s (arc length)
        s = self.cumlen[seg_idx] + t_clamped[np.arange(len(xy)), seg_idx]

        # Compute d (signed perpendicular distance)
        # d = (xy - closest_point) . normal
        chosen_closest = closest[np.arange(len(xy)), seg_idx]
        chosen_normal = self.normals[seg_idx]
        d = ((xy - chosen_closest) * chosen_normal).sum(axis=-1)

        if single:
            return float(s[0]), float(d[0])
        return s, d

    def sd_to_xy(self, s, d):
        """Convert (s, d) back to (x, y). Works on single or batch."""
        single = np.isscalar(s)
        if single:
            s = np.array([s], dtype=np.float32)
            d = np.array([d], dtype=np.float32)
        else:
            s = np.asarray(s, dtype=np.float32)
            d = np.asarray(d, dtype=np.float32)

        # Wrap s to [0, total_len)
        s_wrapped = np.mod(s, self.total_len)

        # Find which segment: cumlen[i] <= s_wrapped < cumlen[i] + seg_lens[i]
        seg_idx = np.searchsorted(self.cumlen, s_wrapped, side='right') - 1
        seg_idx = np.clip(seg_idx, 0, self.n - 1)

        # Position along segment
        t = s_wrapped - self.cumlen[seg_idx]
        point_on_center = self.centers[seg_idx] + t[:, None] * self.tangents[seg_idx]

        # Add lateral offset
        xy = point_on_center + d[:, None] * self.normals[seg_idx]

        if single:
            return float(xy[0, 0]), float(xy[0, 1])
        return xy


def test_roundtrip():
    """Verify xy -> sd -> xy is identity."""
    import glob
    files = sorted(glob.glob('E:/Desktop/PIWM/hsai-predictor-main/hsai-predictor-main/RacingCar/data/train/fixed/*.npz'))
    d = np.load(files[0], allow_pickle=True)
    road_poly = d['map']
    positions = d['position'].astype(np.float32)

    track = TrackCoords(road_poly)
    print(f"Track: {track.n} tiles, total length = {track.total_len:.1f} units")

    # Forward
    s, d_lat = track.xy_to_sd(positions[:50])
    # Inverse
    xy_back = track.sd_to_xy(s, d_lat)

    # Error
    err = np.sqrt(((positions[:50] - xy_back) ** 2).sum(axis=1))
    print(f"Roundtrip error: mean={err.mean():.4f}, max={err.max():.4f}")

    # Print some examples
    print("\nExample conversions:")
    for i in [0, 10, 20, 30, 40]:
        print(f"  xy={positions[i]}, s={s[i]:.2f}, d={d_lat[i]:.3f}, xy_back={xy_back[i]}")

    # Distribution of s and d
    s_all, d_all = track.xy_to_sd(positions)
    print(f"\ns range: [{s_all.min():.1f}, {s_all.max():.1f}]")
    print(f"d range: [{d_all.min():.3f}, {d_all.max():.3f}]")


if __name__ == "__main__":
    test_roundtrip()
