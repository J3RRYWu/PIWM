"""Centerline utility for the donkey track.

The donkey `map_world` field is an unordered set of ~4629 dense points lying on
the lane center of a closed loop. This module:
  1. Orders them via greedy nearest-neighbor chaining (verified to give dense,
     uniform segments of ~1-15 mm).
  2. Exposes `xy_to_sd` / `sd_to_xy` like `track_utils.TrackCoords` so we can
     reuse `lane_utils.extract_lane_waypoints_*` without modification.

Why not reuse TrackCoords directly: TrackCoords expects a `road_poly` of
[polygon, color] pairs from CarRacing. The donkey centerline is just (N, 2).
"""
import numpy as np
from scipy.spatial import cKDTree


def order_centerline(points, k_search=20):
    """Order (N, 2) points along a closed loop via greedy nearest-unvisited chain.

    Returns the ordered (N, 2) array. The choice of start index is arbitrary;
    `xy_to_sd` is rotation-invariant w.r.t. the chosen origin since s wraps.
    """
    points = np.asarray(points, dtype=np.float32)
    n = len(points)
    tree = cKDTree(points)
    visited = np.zeros(n, dtype=bool)
    order = [0]
    visited[0] = True
    cur = 0
    for _ in range(n - 1):
        _, idxs = tree.query(points[cur], k=k_search)
        nxt = -1
        for i in idxs:
            if not visited[i]:
                nxt = int(i); break
        if nxt == -1:
            un = np.where(~visited)[0]
            sub = np.argmin(np.linalg.norm(points[un] - points[cur], axis=1))
            nxt = int(un[sub])
        order.append(nxt); visited[nxt] = True; cur = nxt
    return points[order]


class DonkeyCenterline:
    """Closed-loop polyline centerline. Mirrors TrackCoords API (xy_to_sd, sd_to_xy).

    Geometry:
      centers[i] -> centers[i+1] (wraps to centers[0] for last) define segments.
      cumlen[i] = arc length at centers[i] (cumlen[0] = 0).
      total_len = sum of all segment lengths.
    """

    def __init__(self, map_world):
        ordered = order_centerline(map_world)
        self.centers = ordered.astype(np.float32)
        self.n = len(ordered)

        diffs = np.diff(np.vstack([self.centers, self.centers[0:1]]), axis=0)
        seg_lens = np.sqrt((diffs ** 2).sum(axis=1)).astype(np.float32)
        self.seg_lens = seg_lens
        self.cumlen = np.concatenate(
            [[0.0], np.cumsum(seg_lens[:-1])]).astype(np.float32)
        self.total_len = float(seg_lens.sum())

        eps = 1e-9
        self.tangents = (diffs / (seg_lens[:, None] + eps)).astype(np.float32)
        # Left-hand normal (rotate tangent 90 deg CCW).
        self.normals = np.stack(
            [-self.tangents[:, 1], self.tangents[:, 0]], axis=-1).astype(np.float32)

        # KDTree on centers for fast nearest-segment lookup.
        self._tree = cKDTree(self.centers)

    def xy_to_sd(self, xy, k_search=8):
        """Convert (x, y) -> (s, d). Accepts (2,) or (N, 2)."""
        xy = np.asarray(xy, dtype=np.float32)
        single = (xy.ndim == 1)
        if single:
            xy = xy[None, :]

        # For each query, consider only the k nearest CENTERS and the two
        # segments incident to each (i -> i+1 and i-1 -> i). This is O(N * k)
        # instead of O(N * N_seg) which matters for 4629 segments.
        _, near = self._tree.query(xy, k=k_search)   # (N_q, k)
        cand_segs = np.unique(near.reshape(-1))
        # Include the previous segment for each (handles wraparound).
        cand_segs = np.unique(np.concatenate([cand_segs, (cand_segs - 1) % self.n]))

        starts   = self.centers[cand_segs]      # (M, 2)
        tangents = self.tangents[cand_segs]     # (M, 2)
        seg_lens = self.seg_lens[cand_segs]     # (M,)

        rel = xy[:, None, :] - starts[None, :, :]                       # (Nq, M, 2)
        t = (rel * tangents[None, :, :]).sum(axis=-1)                   # (Nq, M)
        t_clamped = np.clip(t, 0.0, seg_lens[None, :])
        closest = starts[None, :, :] + t_clamped[:, :, None] * tangents[None, :, :]
        dist = np.sqrt(((xy[:, None, :] - closest) ** 2).sum(axis=-1))  # (Nq, M)
        best = np.argmin(dist, axis=1)                                  # (Nq,)

        seg_idx = cand_segs[best]
        t_best = t_clamped[np.arange(len(xy)), best]
        closest_best = closest[np.arange(len(xy)), best]
        normal_best = self.normals[seg_idx]

        s = self.cumlen[seg_idx] + t_best
        d = ((xy - closest_best) * normal_best).sum(axis=-1)

        if single:
            return float(s[0]), float(d[0])
        return s.astype(np.float32), d.astype(np.float32)

    def sd_to_xy(self, s, d):
        single = np.isscalar(s)
        if single:
            s = np.array([s], dtype=np.float32)
            d = np.array([d], dtype=np.float32)
        else:
            s = np.asarray(s, dtype=np.float32)
            d = np.asarray(d, dtype=np.float32)

        s_wrap = np.mod(s, self.total_len)
        seg_idx = np.searchsorted(self.cumlen, s_wrap, side='right') - 1
        seg_idx = np.clip(seg_idx, 0, self.n - 1)
        t = s_wrap - self.cumlen[seg_idx]
        on_center = self.centers[seg_idx] + t[:, None] * self.tangents[seg_idx]
        xy = on_center + d[:, None] * self.normals[seg_idx]
        if single:
            return float(xy[0, 0]), float(xy[0, 1])
        return xy.astype(np.float32)
