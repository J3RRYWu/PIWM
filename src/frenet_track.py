"""Frenet (track-relative) coordinate utility for the fixed donkey track.

The donkey track is FIXED (map_world identical across trajectories), so the road
geometry is a known, deterministic function of arc-length s. This module:

  1. Orders the raw map_world point cloud (NN-chain), resamples it to a uniform
     arc-length grid, and smooths it (the raw 4629-pt cloud has mm-scale noise
     that makes direct curvature estimation explode).
  2. Auto-orients so s increases along the driving direction (verified by the
     sign of d(s)/dt vs forward speed on a sample trajectory).
  3. Provides:
       xy_to_sd(xy)               -> (s, d)             signed lateral offset
       heading_at(s) / kappa_at(s)                      track tangent / curvature
       sd_to_xy(s, d)             -> (x, y)             inverse
       curvature_profile(s, off)  -> kappa(s + off)     road shape ahead (decoder)

Validated: lateral kinematics ḋ = v·sin(ψ_e) holds at corr=0.96 on donkey data
with the smoothed κ(s); see the design validation in the project history.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter1d

from donkey_track import order_centerline


class FrenetTrack:
    def __init__(self, map_world, grid_ds=0.05, smooth_win=9, orient_xy=None,
                 orient_yaw=None):
        ordered = order_centerline(np.asarray(map_world, dtype=np.float32))

        # cumulative arc-length of the ordered cloud
        seg = np.linalg.norm(np.diff(np.vstack([ordered, ordered[:1]]), axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        L = float(cum[-1])

        # resample to a uniform arc-length grid + smooth (closed loop)
        S = np.arange(0.0, L, grid_ds)
        cx = np.interp(S, cum[:-1], ordered[:, 0])
        cy = np.interp(S, cum[:-1], ordered[:, 1])
        cx = uniform_filter1d(cx, smooth_win, mode='wrap')
        cy = uniform_filter1d(cy, smooth_win, mode='wrap')

        self.grid_ds = float(grid_ds)
        self.total_len = float(len(S) * grid_ds)
        self.centers = np.stack([cx, cy], axis=-1).astype(np.float32)   # (M,2)
        self.s_grid = (np.arange(len(S)) * grid_ds).astype(np.float32)
        self.M = len(S)

        tx, ty = np.gradient(cx), np.gradient(cy)
        nrm = np.hypot(tx, ty) + 1e-9
        tx, ty = tx / nrm, ty / nrm
        self.heading = np.unwrap(np.arctan2(ty, tx)).astype(np.float32)   # (M,)
        kappa = np.gradient(self.heading) / grid_ds
        self.kappa = uniform_filter1d(kappa, smooth_win, mode='wrap').astype(np.float32)
        self.normals = np.stack([-np.sin(self.heading), np.cos(self.heading)],
                                axis=-1).astype(np.float32)

        self._tree = cKDTree(self.centers)
        self.flipped = False
        if orient_xy is not None and orient_yaw is not None:
            self._auto_orient(orient_xy, orient_yaw)

    # ------------------------------------------------------------------
    def _auto_orient(self, pos, yaw):
        """Reverse arc-length direction if s decreases as the car drives forward."""
        s, _ = self.xy_to_sd(pos)
        s_un = np.unwrap(s * (2 * np.pi / self.total_len)) * (self.total_len / (2 * np.pi))
        ds = np.gradient(s_un)
        # forward speed sign (PIWM body+y = forward)
        v = np.gradient(pos, axis=0)
        c, sn = np.cos(yaw), np.sin(yaw)
        fwd = -v[:, 0] * sn + v[:, 1] * c
        if np.nanmean(ds * fwd) < 0:
            self._reverse()
            self.flipped = True

    def _reverse(self):
        """Flip the centerline so s runs the other way."""
        self.centers = self.centers[::-1].copy()
        self.kappa = (-self.kappa[::-1]).copy()
        self.heading = (self.heading[::-1] + np.pi).astype(np.float32)
        self.normals = np.stack([-np.sin(self.heading), np.cos(self.heading)],
                                axis=-1).astype(np.float32)
        self.s_grid = (np.arange(self.M) * self.grid_ds).astype(np.float32)
        self._tree = cKDTree(self.centers)

    # ------------------------------------------------------------------
    def xy_to_sd(self, xy):
        """(x,y)->(s,d). d = signed lateral offset (left-normal positive)."""
        xy = np.asarray(xy, dtype=np.float32)
        single = xy.ndim == 1
        if single:
            xy = xy[None]
        _, idx = self._tree.query(xy)
        s = self.s_grid[idx]
        d = np.sum((xy - self.centers[idx]) * self.normals[idx], axis=-1)
        if single:
            return float(s[0]), float(d[0])
        return s.astype(np.float32), d.astype(np.float32)

    def heading_at(self, s):
        i = (np.mod(s, self.total_len) / self.grid_ds).astype(int) % self.M
        return self.heading[i]

    def kappa_at(self, s):
        i = (np.mod(s, self.total_len) / self.grid_ds).astype(int) % self.M
        return self.kappa[i]

    def sd_to_xy(self, s, d):
        i = (np.mod(np.asarray(s), self.total_len) / self.grid_ds).astype(int) % self.M
        return (self.centers[i] + np.asarray(d)[..., None] * self.normals[i]).astype(np.float32)

    def psi_e(self, s, yaw):
        """Heading error = car-forward-angle - track-heading, wrapped to (-pi,pi].
        PIWM forward direction as an angle = atan2(cos yaw, -sin yaw)."""
        fwd = np.arctan2(np.cos(yaw), -np.sin(yaw))
        th = self.heading_at(s)
        return np.arctan2(np.sin(fwd - th), np.cos(fwd - th)).astype(np.float32)

    def curvature_profile(self, s, offsets):
        """kappa at s + offsets (arc-length ahead). s scalar or (N,), offsets (K,).
        Returns (K,) or (N,K)."""
        s = np.asarray(s); offsets = np.asarray(offsets)
        q = s[..., None] + offsets[None, ...] if s.ndim else s + offsets
        i = (np.mod(q, self.total_len) / self.grid_ds).astype(int) % self.M
        return self.kappa[i].astype(np.float32)


def local_road_points(centers, heading, total_len, grid_ds, s, offsets):
    """Centreline geometry ahead, expressed in the LOCAL road frame at s.

    Frame: origin at the centreline point C(s), x-axis along the track tangent
    there, y-axis = left normal (so the frame the car sees at t0, with no
    reference to where on the map it is). Returns the centreline sampled at
    arc-lengths s+offsets:

        P_j = R(-heading(s)) . (C(s + offsets_j) - C(s))         (N, K, 2)

    This is the SHAPE-space counterpart of curvature_profile: same information,
    but readable as a position without integrating twice. `centers`/`heading`
    are the persisted track grid (track.npz), so the convention matches
    kappa = d(heading)/ds and normal = (-sin h, cos h) exactly.

    s (N,) or scalar, offsets (K,). Centreline lookup is linearly interpolated
    on the grid (the nearest-index lookup used elsewhere would put a 2.5 cm
    quantisation floor on targets we care about at the centimetre level).
    """
    centers = np.asarray(centers, np.float32)
    heading = np.asarray(heading, np.float32)
    M = len(centers)
    s = np.atleast_1d(np.asarray(s, np.float32))
    offsets = np.asarray(offsets, np.float32)

    def _c(q):                       # linear interp of the closed centreline at arc-length q
        u = np.mod(q, total_len) / grid_ds
        i0 = np.floor(u).astype(int) % M
        i1 = (i0 + 1) % M
        w = (u - np.floor(u))[..., None]
        return centers[i0] * (1 - w) + centers[i1] * w

    i = (np.mod(s, total_len) / grid_ds).astype(int) % M
    th0 = heading[i]                                       # (N,)
    diff = _c(s[:, None] + offsets[None, :]) - _c(s)[:, None, :]      # (N,K,2)
    c, sn = np.cos(th0)[:, None], np.sin(th0)[:, None]
    x = diff[..., 0] * c + diff[..., 1] * sn
    y = -diff[..., 0] * sn + diff[..., 1] * c
    return np.stack([x, y], -1).astype(np.float32)


if __name__ == "__main__":
    d1 = np.load('../Data_Donkeycar/traj1_64x64.npz', allow_pickle=True)
    st = d1['state'][:].astype(np.float32); m = ~np.isnan(st).any(axis=1)
    pos = st[m, :2]; yaw = np.unwrap(st[m, 2] * np.pi / 180) - np.pi / 2
    tr = FrenetTrack(d1['map_world'], orient_xy=pos, orient_yaw=yaw)
    print(f"M={tr.M} pts, L={tr.total_len:.2f}m, flipped={tr.flipped}")
    print(f"kappa |mean|={np.abs(tr.kappa).mean():.3f} max={np.abs(tr.kappa).max():.3f}")
    s, d = tr.xy_to_sd(pos)
    psi = tr.psi_e(s, yaw)
    print(f"psi_e mean={psi.mean():.3f} std={psi.std():.3f} (small=aligned)")
    # roundtrip
    xy2 = tr.sd_to_xy(s, d)
    print(f"xy roundtrip err: mean={np.linalg.norm(xy2-pos,axis=1).mean():.4f}m")
    # kinematics
    v = st[m, 3]; dt = 1/22.
    ddot = np.gradient(d)/dt; dpred = v*np.sin(psi)
    print(f"dd/dt = v sin(psi_e): corr={np.corrcoef(dpred,ddot)[0,1]:.3f}")
