"""Bridge between the Frenet latent [s,d,psi_e,v,omega] (real metres) and the
existing v6 29-dim observable [car_obs(9) | lane_wp(20)] (x50 working units,
ref-relative, normalized) — so the EXISTING PhysicsEncoderLane / PhysicsDecoderLane
can be reused unchanged.

Conventions recovered from the data pipeline:
  - PIWM yaw: forward direction = (-sin yaw, cos yaw); forward-angle = yaw + pi/2.
  - Frenet psi_e = forward-angle - track_heading  =>  yaw = track_heading + psi_e - pi/2.
  - 31-dim stats (PHYSICS_*_REL) and lane stats are in x50 working units.
"""
import numpy as np
from config import PHYSICS_MEAN_REL, PHYSICS_STD_REL
from lane_utils import LANE_MEAN, LANE_STD, N_LANE_WP, LANE_S_SAMPLES
from donkey_config import DONKEY_SPATIAL_SCALE as SCALE, DONKEY_WHEELBASE_PHYS as WB

CAR_IDX = [2, 3, 4, 5, 6, 7, 8, 9, 10]                 # car_obs dims in 11-dim phys
ENC_MEAN = PHYSICS_MEAN_REL[CAR_IDX]; ENC_STD = PHYSICS_STD_REL[CAR_IDX]
# lane waypoint arc-length offsets used by the v6 decoder (x50 working units -> metres)
LANE_OFF_M = np.asarray(LANE_S_SAMPLES, dtype=np.float32) / SCALE       # [0,0.5,..,4.5] m


class FrenetBridge:
    def __init__(self, track_npz):
        tr = np.load(track_npz)
        self.cen = tr['centers']; self.nrm = tr['normals']; self.head = tr['heading']
        self.L = float(tr['total_len']); self.ds = float(tr['grid_ds']); self.M = len(self.cen)

    def _idx(self, s): return (np.mod(s, self.L) / self.ds).astype(int) % self.M
    def sd_to_xy(self, s, d):
        i = self._idx(s); return self.cen[i] + np.asarray(d)[..., None] * self.nrm[i]
    def heading_at(self, s): return self.head[self._idx(s)]

    def to_decoder_input(self, fseq):
        """(T,5) Frenet seq -> (T,29) decoder input (ref=frame 0, normalized).
        Reproduces the v6 representation: car_obs ref-relative + lane_wp in ref body frame.
        """
        s, d, psi, v, om = [fseq[:, k] for k in range(5)]
        T = len(s)
        yaw = self.heading_at(s) + psi - np.pi / 2                      # absolute yaw
        pos = self.sd_to_xy(s, d)                                       # (T,2) metres
        yaw0 = yaw[0]; pos0 = pos[0]
        c0, s0 = np.cos(yaw0), np.sin(yaw0)

        # --- car_obs(9): [yaw_rel, vx_rel, vy_rel, omega, w0..3, steer] in x50 units ---
        yaw_rel = np.arctan2(np.sin(yaw - yaw0), np.cos(yaw - yaw0))
        vx_w = -v * np.sin(yaw); vy_w = v * np.cos(yaw)                 # world vel (m/s)
        vx_rel = (c0 * vx_w + s0 * vy_w) * SCALE                        # rotate to ref, x50
        vy_rel = (-s0 * vx_w + c0 * vy_w) * SCALE
        wheels = (v * SCALE)[:, None].repeat(4, 1)
        v_safe = np.where(np.abs(v) < 0.05, 0.05, np.abs(v)) * np.sign(v + 1e-9)
        steer = np.clip(np.arctan(om * WB / v_safe), -0.6, 0.6)
        car = np.stack([yaw_rel, vx_rel, vy_rel, om, wheels[:, 0], wheels[:, 1],
                        wheels[:, 2], wheels[:, 3], steer], axis=-1)
        car_n = (car - ENC_MEAN) / ENC_STD

        # --- lane_wp(20): centerline ahead of car_t, in REF body frame, x50 ---
        lane = np.zeros((T, N_LANE_WP, 2), np.float32)
        for t in range(T):
            wpw = self.sd_to_xy(s[t] + LANE_OFF_M, np.zeros(N_LANE_WP))   # world metres
            rel = (wpw - pos0) * SCALE                                    # ref-translated, x50
            x = c0 * rel[:, 0] + s0 * rel[:, 1]
            y = -s0 * rel[:, 0] + c0 * rel[:, 1]
            lane[t] = np.stack([x, y], -1)
        lane_n = (lane.reshape(T, -1) - LANE_MEAN) / LANE_STD
        return np.concatenate([car_n, lane_n], -1).astype(np.float32)
