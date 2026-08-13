"""Frenet world-model dynamics for the fixed donkey track.

State z = [s, d, psi_e, v, omega] in REAL units (metres / rad / rad-per-s).
Road geometry kappa(s) is a DIFFERENTIABLE lookup on the fixed track grid — it
is NOT a predicted state, so it cannot drift over a long rollout. This is the
structural fix for GOKU's win (whose 20 free lane waypoints drift and corrupt
the shared dynamics input).

Per step:
  kappa = kappa_lookup(s)                          (differentiable in s)
  den   = 1 - d*kappa            (clamped away from 0)
  s'    = s    + v*cos(psi)/den * dt               analytic
  d'    = d    + v*sin(psi)       * dt + res_d      analytic + tiny residual
  psi'  = psi  + (omega - kappa*v*cos(psi)/den)*dt + res_psi
  v'    = v    + dv_net(feat)                       learned (throttle -> accel)
  omega'= omega+ dom_net(feat)                      learned (steer -> yaw accel)

Only dv/dom and a small (res_d,res_psi) are learned; the pose kinematics is
analytic and was validated on data (lateral relation corr=0.96). Every state
dim is physically interpretable.
"""
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, i, o, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))
    def forward(self, x): return self.net(x)


class FrenetDynamics(nn.Module):
    def __init__(self, track_npz, stats_npz, dt=1.0 / 22.0):
        super().__init__()
        tr = np.load(track_npz)
        self.M = int(len(tr['kappa']))
        self.register_buffer("kappa_grid", torch.tensor(tr['kappa'], dtype=torch.float32))
        self.total_len = float(tr['total_len'])
        self.grid_ds = float(tr['grid_ds'])
        self.dt = float(dt)

        st = np.load(stats_npz)
        # state order [s,d,psi,v,omega]; s uses a fixed loss scale (0.5 m)
        sm = st['state_mean'].astype(np.float32); ss = st['state_std'].astype(np.float32)
        self.register_buffer("state_mean", torch.tensor(sm))
        self.register_buffer("state_std",  torch.tensor(ss))
        self.register_buffer("kappa_mean", torch.tensor(float(st['kappa_mean'])))
        self.register_buffer("kappa_std",  torch.tensor(float(st['kappa_std'])))

        # feature for the learned heads: [d_n, psi_n, v_n, om_n, kappa_n, action(2)]
        feat_dim = 5 + 2
        self.dv_net  = MLP(feat_dim, 1, h=64)
        self.dom_net = MLP(feat_dim, 1, h=64)
        # TIGHT residual on (res_d, res_psi) only, fixed small scale 0.1.
        # A larger / s-including residual was tested and traded away the
        # bounded long-horizon stability that is Frenet's whole advantage
        # (xy@100 0.246 -> 0.329), so we keep the prior strong.
        self.res_net = MLP(feat_dim, 2, h=64)

    def kappa_at(self, s):
        """Differentiable linear interpolation of kappa at arc-length s (B,)."""
        u = torch.remainder(s, self.total_len) / self.grid_ds        # [0, M)
        i0 = torch.floor(u)
        frac = u - i0
        i0 = i0.long() % self.M
        i1 = (i0 + 1) % self.M
        return self.kappa_grid[i0] * (1 - frac) + self.kappa_grid[i1] * frac

    def _norm_d(self, d):    return (d - self.state_mean[1]) / self.state_std[1]
    def _norm_psi(self, p):  return (p - self.state_mean[2]) / self.state_std[2]
    def _norm_v(self, v):    return (v - self.state_mean[3]) / self.state_std[3]
    def _norm_om(self, o):   return (o - self.state_mean[4]) / self.state_std[4]

    def forward(self, z, action, kappa_override=None):
        """kappa_override (B,): if given, use this PERCEIVED curvature-at-car instead
        of the known-map lookup kappa_at(s). This is the switch from privileged-map
        mode to pure-world-model mode (kappa from the camera, see road_perception.py)."""
        s, d, psi, v, om = z[:, 0], z[:, 1], z[:, 2], z[:, 3], z[:, 4]
        kap = self.kappa_at(s) if kappa_override is None else kappa_override

        den = 1.0 - d * kap
        den = torch.where(den.abs() < 0.2, torch.sign(den + 1e-6) * 0.2, den)
        cospsi = torch.cos(psi); sinpsi = torch.sin(psi)
        sdot = v * cospsi / den
        ddot = v * sinpsi
        psidot = om - kap * v * cospsi / den

        # learned increments for v, omega + small pose residual
        feat = torch.stack([self._norm_d(d), self._norm_psi(psi), self._norm_v(v),
                            self._norm_om(om), (kap - self.kappa_mean) / self.kappa_std],
                           dim=-1)
        feat = torch.cat([feat, action], dim=-1)
        dv = self.dv_net(feat).squeeze(-1)
        dom = self.dom_net(feat).squeeze(-1)
        res = self.res_net(feat) * 0.1               # (B,2): d, psi

        s_n   = s   + sdot * self.dt                 # s: pure analytic (no residual)
        d_n   = d   + ddot * self.dt + res[:, 0]
        psi_n = psi + psidot * self.dt + res[:, 1]
        v_n   = v   + dv
        om_n  = om  + dom

        return torch.stack([s_n, d_n, psi_n, v_n, om_n], dim=-1)

    @staticmethod
    def _interp1d(xp, fp, x):
        """Linear interp of fp(xp) at scalar tensor x; xp ascending. Clamped (hold-last)."""
        x = x.clamp(xp[0], xp[-1])
        i = torch.searchsorted(xp, x).clamp(1, len(xp) - 1)
        x0, x1, y0, y1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
        w = (x - x0) / (x1 - x0 + 1e-9)
        return y0 + w * (y1 - y0)

    @torch.no_grad()
    def rollout_perceived(self, z0, actions, kappa_profile0, offsets):
        """Pure-world-model rollout: kappa comes from a profile PERCEIVED once at t0,
        NOT the map. As the car advances by delta = s_k - s0 (arc-length), kappa at the
        car is interpolated from that already-observed preview (hold-last past its reach
        -- the structured road-context transition is just the shift of the perceived
        profile by the predicted advance).

          z0 (5,), actions (K,2), kappa_profile0 (n_off,) = perceived kappa(s0+offsets),
          offsets (n_off,) arc-length samples in metres. Returns (K+1, 5).
        """
        dev = z0.device
        offs = torch.as_tensor(offsets, dtype=torch.float32, device=dev)
        prof = torch.as_tensor(kappa_profile0, dtype=torch.float32, device=dev)
        s0 = z0[0].clone()
        z = z0.unsqueeze(0)
        out = [z0.clone()]
        for k in range(len(actions)):
            delta = torch.remainder(z[0, 0] - s0, self.total_len)
            kap = self._interp1d(offs, prof, delta).reshape(1)
            z = self.forward(z, actions[k:k + 1], kappa_override=kap)
            out.append(z[0].clone())
        return torch.stack(out)

    @torch.no_grad()
    def rollout_perceived_local(self, z0, actions, kappa_profile0, offsets):
        """Map-FREE rollout: returns the vehicle pose relative to its own start,
        reconstructed entirely from the perceived curvature profile.

        WHY THIS EXISTS
        `rollout_perceived` already takes kappa from the camera rather than the
        map, but it returns the state as (s, d): arc length along a *known*
        centreline and lateral offset from it. Turning that into a position still
        needs the track file, so the model was only map-free in its dynamics, not
        in its state parameterization -- it presupposed a surveyed track. That is
        a real generality limit, and it is also an asymmetry against the
        baselines, which predict a track-agnostic body-relative displacement.

        The fix is that the perceived profile is *already enough* to rebuild the
        road locally. With kappa(sigma) known along the path the vehicle takes,
            theta_road(sigma) = integral of kappa,
            centreline(sigma) = integral of (cos theta_road, sin theta_road),
        and the vehicle sits at centreline(sigma) + d * normal(sigma). So we
        integrate the road forward alongside the vehicle and read the pose off it.
        Nothing here touches self.kappa_grid or the centreline.

        Frame: origin at the centreline point beneath the vehicle at t0, x-axis
        along the road tangent there (theta_road(0) = 0). Matches the track
        convention normal = (-sin theta, cos theta).

          z0 (5,), actions (K,2), kappa_profile0 (n_off,), offsets (n_off,)
          returns (K+1, 3) of (x, y, psi_vehicle) in metres / radians.
        """
        dev = z0.device
        offs = torch.as_tensor(offsets, dtype=torch.float32, device=dev)
        prof = torch.as_tensor(kappa_profile0, dtype=torch.float32, device=dev)
        s0 = z0[0].clone()

        z = z0.unsqueeze(0)
        theta = torch.zeros((), device=dev)          # road heading, relative to t0
        cx = torch.zeros((), device=dev)             # centreline point, local frame
        cy = torch.zeros((), device=dev)

        def _pose(th, cx, cy, d, psi_e):
            # vehicle = centreline + d * left-normal ; heading = road + heading error
            return torch.stack([cx - d * torch.sin(th),
                                cy + d * torch.cos(th),
                                th + psi_e])

        out = [_pose(theta, cx, cy, z0[1], z0[2])]
        for k in range(len(actions)):
            s_prev = z[0, 0].clone()
            delta = torch.remainder(z[0, 0] - s0, self.total_len)
            kap = self._interp1d(offs, prof, delta).reshape(1)
            z = self.forward(z, actions[k:k + 1], kappa_override=kap)
            ds = z[0, 0] - s_prev                    # arc length covered this step
            # advance the road with the same Euler step the dynamics used
            cx = cx + torch.cos(theta) * ds
            cy = cy + torch.sin(theta) * ds
            theta = theta + kap[0] * ds
            out.append(_pose(theta, cx, cy, z[0, 1], z[0, 2]))
        return torch.stack(out)

    @torch.no_grad()
    def rollout_perceived_shape(self, z0, actions, kappa_profile0, road_pts0, offsets):
        """Map-FREE rollout that READS the road shape instead of integrating it.

        Same contract as `rollout_perceived_local` -- pose relative to the start,
        nothing from the map -- but the local road comes from the perception's
        SHAPE head (centreline points at the preview arc-lengths, in the road
        frame at t0) rather than from integrating the curvature profile twice.
        That double integration, not perception, is where the map-free penalty
        came from: with a perfect kappa profile the integrated version still
        lost 0.15 m at 100 steps.

        Per step: advance the Frenet state with perceived kappa (the ODE needs
        curvature at the car), then place the vehicle by interpolating the
        perceived centreline at the arc length covered so far:
            pose = P(delta) + d * normal(delta),  heading = theta(delta) + psi_e
        where theta is the polyline's tangent angle. Past the preview reach the
        interpolation holds the last knot, exactly as the kappa version does.

          z0 (5,), actions (K,2), kappa_profile0 (n_off,),
          road_pts0 (n_off,2) = perceived (X, Y) at s0+offsets, offsets (n_off,)
          returns (K+1, 3) of (x, y, psi_vehicle) in metres / radians.
        """
        dev = z0.device
        offs = torch.as_tensor(offsets, dtype=torch.float32, device=dev)
        prof = torch.as_tensor(kappa_profile0, dtype=torch.float32, device=dev)
        P = torch.as_tensor(road_pts0, dtype=torch.float32, device=dev)     # (n_off,2)
        h = (offs[1] - offs[0]).clamp(min=1e-6)          # knots are uniform in arc length

        # Catmull-Rom tangents. Straight-line interpolation between 0.5 m knots is
        # not good enough here: the tightest corner has R = 0.31 m, so the chord
        # cuts 10 cm off the arc. The cubic drops the readout floor from 3.3 cm to
        # 1.6 cm mean (4.3 -> 2.2 cm at 100 steps).
        m = torch.zeros_like(P)
        m[1:-1] = (P[2:] - P[:-2]) / 2
        m[0] = P[1] - P[0]
        m[-1] = P[-1] - P[-2]
        # knot tangent angles, made continuous, so the spline angle can be
        # unwrapped onto the right branch (the 4.5 m preview can turn > pi)
        dP = torch.zeros_like(P)
        dP[1:-1] = P[2:] - P[:-2]
        dP[0] = P[1] - P[0]
        dP[-1] = P[-1] - P[-2]
        a = torch.atan2(dP[:, 1], dP[:, 0])
        step = torch.remainder(a[1:] - a[:-1] + torch.pi, 2 * torch.pi) - torch.pi
        thk = torch.cat([a[:1], a[:1] + torch.cumsum(step, 0)])
        thk = thk - thk[0]                    # frame convention: theta_road(0) = 0

        n = len(P)
        s0 = z0[0].clone()
        z = z0.unsqueeze(0)

        def _pose(delta, d, psi_e):
            dl = delta.clamp(offs[0], offs[-1])          # hold-last past the preview
            j = ((dl - offs[0]) / h).floor().long().clamp(0, n - 2)
            u = (dl - offs[j]) / h
            u2, u3 = u * u, u * u * u
            pos = ((2*u3 - 3*u2 + 1) * P[j] + (u3 - 2*u2 + u) * m[j]
                   + (-2*u3 + 3*u2) * P[j + 1] + (u3 - u2) * m[j + 1])
            dv = ((6*u2 - 6*u) * P[j] + (3*u2 - 4*u + 1) * m[j]
                  + (-6*u2 + 6*u) * P[j + 1] + (3*u2 - 2*u) * m[j + 1])
            aa = torch.atan2(dv[1], dv[0])
            t = thk[j] + (torch.remainder(aa - thk[j] + torch.pi, 2 * torch.pi) - torch.pi)
            return torch.stack([pos[0] - d * torch.sin(t), pos[1] + d * torch.cos(t),
                                t + psi_e])

        zero = torch.zeros((), device=dev)
        out = [_pose(zero, z0[1], z0[2])]
        for k in range(len(actions)):
            delta = torch.remainder(z[0, 0] - s0, self.total_len)
            kap = self._interp1d(offs, prof, delta).reshape(1)
            z = self.forward(z, actions[k:k + 1], kappa_override=kap)
            delta = torch.remainder(z[0, 0] - s0, self.total_len)
            out.append(_pose(delta, z[0, 1], z[0, 2]))
        return torch.stack(out)

    # --- loss helper: per-dim normalized error (s uses fixed 0.5 m scale) ---
    def state_loss(self, z_pred, z_gt):
        scale = torch.tensor([0.5, self.state_std[1].item(), self.state_std[2].item(),
                              self.state_std[3].item(), self.state_std[4].item()],
                             device=z_pred.device)
        # wrap psi error to (-pi,pi]; wrap s error to (-L/2, L/2]
        e = z_pred - z_gt
        e0 = torch.remainder(e[:, 0] + self.total_len / 2, self.total_len) - self.total_len / 2
        e2 = torch.atan2(torch.sin(e[:, 2]), torch.cos(e[:, 2]))
        e = torch.stack([e0, e[:, 1], e2, e[:, 3], e[:, 4]], dim=-1)
        return ((e / scale) ** 2).mean(dim=0)        # (5,) per-dim
