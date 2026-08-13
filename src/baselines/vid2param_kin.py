"""A faithful Vid2Param baseline for the DonkeyCar.

WHY THIS REPLACES DynamicsVid2ParamLane
Vid2Param (Asenov et al., 2019) is *system identification from video*: an RNN
over the frames produces a posterior over a small set of interpretable PHYSICAL
parameters, and a KNOWN physics model is then rolled out with them. The variant
in shared_dynamics_lane.py keeps the RNN and the KL term but breaks both halves
of that:

  1. its theta is inferred from the frozen encoder's 29-dim output, which is
     car(9) + twenty lane-waypoint dimensions, so theta can carry the shape of
     the road ahead;
  2. theta is then concatenated into three black-box MLPs that predict state
     deltas, including a 20-dim lane delta.

The result is not a system-identification baseline at all -- it is GOKU plus an
8-dimensional summary of the road, which is why it was the strongest baseline
and why it beat GokuNet here while losing badly to it on CarRacing. Comparing a
road-aware model against it and calling the gap ours is not a clean claim.

WHAT THIS DOES INSTEAD
theta is five physical constants of the vehicle, and its only route into the
prediction is through the kinematic Ackermann model that
models/dynamics_bicycle_kin.py already establishes as the right prior for this
platform at <1 m/s:

    steer_{t+1} = steer_t + (k_s * u_steer - steer_t) / tau * dt     (servo lag)
    v_{t+1}     = v_t + (k_a * u_throttle - k_d * v_t) * dt          (drive/drag)
    yaw_rate    = v_t / L * tan(steer_t)                             (Ackermann)

Because theta only sets (L, k_a, k_d, k_s, tau), road geometry has nowhere to
hide in it: there is no term in the kinematics that a road shape could modulate.
That is the point -- it makes the baseline both faithful to the published method
and honest as a comparison.

The lane dimensions are held at their initial value: Vid2Param has no road model,
and inventing one for it would repeat the mistake this file exists to fix. This
does not affect the reported metric, which is position error and reads only
z[:, :2].
"""
import torch
import torch.nn as nn
from config import FULL_STATE_DIM, DT, PHYSICS_STD_REL, PHYSICS_MEAN_REL
from lane_utils import LANE_DIM

TOTAL_DIM = FULL_STATE_DIM + LANE_DIM   # 31


def _bounded(raw, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(raw)


def _logit(value, lo, hi):
    """Raw pre-sigmoid value that makes _bounded(raw, lo, hi) == value."""
    p = min(max((value - lo) / (hi - lo), 1e-4), 1 - 1e-4)
    return float(torch.logit(torch.tensor(p)))


class Vid2ParamKin(nn.Module):
    """Posterior over vehicle parameters from observations + known kinematics."""

    THETA_DIM = 5
    OBS_DIM = 9 + LANE_DIM      # frozen-encoder output width, as for the others

    #: No road model, by design -- see the module docstring. The trainer reads
    #: this and drops the lane term, which this model cannot affect.
    PREDICTS_LANE = False

    # Physical ranges, in the working units the dataset is normalised to
    # (SPATIAL_SCALE = 50, so the donkey's true 0.165 m wheelbase is 8.25).
    L_RANGE   = (4.0, 15.0)     # wheelbase
    KA_RANGE  = (0.0, 60.0)     # throttle -> longitudinal acceleration
    KD_RANGE  = (0.0, 8.0)      # linear drag / rolling resistance, 1/s
    KS_RANGE  = (0.0, 1.2)      # steer command -> physical angle, rad
    TAU_RANGE = (0.02, 1.0)     # steering servo time constant, s

    def __init__(self, latent_dim=TOTAL_DIM, action_dim=3,
                 theta_dim=THETA_DIM, obs_dim=OBS_DIM):
        super().__init__()
        self.theta_rnn = nn.GRU(obs_dim, 64, batch_first=True)
        self.theta_head_mu = nn.Linear(64, theta_dim)
        self.theta_head_lv = nn.Linear(64, theta_dim)
        # Start the posterior nearly deterministic. With the default init the
        # sampled theta carries unit-variance noise, and because theta IS the
        # model here -- it sets the wheelbase, the drag, the steering gain --
        # that noise randomises the physics on every forward pass. The first
        # training run collapsed on exactly this: every parameter stayed at its
        # sigmoid midpoint (mu = 0, logvar = 0), giving 1.59 m at 100 steps,
        # even though moving the drag term alone reaches 0.61 m. The black-box
        # variant does not suffer from it because its theta is only an extra MLP
        # input the network can learn to discount.
        nn.init.zeros_(self.theta_head_lv.weight)
        nn.init.constant_(self.theta_head_lv.bias, -4.0)     # std ~ 0.135
        # Start the parameters at a PHYSICAL PRIOR rather than at the midpoint of
        # each range. This is not tuning, it is what makes the problem solvable:
        # at the midpoints (k_d = 4, k_a = 30) the equilibrium speed is 4.3
        # working units/s against a measured 24.4, and the velocity decays by
        # (1 - k_d*dt)^32 = 0.0016 across the training rollout. The car stops
        # within a few steps, so the gradient reaching the parameters that govern
        # the decay vanishes with it and training never moves: the first two runs
        # sat at train loss 6.00 for all 60 epochs with every parameter still on
        # its midpoint. Seeded here at k_d = 0.5, k_a = 22 the equilibrium speed
        # is 24.9, i.e. the right regime, and the rollout retains gradient.
        nn.init.zeros_(self.theta_head_mu.weight)
        with torch.no_grad():
            self.theta_head_mu.bias.copy_(torch.tensor(
                [_logit(8.25, *self.L_RANGE),    # wheelbase: the donkey's true 0.165 m
                 _logit(22.0, *self.KA_RANGE),   # with k_d below, gives v_eq ~ 24.9
                 _logit(0.5,  *self.KD_RANGE),
                 _logit(0.6,  *self.KS_RANGE),
                 _logit(0.45, *self.TAU_RANGE)]))  # ~10-frame servo lag
        self.register_buffer("mean", torch.tensor(PHYSICS_MEAN_REL, dtype=torch.float32))
        self.register_buffer("std",  torch.tensor(PHYSICS_STD_REL,  dtype=torch.float32))

    # --- system identification -------------------------------------------
    def infer_theta(self, obs_hist):
        """obs_hist: (B, T, obs_dim) -> (theta, mu, logvar), same interface as
        the other observation-conditioned baselines so they share a trainer."""
        out, _ = self.theta_rnn(obs_hist)
        h = out[:, -1]
        mu = self.theta_head_mu(h); lv = self.theta_head_lv(h)
        theta = mu + (0.5 * lv).exp() * torch.randn_like(mu)
        return theta, mu, lv

    def physical_params(self, theta):
        """theta (B,5) -> the five constants, each bounded to a physical range.
        Exposed so the identified parameters can be inspected -- being able to
        read them off is the whole appeal of this method."""
        return (_bounded(theta[:, 0], *self.L_RANGE),
                _bounded(theta[:, 1], *self.KA_RANGE),
                _bounded(theta[:, 2], *self.KD_RANGE),
                _bounded(theta[:, 3], *self.KS_RANGE),
                _bounded(theta[:, 4], *self.TAU_RANGE))

    # --- known-physics rollout -------------------------------------------
    def step(self, z_norm, action, theta):
        L, k_a, k_d, k_s, tau = self.physical_params(theta)

        car = z_norm[:, :FULL_STATE_DIM] * self.std + self.mean
        lane = z_norm[:, FULL_STATE_DIM:]                 # no road model: held
        x, y, yaw = car[:, 0], car[:, 1], car[:, 2]
        vx, vy = car[:, 3], car[:, 4]
        wheels, steer = car[:, 6:10], car[:, 10]
        u_steer, u_thr = action[:, 0], action[:, 1]

        # PIWM convention: body +y is forward.
        c, s = torch.cos(yaw), torch.sin(yaw)
        v_fwd = -vx * s + vy * c

        steer_next = steer + (k_s * u_steer - steer) / tau * DT
        v_next = v_fwd + (k_a * u_thr - k_d * v_fwd) * DT
        yaw_rate = v_fwd / (L + 1e-3) * torch.tan(steer)
        yaw_next = yaw + yaw_rate * DT

        c_n, s_n = torch.cos(yaw_next), torch.sin(yaw_next)
        vx_next, vy_next = -v_next * s_n, v_next * c_n
        x_next, y_next = x + vx * DT, y + vy * DT          # forward Euler

        car_next = torch.cat([
            torch.stack([x_next, y_next, yaw_next, vx_next, vy_next, yaw_rate], -1),
            wheels, steer_next.unsqueeze(-1)], dim=-1)
        return torch.cat([(car_next - self.mean) / self.std, lane], dim=-1)

    def forward(self, z, a, theta):
        return self.step(z, a, theta)
