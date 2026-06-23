"""Donkey-specific dataset utilities.

Adds two pieces on top of the CarRacing `SeqLaneDataset`:

1. **Segment-level train/val split** — the base dataset's `random_split` slices
   the sample-index list, which leaks data across the split because adjacent
   indices share frames (sample i covers frames [t, t+8], sample i+1 covers
   [t+1, t+9] — 8/9 frames in common). For donkey we have 72 clean segments
   from glitch filtering; we hold out whole segments instead.

2. **Horizontal-flip augmentation** — driving data is heavily left-biased
   (mean steer command -0.30 on both trajectories). Mirror-flipping each
   sample with probability 0.5 makes the dataset left/right symmetric,
   doubling the effective coverage. All affected quantities are sign-flipped
   consistently in PIWM's body frame (body+x = lateral, body+y = forward):

       imgs:      flip width axis
       phys[0]:   x_rel        flip sign
       phys[1]:   y_rel        keep
       phys[2]:   yaw_rel      flip sign
       phys[3]:   vx_rel       flip sign     (lateral world component)
       phys[4]:   vy_rel       keep          (forward world component)
       phys[5]:   omega        flip sign
       phys[6..9]: wheels      keep          (forward speed magnitude)
       phys[10]:  steer        flip sign
       wp[2k]:    waypoint x   flip sign
       wp[2k+1]:  waypoint y   keep
       action[0]: steer_cmd    flip sign
       action[1]: throttle     keep
       action[2]: brake        keep

   Flipping happens in NORMALIZED space using
       x_norm_flipped = -x_norm - 2 * (mean / std)
   so the offset constants depend on `config.PHYSICS_MEAN_REL`,
   `config.PHYSICS_STD_REL`, `lane_utils.LANE_MEAN`, `lane_utils.LANE_STD` —
   those must already have been patched to donkey values before this module
   is imported.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from config import PHYSICS_MEAN_REL, PHYSICS_STD_REL
from lane_utils import LANE_MEAN, LANE_STD, LANE_DIM
from train_piwm_lane_v5 import SeqLaneDataset


# Phys-dim flip rules: indices to sign-flip in the 11-dim normalized rel vector.
_PHYS_FLIP_DIMS = (0, 2, 3, 5, 10)        # x_rel, yaw_rel, vx_rel, omega, steer


def _phys_flip_offset():
    """Per-dim offset c such that flipped_norm = -orig_norm + c.

    From flipped_norm = (-orig_unnorm - μ)/σ = -orig_norm - 2μ/σ, so c = -2μ/σ.
    Only the dims in _PHYS_FLIP_DIMS are non-zero.
    """
    off = np.zeros(11, dtype=np.float32)
    for d in _PHYS_FLIP_DIMS:
        off[d] = -2.0 * PHYSICS_MEAN_REL[d] / PHYSICS_STD_REL[d]
    return torch.tensor(off, dtype=torch.float32)


def _lane_flip_offset():
    """Per-element offset for the 20-dim normalized lane vector. Only x-cols
    (even indices) get a non-zero offset."""
    off = np.zeros(LANE_DIM, dtype=np.float32)
    for k in range(LANE_DIM // 2):
        ix = 2 * k                                   # waypoint k's x dim
        off[ix] = -2.0 * LANE_MEAN[ix] / LANE_STD[ix]
    return torch.tensor(off, dtype=torch.float32)


def _phys_flip_sign():
    """Per-dim sign multiplier: -1 for dims that flip, +1 for dims that keep."""
    sgn = torch.ones(11, dtype=torch.float32)
    for d in _PHYS_FLIP_DIMS:
        sgn[d] = -1.0
    return sgn


def _lane_flip_sign():
    sgn = torch.ones(LANE_DIM, dtype=torch.float32)
    for k in range(LANE_DIM // 2):
        sgn[2 * k] = -1.0
    return sgn


_ACTION_FLIP_SIGN = torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float32)


class FlipAugDataset(Dataset):
    """Wrap a base dataset; apply horizontal flip with probability p."""

    def __init__(self, base, p=0.5, seed=None):
        self.base = base
        self.p = float(p)
        # Cache flip constants. PHYSICS_MEAN_REL / LANE_MEAN must already be
        # set to donkey values via donkey_config.patch_globals().
        self.phys_sgn = _phys_flip_sign()                            # (11,)
        self.phys_off = _phys_flip_offset()                          # (11,)
        self.lane_sgn = _lane_flip_sign()                            # (20,)
        self.lane_off = _lane_flip_offset()                          # (20,)
        self.act_sgn  = _ACTION_FLIP_SIGN                            # (3,)
        # Sample-level RNG so flips are reproducible inside a worker.
        self._rng = np.random.default_rng(seed)

    def __len__(self): return len(self.base)

    def __getitem__(self, i):
        stack0, fut_imgs, fut_phys, fut_acts, wp_norm = self.base[i]
        if self._rng.random() >= self.p:
            return stack0, fut_imgs, fut_phys, fut_acts, wp_norm

        # Flip images along width (last dim). stack0: (FS, H, W);
        # fut_imgs: (seq_len, 1, H, W).
        stack0   = torch.flip(stack0,   dims=[-1])
        fut_imgs = torch.flip(fut_imgs, dims=[-1])

        # Flip phys: (seq_len, 11)
        fut_phys = fut_phys * self.phys_sgn + self.phys_off

        # Flip waypoints: (seq_len, 20)
        wp_norm = wp_norm * self.lane_sgn + self.lane_off

        # Flip action: (seq_len-1, 3)
        fut_acts = fut_acts * self.act_sgn

        return stack0, fut_imgs, fut_phys, fut_acts, wp_norm


def make_donkey_loaders(data_dir, seq_len, batch_size,
                        val_frac=0.10, flip_aug=True, seed=0,
                        num_workers=0):
    """Build train/val DataLoaders with segment-level holdout.

    Returns: (train_loader, val_loader, base_ds_for_introspection).

    Reproducibility: the same `seed` always yields the same segment split.
    """
    base = SeqLaneDataset(data_dir, seq_len=seq_len)
    n_eps = len(base.imgs_list)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_eps)
    n_val_eps = max(1, int(round(n_eps * val_frac)))
    val_eps   = set(perm[:n_val_eps].tolist())
    train_eps = set(perm[n_val_eps:].tolist())
    train_idx = [i for i, (ep, _) in enumerate(base.indices) if ep in train_eps]
    val_idx   = [i for i, (ep, _) in enumerate(base.indices) if ep in val_eps]
    print(f"  segments: {n_eps} total  ->  {len(train_eps)} train / {len(val_eps)} val")
    print(f"  samples : {len(train_idx)} train / {len(val_idx)} val")

    train_ds = Subset(base, train_idx)
    val_ds   = Subset(base, val_idx)
    if flip_aug:
        train_ds = FlipAugDataset(train_ds, p=0.5, seed=seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=num_workers)
    return train_loader, val_loader, base


# ----------------------------------------------------------------------
# Segment-level split helpers shared with baseline scripts (SINDYc etc.)
# ----------------------------------------------------------------------
def split_segments(base, val_frac=0.10, seed=0):
    """Return (train_ep_set, val_ep_set) for a SeqLaneDataset `base`, using
    the SAME deterministic split as `make_donkey_loaders`.
    """
    n_eps = len(base.imgs_list)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_eps)
    n_val = max(1, int(round(n_eps * val_frac)))
    return set(perm[n_val:].tolist()), set(perm[:n_val].tolist())


def collect_state31_windows(base, ep_set, window=50, stride=25, with_flip=False):
    """Slide windows over each episode in `ep_set`, emitting normalized
    state31 arrays for SINDYc / numpy-based methods.

    Each window has its own relative-coord reference (frame 0 of the window)
    so yaw_rel, x_rel, y_rel stay within their PIWM normalization range no
    matter how long the underlying segment is. Mirrors what the NN baselines
    see via SeqLaneDataset (which always uses ref_idx=0 of the sub-sequence).

    Returns: list of (state31, action) tuples, both float32.
      state31: (window, 31)
      action : (window, 3)   raw command, not normalized

    If `with_flip` is True, every window also appears mirrored.
    """
    from relative_coords import to_relative_np
    out = []
    phys_sgn_np = _phys_flip_sign().numpy()
    phys_off_np = _phys_flip_offset().numpy()
    lane_sgn_np = _lane_flip_sign().numpy()
    lane_off_np = _lane_flip_offset().numpy()
    act_sgn_np  = _ACTION_FLIP_SIGN.numpy()

    for ep in ep_set:
        phys = base.phys_list[ep]                  # (T, 11)
        acts = base.acts_list[ep]                  # (T, 3)
        wp_world = base.wp_world_list[ep]          # (T, 10, 2)
        T = len(phys)
        if T < window: continue
        for t0 in range(0, T - window + 1, stride):
            seg_phys = phys[t0:t0 + window]
            seg_acts = acts[t0:t0 + window]
            rel = to_relative_np(seg_phys, ref_idx=0)
            car_norm = (rel - PHYSICS_MEAN_REL) / PHYSICS_STD_REL
            pos0 = seg_phys[0, 0:2]; yaw0 = seg_phys[0, 2]
            c, s = np.cos(yaw0), np.sin(yaw0)
            d = wp_world[t0:t0 + window] - pos0
            x =  d[..., 0] * c + d[..., 1] * s
            y = -d[..., 0] * s + d[..., 1] * c
            wp_ref = np.stack([x, y], axis=-1).reshape(window, LANE_DIM)
            lane_norm = (wp_ref - LANE_MEAN) / LANE_STD
            state31 = np.concatenate([car_norm, lane_norm], axis=-1).astype(np.float32)
            out.append((state31, seg_acts.astype(np.float32)))
            if with_flip:
                s_f = state31.copy()
                s_f[:, :11] = state31[:, :11] * phys_sgn_np + phys_off_np
                s_f[:, 11:] = state31[:, 11:] * lane_sgn_np + lane_off_np
                a_f = seg_acts * act_sgn_np
                out.append((s_f.astype(np.float32), a_f.astype(np.float32)))
    return out
