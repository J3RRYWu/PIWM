"""Train baseline Dynamics with shared PIWM encoder/decoder (frozen).

Following the original PIWM paper: all dynamics variants operate on PIWM's
physics latent. We train only the dynamics module here.

Variants:
  - DVBF (unsupervised-style, no physics prior)
  - GOKU (kinematics + learned force, similar to PIWM but lighter)
  - Vid2Param (theta-parameterized physics)
"""

import os, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from config import (DEVICE, DATA_DIR, PHYSICS_MEAN_REL, PHYSICS_STD_REL, FRAME_STACK,
                    LAMBDA_RESIDUAL, ENCODER_STATE_INDICES)
from relative_coords import to_relative_np
from baselines.shared_dynamics import DynamicsDVBF, DynamicsGOKU, DynamicsVid2Param
from models.encoder import PhysicsEncoder
from utils import save_checkpoint, load_checkpoint


ROLLOUT_K = 8
EPOCHS = 60
THETA_HIST = 4  # for Vid2Param: use first 4 states to infer theta


class SeqDataset(Dataset):
    """Dataset for shared-encoder baselines.

    Returns (for each sample):
      - hist_stacks: (THETA_HIST, FRAME_STACK, 64, 64) image stacks for theta inference (V2P only)
      - fut_norm:    (seq_len, 11) GT future physics (for dynamics loss)
      - fut_acts:    (seq_len-1, 3) actions
    Non-V2P methods can ignore hist_stacks.
    """
    def __init__(self, data_dir, seq_len=ROLLOUT_K + 1, need_images=False):
        files = sorted(glob.glob(f"{data_dir}/*.npz"))
        self.seq_len = seq_len
        self.need_images = need_images
        self.imgs_list, self.phys_list, self.acts_list = [], [], []
        self.indices = []
        for ep_idx, f in enumerate(files):
            try:
                d = np.load(f, allow_pickle=True)
                pos = d["position"].astype(np.float32)
                yaw = d["yaw"].astype(np.float32)
                vel = d["velocity"].astype(np.float32)
                omega = d["angular_velocity"].astype(np.float32)
                wheel = d["wheel_omega"].astype(np.float32)
                steer = d["steering_angle"].astype(np.float32)
                acts = d["action"].astype(np.float32)
                phys = np.column_stack([pos, yaw, vel, omega, wheel, steer])
                n = len(phys)
                # Need enough prefix to build THETA_HIST frame stacks at ref t0
                # The earliest stack is centered at t0 - THETA_HIST + 1, requiring
                # frames [t0 - THETA_HIST + 1 - (FRAME_STACK - 1), ..., t0 - THETA_HIST + 1]
                # So we need t0 >= THETA_HIST + FRAME_STACK - 2
                if n < seq_len + THETA_HIST + FRAME_STACK: continue
                imgs = None
                if need_images:
                    imgs = d["imgs"].astype(np.float32)
                    if imgs.max() > 1.0: imgs /= 255.0
                self.imgs_list.append(imgs); self.phys_list.append(phys); self.acts_list.append(acts)
                t_start = THETA_HIST + FRAME_STACK - 2
                for t in range(t_start, n - seq_len):
                    self.indices.append((ep_idx, t))
            except: pass
        print(f"SeqDataset (images={need_images}): {len(self.indices)} samples")

    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        ep, t0 = self.indices[i]
        phys = self.phys_list[ep]; acts = self.acts_list[ep]
        # Future rollout: [t0, t0+seq_len]
        fut_phys = phys[t0:t0 + self.seq_len]
        fut_acts = acts[t0:t0 + self.seq_len - 1]
        rel_fut = to_relative_np(fut_phys, ref_idx=0)
        fut_norm = (rel_fut - PHYSICS_MEAN_REL) / PHYSICS_STD_REL

        if self.need_images:
            # History frame stacks for theta inference via encoder
            # For each of THETA_HIST time steps ending at t0, build a (FRAME_STACK, 64, 64) stack.
            imgs = self.imgs_list[ep]
            hist_stacks = []
            for k in range(THETA_HIST):
                center = t0 - THETA_HIST + 1 + k
                stack = np.stack([imgs[center - FRAME_STACK + 1 + j]
                                  for j in range(FRAME_STACK)], axis=0)
                hist_stacks.append(stack)
            hist_stacks = np.stack(hist_stacks, axis=0)  # (THETA_HIST, FRAME_STACK, 64, 64)
            return (torch.tensor(hist_stacks, dtype=torch.float32),
                    torch.tensor(fut_norm, dtype=torch.float32),
                    torch.tensor(fut_acts, dtype=torch.float32))
        else:
            # Dummy zero tensor for compatibility with DVBF/GOKU loops
            dummy = torch.zeros(1)
            return (dummy,
                    torch.tensor(fut_norm, dtype=torch.float32),
                    torch.tensor(fut_acts, dtype=torch.float32))


def train_dvbf():
    print("=" * 60); print("Train DynamicsDVBF on PIWM latent"); print("=" * 60)
    ds = SeqDataset(DATA_DIR)
    n_val = int(len(ds) * 0.1)
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=128, shuffle=False, drop_last=True)

    model = DynamicsDVBF().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')

    for epoch in range(EPOCHS):
        model.train(); tl, nb = 0, 0
        for hist, fut, acts in tqdm(tr_loader, desc=f"DVBF {epoch+1}/{EPOCHS}"):
            fut, acts = fut.to(DEVICE), acts.to(DEVICE)
            z = fut[:, 0]; ls = 0
            for k in range(ROLLOUT_K):
                z_next = model(z, acts[:, k])
                ls = ls + ((z_next - fut[:, k + 1]) ** 2).mean()
                z = z_next
            ls /= ROLLOUT_K
            opt.zero_grad(); ls.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += ls.item(); nb += 1
        tl /= nb

        model.eval(); vl, vn = 0, 0
        with torch.no_grad():
            for hist, fut, acts in val_loader:
                fut, acts = fut.to(DEVICE), acts.to(DEVICE)
                z = fut[:, 0]; ls = 0
                for k in range(ROLLOUT_K):
                    z = model(z, acts[:, k])
                    ls = ls + ((z - fut[:, k + 1]) ** 2).mean()
                ls /= ROLLOUT_K
                vl += ls.item(); vn += 1
        vl /= vn
        sch.step(vl)
        print(f"  Train {tl:.6f}  Val {vl:.6f}")
        if vl < best:
            best = vl
            save_checkpoint({'model': model.state_dict(), 'val_loss': vl},
                            "checkpoints/shared_dvbf/best.tar")
    print(f"DVBF best: {best:.6f}")


def train_goku():
    print("=" * 60); print("Train DynamicsGOKU on PIWM latent"); print("=" * 60)
    ds = SeqDataset(DATA_DIR)
    n_val = int(len(ds) * 0.1)
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=128, shuffle=False, drop_last=True)

    model = DynamicsGOKU().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')

    for epoch in range(EPOCHS):
        model.train(); tl, nb = 0, 0
        for hist, fut, acts in tqdm(tr_loader, desc=f"GOKU {epoch+1}/{EPOCHS}"):
            fut, acts = fut.to(DEVICE), acts.to(DEVICE)
            z = fut[:, 0]; ls = 0
            for k in range(ROLLOUT_K):
                z_next = model(z, acts[:, k])
                ls = ls + ((z_next - fut[:, k + 1]) ** 2).mean()
                z = z_next
            ls /= ROLLOUT_K
            opt.zero_grad(); ls.backward(); opt.step()
            tl += ls.item(); nb += 1
        tl /= nb

        model.eval(); vl, vn = 0, 0
        with torch.no_grad():
            for hist, fut, acts in val_loader:
                fut, acts = fut.to(DEVICE), acts.to(DEVICE)
                z = fut[:, 0]; ls = 0
                for k in range(ROLLOUT_K):
                    z = model(z, acts[:, k])
                    ls = ls + ((z - fut[:, k + 1]) ** 2).mean()
                ls /= ROLLOUT_K
                vl += ls.item(); vn += 1
        vl /= vn
        sch.step(vl)
        print(f"  Train {tl:.6f}  Val {vl:.6f}")
        if vl < best:
            best = vl
            save_checkpoint({'model': model.state_dict(), 'val_loss': vl},
                            "checkpoints/shared_goku/best.tar")
    print(f"GOKU best: {best:.6f}")


def train_v2p():
    print("=" * 60); print("Train DynamicsVid2Param (FAIR: theta from images via shared encoder)"); print("=" * 60)
    ds = SeqDataset(DATA_DIR, need_images=True)
    n_val = int(len(ds) * 0.1)
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=64, shuffle=False, drop_last=True)

    # Load frozen shared encoder (same weights PIWM/GOKU/DVBF effectively use)
    enc = PhysicsEncoder().to(DEVICE)
    ae_ck = load_checkpoint("checkpoints/piwm_v2_rel/best.tar")
    enc.load_state_dict(ae_ck['encoder'])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False

    model = DynamicsVid2Param().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')

    def encode_history(hist_stacks):
        """hist_stacks: (B, THETA_HIST, FRAME_STACK, 64, 64) -> (B, THETA_HIST, 9)."""
        B, T, C, H, W = hist_stacks.shape
        flat = hist_stacks.reshape(B * T, C, H, W)
        with torch.no_grad():
            obs = enc(flat)  # (B*T, 9)
        return obs.reshape(B, T, -1)

    for epoch in range(EPOCHS):
        model.train(); tl, nb = 0, 0
        for hist_stacks, fut, acts in tqdm(tr_loader, desc=f"V2P {epoch+1}/{EPOCHS}"):
            hist_stacks, fut, acts = hist_stacks.to(DEVICE), fut.to(DEVICE), acts.to(DEVICE)
            hist_obs = encode_history(hist_stacks)          # (B, THETA_HIST, 9)
            theta, mu, lv = model.infer_theta(hist_obs)
            z = fut[:, 0]; ls = 0
            for k in range(ROLLOUT_K):
                z_next = model.step(z, acts[:, k], theta)
                ls = ls + ((z_next - fut[:, k + 1]) ** 2).mean()
                z = z_next
            ls /= ROLLOUT_K
            kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
            loss = ls + 0.001 * kl
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += ls.item(); nb += 1
        tl /= nb

        model.eval(); vl, vn = 0, 0
        with torch.no_grad():
            for hist_stacks, fut, acts in val_loader:
                hist_stacks, fut, acts = hist_stacks.to(DEVICE), fut.to(DEVICE), acts.to(DEVICE)
                hist_obs = encode_history(hist_stacks)
                theta, _, _ = model.infer_theta(hist_obs)
                z = fut[:, 0]; ls = 0
                for k in range(ROLLOUT_K):
                    z = model.step(z, acts[:, k], theta)
                    ls = ls + ((z - fut[:, k + 1]) ** 2).mean()
                ls /= ROLLOUT_K
                vl += ls.item(); vn += 1
        vl /= vn
        sch.step(vl)
        print(f"  Train {tl:.6f}  Val {vl:.6f}")
        if vl < best:
            best = vl
            save_checkpoint({'model': model.state_dict(), 'val_loss': vl},
                            "checkpoints/shared_v2p/best.tar")
    print(f"V2P best: {best:.6f}")


if __name__ == "__main__":
    import sys
    os.makedirs("checkpoints/shared_dvbf", exist_ok=True)
    os.makedirs("checkpoints/shared_goku", exist_ok=True)
    os.makedirs("checkpoints/shared_v2p", exist_ok=True)
    # Allow training only a single method: `python train_shared_baselines.py v2p`
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target in ("all", "dvbf"): train_dvbf()
    if target in ("all", "goku"): train_goku()
    if target in ("all", "v2p"):  train_v2p()
