"""V1: Multi-step rollout loss for Dynamics.

Instead of single-step MSE, train dynamics on k-step rollouts:
    z_pred_1 = dyn(z_0, a_0)
    z_pred_2 = dyn(z_pred_1, a_1)
    ...
    loss = sum_k MSE(z_pred_k, z_true_k)

Encoder/Decoder unchanged (uses V0 AE checkpoint).
Output: checkpoints/dynamics_v1/best.tar
"""

import os, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from config import (DEVICE, DATA_DIR, PHYSICS_MEAN, PHYSICS_STD,
                    FULL_STATE_NAMES, DYN_BATCH_SIZE, DYN_LR, DYN_EPOCHS,
                    LAMBDA_RESIDUAL)
from models.dynamics import PhysicsDynamics
from utils import save_checkpoint

ROLLOUT_K = 8   # train on 8-step rollouts
SAVE_DIR = "checkpoints/dynamics_v1/"


class SeqDataset(Dataset):
    def __init__(self, data_dir, seq_len=ROLLOUT_K + 1):
        files = sorted(glob.glob(f"{data_dir}/*.npz"))
        self.seq_len = seq_len
        self.physics_list = []
        self.actions_list = []
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
                if n < seq_len + 1: continue
                self.physics_list.append(phys)
                self.actions_list.append(acts)
                for t in range(n - seq_len):
                    self.indices.append((ep_idx, t))
            except: pass
        print(f"SeqDataset: {len(self.indices)} sequences of len {seq_len}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep, t = self.indices[idx]
        phys = self.physics_list[ep][t:t + self.seq_len]
        acts = self.actions_list[ep][t:t + self.seq_len - 1]
        phys_norm = (phys - PHYSICS_MEAN) / PHYSICS_STD
        return (torch.tensor(phys_norm, dtype=torch.float32),
                torch.tensor(acts, dtype=torch.float32))


def train():
    print("=" * 60)
    print("V1: Multi-step rollout Dynamics training")
    print("=" * 60)
    print(f"Rollout length: {ROLLOUT_K}")

    ds = SeqDataset(DATA_DIR, seq_len=ROLLOUT_K + 1)
    n_val = int(len(ds) * 0.1)
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=DYN_BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=DYN_BATCH_SIZE, shuffle=False, drop_last=True)

    dyn = PhysicsDynamics().to(DEVICE)
    opt = torch.optim.Adam(dyn.parameters(), lr=DYN_LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    best_val = float('inf')

    for epoch in range(DYN_EPOCHS):
        dyn.train()
        tloss, nb = 0, 0
        pbar = tqdm(tr_loader, desc=f"V1 Epoch {epoch+1}/{DYN_EPOCHS}")
        for phys, acts in pbar:
            phys = phys.to(DEVICE)  # (B, K+1, 11)
            acts = acts.to(DEVICE)  # (B, K, 3)

            z = phys[:, 0]  # (B, 11)
            loss_state = 0
            loss_res = 0
            for k in range(ROLLOUT_K):
                z_next, res = dyn(z, acts[:, k])
                loss_state = loss_state + ((z_next - phys[:, k + 1]) ** 2).mean()
                loss_res = loss_res + (res ** 2).mean()
                z = z_next
            loss_state /= ROLLOUT_K
            loss_res /= ROLLOUT_K
            loss = loss_state + LAMBDA_RESIDUAL * loss_res

            opt.zero_grad(); loss.backward(); opt.step()
            tloss += loss_state.item(); nb += 1
            pbar.set_postfix(state=f"{loss_state.item():.5f}")
        tloss /= nb

        # Val
        dyn.eval()
        vloss, vn = 0, 0
        with torch.no_grad():
            for phys, acts in val_loader:
                phys = phys.to(DEVICE); acts = acts.to(DEVICE)
                z = phys[:, 0]
                ls = 0
                for k in range(ROLLOUT_K):
                    z_next, _ = dyn(z, acts[:, k])
                    ls = ls + ((z_next - phys[:, k + 1]) ** 2).mean()
                    z = z_next
                ls /= ROLLOUT_K
                vloss += ls.item(); vn += 1
        vloss /= vn
        sch.step(vloss)
        print(f"  Train state: {tloss:.6f}  Val state: {vloss:.6f}")

        if vloss < best_val:
            best_val = vloss
            save_checkpoint({'dynamics': dyn.state_dict(), 'val_loss': vloss},
                            os.path.join(SAVE_DIR, 'best.tar'))
    print(f"V1 done. Best: {best_val:.6f}")


if __name__ == "__main__":
    train()
