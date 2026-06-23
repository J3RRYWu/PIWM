"""Train Localization Module and run ablation comparison.

Step 1: Train CoarsePositionHead (image -> x,y)
Step 2: Train FusionGate on multi-step rollout
Step 3: Ablation - compare integration-only vs coarse-only vs fused over N steps
"""

# --- auto-added: make repo root importable when run as a script ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end shim ---


import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from config import (
    DEVICE, DATA_DIR, PHYSICS_MEAN, PHYSICS_STD,
    ENCODER_STATE_INDICES, FRAME_STACK, DT,
    ENCODER_MEAN, ENCODER_STD,
)
from models.localization import CoarsePositionHead, FusionGate
from models.dynamics import PhysicsDynamics
from utils import save_checkpoint, load_checkpoint


# ============================================================
# Dataset for localization training
# ============================================================

class LocalizationDataset(Dataset):
    """Returns (frames, position_xy_normalized) for coarse head training."""

    def __init__(self, data_dir):
        npz_files = sorted(glob.glob(f"{data_dir}/*.npz"))
        all_imgs, all_physics = [], []
        episode_lengths = []

        for f in npz_files:
            try:
                d = np.load(f, allow_pickle=True)
                imgs = d["imgs"].astype(np.float32)
                if imgs.max() > 1.0:
                    imgs /= 255.0
                n = len(imgs)
                if n < FRAME_STACK + 1:
                    continue
                pos = d["position"].astype(np.float32)
                yaw = d["yaw"].astype(np.float32)
                vel = d["velocity"].astype(np.float32)
                omega = d["angular_velocity"].astype(np.float32)
                wheel = d["wheel_omega"].astype(np.float32)
                steer = d["steering_angle"].astype(np.float32)
                physics = np.column_stack([pos, yaw, vel, omega, wheel, steer])

                all_imgs.append(imgs)
                all_physics.append(physics)
                episode_lengths.append(n)
            except Exception as e:
                print(f"Skipping {f}: {e}")

        self.indices = []
        for ep_idx, ep_len in enumerate(episode_lengths):
            for t in range(FRAME_STACK - 1, ep_len):
                self.indices.append((ep_idx, t))

        self.all_imgs = all_imgs
        self.all_physics = all_physics
        print(f"LocalizationDataset: {len(episode_lengths)} episodes, {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, t = self.indices[idx]
        frames = np.stack([
            self.all_imgs[ep_idx][t - FRAME_STACK + 1 + i]
            for i in range(FRAME_STACK)
        ], axis=0)

        # Position (x, y) normalized
        pos_xy = self.all_physics[ep_idx][t, :2]
        pos_norm = (pos_xy - PHYSICS_MEAN[:2]) / PHYSICS_STD[:2]

        return (
            torch.tensor(frames, dtype=torch.float32),
            torch.tensor(pos_norm, dtype=torch.float32),
        )


# ============================================================
# Dataset for multi-step rollout evaluation
# ============================================================

class RolloutDataset(Dataset):
    """Returns sequences for multi-step rollout ablation.

    Each sample: (frames_seq, full_state_seq, action_seq) of length seq_len.
    """

    def __init__(self, data_dir, seq_len=100):
        self.seq_len = seq_len
        npz_files = sorted(glob.glob(f"{data_dir}/*.npz"))
        self.sequences = []

        for f in npz_files:
            try:
                d = np.load(f, allow_pickle=True)
                imgs = d["imgs"].astype(np.float32)
                if imgs.max() > 1.0:
                    imgs /= 255.0
                n = len(imgs)
                pos = d["position"].astype(np.float32)
                yaw = d["yaw"].astype(np.float32)
                vel = d["velocity"].astype(np.float32)
                omega = d["angular_velocity"].astype(np.float32)
                wheel = d["wheel_omega"].astype(np.float32)
                steer = d["steering_angle"].astype(np.float32)
                physics = np.column_stack([pos, yaw, vel, omega, wheel, steer])
                actions = d["action"].astype(np.float32)

                # Extract non-overlapping sequences
                start = FRAME_STACK - 1
                while start + seq_len < n:
                    self.sequences.append((imgs, physics, actions, start))
                    start += seq_len
            except:
                pass

        print(f"RolloutDataset: {len(self.sequences)} sequences of length {seq_len}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        imgs, physics, actions, start = self.sequences[idx]
        T = self.seq_len

        frames_seq = []
        for t in range(start, start + T):
            frames = np.stack([
                imgs[t - FRAME_STACK + 1 + i] for i in range(FRAME_STACK)
            ], axis=0)
            frames_seq.append(frames)

        state_seq = physics[start:start + T]
        action_seq = actions[start:start + T]

        # Normalize states
        state_norm = (state_seq - PHYSICS_MEAN) / PHYSICS_STD

        return (
            torch.tensor(np.array(frames_seq), dtype=torch.float32),  # (T, 3, 64, 64)
            torch.tensor(state_norm, dtype=torch.float32),             # (T, 11)
            torch.tensor(action_seq, dtype=torch.float32),             # (T, 3)
        )


# ============================================================
# Step 1: Train CoarsePositionHead
# ============================================================

def train_coarse_head(data_dir=DATA_DIR):
    print("=" * 60)
    print(f"Step 1: Training CoarsePositionHead on {data_dir}")
    print("=" * 60)

    dataset = LocalizationDataset(data_dir)
    n_val = int(len(dataset) * 0.1)
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, drop_last=True)

    model = CoarsePositionHead().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    mse = nn.MSELoss()
    best_val = float("inf")

    for epoch in range(30):
        model.train()
        train_loss, n = 0, 0
        for frames, pos_gt in tqdm(train_loader, desc=f"Coarse Epoch {epoch+1}/30"):
            frames, pos_gt = frames.to(DEVICE), pos_gt.to(DEVICE)
            pred = model(frames)
            loss = mse(pred, pos_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n += 1
        train_loss /= n

        model.eval()
        val_loss, vn = 0, 0
        with torch.no_grad():
            for frames, pos_gt in val_loader:
                frames, pos_gt = frames.to(DEVICE), pos_gt.to(DEVICE)
                val_loss += mse(model(frames), pos_gt).item()
                vn += 1
        val_loss /= vn
        scheduler.step(val_loss)

        # Convert to real-world MAE
        real_mae_x = val_loss ** 0.5 * PHYSICS_STD[0]
        real_mae_y = val_loss ** 0.5 * PHYSICS_STD[1]
        print(f"  Train: {train_loss:.5f}  Val: {val_loss:.5f}  (~{real_mae_x:.1f}m x, ~{real_mae_y:.1f}m y)")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint({
                "state_dict": model.state_dict(),
                "val_loss": val_loss,
            }, "checkpoints/localization/coarse_best.tar")

    print(f"CoarsePositionHead training complete. Best val: {best_val:.5f}")
    return model


# ============================================================
# Step 2: Ablation comparison
# ============================================================

def run_ablation(coarse_head, data_dir=DATA_DIR):
    print("\n" + "=" * 60)
    print("Step 2: Multi-step rollout ablation")
    print("=" * 60)

    # Load dynamics
    dynamics = PhysicsDynamics().to(DEVICE)
    dyn_ckpt = load_checkpoint("checkpoints/dynamics/best.tar")
    dynamics.load_state_dict(dyn_ckpt["dynamics"])
    dynamics.eval()
    coarse_head.eval()

    dataset = RolloutDataset(data_dir, seq_len=200)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    horizons = [1, 5, 10, 20, 50, 100, 200]
    errors_integ = {h: [] for h in horizons}
    errors_coarse = {h: [] for h in horizons}

    with torch.no_grad():
        for frames_seq, state_seq, action_seq in tqdm(loader, desc="Rollout"):
            frames_seq = frames_seq.squeeze(0).to(DEVICE)  # (T, 3, 64, 64)
            state_seq = state_seq.squeeze(0).to(DEVICE)     # (T, 11)
            action_seq = action_seq.squeeze(0).to(DEVICE)   # (T, 3)
            T = state_seq.size(0)

            # Ground truth positions
            gt_pos = state_seq[:, :2]  # (T, 2) normalized

            # --- Integration path ---
            pos_integ = [state_seq[0, :2]]  # start from GT
            z_current = state_seq[0:1]  # (1, 11)
            for t in range(T - 1):
                z_next, _ = dynamics(z_current, action_seq[t:t+1])
                pos_integ.append(z_next[0, :2])
                # Update full state: use GT for non-position dims, integrated for position
                z_current = state_seq[t+1:t+2].clone()
                z_current[0, 0] = z_next[0, 0]
                z_current[0, 1] = z_next[0, 1]
            pos_integ = torch.stack(pos_integ)  # (T, 2)

            # --- Coarse path ---
            pos_coarse = coarse_head(frames_seq)  # (T, 2)

            # Compute errors at each horizon
            for h in horizons:
                if h >= T:
                    continue
                # Integration error at step h
                err_i = (pos_integ[h] - gt_pos[h]).pow(2).sum().sqrt().item()
                errors_integ[h].append(err_i)
                # Coarse error at step h
                err_c = (pos_coarse[h] - gt_pos[h]).pow(2).sum().sqrt().item()
                errors_coarse[h].append(err_c)

    print("\n--- Ablation Results (normalized L2 error) ---")
    print(f"{'Horizon':>8s} | {'Integration':>12s} | {'Coarse Head':>12s} | {'Better':>10s}")
    print("-" * 50)
    for h in horizons:
        if errors_integ[h]:
            ei = np.mean(errors_integ[h])
            ec = np.mean(errors_coarse[h])
            better = "Integ" if ei < ec else "Coarse"
            print(f"{h:>8d} | {ei:>12.5f} | {ec:>12.5f} | {better:>10s}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=DATA_DIR, help='data directory')
    args = parser.parse_args()
    DATA_DIR_OVERRIDE = args.data

    coarse_head = train_coarse_head(DATA_DIR_OVERRIDE)
    run_ablation(coarse_head, DATA_DIR_OVERRIDE)
