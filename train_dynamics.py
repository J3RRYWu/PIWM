"""Phase 2: Train Physics Dynamics Model.

Freezes Encoder/Decoder, trains dynamics model to predict z_{t+1} from z_t + action.
Loss = state_prediction + lambda_residual * residual_regularization.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config import (
    DEVICE, DATA_DIR, DYN_BATCH_SIZE, DYN_LR, DYN_EPOCHS,
    LAMBDA_RESIDUAL, DYN_SAVE_DIR, AE_SAVE_DIR, FULL_STATE_NAMES,
)
from data.dataset import RacingCarDataset
from models.dynamics import PhysicsDynamics
from utils import save_checkpoint, load_checkpoint


def train():
    print(f"Device: {DEVICE}")

    # Data
    dataset = RacingCarDataset(DATA_DIR, mode="dynamics")
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=DYN_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=DYN_BATCH_SIZE, shuffle=False, drop_last=True, num_workers=0)

    # Model
    dynamics = PhysicsDynamics().to(DEVICE)
    optimizer = torch.optim.Adam(dynamics.parameters(), lr=DYN_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    mse = nn.MSELoss()
    best_val_loss = float("inf")

    for epoch in range(DYN_EPOCHS):
        # === Train ===
        dynamics.train()
        train_state_loss = 0
        train_res_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{DYN_EPOCHS} [train]")
        for z_t, action_t, z_t1_gt in pbar:
            z_t = z_t.to(DEVICE)
            action_t = action_t.to(DEVICE)
            z_t1_gt = z_t1_gt.to(DEVICE)

            z_t1_pred, residual = dynamics(z_t, action_t)

            loss_state = mse(z_t1_pred, z_t1_gt)
            loss_res = (residual ** 2).mean()
            loss = loss_state + LAMBDA_RESIDUAL * loss_res

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_state_loss += loss_state.item()
            train_res_loss += loss_res.item()
            n_batches += 1
            pbar.set_postfix(state=f"{loss_state.item():.5f}", res=f"{loss_res.item():.5f}")

        train_state_loss /= n_batches
        train_res_loss /= n_batches

        # === Validation ===
        dynamics.eval()
        val_state_loss = 0
        val_res_loss = 0
        val_batches = 0

        with torch.no_grad():
            for z_t, action_t, z_t1_gt in val_loader:
                z_t = z_t.to(DEVICE)
                action_t = action_t.to(DEVICE)
                z_t1_gt = z_t1_gt.to(DEVICE)

                z_t1_pred, residual = dynamics(z_t, action_t)
                val_state_loss += mse(z_t1_pred, z_t1_gt).item()
                val_res_loss += (residual ** 2).mean().item()
                val_batches += 1

        val_state_loss /= val_batches
        val_res_loss /= val_batches
        val_total = val_state_loss + LAMBDA_RESIDUAL * val_res_loss

        scheduler.step(val_total)
        lr = optimizer.param_groups[0]["lr"]

        print(f"  Train: state={train_state_loss:.6f} res={train_res_loss:.6f}")
        print(f"  Val:   state={val_state_loss:.6f} res={val_res_loss:.6f} total={val_total:.6f} lr={lr:.6f}")

        # Per-dimension error
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                all_pred, all_gt, all_res = [], [], []
                for z_t, action_t, z_t1_gt in val_loader:
                    z_t1_pred, residual = dynamics(z_t.to(DEVICE), action_t.to(DEVICE))
                    all_pred.append(z_t1_pred.cpu())
                    all_gt.append(z_t1_gt)
                    all_res.append(residual.cpu())
                all_pred = torch.cat(all_pred)
                all_gt = torch.cat(all_gt)
                all_res = torch.cat(all_res)

                per_dim_mae = (all_pred - all_gt).abs().mean(dim=0)
                per_dim_res = all_res.abs().mean(dim=0)
                print("  Per-dim MAE | Residual magnitude (normalized):")
                for i, name in enumerate(FULL_STATE_NAMES):
                    print(f"    {name:6s}: MAE={per_dim_mae[i]:.5f}  |res|={per_dim_res[i]:.5f}")

        # Save
        is_best = val_total < best_val_loss
        if is_best:
            best_val_loss = val_total

        save_checkpoint({
            "epoch": epoch,
            "dynamics": dynamics.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_total,
        }, os.path.join(DYN_SAVE_DIR, "checkpoint.tar"))

        if is_best:
            save_checkpoint({
                "epoch": epoch,
                "dynamics": dynamics.state_dict(),
                "val_loss": val_total,
            }, os.path.join(DYN_SAVE_DIR, "best.tar"))

    print(f"Training complete. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    train()
