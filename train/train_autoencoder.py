"""Phase 1: Train Physics-Disentangled Autoencoder.

Encoder: image (3 stacked frames) -> 9-dim observable physics (no x,y)
Decoder: 9-dim -> reconstructed image
Loss = image_reconstruction + lambda * physics_supervision
"""

# --- auto-added: make repo root importable when run as a script ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end shim ---


import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config import (
    DEVICE, DATA_DIR, AE_BATCH_SIZE, AE_LR, AE_EPOCHS,
    LAMBDA_PHYSICS, AE_SAVE_DIR, ENCODER_NAMES, ENCODER_DIM,
)
from data.dataset import RacingCarDataset
from models.encoder import PhysicsEncoder
from models.decoder import PhysicsDecoder
from utils import save_checkpoint


def train():
    print(f"Device: {DEVICE}")
    print(f"Encoder output: {ENCODER_DIM}-dim {ENCODER_NAMES}")

    # Data
    dataset = RacingCarDataset(DATA_DIR, mode="autoencoder")
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=AE_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=AE_BATCH_SIZE, shuffle=False, drop_last=True, num_workers=0)

    # Models
    encoder = PhysicsEncoder().to(DEVICE)
    decoder = PhysicsDecoder().to(DEVICE)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=AE_LR,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    mse = nn.MSELoss()
    best_val_loss = float("inf")

    for epoch in range(AE_EPOCHS):
        # === Train ===
        encoder.train()
        decoder.train()
        train_recon_loss = 0
        train_phys_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{AE_EPOCHS} [train]")
        for frames, physics_gt, target_img in pbar:
            frames = frames.to(DEVICE)
            physics_gt = physics_gt.to(DEVICE)
            target_img = target_img.to(DEVICE)

            z = encoder(frames)
            recon = decoder(z)

            loss_recon = mse(recon, target_img)
            loss_phys = mse(z, physics_gt)
            loss = loss_recon + LAMBDA_PHYSICS * loss_phys

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_recon_loss += loss_recon.item()
            train_phys_loss += loss_phys.item()
            n_batches += 1
            pbar.set_postfix(recon=f"{loss_recon.item():.4f}", phys=f"{loss_phys.item():.4f}")

        train_recon_loss /= n_batches
        train_phys_loss /= n_batches

        # === Validation ===
        encoder.eval()
        decoder.eval()
        val_recon_loss = 0
        val_phys_loss = 0
        val_batches = 0

        with torch.no_grad():
            for frames, physics_gt, target_img in val_loader:
                frames = frames.to(DEVICE)
                physics_gt = physics_gt.to(DEVICE)
                target_img = target_img.to(DEVICE)

                z = encoder(frames)
                recon = decoder(z)

                val_recon_loss += mse(recon, target_img).item()
                val_phys_loss += mse(z, physics_gt).item()
                val_batches += 1

        val_recon_loss /= val_batches
        val_phys_loss /= val_batches
        val_total = val_recon_loss + LAMBDA_PHYSICS * val_phys_loss

        scheduler.step(val_total)
        lr = optimizer.param_groups[0]["lr"]

        print(f"  Train: recon={train_recon_loss:.5f} phys={train_phys_loss:.5f}")
        print(f"  Val:   recon={val_recon_loss:.5f} phys={val_phys_loss:.5f} total={val_total:.5f} lr={lr:.6f}")

        # Per-dimension physics error
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                all_z, all_gt = [], []
                for frames, physics_gt, _ in val_loader:
                    z = encoder(frames.to(DEVICE))
                    all_z.append(z.cpu())
                    all_gt.append(physics_gt)
                all_z = torch.cat(all_z)
                all_gt = torch.cat(all_gt)
                per_dim_mae = (all_z - all_gt).abs().mean(dim=0)
                print("  Per-dim MAE (normalized):")
                for i, name in enumerate(ENCODER_NAMES):
                    print(f"    {name:6s}: {per_dim_mae[i]:.4f}")

        # Save
        is_best = val_total < best_val_loss
        if is_best:
            best_val_loss = val_total

        save_checkpoint({
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_total,
        }, os.path.join(AE_SAVE_DIR, "checkpoint.tar"))

        if is_best:
            save_checkpoint({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "val_loss": val_total,
            }, os.path.join(AE_SAVE_DIR, "best.tar"))

    print(f"Training complete. Best val loss: {best_val_loss:.5f}")


if __name__ == "__main__":
    train()
