"""Visualize Decoder reconstruction quality."""

import numpy as np
import torch
from PIL import Image

from config import DEVICE, DATA_DIR, AE_SAVE_DIR, ENCODER_NAMES
from data.dataset import RacingCarDataset
from models.encoder import PhysicsEncoder
from models.decoder import PhysicsDecoder
from utils import load_checkpoint

import os


def tensor_to_grid(tensors, nrow=8, padding=2):
    """Convert batch of (N, 1, H, W) tensors to a PIL grid image."""
    N, C, H, W = tensors.shape
    ncol = nrow
    nrows = (N + ncol - 1) // ncol
    grid_h = nrows * H + (nrows + 1) * padding
    grid_w = ncol * W + (ncol + 1) * padding
    grid = np.ones((grid_h, grid_w), dtype=np.uint8) * 255

    for idx in range(N):
        row = idx // ncol
        col = idx % ncol
        y = padding + row * (H + padding)
        x = padding + col * (W + padding)
        img = (tensors[idx, 0].numpy() * 255).clip(0, 255).astype(np.uint8)
        grid[y:y+H, x:x+W] = img

    return Image.fromarray(grid, mode='L')


def main():
    # Load models
    ckpt = load_checkpoint(os.path.join(AE_SAVE_DIR, "best.tar"))
    encoder = PhysicsEncoder().to(DEVICE)
    decoder = PhysicsDecoder().to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    dataset = RacingCarDataset(DATA_DIR, mode="autoencoder")
    os.makedirs("vis", exist_ok=True)

    indices = np.linspace(0, len(dataset) - 1, 32, dtype=int)

    originals = []
    reconstructed = []
    physics_errors = []

    with torch.no_grad():
        for i in indices:
            frames, physics_gt, target_img = dataset[i]
            frames = frames.unsqueeze(0).to(DEVICE)
            physics_gt = physics_gt.unsqueeze(0).to(DEVICE)

            z = encoder(frames)
            recon = decoder(z)

            originals.append(target_img.unsqueeze(0))
            reconstructed.append(recon.cpu())
            physics_errors.append((z - physics_gt).abs().cpu())

    originals = torch.cat(originals)         # (32, 1, 64, 64)
    reconstructed = torch.cat(reconstructed)  # (32, 1, 64, 64)
    physics_errors = torch.cat(physics_errors)

    # Save grids
    grid_orig = tensor_to_grid(originals, nrow=8)
    grid_recon = tensor_to_grid(reconstructed, nrow=8)
    grid_orig.save("vis/originals.png")
    grid_recon.save("vis/reconstructed.png")
    print("Saved vis/originals.png")
    print("Saved vis/reconstructed.png")

    # Side-by-side: each pair is (original, reconstruction) stacked vertically
    pairs = []
    for i in range(32):
        o = (originals[i, 0].numpy() * 255).clip(0, 255).astype(np.uint8)
        r = (reconstructed[i, 0].numpy() * 255).clip(0, 255).astype(np.uint8)
        pair = np.vstack([o, np.ones((2, 64), dtype=np.uint8) * 255, r])  # orig on top, recon below
        pairs.append(pair)

    # Arrange pairs in a grid: 8 columns
    ncol = 8
    nrows = 4
    pad = 4
    cell_h = 64 + 2 + 64  # orig + gap + recon
    grid_h = nrows * cell_h + (nrows + 1) * pad
    grid_w = ncol * 64 + (ncol + 1) * pad
    comparison = np.ones((grid_h, grid_w), dtype=np.uint8) * 200

    for idx, pair in enumerate(pairs):
        row = idx // ncol
        col = idx % ncol
        y = pad + row * (cell_h + pad)
        x = pad + col * (64 + pad)
        comparison[y:y+cell_h, x:x+64] = pair

    Image.fromarray(comparison, mode='L').save("vis/comparison.png")
    print("Saved vis/comparison.png (top=original, bottom=reconstruction per cell)")

    # Physics errors
    mean_err = physics_errors.mean(dim=0)
    print("\nPer-dim MAE (normalized):")
    for i, name in enumerate(ENCODER_NAMES):
        print(f"  {name:6s}: {mean_err[i]:.4f}")


if __name__ == "__main__":
    main()
