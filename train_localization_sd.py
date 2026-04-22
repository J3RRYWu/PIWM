"""Train CoarsePositionHeadSD: predict track-relative (s, d) coords.

The s coordinate is encoded as (sin(2πs/L), cos(2πs/L)) to handle wrap-around.
d is normalized by 10 (track half-width is ~7).
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from config import DEVICE, FRAME_STACK
from models.localization_sd import CoarsePositionHeadSD
from track_utils import TrackCoords
from utils import save_checkpoint, load_checkpoint

TRAIN_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/train/fixed/"
TEST_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/test/fixed/"
D_NORM = 10.0  # normalize d by this (track half-width ~7)


class SDDataset(Dataset):
    """Returns (frames, target_sd_encoded) where target is (s_sin, s_cos, d_norm)."""

    def __init__(self, data_dir, track):
        self.track = track
        files = sorted(glob.glob(f"{data_dir}/*.npz"))
        self.items = []  # (frames, s_sin, s_cos, d_norm)

        for f in files:
            try:
                d = np.load(f, allow_pickle=True)
                imgs = d["imgs"].astype(np.float32)
                if imgs.max() > 1.0:
                    imgs /= 255.0
                pos = d["position"].astype(np.float32)
                n = len(imgs)
                if n < FRAME_STACK + 1:
                    continue

                # Convert all positions to (s, d)
                s_arr, d_arr = track.xy_to_sd(pos)

                # Encode s as (sin, cos)
                theta = 2 * np.pi * s_arr / track.total_len
                s_sin = np.sin(theta).astype(np.float32)
                s_cos = np.cos(theta).astype(np.float32)
                d_norm = (d_arr / D_NORM).astype(np.float32)

                # Store per-frame data
                for t in range(FRAME_STACK - 1, n):
                    frames = np.stack([imgs[t - FRAME_STACK + 1 + i] for i in range(FRAME_STACK)], axis=0)
                    self.items.append((frames, s_sin[t], s_cos[t], d_norm[t]))
            except Exception as e:
                print(f"Skipping {f}: {e}")

        print(f"SDDataset @ {data_dir}: {len(self.items)} samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        frames, s_sin, s_cos, d_norm = self.items[idx]
        target = torch.tensor([s_sin, s_cos, d_norm], dtype=torch.float32)
        return torch.tensor(frames, dtype=torch.float32), target


def decode_sd(pred_sd, track_total_len):
    """Decode (s_sin, s_cos, d_norm) back to (s, d).

    Args:
        pred_sd: (N, 3) numpy array
    Returns:
        s: (N,) in [0, track_total_len)
        d: (N,) in original units
    """
    s_sin = pred_sd[:, 0]
    s_cos = pred_sd[:, 1]
    d_norm = pred_sd[:, 2]

    theta = np.arctan2(s_sin, s_cos)  # [-pi, pi]
    theta = np.mod(theta, 2 * np.pi)   # [0, 2pi)
    s = theta * track_total_len / (2 * np.pi)
    d = d_norm * D_NORM
    return s, d


def train():
    # Build track from any training episode (all same track)
    files = sorted(glob.glob(f"{TRAIN_DIR}/*.npz"))
    d0 = np.load(files[0], allow_pickle=True)
    track = TrackCoords(d0["map"])
    print(f"Track total length: {track.total_len:.1f} units, tiles: {track.n}")

    # Datasets
    full_ds = SDDataset(TRAIN_DIR, track)
    n_val = int(len(full_ds) * 0.1)
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, drop_last=True)

    # Model
    model = CoarsePositionHeadSD().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    mse = nn.MSELoss()

    best_val = float("inf")
    epochs = 30

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        n_batches = 0
        for frames, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            frames, target = frames.to(DEVICE), target.to(DEVICE)
            pred = model(frames)
            loss = mse(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches

        # Val
        model.eval()
        val_loss = 0
        vn = 0
        all_pred, all_gt = [], []
        with torch.no_grad():
            for frames, target in val_loader:
                frames, target = frames.to(DEVICE), target.to(DEVICE)
                pred = model(frames)
                val_loss += mse(pred, target).item()
                vn += 1
                all_pred.append(pred.cpu().numpy())
                all_gt.append(target.cpu().numpy())
        val_loss /= vn
        scheduler.step(val_loss)

        # Decode and compute real-world errors
        all_pred = np.concatenate(all_pred)
        all_gt = np.concatenate(all_gt)
        s_pred, d_pred = decode_sd(all_pred, track.total_len)
        s_gt, d_gt = decode_sd(all_gt, track.total_len)

        # s error with wrap-around
        ds = np.abs(s_pred - s_gt)
        ds = np.minimum(ds, track.total_len - ds)
        dd = np.abs(d_pred - d_gt)

        # Convert to (x, y) and compute L2 error
        xy_pred = track.sd_to_xy(s_pred, d_pred)
        xy_gt = track.sd_to_xy(s_gt, d_gt)
        xy_err = np.sqrt(((xy_pred - xy_gt) ** 2).sum(axis=1))

        print(f"  Train: {train_loss:.5f}  Val: {val_loss:.5f}")
        print(f"  Val s_err: mean={ds.mean():.2f} (track_len={track.total_len:.0f}), "
              f"d_err: mean={dd.mean():.3f}, xy_err: mean={xy_err.mean():.2f} units")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint({
                "state_dict": model.state_dict(),
                "val_loss": val_loss,
                "track_total_len": track.total_len,
            }, "checkpoints/localization_sd/best.tar")

    print(f"Training complete. Best val loss: {best_val:.5f}")
    return model, track


def evaluate_on_test(model, track):
    """Evaluate on UNSEEN test episodes."""
    print("\n" + "=" * 60)
    print("Evaluating on UNSEEN test episodes")
    print("=" * 60)

    test_ds = SDDataset(TEST_DIR, track)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, drop_last=False)

    model.eval()
    all_pred, all_gt = [], []
    with torch.no_grad():
        for frames, target in loader:
            pred = model(frames.to(DEVICE)).cpu().numpy()
            all_pred.append(pred)
            all_gt.append(target.numpy())

    all_pred = np.concatenate(all_pred)
    all_gt = np.concatenate(all_gt)

    s_pred, d_pred = decode_sd(all_pred, track.total_len)
    s_gt, d_gt = decode_sd(all_gt, track.total_len)

    ds = np.abs(s_pred - s_gt)
    ds = np.minimum(ds, track.total_len - ds)
    dd = np.abs(d_pred - d_gt)

    xy_pred = track.sd_to_xy(s_pred, d_pred)
    xy_gt = track.sd_to_xy(s_gt, d_gt)
    xy_err = np.sqrt(((xy_pred - xy_gt) ** 2).sum(axis=1))

    print(f"Test samples: {len(xy_err)}")
    print(f"  s error:    mean={ds.mean():.2f}, median={np.median(ds):.2f}, max={ds.max():.2f} "
          f"(track_len={track.total_len:.0f})")
    print(f"  d error:    mean={dd.mean():.3f}, median={np.median(dd):.3f}")
    print(f"  xy error:   mean={xy_err.mean():.2f}, median={np.median(xy_err):.2f}, "
          f"max={xy_err.max():.2f}")
    print(f"  xy <5:   {(xy_err < 5).mean():.1%}")
    print(f"  xy <10:  {(xy_err < 10).mean():.1%}")
    print(f"  xy <20:  {(xy_err < 20).mean():.1%}")


if __name__ == "__main__":
    model, track = train()
    evaluate_on_test(model, track)
