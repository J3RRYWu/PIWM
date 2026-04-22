"""Train LSTM-based localization with ablation.

Runs both Plan A (xy) and Plan A+B (sd) and reports results.
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from config import DEVICE, FRAME_STACK
from models.localization_lstm import LSTMLocalizer
from track_utils import TrackCoords
from utils import save_checkpoint, load_checkpoint

TRAIN_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/train/fixed/"
TEST_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/test/fixed/"
SEQ_LEN = 10  # number of frame-stacks to feed into LSTM
D_NORM = 10.0


def build_track(data_dir):
    files = sorted(glob.glob(f"{data_dir}/*.npz"))
    d0 = np.load(files[0], allow_pickle=True)
    return TrackCoords(d0["map"])


class LSTMDataset(Dataset):
    """Returns (frame_seq, target) where frame_seq has SEQ_LEN frame-stacks.

    mode='xy':  target = (x_norm, y_norm) using simple (mean, std) normalization
    mode='sd':  target = (s_sin, s_cos, d_norm)
    """

    def __init__(self, data_dir, track, mode='xy', xy_mean=None, xy_std=None):
        self.mode = mode
        self.track = track
        files = sorted(glob.glob(f"{data_dir}/*.npz"))

        all_imgs = []
        all_targets = []
        episode_frame_offsets = [0]

        all_positions = []

        for f in files:
            try:
                d = np.load(f, allow_pickle=True)
                imgs = d["imgs"].astype(np.float32)
                if imgs.max() > 1.0:
                    imgs /= 255.0
                pos = d["position"].astype(np.float32)
                n = len(imgs)
                if n < FRAME_STACK + SEQ_LEN:
                    continue

                all_imgs.append(imgs)
                all_positions.append(pos)
                episode_frame_offsets.append(episode_frame_offsets[-1] + n)
            except:
                pass

        self.all_imgs = all_imgs
        self.all_positions = all_positions

        # Normalization for xy
        concat_pos = np.concatenate(all_positions)
        if xy_mean is None:
            self.xy_mean = concat_pos.mean(axis=0).astype(np.float32)
            self.xy_std = concat_pos.std(axis=0).astype(np.float32)
        else:
            self.xy_mean = xy_mean
            self.xy_std = xy_std

        # Build valid sample indices: (ep_idx, end_frame)
        # Sample is a sequence ending at end_frame, of length SEQ_LEN
        # Each item in sequence is a stack of FRAME_STACK frames
        # So we need the first frame of first item to be valid: end_frame - SEQ_LEN + 1 - (FRAME_STACK - 1) >= 0
        self.indices = []
        for ep_idx, imgs in enumerate(all_imgs):
            n = len(imgs)
            min_end = (SEQ_LEN - 1) + (FRAME_STACK - 1)
            for end_frame in range(min_end, n):
                self.indices.append((ep_idx, end_frame))

        print(f"LSTMDataset @ {data_dir} [{mode}]: {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, end_frame = self.indices[idx]
        imgs = self.all_imgs[ep_idx]

        # Build sequence of SEQ_LEN frame-stacks, ending at end_frame
        seq = []
        for offset in range(SEQ_LEN):
            # Frame-stack centered at (end_frame - SEQ_LEN + 1 + offset)
            center = end_frame - SEQ_LEN + 1 + offset
            stack = np.stack([
                imgs[center - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)
            ], axis=0)
            seq.append(stack)
        seq = np.stack(seq, axis=0)  # (SEQ_LEN, FRAME_STACK, 64, 64)

        pos = self.all_positions[ep_idx][end_frame]

        if self.mode == 'xy':
            target = (pos - self.xy_mean) / self.xy_std
            target = target.astype(np.float32)
        else:  # sd
            s, d = self.track.xy_to_sd(pos)
            theta = 2 * np.pi * s / self.track.total_len
            target = np.array([np.sin(theta), np.cos(theta), d / D_NORM], dtype=np.float32)

        return (torch.tensor(seq, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32),
                torch.tensor(pos, dtype=torch.float32))  # also return real xy for eval


def decode_sd(pred, total_len):
    s_sin, s_cos, d_norm = pred[:, 0], pred[:, 1], pred[:, 2]
    theta = np.arctan2(s_sin, s_cos)
    theta = np.mod(theta, 2 * np.pi)
    s = theta * total_len / (2 * np.pi)
    d = d_norm * D_NORM
    return s, d


def train_variant(mode, track, epochs=20):
    """Train LSTM localizer with given target mode ('xy' or 'sd')."""
    print(f"\n{'='*60}\nTraining LSTM [{mode}]\n{'='*60}")

    # Build dataset
    full_ds = LSTMDataset(TRAIN_DIR, track, mode=mode)
    xy_mean, xy_std = full_ds.xy_mean, full_ds.xy_std

    n_val = int(len(full_ds) * 0.1)
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=True)

    out_dim = 2 if mode == 'xy' else 3
    model = LSTMLocalizer(out_dim=out_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    mse = nn.MSELoss()

    best_val = float('inf')
    save_dir = f"checkpoints/localization_lstm_{mode}/"

    for epoch in range(epochs):
        model.train()
        tloss, nb = 0, 0
        for seq, target, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            pred = model(seq)
            loss = mse(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tloss += loss.item()
            nb += 1
        tloss /= nb

        model.eval()
        vloss, vn = 0, 0
        all_pred, all_pos = [], []
        with torch.no_grad():
            for seq, target, pos in val_loader:
                seq, target = seq.to(DEVICE), target.to(DEVICE)
                pred = model(seq)
                vloss += mse(pred, target).item()
                vn += 1
                all_pred.append(pred.cpu().numpy())
                all_pos.append(pos.numpy())
        vloss /= vn
        scheduler.step(vloss)

        all_pred = np.concatenate(all_pred)
        all_pos = np.concatenate(all_pos)

        if mode == 'xy':
            pred_xy = all_pred * xy_std + xy_mean
        else:
            s_p, d_p = decode_sd(all_pred, track.total_len)
            pred_xy = track.sd_to_xy(s_p, d_p)

        xy_err = np.sqrt(((pred_xy - all_pos) ** 2).sum(axis=1))
        print(f"  Train: {tloss:.5f}  Val: {vloss:.5f}  xy_err mean={xy_err.mean():.2f} median={np.median(xy_err):.2f}")

        if vloss < best_val:
            best_val = vloss
            save_checkpoint({
                'state_dict': model.state_dict(),
                'val_loss': vloss,
                'xy_mean': xy_mean, 'xy_std': xy_std,
                'track_total_len': track.total_len,
            }, os.path.join(save_dir, 'best.tar'))

    return model, xy_mean, xy_std


def evaluate_on_test(model, mode, track, xy_mean, xy_std):
    print(f"\n--- Test on UNSEEN episodes [{mode}] ---")
    test_ds = LSTMDataset(TEST_DIR, track, mode=mode, xy_mean=xy_mean, xy_std=xy_std)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, drop_last=False)

    model.eval()
    all_pred, all_pos = [], []
    with torch.no_grad():
        for seq, _, pos in loader:
            pred = model(seq.to(DEVICE)).cpu().numpy()
            all_pred.append(pred)
            all_pos.append(pos.numpy())
    all_pred = np.concatenate(all_pred)
    all_pos = np.concatenate(all_pos)

    if mode == 'xy':
        pred_xy = all_pred * xy_std + xy_mean
    else:
        s_p, d_p = decode_sd(all_pred, track.total_len)
        pred_xy = track.sd_to_xy(s_p, d_p)

    xy_err = np.sqrt(((pred_xy - all_pos) ** 2).sum(axis=1))
    print(f"Samples: {len(xy_err)}")
    print(f"  Mean:   {xy_err.mean():.2f}")
    print(f"  Median: {np.median(xy_err):.2f}")
    print(f"  Max:    {xy_err.max():.2f}")
    print(f"  <5:     {(xy_err < 5).mean():.1%}")
    print(f"  <10:    {(xy_err < 10).mean():.1%}")
    print(f"  <20:    {(xy_err < 20).mean():.1%}")
    return xy_err


if __name__ == "__main__":
    track = build_track(TRAIN_DIR)
    print(f"Track total length: {track.total_len:.1f}")

    # Plan A: LSTM + xy
    model_xy, xy_mean, xy_std = train_variant('xy', track, epochs=20)
    err_xy = evaluate_on_test(model_xy, 'xy', track, xy_mean, xy_std)

    # Plan A+B: LSTM + sd
    model_sd, _, _ = train_variant('sd', track, epochs=20)
    err_sd = evaluate_on_test(model_sd, 'sd', track, xy_mean, xy_std)

    # Summary
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY (xy error on UNSEEN test)")
    print("=" * 60)
    print(f"{'Method':<30s} {'Mean':>8s} {'Median':>8s} {'<10':>8s} {'<20':>8s}")
    print("-" * 70)
    print(f"{'A (LSTM + xy)':<30s} {err_xy.mean():>8.2f} {np.median(err_xy):>8.2f} "
          f"{(err_xy < 10).mean()*100:>7.1f}% {(err_xy < 20).mean()*100:>7.1f}%")
    print(f"{'A+B (LSTM + sd)':<30s} {err_sd.mean():>8.2f} {np.median(err_sd):>8.2f} "
          f"{(err_sd < 10).mean()*100:>7.1f}% {(err_sd < 20).mean()*100:>7.1f}%")
    print("\nPrevious results (single-frame-stack baselines):")
    print(f"{'B only (CNN + sd)':<30s} {'17.39':>8s} {'7.39':>8s} {'61.5':>7s}% {'82.5':>7s}%")
    print(f"{'A only (CNN + xy, original)':<30s} {'26.27':>8s} {'12.82':>8s} {'38.5':>7s}% {'68.6':>7s}%")
