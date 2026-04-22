"""Train action-augmented LSTM localizer, xy and sd variants + ablation."""

import os, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from config import DEVICE, FRAME_STACK
from models.localization_lstm_action import LSTMActionLocalizer
from track_utils import TrackCoords
from utils import save_checkpoint, load_checkpoint

TRAIN_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/train/fixed/"
TEST_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/test/fixed/"
SEQ_LEN = 10
D_NORM = 10.0


class LSTMActionDataset(Dataset):
    def __init__(self, data_dir, track, mode='sd', xy_mean=None, xy_std=None):
        self.mode = mode
        self.track = track
        files = sorted(glob.glob(f"{data_dir}/*.npz"))
        all_imgs, all_pos, all_actions = [], [], []
        for f in files:
            try:
                d = np.load(f, allow_pickle=True)
                imgs = d["imgs"].astype(np.float32)
                if imgs.max() > 1.0: imgs /= 255.0
                pos = d["position"].astype(np.float32)
                actions = d["action"].astype(np.float32)
                if len(imgs) < FRAME_STACK + SEQ_LEN:
                    continue
                all_imgs.append(imgs)
                all_pos.append(pos)
                all_actions.append(actions)
            except: pass
        self.all_imgs = all_imgs
        self.all_pos = all_pos
        self.all_actions = all_actions

        concat = np.concatenate(all_pos)
        if xy_mean is None:
            self.xy_mean = concat.mean(axis=0).astype(np.float32)
            self.xy_std = concat.std(axis=0).astype(np.float32)
        else:
            self.xy_mean, self.xy_std = xy_mean, xy_std

        self.indices = []
        for ep_idx, imgs in enumerate(all_imgs):
            min_end = (SEQ_LEN - 1) + (FRAME_STACK - 1)
            for end_frame in range(min_end, len(imgs)):
                self.indices.append((ep_idx, end_frame))
        print(f"LSTMActionDataset @ {data_dir} [{mode}]: {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, end = self.indices[idx]
        imgs = self.all_imgs[ep_idx]
        actions = self.all_actions[ep_idx]

        seq_frames = []
        seq_actions = []
        for off in range(SEQ_LEN):
            center = end - SEQ_LEN + 1 + off
            stack = np.stack([imgs[center - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
            seq_frames.append(stack)
            seq_actions.append(actions[center])  # action taken AT that frame
        seq_frames = np.stack(seq_frames, axis=0)
        seq_actions = np.stack(seq_actions, axis=0)

        pos = self.all_pos[ep_idx][end]
        if self.mode == 'xy':
            target = ((pos - self.xy_mean) / self.xy_std).astype(np.float32)
        else:
            s, d = self.track.xy_to_sd(pos)
            theta = 2 * np.pi * s / self.track.total_len
            target = np.array([np.sin(theta), np.cos(theta), d / D_NORM], dtype=np.float32)

        return (torch.tensor(seq_frames, dtype=torch.float32),
                torch.tensor(seq_actions, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32),
                torch.tensor(pos, dtype=torch.float32))


def decode_sd(pred, total_len):
    theta = np.arctan2(pred[:, 0], pred[:, 1])
    theta = np.mod(theta, 2 * np.pi)
    s = theta * total_len / (2 * np.pi)
    d = pred[:, 2] * D_NORM
    return s, d


def train_variant(mode, track, epochs=20):
    print(f"\n{'='*60}\nTraining LSTM+Action [{mode}]\n{'='*60}")
    full_ds = LSTMActionDataset(TRAIN_DIR, track, mode=mode)
    xy_mean, xy_std = full_ds.xy_mean, full_ds.xy_std

    n_val = int(len(full_ds) * 0.1)
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=True)

    out_dim = 2 if mode == 'xy' else 3
    model = LSTMActionLocalizer(out_dim=out_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    mse = nn.MSELoss()
    best_val = float('inf')
    save_dir = f"checkpoints/localization_lstm_action_{mode}/"

    for epoch in range(epochs):
        model.train()
        tloss, nb = 0, 0
        for fr, ac, tgt, _ in tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs}"):
            fr, ac, tgt = fr.to(DEVICE), ac.to(DEVICE), tgt.to(DEVICE)
            pred = model(fr, ac)
            loss = mse(pred, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            tloss += loss.item(); nb += 1
        tloss /= nb

        model.eval()
        vloss, vn = 0, 0
        all_p, all_pos = [], []
        with torch.no_grad():
            for fr, ac, tgt, pos in val_loader:
                fr, ac, tgt = fr.to(DEVICE), ac.to(DEVICE), tgt.to(DEVICE)
                pred = model(fr, ac)
                vloss += mse(pred, tgt).item(); vn += 1
                all_p.append(pred.cpu().numpy()); all_pos.append(pos.numpy())
        vloss /= vn
        sch.step(vloss)

        all_p = np.concatenate(all_p); all_pos = np.concatenate(all_pos)
        if mode == 'xy':
            pred_xy = all_p * xy_std + xy_mean
        else:
            s_p, d_p = decode_sd(all_p, track.total_len)
            pred_xy = track.sd_to_xy(s_p, d_p)
        err = np.sqrt(((pred_xy - all_pos) ** 2).sum(axis=1))
        print(f"  Train:{tloss:.5f} Val:{vloss:.5f} xy_err mean={err.mean():.2f} median={np.median(err):.2f}")

        if vloss < best_val:
            best_val = vloss
            save_checkpoint({
                'state_dict': model.state_dict(), 'val_loss': vloss,
                'xy_mean': xy_mean, 'xy_std': xy_std,
                'track_total_len': track.total_len,
            }, os.path.join(save_dir, 'best.tar'))
    return model, xy_mean, xy_std


def eval_test(model, mode, track, xy_mean, xy_std):
    print(f"\n--- Test UNSEEN [{mode} + action] ---")
    ds = LSTMActionDataset(TEST_DIR, track, mode=mode, xy_mean=xy_mean, xy_std=xy_std)
    loader = DataLoader(ds, batch_size=32, shuffle=False, drop_last=False)
    model.eval()
    all_p, all_pos = [], []
    with torch.no_grad():
        for fr, ac, _, pos in loader:
            pred = model(fr.to(DEVICE), ac.to(DEVICE)).cpu().numpy()
            all_p.append(pred); all_pos.append(pos.numpy())
    all_p = np.concatenate(all_p); all_pos = np.concatenate(all_pos)
    if mode == 'xy':
        pred_xy = all_p * xy_std + xy_mean
    else:
        s_p, d_p = decode_sd(all_p, track.total_len)
        pred_xy = track.sd_to_xy(s_p, d_p)
    err = np.sqrt(((pred_xy - all_pos) ** 2).sum(axis=1))
    print(f"  Samples: {len(err)}")
    print(f"  Mean:{err.mean():.2f} Median:{np.median(err):.2f} Max:{err.max():.2f}")
    print(f"  <5:{(err<5).mean():.1%} <10:{(err<10).mean():.1%} <20:{(err<20).mean():.1%}")
    return err


if __name__ == "__main__":
    files = sorted(glob.glob(f"{TRAIN_DIR}/*.npz"))
    d0 = np.load(files[0], allow_pickle=True)
    track = TrackCoords(d0["map"])
    print(f"Track total_len: {track.total_len:.1f}")

    # xy variant
    m_xy, xy_m, xy_s = train_variant('xy', track, epochs=20)
    err_xy = eval_test(m_xy, 'xy', track, xy_m, xy_s)

    # sd variant
    m_sd, _, _ = train_variant('sd', track, epochs=20)
    err_sd = eval_test(m_sd, 'sd', track, xy_m, xy_s)

    print("\n" + "=" * 75)
    print("FULL ABLATION (xy error on UNSEEN test)")
    print("=" * 75)
    print(f"{'Method':<34s} {'Mean':>8s} {'Median':>8s} {'<10':>8s} {'<20':>8s}")
    print("-" * 75)
    print(f"{'CNN + xy (baseline)':<34s} {'26.27':>8s} {'12.82':>8s} {'38.5':>7s}% {'68.6':>7s}%")
    print(f"{'CNN + sd':<34s} {'17.39':>8s} {'7.39':>8s} {'61.5':>7s}% {'82.5':>7s}%")
    print(f"{'LSTM + xy':<34s} {'10.16':>8s} {'3.21':>8s} {'86.6':>7s}% {'92.5':>7s}%")
    print(f"{'LSTM + sd':<34s} {'9.76':>8s} {'2.91':>8s} {'88.2':>7s}% {'94.4':>7s}%")
    print(f"{'LSTM + action + xy':<34s} {err_xy.mean():>8.2f} {np.median(err_xy):>8.2f} "
          f"{(err_xy<10).mean()*100:>7.1f}% {(err_xy<20).mean()*100:>7.1f}%")
    print(f"{'LSTM + action + sd':<34s} {err_sd.mean():>8.2f} {np.median(err_sd):>8.2f} "
          f"{(err_sd<10).mean()*100:>7.1f}% {(err_sd<20).mean()*100:>7.1f}%")
