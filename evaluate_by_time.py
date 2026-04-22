"""Evaluate all localization methods as a function of time since episode start.

Answers: for each time step t after episode start, which method is best?
"""

import numpy as np
import torch
import glob
import os
from PIL import Image, ImageDraw

from config import DEVICE, PHYSICS_MEAN, PHYSICS_STD, FRAME_STACK
from models.localization import CoarsePositionHead
from models.localization_sd import CoarsePositionHeadSD
from models.localization_lstm import LSTMLocalizer
from track_utils import TrackCoords
from utils import load_checkpoint

TEST_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/test/fixed/"
TRAIN_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/train/fixed/"
D_NORM = 10.0
SEQ_LEN = 10


def decode_sd(pred, total_len):
    s_sin, s_cos, d_norm = pred[:, 0], pred[:, 1], pred[:, 2]
    theta = np.arctan2(s_sin, s_cos)
    theta = np.mod(theta, 2 * np.pi)
    s = theta * total_len / (2 * np.pi)
    d = d_norm * D_NORM
    return s, d


def load_all_models(track):
    models = {}

    # A: CNN + xy
    m = CoarsePositionHead().to(DEVICE)
    ckpt = load_checkpoint("checkpoints/localization/coarse_best.tar")
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    models['A: CNN+xy'] = ('cnn_xy', m)

    # B: CNN + sd
    m = CoarsePositionHeadSD().to(DEVICE)
    ckpt = load_checkpoint("checkpoints/localization_sd/best.tar")
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    models['B: CNN+sd'] = ('cnn_sd', m)

    # Plan A: LSTM + xy
    m = LSTMLocalizer(out_dim=2).to(DEVICE)
    ckpt = load_checkpoint("checkpoints/localization_lstm_xy/best.tar")
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    xy_mean = ckpt.get("xy_mean", PHYSICS_MEAN[:2])
    xy_std = ckpt.get("xy_std", PHYSICS_STD[:2])
    models['A: LSTM+xy'] = ('lstm_xy', m, xy_mean, xy_std)

    # Plan A+B: LSTM + sd
    m = LSTMLocalizer(out_dim=3).to(DEVICE)
    ckpt = load_checkpoint("checkpoints/localization_lstm_sd/best.tar")
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    models['A+B: LSTM+sd'] = ('lstm_sd', m)

    return models


def evaluate_at_time(methods, track, max_t=50):
    """For each method and each time t, compute xy error over all test episodes."""
    files = sorted(glob.glob(f"{TEST_DIR}/*.npz"))

    # errors[method_name][t] = list of errors at time t across episodes
    errors = {name: {t: [] for t in range(max_t)} for name in methods}

    for f in files:
        d = np.load(f, allow_pickle=True)
        imgs = d["imgs"].astype(np.float32)
        if imgs.max() > 1.0: imgs /= 255.0
        pos = d["position"].astype(np.float32)
        n = min(max_t + FRAME_STACK, len(imgs))

        with torch.no_grad():
            # Precompute all frame-stacks for this episode
            stacks = []
            for t in range(FRAME_STACK - 1, n):
                stack = np.stack([imgs[t - FRAME_STACK + 1 + i] for i in range(FRAME_STACK)], axis=0)
                stacks.append(stack)
            stacks = np.array(stacks, dtype=np.float32)  # (T, 3, 64, 64)

            for name, info in methods.items():
                kind = info[0]
                model = info[1]

                if kind == 'cnn_xy':
                    inp = torch.tensor(stacks).to(DEVICE)
                    pred = model(inp).cpu().numpy()
                    pred_xy = pred * PHYSICS_STD[:2] + PHYSICS_MEAN[:2]
                    for idx in range(len(pred_xy)):
                        t_frame = (FRAME_STACK - 1) + idx  # absolute time
                        t_rel = idx  # time since first valid frame
                        if t_rel < max_t:
                            err = np.sqrt(((pred_xy[idx] - pos[t_frame]) ** 2).sum())
                            errors[name][t_rel].append(err)

                elif kind == 'cnn_sd':
                    inp = torch.tensor(stacks).to(DEVICE)
                    pred = model(inp).cpu().numpy()
                    s, d = decode_sd(pred, track.total_len)
                    pred_xy = track.sd_to_xy(s, d)
                    for idx in range(len(pred_xy)):
                        t_frame = (FRAME_STACK - 1) + idx
                        t_rel = idx
                        if t_rel < max_t:
                            err = np.sqrt(((pred_xy[idx] - pos[t_frame]) ** 2).sum())
                            errors[name][t_rel].append(err)

                elif kind in ('lstm_xy', 'lstm_sd'):
                    # Need SEQ_LEN stacks to make first prediction
                    # At stack index i, we predict using stacks[i-SEQ_LEN+1 : i+1]
                    for i in range(SEQ_LEN - 1, len(stacks)):
                        seq = stacks[i - SEQ_LEN + 1 : i + 1]  # (SEQ_LEN, 3, 64, 64)
                        seq_t = torch.tensor(seq).unsqueeze(0).to(DEVICE)
                        pred = model(seq_t).cpu().numpy()[0]
                        t_frame = (FRAME_STACK - 1) + i
                        t_rel = i

                        if kind == 'lstm_xy':
                            if name == 'A: LSTM+xy':
                                xy_mean, xy_std = info[2], info[3]
                            pred_xy = pred * xy_std + xy_mean
                        else:  # lstm_sd
                            s_val, d_val = decode_sd(pred[None], track.total_len)
                            pred_xy = track.sd_to_xy(s_val, d_val)[0]

                        if t_rel < max_t:
                            err = np.sqrt(((pred_xy - pos[t_frame]) ** 2).sum())
                            errors[name][t_rel].append(err)

    return errors


def draw_curves(errors, max_t, save_path):
    """Plot error vs time since episode start for all methods."""
    width, height = 900, 550
    ml, mr, mt, mb = 70, 30, 50, 70
    pw = width - ml - mr
    ph = height - mt - mb

    # Compute median error per time step
    curves = {}
    for name, t_dict in errors.items():
        ts = []
        meds = []
        for t in range(max_t):
            if t_dict[t]:
                ts.append(t)
                meds.append(np.median(t_dict[t]))
        curves[name] = (ts, meds)

    x_max = max_t
    y_max = max(max(meds) for _, meds in curves.values() if meds) * 1.1

    def to_px(x, y):
        return int(ml + x / x_max * pw), int(mt + ph - y / y_max * ph)

    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Grid
    for i in range(7):
        yv = i * y_max / 6
        _, py = to_px(0, yv)
        draw.line([(ml, py), (width - mr, py)], fill=(230, 230, 230))
        draw.text((ml - 45, py - 6), f"{yv:.1f}", fill=(80, 80, 80))
    for i in range(0, max_t + 1, 5):
        px, _ = to_px(i, 0)
        draw.line([(px, mt), (px, height - mb)], fill=(230, 230, 230))
        draw.text((px - 5, height - mb + 8), f"{i}", fill=(80, 80, 80))

    # Axes
    draw.line([(ml, mt), (ml, height - mb)], fill=(0, 0, 0), width=2)
    draw.line([(ml, height - mb), (width - mr, height - mb)], fill=(0, 0, 0), width=2)

    colors = {
        'A: CNN+xy': (200, 40, 40),       # red
        'B: CNN+sd': (40, 180, 80),       # green
        'A: LSTM+xy': (220, 120, 0),      # orange
        'A+B: LSTM+sd': (0, 80, 220),     # blue
    }

    # Plot curves
    for name, (ts, meds) in curves.items():
        color = colors.get(name, (100, 100, 100))
        pts = [to_px(t, m) for t, m in zip(ts, meds)]
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i+1]], fill=color, width=3)
        for p in pts:
            draw.ellipse([p[0]-3, p[1]-3, p[0]+3, p[1]+3], fill=color)

    # Labels
    draw.text((width // 2 - 100, height - 20), "Time since episode start (frames)", fill=(0, 0, 0))
    draw.text((5, mt + ph // 2 - 20), "Median", fill=(0, 0, 0))
    draw.text((5, mt + ph // 2 - 5), "xy err", fill=(0, 0, 0))
    draw.text((5, mt + ph // 2 + 10), "(units)", fill=(0, 0, 0))
    draw.text((ml + 10, 10), "Localization error vs time since episode start", fill=(0, 0, 0))

    # Legend
    lx, ly = width - 220, mt + 10
    draw.rectangle([lx, ly, lx + 205, ly + 90], fill=(255, 255, 255), outline=(0, 0, 0))
    y_off = ly + 8
    for name, color in colors.items():
        draw.line([(lx+10, y_off+6), (lx+40, y_off+6)], fill=color, width=3)
        draw.text((lx+45, y_off), name, fill=(0, 0, 0))
        y_off += 20

    img.save(save_path)
    print(f"Saved {save_path}")


def main():
    track_files = sorted(glob.glob(f"{TRAIN_DIR}/*.npz"))
    d = np.load(track_files[0], allow_pickle=True)
    track = TrackCoords(d["map"])

    print("Loading models...")
    models = load_all_models(track)

    print("Evaluating...")
    max_t = 30
    errors = evaluate_at_time(models, track, max_t=max_t)

    # Print summary table
    print(f"\n{'Time':>5s}", end="")
    for name in models:
        print(f" {name:>14s}", end="")
    print()
    print("-" * (5 + 15 * len(models)))

    for t in [0, 2, 4, 8, 10, 15, 20, 25, 29]:
        print(f"{t:>5d}", end="")
        for name in models:
            if errors[name][t]:
                med = np.median(errors[name][t])
                print(f" {med:>14.2f}", end="")
            else:
                print(f" {'--':>14s}", end="")
        print()

    draw_curves(errors, max_t, "vis/error_vs_time.png")


if __name__ == "__main__":
    main()
