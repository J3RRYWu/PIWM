"""Visualize CoarsePositionHead predictions vs GT on fixed track."""

# --- auto-added: make repo root importable when run as a script ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end shim ---


import numpy as np
import torch
from PIL import Image, ImageDraw
import glob
import os

from config import DEVICE, PHYSICS_MEAN, PHYSICS_STD, FRAME_STACK
from models.localization import CoarsePositionHead
from utils import load_checkpoint

FIXED_DATA_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/train/fixed/"


def load_all_data():
    files = sorted(glob.glob(f"{FIXED_DATA_DIR}/*.npz"))
    all_frames = []
    all_gt_pos = []
    episode_ids = []

    for ep_idx, f in enumerate(files):
        try:
            d = np.load(f, allow_pickle=True)
            imgs = d["imgs"].astype(np.float32)
            if imgs.max() > 1.0:
                imgs /= 255.0
            pos = d["position"].astype(np.float32)
            n = len(imgs)
            if n < FRAME_STACK + 1:
                continue

            for t in range(FRAME_STACK - 1, n):
                frames = np.stack([imgs[t - FRAME_STACK + 1 + i] for i in range(FRAME_STACK)], axis=0)
                all_frames.append(frames)
                all_gt_pos.append(pos[t])
                episode_ids.append(ep_idx)
        except:
            pass

    return np.array(all_frames), np.array(all_gt_pos), np.array(episode_ids)


def main():
    os.makedirs("vis", exist_ok=True)

    # Load model
    ckpt = load_checkpoint("checkpoints/localization/coarse_best.tar")
    model = CoarsePositionHead().to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Load data
    print("Loading data...")
    all_frames, all_gt_pos, episode_ids = load_all_data()
    print(f"Total samples: {len(all_frames)}")

    # Predict all positions
    print("Predicting...")
    all_pred_pos = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(all_frames), batch_size):
            batch = torch.tensor(all_frames[i:i+batch_size], dtype=torch.float32).to(DEVICE)
            pred_norm = model(batch).cpu().numpy()
            pred_real = pred_norm * PHYSICS_STD[:2] + PHYSICS_MEAN[:2]
            all_pred_pos.append(pred_real)
    all_pred_pos = np.concatenate(all_pred_pos)

    gt = all_gt_pos
    pred = all_pred_pos

    # === Plot 1: Full track overlay ===
    size = 800
    margin = 80
    all_x = np.concatenate([gt[:, 0], pred[:, 0]])
    all_y = np.concatenate([gt[:, 1], pred[:, 1]])
    pad = 0.05
    x_min = all_x.min() - (all_x.max() - all_x.min()) * pad - 1
    x_max = all_x.max() + (all_x.max() - all_x.min()) * pad + 1
    y_min = all_y.min() - (all_y.max() - all_y.min()) * pad - 1
    y_max = all_y.max() + (all_y.max() - all_y.min()) * pad + 1
    x_range = x_max - x_min
    y_range = y_max - y_min
    plot_size = size - 2 * margin
    scale = min(plot_size / x_range, plot_size / y_range)
    cx = margin + (plot_size - x_range * scale) / 2
    cy = margin + (plot_size - y_range * scale) / 2

    def to_px(x, y):
        return int(cx + (x - x_min) * scale), int(cy + (y_max - y) * scale)

    img = Image.new('RGB', (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Axes
    ax_l, ax_r = margin - 5, size - margin + 5
    ax_t, ax_b = margin - 5, size - margin + 5
    draw.line([(ax_l, ax_b), (ax_r, ax_b)], fill=(0, 0, 0), width=2)
    draw.line([(ax_l, ax_t), (ax_l, ax_b)], fill=(0, 0, 0), width=2)
    for i in range(6):
        vx = x_min + i * x_range / 5
        px, _ = to_px(vx, y_min)
        draw.line([(px, ax_b), (px, ax_b + 5)], fill=(0, 0, 0))
        draw.text((px - 15, ax_b + 8), f"{vx:.0f}", fill=(80, 80, 80))
        draw.line([(px, ax_t), (px, ax_b)], fill=(240, 240, 240))
        vy = y_min + i * y_range / 5
        _, py = to_px(x_min, vy)
        draw.line([(ax_l - 5, py), (ax_l, py)], fill=(0, 0, 0))
        draw.text((ax_l - 45, py - 5), f"{vy:.0f}", fill=(80, 80, 80))
        draw.line([(ax_l, py), (ax_r, py)], fill=(240, 240, 240))

    # Draw GT points (blue, small)
    for i in range(0, len(gt), 3):  # subsample for speed
        px, py = to_px(gt[i, 0], gt[i, 1])
        draw.ellipse([px-1, py-1, px+1, py+1], fill=(0, 80, 255))

    # Draw pred points (red, small)
    for i in range(0, len(pred), 3):
        px, py = to_px(pred[i, 0], pred[i, 1])
        draw.ellipse([px-1, py-1, px+1, py+1], fill=(255, 40, 40, 128))

    # Legend
    lx, ly = size - 200, 10
    draw.rectangle([lx, ly, lx + 185, ly + 55], fill=(255, 255, 255), outline=(0, 0, 0))
    draw.ellipse([lx+10, ly+10, lx+20, ly+20], fill=(0, 80, 255))
    draw.text((lx+25, ly+8), "Ground Truth", fill=(0, 0, 0))
    draw.ellipse([lx+10, ly+32, lx+20, ly+42], fill=(255, 40, 40))
    draw.text((lx+25, ly+30), "Coarse Prediction", fill=(0, 0, 0))

    draw.text((margin, 5), "CoarsePositionHead: All predictions on fixed track", fill=(0, 0, 0))
    draw.text((margin, size - 18), "X", fill=(0, 0, 0))
    draw.text((5, size // 2), "Y", fill=(0, 0, 0))

    img.save("vis/coarse_all.png")
    print("Saved vis/coarse_all.png")

    # === Plot 2: Error arrows (GT -> Pred) for a single episode ===
    ep0_mask = episode_ids == 0
    gt_ep = gt[ep0_mask]
    pred_ep = pred[ep0_mask]

    img2 = Image.new('RGB', (size, size), (255, 255, 255))
    draw2 = ImageDraw.Draw(img2)

    # Same axes
    draw2.line([(ax_l, ax_b), (ax_r, ax_b)], fill=(0, 0, 0), width=2)
    draw2.line([(ax_l, ax_t), (ax_l, ax_b)], fill=(0, 0, 0), width=2)
    for i in range(6):
        vx = x_min + i * x_range / 5
        px, _ = to_px(vx, y_min)
        draw2.line([(px, ax_b), (px, ax_b + 5)], fill=(0, 0, 0))
        draw2.text((px - 15, ax_b + 8), f"{vx:.0f}", fill=(80, 80, 80))
        draw2.line([(px, ax_t), (px, ax_b)], fill=(240, 240, 240))
        vy = y_min + i * y_range / 5
        _, py = to_px(x_min, vy)
        draw2.line([(ax_l - 5, py), (ax_l, py)], fill=(0, 0, 0))
        draw2.text((ax_l - 45, py - 5), f"{vy:.0f}", fill=(80, 80, 80))
        draw2.line([(ax_l, py), (ax_r, py)], fill=(240, 240, 240))

    # GT trajectory (blue line)
    for i in range(len(gt_ep) - 1):
        p1 = to_px(gt_ep[i, 0], gt_ep[i, 1])
        p2 = to_px(gt_ep[i + 1, 0], gt_ep[i + 1, 1])
        draw2.line([p1, p2], fill=(0, 80, 255), width=3)

    # Error arrows every N frames
    step = max(1, len(gt_ep) // 40)
    for i in range(0, len(gt_ep), step):
        pg = to_px(gt_ep[i, 0], gt_ep[i, 1])
        pp = to_px(pred_ep[i, 0], pred_ep[i, 1])
        draw2.line([pg, pp], fill=(255, 40, 40), width=1)
        draw2.ellipse([pp[0]-3, pp[1]-3, pp[0]+3, pp[1]+3], fill=(255, 40, 40))

    # Error stats
    errors = np.sqrt(((gt_ep - pred_ep) ** 2).sum(axis=1))
    draw2.text((margin, 5),
        f"Episode 0: mean err={errors.mean():.2f}, max={errors.max():.2f}, median={np.median(errors):.2f} units",
        fill=(0, 0, 0))

    lx, ly = size - 220, 25
    draw2.rectangle([lx, ly, lx + 205, ly + 55], fill=(255, 255, 255), outline=(0, 0, 0))
    draw2.line([(lx+10, ly+15), (lx+40, ly+15)], fill=(0, 80, 255), width=3)
    draw2.text((lx+45, ly+8), "GT trajectory", fill=(0, 0, 0))
    draw2.line([(lx+10, ly+37), (lx+40, ly+37)], fill=(255, 40, 40), width=1)
    draw2.ellipse([lx+22, ly+33, lx+28, ly+39], fill=(255, 40, 40))
    draw2.text((lx+45, ly+30), "Prediction error", fill=(0, 0, 0))

    img2.save("vis/coarse_errors.png")
    print("Saved vis/coarse_errors.png")

    # Print stats
    all_errors = np.sqrt(((gt - pred) ** 2).sum(axis=1))
    print(f"\nOverall stats:")
    print(f"  Mean error:   {all_errors.mean():.2f} units")
    print(f"  Median error: {np.median(all_errors):.2f} units")
    print(f"  Max error:    {all_errors.max():.2f} units")
    print(f"  <1 unit:      {(all_errors < 1).mean():.1%}")
    print(f"  <5 units:     {(all_errors < 5).mean():.1%}")
    print(f"  <10 units:    {(all_errors < 10).mean():.1%}")


if __name__ == "__main__":
    main()
