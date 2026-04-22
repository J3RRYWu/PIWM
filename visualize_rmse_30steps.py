"""Plot RMSE vs Prediction Horizon for first 30 steps only."""

import numpy as np
import torch
import glob
import os
from PIL import Image, ImageDraw

from config import DEVICE, PHYSICS_MEAN, PHYSICS_STD, DYN_SAVE_DIR
from models.dynamics import PhysicsDynamics
from utils import load_checkpoint

DATA_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/train/controller_5/"


def main():
    os.makedirs("vis", exist_ok=True)

    dynamics = PhysicsDynamics().to(DEVICE)
    ckpt = load_checkpoint(os.path.join(DYN_SAVE_DIR, "best.tar"))
    dynamics.load_state_dict(ckpt["dynamics"])
    dynamics.eval()

    files = sorted(glob.glob(f"{DATA_DIR}/*.npz"))
    max_horizon = 30

    errors_at_step = {t: [] for t in range(max_horizon)}

    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            pos = d["position"].astype(np.float32)
            yaw = d["yaw"].astype(np.float32)
            vel = d["velocity"].astype(np.float32)
            omega = d["angular_velocity"].astype(np.float32)
            wheel = d["wheel_omega"].astype(np.float32)
            steer = d["steering_angle"].astype(np.float32)
            actions = d["action"].astype(np.float32)
            physics = np.column_stack([pos, yaw, vel, omega, wheel, steer])

            T = min(len(actions), max_horizon)
            if T < max_horizon:
                continue

            state_norm = (physics[:T+1] - PHYSICS_MEAN) / PHYSICS_STD
            z = torch.tensor(state_norm[0], dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                for t in range(T):
                    a = torch.tensor(actions[t], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    z_next, _ = dynamics(z, a)
                    pred_pos = z_next[0, :2].cpu().numpy() * PHYSICS_STD[:2] + PHYSICS_MEAN[:2]
                    gt_pos = physics[t+1, :2]
                    err = np.sqrt(((pred_pos - gt_pos) ** 2).sum())
                    errors_at_step[t].append(err)
                    z = z_next
        except:
            pass

    # Compute RMSE and std
    horizons = list(range(1, max_horizon + 1))
    rmse_vals = []
    std_vals = []
    for h in horizons:
        errs = np.array(errors_at_step[h-1])
        rmse_vals.append(np.sqrt((errs ** 2).mean()))
        std_vals.append(errs.std())

    # Print table
    print(f"{'Step':>6s} | {'RMSE':>8s} | {'Std':>8s} | {'N':>5s}")
    print("-" * 35)
    for h, r, s in zip(horizons, rmse_vals, std_vals):
        print(f"{h:>6d} | {r:>8.3f} | {s:>8.3f} | {len(errors_at_step[h-1]):>5d}")

    # Draw plot
    width, height = 800, 500
    ml, mr, mt, mb = 80, 30, 50, 60
    pw = width - ml - mr
    ph = height - mt - mb

    x_max = max_horizon
    y_max = max(r + s for r, s in zip(rmse_vals, std_vals)) * 1.15

    def to_px(x, y):
        return int(ml + x / x_max * pw), int(mt + ph - y / y_max * ph)

    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Grid + ticks
    for i in range(7):
        y_val = i * y_max / 6
        _, py = to_px(0, y_val)
        draw.line([(ml, py), (width - mr, py)], fill=(230, 230, 230))
        draw.text((ml - 45, py - 6), f"{y_val:.2f}", fill=(80, 80, 80))

    for i in range(0, max_horizon + 1, 5):
        px, _ = to_px(i, 0)
        draw.line([(px, mt), (px, height - mb)], fill=(230, 230, 230))
        draw.text((px - 5, height - mb + 8), f"{i}", fill=(80, 80, 80))

    # Axes
    draw.line([(ml, mt), (ml, height - mb)], fill=(0, 0, 0), width=2)
    draw.line([(ml, height - mb), (width - mr, height - mb)], fill=(0, 0, 0), width=2)

    # Std band
    upper = [to_px(h, r + s) for h, r, s in zip(horizons, rmse_vals, std_vals)]
    lower = [to_px(h, max(0, r - s)) for h, r, s in zip(horizons, rmse_vals, std_vals)]
    poly = upper + lower[::-1]
    if len(poly) > 2:
        draw.polygon(poly, fill=(200, 220, 255))

    # RMSE line
    points = [to_px(h, r) for h, r in zip(horizons, rmse_vals)]
    for i in range(len(points) - 1):
        draw.line([points[i], points[i+1]], fill=(0, 80, 220), width=3)
    for p in points:
        draw.ellipse([p[0]-3, p[1]-3, p[0]+3, p[1]+3], fill=(0, 80, 220))

    # Labels
    draw.text((width // 2 - 60, height - 18), "Prediction Horizon (steps)", fill=(0, 0, 0))
    draw.text((5, mt + ph // 2 - 30), "RMSE", fill=(0, 0, 0))
    draw.text((5, mt + ph // 2 - 15), "(units)", fill=(0, 0, 0))
    draw.text((ml + 10, 10), "PIWM Rollout: Position RMSE (first 30 steps)", fill=(0, 0, 0))

    # Legend
    lx, ly = width - 200, mt + 10
    draw.rectangle([lx, ly, lx + 165, ly + 40], fill=(255, 255, 255), outline=(0, 0, 0))
    draw.line([(lx+10, ly+12), (lx+40, ly+12)], fill=(0, 80, 220), width=3)
    draw.text((lx+45, ly+5), "RMSE (mean)", fill=(0, 0, 0))
    draw.rectangle([lx+10, ly+25, lx+40, ly+33], fill=(200, 220, 255))
    draw.text((lx+45, ly+23), "+/- 1 std", fill=(80, 80, 80))

    img.save("vis/rmse_30steps.png")
    print(f"\nSaved vis/rmse_30steps.png")


if __name__ == "__main__":
    main()
