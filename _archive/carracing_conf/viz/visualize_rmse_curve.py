"""Plot RMSE vs Prediction Horizon curve (like PIWM paper)."""

# --- auto-added: make repo root importable when run as a script ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end shim ---


import numpy as np
import torch
import glob
import os
from PIL import Image, ImageDraw

from config import DEVICE, PHYSICS_MEAN, PHYSICS_STD, FRAME_STACK, DYN_SAVE_DIR
from models.dynamics import PhysicsDynamics
from utils import load_checkpoint

DATA_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/train/controller_5/"


def rollout_and_collect_errors(dynamics, data_dir, max_horizon=500):
    """Run rollout on all episodes, collect position error at each step."""
    files = sorted(glob.glob(f"{data_dir}/*.npz"))

    # errors_at_step[t] = list of position errors at step t across all episodes
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
            if T < 10:
                continue

            state_norm = (physics[:T+1] - PHYSICS_MEAN) / PHYSICS_STD
            init = torch.tensor(state_norm[0], dtype=torch.float32).unsqueeze(0).to(DEVICE)

            z = init
            with torch.no_grad():
                for t in range(T):
                    a = torch.tensor(actions[t], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    z_next, _ = dynamics(z, a)

                    # Denormalize predicted position
                    pred_pos = z_next[0, :2].cpu().numpy() * PHYSICS_STD[:2] + PHYSICS_MEAN[:2]
                    gt_pos = physics[t+1, :2]

                    err = np.sqrt(((pred_pos - gt_pos) ** 2).sum())
                    errors_at_step[t].append(err)

                    z = z_next
        except Exception as e:
            print(f"Skipping {f}: {e}")

    return errors_at_step


def draw_rmse_plot(horizons, rmse_values, std_values, save_path, width=800, height=500, margin_l=80, margin_r=30, margin_t=50, margin_b=60):
    """Draw RMSE vs Prediction Horizon plot."""
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    x_max = max(horizons)
    y_max = max(r + s for r, s in zip(rmse_values, std_values)) * 1.1

    def to_px(x_val, y_val):
        px = margin_l + x_val / x_max * plot_w
        py = margin_t + plot_h - y_val / y_max * plot_h
        return int(px), int(py)

    # Grid
    for i in range(6):
        y_val = i * y_max / 5
        _, py = to_px(0, y_val)
        draw.line([(margin_l, py), (width - margin_r, py)], fill=(230, 230, 230), width=1)
        draw.text((margin_l - 45, py - 6), f"{y_val:.0f}", fill=(80, 80, 80))

    for i in range(6):
        x_val = i * x_max / 5
        px, _ = to_px(x_val, 0)
        draw.line([(px, margin_t), (px, height - margin_b)], fill=(230, 230, 230), width=1)
        draw.text((px - 10, height - margin_b + 8), f"{int(x_val)}", fill=(80, 80, 80))

    # Axes
    draw.line([(margin_l, margin_t), (margin_l, height - margin_b)], fill=(0, 0, 0), width=2)
    draw.line([(margin_l, height - margin_b), (width - margin_r, height - margin_b)], fill=(0, 0, 0), width=2)

    # Shaded std region
    upper_points = []
    lower_points = []
    for h, r, s in zip(horizons, rmse_values, std_values):
        px_u, py_u = to_px(h, r + s)
        px_l, py_l = to_px(h, max(0, r - s))
        upper_points.append((px_u, py_u))
        lower_points.append((px_l, py_l))

    # Draw filled polygon for std band
    poly_points = upper_points + lower_points[::-1]
    if len(poly_points) > 2:
        draw.polygon(poly_points, fill=(200, 220, 255))

    # RMSE line
    points = [to_px(h, r) for h, r in zip(horizons, rmse_values)]
    for i in range(len(points) - 1):
        draw.line([points[i], points[i+1]], fill=(0, 80, 220), width=3)

    # Dots at each point
    for p in points:
        draw.ellipse([p[0]-3, p[1]-3, p[0]+3, p[1]+3], fill=(0, 80, 220))

    # Labels
    draw.text((width // 2 - 60, height - 20), "Prediction Horizon (steps)", fill=(0, 0, 0))
    # Y label (vertical)
    for i, c in enumerate("Position RMSE (units)"):
        draw.text((8, margin_t + 20 + i * 12), c, fill=(0, 0, 0))

    draw.text((margin_l + 10, 10), "PIWM Rollout: Position RMSE vs Prediction Horizon", fill=(0, 0, 0))

    # Legend
    lx, ly = width - 220, margin_t + 10
    draw.rectangle([lx, ly, lx + 185, ly + 40], fill=(255, 255, 255), outline=(0, 0, 0))
    draw.line([(lx+10, ly+12), (lx+40, ly+12)], fill=(0, 80, 220), width=3)
    draw.text((lx+45, ly+5), "RMSE (mean)", fill=(0, 0, 0))
    draw.rectangle([lx+10, ly+25, lx+40, ly+33], fill=(200, 220, 255))
    draw.text((lx+45, ly+23), "+/- 1 std", fill=(80, 80, 80))

    img.save(save_path)
    print(f"Saved {save_path}")


def main():
    os.makedirs("vis", exist_ok=True)

    dynamics = PhysicsDynamics().to(DEVICE)
    ckpt = load_checkpoint(os.path.join(DYN_SAVE_DIR, "best.tar"))
    dynamics.load_state_dict(ckpt["dynamics"])
    dynamics.eval()

    print("Running rollouts...")
    errors_at_step = rollout_and_collect_errors(dynamics, DATA_DIR, max_horizon=500)

    # Compute RMSE and std at selected horizons
    horizons = list(range(1, 501, 5))  # every 5 steps
    rmse_values = []
    std_values = []

    for h in horizons:
        if errors_at_step[h-1]:
            errs = np.array(errors_at_step[h-1])
            rmse_values.append(np.sqrt((errs ** 2).mean()))
            std_values.append(errs.std())
        else:
            rmse_values.append(0)
            std_values.append(0)

    # Print table
    print(f"\n{'Horizon':>8s} | {'RMSE':>10s} | {'Std':>10s} | {'N episodes':>10s}")
    print("-" * 45)
    for h in [1, 5, 10, 20, 50, 100, 200, 300, 400, 500]:
        if errors_at_step[h-1]:
            errs = np.array(errors_at_step[h-1])
            rmse = np.sqrt((errs ** 2).mean())
            print(f"{h:>8d} | {rmse:>10.2f} | {errs.std():>10.2f} | {len(errs):>10d}")

    draw_rmse_plot(horizons, rmse_values, std_values, "vis/rmse_vs_horizon.png")


if __name__ == "__main__":
    main()
