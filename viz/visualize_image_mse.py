"""Image reconstruction MSE vs rollout step.

At step k:
  1. Start from GT state at t=0
  2. Dynamics rollout k steps -> state_k
  3. Decode 9-dim (encoder dims) to image
  4. Compare to actual image at t=k (MSE)
"""

# --- auto-added: make repo root importable when run as a script ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end shim ---


import numpy as np
import torch
import torch.nn.functional as F
import glob
import os
from PIL import Image, ImageDraw

from config import (DEVICE, PHYSICS_MEAN, PHYSICS_STD, FRAME_STACK,
                    DYN_SAVE_DIR, AE_SAVE_DIR, ENCODER_STATE_INDICES,
                    ENCODER_MEAN, ENCODER_STD)
from models.dynamics import PhysicsDynamics
from models.decoder import PhysicsDecoder
from utils import load_checkpoint

DATA_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/train/controller_5/"
MAX_STEPS = 100


def main():
    os.makedirs("vis", exist_ok=True)

    dynamics = PhysicsDynamics().to(DEVICE)
    dynamics.load_state_dict(load_checkpoint(os.path.join(DYN_SAVE_DIR, "best.tar"))["dynamics"])
    dynamics.eval()

    decoder = PhysicsDecoder().to(DEVICE)
    ae_ckpt = load_checkpoint(os.path.join(AE_SAVE_DIR, "best.tar"))
    decoder.load_state_dict(ae_ckpt["decoder"])
    decoder.eval()

    files = sorted(glob.glob(f"{DATA_DIR}/*.npz"))

    mses_at_step = {k: [] for k in range(MAX_STEPS + 1)}

    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            imgs = d["imgs"].astype(np.float32)
            if imgs.max() > 1.0:
                imgs /= 255.0
            pos = d["position"].astype(np.float32)
            yaw = d["yaw"].astype(np.float32)
            vel = d["velocity"].astype(np.float32)
            omega = d["angular_velocity"].astype(np.float32)
            wheel = d["wheel_omega"].astype(np.float32)
            steer = d["steering_angle"].astype(np.float32)
            actions = d["action"].astype(np.float32)
            physics = np.column_stack([pos, yaw, vel, omega, wheel, steer])

            T = min(MAX_STEPS + 1, len(imgs), len(actions) + 1)
            if T < 10:
                continue

            # Start from GT state at t=0
            init_state = physics[0]
            init_norm = (init_state - PHYSICS_MEAN) / PHYSICS_STD
            z = torch.tensor(init_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # At step 0: decode init state
            with torch.no_grad():
                enc_dims = z[:, ENCODER_STATE_INDICES]  # (1, 9) normalized with PHYSICS_STD (for those indices)
                # But the encoder normalization uses ENCODER_MEAN/STD, which equals PHYSICS_MEAN/STD at those indices
                # So enc_dims is already in encoder's normalized space
                recon = decoder(enc_dims).cpu().numpy()[0, 0]
            mse = ((recon - imgs[0]) ** 2).mean()
            mses_at_step[0].append(mse)

            # Rollout
            with torch.no_grad():
                for k in range(1, T):
                    a = torch.tensor(actions[k - 1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    z_next, _ = dynamics(z, a)
                    enc_dims = z_next[:, ENCODER_STATE_INDICES]
                    recon = decoder(enc_dims).cpu().numpy()[0, 0]
                    mse = ((recon - imgs[k]) ** 2).mean()
                    mses_at_step[k].append(mse)
                    z = z_next
        except Exception as e:
            print(f"Skipping {f}: {e}")

    # Compute mean and std
    steps = []
    means = []
    stds = []
    for k in range(MAX_STEPS + 1):
        if mses_at_step[k]:
            arr = np.array(mses_at_step[k])
            steps.append(k)
            means.append(arr.mean())
            stds.append(arr.std())

    # Print table
    print(f"{'Step':>5s} {'MSE mean':>12s} {'MSE std':>12s} {'N':>5s}")
    print("-" * 40)
    for k in [0, 1, 5, 10, 20, 30, 50, 75, 100]:
        if k < len(steps) and mses_at_step[k]:
            arr = np.array(mses_at_step[k])
            print(f"{k:>5d} {arr.mean():>12.6f} {arr.std():>12.6f} {len(arr):>5d}")

    # Plot
    width, height = 800, 500
    ml, mr, mt, mb = 80, 30, 50, 60
    pw = width - ml - mr
    ph = height - mt - mb

    x_max = max(steps)
    y_max = max(m + s for m, s in zip(means, stds)) * 1.1

    def to_px(x, y):
        return int(ml + x / x_max * pw), int(mt + ph - y / y_max * ph)

    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for i in range(6):
        yv = i * y_max / 5
        _, py = to_px(0, yv)
        draw.line([(ml, py), (width - mr, py)], fill=(230, 230, 230))
        draw.text((ml - 55, py - 6), f"{yv:.4f}", fill=(80, 80, 80))
    for i in range(0, int(x_max) + 1, 10):
        px, _ = to_px(i, 0)
        draw.line([(px, mt), (px, height - mb)], fill=(230, 230, 230))
        draw.text((px - 6, height - mb + 8), f"{i}", fill=(80, 80, 80))

    draw.line([(ml, mt), (ml, height - mb)], fill=(0, 0, 0), width=2)
    draw.line([(ml, height - mb), (width - mr, height - mb)], fill=(0, 0, 0), width=2)

    # Std band
    upper = [to_px(s, m + st) for s, m, st in zip(steps, means, stds)]
    lower = [to_px(s, max(0, m - st)) for s, m, st in zip(steps, means, stds)]
    if len(upper) > 2:
        draw.polygon(upper + lower[::-1], fill=(220, 230, 245))

    # Mean line
    pts = [to_px(s, m) for s, m in zip(steps, means)]
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i+1]], fill=(0, 80, 220), width=3)
    for p in pts:
        draw.ellipse([p[0]-2, p[1]-2, p[0]+2, p[1]+2], fill=(0, 80, 220))

    draw.text((width // 2 - 100, height - 20), "Prediction Horizon (steps)", fill=(0, 0, 0))
    draw.text((5, mt + ph // 2 - 20), "Image", fill=(0, 0, 0))
    draw.text((5, mt + ph // 2 - 5), "MSE", fill=(0, 0, 0))
    draw.text((ml + 10, 10), "PIWM Decoder Image MSE vs Rollout Step", fill=(0, 0, 0))

    # Legend
    lx, ly = width - 200, mt + 10
    draw.rectangle([lx, ly, lx + 165, ly + 40], fill=(255, 255, 255), outline=(0, 0, 0))
    draw.line([(lx+10, ly+12), (lx+40, ly+12)], fill=(0, 80, 220), width=3)
    draw.text((lx+45, ly+5), "Mean", fill=(0, 0, 0))
    draw.rectangle([lx+10, ly+25, lx+40, ly+33], fill=(220, 230, 245))
    draw.text((lx+45, ly+23), "+/- 1 std", fill=(80, 80, 80))

    img.save("vis/image_mse_vs_step.png")
    print("Saved vis/image_mse_vs_step.png")


if __name__ == "__main__":
    main()
