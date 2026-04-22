"""Visualize full pipeline: CoarseHead initial position + Dynamics integration for 100 steps.
Shows 5 episodes side by side, GT vs predicted trajectory."""

import numpy as np
import torch
import glob
import os
from PIL import Image, ImageDraw

from config import DEVICE, PHYSICS_MEAN, PHYSICS_STD, FRAME_STACK, DYN_SAVE_DIR
from models.dynamics import PhysicsDynamics
from models.localization import CoarsePositionHead
from utils import load_checkpoint

TEST_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/test/fixed/"
HORIZON = 100


def load_episode(data_dir, ep_idx):
    files = sorted(glob.glob(f"{data_dir}/*.npz"))
    d = np.load(files[ep_idx], allow_pickle=True)
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
    return imgs, physics, actions


def run_pipeline(coarse_head, dynamics, imgs, physics, actions, horizon):
    """Run full pipeline: CoarseHead for initial pos, then Dynamics integration."""
    start_t = FRAME_STACK - 1  # first valid frame

    # Step 1: CoarseHead estimates initial (x, y)
    frames_init = np.stack([imgs[start_t - FRAME_STACK + 1 + i] for i in range(FRAME_STACK)], axis=0)
    with torch.no_grad():
        inp = torch.tensor(frames_init, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        pred_pos_norm = coarse_head(inp).cpu().numpy()[0]
    pred_xy = pred_pos_norm * PHYSICS_STD[:2] + PHYSICS_MEAN[:2]

    # Step 2: Build initial full state using CoarseHead xy + GT for other dims
    gt_state = physics[start_t]
    init_state = gt_state.copy()
    init_state[0] = pred_xy[0]  # x from CoarseHead
    init_state[1] = pred_xy[1]  # y from CoarseHead
    init_norm = (init_state - PHYSICS_MEAN) / PHYSICS_STD

    # Step 3: Dynamics integration
    T = min(horizon, len(actions) - start_t - 1)
    pred_positions = [pred_xy.copy()]
    z = torch.tensor(init_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        for t in range(T):
            a = torch.tensor(actions[start_t + t], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            z_next, _ = dynamics(z, a)
            pos = z_next[0, :2].cpu().numpy() * PHYSICS_STD[:2] + PHYSICS_MEAN[:2]
            pred_positions.append(pos)
            z = z_next

    pred_positions = np.array(pred_positions)
    gt_positions = physics[start_t:start_t + T + 1, :2]

    return gt_positions, pred_positions


def draw_single(draw, gt, pred, x_min, x_max, y_min, y_max,
                ox, oy, pw, ph, margin, ep_idx):
    """Draw one episode panel."""
    plot_w = pw - 2 * margin
    plot_h = ph - 2 * margin
    x_range = x_max - x_min + 1e-6
    y_range = y_max - y_min + 1e-6
    scale = min(plot_w / x_range, plot_h / y_range)
    cx = ox + margin + (plot_w - x_range * scale) / 2
    cy = oy + margin + (plot_h - y_range * scale) / 2

    def to_px(x, y):
        return int(cx + (x - x_min) * scale), int(cy + (y_max - y) * scale)

    # Grid
    al = ox + margin - 3
    ar = ox + pw - margin + 3
    at = oy + margin - 3
    ab = oy + ph - margin + 3
    draw.line([(al, ab), (ar, ab)], fill=(0, 0, 0), width=1)
    draw.line([(al, at), (al, ab)], fill=(0, 0, 0), width=1)

    for i in range(4):
        vx = x_min + i * x_range / 3
        px, _ = to_px(vx, y_min)
        draw.text((px - 10, ab + 3), f"{vx:.0f}", fill=(120, 120, 120))
        vy = y_min + i * y_range / 3
        _, py = to_px(x_min, vy)
        draw.text((al - 30, py - 5), f"{vy:.0f}", fill=(120, 120, 120))

    # GT (blue solid)
    for i in range(len(gt) - 1):
        p1, p2 = to_px(gt[i, 0], gt[i, 1]), to_px(gt[i+1, 0], gt[i+1, 1])
        draw.line([p1, p2], fill=(0, 80, 255), width=3)

    # Pred (red dashed)
    pts = [to_px(pred[i, 0], pred[i, 1]) for i in range(len(pred))]
    dash_on, dash_off = 12, 10
    accum, drawing = 0.0, True
    for i in range(len(pts) - 1):
        dx = pts[i+1][0] - pts[i][0]
        dy = pts[i+1][1] - pts[i][1]
        seg_len = (dx**2 + dy**2) ** 0.5
        if seg_len < 0.5:
            continue
        pos_in = 0.0
        while pos_in < seg_len:
            cl = dash_on if drawing else dash_off
            can = min(cl - accum, seg_len - pos_in)
            if drawing:
                t0, t1 = pos_in / seg_len, (pos_in + can) / seg_len
                s = (int(pts[i][0] + dx*t0), int(pts[i][1] + dy*t0))
                e = (int(pts[i][0] + dx*t1), int(pts[i][1] + dy*t1))
                draw.line([s, e], fill=(255, 40, 40), width=2)
            pos_in += can
            accum += can
            if accum >= cl:
                accum = 0
                drawing = not drawing

    # Start
    sp = to_px(gt[0, 0], gt[0, 1])
    draw.ellipse([sp[0]-5, sp[1]-5, sp[0]+5, sp[1]+5], fill=(0, 200, 0), outline=(0, 0, 0))

    # CoarseHead initial estimate (orange diamond)
    cp = to_px(pred[0, 0], pred[0, 1])
    draw.polygon([(cp[0], cp[1]-6), (cp[0]+6, cp[1]), (cp[0], cp[1]+6), (cp[0]-6, cp[1])],
                 fill=(255, 165, 0), outline=(0, 0, 0))

    # End points
    eg = to_px(gt[-1, 0], gt[-1, 1])
    ep = to_px(pred[-1, 0], pred[-1, 1])
    draw.ellipse([eg[0]-4, eg[1]-4, eg[0]+4, eg[1]+4], fill=(0, 80, 255), outline=(0, 0, 0))
    draw.ellipse([ep[0]-4, ep[1]-4, ep[0]+4, ep[1]+4], fill=(255, 40, 40), outline=(0, 0, 0))

    # Error
    err_init = np.sqrt(((gt[0] - pred[0]) ** 2).sum())
    err_final = np.sqrt(((gt[-1] - pred[-1]) ** 2).sum())
    draw.text((ox + margin, oy + 5),
              f"Ep {ep_idx}  init_err={err_init:.1f}  final_err={err_final:.1f}",
              fill=(0, 0, 0))


def main():
    os.makedirs("vis", exist_ok=True)

    # Load models
    dynamics = PhysicsDynamics().to(DEVICE)
    dyn_ckpt = load_checkpoint(os.path.join(DYN_SAVE_DIR, "best.tar"))
    dynamics.load_state_dict(dyn_ckpt["dynamics"])
    dynamics.eval()

    coarse = CoarsePositionHead().to(DEVICE)
    c_ckpt = load_checkpoint("checkpoints/localization/coarse_best.tar")
    coarse.load_state_dict(c_ckpt["state_dict"])
    coarse.eval()

    files = sorted(glob.glob(f"{TEST_DIR}/*.npz"))
    n_eps = min(5, len(files))

    # Run pipeline for each episode
    all_gt, all_pred = [], []
    for i in range(n_eps):
        imgs, physics, actions = load_episode(TEST_DIR, i)
        gt, pred = run_pipeline(coarse, dynamics, imgs, physics, actions, HORIZON)
        all_gt.append(gt)
        all_pred.append(pred)

    # Global coordinate range
    all_x = np.concatenate([g[:, 0] for g in all_gt] + [p[:, 0] for p in all_pred])
    all_y = np.concatenate([g[:, 1] for g in all_gt] + [p[:, 1] for p in all_pred])
    pad = 0.1
    x_min = all_x.min() - (all_x.max() - all_x.min()) * pad - 1
    x_max = all_x.max() + (all_x.max() - all_x.min()) * pad + 1
    y_min = all_y.min() - (all_y.max() - all_y.min()) * pad - 1
    y_max = all_y.max() + (all_y.max() - all_y.min()) * pad + 1

    # Draw 5 panels: top row 3, bottom row 2
    pw, ph = 400, 350
    margin = 50
    img_w = pw * 3
    img_h = ph * 2 + 80
    img = Image.new('RGB', (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    positions = [(0, 30), (pw, 30), (2*pw, 30), (0, ph+40), (pw, ph+40)]

    for i in range(n_eps):
        ox, oy = positions[i]
        draw_single(draw, all_gt[i], all_pred[i],
                    x_min, x_max, y_min, y_max,
                    ox, oy, pw, ph, margin, i)

    # Title
    draw.text((10, 5), f"Full Pipeline: CoarseHead init + Dynamics integration ({HORIZON} steps)  |  Blue=GT, Red dashed=Predicted, Orange=CoarseHead init",
              fill=(0, 0, 0))

    # Legend in last panel space if <5 episodes, or bottom right
    lx, ly = 2 * pw + 50, ph + 80
    draw.rectangle([lx, ly, lx + 300, ly + 100], fill=(255, 255, 255), outline=(0, 0, 0))
    draw.line([(lx+10, ly+15), (lx+45, ly+15)], fill=(0, 80, 255), width=3)
    draw.text((lx+50, ly+8), "Ground Truth", fill=(0, 0, 0))
    draw.line([(lx+10, ly+35), (lx+22, ly+35)], fill=(255, 40, 40), width=2)
    draw.line([(lx+30, ly+35), (lx+45, ly+35)], fill=(255, 40, 40), width=2)
    draw.text((lx+50, ly+28), "Pipeline prediction", fill=(0, 0, 0))
    draw.ellipse([lx+22, ly+48, lx+34, ly+60], fill=(0, 200, 0))
    draw.text((lx+50, ly+48), "GT start", fill=(0, 0, 0))
    draw.polygon([(lx+28, ly+65), (lx+34, ly+71), (lx+28, ly+77), (lx+22, ly+71)],
                 fill=(255, 165, 0))
    draw.text((lx+50, ly+68), "CoarseHead estimate", fill=(0, 0, 0))

    img.save("vis/pipeline_100steps_5eps.png")
    print("Saved vis/pipeline_100steps_5eps.png")

    # Print summary
    for i in range(n_eps):
        err_init = np.sqrt(((all_gt[i][0] - all_pred[i][0]) ** 2).sum())
        err_final = np.sqrt(((all_gt[i][-1] - all_pred[i][-1]) ** 2).sum())
        print(f"Episode {i}: init_err={err_init:.2f}, final_err={err_final:.2f}")


if __name__ == "__main__":
    main()
