"""Visualize rollout: GT vs Integration in two separate panels."""

import numpy as np
import torch
from PIL import Image, ImageDraw

from config import DEVICE, DATA_DIR, PHYSICS_MEAN, PHYSICS_STD, DYN_SAVE_DIR
from models.dynamics import PhysicsDynamics
from utils import load_checkpoint

import os
import glob


def load_episode(data_dir, episode_idx=0):
    files = sorted(glob.glob(f"{data_dir}/*.npz"))
    d = np.load(files[episode_idx], allow_pickle=True)
    pos = d["position"].astype(np.float32)
    yaw = d["yaw"].astype(np.float32)
    vel = d["velocity"].astype(np.float32)
    omega = d["angular_velocity"].astype(np.float32)
    wheel = d["wheel_omega"].astype(np.float32)
    steer = d["steering_angle"].astype(np.float32)
    actions = d["action"].astype(np.float32)
    physics = np.column_stack([pos, yaw, vel, omega, wheel, steer])
    return physics, actions


def rollout_dynamics(dynamics, init_state, actions):
    T = len(actions)
    states = [init_state]
    z = init_state.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        for t in range(T):
            a = torch.tensor(actions[t], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            z_next, _ = dynamics(z, a)
            states.append(z_next.squeeze(0).cpu())
            z = z_next
    return torch.stack(states)


def draw_panel(draw, positions, color, title, x_min, x_max, y_min, y_max,
               offset_x, offset_y, panel_w, panel_h, margin):
    """Draw one trajectory panel."""
    plot_w = panel_w - 2 * margin
    plot_h = panel_h - 2 * margin
    x_range = x_max - x_min + 1e-6
    y_range = y_max - y_min + 1e-6
    scale = min(plot_w / x_range, plot_h / y_range)

    cx = margin + (plot_w - x_range * scale) / 2
    cy = margin + (plot_h - y_range * scale) / 2

    def to_px(x, y):
        px = offset_x + cx + (x - x_min) * scale
        py = offset_y + cy + (y_max - y) * scale
        return int(px), int(py)

    # Axes
    ax_l = offset_x + margin - 5
    ax_r = offset_x + panel_w - margin + 5
    ax_t = offset_y + margin - 5
    ax_b = offset_y + panel_h - margin + 5

    draw.line([(ax_l, ax_b), (ax_r, ax_b)], fill=(0, 0, 0), width=1)
    draw.line([(ax_l, ax_t), (ax_l, ax_b)], fill=(0, 0, 0), width=1)

    # Ticks
    for i in range(6):
        vx = x_min + i * x_range / 5
        px, _ = to_px(vx, y_min)
        draw.line([(px, ax_b), (px, ax_b + 4)], fill=(0, 0, 0))
        draw.text((px - 12, ax_b + 6), f"{vx:.0f}", fill=(80, 80, 80))

        vy = y_min + i * y_range / 5
        _, py = to_px(x_min, vy)
        draw.line([(ax_l - 4, py), (ax_l, py)], fill=(0, 0, 0))
        draw.text((ax_l - 40, py - 5), f"{vy:.0f}", fill=(80, 80, 80))

    # Grid
    for i in range(6):
        vx = x_min + i * x_range / 5
        px, _ = to_px(vx, y_min)
        draw.line([(px, ax_t), (px, ax_b)], fill=(235, 235, 235), width=1)
        vy = y_min + i * y_range / 5
        _, py = to_px(x_min, vy)
        draw.line([(ax_l, py), (ax_r, py)], fill=(235, 235, 235), width=1)

    # Trajectory
    for i in range(len(positions) - 1):
        p1 = to_px(positions[i, 0], positions[i, 1])
        p2 = to_px(positions[i + 1, 0], positions[i + 1, 1])
        draw.line([p1, p2], fill=color, width=3)

    # Start (green) and end dots
    sp = to_px(positions[0, 0], positions[0, 1])
    draw.ellipse([sp[0]-5, sp[1]-5, sp[0]+5, sp[1]+5], fill=(0, 200, 0), outline=(0, 0, 0))
    ep = to_px(positions[-1, 0], positions[-1, 1])
    draw.ellipse([ep[0]-5, ep[1]-5, ep[0]+5, ep[1]+5], fill=color, outline=(0, 0, 0))

    # Title
    draw.text((offset_x + margin, offset_y + 5), title, fill=color)


def draw_comparison(gt_pos, pred_pos, title, size=700, margin=80):
    """Draw GT and Integration overlaid on the same plot."""
    all_x = np.concatenate([gt_pos[:, 0], pred_pos[:, 0]])
    all_y = np.concatenate([gt_pos[:, 1], pred_pos[:, 1]])
    pad = 0.1
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
        px = cx + (x - x_min) * scale
        py = cy + (y_max - y) * scale
        return int(px), int(py)

    img = Image.new('RGB', (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Axes
    ax_l = margin - 5
    ax_r = size - margin + 5
    ax_t = margin - 5
    ax_b = size - margin + 5
    draw.line([(ax_l, ax_b), (ax_r, ax_b)], fill=(0, 0, 0), width=2)
    draw.line([(ax_l, ax_t), (ax_l, ax_b)], fill=(0, 0, 0), width=2)

    # Ticks & grid
    for i in range(6):
        vx = x_min + i * x_range / 5
        px, _ = to_px(vx, y_min)
        draw.line([(px, ax_b), (px, ax_b + 5)], fill=(0, 0, 0))
        draw.text((px - 15, ax_b + 8), f"{vx:.0f}", fill=(80, 80, 80))
        draw.line([(px, ax_t), (px, ax_b)], fill=(235, 235, 235), width=1)

        vy = y_min + i * y_range / 5
        _, py = to_px(x_min, vy)
        draw.line([(ax_l - 5, py), (ax_l, py)], fill=(0, 0, 0))
        draw.text((ax_l - 45, py - 5), f"{vy:.0f}", fill=(80, 80, 80))
        draw.line([(ax_l, py), (ax_r, py)], fill=(235, 235, 235), width=1)

    draw.text((size // 2 - 10, size - 18), "X", fill=(0, 0, 0))
    draw.text((5, size // 2), "Y", fill=(0, 0, 0))

    # GT trajectory (blue, solid, thick)
    for i in range(len(gt_pos) - 1):
        p1 = to_px(gt_pos[i, 0], gt_pos[i, 1])
        p2 = to_px(gt_pos[i + 1, 0], gt_pos[i + 1, 1])
        draw.line([p1, p2], fill=(0, 80, 255), width=4)

    # Integration trajectory (red, dashed, on top)
    dash_on = 12
    dash_off = 10
    # Collect all pixel points first
    px_points = [to_px(pred_pos[i, 0], pred_pos[i, 1]) for i in range(len(pred_pos))]
    # Walk along polyline with dash pattern
    accum = 0.0
    drawing = True
    for i in range(len(px_points) - 1):
        p1 = px_points[i]
        p2 = px_points[i + 1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        seg_len = (dx**2 + dy**2) ** 0.5
        if seg_len < 0.5:
            continue
        # How far along this segment
        pos_in_seg = 0.0
        while pos_in_seg < seg_len:
            cycle_len = dash_on if drawing else dash_off
            remaining = cycle_len - accum
            can_go = min(remaining, seg_len - pos_in_seg)
            if drawing:
                t0 = pos_in_seg / seg_len
                t1 = (pos_in_seg + can_go) / seg_len
                sp = (int(p1[0] + dx * t0), int(p1[1] + dy * t0))
                ep = (int(p1[0] + dx * t1), int(p1[1] + dy * t1))
                draw.line([sp, ep], fill=(255, 40, 40), width=3)
            pos_in_seg += can_go
            accum += can_go
            if accum >= cycle_len:
                accum = 0.0
                drawing = not drawing

    # Start point (green)
    sp = to_px(gt_pos[0, 0], gt_pos[0, 1])
    draw.ellipse([sp[0]-7, sp[1]-7, sp[0]+7, sp[1]+7], fill=(0, 200, 0), outline=(0, 0, 0), width=2)

    # GT end (blue)
    ep_gt = to_px(gt_pos[-1, 0], gt_pos[-1, 1])
    draw.ellipse([ep_gt[0]-6, ep_gt[1]-6, ep_gt[0]+6, ep_gt[1]+6], fill=(0, 80, 255), outline=(0, 0, 0), width=2)

    # Pred end (red)
    ep_pred = to_px(pred_pos[-1, 0], pred_pos[-1, 1])
    draw.ellipse([ep_pred[0]-6, ep_pred[1]-6, ep_pred[0]+6, ep_pred[1]+6], fill=(255, 40, 40), outline=(0, 0, 0), width=2)

    # Error line between endpoints
    draw.line([ep_gt, ep_pred], fill=(200, 200, 200), width=1)

    # Legend
    lx, ly = size - 200, 10
    draw.rectangle([lx, ly, lx + 185, ly + 85], fill=(255, 255, 255), outline=(0, 0, 0))
    draw.line([(lx+10, ly+15), (lx+45, ly+15)], fill=(0, 80, 255), width=4)
    draw.text((lx+50, ly+8), "Ground Truth", fill=(0, 0, 0))
    # dashed red line in legend
    draw.line([(lx+10, ly+35), (lx+22, ly+35)], fill=(255, 40, 40), width=3)
    draw.line([(lx+30, ly+35), (lx+45, ly+35)], fill=(255, 40, 40), width=3)
    draw.text((lx+50, ly+28), "Integration", fill=(0, 0, 0))
    draw.ellipse([lx+22, ly+48, lx+34, ly+60], fill=(0, 200, 0))
    draw.text((lx+50, ly+48), "Start", fill=(0, 0, 0))

    # Title + error
    err = np.sqrt(((gt_pos[-1] - pred_pos[-1]) ** 2).sum())
    draw.text((margin, 5), f"{title}  |  Final error: {err:.1f} units", fill=(0, 0, 0))

    return img


def main():
    os.makedirs("vis", exist_ok=True)

    dynamics = PhysicsDynamics().to(DEVICE)
    ckpt = load_checkpoint(os.path.join(DYN_SAVE_DIR, "best.tar"))
    dynamics.load_state_dict(ckpt["dynamics"])
    dynamics.eval()

    horizons = [100, 200, 500]
    episodes = [0, 5, 10, 20]

    for ep_idx in episodes:
        physics, actions = load_episode(DATA_DIR, ep_idx)
        T = min(len(actions), max(horizons))
        if T < 100:
            continue

        state_norm = (physics[:T+1] - PHYSICS_MEAN) / PHYSICS_STD
        init = torch.tensor(state_norm[0], dtype=torch.float32)
        pred_states = rollout_dynamics(dynamics, init, actions[:T]).numpy()

        gt_pos = physics[:T+1, :2]
        pred_pos = pred_states[:T+1, :2] * PHYSICS_STD[:2] + PHYSICS_MEAN[:2]

        for h in horizons:
            if h > T:
                continue
            img = draw_comparison(gt_pos[:h+1], pred_pos[:h+1],
                                  f"Episode {ep_idx}, {h} steps")
            img.save(f"vis/rollout_ep{ep_idx}_{h}steps.png")
            err = np.sqrt(((gt_pos[h] - pred_pos[h]) ** 2).sum())
            print(f"Episode {ep_idx}, {h:4d} steps: final error = {err:.2f} units")

    print("\nSaved to vis/rollout_ep*_*steps.png")


if __name__ == "__main__":
    main()
