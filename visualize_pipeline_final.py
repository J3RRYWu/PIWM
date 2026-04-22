"""Visualize optimal pipeline: LSTM+sd (initial localization) + Dynamics integration.

Pipeline:
  t=0..9:   Car drives, collect 10 frames
  t=9:      LSTM+sd estimates initial (x, y)
  t=10+:    Dynamics integration for 100 steps
"""

import numpy as np
import torch
import glob
import os
from PIL import Image, ImageDraw

from config import DEVICE, PHYSICS_MEAN, PHYSICS_STD, FRAME_STACK, DYN_SAVE_DIR
from models.dynamics import PhysicsDynamics
from models.localization_lstm import LSTMLocalizer
from track_utils import TrackCoords
from utils import load_checkpoint

TEST_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/test/fixed/"
TRAIN_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/train/fixed/"
HORIZON = 100
SEQ_LEN = 10
D_NORM = 10.0


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


def decode_sd(pred_single, total_len):
    s_sin, s_cos, d_norm = pred_single[0], pred_single[1], pred_single[2]
    theta = np.arctan2(s_sin, s_cos)
    theta = np.mod(theta, 2 * np.pi)
    return float(theta * total_len / (2 * np.pi)), float(d_norm * D_NORM)


def run_pipeline(lstm_model, dynamics, track, imgs, physics, actions, horizon):
    """
    Returns:
        t_start: frame index where pipeline prediction starts
        gt_pos: (horizon+1, 2) ground truth
        pred_pos: (horizon+1, 2) predicted
        init_pos: initial position from LSTM+sd
    """
    # Build sequence of SEQ_LEN frame-stacks ending at frame index (SEQ_LEN - 1) + (FRAME_STACK - 1)
    t_start = (SEQ_LEN - 1) + (FRAME_STACK - 1)  # first valid LSTM-predictable frame

    seq = []
    for offset in range(SEQ_LEN):
        center = t_start - SEQ_LEN + 1 + offset
        stack = np.stack([imgs[center - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
        seq.append(stack)
    seq = np.stack(seq, axis=0)

    # LSTM+sd -> initial (x, y)
    with torch.no_grad():
        seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        pred_sd = lstm_model(seq_t).cpu().numpy()[0]
    s_init, d_init = decode_sd(pred_sd, track.total_len)
    xy_init = track.sd_to_xy(s_init, d_init)  # (x, y)

    # Build initial full state: use LSTM's xy but GT for other dims
    init_state = physics[t_start].copy()
    init_state[0] = xy_init[0]
    init_state[1] = xy_init[1]
    init_norm = (init_state - PHYSICS_MEAN) / PHYSICS_STD

    # Integrate with Dynamics
    T = min(horizon, len(actions) - t_start - 1)
    pred_positions = [np.array(xy_init)]
    z = torch.tensor(init_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        for t in range(T):
            a = torch.tensor(actions[t_start + t], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            z_next, _ = dynamics(z, a)
            p = z_next[0, :2].cpu().numpy() * PHYSICS_STD[:2] + PHYSICS_MEAN[:2]
            pred_positions.append(p)
            z = z_next

    pred_positions = np.array(pred_positions)
    gt_positions = physics[t_start:t_start + T + 1, :2]
    return t_start, gt_positions, pred_positions, np.array(xy_init)


def draw_single(draw, gt, pred, xy_init,
                x_min, x_max, y_min, y_max,
                ox, oy, pw, ph, margin, title):
    plot_w = pw - 2 * margin
    plot_h = ph - 2 * margin
    xr = x_max - x_min + 1e-6
    yr = y_max - y_min + 1e-6
    scale = min(plot_w / xr, plot_h / yr)
    cx = ox + margin + (plot_w - xr * scale) / 2
    cy = oy + margin + (plot_h - yr * scale) / 2

    def to_px(x, y):
        return int(cx + (x - x_min) * scale), int(cy + (y_max - y) * scale)

    # Axes + grid
    al, ar = ox + margin - 3, ox + pw - margin + 3
    at, ab = oy + margin - 3, oy + ph - margin + 3
    draw.line([(al, ab), (ar, ab)], fill=(0, 0, 0))
    draw.line([(al, at), (al, ab)], fill=(0, 0, 0))
    for i in range(4):
        vx = x_min + i * xr / 3
        px, _ = to_px(vx, y_min)
        draw.text((px - 10, ab + 3), f"{vx:.0f}", fill=(120, 120, 120))
        vy = y_min + i * yr / 3
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
        seg = (dx**2 + dy**2) ** 0.5
        if seg < 0.5:
            continue
        pos_in = 0.0
        while pos_in < seg:
            cl = dash_on if drawing else dash_off
            can = min(cl - accum, seg - pos_in)
            if drawing:
                t0, t1 = pos_in/seg, (pos_in+can)/seg
                s = (int(pts[i][0]+dx*t0), int(pts[i][1]+dy*t0))
                e = (int(pts[i][0]+dx*t1), int(pts[i][1]+dy*t1))
                draw.line([s, e], fill=(255, 40, 40), width=2)
            pos_in += can
            accum += can
            if accum >= cl:
                accum = 0; drawing = not drawing

    # GT start (green)
    sp = to_px(gt[0, 0], gt[0, 1])
    draw.ellipse([sp[0]-5, sp[1]-5, sp[0]+5, sp[1]+5], fill=(0, 200, 0), outline=(0, 0, 0))
    # LSTM init estimate (orange diamond)
    cp = to_px(xy_init[0], xy_init[1])
    draw.polygon([(cp[0], cp[1]-6), (cp[0]+6, cp[1]), (cp[0], cp[1]+6), (cp[0]-6, cp[1])],
                 fill=(255, 165, 0), outline=(0, 0, 0))
    # End points
    eg = to_px(gt[-1, 0], gt[-1, 1])
    ep = to_px(pred[-1, 0], pred[-1, 1])
    draw.ellipse([eg[0]-4, eg[1]-4, eg[0]+4, eg[1]+4], fill=(0, 80, 255), outline=(0, 0, 0))
    draw.ellipse([ep[0]-4, ep[1]-4, ep[0]+4, ep[1]+4], fill=(255, 40, 40), outline=(0, 0, 0))

    init_err = np.sqrt(((gt[0] - xy_init) ** 2).sum())
    final_err = np.sqrt(((gt[-1] - pred[-1]) ** 2).sum())
    draw.text((ox + margin, oy + 5),
              f"{title}  init={init_err:.1f}  final={final_err:.1f}",
              fill=(0, 0, 0))


def main():
    os.makedirs("vis", exist_ok=True)

    # Track
    track_files = sorted(glob.glob(f"{TRAIN_DIR}/*.npz"))
    d0 = np.load(track_files[0], allow_pickle=True)
    track = TrackCoords(d0["map"])

    # Dynamics
    dynamics = PhysicsDynamics().to(DEVICE)
    ckpt = load_checkpoint(os.path.join(DYN_SAVE_DIR, "best.tar"))
    dynamics.load_state_dict(ckpt["dynamics"])
    dynamics.eval()

    # LSTM+sd (best localizer)
    lstm = LSTMLocalizer(out_dim=3).to(DEVICE)
    lckpt = load_checkpoint("checkpoints/localization_lstm_sd/best.tar")
    lstm.load_state_dict(lckpt["state_dict"])
    lstm.eval()

    # Run on 5 test episodes
    files = sorted(glob.glob(f"{TEST_DIR}/*.npz"))
    n_eps = min(5, len(files))

    results = []
    for i in range(n_eps):
        imgs, physics, actions = load_episode(TEST_DIR, i)
        t_start, gt, pred, xy_init = run_pipeline(lstm, dynamics, track, imgs, physics, actions, HORIZON)
        results.append((gt, pred, xy_init))
        print(f"Episode {i}: init_err={np.sqrt(((gt[0]-xy_init)**2).sum()):.2f}, "
              f"final_err={np.sqrt(((gt[-1]-pred[-1])**2).sum()):.2f}")

    # Global coordinate range
    all_x = np.concatenate([g[:, 0] for g, _, _ in results] + [p[:, 0] for _, p, _ in results])
    all_y = np.concatenate([g[:, 1] for g, _, _ in results] + [p[:, 1] for _, p, _ in results])
    pad = 0.1
    x_min = all_x.min() - (all_x.max() - all_x.min()) * pad - 1
    x_max = all_x.max() + (all_x.max() - all_x.min()) * pad + 1
    y_min = all_y.min() - (all_y.max() - all_y.min()) * pad - 1
    y_max = all_y.max() + (all_y.max() - all_y.min()) * pad + 1

    pw, ph = 400, 350
    margin = 50
    img_w = pw * 3
    img_h = ph * 2 + 80
    img = Image.new('RGB', (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    positions = [(0, 30), (pw, 30), (2*pw, 30), (0, ph+40), (pw, ph+40)]

    for i in range(n_eps):
        gt, pred, xy_init = results[i]
        ox, oy = positions[i]
        draw_single(draw, gt, pred, xy_init,
                    x_min, x_max, y_min, y_max,
                    ox, oy, pw, ph, margin, f"Ep {i}")

    draw.text((10, 5),
              f"Full Pipeline: LSTM+sd initial localization (t=9) + Dynamics integration ({HORIZON} steps)",
              fill=(0, 0, 0))

    # Legend in empty panel space (bottom right)
    lx, ly = 2 * pw + 50, ph + 100
    draw.rectangle([lx, ly, lx + 310, ly + 105], fill=(255, 255, 255), outline=(0, 0, 0))
    draw.line([(lx+10, ly+15), (lx+45, ly+15)], fill=(0, 80, 255), width=3)
    draw.text((lx+50, ly+8), "Ground Truth", fill=(0, 0, 0))
    draw.line([(lx+10, ly+35), (lx+22, ly+35)], fill=(255, 40, 40), width=2)
    draw.line([(lx+30, ly+35), (lx+45, ly+35)], fill=(255, 40, 40), width=2)
    draw.text((lx+50, ly+28), "Pipeline prediction", fill=(0, 0, 0))
    draw.ellipse([lx+22, ly+51, lx+34, ly+63], fill=(0, 200, 0))
    draw.text((lx+50, ly+51), "GT start (t=9)", fill=(0, 0, 0))
    draw.polygon([(lx+28, ly+73), (lx+34, ly+79), (lx+28, ly+85), (lx+22, ly+79)],
                 fill=(255, 165, 0))
    draw.text((lx+50, ly+74), "LSTM+sd initial estimate", fill=(0, 0, 0))

    img.save("vis/pipeline_final_5eps.png")
    print("\nSaved vis/pipeline_final_5eps.png")


if __name__ == "__main__":
    main()
