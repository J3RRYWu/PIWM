"""Animated GIF: rolling prediction using full pipeline.

Each frame t:
  - t < 10:   only GT dots accumulated (car "observing")
  - t >= 10:  LSTM+sd estimates current position from last 10 frames
              Dynamics predicts 30 steps forward
              Animation shows sliding prediction trajectory vs GT
"""

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
from models.localization_lstm import LSTMLocalizer
from track_utils import TrackCoords
from utils import load_checkpoint

TEST_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/test/fixed/"
TRAIN_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/train/fixed/"

SEQ_LEN = 10
HORIZON_PRED = 30  # predict 30 steps ahead
N_FRAMES = 120    # total animation length
D_NORM = 10.0


def decode_sd(pred, total_len):
    s_sin, s_cos, d_norm = pred[0], pred[1], pred[2]
    theta = np.arctan2(s_sin, s_cos)
    theta = np.mod(theta, 2 * np.pi)
    return float(theta * total_len / (2 * np.pi)), float(d_norm * D_NORM)


def estimate_state(lstm, track, imgs, physics, t):
    """Use LSTM+sd on frames [t-SEQ_LEN+1, t] to estimate position at time t.
    Returns full 11-dim state (with LSTM xy, GT others)."""
    seq = []
    for offset in range(SEQ_LEN):
        center = t - SEQ_LEN + 1 + offset
        stack = np.stack([imgs[center - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
        seq.append(stack)
    seq = np.stack(seq, axis=0)

    with torch.no_grad():
        pred_sd = lstm(torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
    s_v, d_v = decode_sd(pred_sd, track.total_len)
    xy = track.sd_to_xy(s_v, d_v)  # (x, y)

    state = physics[t].copy()
    state[0] = xy[0]
    state[1] = xy[1]
    return state, np.array([xy[0], xy[1]])


def predict_forward(dynamics, init_state, actions, horizon):
    """Predict horizon steps forward from init_state."""
    z = torch.tensor((init_state - PHYSICS_MEAN) / PHYSICS_STD,
                     dtype=torch.float32).unsqueeze(0).to(DEVICE)
    positions = [init_state[:2].copy()]
    with torch.no_grad():
        for t in range(horizon):
            if t >= len(actions):
                break
            a = torch.tensor(actions[t], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            z_next, _ = dynamics(z, a)
            p = z_next[0, :2].cpu().numpy() * PHYSICS_STD[:2] + PHYSICS_MEAN[:2]
            positions.append(p)
            z = z_next
    return np.array(positions)


def render_frame(t, gt_full, pred_traj, est_pos,
                 x_min, x_max, y_min, y_max,
                 width=700, height=600, margin=80):
    """Render a single animation frame."""
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    pw = width - 2 * margin
    ph = height - 2 * margin - 40
    xr = x_max - x_min
    yr = y_max - y_min
    scale = min(pw / xr, ph / yr)
    cx = margin + (pw - xr * scale) / 2
    cy = margin + 40 + (ph - yr * scale) / 2

    def to_px(x, y):
        return int(cx + (x - x_min) * scale), int(cy + (y_max - y) * scale)

    # Axes
    al, ar = margin - 3, width - margin + 3
    at, ab = margin + 37, height - margin + 3
    draw.line([(al, ab), (ar, ab)], fill=(0, 0, 0), width=1)
    draw.line([(al, at), (al, ab)], fill=(0, 0, 0), width=1)
    for i in range(6):
        vx = x_min + i * xr / 5
        px, _ = to_px(vx, y_min)
        draw.line([(px, ab), (px, ab + 4)], fill=(0, 0, 0))
        draw.line([(px, at), (px, ab)], fill=(240, 240, 240))
        draw.text((px - 12, ab + 6), f"{vx:.0f}", fill=(80, 80, 80))
        vy = y_min + i * yr / 5
        _, py = to_px(x_min, vy)
        draw.line([(al - 4, py), (al, py)], fill=(0, 0, 0))
        draw.line([(al, py), (ar, py)], fill=(240, 240, 240))
        draw.text((al - 45, py - 6), f"{vy:.0f}", fill=(80, 80, 80))

    # === GT past trajectory (blue solid) ===
    gt_past = gt_full[:t+1]
    if len(gt_past) >= 2:
        for i in range(len(gt_past) - 1):
            p1, p2 = to_px(gt_past[i, 0], gt_past[i, 1]), to_px(gt_past[i+1, 0], gt_past[i+1, 1])
            draw.line([p1, p2], fill=(0, 80, 255), width=3)

    # === GT future trajectory (light blue) for context ===
    gt_future = gt_full[t:t + HORIZON_PRED + 1] if t + HORIZON_PRED + 1 <= len(gt_full) else gt_full[t:]
    if len(gt_future) >= 2:
        for i in range(len(gt_future) - 1):
            p1, p2 = to_px(gt_future[i, 0], gt_future[i, 1]), to_px(gt_future[i+1, 0], gt_future[i+1, 1])
            draw.line([p1, p2], fill=(150, 200, 255), width=2)

    # === Current GT position ===
    cur = to_px(gt_full[t, 0], gt_full[t, 1])
    draw.ellipse([cur[0]-6, cur[1]-6, cur[0]+6, cur[1]+6], fill=(0, 80, 255), outline=(0, 0, 0), width=2)

    # === Prediction (only if we have 10+ frames) ===
    if pred_traj is not None:
        # Red dashed
        pts = [to_px(p[0], p[1]) for p in pred_traj]
        dash_on, dash_off = 10, 8
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

        # Prediction endpoint marker
        end_px = to_px(pred_traj[-1, 0], pred_traj[-1, 1])
        draw.ellipse([end_px[0]-4, end_px[1]-4, end_px[0]+4, end_px[1]+4], fill=(255, 40, 40), outline=(0, 0, 0))

        # LSTM initial estimate marker
        if est_pos is not None:
            ep = to_px(est_pos[0], est_pos[1])
            draw.polygon([(ep[0], ep[1]-6), (ep[0]+6, ep[1]), (ep[0], ep[1]+6), (ep[0]-6, ep[1])],
                         fill=(255, 165, 0), outline=(0, 0, 0))

    # === Title/status ===
    if t < SEQ_LEN:
        status = f"Frame {t}: Observing (need {SEQ_LEN} frames to start prediction)"
    else:
        init_err = np.sqrt(((gt_full[t] - est_pos) ** 2).sum()) if est_pos is not None else 0
        status = f"Frame {t}: LSTM+sd localization (init_err={init_err:.1f}) + Dynamics {HORIZON_PRED}-step prediction"
    draw.text((margin, 10), status, fill=(0, 0, 0))

    # Legend (top right)
    lx, ly = width - 210, 30
    draw.rectangle([lx, ly, lx + 195, ly + 85], fill=(255, 255, 255), outline=(200, 200, 200))
    draw.line([(lx+10, ly+12), (lx+40, ly+12)], fill=(0, 80, 255), width=3)
    draw.text((lx+45, ly+5), "GT past", fill=(0, 0, 0))
    draw.line([(lx+10, ly+30), (lx+40, ly+30)], fill=(150, 200, 255), width=2)
    draw.text((lx+45, ly+23), "GT future", fill=(0, 0, 0))
    draw.line([(lx+10, ly+48), (lx+20, ly+48)], fill=(255, 40, 40), width=2)
    draw.line([(lx+25, ly+48), (lx+40, ly+48)], fill=(255, 40, 40), width=2)
    draw.text((lx+45, ly+41), "Prediction", fill=(0, 0, 0))
    draw.polygon([(lx+23, ly+60), (lx+29, ly+66), (lx+23, ly+72), (lx+17, ly+66)], fill=(255, 165, 0))
    draw.text((lx+45, ly+60), "LSTM estimate", fill=(0, 0, 0))

    return img


def animate_episode(lstm, dynamics, track, imgs, physics, actions, save_path):
    # GT positions
    gt_full = physics[:, :2]
    n_total = min(N_FRAMES, len(gt_full))

    # Determine bounding box (full GT + pred margin)
    margin_xy = 30
    x_min = gt_full[:n_total, 0].min() - margin_xy
    x_max = gt_full[:n_total, 0].max() + margin_xy
    y_min = gt_full[:n_total, 1].min() - margin_xy
    y_max = gt_full[:n_total, 1].max() + margin_xy

    # Make square
    xr = x_max - x_min
    yr = y_max - y_min
    if xr > yr:
        mid = (y_max + y_min) / 2
        y_min = mid - xr / 2
        y_max = mid + xr / 2
    else:
        mid = (x_max + x_min) / 2
        x_min = mid - yr / 2
        x_max = mid + yr / 2

    frames = []
    min_t = (FRAME_STACK - 1)  # first valid frame
    for t in range(min_t, n_total):
        pred_traj = None
        est_pos = None
        if t >= (SEQ_LEN - 1) + (FRAME_STACK - 1):
            # Estimate state at current t using LSTM
            init_state, est_pos = estimate_state(lstm, track, imgs, physics, t)
            # Predict forward
            horizon = min(HORIZON_PRED, n_total - t - 1)
            if horizon > 0:
                pred_traj = predict_forward(dynamics, init_state, actions[t:], horizon)

        frame = render_frame(t, gt_full, pred_traj, est_pos,
                             x_min, x_max, y_min, y_max)
        frames.append(frame)

    # Save GIF
    frames[0].save(save_path, save_all=True, append_images=frames[1:],
                   duration=80, loop=0, optimize=True)
    print(f"Saved {save_path} ({len(frames)} frames)")


def main():
    os.makedirs("vis", exist_ok=True)

    # Build track
    track_files = sorted(glob.glob(f"{TRAIN_DIR}/*.npz"))
    d0 = np.load(track_files[0], allow_pickle=True)
    track = TrackCoords(d0["map"])

    # Load models
    dynamics = PhysicsDynamics().to(DEVICE)
    ckpt = load_checkpoint(os.path.join(DYN_SAVE_DIR, "best.tar"))
    dynamics.load_state_dict(ckpt["dynamics"])
    dynamics.eval()

    lstm = LSTMLocalizer(out_dim=3).to(DEVICE)
    lckpt = load_checkpoint("checkpoints/localization_lstm_sd/best.tar")
    lstm.load_state_dict(lckpt["state_dict"])
    lstm.eval()

    # Pick 2 test episodes to animate
    files = sorted(glob.glob(f"{TEST_DIR}/*.npz"))
    for ep_idx in [0, 3]:
        d = np.load(files[ep_idx], allow_pickle=True)
        imgs = d["imgs"].astype(np.float32)
        if imgs.max() > 1.0: imgs /= 255.0
        pos = d["position"].astype(np.float32)
        yaw = d["yaw"].astype(np.float32)
        vel = d["velocity"].astype(np.float32)
        omega = d["angular_velocity"].astype(np.float32)
        wheel = d["wheel_omega"].astype(np.float32)
        steer = d["steering_angle"].astype(np.float32)
        actions = d["action"].astype(np.float32)
        physics = np.column_stack([pos, yaw, vel, omega, wheel, steer])

        animate_episode(lstm, dynamics, track, imgs, physics, actions,
                       f"vis/rolling_pred_ep{ep_idx}.gif")


if __name__ == "__main__":
    main()
