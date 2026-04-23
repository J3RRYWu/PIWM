"""Animated GIF: state tracking with rolling LSTM correction.

Pipeline (continuous estimation with rolling updates):
  t=FRAME_STACK-1:                   CNN+sd estimates initial position
  t=FRAME_STACK .. SEQ_LEN+FS-2:     Dynamics integration from CNN init
  t >= SEQ_LEN+FRAME_STACK-1:        LSTM+sd re-estimates at EVERY frame (rolling)

Shows the estimated car position at each time step vs GT.
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
from models.localization import CoarsePositionHead
from models.localization_sd import CoarsePositionHeadSD
from models.localization_lstm import LSTMLocalizer
from track_utils import TrackCoords
from utils import load_checkpoint

TEST_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/test/fixed/"
TRAIN_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/train/fixed/"

SEQ_LEN = 10
N_FRAMES = 150
D_NORM = 10.0


def decode_sd_single(pred, total_len):
    s_sin, s_cos, d_norm = pred[0], pred[1], pred[2]
    theta = np.arctan2(s_sin, s_cos)
    theta = np.mod(theta, 2 * np.pi)
    return float(theta * total_len / (2 * np.pi)), float(d_norm * D_NORM)


def run_state_tracking(cnn_sd, lstm, dynamics, track, imgs, physics, actions, n_total):
    """Run full state-tracking pipeline with ROLLING LSTM updates.

    For each t:
      - t < FRAME_STACK-1: no estimate
      - t == FRAME_STACK-1: CNN+sd
      - FRAME_STACK-1 < t < SEQ_LEN+FRAME_STACK-1: Dynamics integration from CNN init
      - t >= SEQ_LEN+FRAME_STACK-1: LSTM+sd rolling re-estimation at every step

    Returns:
        est_positions: (n_total, 2) estimated (x, y) at each frame
        cnn_xy: initial CNN estimate
        first_lstm_xy: LSTM estimate at first LSTM-valid frame
        t_cnn, t_lstm: timing constants
    """
    est_positions = np.full((n_total, 2), np.nan)

    t_cnn = FRAME_STACK - 1
    t_lstm = (SEQ_LEN - 1) + (FRAME_STACK - 1)

    # === CNN+sd at t_cnn ===
    stack = np.stack([imgs[t_cnn - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
    with torch.no_grad():
        pred_cnn = cnn_sd(torch.tensor(stack, dtype=torch.float32).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
    s_c, d_c = decode_sd_single(pred_cnn, track.total_len)
    cnn_xy = track.sd_to_xy(s_c, d_c)
    est_positions[t_cnn] = cnn_xy

    # === Dynamics integration from CNN init to t_lstm-1 ===
    init_state = physics[t_cnn].copy()
    init_state[0] = cnn_xy[0]
    init_state[1] = cnn_xy[1]
    z = torch.tensor((init_state - PHYSICS_MEAN) / PHYSICS_STD, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        for t in range(t_cnn, t_lstm):
            a = torch.tensor(actions[t], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            z_next, _ = dynamics(z, a)
            p = z_next[0, :2].cpu().numpy() * PHYSICS_STD[:2] + PHYSICS_MEAN[:2]
            est_positions[t + 1] = p
            z = z_next

    # === Rolling LSTM updates from t_lstm onwards ===
    first_lstm_xy = None
    with torch.no_grad():
        for t in range(t_lstm, n_total):
            seq = []
            for offset in range(SEQ_LEN):
                center = t - SEQ_LEN + 1 + offset
                s_stack = np.stack([imgs[center - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
                seq.append(s_stack)
            seq = np.stack(seq, axis=0)
            pred = lstm(torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
            s_v, d_v = decode_sd_single(pred, track.total_len)
            xy = track.sd_to_xy(s_v, d_v)
            est_positions[t] = xy
            if first_lstm_xy is None:
                first_lstm_xy = np.array(xy)

    return est_positions, np.array(cnn_xy), first_lstm_xy, t_cnn, t_lstm


def render_frame(t, gt_full, est_positions, cnn_xy, lstm_xy, t_cnn, t_lstm,
                 x_min, x_max, y_min, y_max,
                 width=700, height=650, margin=80):
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    pw = width - 2 * margin
    ph = height - 2 * margin - 40
    xr = x_max - x_min
    yr = y_max - y_min
    scale = min(pw / xr, ph / yr)
    cx = margin + (pw - xr * scale) / 2
    cy = margin + 60 + (ph - yr * scale) / 2

    def to_px(x, y):
        return int(cx + (x - x_min) * scale), int(cy + (y_max - y) * scale)

    # Axes
    al, ar = margin - 3, width - margin + 3
    at, ab = margin + 57, height - margin + 3
    draw.line([(al, ab), (ar, ab)], fill=(0, 0, 0), width=1)
    draw.line([(al, at), (al, ab)], fill=(0, 0, 0), width=1)
    for i in range(6):
        vx = x_min + i * xr / 5
        px, _ = to_px(vx, y_min)
        draw.line([(px, at), (px, ab)], fill=(240, 240, 240))
        draw.text((px - 12, ab + 6), f"{vx:.0f}", fill=(80, 80, 80))
        vy = y_min + i * yr / 5
        _, py = to_px(x_min, vy)
        draw.line([(al, py), (ar, py)], fill=(240, 240, 240))
        draw.text((al - 45, py - 6), f"{vy:.0f}", fill=(80, 80, 80))

    # GT past
    for i in range(t):
        p1, p2 = to_px(gt_full[i, 0], gt_full[i, 1]), to_px(gt_full[i+1, 0], gt_full[i+1, 1])
        draw.line([p1, p2], fill=(0, 80, 255), width=3)

    # GT future (faded)
    for i in range(t, min(t + 40, len(gt_full) - 1)):
        p1, p2 = to_px(gt_full[i, 0], gt_full[i, 1]), to_px(gt_full[i+1, 0], gt_full[i+1, 1])
        draw.line([p1, p2], fill=(180, 210, 255), width=2)

    # Estimated trajectory up to t (red, dashed between CNN and LSTM segments)
    # Pre-LSTM segment: t_cnn to t_lstm-1 (CNN init + dynamics)
    # Post-LSTM segment: t_lstm onwards (dynamics from correction)
    # Break the line at t_lstm to show the correction "jump"
    for start_t, end_t in [(t_cnn, t_lstm), (t_lstm, t + 1)]:
        seg = est_positions[start_t:min(end_t, t+1)]
        if len(seg) < 2:
            continue
        # Filter NaN
        valid = ~np.isnan(seg[:, 0])
        seg = seg[valid]
        if len(seg) < 2:
            continue
        pts = [to_px(p[0], p[1]) for p in seg]
        # dashed
        dash_on, dash_off = 10, 8
        accum, drawing = 0.0, True
        for i in range(len(pts) - 1):
            dx = pts[i+1][0] - pts[i][0]
            dy = pts[i+1][1] - pts[i][1]
            sl = (dx**2 + dy**2) ** 0.5
            if sl < 0.5:
                continue
            pos_in = 0.0
            while pos_in < sl:
                cl = dash_on if drawing else dash_off
                can = min(cl - accum, sl - pos_in)
                if drawing:
                    t0, t1 = pos_in/sl, (pos_in+can)/sl
                    s = (int(pts[i][0]+dx*t0), int(pts[i][1]+dy*t0))
                    e = (int(pts[i][0]+dx*t1), int(pts[i][1]+dy*t1))
                    draw.line([s, e], fill=(255, 40, 40), width=2)
                pos_in += can
                accum += can
                if accum >= cl:
                    accum = 0; drawing = not drawing

    # Current positions
    cur_gt = to_px(gt_full[t, 0], gt_full[t, 1])
    draw.ellipse([cur_gt[0]-7, cur_gt[1]-7, cur_gt[0]+7, cur_gt[1]+7],
                 fill=(0, 80, 255), outline=(0, 0, 0), width=2)

    if t >= t_cnn and not np.isnan(est_positions[t, 0]):
        cur_est = to_px(est_positions[t, 0], est_positions[t, 1])
        draw.ellipse([cur_est[0]-7, cur_est[1]-7, cur_est[0]+7, cur_est[1]+7],
                     fill=(255, 40, 40), outline=(0, 0, 0), width=2)

    # CNN and LSTM markers
    if t >= t_cnn:
        p_cnn = to_px(cnn_xy[0], cnn_xy[1])
        draw.polygon([(p_cnn[0], p_cnn[1]-8), (p_cnn[0]+8, p_cnn[1]),
                      (p_cnn[0], p_cnn[1]+8), (p_cnn[0]-8, p_cnn[1])],
                     fill=(255, 140, 0), outline=(0, 0, 0))
    if t >= t_lstm:
        p_lstm = to_px(lstm_xy[0], lstm_xy[1])
        draw.polygon([(p_lstm[0], p_lstm[1]-8), (p_lstm[0]+8, p_lstm[1]),
                      (p_lstm[0], p_lstm[1]+8), (p_lstm[0]-8, p_lstm[1])],
                     fill=(180, 0, 200), outline=(0, 0, 0))

    # Title
    phase = ""
    if t < t_cnn:
        phase = "(collecting first frames)"
    elif t < t_lstm:
        phase = "(CNN+sd init + Dynamics integration)"
    elif t == t_lstm:
        phase = "(LSTM+sd CORRECTION)"
    else:
        phase = "(Dynamics integration from LSTM correction)"

    err = 0
    if t >= t_cnn and not np.isnan(est_positions[t, 0]):
        err = np.sqrt(((gt_full[t] - est_positions[t]) ** 2).sum())

    draw.text((margin, 10), f"Frame {t} {phase}", fill=(0, 0, 0))
    draw.text((margin, 30), f"Current tracking error: {err:.2f} units", fill=(0, 0, 0))

    # Legend
    lx, ly = width - 230, 60
    draw.rectangle([lx, ly, lx + 215, ly + 115], fill=(255, 255, 255), outline=(200, 200, 200))
    draw.line([(lx+10, ly+12), (lx+40, ly+12)], fill=(0, 80, 255), width=3)
    draw.text((lx+45, ly+5), "GT (past)", fill=(0, 0, 0))
    draw.line([(lx+10, ly+30), (lx+40, ly+30)], fill=(180, 210, 255), width=2)
    draw.text((lx+45, ly+23), "GT (future)", fill=(0, 0, 0))
    draw.line([(lx+10, ly+48), (lx+20, ly+48)], fill=(255, 40, 40), width=2)
    draw.line([(lx+25, ly+48), (lx+40, ly+48)], fill=(255, 40, 40), width=2)
    draw.text((lx+45, ly+41), "Pipeline estimate", fill=(0, 0, 0))
    draw.polygon([(lx+18, ly+58), (lx+26, ly+66), (lx+18, ly+74), (lx+10, ly+66)], fill=(255, 140, 0))
    draw.text((lx+32, ly+61), "CNN+sd init", fill=(0, 0, 0))
    draw.polygon([(lx+18, ly+80), (lx+26, ly+88), (lx+18, ly+96), (lx+10, ly+88)], fill=(180, 0, 200))
    draw.text((lx+32, ly+83), "LSTM+sd correction", fill=(0, 0, 0))

    return img


def animate_episode(cnn_sd, lstm, dynamics, track, imgs, physics, actions, save_path):
    gt_full = physics[:, :2]
    n_total = min(N_FRAMES, len(gt_full))

    est_positions, cnn_xy, lstm_xy, t_cnn, t_lstm = run_state_tracking(
        cnn_sd, lstm, dynamics, track, imgs, physics, actions, n_total)

    # Bounding box (full GT + estimate)
    all_x = np.concatenate([gt_full[:n_total, 0], est_positions[~np.isnan(est_positions[:, 0]), 0]])
    all_y = np.concatenate([gt_full[:n_total, 1], est_positions[~np.isnan(est_positions[:, 1]), 1]])
    margin_xy = 20
    x_min, x_max = all_x.min() - margin_xy, all_x.max() + margin_xy
    y_min, y_max = all_y.min() - margin_xy, all_y.max() + margin_xy
    # make square
    xr, yr = x_max - x_min, y_max - y_min
    if xr > yr:
        mid = (y_max + y_min) / 2
        y_min, y_max = mid - xr/2, mid + xr/2
    else:
        mid = (x_max + x_min) / 2
        x_min, x_max = mid - yr/2, mid + yr/2

    frames = []
    for t in range(n_total):
        frames.append(render_frame(t, gt_full, est_positions,
                                   cnn_xy, lstm_xy, t_cnn, t_lstm,
                                   x_min, x_max, y_min, y_max))

    frames[0].save(save_path, save_all=True, append_images=frames[1:],
                   duration=80, loop=0, optimize=True)
    print(f"Saved {save_path} ({len(frames)} frames)")


def main():
    os.makedirs("vis", exist_ok=True)
    track_files = sorted(glob.glob(f"{TRAIN_DIR}/*.npz"))
    d0 = np.load(track_files[0], allow_pickle=True)
    track = TrackCoords(d0["map"])

    dynamics = PhysicsDynamics().to(DEVICE)
    dynamics.load_state_dict(load_checkpoint(os.path.join(DYN_SAVE_DIR, "best.tar"))["dynamics"])
    dynamics.eval()

    cnn_sd = CoarsePositionHeadSD().to(DEVICE)
    cnn_sd.load_state_dict(load_checkpoint("checkpoints/localization_sd/best.tar")["state_dict"])
    cnn_sd.eval()

    lstm = LSTMLocalizer(out_dim=3).to(DEVICE)
    lstm.load_state_dict(load_checkpoint("checkpoints/localization_lstm_sd/best.tar")["state_dict"])
    lstm.eval()

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
        animate_episode(cnn_sd, lstm, dynamics, track, imgs, physics, actions,
                       f"vis/tracking_ep{ep_idx}.gif")


if __name__ == "__main__":
    main()
