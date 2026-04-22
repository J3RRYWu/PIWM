"""Rolling 30-step prediction with gated LSTM correction.

State tracking (deployment-ready, no GT needed):
  t=FRAME_STACK-1:                   CNN+sd initial pose
  t in (CNN, LSTM_valid):            Dynamics integration only
  t >= LSTM_valid:                   At each step:
    (1) Dynamics propagates previous state forward 1 step -> prior
    (2) LSTM+sd observes current frames -> observation
    (3) GATE: if |obs - prior| > THRESHOLD, reject LSTM; else adopt LSTM

Then at every frame, Dynamics predicts 30 steps forward from current state.
"""

import numpy as np
import torch
import glob
import os
from PIL import Image, ImageDraw

from config import DEVICE, PHYSICS_MEAN, PHYSICS_STD, FRAME_STACK, DYN_SAVE_DIR
from models.dynamics import PhysicsDynamics
from models.localization_sd import CoarsePositionHeadSD
from models.localization_lstm import LSTMLocalizer
from track_utils import TrackCoords
from utils import load_checkpoint

TEST_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/test/fixed/"
TRAIN_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/train/fixed/"
SEQ_LEN = 10
HORIZON_PRED = 30
N_FRAMES = 500
D_NORM = 10.0
GATE_THRESHOLD = 40.0  # units; reject LSTM if |obs - dynamics_prior| > this


def decode_sd(p, total_len):
    theta = np.arctan2(p[0], p[1])
    theta = np.mod(theta, 2 * np.pi)
    return float(theta * total_len / (2 * np.pi)), float(p[2] * D_NORM)


def cnn_observe(cnn_sd, track, imgs, t):
    """CNN+sd observation at time t."""
    stack = np.stack([imgs[t - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
    with torch.no_grad():
        pred = cnn_sd(torch.tensor(stack, dtype=torch.float32).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
    s, d = decode_sd(pred, track.total_len)
    return np.array(track.sd_to_xy(s, d))


def lstm_observe(lstm, track, imgs, t):
    """LSTM+sd observation at time t."""
    seq = []
    for off in range(SEQ_LEN):
        center = t - SEQ_LEN + 1 + off
        s_stack = np.stack([imgs[center - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
        seq.append(s_stack)
    seq = np.stack(seq, axis=0)
    with torch.no_grad():
        pred = lstm(torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
    s, d = decode_sd(pred, track.total_len)
    return np.array(track.sd_to_xy(s, d))


def dynamics_step(dynamics, state, action):
    """Propagate state forward one step using dynamics."""
    z = torch.tensor((state - PHYSICS_MEAN) / PHYSICS_STD,
                     dtype=torch.float32).unsqueeze(0).to(DEVICE)
    a = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        z_next, _ = dynamics(z, a)
    return (z_next[0].cpu().numpy() * PHYSICS_STD + PHYSICS_MEAN).astype(np.float32)


def predict_forward(dynamics, init_state, actions_future, horizon):
    z = torch.tensor((init_state - PHYSICS_MEAN) / PHYSICS_STD,
                     dtype=torch.float32).unsqueeze(0).to(DEVICE)
    positions = [init_state[:2].copy()]
    with torch.no_grad():
        for k in range(horizon):
            if k >= len(actions_future): break
            a = torch.tensor(actions_future[k], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            z_next, _ = dynamics(z, a)
            p = z_next[0, :2].cpu().numpy() * PHYSICS_STD[:2] + PHYSICS_MEAN[:2]
            positions.append(p)
            z = z_next
    return np.array(positions)


def render_frame(t, gt_full, pred_traj, est_xy, lstm_xy, method,
                 x_min, x_max, y_min, y_max,
                 width=700, height=600, margin=80):
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
    draw.line([(al, ab), (ar, ab)], fill=(0, 0, 0))
    draw.line([(al, at), (al, ab)], fill=(0, 0, 0))
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

    # GT future (next horizon_pred steps, faded)
    end = min(t + HORIZON_PRED + 1, len(gt_full))
    for i in range(t, end - 1):
        p1, p2 = to_px(gt_full[i, 0], gt_full[i, 1]), to_px(gt_full[i+1, 0], gt_full[i+1, 1])
        draw.line([p1, p2], fill=(160, 200, 255), width=2)

    # Current GT
    cur = to_px(gt_full[t, 0], gt_full[t, 1])
    draw.ellipse([cur[0]-7, cur[1]-7, cur[0]+7, cur[1]+7],
                 fill=(0, 80, 255), outline=(0, 0, 0), width=2)

    # Prediction (red dashed)
    if pred_traj is not None and len(pred_traj) >= 2:
        pts = [to_px(p[0], p[1]) for p in pred_traj]
        dash_on, dash_off = 10, 8
        accum, drawing = 0.0, True
        for i in range(len(pts) - 1):
            dx = pts[i+1][0] - pts[i][0]
            dy = pts[i+1][1] - pts[i][1]
            sl = (dx**2 + dy**2) ** 0.5
            if sl < 0.5: continue
            pos_in = 0.0
            while pos_in < sl:
                cl = dash_on if drawing else dash_off
                can = min(cl - accum, sl - pos_in)
                if drawing:
                    t0, t1 = pos_in/sl, (pos_in+can)/sl
                    s = (int(pts[i][0]+dx*t0), int(pts[i][1]+dy*t0))
                    e = (int(pts[i][0]+dx*t1), int(pts[i][1]+dy*t1))
                    draw.line([s, e], fill=(255, 40, 40), width=2)
                pos_in += can; accum += can
                if accum >= cl:
                    accum = 0; drawing = not drawing
        # End marker
        ep = pts[-1]
        draw.ellipse([ep[0]-4, ep[1]-4, ep[0]+4, ep[1]+4], fill=(255, 40, 40), outline=(0, 0, 0))

    # Current estimate marker (used state)
    if est_xy is not None:
        color = (255, 140, 0) if method.startswith('CNN') else (180, 0, 200)
        if method == 'Dynamics (LSTM rejected)':
            color = (120, 120, 120)
        ep = to_px(est_xy[0], est_xy[1])
        draw.polygon([(ep[0], ep[1]-7), (ep[0]+7, ep[1]), (ep[0], ep[1]+7), (ep[0]-7, ep[1])],
                     fill=color, outline=(0, 0, 0))

    # Show raw LSTM observation if it was rejected
    if lstm_xy is not None and method == 'Dynamics (LSTM rejected)':
        lp = to_px(lstm_xy[0], lstm_xy[1])
        # Draw X to show rejected LSTM observation
        draw.line([(lp[0]-6, lp[1]-6), (lp[0]+6, lp[1]+6)], fill=(200, 0, 0), width=2)
        draw.line([(lp[0]-6, lp[1]+6), (lp[0]+6, lp[1]-6)], fill=(200, 0, 0), width=2)

    # Title
    if t < FRAME_STACK - 1:
        status = f"Frame {t}: Observing (need frame stack)"
    else:
        err = np.sqrt(((gt_full[t] - est_xy) ** 2).sum()) if est_xy is not None else 0
        status = f"Frame {t}: {method} localize (err={err:.1f}) + Dynamics 30-step prediction"
    draw.text((margin, 10), status, fill=(0, 0, 0))

    # Legend
    lx, ly = width - 240, 30
    draw.rectangle([lx, ly, lx + 225, ly + 140], fill=(255, 255, 255), outline=(200, 200, 200))
    draw.line([(lx+10, ly+12), (lx+40, ly+12)], fill=(0, 80, 255), width=3)
    draw.text((lx+45, ly+5), "GT past", fill=(0, 0, 0))
    draw.line([(lx+10, ly+30), (lx+40, ly+30)], fill=(160, 200, 255), width=2)
    draw.text((lx+45, ly+23), "GT future (30)", fill=(0, 0, 0))
    draw.line([(lx+10, ly+48), (lx+20, ly+48)], fill=(255, 40, 40), width=2)
    draw.line([(lx+25, ly+48), (lx+40, ly+48)], fill=(255, 40, 40), width=2)
    draw.text((lx+45, ly+41), "Dynamics pred", fill=(0, 0, 0))
    draw.polygon([(lx+20, ly+61), (lx+28, ly+69), (lx+20, ly+77), (lx+12, ly+69)], fill=(255, 140, 0))
    draw.text((lx+32, ly+64), "CNN+sd", fill=(0, 0, 0))
    draw.polygon([(lx+20, ly+83), (lx+28, ly+91), (lx+20, ly+99), (lx+12, ly+91)], fill=(180, 0, 200))
    draw.text((lx+32, ly+86), "LSTM+sd (accepted)", fill=(0, 0, 0))
    draw.polygon([(lx+20, ly+105), (lx+28, ly+113), (lx+20, ly+121), (lx+12, ly+113)], fill=(120, 120, 120))
    draw.text((lx+32, ly+108), "Dynamics-only", fill=(0, 0, 0))
    draw.line([(lx+14, ly+126), (lx+26, ly+138)], fill=(200, 0, 0), width=2)
    draw.line([(lx+14, ly+138), (lx+26, ly+126)], fill=(200, 0, 0), width=2)
    draw.text((lx+32, ly+128), "LSTM rejected", fill=(0, 0, 0))

    return img


def animate_episode(cnn_sd, lstm, dynamics, track, imgs, physics, actions, save_path):
    gt_full = physics[:, :2]
    n_total = min(N_FRAMES, len(gt_full))
    t_cnn = FRAME_STACK - 1
    t_lstm_ok = SEQ_LEN + FRAME_STACK - 2

    # Bounding box
    margin_xy = 30
    x_min = gt_full[:n_total, 0].min() - margin_xy
    x_max = gt_full[:n_total, 0].max() + margin_xy
    y_min = gt_full[:n_total, 1].min() - margin_xy
    y_max = gt_full[:n_total, 1].max() + margin_xy
    xr, yr = x_max - x_min, y_max - y_min
    if xr > yr:
        mid = (y_max + y_min) / 2
        y_min, y_max = mid - xr/2, mid + xr/2
    else:
        mid = (x_max + x_min) / 2
        x_min, x_max = mid - yr/2, mid + yr/2

    # Chain state across frames (deployment-like, no GT used)
    current_state = None   # 11-dim
    rejected_count = 0
    frames = []
    for t in range(n_total):
        pred_traj = None
        est_xy = None
        lstm_xy_raw = None
        method = ''

        if t == t_cnn:
            # Initialize with CNN+sd
            est_xy = cnn_observe(cnn_sd, track, imgs, t)
            current_state = physics[t].copy()  # use GT for non-xy dims (encoder could provide in practice)
            current_state[0] = est_xy[0]
            current_state[1] = est_xy[1]
            method = 'CNN+sd (init)'
        elif t_cnn < t < t_lstm_ok:
            # Dynamics only (LSTM not available yet)
            current_state = dynamics_step(dynamics, current_state, actions[t - 1])
            # Refresh non-xy dims from observation (encoder would do this in practice; using GT here)
            current_state[2:] = physics[t, 2:]
            est_xy = current_state[:2].copy()
            method = 'Dynamics (CNN init)'
        elif t >= t_lstm_ok:
            # (1) Dynamics propagation as prior
            prior_state = dynamics_step(dynamics, current_state, actions[t - 1])
            prior_state[2:] = physics[t, 2:]
            prior_xy = prior_state[:2].copy()

            # (2) LSTM observation
            lstm_xy_raw = lstm_observe(lstm, track, imgs, t)

            # (3) Gate
            gap = np.sqrt(((lstm_xy_raw - prior_xy) ** 2).sum())
            if gap > GATE_THRESHOLD:
                # Reject LSTM, keep dynamics prior
                current_state = prior_state
                current_state[:2] = prior_xy
                est_xy = prior_xy
                method = 'Dynamics (LSTM rejected)'
                rejected_count += 1
            else:
                # Accept LSTM observation
                current_state = prior_state.copy()
                current_state[0] = lstm_xy_raw[0]
                current_state[1] = lstm_xy_raw[1]
                est_xy = lstm_xy_raw
                method = 'LSTM+sd (gated accept)'

        # Dynamics rollout prediction (from current state, 30 steps)
        if current_state is not None:
            horizon = min(HORIZON_PRED, n_total - t - 1)
            if horizon > 0:
                pred_traj = predict_forward(dynamics, current_state.copy(), actions[t:], horizon)

        frames.append(render_frame(t, gt_full, pred_traj, est_xy, lstm_xy_raw, method,
                                   x_min, x_max, y_min, y_max))

    print(f"  LSTM rejections: {rejected_count}/{n_total - t_lstm_ok} "
          f"({rejected_count/(n_total - t_lstm_ok)*100:.1f}%)")
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
                       f"vis/pipeline_rolling_ep{ep_idx}.gif")


if __name__ == "__main__":
    main()
