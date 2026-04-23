"""Rolling 30-step prediction with SOFT Kalman-style fusion.

Instead of hard gating, use smooth blending:
    alpha = exp(-gap^2 / sigma^2)
    state = alpha * lstm_obs + (1 - alpha) * dynamics_prior

Smooth: small gap -> trust LSTM, large gap -> trust dynamics.
No sudden jumps or hard rejections.
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
SIGMA = 15.0  # Kalman-ish fusion width: alpha = exp(-gap^2 / sigma^2)


def decode_sd(p, total_len):
    theta = np.arctan2(p[0], p[1])
    theta = np.mod(theta, 2 * np.pi)
    return float(theta * total_len / (2 * np.pi)), float(p[2] * D_NORM)


def cnn_observe(cnn_sd, track, imgs, t):
    stack = np.stack([imgs[t - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
    with torch.no_grad():
        pred = cnn_sd(torch.tensor(stack, dtype=torch.float32).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
    s, d = decode_sd(pred, track.total_len)
    return np.array(track.sd_to_xy(s, d))


def lstm_observe(lstm, track, imgs, t):
    seq = []
    for off in range(SEQ_LEN):
        center = t - SEQ_LEN + 1 + off
        st = np.stack([imgs[center - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
        seq.append(st)
    seq = np.stack(seq, axis=0)
    with torch.no_grad():
        pred = lstm(torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
    s, d = decode_sd(pred, track.total_len)
    return np.array(track.sd_to_xy(s, d))


def dynamics_step(dynamics, state, action):
    z = torch.tensor((state - PHYSICS_MEAN) / PHYSICS_STD,
                     dtype=torch.float32).unsqueeze(0).to(DEVICE)
    a = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        z_next, _ = dynamics(z, a)
    return (z_next[0].cpu().numpy() * PHYSICS_STD + PHYSICS_MEAN).astype(np.float32)


def predict_forward(dynamics, state, actions_fut, horizon):
    z = torch.tensor((state - PHYSICS_MEAN) / PHYSICS_STD,
                     dtype=torch.float32).unsqueeze(0).to(DEVICE)
    positions = [state[:2].copy()]
    with torch.no_grad():
        for k in range(horizon):
            if k >= len(actions_fut): break
            a = torch.tensor(actions_fut[k], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            z_next, _ = dynamics(z, a)
            p = z_next[0, :2].cpu().numpy() * PHYSICS_STD[:2] + PHYSICS_MEAN[:2]
            positions.append(p)
            z = z_next
    return np.array(positions)


def render_frame(t, gt_full, pred_traj, est_xy, lstm_xy, dyn_xy, alpha,
                 x_min, x_max, y_min, y_max,
                 width=720, height=600, margin=80):
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    pw = width - 2 * margin
    ph = height - 2 * margin - 40
    xr, yr = x_max - x_min, y_max - y_min
    scale = min(pw / xr, ph / yr)
    cx = margin + (pw - xr * scale) / 2
    cy = margin + 60 + (ph - yr * scale) / 2

    def to_px(x, y):
        return int(cx + (x - x_min) * scale), int(cy + (y_max - y) * scale)

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

    # GT past (blue solid)
    for i in range(t):
        p1, p2 = to_px(gt_full[i, 0], gt_full[i, 1]), to_px(gt_full[i+1, 0], gt_full[i+1, 1])
        draw.line([p1, p2], fill=(0, 80, 255), width=3)
    # GT future (faded)
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
        ep = pts[-1]
        draw.ellipse([ep[0]-4, ep[1]-4, ep[0]+4, ep[1]+4], fill=(255, 40, 40), outline=(0, 0, 0))

    # Dynamics prior (gray small dot, when we have it)
    if dyn_xy is not None:
        dp = to_px(dyn_xy[0], dyn_xy[1])
        draw.ellipse([dp[0]-4, dp[1]-4, dp[0]+4, dp[1]+4], fill=(120, 120, 120), outline=(0, 0, 0))

    # Raw LSTM observation (small purple dot)
    if lstm_xy is not None:
        lp = to_px(lstm_xy[0], lstm_xy[1])
        draw.ellipse([lp[0]-4, lp[1]-4, lp[0]+4, lp[1]+4], fill=(180, 0, 200), outline=(0, 0, 0))

    # Fused estimate (green diamond, this is the actual used state)
    if est_xy is not None:
        ep = to_px(est_xy[0], est_xy[1])
        draw.polygon([(ep[0], ep[1]-7), (ep[0]+7, ep[1]), (ep[0], ep[1]+7), (ep[0]-7, ep[1])],
                     fill=(0, 200, 80), outline=(0, 0, 0))

    # Title
    if est_xy is None:
        status = f"Frame {t}: observing"
    else:
        err = np.sqrt(((gt_full[t] - est_xy) ** 2).sum())
        if alpha is None:
            status = f"Frame {t}: CNN/Dynamics init  err={err:.1f}"
        else:
            status = f"Frame {t}: alpha={alpha:.2f} (LSTM weight), fused err={err:.1f}"
    draw.text((margin, 10), status, fill=(0, 0, 0))

    # Alpha bar
    if alpha is not None:
        bar_x, bar_y = margin, 35
        bar_w, bar_h = 200, 12
        draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], outline=(0, 0, 0))
        fill_w = int(bar_w * alpha)
        draw.rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + bar_h], fill=(180, 0, 200))
        draw.text((bar_x + bar_w + 10, bar_y), f"LSTM {alpha*100:.0f}% / Dynamics {(1-alpha)*100:.0f}%", fill=(0, 0, 0))

    # Legend
    lx, ly = width - 240, 60
    draw.rectangle([lx, ly, lx + 225, ly + 140], fill=(255, 255, 255), outline=(200, 200, 200))
    draw.line([(lx+10, ly+12), (lx+40, ly+12)], fill=(0, 80, 255), width=3)
    draw.text((lx+45, ly+5), "GT past", fill=(0, 0, 0))
    draw.line([(lx+10, ly+30), (lx+40, ly+30)], fill=(160, 200, 255), width=2)
    draw.text((lx+45, ly+23), "GT future (30)", fill=(0, 0, 0))
    draw.line([(lx+10, ly+48), (lx+20, ly+48)], fill=(255, 40, 40), width=2)
    draw.line([(lx+25, ly+48), (lx+40, ly+48)], fill=(255, 40, 40), width=2)
    draw.text((lx+45, ly+41), "Dynamics 30-pred", fill=(0, 0, 0))
    draw.ellipse([lx+19, ly+62, lx+31, ly+74], fill=(120, 120, 120), outline=(0, 0, 0))
    draw.text((lx+38, ly+63), "Dynamics prior", fill=(0, 0, 0))
    draw.ellipse([lx+19, ly+84, lx+31, ly+96], fill=(180, 0, 200), outline=(0, 0, 0))
    draw.text((lx+38, ly+85), "LSTM observation", fill=(0, 0, 0))
    draw.polygon([(lx+25, ly+103), (lx+32, ly+110), (lx+25, ly+117), (lx+18, ly+110)],
                 fill=(0, 200, 80), outline=(0, 0, 0))
    draw.text((lx+38, ly+105), "Fused estimate", fill=(0, 0, 0))
    draw.text((lx+10, ly+125), f"sigma={SIGMA:.0f}  alpha=exp(-gap2/sigma2)", fill=(0, 0, 0))

    return img


def animate_episode(cnn_sd, lstm, dynamics, track, imgs, physics, actions, save_path):
    gt_full = physics[:, :2]
    n_total = min(N_FRAMES, len(gt_full))
    t_cnn = FRAME_STACK - 1
    t_lstm_ok = SEQ_LEN + FRAME_STACK - 2

    margin_xy = 30
    x_min = gt_full[:n_total, 0].min() - margin_xy
    x_max = gt_full[:n_total, 0].max() + margin_xy
    y_min = gt_full[:n_total, 1].min() - margin_xy
    y_max = gt_full[:n_total, 1].max() + margin_xy
    xr, yr = x_max - x_min, y_max - y_min
    if xr > yr:
        mid = (y_max + y_min) / 2; y_min, y_max = mid - xr/2, mid + xr/2
    else:
        mid = (x_max + x_min) / 2; x_min, x_max = mid - yr/2, mid + yr/2

    current_state = None
    alphas_history = []
    frames = []
    for t in range(n_total):
        pred_traj = None
        est_xy = None
        lstm_xy = None
        dyn_xy = None
        alpha = None

        if t == t_cnn:
            cnn_xy = cnn_observe(cnn_sd, track, imgs, t)
            current_state = physics[t].copy()
            current_state[0] = cnn_xy[0]; current_state[1] = cnn_xy[1]
            est_xy = cnn_xy.copy()
        elif t_cnn < t < t_lstm_ok:
            current_state = dynamics_step(dynamics, current_state, actions[t - 1])
            current_state[2:] = physics[t, 2:]
            est_xy = current_state[:2].copy()
        elif t >= t_lstm_ok:
            # Dynamics prior
            prior = dynamics_step(dynamics, current_state, actions[t - 1])
            prior[2:] = physics[t, 2:]
            dyn_xy = prior[:2].copy()

            # LSTM observation
            lstm_xy = lstm_observe(lstm, track, imgs, t)

            # Soft blend (Kalman-ish)
            gap = np.sqrt(((lstm_xy - dyn_xy) ** 2).sum())
            alpha = float(np.exp(-(gap ** 2) / (SIGMA ** 2)))
            fused_xy = alpha * lstm_xy + (1 - alpha) * dyn_xy

            current_state = prior.copy()
            current_state[0] = fused_xy[0]; current_state[1] = fused_xy[1]
            est_xy = fused_xy
            alphas_history.append(alpha)

        if current_state is not None:
            horizon = min(HORIZON_PRED, n_total - t - 1)
            if horizon > 0:
                pred_traj = predict_forward(dynamics, current_state.copy(), actions[t:], horizon)

        frames.append(render_frame(t, gt_full, pred_traj, est_xy, lstm_xy, dyn_xy, alpha,
                                   x_min, x_max, y_min, y_max))

    if alphas_history:
        a = np.array(alphas_history)
        print(f"  Alpha stats: mean={a.mean():.2f} median={np.median(a):.2f} "
              f">0.5: {(a > 0.5).mean()*100:.1f}%  >0.9: {(a > 0.9).mean()*100:.1f}%")
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
                       f"vis/pipeline_kalman_ep{ep_idx}.gif")


if __name__ == "__main__":
    main()
