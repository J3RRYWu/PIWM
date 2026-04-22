"""Train DVBF/GOKU/V2P on 31-dim state using v5's frozen encoder/decoder.

For fair comparison with PIWM-lane-v5: baselines get the SAME lane info
through the same encoder, but their dynamics have NO physics prior on the
lane (everything learned).
"""
import os, glob, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from config import (DEVICE, DATA_DIR, PHYSICS_MEAN_REL, PHYSICS_STD_REL,
                    ENCODER_DIM as CAR_ENCODER_DIM)
from lane_utils import LANE_DIM, LANE_MEAN, LANE_STD, LANE_FRAME_STACK as FRAME_STACK
from relative_coords import to_relative_np
from baselines.shared_dynamics_lane import (DynamicsDVBFLane, DynamicsGOKULane,
                                             DynamicsVid2ParamLane)
from models.encoder_lane import PhysicsEncoderLane
from utils import save_checkpoint, load_checkpoint


ROLLOUT_K = 8
EPOCHS = 60
THETA_HIST = 4


class SeqLaneStateDataset(Dataset):
    """Returns (image_history_for_v2p, fut_state31, fut_acts).

    fut_state31: (seq_len, 31) = car (11, normalized REL) || lane (20, normalized)
    image_history_for_v2p: (THETA_HIST, FRAME_STACK, 64, 64)  optional
    """
    def __init__(self, data_dir, seq_len=ROLLOUT_K + 1, need_images=False):
        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        files = [f for f in files if ".lane." not in f]
        self.seq_len = seq_len
        self.need_images = need_images
        self.imgs_list, self.phys_list, self.acts_list, self.wp_world_list = [], [], [], []
        self.indices = []
        for ep_idx, f in enumerate(files):
            lane_f = f.replace(".npz", ".lane.npz")
            if not os.path.exists(lane_f): continue
            try:
                d = np.load(f, allow_pickle=True)
                pos = d["position"].astype(np.float32); yaw = d["yaw"].astype(np.float32)
                vel = d["velocity"].astype(np.float32); omega = d["angular_velocity"].astype(np.float32)
                wheel = d["wheel_omega"].astype(np.float32); steer = d["steering_angle"].astype(np.float32)
                acts = d["action"].astype(np.float32)
                phys = np.column_stack([pos, yaw, vel, omega, wheel, steer])
                wp_body = np.load(lane_f)["lane_wp_body"].astype(np.float32)
                cy = np.cos(yaw); sy = np.sin(yaw)
                wp_world = np.zeros_like(wp_body)
                wp_world[..., 0] = wp_body[..., 0] * cy[:, None] - wp_body[..., 1] * sy[:, None] + pos[:, 0:1]
                wp_world[..., 1] = wp_body[..., 0] * sy[:, None] + wp_body[..., 1] * cy[:, None] + pos[:, 1:2]
                imgs = None
                if need_images:
                    imgs = d["imgs"].astype(np.float32)
                    if imgs.max() > 1.0: imgs /= 255.0
                self.imgs_list.append(imgs); self.phys_list.append(phys)
                self.acts_list.append(acts); self.wp_world_list.append(wp_world)
                n = len(phys)
                if n < seq_len + THETA_HIST + FRAME_STACK: continue
                t_start = (THETA_HIST + FRAME_STACK - 2) if need_images else (FRAME_STACK - 1)
                for t in range(t_start, n - seq_len):
                    self.indices.append((ep_idx, t))
            except Exception as e:
                print(f"  skip {os.path.basename(f)}: {e}")
        print(f"SeqLaneStateDataset (images={need_images}): {len(self.indices)}")

    def __len__(self): return len(self.indices)

    def _wp_in_ref_body(self, ep, t0):
        phys = self.phys_list[ep]; wp_world = self.wp_world_list[ep]
        wp_w_seq = wp_world[t0:t0 + self.seq_len]                 # (S, 10, 2)
        pos0 = phys[t0, 0:2]; yaw0 = phys[t0, 2]
        rel = wp_w_seq - pos0
        c = np.cos(yaw0); s = np.sin(yaw0)
        x =  rel[..., 0] * c + rel[..., 1] * s
        y = -rel[..., 0] * s + rel[..., 1] * c
        return np.stack([x, y], axis=-1).astype(np.float32)

    def __getitem__(self, i):
        ep, t0 = self.indices[i]
        phys = self.phys_list[ep]; acts = self.acts_list[ep]
        fut_phys = phys[t0:t0 + self.seq_len]
        fut_acts = acts[t0:t0 + self.seq_len - 1]
        rel_phys = to_relative_np(fut_phys, ref_idx=0)
        car_norm = (rel_phys - PHYSICS_MEAN_REL) / PHYSICS_STD_REL  # (S, 11)
        wp_ref = self._wp_in_ref_body(ep, t0)                       # (S, 10, 2)
        lane_norm = (wp_ref.reshape(self.seq_len, LANE_DIM) - LANE_MEAN) / LANE_STD
        state31 = np.concatenate([car_norm, lane_norm], axis=-1)    # (S, 31)

        if self.need_images:
            imgs = self.imgs_list[ep]
            hist = []
            for k in range(THETA_HIST):
                center = t0 - THETA_HIST + 1 + k
                stack = np.stack([imgs[center - FRAME_STACK + 1 + j]
                                  for j in range(FRAME_STACK)], axis=0)
                hist.append(stack)
            hist = np.stack(hist, axis=0)                           # (T_hist, FS, 64, 64)
            return (torch.tensor(hist, dtype=torch.float32),
                    torch.tensor(state31, dtype=torch.float32),
                    torch.tensor(fut_acts, dtype=torch.float32))
        else:
            dummy = torch.zeros(1)
            return (dummy,
                    torch.tensor(state31, dtype=torch.float32),
                    torch.tensor(fut_acts, dtype=torch.float32))


def _train_simple(model, name, save_path):
    print("=" * 60); print(f"Train {name}"); print("=" * 60)
    ds = SeqLaneStateDataset(DATA_DIR)
    n_val = int(len(ds) * 0.1)
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=128, shuffle=False, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')
    for epoch in range(EPOCHS):
        model.train(); tl, nb = 0, 0
        for _, fut, acts in tqdm(tr_loader, desc=f"{name} {epoch+1}/{EPOCHS}"):
            fut, acts = fut.to(DEVICE), acts.to(DEVICE)
            z = fut[:, 0]; ls = 0
            for k in range(ROLLOUT_K):
                z = model(z, acts[:, k])
                ls = ls + ((z - fut[:, k + 1]) ** 2).mean()
            ls /= ROLLOUT_K
            opt.zero_grad(); ls.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += ls.item(); nb += 1
        tl /= nb
        model.eval(); vl, vn = 0, 0
        with torch.no_grad():
            for _, fut, acts in val_loader:
                fut, acts = fut.to(DEVICE), acts.to(DEVICE)
                z = fut[:, 0]; ls = 0
                for k in range(ROLLOUT_K):
                    z = model(z, acts[:, k])
                    ls = ls + ((z - fut[:, k + 1]) ** 2).mean()
                ls /= ROLLOUT_K
                vl += ls.item(); vn += 1
        vl /= vn
        sch.step(vl)
        print(f"  Train {tl:.6f}  Val {vl:.6f}")
        if vl < best:
            best = vl
            save_checkpoint({'model': model.state_dict(), 'val_loss': vl}, save_path)
    print(f"{name} best: {best:.6f}")


def train_dvbf_lane():
    _train_simple(DynamicsDVBFLane().to(DEVICE), "DVBF-lane",
                  "checkpoints/shared_dvbf_lane/best.tar")


def train_goku_lane():
    _train_simple(DynamicsGOKULane().to(DEVICE), "GOKU-lane",
                  "checkpoints/shared_goku_lane/best.tar")


def train_v2p_lane():
    print("=" * 60); print("Train V2P-lane (theta from v5 encoder)"); print("=" * 60)
    ds = SeqLaneStateDataset(DATA_DIR, need_images=True)
    n_val = int(len(ds) * 0.1)
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=64, shuffle=False, drop_last=True)
    enc = PhysicsEncoderLane().to(DEVICE)
    # Use v5 ae.tar encoder (or best.tar if available, post-stage-3)
    if os.path.exists("checkpoints/piwm_lane_v5/best.tar"):
        ck = load_checkpoint("checkpoints/piwm_lane_v5/best.tar")
    else:
        ck = load_checkpoint("checkpoints/piwm_lane_v5/ae.tar")
    enc.load_state_dict(ck['encoder']); enc.eval()
    for p in enc.parameters(): p.requires_grad = False
    model = DynamicsVid2ParamLane().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')

    def encode_history(hist_stacks):
        B, T, C, H, W = hist_stacks.shape
        with torch.no_grad():
            obs = enc(hist_stacks.reshape(B * T, C, H, W))
        return obs.reshape(B, T, -1)   # (B, T_hist, 29)

    for epoch in range(EPOCHS):
        model.train(); tl, nb = 0, 0
        for hist, fut, acts in tqdm(tr_loader, desc=f"V2P-lane {epoch+1}/{EPOCHS}"):
            hist, fut, acts = hist.to(DEVICE), fut.to(DEVICE), acts.to(DEVICE)
            obs_hist = encode_history(hist)
            theta, mu, lv = model.infer_theta(obs_hist)
            z = fut[:, 0]; ls = 0
            for k in range(ROLLOUT_K):
                z = model.step(z, acts[:, k], theta)
                ls = ls + ((z - fut[:, k + 1]) ** 2).mean()
            ls /= ROLLOUT_K
            kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
            loss = ls + 0.001 * kl
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tl += ls.item(); nb += 1
        tl /= nb
        model.eval(); vl, vn = 0, 0
        with torch.no_grad():
            for hist, fut, acts in val_loader:
                hist, fut, acts = hist.to(DEVICE), fut.to(DEVICE), acts.to(DEVICE)
                obs_hist = encode_history(hist)
                theta, _, _ = model.infer_theta(obs_hist)
                z = fut[:, 0]; ls = 0
                for k in range(ROLLOUT_K):
                    z = model.step(z, acts[:, k], theta)
                    ls = ls + ((z - fut[:, k + 1]) ** 2).mean()
                ls /= ROLLOUT_K
                vl += ls.item(); vn += 1
        vl /= vn; sch.step(vl)
        print(f"  Train {tl:.6f}  Val {vl:.6f}")
        if vl < best:
            best = vl
            save_checkpoint({'model': model.state_dict(), 'val_loss': vl},
                            "checkpoints/shared_v2p_lane/best.tar")
    print(f"V2P-lane best: {best:.6f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="all", choices=["dvbf", "goku", "v2p", "all"])
    args = p.parse_args()
    os.makedirs("checkpoints/shared_dvbf_lane", exist_ok=True)
    os.makedirs("checkpoints/shared_goku_lane", exist_ok=True)
    os.makedirs("checkpoints/shared_v2p_lane", exist_ok=True)
    if args.variant in ("dvbf", "all"): train_dvbf_lane()
    if args.variant in ("goku", "all"): train_goku_lane()
    if args.variant in ("v2p",  "all"): train_v2p_lane()
