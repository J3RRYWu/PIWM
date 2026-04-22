"""PIWM-v5: Train lane-augmented bicycle model.

Three stages:
  Stage 1: AE training (encoder 29-dim: 9 car + 20 lane) with GT supervision
  Stage 2: Dynamics rollout training on 31-dim state (11 car + 20 lane)
  Stage 3: E2E fine-tune (image + car + lane losses)

Output: checkpoints/piwm_lane_v5/{ae.tar, dyn.tar, best.tar}
"""

import os, glob, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from config import (DEVICE, DATA_DIR, PHYSICS_MEAN_REL, PHYSICS_STD_REL,
                    LAMBDA_RESIDUAL, FULL_STATE_DIM,
                    ENCODER_DIM as CAR_ENCODER_DIM, ENCODER_STATE_INDICES,
                    ENCODER_MEAN_REL, ENCODER_STD_REL)
from lane_utils import LANE_DIM, LANE_MEAN, LANE_STD, N_LANE_WP, LANE_FRAME_STACK as FRAME_STACK
from relative_coords import to_relative_np
from models.encoder_lane import PhysicsEncoderLane
from models.decoder_lane import PhysicsDecoderLane
from models.dynamics_lane_v5 import LaneAugmentedV5
from utils import save_checkpoint, load_checkpoint


SAVE_DIR = "checkpoints/piwm_lane_v5/"
ROLLOUT_K = 8
AE_EPOCHS = 40
DYN_EPOCHS = 60
E2E_EPOCHS = 20
LAMBDA_IMG = 10.0
LAMBDA_CAR = 1.0
LAMBDA_LANE = 0.5


def _rotate_world_points_to_ref(points_world, pos0, yaw0):
    """Convert world-frame lane waypoints to body frame of ref pose (pos0, yaw0)."""
    rel = points_world - pos0                       # (T, N, 2)
    c = np.cos(yaw0); s = np.sin(yaw0)
    x =  rel[..., 0] * c + rel[..., 1] * s
    y = -rel[..., 0] * s + rel[..., 1] * c
    return np.stack([x, y], axis=-1).astype(np.float32)


class SeqLaneDataset(Dataset):
    """Loads imgs, phys, actions, AND lane waypoints (from .lane.npz sidecar).

    Lane waypoints stored per-frame in body frame OF THAT FRAME. We re-express
    them in the body frame of the SEQUENCE's ref frame (ref_idx=0) so they
    match the relative-coords car state.

    Actually simpler: store GT lane wp in world frame in the dataset, then
    rotate into ref body frame per sample. But preprocessing already has them
    in per-frame body frame. We convert back to world using GT car pose, then
    into ref body frame.
    """

    def __init__(self, data_dir, seq_len=ROLLOUT_K + 1):
        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        files = [f for f in files if ".lane." not in f]
        self.seq_len = seq_len

        self.imgs_list, self.phys_list, self.acts_list, self.wp_world_list = [], [], [], []
        self.indices = []
        for ep_idx, f in enumerate(files):
            lane_f = f.replace(".npz", ".lane.npz")
            if not os.path.exists(lane_f):
                continue
            try:
                d = np.load(f, allow_pickle=True)
                imgs = d["imgs"].astype(np.float32)
                if imgs.max() > 1.0: imgs /= 255.0
                pos = d["position"].astype(np.float32)
                yaw = d["yaw"].astype(np.float32)
                vel = d["velocity"].astype(np.float32)
                omega = d["angular_velocity"].astype(np.float32)
                wheel = d["wheel_omega"].astype(np.float32)
                steer = d["steering_angle"].astype(np.float32)
                acts = d["action"].astype(np.float32)
                phys = np.column_stack([pos, yaw, vel, omega, wheel, steer])

                # Load per-frame lane waypoints (in their own body frames)
                wp_body = np.load(lane_f)["lane_wp_body"].astype(np.float32)  # (T, 10, 2)

                # Convert each frame's wp back to world frame
                # wp_world[t] = R(yaw[t]) wp_body[t] + pos[t]
                T = len(pos)
                cy = np.cos(yaw); sy = np.sin(yaw)             # (T,)
                wp_world = np.zeros_like(wp_body)
                wp_world[..., 0] = wp_body[..., 0] * cy[:, None] - wp_body[..., 1] * sy[:, None] + pos[:, 0:1]
                wp_world[..., 1] = wp_body[..., 0] * sy[:, None] + wp_body[..., 1] * cy[:, None] + pos[:, 1:2]

                n = len(imgs)
                if n < seq_len + FRAME_STACK: continue
                self.imgs_list.append(imgs)
                self.phys_list.append(phys)
                self.acts_list.append(acts)
                self.wp_world_list.append(wp_world)
                for t in range(FRAME_STACK - 1, n - seq_len):
                    self.indices.append((ep_idx, t))
            except Exception as e:
                print(f"  skip {os.path.basename(f)}: {e}")
        print(f"SeqLaneDataset: {len(self.indices)} samples from {len(self.imgs_list)} eps")

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        ep, t0 = self.indices[i]
        imgs = self.imgs_list[ep]; phys = self.phys_list[ep]
        acts = self.acts_list[ep]; wp_world = self.wp_world_list[ep]

        stack0 = np.stack([imgs[t0 - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
        fut_imgs = imgs[t0:t0 + self.seq_len]
        fut_phys = phys[t0:t0 + self.seq_len]
        fut_acts = acts[t0:t0 + self.seq_len - 1]
        fut_wp_w = wp_world[t0:t0 + self.seq_len]   # (seq_len, 10, 2) world frame

        # Relative-coords car state (ref = first frame)
        rel_phys = to_relative_np(fut_phys, ref_idx=0)
        phys_norm = (rel_phys - PHYSICS_MEAN_REL) / PHYSICS_STD_REL

        # Lane wp in REF body frame (ref = first frame of sequence)
        pos0 = fut_phys[0, 0:2]; yaw0 = fut_phys[0, 2]
        wp_ref_body = _rotate_world_points_to_ref(fut_wp_w, pos0, yaw0)   # (seq_len, 10, 2)

        # Normalize lane wp
        wp_norm = (wp_ref_body.reshape(self.seq_len, LANE_DIM) - LANE_MEAN) / LANE_STD

        return (torch.tensor(stack0, dtype=torch.float32),
                torch.tensor(fut_imgs, dtype=torch.float32).unsqueeze(1),
                torch.tensor(phys_norm, dtype=torch.float32),
                torch.tensor(fut_acts, dtype=torch.float32),
                torch.tensor(wp_norm, dtype=torch.float32))


# ================================================================
# Stage 1: AE with lane supervision
# ================================================================
def stage1_ae():
    print("=" * 60); print("Stage 1 [v5]: AE (encoder+decoder) with car+lane supervision"); print("=" * 60)
    ds = SeqLaneDataset(DATA_DIR, seq_len=1)   # just need single-frame samples
    # Override: use seq_len=1 via a different dataset wouldn't work because indices skip seq_len.
    # Just use seq_len>1 and always pick frame 0.
    ds = SeqLaneDataset(DATA_DIR, seq_len=ROLLOUT_K + 1)
    n_val = int(len(ds) * 0.1)
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=64, shuffle=False, drop_last=True)

    enc = PhysicsEncoderLane().to(DEVICE)
    dec = PhysicsDecoderLane().to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')

    # Targets for encoder: first-frame car-observable (9 dim) + lane wp (20 dim)
    # Both already normalized. Car observable = phys_norm[:, 0, 2:11].

    for epoch in range(AE_EPOCHS):
        enc.train(); dec.train()
        ti, tc, tl_lane, nb = 0, 0, 0, 0
        for stack0, fut_imgs, fut_phys, fut_acts, wp_norm in tqdm(tr_loader, desc=f"v5 AE {epoch+1}/{AE_EPOCHS}"):
            stack0 = stack0.to(DEVICE); fut_imgs = fut_imgs.to(DEVICE)
            fut_phys = fut_phys.to(DEVICE); wp_norm = wp_norm.to(DEVICE)
            z_obs = enc(stack0)                    # (B, 29)
            car_obs = z_obs[:, :CAR_ENCODER_DIM]   # (B, 9)
            lane_obs = z_obs[:, CAR_ENCODER_DIM:]  # (B, 20)

            rec = dec(z_obs)                       # (B, 1, 64, 64)

            gt_car  = fut_phys[:, 0, 2:11]
            gt_lane = wp_norm[:, 0]

            img_l  = ((rec - fut_imgs[:, 0]) ** 2).mean()
            car_l  = ((car_obs  - gt_car)  ** 2).mean()
            lane_l = ((lane_obs - gt_lane) ** 2).mean()
            loss = LAMBDA_IMG * img_l + LAMBDA_CAR * car_l + LAMBDA_LANE * lane_l
            opt.zero_grad(); loss.backward(); opt.step()
            ti += img_l.item(); tc += car_l.item(); tl_lane += lane_l.item(); nb += 1
        ti /= nb; tc /= nb; tl_lane /= nb

        enc.eval(); dec.eval()
        vi, vc, vl, vn = 0, 0, 0, 0
        with torch.no_grad():
            for stack0, fut_imgs, fut_phys, fut_acts, wp_norm in val_loader:
                stack0 = stack0.to(DEVICE); fut_imgs = fut_imgs.to(DEVICE)
                fut_phys = fut_phys.to(DEVICE); wp_norm = wp_norm.to(DEVICE)
                z_obs = enc(stack0); rec = dec(z_obs)
                vi += ((rec - fut_imgs[:, 0]) ** 2).mean().item()
                vc += ((z_obs[:, :CAR_ENCODER_DIM] - fut_phys[:, 0, 2:11]) ** 2).mean().item()
                vl += ((z_obs[:, CAR_ENCODER_DIM:] - wp_norm[:, 0]) ** 2).mean().item()
                vn += 1
        vi /= vn; vc /= vn; vl /= vn
        vt = LAMBDA_IMG * vi + LAMBDA_CAR * vc + LAMBDA_LANE * vl
        sch.step(vt)
        print(f"  Train img={ti:.5f} car={tc:.5f} lane={tl_lane:.5f}"
              f"  Val img={vi:.5f} car={vc:.5f} lane={vl:.5f}")
        if vt < best:
            best = vt
            save_checkpoint({'encoder': enc.state_dict(), 'decoder': dec.state_dict(),
                             'val_loss': vt},
                            os.path.join(SAVE_DIR, 'ae.tar'))
    print(f"v5 AE best: {best:.5f}")


# ================================================================
# Stage 2: Dynamics rollout (31-dim)
# ================================================================
def stage2_dyn():
    print("\n" + "=" * 60); print("Stage 2 [v5]: Lane-augmented dynamics rollout"); print("=" * 60)
    ds = SeqLaneDataset(DATA_DIR, seq_len=ROLLOUT_K + 1)
    n_val = int(len(ds) * 0.1)
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=128, shuffle=False, drop_last=True)

    dyn = LaneAugmentedV5().to(DEVICE)
    opt = torch.optim.Adam(dyn.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')

    for epoch in range(DYN_EPOCHS):
        dyn.train()
        tc, tl_lane, nb = 0, 0, 0
        for stack0, fut_imgs, fut_phys, fut_acts, wp_norm in tqdm(tr_loader, desc=f"v5 Dyn {epoch+1}/{DYN_EPOCHS}"):
            fut_phys = fut_phys.to(DEVICE); fut_acts = fut_acts.to(DEVICE)
            wp_norm = wp_norm.to(DEVICE)

            z0 = torch.cat([fut_phys[:, 0], wp_norm[:, 0]], dim=-1)   # (B, 31)
            z = z0
            ls_car, ls_lane, lres = 0, 0, 0
            for k in range(ROLLOUT_K):
                z_next, car_res = dyn(z, fut_acts[:, k])
                gt = torch.cat([fut_phys[:, k + 1], wp_norm[:, k + 1]], dim=-1)
                ls_car  = ls_car  + ((z_next[:, :11] - gt[:, :11]) ** 2).mean()
                ls_lane = ls_lane + ((z_next[:, 11:] - gt[:, 11:]) ** 2).mean()
                lres = lres + (car_res ** 2).mean()
                z = z_next
            ls_car /= ROLLOUT_K; ls_lane /= ROLLOUT_K; lres /= ROLLOUT_K
            loss = ls_car + LAMBDA_LANE * ls_lane + LAMBDA_RESIDUAL * lres
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(dyn.parameters(), 5.0)
            opt.step()
            tc += ls_car.item(); tl_lane += ls_lane.item(); nb += 1
        tc /= nb; tl_lane /= nb

        dyn.eval(); vc, vl_lane, vn = 0, 0, 0
        with torch.no_grad():
            for stack0, fut_imgs, fut_phys, fut_acts, wp_norm in val_loader:
                fut_phys = fut_phys.to(DEVICE); fut_acts = fut_acts.to(DEVICE)
                wp_norm = wp_norm.to(DEVICE)
                z = torch.cat([fut_phys[:, 0], wp_norm[:, 0]], dim=-1)
                ls_car, ls_lane = 0, 0
                for k in range(ROLLOUT_K):
                    z, _ = dyn(z, fut_acts[:, k])
                    gt = torch.cat([fut_phys[:, k + 1], wp_norm[:, k + 1]], dim=-1)
                    ls_car  = ls_car  + ((z[:, :11] - gt[:, :11]) ** 2).mean()
                    ls_lane = ls_lane + ((z[:, 11:] - gt[:, 11:]) ** 2).mean()
                ls_car /= ROLLOUT_K; ls_lane /= ROLLOUT_K
                vc += ls_car.item(); vl_lane += ls_lane.item(); vn += 1
        vc /= vn; vl_lane /= vn
        vt = vc + LAMBDA_LANE * vl_lane
        sch.step(vt)
        print(f"  Train car={tc:.5f} lane={tl_lane:.5f}  Val car={vc:.5f} lane={vl_lane:.5f}")
        if vt < best:
            best = vt
            save_checkpoint({'dynamics': dyn.state_dict(), 'val_loss': vt},
                            os.path.join(SAVE_DIR, 'dyn.tar'))
    print(f"v5 Dyn best: {best:.5f}")


# ================================================================
# Stage 3: E2E fine-tune
# ================================================================
def stage3_e2e():
    print("\n" + "=" * 60); print("Stage 3 [v5]: E2E fine-tune"); print("=" * 60)
    ds = SeqLaneDataset(DATA_DIR, seq_len=ROLLOUT_K + 1)
    n_val = int(len(ds) * 0.1)
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=32, shuffle=False, drop_last=True)

    enc = PhysicsEncoderLane().to(DEVICE)
    dec = PhysicsDecoderLane().to(DEVICE)
    dyn = LaneAugmentedV5().to(DEVICE)
    ae_ck = load_checkpoint(os.path.join(SAVE_DIR, 'ae.tar'))
    enc.load_state_dict(ae_ck['encoder']); dec.load_state_dict(ae_ck['decoder'])
    dy_ck = load_checkpoint(os.path.join(SAVE_DIR, 'dyn.tar'))
    dyn.load_state_dict(dy_ck['dynamics'])

    params = list(enc.parameters()) + list(dec.parameters()) + list(dyn.parameters())
    opt = torch.optim.Adam(params, lr=3e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')

    for epoch in range(E2E_EPOCHS):
        enc.train(); dec.train(); dyn.train()
        ti, tc, tl_lane, nb = 0, 0, 0, 0
        for stack0, fut_imgs, fut_phys, fut_acts, wp_norm in tqdm(tr_loader, desc=f"v5 E2E {epoch+1}/{E2E_EPOCHS}"):
            stack0 = stack0.to(DEVICE); fut_imgs = fut_imgs.to(DEVICE)
            fut_phys = fut_phys.to(DEVICE); fut_acts = fut_acts.to(DEVICE)
            wp_norm = wp_norm.to(DEVICE)
            B = stack0.size(0)

            z_obs = enc(stack0)
            # Build 31-dim z: car[0:2]=fut_phys[:,0,0:2] (x,y ref rel), car[2:11]=z_obs car part, lane=z_obs lane
            z = torch.zeros(B, 31, device=DEVICE)
            z[:, 0:2]  = fut_phys[:, 0, 0:2]
            z[:, 2:11] = z_obs[:, :CAR_ENCODER_DIM]
            z[:, 11:]  = z_obs[:, CAR_ENCODER_DIM:]

            img_l, car_l, lane_l = 0, 0, 0
            # Reconstruct frame 0
            r0 = dec(torch.cat([z[:, 2:11], z[:, 11:]], dim=-1))
            img_l  += ((r0 - fut_imgs[:, 0]) ** 2).mean()
            car_l  += ((z[:, 2:11] - fut_phys[:, 0, 2:11]) ** 2).mean()
            lane_l += ((z[:, 11:]  - wp_norm[:, 0])         ** 2).mean()

            for k in range(ROLLOUT_K):
                z, _ = dyn(z, fut_acts[:, k])
                rec = dec(torch.cat([z[:, 2:11], z[:, 11:]], dim=-1))
                img_l  += ((rec - fut_imgs[:, k + 1]) ** 2).mean()
                car_l  += ((z[:, 2:11] - fut_phys[:, k + 1, 2:11]) ** 2).mean()
                lane_l += ((z[:, 11:]  - wp_norm[:, k + 1])        ** 2).mean()
            img_l  /= (ROLLOUT_K + 1); car_l /= (ROLLOUT_K + 1); lane_l /= (ROLLOUT_K + 1)
            loss = LAMBDA_IMG * img_l + LAMBDA_CAR * car_l + LAMBDA_LANE * lane_l
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            ti += img_l.item(); tc += car_l.item(); tl_lane += lane_l.item(); nb += 1
        ti /= nb; tc /= nb; tl_lane /= nb

        enc.eval(); dec.eval(); dyn.eval()
        vi, vc, vl_lane, vn = 0, 0, 0, 0
        with torch.no_grad():
            for stack0, fut_imgs, fut_phys, fut_acts, wp_norm in val_loader:
                stack0 = stack0.to(DEVICE); fut_imgs = fut_imgs.to(DEVICE)
                fut_phys = fut_phys.to(DEVICE); fut_acts = fut_acts.to(DEVICE)
                wp_norm = wp_norm.to(DEVICE)
                B = stack0.size(0)
                z_obs = enc(stack0)
                z = torch.zeros(B, 31, device=DEVICE)
                z[:, 0:2]  = fut_phys[:, 0, 0:2]
                z[:, 2:11] = z_obs[:, :CAR_ENCODER_DIM]
                z[:, 11:]  = z_obs[:, CAR_ENCODER_DIM:]
                il, cl, ll = 0, 0, 0
                r0 = dec(torch.cat([z[:, 2:11], z[:, 11:]], dim=-1))
                il += ((r0 - fut_imgs[:, 0]) ** 2).mean().item()
                cl += ((z[:, 2:11] - fut_phys[:, 0, 2:11]) ** 2).mean().item()
                ll += ((z[:, 11:]  - wp_norm[:, 0]) ** 2).mean().item()
                for k in range(ROLLOUT_K):
                    z, _ = dyn(z, fut_acts[:, k])
                    rec = dec(torch.cat([z[:, 2:11], z[:, 11:]], dim=-1))
                    il += ((rec - fut_imgs[:, k + 1]) ** 2).mean().item()
                    cl += ((z[:, 2:11] - fut_phys[:, k + 1, 2:11]) ** 2).mean().item()
                    ll += ((z[:, 11:]  - wp_norm[:, k + 1]) ** 2).mean().item()
                il /= (ROLLOUT_K + 1); cl /= (ROLLOUT_K + 1); ll /= (ROLLOUT_K + 1)
                vi += il; vc += cl; vl_lane += ll; vn += 1
        vi /= vn; vc /= vn; vl_lane /= vn
        vt = LAMBDA_IMG * vi + LAMBDA_CAR * vc + LAMBDA_LANE * vl_lane
        sch.step(vt)
        print(f"  Train img={ti:.5f} car={tc:.5f} lane={tl_lane:.5f}"
              f"  Val img={vi:.5f} car={vc:.5f} lane={vl_lane:.5f}")
        if vt < best:
            best = vt
            save_checkpoint({'encoder': enc.state_dict(),
                             'decoder': dec.state_dict(),
                             'dynamics': dyn.state_dict(),
                             'val_loss': vt},
                            os.path.join(SAVE_DIR, 'best.tar'))
    print(f"v5 E2E best: {best:.5f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["ae", "dyn", "e2e", "all"])
    args = p.parse_args()
    os.makedirs(SAVE_DIR, exist_ok=True)
    if args.stage in ("ae", "all"):  stage1_ae()
    if args.stage in ("dyn", "all"): stage2_dyn()
    if args.stage in ("e2e", "all"): stage3_e2e()
