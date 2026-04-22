"""V2_rel: PIWM with RELATIVE coords + multistep + e2e training.

Pipeline:
  Stage 1: Train Encoder/Decoder (9-dim, physics-supervised, relative coords)
  Stage 2: Train Dynamics (multistep rollout, relative coords)
  Stage 3: E2E fine-tune (rollout image MSE + physics loss)

All in one script. Output: checkpoints/piwm_v2_rel/{ae.tar, dyn.tar, best.tar}
"""

import os, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from config import (DEVICE, DATA_DIR, PHYSICS_MEAN_REL, PHYSICS_STD_REL,
                    ENCODER_STATE_INDICES, FRAME_STACK, AE_BATCH_SIZE, AE_LR,
                    LAMBDA_PHYSICS, LAMBDA_RESIDUAL)
from models.encoder import PhysicsEncoder
from models.decoder import PhysicsDecoder
from models.dynamics import PhysicsDynamics
from relative_coords import to_relative_np
from utils import save_checkpoint, load_checkpoint


SAVE_DIR = "checkpoints/piwm_v2_rel/"
ROLLOUT_K = 8
AE_EPOCHS = 30
DYN_EPOCHS = 60
E2E_EPOCHS = 20


# Monkey-patch Dynamics to use relative-coord normalization
def patch_dynamics_rel(dyn):
    """Rebuild kinematic scale constants using PHYSICS_*_REL."""
    from config import DT
    std = PHYSICS_STD_REL
    mean = PHYSICS_MEAN_REL
    dyn.dt_vx_x = torch.tensor(float(std[3] * DT / std[0])).to(dyn.dt_vx_x)
    dyn.dt_vy_y = torch.tensor(float(std[4] * DT / std[1])).to(dyn.dt_vy_y)
    dyn.dt_om_yaw = torch.tensor(float(std[5] * DT / std[2])).to(dyn.dt_om_yaw)
    dyn.mean_vx_dt_over_std_x = torch.tensor(float(mean[3] * DT / std[0])).to(dyn.mean_vx_dt_over_std_x)
    dyn.mean_vy_dt_over_std_y = torch.tensor(float(mean[4] * DT / std[1])).to(dyn.mean_vy_dt_over_std_y)
    dyn.mean_om_dt_over_std_yaw = torch.tensor(float(mean[5] * DT / std[2])).to(dyn.mean_om_dt_over_std_yaw)
    return dyn


class AEDataset(Dataset):
    """Single frame stack + relative physics at that frame (ref=frame itself)."""
    def __init__(self, data_dir):
        files = sorted(glob.glob(f"{data_dir}/*.npz"))
        self.items = []
        for f in files:
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
                physics = np.column_stack([pos, yaw, vel, omega, wheel, steer])
                n = len(imgs)
                if n < FRAME_STACK: continue
                # For AE: reference frame = current frame. All positions relative to self => x,y,yaw all zero.
                # Better approach: use FRAME_STACK-1 as ref (the oldest frame of the stack).
                # That way, the current frame's yaw_rel is meaningful (how much car has turned in FRAME_STACK frames).
                for t in range(FRAME_STACK - 1, n):
                    frames = np.stack([imgs[t - FRAME_STACK + 1 + i] for i in range(FRAME_STACK)], axis=0)
                    # Ref: earliest frame in stack
                    ref_phys = physics[t - FRAME_STACK + 1:t + 1]
                    rel = to_relative_np(ref_phys, ref_idx=0)
                    self.items.append((frames, rel[-1], imgs[t]))  # (stack, rel_state_at_t, target_img)
            except: pass
        print(f"AEDataset: {len(self.items)} samples")

    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        frames, rel_state, img = self.items[i]
        phys_norm = (rel_state - PHYSICS_MEAN_REL) / PHYSICS_STD_REL
        enc_target = phys_norm[ENCODER_STATE_INDICES]
        return (torch.tensor(frames, dtype=torch.float32),
                torch.tensor(enc_target, dtype=torch.float32),
                torch.tensor(img, dtype=torch.float32).unsqueeze(0))


class SeqDataset(Dataset):
    """Sequences for dynamics/e2e training, each with its own rel frame."""
    def __init__(self, data_dir, seq_len=ROLLOUT_K + 1):
        files = sorted(glob.glob(f"{data_dir}/*.npz"))
        self.seq_len = seq_len
        self.imgs_list, self.phys_list, self.acts_list = [], [], []
        self.indices = []
        for ep_idx, f in enumerate(files):
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
                n = len(imgs)
                if n < seq_len + FRAME_STACK: continue
                self.imgs_list.append(imgs); self.phys_list.append(phys); self.acts_list.append(acts)
                for t in range(FRAME_STACK - 1, n - seq_len):
                    self.indices.append((ep_idx, t))
            except: pass
        print(f"SeqDataset: {len(self.indices)} samples")

    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        ep, t0 = self.indices[i]
        imgs = self.imgs_list[ep]; phys = self.phys_list[ep]; acts = self.acts_list[ep]
        stack0 = np.stack([imgs[t0 - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
        fut_imgs = imgs[t0:t0 + self.seq_len]
        fut_phys = phys[t0:t0 + self.seq_len]
        fut_acts = acts[t0:t0 + self.seq_len - 1]
        rel_phys = to_relative_np(fut_phys, ref_idx=0)  # relative to t0's pose
        phys_norm = (rel_phys - PHYSICS_MEAN_REL) / PHYSICS_STD_REL
        return (torch.tensor(stack0, dtype=torch.float32),
                torch.tensor(fut_imgs, dtype=torch.float32).unsqueeze(1),
                torch.tensor(phys_norm, dtype=torch.float32),
                torch.tensor(fut_acts, dtype=torch.float32))


def stage1_ae():
    print("=" * 60)
    print("Stage 1: AE (relative coords)")
    print("=" * 60)
    ds = AEDataset(DATA_DIR)
    n_val = int(len(ds) * 0.1)
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=AE_BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=AE_BATCH_SIZE, shuffle=False, drop_last=True)

    enc = PhysicsEncoder().to(DEVICE)
    dec = PhysicsDecoder().to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=AE_LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    mse = nn.MSELoss()
    best = float('inf')

    for epoch in range(AE_EPOCHS):
        enc.train(); dec.train()
        tl, nb = 0, 0
        for fr, target_phys, target_img in tqdm(tr_loader, desc=f"AE {epoch+1}/{AE_EPOCHS}"):
            fr, target_phys, target_img = fr.to(DEVICE), target_phys.to(DEVICE), target_img.to(DEVICE)
            z = enc(fr)
            rec = dec(z)
            loss = mse(rec, target_img) + LAMBDA_PHYSICS * mse(z, target_phys)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        tl /= nb

        enc.eval(); dec.eval()
        vl, vn = 0, 0
        with torch.no_grad():
            for fr, target_phys, target_img in val_loader:
                fr, target_phys, target_img = fr.to(DEVICE), target_phys.to(DEVICE), target_img.to(DEVICE)
                z = enc(fr); rec = dec(z)
                vl += (mse(rec, target_img) + LAMBDA_PHYSICS * mse(z, target_phys)).item(); vn += 1
        vl /= vn
        sch.step(vl)
        print(f"  Train {tl:.5f}  Val {vl:.5f}")
        if vl < best:
            best = vl
            save_checkpoint({'encoder': enc.state_dict(), 'decoder': dec.state_dict(), 'val_loss': vl},
                            os.path.join(SAVE_DIR, 'ae.tar'))
    print(f"AE best: {best:.5f}")


def stage2_dyn():
    print("\n" + "=" * 60)
    print("Stage 2: Dynamics (multistep rollout, relative coords)")
    print("=" * 60)
    ds = SeqDataset(DATA_DIR, seq_len=ROLLOUT_K + 1)
    n_val = int(len(ds) * 0.1)
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=128, shuffle=False, drop_last=True)

    dyn = PhysicsDynamics().to(DEVICE)
    patch_dynamics_rel(dyn)
    opt = torch.optim.Adam(dyn.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')

    for epoch in range(DYN_EPOCHS):
        dyn.train()
        tl, nb = 0, 0
        for stack0, fut_imgs, fut_phys, fut_acts in tqdm(tr_loader, desc=f"Dyn {epoch+1}/{DYN_EPOCHS}"):
            fut_phys = fut_phys.to(DEVICE); fut_acts = fut_acts.to(DEVICE)
            z = fut_phys[:, 0]
            ls = 0; lres = 0
            for k in range(ROLLOUT_K):
                z_next, res = dyn(z, fut_acts[:, k])
                ls = ls + ((z_next - fut_phys[:, k + 1]) ** 2).mean()
                lres = lres + (res ** 2).mean()
                z = z_next
            ls /= ROLLOUT_K; lres /= ROLLOUT_K
            loss = ls + LAMBDA_RESIDUAL * lres
            opt.zero_grad(); loss.backward(); opt.step()
            tl += ls.item(); nb += 1
        tl /= nb

        dyn.eval()
        vl, vn = 0, 0
        with torch.no_grad():
            for stack0, fut_imgs, fut_phys, fut_acts in val_loader:
                fut_phys = fut_phys.to(DEVICE); fut_acts = fut_acts.to(DEVICE)
                z = fut_phys[:, 0]; ls = 0
                for k in range(ROLLOUT_K):
                    z_next, _ = dyn(z, fut_acts[:, k])
                    ls = ls + ((z_next - fut_phys[:, k + 1]) ** 2).mean()
                    z = z_next
                ls /= ROLLOUT_K
                vl += ls.item(); vn += 1
        vl /= vn
        sch.step(vl)
        print(f"  Train {tl:.6f}  Val {vl:.6f}")
        if vl < best:
            best = vl
            save_checkpoint({'dynamics': dyn.state_dict(), 'val_loss': vl},
                            os.path.join(SAVE_DIR, 'dyn.tar'))
    print(f"Dyn best: {best:.6f}")


def stage3_e2e():
    print("\n" + "=" * 60)
    print("Stage 3: E2E fine-tune (rollout image MSE + physics)")
    print("=" * 60)
    ds = SeqDataset(DATA_DIR, seq_len=ROLLOUT_K + 1)
    n_val = int(len(ds) * 0.1)
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=32, shuffle=False, drop_last=True)

    enc = PhysicsEncoder().to(DEVICE)
    dec = PhysicsDecoder().to(DEVICE)
    dyn = PhysicsDynamics().to(DEVICE)
    patch_dynamics_rel(dyn)
    ae_ck = load_checkpoint(os.path.join(SAVE_DIR, 'ae.tar'))
    enc.load_state_dict(ae_ck['encoder']); dec.load_state_dict(ae_ck['decoder'])
    dy_ck = load_checkpoint(os.path.join(SAVE_DIR, 'dyn.tar'))
    dyn.load_state_dict(dy_ck['dynamics'])
    patch_dynamics_rel(dyn)  # re-apply after loading

    params = list(enc.parameters()) + list(dec.parameters()) + list(dyn.parameters())
    opt = torch.optim.Adam(params, lr=3e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')
    LAMBDA_IMG = 10.0; LAMBDA_PHYS = 1.0

    for epoch in range(E2E_EPOCHS):
        enc.train(); dec.train(); dyn.train()
        ti, tp, nb = 0, 0, 0
        for stack0, fut_imgs, fut_phys, fut_acts in tqdm(tr_loader, desc=f"E2E {epoch+1}/{E2E_EPOCHS}"):
            stack0, fut_imgs, fut_phys, fut_acts = stack0.to(DEVICE), fut_imgs.to(DEVICE), fut_phys.to(DEVICE), fut_acts.to(DEVICE)
            B = stack0.size(0)
            z_enc = enc(stack0)
            z = torch.zeros(B, 11, device=DEVICE)
            z[:, 0:2] = fut_phys[:, 0, 0:2]  # x_rel, y_rel from GT (should be ~0 since t0 is origin)
            z[:, 2:11] = z_enc

            img_l = 0; phys_l = 0
            recon0 = dec(z[:, 2:11])
            img_l += ((recon0 - fut_imgs[:, 0]) ** 2).mean()
            phys_l += ((z_enc - fut_phys[:, 0, 2:11]) ** 2).mean()
            for k in range(ROLLOUT_K):
                z, _ = dyn(z, fut_acts[:, k])
                rec = dec(z[:, 2:11])
                img_l += ((rec - fut_imgs[:, k + 1]) ** 2).mean()
                phys_l += ((z[:, 2:11] - fut_phys[:, k + 1, 2:11]) ** 2).mean()
            img_l /= (ROLLOUT_K + 1); phys_l /= (ROLLOUT_K + 1)
            loss = LAMBDA_IMG * img_l + LAMBDA_PHYS * phys_l

            opt.zero_grad(); loss.backward(); opt.step()
            ti += img_l.item(); tp += phys_l.item(); nb += 1
        ti /= nb; tp /= nb

        enc.eval(); dec.eval(); dyn.eval()
        vi, vp, vn = 0, 0, 0
        with torch.no_grad():
            for stack0, fut_imgs, fut_phys, fut_acts in val_loader:
                stack0, fut_imgs, fut_phys, fut_acts = stack0.to(DEVICE), fut_imgs.to(DEVICE), fut_phys.to(DEVICE), fut_acts.to(DEVICE)
                B = stack0.size(0)
                z_enc = enc(stack0)
                z = torch.zeros(B, 11, device=DEVICE)
                z[:, 0:2] = fut_phys[:, 0, 0:2]; z[:, 2:11] = z_enc
                il, pl = 0, 0
                r0 = dec(z[:, 2:11])
                il += ((r0 - fut_imgs[:, 0]) ** 2).mean()
                pl += ((z_enc - fut_phys[:, 0, 2:11]) ** 2).mean()
                for k in range(ROLLOUT_K):
                    z, _ = dyn(z, fut_acts[:, k])
                    r = dec(z[:, 2:11])
                    il += ((r - fut_imgs[:, k + 1]) ** 2).mean()
                    pl += ((z[:, 2:11] - fut_phys[:, k + 1, 2:11]) ** 2).mean()
                il /= (ROLLOUT_K + 1); pl /= (ROLLOUT_K + 1)
                vi += il.item(); vp += pl.item(); vn += 1
        vi /= vn; vp /= vn
        vt = LAMBDA_IMG * vi + LAMBDA_PHYS * vp
        sch.step(vt)
        print(f"  Train img={ti:.5f} phys={tp:.5f}  Val img={vi:.5f} phys={vp:.5f}")
        if vt < best:
            best = vt
            save_checkpoint({
                'encoder': enc.state_dict(),
                'decoder': dec.state_dict(),
                'dynamics': dyn.state_dict(),
                'val_loss': vt,
            }, os.path.join(SAVE_DIR, 'best.tar'))
    print(f"E2E best: {best:.5f}")


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    stage1_ae()
    stage2_dyn()
    stage3_e2e()
