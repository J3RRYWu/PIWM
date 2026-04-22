"""V3: Full PIWM with free latent dims (9 physics + 8 free), multi-step + e2e.

Unified training: encoder+decoder+dynamics trained together with image rollout
MSE and physics supervision (on first 9 dims of encoder output).

Output: checkpoints/piwm_v3/best.tar
"""

import os, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from config import (DEVICE, DATA_DIR, PHYSICS_MEAN, PHYSICS_STD, FRAME_STACK,
                    ENCODER_STATE_INDICES, ENCODER_MEAN, ENCODER_STD)
from models.piwm_v3 import EncoderV3, DecoderV3, DynamicsV3
from utils import save_checkpoint, load_checkpoint

ROLLOUT_K = 8
SAVE_DIR = "checkpoints/piwm_v3/"
EPOCHS = 50
LR = 1e-3
LAMBDA_PHYS = 1.0
LAMBDA_IMG = 10.0


class V3Dataset(Dataset):
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
        print(f"V3Dataset: {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep, t0 = self.indices[idx]
        imgs = self.imgs_list[ep]
        phys = self.phys_list[ep]
        acts = self.acts_list[ep]
        stack0 = np.stack([imgs[t0 - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
        # Also provide stack at each future step (for teacher encoding of free dims is NOT needed; we roll out)
        fut_imgs = imgs[t0:t0 + self.seq_len]
        fut_phys = phys[t0:t0 + self.seq_len]
        fut_acts = acts[t0:t0 + self.seq_len - 1]
        phys_norm = (fut_phys - PHYSICS_MEAN) / PHYSICS_STD
        return (torch.tensor(stack0, dtype=torch.float32),
                torch.tensor(fut_imgs, dtype=torch.float32).unsqueeze(1),
                torch.tensor(phys_norm, dtype=torch.float32),
                torch.tensor(fut_acts, dtype=torch.float32))


def train():
    print("=" * 60)
    print("V3: PIWM with free dims (9 physics + 8 free), rollout training")
    print("=" * 60)

    enc = EncoderV3().to(DEVICE)
    dec = DecoderV3().to(DEVICE)
    dyn = DynamicsV3().to(DEVICE)

    ds = V3Dataset(DATA_DIR)
    n_val = int(len(ds) * 0.1)
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=32, shuffle=False, drop_last=True)

    params = list(enc.parameters()) + list(dec.parameters()) + list(dyn.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best_val = float('inf')

    for epoch in range(EPOCHS):
        enc.train(); dec.train(); dyn.train()
        t_img, t_phys, nb = 0, 0, 0
        pbar = tqdm(tr_loader, desc=f"V3 {epoch+1}/{EPOCHS}")
        for stack0, fut_imgs, fut_phys, fut_acts in pbar:
            stack0 = stack0.to(DEVICE)
            fut_imgs = fut_imgs.to(DEVICE)
            fut_phys = fut_phys.to(DEVICE)
            fut_acts = fut_acts.to(DEVICE)
            B = stack0.size(0)

            z_enc = enc(stack0)  # (B, 17) = [9 physics + 8 free]
            physics_pred = z_enc[:, :9]
            free_pred = z_enc[:, 9:]

            # Build 19-dim state: [x, y, physics_pred(9), free(8)]
            z = torch.zeros(B, 19, device=DEVICE)
            z[:, 0:2] = fut_phys[:, 0, 0:2]   # x, y from GT
            z[:, 2:11] = physics_pred          # 9 physics dims
            z[:, 11:19] = free_pred            # 8 free dims

            img_loss = 0
            phys_loss = 0

            # Step 0 decode
            recon0 = dec(torch.cat([physics_pred, free_pred], dim=-1))
            img_loss = img_loss + ((recon0 - fut_imgs[:, 0]) ** 2).mean()
            phys_loss = phys_loss + ((physics_pred - fut_phys[:, 0, 2:11]) ** 2).mean()

            for k in range(ROLLOUT_K):
                z, _ = dyn(z, fut_acts[:, k])
                phys_part = z[:, 2:11]
                free_part = z[:, 11:19]
                recon = dec(torch.cat([phys_part, free_part], dim=-1))
                img_loss = img_loss + ((recon - fut_imgs[:, k + 1]) ** 2).mean()
                phys_loss = phys_loss + ((phys_part - fut_phys[:, k + 1, 2:11]) ** 2).mean()

            img_loss /= (ROLLOUT_K + 1)
            phys_loss /= (ROLLOUT_K + 1)
            loss = LAMBDA_IMG * img_loss + LAMBDA_PHYS * phys_loss

            opt.zero_grad(); loss.backward(); opt.step()
            t_img += img_loss.item(); t_phys += phys_loss.item(); nb += 1
            pbar.set_postfix(img=f"{img_loss.item():.4f}", phys=f"{phys_loss.item():.4f}")

        t_img /= nb; t_phys /= nb

        enc.eval(); dec.eval(); dyn.eval()
        v_img, v_phys, vn = 0, 0, 0
        with torch.no_grad():
            for stack0, fut_imgs, fut_phys, fut_acts in val_loader:
                stack0 = stack0.to(DEVICE); fut_imgs = fut_imgs.to(DEVICE)
                fut_phys = fut_phys.to(DEVICE); fut_acts = fut_acts.to(DEVICE)
                B = stack0.size(0)

                z_enc = enc(stack0)
                physics_pred = z_enc[:, :9]; free_pred = z_enc[:, 9:]
                z = torch.zeros(B, 19, device=DEVICE)
                z[:, 0:2] = fut_phys[:, 0, 0:2]
                z[:, 2:11] = physics_pred; z[:, 11:19] = free_pred

                il, pl = 0, 0
                recon0 = dec(torch.cat([physics_pred, free_pred], dim=-1))
                il += ((recon0 - fut_imgs[:, 0]) ** 2).mean()
                pl += ((physics_pred - fut_phys[:, 0, 2:11]) ** 2).mean()
                for k in range(ROLLOUT_K):
                    z, _ = dyn(z, fut_acts[:, k])
                    pp = z[:, 2:11]; fp = z[:, 11:19]
                    recon = dec(torch.cat([pp, fp], dim=-1))
                    il += ((recon - fut_imgs[:, k + 1]) ** 2).mean()
                    pl += ((pp - fut_phys[:, k + 1, 2:11]) ** 2).mean()
                il /= (ROLLOUT_K + 1); pl /= (ROLLOUT_K + 1)
                v_img += il.item(); v_phys += pl.item(); vn += 1
        v_img /= vn; v_phys /= vn
        val_total = LAMBDA_IMG * v_img + LAMBDA_PHYS * v_phys
        sch.step(val_total)
        print(f"  Train img={t_img:.5f} phys={t_phys:.5f}  Val img={v_img:.5f} phys={v_phys:.5f}")

        if val_total < best_val:
            best_val = val_total
            save_checkpoint({
                'encoder': enc.state_dict(),
                'decoder': dec.state_dict(),
                'dynamics': dyn.state_dict(),
                'val_loss': val_total,
            }, os.path.join(SAVE_DIR, 'best.tar'))
    print(f"V3 done. Best: {best_val:.6f}")


if __name__ == "__main__":
    train()
