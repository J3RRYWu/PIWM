"""V2: End-to-end fine-tune.

Start from V1 (multistep dynamics + original AE).
Unfreeze everything and train with ROLLOUT IMAGE MSE:
    z_0 = encoder(image_0)
    z_k = dyn(z_{k-1}, a_{k-1})  k = 1..K
    loss = sum_k MSE(decoder(z_k[9dims]), image_k)

Plus physics supervision on z to prevent latent drift.

Output: checkpoints/piwm_v2/best.tar
"""

# --- auto-added: make repo root importable when run as a script ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end shim ---


import os, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from config import (DEVICE, DATA_DIR, PHYSICS_MEAN, PHYSICS_STD, FRAME_STACK,
                    ENCODER_STATE_INDICES, ENCODER_MEAN, ENCODER_STD)
from models.encoder import PhysicsEncoder
from models.decoder import PhysicsDecoder
from models.dynamics import PhysicsDynamics
from utils import save_checkpoint, load_checkpoint

ROLLOUT_K = 8
SAVE_DIR = "checkpoints/piwm_v2/"
EPOCHS = 30
LR = 3e-4
LAMBDA_PHYS = 1.0      # physics supervision on first 9 dims
LAMBDA_IMG = 10.0      # image reconstruction


class RolloutDataset(Dataset):
    def __init__(self, data_dir, seq_len=ROLLOUT_K + 1):
        files = sorted(glob.glob(f"{data_dir}/*.npz"))
        self.seq_len = seq_len
        self.imgs_list = []
        self.phys_list = []
        self.acts_list = []
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
                self.imgs_list.append(imgs)
                self.phys_list.append(phys)
                self.acts_list.append(acts)
                for t in range(FRAME_STACK - 1, n - seq_len):
                    self.indices.append((ep_idx, t))
            except: pass
        print(f"RolloutDataset: {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep, t0 = self.indices[idx]
        imgs = self.imgs_list[ep]
        phys = self.phys_list[ep]
        acts = self.acts_list[ep]
        stack0 = np.stack([imgs[t0 - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
        future_imgs = imgs[t0:t0 + self.seq_len]
        future_phys = phys[t0:t0 + self.seq_len]
        future_acts = acts[t0:t0 + self.seq_len - 1]
        phys_norm = (future_phys - PHYSICS_MEAN) / PHYSICS_STD
        return (torch.tensor(stack0, dtype=torch.float32),
                torch.tensor(future_imgs, dtype=torch.float32).unsqueeze(1),  # (K+1, 1, 64, 64)
                torch.tensor(phys_norm, dtype=torch.float32),
                torch.tensor(future_acts, dtype=torch.float32))


def train():
    print("=" * 60)
    print("V2: End-to-end fine-tune (rollout image MSE)")
    print("=" * 60)

    # Load V0 AE + V1 Dynamics as init
    enc = PhysicsEncoder().to(DEVICE)
    dec = PhysicsDecoder().to(DEVICE)
    dyn = PhysicsDynamics().to(DEVICE)
    ae_ck = load_checkpoint("checkpoints/autoencoder/best.tar")
    enc.load_state_dict(ae_ck['encoder'])
    dec.load_state_dict(ae_ck['decoder'])
    try:
        dy_ck = load_checkpoint("checkpoints/dynamics_v1/best.tar")
        dyn.load_state_dict(dy_ck['dynamics'])
        print("Loaded V1 multistep dynamics.")
    except:
        dy_ck = load_checkpoint("checkpoints/dynamics/best.tar")
        dyn.load_state_dict(dy_ck['dynamics'])
        print("Fell back to V0 dynamics.")

    ds = RolloutDataset(DATA_DIR)
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
        tloss, t_img, t_phys, nb = 0, 0, 0, 0
        pbar = tqdm(tr_loader, desc=f"E2E {epoch+1}/{EPOCHS}")
        for stack0, future_imgs, future_phys, future_acts in pbar:
            stack0 = stack0.to(DEVICE)
            future_imgs = future_imgs.to(DEVICE)   # (B, K+1, 1, 64, 64)
            future_phys = future_phys.to(DEVICE)   # (B, K+1, 11)
            future_acts = future_acts.to(DEVICE)   # (B, K, 3)

            # z_0 from encoder (9-dim); combine with GT x,y to form 11-dim state
            z_enc = enc(stack0)  # (B, 9)
            z = torch.zeros(stack0.size(0), 11, device=DEVICE)
            z[:, 0:2] = future_phys[:, 0, 0:2]   # x, y from GT
            z[:, 2:11] = z_enc

            img_loss = 0
            phys_loss = 0

            # Step 0 reconstruction
            recon0 = dec(z[:, 2:11])
            img_loss = img_loss + ((recon0 - future_imgs[:, 0]) ** 2).mean()
            phys_loss = phys_loss + ((z_enc - future_phys[:, 0, 2:11]) ** 2).mean()

            for k in range(ROLLOUT_K):
                z, _ = dyn(z, future_acts[:, k])
                recon = dec(z[:, 2:11])
                img_loss = img_loss + ((recon - future_imgs[:, k + 1]) ** 2).mean()
                phys_loss = phys_loss + ((z[:, 2:11] - future_phys[:, k + 1, 2:11]) ** 2).mean()

            img_loss /= (ROLLOUT_K + 1)
            phys_loss /= (ROLLOUT_K + 1)
            loss = LAMBDA_IMG * img_loss + LAMBDA_PHYS * phys_loss

            opt.zero_grad(); loss.backward(); opt.step()
            tloss += loss.item(); t_img += img_loss.item(); t_phys += phys_loss.item(); nb += 1
            pbar.set_postfix(img=f"{img_loss.item():.4f}", phys=f"{phys_loss.item():.4f}")

        tloss /= nb; t_img /= nb; t_phys /= nb

        # Val
        enc.eval(); dec.eval(); dyn.eval()
        vloss, v_img, v_phys, vn = 0, 0, 0, 0
        with torch.no_grad():
            for stack0, future_imgs, future_phys, future_acts in val_loader:
                stack0 = stack0.to(DEVICE)
                future_imgs = future_imgs.to(DEVICE)
                future_phys = future_phys.to(DEVICE)
                future_acts = future_acts.to(DEVICE)
                z_enc = enc(stack0)
                z = torch.zeros(stack0.size(0), 11, device=DEVICE)
                z[:, 0:2] = future_phys[:, 0, 0:2]
                z[:, 2:11] = z_enc

                img_l, ph_l = 0, 0
                recon0 = dec(z[:, 2:11])
                img_l += ((recon0 - future_imgs[:, 0]) ** 2).mean()
                ph_l += ((z_enc - future_phys[:, 0, 2:11]) ** 2).mean()
                for k in range(ROLLOUT_K):
                    z, _ = dyn(z, future_acts[:, k])
                    recon = dec(z[:, 2:11])
                    img_l += ((recon - future_imgs[:, k + 1]) ** 2).mean()
                    ph_l += ((z[:, 2:11] - future_phys[:, k + 1, 2:11]) ** 2).mean()
                img_l /= (ROLLOUT_K + 1); ph_l /= (ROLLOUT_K + 1)
                vloss += (LAMBDA_IMG * img_l + LAMBDA_PHYS * ph_l).item()
                v_img += img_l.item(); v_phys += ph_l.item(); vn += 1
        vloss /= vn; v_img /= vn; v_phys /= vn
        sch.step(vloss)
        print(f"  Train img={t_img:.5f} phys={t_phys:.5f}  Val img={v_img:.5f} phys={v_phys:.5f}")

        if vloss < best_val:
            best_val = vloss
            save_checkpoint({
                'encoder': enc.state_dict(),
                'decoder': dec.state_dict(),
                'dynamics': dyn.state_dict(),
                'val_loss': vloss,
            }, os.path.join(SAVE_DIR, 'best.tar'))
    print(f"V2 done. Best: {best_val:.6f}")


if __name__ == "__main__":
    train()
