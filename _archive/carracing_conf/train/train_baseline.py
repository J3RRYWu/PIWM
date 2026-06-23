"""Two-stage training of baseline world model (mirrors PIWM training).

Stage 1: Train BaselineEncoder + BaselineDecoder as pure autoencoder.
Stage 2: Freeze encoder/decoder, train BaselineDynamics on latent transitions.
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

from config import DEVICE, DATA_DIR, FRAME_STACK
from models.baseline import BaselineEncoder, BaselineDecoder, BaselineDynamics
from utils import save_checkpoint, load_checkpoint


AE_SAVE_DIR = "checkpoints/baseline_ae/"
DYN_SAVE_DIR = "checkpoints/baseline_dynamics/"
AE_EPOCHS = 50
DYN_EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3


class ImageDataset(Dataset):
    """Returns (stacked_frames, target_image) for autoencoder training."""

    def __init__(self, data_dir):
        files = sorted(glob.glob(f"{data_dir}/*.npz"))
        self.imgs_list = []
        self.indices = []
        for ep_idx, f in enumerate(files):
            try:
                d = np.load(f, allow_pickle=True)
                imgs = d["imgs"].astype(np.float32)
                if imgs.max() > 1.0: imgs /= 255.0
                n = len(imgs)
                if n < FRAME_STACK:
                    continue
                self.imgs_list.append(imgs)
                for t in range(FRAME_STACK - 1, n):
                    self.indices.append((ep_idx, t))
            except: pass
        print(f"ImageDataset: {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, t = self.indices[idx]
        imgs = self.imgs_list[ep_idx]
        stacked = np.stack([imgs[t - FRAME_STACK + 1 + i] for i in range(FRAME_STACK)], axis=0)
        target = imgs[t]
        return (torch.tensor(stacked, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32).unsqueeze(0))


class TransitionDataset(Dataset):
    """Returns (stacked_t, stacked_t+1, action_t) for dynamics training."""

    def __init__(self, data_dir):
        files = sorted(glob.glob(f"{data_dir}/*.npz"))
        self.imgs_list = []
        self.act_list = []
        self.indices = []
        for ep_idx, f in enumerate(files):
            try:
                d = np.load(f, allow_pickle=True)
                imgs = d["imgs"].astype(np.float32)
                if imgs.max() > 1.0: imgs /= 255.0
                acts = d["action"].astype(np.float32)
                n = len(imgs)
                if n < FRAME_STACK + 1:
                    continue
                self.imgs_list.append(imgs)
                self.act_list.append(acts)
                for t in range(FRAME_STACK - 1, n - 1):
                    self.indices.append((ep_idx, t))
            except: pass
        print(f"TransitionDataset: {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, t = self.indices[idx]
        imgs = self.imgs_list[ep_idx]
        acts = self.act_list[ep_idx]
        st_t = np.stack([imgs[t - FRAME_STACK + 1 + i] for i in range(FRAME_STACK)], axis=0)
        st_t1 = np.stack([imgs[t + 1 - FRAME_STACK + 1 + i] for i in range(FRAME_STACK)], axis=0)
        action = acts[t]
        return (torch.tensor(st_t, dtype=torch.float32),
                torch.tensor(st_t1, dtype=torch.float32),
                torch.tensor(action, dtype=torch.float32))


def train_ae():
    print("=" * 60)
    print("Stage 1: Baseline AE (pure reconstruction)")
    print("=" * 60)

    ds = ImageDataset(DATA_DIR)
    n_val = int(len(ds) * 0.1)
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    enc = BaselineEncoder().to(DEVICE)
    dec = BaselineDecoder().to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    mse = nn.MSELoss()
    best_val = float('inf')

    for epoch in range(AE_EPOCHS):
        enc.train(); dec.train()
        tloss, nb = 0, 0
        for fr, tg in tqdm(tr_loader, desc=f"AE {epoch+1}/{AE_EPOCHS}"):
            fr, tg = fr.to(DEVICE), tg.to(DEVICE)
            z = enc(fr)
            recon = dec(z)
            loss = mse(recon, tg)
            opt.zero_grad(); loss.backward(); opt.step()
            tloss += loss.item(); nb += 1
        tloss /= nb

        enc.eval(); dec.eval()
        vloss, vn = 0, 0
        with torch.no_grad():
            for fr, tg in val_loader:
                fr, tg = fr.to(DEVICE), tg.to(DEVICE)
                vloss += mse(dec(enc(fr)), tg).item(); vn += 1
        vloss /= vn
        sch.step(vloss)
        print(f"  Train: {tloss:.6f}  Val: {vloss:.6f}")

        if vloss < best_val:
            best_val = vloss
            save_checkpoint({'encoder': enc.state_dict(), 'decoder': dec.state_dict(),
                             'val_loss': vloss}, os.path.join(AE_SAVE_DIR, 'best.tar'))
    print(f"Baseline AE done. Best val loss: {best_val:.6f}")


def train_dyn():
    print("\n" + "=" * 60)
    print("Stage 2: Baseline Dynamics (MLP on latent)")
    print("=" * 60)

    enc = BaselineEncoder().to(DEVICE)
    dec = BaselineDecoder().to(DEVICE)
    ae_ckpt = load_checkpoint(os.path.join(AE_SAVE_DIR, 'best.tar'))
    enc.load_state_dict(ae_ckpt['encoder'])
    dec.load_state_dict(ae_ckpt['decoder'])
    enc.eval(); dec.eval()
    for p in enc.parameters(): p.requires_grad_(False)
    for p in dec.parameters(): p.requires_grad_(False)

    ds = TransitionDataset(DATA_DIR)
    n_val = int(len(ds) * 0.1)
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=128, shuffle=False, drop_last=True)

    dyn = BaselineDynamics().to(DEVICE)
    opt = torch.optim.Adam(dyn.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    mse = nn.MSELoss()
    best_val = float('inf')

    for epoch in range(DYN_EPOCHS):
        dyn.train()
        tloss, nb = 0, 0
        for st_t, st_t1, act in tqdm(tr_loader, desc=f"Dyn {epoch+1}/{DYN_EPOCHS}"):
            st_t, st_t1, act = st_t.to(DEVICE), st_t1.to(DEVICE), act.to(DEVICE)
            with torch.no_grad():
                z_t = enc(st_t)
                z_t1 = enc(st_t1)
            z_pred = dyn(z_t, act)
            loss = mse(z_pred, z_t1)
            opt.zero_grad(); loss.backward(); opt.step()
            tloss += loss.item(); nb += 1
        tloss /= nb

        dyn.eval()
        vloss, vn = 0, 0
        with torch.no_grad():
            for st_t, st_t1, act in val_loader:
                st_t, st_t1, act = st_t.to(DEVICE), st_t1.to(DEVICE), act.to(DEVICE)
                z_t, z_t1 = enc(st_t), enc(st_t1)
                vloss += mse(dyn(z_t, act), z_t1).item(); vn += 1
        vloss /= vn
        sch.step(vloss)
        print(f"  Train: {tloss:.6f}  Val: {vloss:.6f}")

        if vloss < best_val:
            best_val = vloss
            save_checkpoint({'dynamics': dyn.state_dict(), 'val_loss': vloss},
                           os.path.join(DYN_SAVE_DIR, 'best.tar'))
    print(f"Baseline Dynamics done. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    train_ae()
    train_dyn()
