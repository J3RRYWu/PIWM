"""Ablation training for PIWM-bicycle variants.

Variants (select with --variant):
  v1  : original BicycleDynamics (keeps dynamics_bicycle.py intact)
  v2  : + residual masked on [x, y]
  v3  : v2 + physically constrained wheelbase L ∈ [2, 8]
  v4  : dynamic bicycle (tire forces) + masked residual

All variants use the SAME 3-stage protocol as train_piwm_bicycle.py.
Each variant writes to its own checkpoint dir:
  checkpoints/piwm_bike_v{1,2,3,4}/{dyn.tar, best.tar}
"""

import os, glob, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from config import (DEVICE, DATA_DIR, PHYSICS_MEAN_REL, PHYSICS_STD_REL,
                    FRAME_STACK, LAMBDA_RESIDUAL)
from models.encoder import PhysicsEncoder
from models.decoder import PhysicsDecoder
from models.dynamics_bicycle import BicycleDynamics               # v1 (original)
from models.dynamics_bicycle_v2 import BicycleDynamicsV2           # v2: masked residual
from models.dynamics_bicycle_v3 import BicycleDynamicsV3           # v3: masked + L-cap
from models.dynamics_bicycle_v4 import BicycleDynamicsV4           # v4: dynamic bicycle
from relative_coords import to_relative_np
from utils import save_checkpoint, load_checkpoint


ROLLOUT_K = 8
DYN_EPOCHS = 60
E2E_EPOCHS = 20


def make_model(variant):
    if variant == "v1": return BicycleDynamics()
    if variant == "v2": return BicycleDynamicsV2()
    if variant == "v3": return BicycleDynamicsV3()
    if variant == "v4": return BicycleDynamicsV4()
    raise ValueError(f"Unknown variant: {variant}")


class SeqDataset(Dataset):
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
        rel_phys = to_relative_np(fut_phys, ref_idx=0)
        phys_norm = (rel_phys - PHYSICS_MEAN_REL) / PHYSICS_STD_REL
        return (torch.tensor(stack0, dtype=torch.float32),
                torch.tensor(fut_imgs, dtype=torch.float32).unsqueeze(1),
                torch.tensor(phys_norm, dtype=torch.float32),
                torch.tensor(fut_acts, dtype=torch.float32))


def stage2_dyn(variant, save_dir):
    print("=" * 60); print(f"Stage 2 [{variant}]: Bicycle Dynamics"); print("=" * 60)
    ds = SeqDataset(DATA_DIR)
    n_val = int(len(ds) * 0.1)
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=128, shuffle=False, drop_last=True)

    dyn = make_model(variant).to(DEVICE)
    opt = torch.optim.Adam(dyn.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')

    for epoch in range(DYN_EPOCHS):
        dyn.train()
        tl, nb = 0, 0
        for stack0, fut_imgs, fut_phys, fut_acts in tqdm(tr_loader, desc=f"{variant} Dyn {epoch+1}/{DYN_EPOCHS}"):
            fut_phys = fut_phys.to(DEVICE); fut_acts = fut_acts.to(DEVICE)
            z = fut_phys[:, 0]
            ls, lres = 0, 0
            for k in range(ROLLOUT_K):
                z_next, res = dyn(z, fut_acts[:, k])
                ls = ls + ((z_next - fut_phys[:, k + 1]) ** 2).mean()
                lres = lres + (res ** 2).mean()
                z = z_next
            ls /= ROLLOUT_K; lres /= ROLLOUT_K
            loss = ls + LAMBDA_RESIDUAL * lres
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(dyn.parameters(), 5.0)
            opt.step()
            tl += ls.item(); nb += 1
        tl /= nb

        dyn.eval(); vl, vn = 0, 0
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

        extras = ""
        if hasattr(dyn, 'L'): extras += f"  L={float(dyn.L):.2f}"
        if hasattr(dyn, 'm'):
            extras += f"  m={float(dyn.m):.0f}  Iz={float(dyn.Iz):.0f}"
            extras += f"  Cf={float(dyn.Cf):.0f}  Cr={float(dyn.Cr):.0f}"
            extras += f"  Lf={float(dyn.Lf):.2f}  Lr={float(dyn.Lr):.2f}"
        print(f"  Train {tl:.6f}  Val {vl:.6f}{extras}")

        if vl < best:
            best = vl
            save_checkpoint({'dynamics': dyn.state_dict(), 'val_loss': vl,
                             'variant': variant},
                            os.path.join(save_dir, 'dyn.tar'))
    print(f"[{variant}] Dyn best: {best:.6f}")


def stage3_e2e(variant, save_dir):
    print("\n" + "=" * 60); print(f"Stage 3 [{variant}]: E2E fine-tune"); print("=" * 60)
    ds = SeqDataset(DATA_DIR)
    n_val = int(len(ds) * 0.1)
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=32, shuffle=False, drop_last=True)

    enc = PhysicsEncoder().to(DEVICE)
    dec = PhysicsDecoder().to(DEVICE)
    dyn = make_model(variant).to(DEVICE)
    ae_ck = load_checkpoint("checkpoints/piwm_v2_rel/ae.tar")
    enc.load_state_dict(ae_ck['encoder']); dec.load_state_dict(ae_ck['decoder'])
    dy_ck = load_checkpoint(os.path.join(save_dir, 'dyn.tar'))
    dyn.load_state_dict(dy_ck['dynamics'])

    params = list(enc.parameters()) + list(dec.parameters()) + list(dyn.parameters())
    opt = torch.optim.Adam(params, lr=3e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')
    LAMBDA_IMG = 10.0; LAMBDA_PHYS = 1.0

    for epoch in range(E2E_EPOCHS):
        enc.train(); dec.train(); dyn.train()
        ti, tp, nb = 0, 0, 0
        for stack0, fut_imgs, fut_phys, fut_acts in tqdm(tr_loader, desc=f"{variant} E2E {epoch+1}/{E2E_EPOCHS}"):
            stack0, fut_imgs, fut_phys, fut_acts = stack0.to(DEVICE), fut_imgs.to(DEVICE), fut_phys.to(DEVICE), fut_acts.to(DEVICE)
            B = stack0.size(0)
            z_enc = enc(stack0)
            z = torch.zeros(B, 11, device=DEVICE)
            z[:, 0:2] = fut_phys[:, 0, 0:2]
            z[:, 2:11] = z_enc

            img_l, phys_l = 0, 0
            r0 = dec(z[:, 2:11])
            img_l += ((r0 - fut_imgs[:, 0]) ** 2).mean()
            phys_l += ((z_enc - fut_phys[:, 0, 2:11]) ** 2).mean()
            for k in range(ROLLOUT_K):
                z, _ = dyn(z, fut_acts[:, k])
                rec = dec(z[:, 2:11])
                img_l += ((rec - fut_imgs[:, k + 1]) ** 2).mean()
                phys_l += ((z[:, 2:11] - fut_phys[:, k + 1, 2:11]) ** 2).mean()
            img_l /= (ROLLOUT_K + 1); phys_l /= (ROLLOUT_K + 1)
            loss = LAMBDA_IMG * img_l + LAMBDA_PHYS * phys_l
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
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

        extras = ""
        if hasattr(dyn, 'L'): extras += f"  L={float(dyn.L):.2f}"
        print(f"  Train img={ti:.5f} phys={tp:.5f}  Val img={vi:.5f} phys={vp:.5f}{extras}")

        if vt < best:
            best = vt
            save_checkpoint({
                'encoder': enc.state_dict(),
                'decoder': dec.state_dict(),
                'dynamics': dyn.state_dict(),
                'val_loss': vt, 'variant': variant,
            }, os.path.join(save_dir, 'best.tar'))
    print(f"[{variant}] E2E best: {best:.5f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["v1", "v2", "v3", "v4"])
    p.add_argument("--stage", default="both", choices=["dyn", "e2e", "both"])
    args = p.parse_args()

    save_dir = f"checkpoints/piwm_bike_{args.variant}/"
    os.makedirs(save_dir, exist_ok=True)

    if args.stage in ("dyn", "both"):
        stage2_dyn(args.variant, save_dir)
    if args.stage in ("e2e", "both"):
        stage3_e2e(args.variant, save_dir)
