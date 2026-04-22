"""GOKU-net baseline: Generative ODE Modeling with Known Unknowns.

Architecture:
  - Encoder (RNN over image sequence) -> initial latent + ODE params (mu, logvar, VAE style)
  - ODE block (known kinematics + learned force/wheel) integrated forward
  - Decoder (latent -> image)

Known ODE (kinematics): x' = vx, y' = vy, yaw' = omega
Unknowns (learned MLP): force -> dvx, dvy, domega; wheels & steer delta
"""

import os, sys, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DEVICE, DATA_DIR, PHYSICS_MEAN_REL, PHYSICS_STD_REL, DT,
                    FRAME_STACK)
from relative_coords import to_relative_np
from utils import save_checkpoint, load_checkpoint


SAVE_DIR = "checkpoints/goku/"
SEQ_LEN = 16
ENCODER_SEQ = 4   # first 4 frame-stacks used for inference
LATENT_DIM = 11


class GokuEncoder(nn.Module):
    """RNN-VAE encoder: image sequence -> (mu, logvar) over initial latent state."""

    def __init__(self, latent_dim=LATENT_DIM, hidden=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
        )
        self.feat = nn.Linear(256 * 4 * 4, 128)
        self.rnn = nn.GRU(128, hidden, batch_first=True)
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

    def forward(self, seq_stack):
        """seq_stack: (B, T_enc, FRAME_STACK, 64, 64)"""
        B, T = seq_stack.shape[:2]
        flat = seq_stack.reshape(B * T, *seq_stack.shape[2:])
        h = self.conv(flat).reshape(B * T, -1)
        feats = self.feat(h).reshape(B, T, -1)
        out, _ = self.rnn(feats)
        last = out[:, -1, :]
        return self.fc_mu(last), self.fc_logvar(last)


class GokuDecoder(nn.Module):
    """Decodes latent state -> 64x64 image."""
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 256 * 4 * 4), nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Sigmoid(),
        )
    def forward(self, z):
        h = self.fc(z).view(z.size(0), 256, 4, 4)
        return self.deconv(h)


class GokuODE(nn.Module):
    """Known ODE for kinematics + learned forces.

    Operates in NORMALIZED space.
    State: [x, y, yaw, vx, vy, omega, w0..w3, steer] (11)
    Action: 3
    """
    def __init__(self):
        super().__init__()
        inp = LATENT_DIM + 3
        self.force = nn.Sequential(
            nn.Linear(inp, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3),  # dvx, dvy, domega
        )
        self.wheel = nn.Sequential(
            nn.Linear(inp, 64), nn.Tanh(),
            nn.Linear(64, 5),  # dw0..3, dsteer
        )

        std = PHYSICS_STD_REL; mean = PHYSICS_MEAN_REL
        self.register_buffer("dt_vx_x", torch.tensor(float(std[3] * DT / std[0])))
        self.register_buffer("dt_vy_y", torch.tensor(float(std[4] * DT / std[1])))
        self.register_buffer("dt_om_yaw", torch.tensor(float(std[5] * DT / std[2])))
        self.register_buffer("m_vx_x", torch.tensor(float(mean[3] * DT / std[0])))
        self.register_buffer("m_vy_y", torch.tensor(float(mean[4] * DT / std[1])))
        self.register_buffer("m_om_yaw", torch.tensor(float(mean[5] * DT / std[2])))

    def step(self, z, a):
        """Euler step: z_{t+1} = z_t + dt * f(z_t, a_t)  (implemented discrete)."""
        inp = torch.cat([z, a], dim=-1)
        x, y, yaw = z[:, 0], z[:, 1], z[:, 2]
        vx, vy, omega = z[:, 3], z[:, 4], z[:, 5]
        wheels, steer = z[:, 6:10], z[:, 10]

        # Known kinematics
        x_n = x + vx * self.dt_vx_x + self.m_vx_x
        y_n = y + vy * self.dt_vy_y + self.m_vy_y
        yaw_n = yaw + omega * self.dt_om_yaw + self.m_om_yaw

        # Learned forces
        df = self.force(inp)
        vx_n = vx + df[:, 0]; vy_n = vy + df[:, 1]; om_n = omega + df[:, 2]

        dw = self.wheel(inp)
        w_n = wheels + dw[:, :4]
        s_n = steer + dw[:, 4]

        return torch.stack([x_n, y_n, yaw_n, vx_n, vy_n, om_n,
                            w_n[:, 0], w_n[:, 1], w_n[:, 2], w_n[:, 3], s_n], dim=-1)


class GOKU(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GokuEncoder()
        self.decoder = GokuDecoder()
        self.ode = GokuODE()

    def forward(self, seq_stack, action_seq, T_pred):
        """
        seq_stack: (B, T_enc, 3, 64, 64) image stacks for encoding
        action_seq: (B, T_pred, 3) action sequence for rollout
        Returns: reconstructed images (B, T_pred+1, 1, 64, 64), latent states (B, T_pred+1, 11), mu, logvar
        """
        mu, logvar = self.encoder(seq_stack)  # (B, 11)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z0 = mu + eps * std

        states = [z0]
        z = z0
        for t in range(T_pred):
            z = self.ode.step(z, action_seq[:, t])
            states.append(z)
        states = torch.stack(states, dim=1)  # (B, T+1, 11)

        recons = self.decoder(states.reshape(-1, LATENT_DIM)).reshape(
            states.size(0), states.size(1), 1, 64, 64)
        return recons, states, mu, logvar


class GokuDataset(Dataset):
    """Image stacks for encoder + future images + relative physics + actions."""
    def __init__(self, data_dir):
        files = sorted(glob.glob(f"{data_dir}/*.npz"))
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
                if n < ENCODER_SEQ + SEQ_LEN + FRAME_STACK: continue
                self.imgs_list.append(imgs); self.phys_list.append(phys); self.acts_list.append(acts)
                # Valid t0: need ENCODER_SEQ frame-stacks before t0, and SEQ_LEN frames after
                for t0 in range(FRAME_STACK - 1 + ENCODER_SEQ - 1, n - SEQ_LEN):
                    self.indices.append((ep_idx, t0))
            except: pass
        print(f"GokuDataset: {len(self.indices)} samples")

    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        ep, t0 = self.indices[i]
        imgs = self.imgs_list[ep]; phys = self.phys_list[ep]; acts = self.acts_list[ep]
        # ENCODER input: ENCODER_SEQ frame-stacks ending at t0
        enc_seq = []
        for k in range(ENCODER_SEQ):
            center = t0 - ENCODER_SEQ + 1 + k
            stack = np.stack([imgs[center - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
            enc_seq.append(stack)
        enc_seq = np.stack(enc_seq, axis=0)  # (T_enc, 3, 64, 64)
        # Future images (SEQ_LEN+1 from t0 onwards)
        fut_imgs = imgs[t0:t0 + SEQ_LEN + 1]
        fut_acts = acts[t0:t0 + SEQ_LEN]
        fut_phys = phys[t0:t0 + SEQ_LEN + 1]
        rel_phys = to_relative_np(fut_phys, ref_idx=0)
        phys_norm = (rel_phys - PHYSICS_MEAN_REL) / PHYSICS_STD_REL
        return (torch.tensor(enc_seq, dtype=torch.float32),
                torch.tensor(fut_imgs, dtype=torch.float32).unsqueeze(1),
                torch.tensor(phys_norm, dtype=torch.float32),
                torch.tensor(fut_acts, dtype=torch.float32))


def train():
    print("=" * 60)
    print("Training GOKU-net")
    print("=" * 60)
    ds = GokuDataset(DATA_DIR)
    n_val = int(len(ds) * 0.1)
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=32, shuffle=False, drop_last=True)

    model = GOKU().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')
    EPOCHS = 30

    for epoch in range(EPOCHS):
        model.train()
        tl, nb = 0, 0
        for enc_seq, fut_imgs, fut_phys, fut_acts in tqdm(tr_loader, desc=f"GOKU {epoch+1}/{EPOCHS}"):
            enc_seq = enc_seq.to(DEVICE); fut_imgs = fut_imgs.to(DEVICE)
            fut_phys = fut_phys.to(DEVICE); fut_acts = fut_acts.to(DEVICE)

            recons, states, mu, logvar = model(enc_seq, fut_acts, SEQ_LEN)
            rec_loss = ((recons - fut_imgs) ** 2).mean()
            phys_loss = ((states - fut_phys) ** 2).mean()
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            loss = 10 * rec_loss + 1.0 * phys_loss + 0.01 * kl

            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        tl /= nb

        model.eval()
        vl, vi, vp, vn = 0, 0, 0, 0
        with torch.no_grad():
            for enc_seq, fut_imgs, fut_phys, fut_acts in val_loader:
                enc_seq = enc_seq.to(DEVICE); fut_imgs = fut_imgs.to(DEVICE)
                fut_phys = fut_phys.to(DEVICE); fut_acts = fut_acts.to(DEVICE)
                recons, states, mu, logvar = model(enc_seq, fut_acts, SEQ_LEN)
                rec = ((recons - fut_imgs) ** 2).mean()
                ph = ((states - fut_phys) ** 2).mean()
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
                vl += (10 * rec + ph + 0.01 * kl).item()
                vi += rec.item(); vp += ph.item(); vn += 1
        vl /= vn; vi /= vn; vp /= vn
        sch.step(vl)
        print(f"  Train {tl:.5f}  Val {vl:.5f} (img={vi:.5f}, phys={vp:.5f})")

        if vl < best:
            best = vl
            save_checkpoint({'model': model.state_dict(), 'val_loss': vl},
                            os.path.join(SAVE_DIR, 'best.tar'))
    print(f"GOKU best: {best:.5f}")


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    train()
