"""Vid2Param baseline: RNN-VAE that estimates physical parameters from video.

Architecture (following arXiv 1907.06422):
  - RNN-VAE encoder: image sequence -> (mu, logvar) over initial state + physical params
  - Physics-based dynamics (kinematics + param-scaled force/torque)
  - Decoder: latent -> image

Differs from GOKU:
  - Separate "theta" (physical params) explicitly predicted
  - Theta stays constant during rollout (system identification)
  - State evolves via theta-parameterized physics
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


SAVE_DIR = "checkpoints/vid2param/"
LATENT_DIM = 11
THETA_DIM = 8  # physical parameters (mass, friction, steering gain, etc. -- learned)
SEQ_LEN = 16
ENCODER_SEQ = 4


class V2PEncoder(nn.Module):
    """RNN-VAE encoder: image sequence -> initial state + theta parameters."""
    def __init__(self, latent_dim=LATENT_DIM, theta_dim=THETA_DIM, hidden=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
        )
        self.feat = nn.Linear(256 * 4 * 4, 128)
        self.rnn = nn.LSTM(128, hidden, num_layers=1, batch_first=True)
        # Two heads: initial state and theta
        self.fc_z0_mu = nn.Linear(hidden, latent_dim)
        self.fc_z0_lv = nn.Linear(hidden, latent_dim)
        self.fc_theta_mu = nn.Linear(hidden, theta_dim)
        self.fc_theta_lv = nn.Linear(hidden, theta_dim)

    def forward(self, seq_stack):
        B, T = seq_stack.shape[:2]
        flat = seq_stack.reshape(B * T, *seq_stack.shape[2:])
        h = self.conv(flat).reshape(B * T, -1)
        f = self.feat(h).reshape(B, T, -1)
        out, _ = self.rnn(f)
        last = out[:, -1, :]
        return (self.fc_z0_mu(last), self.fc_z0_lv(last),
                self.fc_theta_mu(last), self.fc_theta_lv(last))


class V2PDecoder(nn.Module):
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
        return self.deconv(self.fc(z).view(z.size(0), 256, 4, 4))


class V2PDynamics(nn.Module):
    """Physics-based dynamics parameterized by theta (constant during rollout)."""
    def __init__(self, latent_dim=LATENT_DIM, theta_dim=THETA_DIM, action_dim=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.force_net = nn.Sequential(
            nn.Linear(latent_dim + theta_dim + action_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3),  # dvx, dvy, domega
        )
        self.wheel_net = nn.Sequential(
            nn.Linear(latent_dim + theta_dim + action_dim, 64), nn.Tanh(),
            nn.Linear(64, 5),
        )

        std = PHYSICS_STD_REL; mean = PHYSICS_MEAN_REL
        self.register_buffer("dt_vx_x", torch.tensor(float(std[3] * DT / std[0])))
        self.register_buffer("dt_vy_y", torch.tensor(float(std[4] * DT / std[1])))
        self.register_buffer("dt_om_yaw", torch.tensor(float(std[5] * DT / std[2])))
        self.register_buffer("m_vx_x", torch.tensor(float(mean[3] * DT / std[0])))
        self.register_buffer("m_vy_y", torch.tensor(float(mean[4] * DT / std[1])))
        self.register_buffer("m_om_yaw", torch.tensor(float(mean[5] * DT / std[2])))

    def step(self, z, theta, a):
        inp = torch.cat([z, theta, a], dim=-1)
        x, y, yaw = z[:, 0], z[:, 1], z[:, 2]
        vx, vy, omega = z[:, 3], z[:, 4], z[:, 5]
        wheels, steer = z[:, 6:10], z[:, 10]

        x_n = x + vx * self.dt_vx_x + self.m_vx_x
        y_n = y + vy * self.dt_vy_y + self.m_vy_y
        yaw_n = yaw + omega * self.dt_om_yaw + self.m_om_yaw

        df = self.force_net(inp)
        vx_n = vx + df[:, 0]; vy_n = vy + df[:, 1]; om_n = omega + df[:, 2]
        dw = self.wheel_net(inp)
        w_n = wheels + dw[:, :4]; s_n = steer + dw[:, 4]

        return torch.stack([x_n, y_n, yaw_n, vx_n, vy_n, om_n,
                            w_n[:, 0], w_n[:, 1], w_n[:, 2], w_n[:, 3], s_n], dim=-1)


class Vid2Param(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = V2PEncoder()
        self.decoder = V2PDecoder()
        self.dynamics = V2PDynamics()

    def forward(self, seq_stack, action_seq, T_pred):
        mu_z, lv_z, mu_t, lv_t = self.encoder(seq_stack)
        std_z = (0.5 * lv_z).exp()
        std_t = (0.5 * lv_t).exp()
        z0 = mu_z + std_z * torch.randn_like(std_z)
        theta = mu_t + std_t * torch.randn_like(std_t)

        states = [z0]; z = z0
        for t in range(T_pred):
            z = self.dynamics.step(z, theta, action_seq[:, t])
            states.append(z)
        states = torch.stack(states, dim=1)
        recons = self.decoder(states.reshape(-1, LATENT_DIM)).reshape(
            states.size(0), states.size(1), 1, 64, 64)
        return recons, states, (mu_z, lv_z), (mu_t, lv_t)


class V2PDataset(Dataset):
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
                for t0 in range(FRAME_STACK - 1 + ENCODER_SEQ - 1, n - SEQ_LEN):
                    self.indices.append((ep_idx, t0))
            except: pass
        print(f"V2PDataset: {len(self.indices)} samples")

    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        ep, t0 = self.indices[i]
        imgs = self.imgs_list[ep]; phys = self.phys_list[ep]; acts = self.acts_list[ep]
        enc_seq = []
        for k in range(ENCODER_SEQ):
            center = t0 - ENCODER_SEQ + 1 + k
            stack = np.stack([imgs[center - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
            enc_seq.append(stack)
        enc_seq = np.stack(enc_seq, axis=0)
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
    print("=" * 60); print("Training Vid2Param"); print("=" * 60)
    ds = V2PDataset(DATA_DIR)
    n_val = int(len(ds) * 0.1)
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=32, shuffle=False, drop_last=True)

    model = Vid2Param().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')
    EPOCHS = 30

    for epoch in range(EPOCHS):
        model.train()
        tl, nb = 0, 0
        for enc_seq, fut_imgs, fut_phys, fut_acts in tqdm(tr_loader, desc=f"V2P {epoch+1}/{EPOCHS}"):
            enc_seq = enc_seq.to(DEVICE); fut_imgs = fut_imgs.to(DEVICE)
            fut_phys = fut_phys.to(DEVICE); fut_acts = fut_acts.to(DEVICE)
            recons, states, (mu_z, lv_z), (mu_t, lv_t) = model(enc_seq, fut_acts, SEQ_LEN)
            rec = ((recons - fut_imgs) ** 2).mean()
            ph = ((states - fut_phys) ** 2).mean()
            kl_z = -0.5 * (1 + lv_z - mu_z.pow(2) - lv_z.exp()).mean()
            kl_t = -0.5 * (1 + lv_t - mu_t.pow(2) - lv_t.exp()).mean()
            loss = 10 * rec + 1.0 * ph + 0.01 * (kl_z + kl_t)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        tl /= nb

        model.eval()
        vl, vi, vp, vn = 0, 0, 0, 0
        with torch.no_grad():
            for enc_seq, fut_imgs, fut_phys, fut_acts in val_loader:
                enc_seq = enc_seq.to(DEVICE); fut_imgs = fut_imgs.to(DEVICE)
                fut_phys = fut_phys.to(DEVICE); fut_acts = fut_acts.to(DEVICE)
                recons, states, (mu_z, lv_z), (mu_t, lv_t) = model(enc_seq, fut_acts, SEQ_LEN)
                rec = ((recons - fut_imgs) ** 2).mean()
                ph = ((states - fut_phys) ** 2).mean()
                vl += (10 * rec + ph).item()
                vi += rec.item(); vp += ph.item(); vn += 1
        vl /= vn; vi /= vn; vp /= vn
        sch.step(vl)
        print(f"  Train {tl:.5f}  Val {vl:.5f} (img={vi:.5f}, phys={vp:.5f})")

        if vl < best:
            best = vl
            save_checkpoint({'model': model.state_dict(), 'val_loss': vl},
                            os.path.join(SAVE_DIR, 'best.tar'))
    print(f"V2P best: {best:.5f}")


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    train()
