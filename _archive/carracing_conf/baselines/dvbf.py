"""DVBF baseline: Deep Variational Bayes Filter.

Learns a latent state-space model from raw observations. No physics prior.

Key ideas (simplified from paper, focusing on rollout-compatible form):
  - Latent state z_t (dim = latent_dim)
  - Locally-linear transition with stochastic perturbation w_t:
        z_t = A_t * z_{t-1} + B_t * a_{t-1} + C_t * w_t
    where A_t, B_t, C_t are mixed from a small set of matrices via weights produced
    by a recognition net.
  - Recognition: q(w_t | z_{t-1}, x_t, a_{t-1}) = Normal
  - Emission: p(x_t | z_t) via decoder

We use latent_dim = 11 to match PIWM.
"""

import os, sys, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE, DATA_DIR, FRAME_STACK
from utils import save_checkpoint, load_checkpoint


SAVE_DIR = "checkpoints/dvbf/"
LATENT_DIM = 11
W_DIM = 8
N_MATRICES = 4       # number of base matrices to mix
SEQ_LEN = 16
ENCODER_SEQ = 4


class DVBFEncoder(nn.Module):
    """Image -> feature (used at each step to condition transition)."""
    def __init__(self, feat_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
        )
        self.fc = nn.Linear(256 * 4 * 4, feat_dim)

    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))


class DVBFDecoder(nn.Module):
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


class DVBF(nn.Module):
    """Simplified DVBF.
    - q(w | z_{t-1}, phi_t, a_{t-1}) Normal
    - z_t = A(alpha) z_{t-1} + B(alpha) a_{t-1} + C(alpha) w_t
    - Initial z_0 from recognition q(z_0 | phi_0..phi_k)
    """
    def __init__(self, latent_dim=LATENT_DIM, w_dim=W_DIM, action_dim=3, n_mat=N_MATRICES):
        super().__init__()
        self.latent_dim = latent_dim; self.w_dim = w_dim; self.n_mat = n_mat
        self.encoder = DVBFEncoder(feat_dim=64)
        self.decoder = DVBFDecoder(latent_dim=latent_dim)

        # Base matrices (learnable)
        self.A_k = nn.Parameter(torch.randn(n_mat, latent_dim, latent_dim) * 0.05)
        self.B_k = nn.Parameter(torch.randn(n_mat, latent_dim, action_dim) * 0.05)
        self.C_k = nn.Parameter(torch.randn(n_mat, latent_dim, w_dim) * 0.05)
        # initialize A toward identity
        with torch.no_grad():
            for k in range(n_mat):
                self.A_k[k] += torch.eye(latent_dim)

        # Mixing weights network: (z_{t-1}, feat_t, a_{t-1}) -> alpha (n_mat,)
        self.alpha_net = nn.Sequential(
            nn.Linear(latent_dim + 64 + action_dim, 64), nn.ReLU(),
            nn.Linear(64, n_mat), nn.Softmax(dim=-1),
        )

        # Recognition of w_t: (z_{t-1}, feat_t, a_{t-1}) -> (mu, logvar) of w
        self.w_mu = nn.Sequential(
            nn.Linear(latent_dim + 64 + action_dim, 64), nn.ReLU(),
            nn.Linear(64, w_dim),
        )
        self.w_logvar = nn.Sequential(
            nn.Linear(latent_dim + 64 + action_dim, 64), nn.ReLU(),
            nn.Linear(64, w_dim),
        )

        # Initial z_0 recognition from first k features
        self.init_mu = nn.Linear(64 * ENCODER_SEQ, latent_dim)
        self.init_logvar = nn.Linear(64 * ENCODER_SEQ, latent_dim)

    def transition(self, z_prev, feat_t, a_prev, sample=True):
        """One transition step; returns (z_t, w_t, mu_w, logvar_w)."""
        inp = torch.cat([z_prev, feat_t, a_prev], dim=-1)
        alpha = self.alpha_net(inp)  # (B, n_mat)
        A = torch.einsum("bk,kij->bij", alpha, self.A_k)
        B = torch.einsum("bk,kij->bij", alpha, self.B_k)
        C = torch.einsum("bk,kij->bij", alpha, self.C_k)

        mu_w = self.w_mu(inp)
        logvar_w = self.w_logvar(inp)
        if sample:
            std = (0.5 * logvar_w).exp()
            w = mu_w + std * torch.randn_like(std)
        else:
            w = mu_w

        z_new = torch.einsum("bij,bj->bi", A, z_prev) + \
                torch.einsum("bij,bj->bi", B, a_prev) + \
                torch.einsum("bij,bj->bi", C, w)
        return z_new, w, mu_w, logvar_w

    def infer_z0(self, enc_seq):
        """enc_seq: (B, T_enc, FRAME_STACK, 64, 64). Uses first T_enc features to seed z_0."""
        B, T = enc_seq.shape[:2]
        flat = enc_seq.reshape(B * T, *enc_seq.shape[2:])
        feats = self.encoder(flat).reshape(B, T * 64)
        mu = self.init_mu(feats); logvar = self.init_logvar(feats)
        std = (0.5 * logvar).exp()
        z0 = mu + std * torch.randn_like(std)
        return z0, mu, logvar

    def forward(self, enc_seq, fut_stacks, action_seq):
        """Full forward: infer z_0, then roll forward using fut_stacks features.

        enc_seq: (B, T_enc, 3, 64, 64) -- used for z_0
        fut_stacks: (B, T+1, 3, 64, 64) -- used for recognition of w_t at each step
        action_seq: (B, T, 3)
        """
        B, Tp1 = fut_stacks.shape[:2]
        T = Tp1 - 1

        # Features at each future step
        flat = fut_stacks.reshape(B * Tp1, *fut_stacks.shape[2:])
        feats = self.encoder(flat).reshape(B, Tp1, 64)

        z0, mu0, logvar0 = self.infer_z0(enc_seq)
        zs = [z0]; w_mus = []; w_logvars = []
        z = z0
        for t in range(T):
            z_new, w, mu_w, logvar_w = self.transition(z, feats[:, t + 1], action_seq[:, t], sample=True)
            zs.append(z_new)
            w_mus.append(mu_w); w_logvars.append(logvar_w)
            z = z_new
        zs = torch.stack(zs, dim=1)  # (B, T+1, latent)

        # Decode
        recons = self.decoder(zs.reshape(-1, self.latent_dim)).reshape(B, Tp1, 1, 64, 64)
        return recons, zs, mu0, logvar0, torch.stack(w_mus, 1), torch.stack(w_logvars, 1)

    def rollout(self, z0, action_seq, T):
        """Deterministic rollout (no observations after z_0) using mean w = 0."""
        zs = [z0]; z = z0
        # During rollout without observations, we can't use feat_t. Substitute with zeros.
        dummy_feat = torch.zeros(z0.size(0), 64, device=z0.device)
        for t in range(T):
            z, _, _, _ = self.transition(z, dummy_feat, action_seq[:, t], sample=False)
            zs.append(z)
        return torch.stack(zs, dim=1)


class DVBFDataset(Dataset):
    """Provides (enc_seq, fut_stacks, fut_imgs, actions)."""
    def __init__(self, data_dir):
        files = sorted(glob.glob(f"{data_dir}/*.npz"))
        self.imgs_list, self.acts_list = [], []
        self.indices = []
        for ep_idx, f in enumerate(files):
            try:
                d = np.load(f, allow_pickle=True)
                imgs = d["imgs"].astype(np.float32)
                if imgs.max() > 1.0: imgs /= 255.0
                acts = d["action"].astype(np.float32)
                n = len(imgs)
                if n < ENCODER_SEQ + SEQ_LEN + FRAME_STACK: continue
                self.imgs_list.append(imgs); self.acts_list.append(acts)
                for t0 in range(FRAME_STACK - 1 + ENCODER_SEQ - 1, n - SEQ_LEN - FRAME_STACK):
                    self.indices.append((ep_idx, t0))
            except: pass
        print(f"DVBFDataset: {len(self.indices)} samples")

    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        ep, t0 = self.indices[i]
        imgs = self.imgs_list[ep]; acts = self.acts_list[ep]
        enc_seq = []
        for k in range(ENCODER_SEQ):
            center = t0 - ENCODER_SEQ + 1 + k
            stack = np.stack([imgs[center - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
            enc_seq.append(stack)
        enc_seq = np.stack(enc_seq, axis=0)
        # Future stacks and target images
        fut_stacks = []
        for k in range(SEQ_LEN + 1):
            center = t0 + k
            stack = np.stack([imgs[center - FRAME_STACK + 1 + j] for j in range(FRAME_STACK)], axis=0)
            fut_stacks.append(stack)
        fut_stacks = np.stack(fut_stacks, axis=0)
        fut_imgs = imgs[t0:t0 + SEQ_LEN + 1]
        fut_acts = acts[t0:t0 + SEQ_LEN]
        return (torch.tensor(enc_seq, dtype=torch.float32),
                torch.tensor(fut_stacks, dtype=torch.float32),
                torch.tensor(fut_imgs, dtype=torch.float32).unsqueeze(1),
                torch.tensor(fut_acts, dtype=torch.float32))


def train():
    print("=" * 60); print("Training DVBF"); print("=" * 60)
    ds = DVBFDataset(DATA_DIR)
    n_val = int(len(ds) * 0.1)
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=16, shuffle=False, drop_last=True)

    model = DVBF().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')
    EPOCHS = 25

    for epoch in range(EPOCHS):
        model.train()
        tl, nb = 0, 0
        for enc_seq, fut_stacks, fut_imgs, fut_acts in tqdm(tr_loader, desc=f"DVBF {epoch+1}/{EPOCHS}"):
            enc_seq = enc_seq.to(DEVICE); fut_stacks = fut_stacks.to(DEVICE)
            fut_imgs = fut_imgs.to(DEVICE); fut_acts = fut_acts.to(DEVICE)

            recons, zs, mu0, logvar0, w_mus, w_logvars = model(enc_seq, fut_stacks, fut_acts)
            rec = ((recons - fut_imgs) ** 2).mean()
            kl0 = -0.5 * (1 + logvar0 - mu0.pow(2) - logvar0.exp()).mean()
            klw = -0.5 * (1 + w_logvars - w_mus.pow(2) - w_logvars.exp()).mean()
            loss = 10 * rec + 0.01 * (kl0 + klw)

            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        tl /= nb

        model.eval()
        vl, vi, vn = 0, 0, 0
        with torch.no_grad():
            for enc_seq, fut_stacks, fut_imgs, fut_acts in val_loader:
                enc_seq = enc_seq.to(DEVICE); fut_stacks = fut_stacks.to(DEVICE)
                fut_imgs = fut_imgs.to(DEVICE); fut_acts = fut_acts.to(DEVICE)
                recons, zs, mu0, logvar0, w_mus, w_logvars = model(enc_seq, fut_stacks, fut_acts)
                rec = ((recons - fut_imgs) ** 2).mean()
                vl += rec.item(); vi += rec.item(); vn += 1
        vl /= vn; vi /= vn
        sch.step(vl)
        print(f"  Train {tl:.5f}  Val rec={vi:.5f}")

        if vl < best:
            best = vl
            save_checkpoint({'model': model.state_dict(), 'val_loss': vl},
                            os.path.join(SAVE_DIR, 'best.tar'))
    print(f"DVBF best: {best:.5f}")


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    train()
