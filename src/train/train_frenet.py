"""Train the Frenet dynamics and evaluate 100-step xy drift vs GOKU (both from
GT init, in real metres — the fair, interpretable comparison).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os, glob, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from models.frenet_dynamics import FrenetDynamics

DATA = _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "Data_Donkeycar_frenet")
META = _os.path.join(DATA, "_meta")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FrenetSeqDataset(Dataset):
    def __init__(self, K):
        self.K = K
        self.files = sorted([f for f in glob.glob(_os.path.join(DATA, "*.npz"))
                             if "_meta" not in f])
        self.state, self.act, self.prof = [], [], []
        self.idx = []
        for ep, f in enumerate(self.files):
            d = np.load(f)
            st = d["state"].astype(np.float32); ac = d["action"].astype(np.float32)
            kp = d["kappa_profile"].astype(np.float32)
            self.state.append(st); self.act.append(ac); self.prof.append(kp)
            for t in range(0, len(st) - K - 1):
                self.idx.append((ep, t))
        print(f"FrenetSeqDataset K={K}: {len(self.idx)} windows from {len(self.files)} segs")

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        ep, t = self.idx[i]
        st = self.state[ep]; ac = self.act[ep]
        # kappa_profile at the WINDOW START t0 = the t0 camera preview (for profile mode)
        return (torch.tensor(st[t:t + self.K + 1]),
                torch.tensor(ac[t:t + self.K]),
                torch.tensor(self.prof[ep][t]))


def split(ds, val_frac=0.10, seed=0):
    n_eps = len(ds.files)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_eps)
    nval = max(1, int(round(n_eps * val_frac)))
    val_eps = set(perm[:nval].tolist())
    tr = [i for i, (e, _) in enumerate(ds.idx) if e not in val_eps]
    va = [i for i, (e, _) in enumerate(ds.idx) if e in val_eps]
    return Subset(ds, tr), Subset(ds, va), val_eps


# track for xy conversion (eval)
_tr = np.load(_os.path.join(META, "track.npz"))
_cen, _nrm = _tr["centers"], _tr["normals"]
_L, _ds, _M = float(_tr["total_len"]), float(_tr["grid_ds"]), len(_tr["centers"])
def sd2xy(s, d):
    i = (np.mod(s, _L) / _ds).astype(int) % _M
    return _cen[i] + d[..., None] * _nrm[i]

OFFSETS = torch.tensor(np.load(_os.path.join(META, "stats.npz"))["kappa_offsets"],
                       dtype=torch.float32, device=DEVICE)


def kappa_from_profile(prof, delta):
    """Batched linear interp: prof (B,n_off) perceived at t0, delta (B,) arc-length
    advanced since t0 -> kappa-at-car (B,). Clamped (hold-last past the preview)."""
    delta = delta.clamp(OFFSETS[0], OFFSETS[-1])
    i = torch.searchsorted(OFFSETS, delta).clamp(1, len(OFFSETS) - 1)
    x0, x1 = OFFSETS[i - 1], OFFSETS[i]
    y0 = prof.gather(1, (i - 1).unsqueeze(1)).squeeze(1)
    y1 = prof.gather(1, i.unsqueeze(1)).squeeze(1)
    w = (delta - x0) / (x1 - x0 + 1e-9)
    return y0 + w * (y1 - y0)


def eval_xy(dyn, ds, val_eps, K=100, n=60, kappa_mode="map"):
    dyn.eval()
    rng = np.random.default_rng(1)
    starts = []
    for ep in val_eps:
        st = ds.state[ep]
        if len(st) > K + 1:
            for t in rng.choice(len(st) - K - 1, size=min(6, len(st) - K - 1), replace=False):
                starts.append((ep, int(t)))
    rng.shuffle(starts); starts = starts[:n]
    e50, e100 = [], []
    with torch.no_grad():
        for ep, t in starts:
            st = ds.state[ep]; ac = ds.act[ep]
            z = torch.tensor(st[t:t + 1]).to(DEVICE)
            s0 = z[0, 0].clone()
            prof = torch.tensor(ds.prof[ep][t]).unsqueeze(0).to(DEVICE)
            preds = [z[0].cpu().numpy()]
            for k in range(K):
                kov = None
                if kappa_mode == "profile":
                    delta = torch.remainder(z[:, 0] - s0, dyn.total_len)
                    kov = kappa_from_profile(prof, delta)
                z = dyn(z, torch.tensor(ac[t + k:t + k + 1]).to(DEVICE), kappa_override=kov)
                preds.append(z[0].cpu().numpy())
            preds = np.array(preds)
            gt = st[t:t + K + 1]
            xy_p = sd2xy(preds[:, 0], preds[:, 1])
            xy_g = sd2xy(gt[:, 0], gt[:, 1])
            er = np.linalg.norm(xy_p - xy_g, axis=-1)
            e50.append(er[50]); e100.append(er[100])
    return float(np.mean(e50)), float(np.median(e100)), float(np.mean(e100))


def run(K, epochs, save, kappa_mode="map", init=None):
    ds = FrenetSeqDataset(K)
    tr, va, val_eps = split(ds)
    trl = DataLoader(tr, batch_size=128, shuffle=True, drop_last=True)
    dyn = FrenetDynamics(_os.path.join(META, "track.npz"),
                         _os.path.join(META, "stats.npz")).to(DEVICE)
    if init:
        from utils import load_checkpoint
        dyn.load_state_dict(load_checkpoint(init)["dynamics"])
        print(f"warm-started dynamics from {init}")
    print(f"kappa_mode={kappa_mode}  (profile = perceived/preview regime, NOT map lookup)")
    opt = torch.optim.Adam(dyn.parameters(), lr=1e-3 if not init else 3e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float("inf")
    names = ["s", "d", "psi", "v", "om"]
    for ep in range(epochs):
        dyn.train(); tot = 0; nb = 0
        for Z, A, P in tqdm(trl, desc=f"frenet K{K} {ep+1}/{epochs}"):
            Z = Z.to(DEVICE); A = A.to(DEVICE); P = P.to(DEVICE)
            s0 = Z[:, 0, 0].clone()
            z = Z[:, 0]; preds = [z]
            for k in range(K):
                kov = None
                if kappa_mode == "profile":
                    delta = torch.remainder(z[:, 0] - s0, dyn.total_len)
                    kov = kappa_from_profile(P, delta)
                z = dyn(z, A[:, k], kappa_override=kov); preds.append(z)
            preds = torch.stack(preds, 1)
            per = dyn.state_loss(preds.reshape(-1, 5), Z.reshape(-1, 5))
            loss = per.sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        e50, e100m, e100 = eval_xy(dyn, ds, val_eps, kappa_mode=kappa_mode)
        sch.step(e100)
        print(f"  train_loss={tot/nb:.4f}  xy@50={e50:.3f}m  xy@100 med={e100m:.3f} mean={e100:.3f}m")
        if e100 < best:
            best = e100
            os.makedirs(_os.path.dirname(save), exist_ok=True)
            torch.save({"dynamics": dyn.state_dict(), "xy100": e100, "kappa_mode": kappa_mode}, save)
    print(f"BEST xy@100 mean = {best:.3f}m   (map-oracle ref 0.246m, baselines 0.44-0.47m)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--save", type=str, default="checkpoints/frenet/dyn.tar")
    p.add_argument("--kappa_mode", choices=["map", "profile"], default="map",
                   help="map=known-track lookup (oracle); profile=t0 preview + shift (pure WM)")
    p.add_argument("--init", type=str, default=None, help="warm-start dynamics checkpoint")
    a = p.parse_args()
    run(a.K, a.epochs, a.save, a.kappa_mode, a.init)
