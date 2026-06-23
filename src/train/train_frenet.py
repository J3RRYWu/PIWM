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
        self.state, self.act = [], []
        self.idx = []
        for ep, f in enumerate(self.files):
            d = np.load(f)
            st = d["state"].astype(np.float32); ac = d["action"].astype(np.float32)
            self.state.append(st); self.act.append(ac)
            for t in range(0, len(st) - K - 1):
                self.idx.append((ep, t))
        print(f"FrenetSeqDataset K={K}: {len(self.idx)} windows from {len(self.files)} segs")

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        ep, t = self.idx[i]
        st = self.state[ep]; ac = self.act[ep]
        return (torch.tensor(st[t:t + self.K + 1]),
                torch.tensor(ac[t:t + self.K]))


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


def eval_xy(dyn, ds, val_eps, K=100, n=60):
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
            preds = [z[0].cpu().numpy()]
            for k in range(K):
                z = dyn(z, torch.tensor(ac[t + k:t + k + 1]).to(DEVICE))
                preds.append(z[0].cpu().numpy())
            preds = np.array(preds)
            gt = st[t:t + K + 1]
            xy_p = sd2xy(preds[:, 0], preds[:, 1])
            xy_g = sd2xy(gt[:, 0], gt[:, 1])
            er = np.linalg.norm(xy_p - xy_g, axis=-1)
            e50.append(er[50]); e100.append(er[100])
    return float(np.mean(e50)), float(np.median(e100)), float(np.mean(e100))


def run(K, epochs, save):
    ds = FrenetSeqDataset(K)
    tr, va, val_eps = split(ds)
    trl = DataLoader(tr, batch_size=128, shuffle=True, drop_last=True)
    dyn = FrenetDynamics(_os.path.join(META, "track.npz"),
                         _os.path.join(META, "stats.npz")).to(DEVICE)
    opt = torch.optim.Adam(dyn.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float("inf")
    names = ["s", "d", "psi", "v", "om"]
    for ep in range(epochs):
        dyn.train(); tot = 0; nb = 0
        for Z, A in tqdm(trl, desc=f"frenet K{K} {ep+1}/{epochs}"):
            Z = Z.to(DEVICE); A = A.to(DEVICE)
            z = Z[:, 0]; preds = [z]
            for k in range(K):
                z = dyn(z, A[:, k]); preds.append(z)
            preds = torch.stack(preds, 1)
            per = dyn.state_loss(preds.reshape(-1, 5), Z.reshape(-1, 5))
            loss = per.sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        e50, e100m, e100 = eval_xy(dyn, ds, val_eps)
        sch.step(e100)
        print(f"  train_loss={tot/nb:.4f}  xy@50={e50:.3f}m  xy@100 med={e100m:.3f} mean={e100:.3f}m")
        if e100 < best:
            best = e100
            os.makedirs(_os.path.dirname(save), exist_ok=True)
            torch.save({"dynamics": dyn.state_dict(), "xy100": e100}, save)
    print(f"BEST xy@100 mean = {best:.3f}m   (GOKU ref ~0.70m, V2P ~0.60m, analytic UB 0.33m)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--save", type=str, default="checkpoints/frenet/dyn.tar")
    a = p.parse_args()
    run(a.K, a.epochs, a.save)
