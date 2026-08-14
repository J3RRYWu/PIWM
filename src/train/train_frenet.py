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
from folds import fold_split, describe as describe_split

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


def set_train_seed(seed):
    """Seed weight init and batch order ONLY.

    The train/val split is deliberately NOT reseeded: it stays at seed=0 in
    `split` below, so repeats of an experiment are paired on the same validation
    episodes and their spread measures training noise rather than a change of
    data. Without this, a fresh run of an identical configuration moved the
    100-step error from 0.246 m to 0.34-0.40 m with no way to attribute it.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def split(ds, val_frac=0.10, seed=0, fold=-1, nfolds=5):
    """fold=-1 is the legacy single hold-out; see src/folds.py."""
    _, val_eps = fold_split(len(ds.files), fold, nfolds, val_frac, seed)
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


def delta_supervision_noise(Z, half):
    """Conference weak-supervision noise (Sec. 4.1): per-dim biased uniform.
    mu_tilde = x + Δ, Δ~Unif[-half,half]; label = mean of 50 samples ~Unif[mu_tilde±half].
    `half` = 0.5*δ*|X_i| per dim. Returns the noisy proxy labels (model never sees clean x)."""
    b, t, d = Z.shape
    bias = (torch.rand(b, t, d, device=Z.device) * 2 - 1) * half               # (b,t,d) * (d,)
    resid = ((torch.rand(b, t, d, 50, device=Z.device) * 2 - 1) * half[:, None]).mean(-1)
    return Z + bias + resid


#: matched-magnitude Gaussian: same per-dim std as delta_supervision_noise, whose
#: variance is Var(bias) + Var(resid) = h^2/3 + h^2/150 = h^2 * 51/150.
_GAUSS_SCALE = float(np.sqrt(51.0 / 150.0))


def gaussian_supervision_noise(Z, half):
    """Control for the delta noise: SAME per-dim magnitude, plain zero-mean Gaussian.

    The 5-fold run found label noise makes this model monotonically BETTER while it
    makes every baseline worse. That is only a statement about weak supervision if
    the conference's biased-uniform structure is what does it; if a matched Gaussian
    helps just as much, the effect is ordinary regularisation and must be described
    as such. This exists to tell those two apart."""
    return Z + torch.randn_like(Z) * (half * _GAUSS_SCALE)


def run(K, epochs, save, kappa_mode="map", init=None, delta_sup=0.0, seed=0,
        fold=-1, nfolds=5, noise_kind="delta"):
    set_train_seed(seed)
    ds = FrenetSeqDataset(K)
    tr, va, val_eps = split(ds, fold=fold, nfolds=nfolds)
    print(f"  split -> {describe_split(len(ds.files), fold, nfolds)}")
    trl = DataLoader(tr, batch_size=128, shuffle=True, drop_last=True)
    dyn = FrenetDynamics(_os.path.join(META, "track.npz"),
                         _os.path.join(META, "stats.npz")).to(DEVICE)
    if init:
        from utils import load_checkpoint
        dyn.load_state_dict(load_checkpoint(init)["dynamics"])
        print(f"warm-started dynamics from {init}")
    # Per-dim valid range |X_i| for the delta weak-supervision noise, taken over the
    # TRAINING episodes only. Using every episode would let the held-out ones set the
    # noise scale -- negligible in size but still information crossing the split, and
    # train_baselines_donkey.py already computes its own range from train windows,
    # so this also makes "the same delta" mean the same thing in both scripts.
    _train_eps = [e for e in range(len(ds.state)) if e not in val_eps]
    _allst = np.concatenate([ds.state[e] for e in _train_eps], 0)
    _rng = torch.tensor(_allst.max(0) - _allst.min(0), dtype=torch.float32, device=DEVICE)
    half_sup = 0.5 * delta_sup * _rng
    noise_fn = {"delta": delta_supervision_noise,
                "gauss": gaussian_supervision_noise}[noise_kind]
    print(f"kappa_mode={kappa_mode}  delta_sup={delta_sup}  noise={noise_kind}  "
          f"(|X| over {len(_train_eps)} train eps"
          f"={np.round(_allst.max(0) - _allst.min(0), 3).tolist()}, "
          f"half-width={np.round(half_sup.cpu().numpy(), 3).tolist()})")
    opt = torch.optim.Adam(dyn.parameters(), lr=1e-3 if not init else 3e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float("inf")
    names = ["s", "d", "psi", "v", "om"]
    for ep in range(epochs):
        dyn.train(); tot = 0; nb = 0
        for Z, A, P in tqdm(trl, desc=f"frenet K{K} {ep+1}/{epochs}"):
            Z = Z.to(DEVICE); A = A.to(DEVICE); P = P.to(DEVICE)
            # weak supervision: model only ever sees the δ-noised proxy labels (init + targets)
            Zsup = noise_fn(Z, half_sup) if delta_sup > 0 else Z
            s0 = Zsup[:, 0, 0].clone()
            z = Zsup[:, 0]; preds = [z]
            for k in range(K):
                kov = None
                if kappa_mode == "profile":
                    delta = torch.remainder(z[:, 0] - s0, dyn.total_len)
                    kov = kappa_from_profile(P, delta)
                z = dyn(z, A[:, k], kappa_override=kov); preds.append(z)
            preds = torch.stack(preds, 1)
            per = dyn.state_loss(preds.reshape(-1, 5), Zsup.reshape(-1, 5))
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
    p.add_argument("--seed", type=int, default=0,
                   help="seeds weight init + batch order; train/val split stays fixed")
    p.add_argument("--delta", type=float, default=0.0,
                   help="conference δ weak-supervision noise on state labels (e.g. 0.05, 0.10)")
    p.add_argument("--fold", type=int, default=-1,
                   help="-1 = legacy single hold-out (default, reproduces every "
                        "existing checkpoint); 0..nfolds-1 selects a CV fold")
    p.add_argument("--nfolds", type=int, default=5)
    p.add_argument("--noise_kind", choices=["delta", "gauss"], default="delta",
                   help="delta = the conference's biased-uniform weak supervision; "
                        "gauss = matched-magnitude Gaussian control (see the note on "
                        "gaussian_supervision_noise)")
    a = p.parse_args()
    run(a.K, a.epochs, a.save, a.kappa_mode, a.init, a.delta, a.seed,
        a.fold, a.nfolds, a.noise_kind)
