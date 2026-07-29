"""Train the road-context perception head: front-view 15-frame stack -> upcoming
curvature profile kappa(s + [0,0.5,..,4.5] m). This is the perception that replaces
the privileged known-map kappa lookup, making the Frenet model a PURE world model.

Two regimes, trained the SAME way on the SAME (image, kappa_profile) targets so the
only difference is whether the v6 encoder backbone adapts to the curvature task:
    python src/train/train_kappa_perception.py --mode frozen    --save checkpoints/frenet/kappa_frozen.tar
    python src/train/train_kappa_perception.py --mode finetune  --save checkpoints/frenet/kappa_finetune.tar
(GPU: use the py311 interpreter.) Reports per-offset val RMSE (real 1/m units) =
"how far ahead the camera can actually read curvature".
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # src/

import os, glob, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from models.road_perception import RoadContextEncoder
from models.road_perception_vqformer import RoadContextVQFormer
from lane_utils import LANE_FRAME_STACK as FS
from utils import load_checkpoint

DATA = _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "Data_Donkeycar_frenet")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KappaFrameDataset(Dataset):
    """(15-frame stack ending at t, kappa_profile[t]) over all clean frenet segments."""
    def __init__(self):
        self.files = sorted(f for f in glob.glob(_os.path.join(DATA, "*.npz"))
                            if "_meta" not in f)
        self.imgs, self.kap, self.idx = [], [], []
        for ep, f in enumerate(self.files):
            d = np.load(f)
            im = d["imgs"].astype(np.float32)            # (T,64,64) in [0,1]
            kp = d["kappa_profile"].astype(np.float32)   # (T,10)
            self.imgs.append(im); self.kap.append(kp)
            for t in range(FS - 1, len(im)):
                self.idx.append((ep, t))
        self.n_off = self.kap[0].shape[1]
        print(f"KappaFrameDataset: {len(self.idx)} frames from {len(self.files)} segs, "
              f"n_offsets={self.n_off}")

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        ep, t = self.idx[i]
        stack = self.imgs[ep][t - FS + 1:t + 1]          # (FS,64,64)
        return torch.from_numpy(stack), torch.from_numpy(self.kap[ep][t])


def split(ds, val_frac=0.10, seed=0):
    """Episode-level split identical to train_frenet (same val segments)."""
    n_eps = len(ds.files)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_eps)
    nval = max(1, int(round(n_eps * val_frac)))
    val_eps = set(perm[:nval].tolist())
    tr = [i for i, (e, _) in enumerate(ds.idx) if e not in val_eps]
    va = [i for i, (e, _) in enumerate(ds.idx) if e in val_eps]
    return Subset(ds, tr), Subset(ds, va), val_eps


@torch.no_grad()
def evaluate(model, loader, offsets):
    model.eval()
    se, n = np.zeros(len(offsets)), 0
    for x, y in loader:
        p = model(x.to(DEVICE)).cpu().numpy()
        se += ((p - y.numpy()) ** 2).sum(0); n += len(y)
    rmse = np.sqrt(se / n)
    return rmse


def run(mode, save, epochs, backbone_ckpt, arch="cnn"):
    ds = KappaFrameDataset()
    offsets = np.load(_os.path.join(DATA, "_meta", "stats.npz"))["kappa_offsets"]
    tr, va, val_eps = split(ds)
    print(f"train={len(tr)} val={len(va)} (val segs={sorted(val_eps)})")
    trl = DataLoader(tr, batch_size=128, shuffle=True, drop_last=True, num_workers=0)
    val = DataLoader(va, batch_size=256, shuffle=False, num_workers=0)

    if arch == "vqformer":
        # per-frame CNN -> VQ(512) -> Transformer over the window. Always trained
        # from scratch: there is no pretrained VQ/Transformer backbone in this repo,
        # so --mode / --backbone do not apply and are ignored on purpose.
        model = RoadContextVQFormer(n_offsets=ds.n_off, frame_stack=FS).to(DEVICE)
        print("arch=vqformer  (per-frame CNN + VQ-512 + 2-layer Transformer, from scratch)")
        if mode == "frozen" or backbone_ckpt.lower() != "scratch":
            print("  note: --mode/--backbone ignored for this arch (nothing to warm-start from)")
    else:
        model = RoadContextEncoder(n_offsets=ds.n_off,
                                   freeze_backbone=(mode == "frozen")).to(DEVICE)
        if backbone_ckpt.lower() == "scratch":
            print("backbone: RANDOM init (no v6 warm-start) -- native-from-scratch test")
        else:
            # warm-start the backbone from the trained v6 encoder
            model.load_backbone(load_checkpoint(backbone_ckpt)["encoder"])
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"arch={arch} mode={mode}  trainable params={n_train}  "
          f"(head only={sum(p.numel() for p in model.kappa_head.parameters())})")

    # same optimizer / schedule / budget for both archs so the comparison is about
    # architecture rather than tuning
    lr = 1e-3 if (mode == "frozen" and arch == "cnn") else 3e-4
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    lossfn = nn.MSELoss()
    best = float("inf")
    for ep in range(epochs):
        model.train(); tot = nb = 0
        perp_sum = 0.0
        for x, y in tqdm(trl, desc=f"kappa-{arch}-{mode} {ep+1}/{epochs}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = lossfn(pred, y)
            if arch == "vqformer":
                loss = loss + model.last_vq_loss        # VQ codebook + commitment
                perp_sum += float(model.last_perplexity)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        rmse = evaluate(model, val, offsets)
        score = float(rmse.mean())
        sch.step(score)
        extra = f"  codebook_perplexity={perp_sum/max(nb,1):.1f}/{model.vq.n_codes}" \
                if arch == "vqformer" else ""
        print(f"  ep{ep+1} train_loss={tot/nb:.4f}  val_RMSE mean={score:.4f}  "
              f"per-offset={np.round(rmse, 3).tolist()}{extra}")
        if score < best:
            best = score
            os.makedirs(_os.path.dirname(save), exist_ok=True)
            torch.save({"model": model.state_dict(), "mode": mode, "arch": arch,
                        "val_rmse": rmse.tolist(), "offsets": offsets.tolist()}, save)
    print(f"\nBEST val RMSE(mean over offsets) = {best:.4f} 1/m  -> {save}")
    print(f"  offsets (m) = {np.round(offsets, 1).tolist()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["frozen", "finetune"], default="finetune")
    p.add_argument("--arch", choices=["cnn", "vqformer"], default="cnn",
                   help="cnn = shipped channel-stacked backbone; "
                        "vqformer = per-frame CNN + VQ-512 + Transformer")
    p.add_argument("--save", required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--backbone", default="checkpoints/piwm_lane_v6_donkey/ae.tar")
    a = p.parse_args()
    run(a.mode, a.save, a.epochs, a.backbone, arch=a.arch)
