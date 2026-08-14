"""Train the road-context perception head: front-view 15-frame stack -> upcoming
curvature profile kappa(s + [0,0.5,..,4.5] m). This is the perception that replaces
the privileged known-map kappa lookup, making the Frenet model a PURE world model.

Two regimes, trained the SAME way on the SAME (image, kappa_profile) targets so the
only difference is whether the v6 encoder backbone adapts to the curvature task:
    python src/train/train_kappa_perception.py --mode frozen    --save checkpoints/frenet/kappa_frozen.tar
    python src/train/train_kappa_perception.py --mode finetune  --save checkpoints/frenet/kappa_finetune.tar
(GPU: use the py311 interpreter.) Reports per-offset val RMSE (real 1/m units) =
"how far ahead the camera can actually read curvature".

--target shape additionally trains the SHAPE head: the same road as local
centreline geometry, which a map-free rollout can read a position off directly
instead of integrating curvature twice. See models/road_perception.py.
    python src/train/train_kappa_perception.py --target shape --mode finetune \
        --backbone scratch --save checkpoints/frenet/road_shape.tar
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
from models.road_perception_ae import (ExtrinsicVisionVAE, ExtrinsicRoadEncoder,
                                       IntrinsicRoadEncoder)
from baselines.road_seq_baselines import LSTMRoadEncoder, TransformerRoadEncoder
from frenet_track import local_road_points
from folds import fold_split, describe as describe_split
from lane_utils import LANE_FRAME_STACK as FS
from utils import load_checkpoint

DATA = _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "Data_Donkeycar_frenet")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KappaFrameDataset(Dataset):
    """(15-frame stack ending at t, kappa_profile[t]) over all clean frenet segments.

    with_shape also returns the SHAPE target: the local centreline sampled at the
    same arc-length offsets, in the road frame at the car, as (X - offset, Y).
    It is derived on the fly from track.npz + the stored s, so no dataset rebuild
    is needed -- and like the kappa target it is supervision only; the model sees
    images alone at test time.
    """
    def __init__(self, with_shape=False):
        self.files = sorted(f for f in glob.glob(_os.path.join(DATA, "*.npz"))
                            if "_meta" not in f)
        self.with_shape = with_shape
        offsets = np.load(_os.path.join(DATA, "_meta", "stats.npz"))["kappa_offsets"]
        if with_shape:
            tr = np.load(_os.path.join(DATA, "_meta", "track.npz"))
            cen, hd = tr["centers"], tr["heading"]
            L, gds = float(tr["total_len"]), float(tr["grid_ds"])
        self.imgs, self.kap, self.shp, self.idx = [], [], [], []
        for ep, f in enumerate(self.files):
            d = np.load(f)
            im = d["imgs"].astype(np.float32)            # (T,64,64) in [0,1]
            kp = d["kappa_profile"].astype(np.float32)   # (T,10)
            self.imgs.append(im); self.kap.append(kp)
            if with_shape:
                P = local_road_points(cen, hd, L, gds, d["state"][:, 0], offsets)  # (T,10,2)
                P[..., 0] -= offsets                     # predict the along-track residual
                self.shp.append(P.astype(np.float32))
            for t in range(FS - 1, len(im)):
                self.idx.append((ep, t))
        self.n_off = self.kap[0].shape[1]
        # per-target variance, so the two losses are combined on equal footing
        self.kap_var = float(np.concatenate(self.kap, 0).var())
        self.shp_var = float(np.concatenate(self.shp, 0).var()) if with_shape else 1.0
        print(f"KappaFrameDataset: {len(self.idx)} frames from {len(self.files)} segs, "
              f"n_offsets={self.n_off}, with_shape={with_shape}")

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        ep, t = self.idx[i]
        stack = self.imgs[ep][t - FS + 1:t + 1]          # (FS,64,64)
        out = [torch.from_numpy(stack), torch.from_numpy(self.kap[ep][t])]
        if self.with_shape:
            out.append(torch.from_numpy(self.shp[ep][t]))
        return tuple(out)


def set_train_seed(seed):
    """Seed weight init and batch order ONLY -- the split below stays at seed=0
    so repeats are paired on the same validation episodes."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def split(ds, val_frac=0.10, seed=0, fold=-1, nfolds=5):
    """Episode-level split identical to train_frenet (same val segments).

    Under k-fold this identity is load-bearing: the encoder must not have seen
    the dynamics' validation episodes, so both call the same `folds.fold_split`.
    """
    _, val_eps = fold_split(len(ds.files), fold, nfolds, val_frac, seed)
    tr = [i for i, (e, _) in enumerate(ds.idx) if e not in val_eps]
    va = [i for i, (e, _) in enumerate(ds.idx) if e in val_eps]
    return Subset(ds, tr), Subset(ds, va), val_eps


@torch.no_grad()
def evaluate(model, loader, offsets):
    model.eval()
    se, n = np.zeros(len(offsets)), 0
    for batch in loader:
        x, y = batch[0], batch[1]
        p = model(x.to(DEVICE)).cpu().numpy()
        se += ((p - y.numpy()) ** 2).sum(0); n += len(y)
    rmse = np.sqrt(se / n)
    return rmse


@torch.no_grad()
def evaluate_shape(model, loader, offsets):
    """Per-offset RMSE of the predicted road geometry, in METRES: along-track (X)
    and lateral (Y). Y is the one that matters for placing the car."""
    model.eval()
    se, n = np.zeros((len(offsets), 2)), 0
    for x, _, sp in loader:
        _, p = model.forward_both(x.to(DEVICE))
        se += ((p.cpu().numpy() - sp.numpy()) ** 2).sum(0); n += len(sp)
    return np.sqrt(se / n)


def run_stage1_vae(save, epochs, seed=0, fold=-1, nfolds=5, latent_dim=128,
                   kl_weight=1.0, lr=1e-4):
    """Extrinsic stage 1: a general-purpose vision VAE, reconstruction and KL only.

    It never receives the curvature target. That is deliberate and it is the whole
    content of the intrinsic/extrinsic distinction -- an encoder that has already
    seen the physical labels is not a general-purpose vision encoder, which is why
    `checkpoints/piwm_lane_v6_donkey/ae.tar` cannot stand in for this stage.
    """
    set_train_seed(seed)
    ds = KappaFrameDataset(with_shape=False)
    tr, va, val_eps = split(ds, fold=fold, nfolds=nfolds)
    print(f"  split -> {describe_split(len(ds.files), fold, nfolds)}")
    trl = DataLoader(tr, batch_size=64, shuffle=True, drop_last=True)
    val = DataLoader(va, batch_size=128, shuffle=False)

    model = ExtrinsicVisionVAE(latent_dim=latent_dim, frame_stack=FS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"arch=extrinsic_vae  latent_dim={latent_dim}  kl_weight={kl_weight}  "
          f"params={sum(p.numel() for p in model.parameters()):,}  (NO physical target)")

    best = float("inf")
    for ep in range(epochs):
        model.train(); tot = nb = 0
        for batch in tqdm(trl, desc=f"vae {ep+1}/{epochs}"):
            x = batch[0].to(DEVICE)
            loss, _ = model.loss(x, kl_weight=kl_weight)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        model.eval(); vtot = vnb = 0
        with torch.no_grad():
            for batch in val:
                loss, _ = model.loss(batch[0].to(DEVICE), kl_weight=kl_weight)
                vtot += loss.item(); vnb += 1
        vloss = vtot / max(vnb, 1)
        print(f"  ep{ep+1} train={tot/max(nb,1):.1f}  val={vloss:.1f}")
        if vloss < best:
            best = vloss
            os.makedirs(_os.path.dirname(save), exist_ok=True)
            torch.save({"model": model.state_dict(), "latent_dim": latent_dim,
                        "val_loss": vloss, "arch": "extrinsic_vae"}, save)
    print(f"\nBEST stage-1 val loss = {best:.1f}  -> {save}")


def run(mode, save, epochs, backbone_ckpt, arch="cnn", target="kappa", seed=0,
        fold=-1, nfolds=5, vae_ckpt="", head="mlp", lambda_interp=1000.0, beta=1.0,
        unfreeze_stage1=False):
    set_train_seed(seed)
    ds = KappaFrameDataset(with_shape=(target == "shape"))
    offsets = np.load(_os.path.join(DATA, "_meta", "stats.npz"))["kappa_offsets"]
    tr, va, val_eps = split(ds, fold=fold, nfolds=nfolds)
    print(f"  split -> {describe_split(len(ds.files), fold, nfolds)}")
    print(f"train={len(tr)} val={len(va)} (val segs={sorted(val_eps)})")
    trl = DataLoader(tr, batch_size=128, shuffle=True, drop_last=True, num_workers=0)
    val = DataLoader(va, batch_size=256, shuffle=False, num_workers=0)

    if arch == "extrinsic":
        # stage 2. Stage 1 must already exist -- train it with --arch extrinsic_vae,
        # which never sees a physical target, and that separation is what makes this
        # extrinsic rather than just two networks in a row.
        if not _os.path.exists(vae_ckpt):
            raise SystemExit(
                f"extrinsic stage 2 needs a stage-1 VAE at {vae_ckpt}.\n"
                f"Train it first:  --arch extrinsic_vae --save {vae_ckpt}")
        vck = load_checkpoint(vae_ckpt)
        vae = ExtrinsicVisionVAE(latent_dim=vck.get("latent_dim", 128), frame_stack=FS)
        vae.load_state_dict(vck["model"])
        # Stage 1 is frozen by default: the conference recipe freezes it, and
        # unfreezing turns this back into a jointly trained encoder, i.e. no longer
        # the variant being tested. --mode is ignored here on purpose (it defaults to
        # finetune for the CNN backbone, which would silently deviate).
        model = ExtrinsicRoadEncoder(vae, n_offsets=ds.n_off, frame_stack=FS,
                                     head=head, freeze=not unfreeze_stage1).to(DEVICE)
        print(f"arch=extrinsic  stage1={vae_ckpt}  head={head}  "
              f"stage1={'UNFROZEN (deviates from the recipe)' if unfreeze_stage1 else 'frozen'}")
    elif arch == "intrinsic":
        model = IntrinsicRoadEncoder(n_offsets=ds.n_off, frame_stack=FS).to(DEVICE)
        print("arch=intrinsic  (single encoder; latent split into physical + visual)")
    elif arch == "lstm":
        model = LSTMRoadEncoder(n_offsets=ds.n_off, frame_stack=FS).to(DEVICE)
        print("arch=lstm  (non-physical sequence benchmark)")
    elif arch == "transformer":
        model = TransformerRoadEncoder(n_offsets=ds.n_off, frame_stack=FS).to(DEVICE)
        print("arch=transformer  (non-physical sequence benchmark)")
    elif arch == "vqformer":
        # per-frame CNN -> VQ(512) -> Transformer over the window. Always trained
        # from scratch: there is no pretrained VQ/Transformer backbone in this repo,
        # so --mode / --backbone do not apply and are ignored on purpose.
        model = RoadContextVQFormer(n_offsets=ds.n_off, frame_stack=FS).to(DEVICE)
        print("arch=vqformer  (per-frame CNN + VQ-512 + 2-layer Transformer, from scratch)")
        if mode == "frozen" or backbone_ckpt.lower() != "scratch":
            print("  note: --mode/--backbone ignored for this arch (nothing to warm-start from)")
    else:
        model = RoadContextEncoder(n_offsets=ds.n_off,
                                   freeze_backbone=(mode == "frozen"),
                                   predict_shape=(target == "shape")).to(DEVICE)
        if backbone_ckpt.lower() == "scratch":
            print("backbone: RANDOM init (no v6 warm-start) -- native-from-scratch test")
        else:
            # warm-start the backbone from the trained v6 encoder
            model.load_backbone(load_checkpoint(backbone_ckpt)["encoder"])
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _head = getattr(model, "kappa_head", None) or getattr(model, "head", None)
    _hn = sum(p.numel() for p in _head.parameters()) if _head is not None else 0
    print(f"arch={arch} mode={mode}  trainable params={n_train}  (head only={_hn})")

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
        aux_sum = {}
        for batch in tqdm(trl, desc=f"{target}-{arch}-{mode} {ep+1}/{epochs}"):
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            if target == "shape":
                sp = batch[2].to(DEVICE)
                kap, shp = model.forward_both(x)
                # variance-normalised so curvature (1/m) and geometry (m) weigh alike
                loss = lossfn(kap, y) / ds.kap_var + lossfn(shp, sp) / ds.shp_var
            elif arch == "intrinsic":
                # reconstruction + interpretability + KL, per the conference objective;
                # the interpretability term is what ties z_p to the curvature profile
                loss, parts = model.loss(x, y, lambda_interp=lambda_interp, beta=beta)
                aux_sum = {k: aux_sum.get(k, 0.0) + v for k, v in parts.items()}
            else:
                pred = model(x)
                loss = lossfn(pred, y)
            if arch == "vqformer":
                loss = loss + model.last_vq_loss        # VQ codebook + commitment
                perp_sum += float(model.last_perplexity)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        rmse = evaluate(model, val, offsets)
        if target == "shape":
            srmse = evaluate_shape(model, val, offsets)
            # same normalisation as the loss, so checkpoint selection matches training
            score = float((rmse ** 2).mean() / ds.kap_var
                          + (srmse ** 2).mean() / ds.shp_var)
        else:
            score = float(rmse.mean())
        sch.step(score)
        extra = f"  codebook_perplexity={perp_sum/max(nb,1):.1f}/{model.vq.n_codes}" \
                if arch == "vqformer" else ""
        if aux_sum:
            extra += "  " + "  ".join(f"{k}={v/max(nb,1):.4f}" for k, v in aux_sum.items())
        if target == "shape":
            extra += (f"\n        shape_RMSE m: along={np.round(srmse[:,0],3).tolist()}"
                      f" lat={np.round(srmse[:,1],3).tolist()}")
        print(f"  ep{ep+1} train_loss={tot/nb:.4f}  val_score={score:.4f}  "
              f"kappa_RMSE mean={rmse.mean():.4f} "
              f"per-offset={np.round(rmse, 3).tolist()}{extra}")
        if score < best:
            best = score
            os.makedirs(_os.path.dirname(save), exist_ok=True)
            ck = {"model": model.state_dict(), "mode": mode, "arch": arch,
                  "target": target, "val_rmse": rmse.tolist(),
                  "offsets": offsets.tolist()}
            if target == "shape":
                ck["val_shape_rmse"] = srmse.tolist()
            torch.save(ck, save)
    print(f"\nBEST val score = {best:.4f}  -> {save}")
    print(f"  offsets (m) = {np.round(offsets, 1).tolist()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["frozen", "finetune"], default="finetune")
    p.add_argument("--arch", default="cnn",
                   choices=["cnn", "vqformer", "extrinsic_vae", "extrinsic",
                            "intrinsic", "lstm", "transformer"],
                   help="cnn = shipped channel-stacked backbone (directly supervised); "
                        "vqformer = per-frame CNN + VQ-512 + Transformer; "
                        "extrinsic_vae = conference stage 1, reconstruction+KL only; "
                        "extrinsic = conference stage 2 on a frozen stage-1 latent; "
                        "intrinsic = conference single-encoder variant; "
                        "lstm / transformer = non-physical sequence benchmarks")
    p.add_argument("--vae-ckpt", dest="vae_ckpt", default="",
                   help="stage-1 checkpoint, required by --arch extrinsic")
    p.add_argument("--head", choices=["mlp", "transformer"], default="mlp",
                   help="extrinsic stage-2 head. mlp follows the reference code; "
                        "transformer follows the paper's text. They disagree.")
    p.add_argument("--lambda-interp", dest="lambda_interp", type=float, default=1000.0,
                   help="intrinsic: weight on the interpretability term")
    p.add_argument("--beta", type=float, default=1.0,
                   help="intrinsic: KL weight on the visual latent")
    p.add_argument("--latent-dim", dest="latent_dim", type=int, default=128)
    p.add_argument("--unfreeze-stage1", dest="unfreeze_stage1", action="store_true",
                   help="extrinsic: train stage 1 jointly. Deviates from the recipe.")
    p.add_argument("--target", choices=["kappa", "shape"], default="kappa",
                   help="kappa = curvature profile only (original); "
                        "shape = curvature AND the local centreline geometry, so a "
                        "map-free rollout can read position instead of integrating it")
    p.add_argument("--save", required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--backbone", default="checkpoints/piwm_lane_v6_donkey/ae.tar")
    p.add_argument("--seed", type=int, default=0,
                   help="seeds weight init + batch order; train/val split stays fixed")
    p.add_argument("--fold", type=int, default=-1,
                   help="-1 = legacy single hold-out (default); 0..nfolds-1 selects a CV "
                        "fold. MUST match the fold the dynamics is trained on.")
    p.add_argument("--nfolds", type=int, default=5)
    a = p.parse_args()
    if a.arch == "extrinsic_vae":
        run_stage1_vae(a.save, a.epochs, seed=a.seed, fold=a.fold, nfolds=a.nfolds,
                       latent_dim=a.latent_dim)
    else:
        run(a.mode, a.save, a.epochs, a.backbone, arch=a.arch, target=a.target,
            seed=a.seed, fold=a.fold, nfolds=a.nfolds, vae_ckpt=a.vae_ckpt,
            head=a.head, lambda_interp=a.lambda_interp, beta=a.beta,
            unfreeze_stage1=a.unfreeze_stage1)
