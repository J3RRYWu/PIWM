"""Full Frenet world-model pipeline + image-prediction comparison.

For matched val windows (GT init), roll out each model and DECODE every step to
a front-view image via the shared v6 decoder; report image MSE vs the real
future frames (the front-view deliverable) alongside the pos MSE.
  - Frenet:  rollout -> FrenetBridge -> shared decoder
  - GOKU/V2P: rollout (31-dim) -> shared decoder
Also saves a filmstrip: real vs Frenet-imagined future frames.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "train"))
import donkey_config; donkey_config.patch_globals()

import glob
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from config import DATA_DIR, PHYSICS_MEAN_REL, PHYSICS_STD_REL, ENCODER_DIM as CARENC
from lane_utils import LANE_MEAN, LANE_STD, LANE_DIM, LANE_FRAME_STACK as FS
from donkey_config import DONKEY_SPATIAL_SCALE as SCALE
from relative_coords import to_relative_np
from donkey_dataset import split_segments
from train_piwm_lane_v5 import SeqLaneDataset
from baselines.shared_dynamics_lane import (DynamicsGOKULane, DynamicsVid2ParamLane)
from models.frenet_dynamics import FrenetDynamics
from models.encoder_lane import PhysicsEncoderLane
from models.decoder_lane import PhysicsDecoderLane
from frenet_bridge import FrenetBridge
from utils import load_checkpoint

K = 100
META = _os.path.join(_os.path.dirname(__file__), "..", "Data_Donkeycar_frenet", "_meta")
FRENET_DIR = _os.path.join(_os.path.dirname(__file__), "..", "Data_Donkeycar_frenet")
_tr = np.load(_os.path.join(META, "track.npz"))
_cen, _nrm, _L, _ds, _M = _tr["centers"], _tr["normals"], float(_tr["total_len"]), float(_tr["grid_ds"]), len(_tr["centers"])
def sd2xy(s, d): i = (np.mod(s, _L) / _ds).astype(int) % _M; return _cen[i] + d[..., None] * _nrm[i]


def build_state31(phys, wpw, t0, T):
    seg = phys[t0:t0 + T]; rel = to_relative_np(seg, 0)
    car = (rel - PHYSICS_MEAN_REL) / PHYSICS_STD_REL
    p0 = seg[0, :2]; y0 = seg[0, 2]; c, s = np.cos(y0), np.sin(y0)
    dd = wpw[t0:t0 + T] - p0
    x = dd[..., 0]*c + dd[..., 1]*s; y = -dd[..., 0]*s + dd[..., 1]*c
    lane = (np.stack([x, y], -1).reshape(T, LANE_DIM) - LANE_MEAN) / LANE_STD
    return np.concatenate([car, lane], -1).astype(np.float32)


@torch.no_grad()
def decode_imgs(dec, dec_in):                      # dec_in (T,29) -> (T,64,64)
    out = []
    for i in range(0, len(dec_in), 32):
        out.append(dec(torch.tensor(dec_in[i:i+32], dtype=torch.float32))[:, 0].numpy())
    return np.concatenate(out, 0)


def main():
    base = SeqLaneDataset(DATA_DIR, seq_len=2)
    _, val_eps = split_segments(base, val_frac=0.10, seed=0)
    fr_files = sorted([f for f in glob.glob(_os.path.join(FRENET_DIR, "*.npz")) if "_meta" not in f])

    ck = load_checkpoint("checkpoints/piwm_lane_v6_donkey/ae.tar")
    enc = PhysicsEncoderLane(); enc.load_state_dict(ck["encoder"]); enc.eval()
    dec = PhysicsDecoderLane(); dec.load_state_dict(ck["decoder"]); dec.eval()
    for m in (enc, dec):
        for p in m.parameters(): p.requires_grad = False
    goku = DynamicsGOKULane(); goku.load_state_dict(load_checkpoint("checkpoints/goku_lane_donkey_longK/best.tar")["model"]); goku.eval()
    v2p = DynamicsVid2ParamLane(); v2p.load_state_dict(load_checkpoint("checkpoints/v2p_lane_donkey_longK/best.tar")["model"]); v2p.eval()
    fr = FrenetDynamics(_os.path.join(META, "track.npz"), _os.path.join(META, "stats.npz"))
    fr.load_state_dict(load_checkpoint("checkpoints/frenet/dyn_k16.tar")["dynamics"]); fr.eval()
    br = FrenetBridge(_os.path.join(META, "track.npz"))

    std_xy = PHYSICS_STD_REL[:2]
    pos = {k: [] for k in ["Frenet", "GOKU", "V2P"]}
    img = {k: [] for k in ["Frenet", "GOKU", "V2P"]}
    sample = None
    for ep in sorted(val_eps):
        phys = base.phys_list[ep]; acts = base.acts_list[ep]; wpw = base.wp_world_list[ep]
        imgs = base.imgs_list[ep]
        frd = np.load(fr_files[ep]); fst = frd["state"]; fac = frd["action"]
        T = min(len(phys), len(fst), len(imgs), len(acts) + 1, len(fac) + 1)
        for t0 in range(FS - 1, T - K - 1, 6):
            real = imgs[t0:t0 + K + 1]
            if len(real) != K + 1: continue
            s31 = build_state31(phys, wpw, t0, K + 1)
            stack0 = np.stack([imgs[t0 - FS + 1 + j] for j in range(FS)], 0)[None]
            with torch.no_grad():
                theta, _, _ = v2p.infer_theta(enc(torch.tensor(stack0, dtype=torch.float32)).unsqueeze(1))
                # baselines rollout (GT init)
                for name, fn in [("GOKU", lambda z, a: goku(z, a)),
                                 ("V2P", lambda z, a: v2p.step(z, a, theta))]:
                    z = torch.tensor(s31[0]).unsqueeze(0); Z = [s31[0]]
                    for k in range(K):
                        z = fn(z, torch.tensor(acts[t0 + k]).unsqueeze(0)); Z.append(z[0].numpy())
                    Z = np.array(Z)
                    pos[name].append(np.linalg.norm((Z[:, :2] - s31[:, :2]) * std_xy / SCALE, axis=-1))
                    di = np.concatenate([Z[:, 2:11], Z[:, 11:]], -1)
                    img[name].append(((decode_imgs(dec, di) - real) ** 2).mean((1, 2)))
                # frenet rollout (GT init)
                z = torch.tensor(fst[t0]).unsqueeze(0); F = [fst[t0]]
                for k in range(K):
                    z = fr(z, torch.tensor(fac[t0 + k]).unsqueeze(0)); F.append(z[0].numpy())
                F = np.array(F)
                pos["Frenet"].append(np.linalg.norm(sd2xy(F[:, 0], F[:, 1]) - sd2xy(fst[t0:t0+K+1, 0], fst[t0:t0+K+1, 1]), axis=-1))
                fr_imgs = decode_imgs(dec, br.to_decoder_input(F))
                img["Frenet"].append(((fr_imgs - real) ** 2).mean((1, 2)))
                if sample is None:
                    sample = (real, fr_imgs)

    def stack(lst):
        good = [np.asarray(a).reshape(-1) for a in lst if np.asarray(a).size == K + 1]
        return np.stack(good) if good else np.zeros((1, K + 1))
    print(f"\nwindows: {len(pos['Frenet'])}")
    print(f"\n{'model':<8} {'pos@100':>9} {'img@1':>8} {'img@25':>8} {'img@50':>8} {'img@100':>8}")
    print("-" * 50)
    img_curves = {}
    for name in ["Frenet", "GOKU", "V2P"]:
        P = stack(pos[name]); I = stack(img[name]); img_curves[name] = I.mean(0)
        print(f"{name:<8} {P[:,100].mean():>8.3f}m {I[:,1].mean():>8.4f} {I[:,25].mean():>8.4f} "
              f"{I[:,50].mean():>8.4f} {I[:,100].mean():>8.4f}")

    # plot image MSE vs step
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    sty = {"Frenet": ("#5B2C9B", 3, "-"), "GOKU": ("#9DDA9D", 2, "--"), "V2P": ("#F2A24C", 2, "--")}
    for name in ["Frenet", "GOKU", "V2P"]:
        I = img_curves[name]; c, lw, ls = sty[name]
        ax.plot(np.arange(K + 1), I, color=c, lw=lw, ls=ls, label=f"{name} (@100={I[100]:.4f})")
    ax.set_xlabel("rollout step"); ax.set_ylabel("front-view image MSE")
    ax.set_title("Imagined future front-view quality (GT init)"); ax.grid(alpha=.3); ax.legend()
    _os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig_frenet_image_mse.png", dpi=130, bbox_inches="tight")

    # filmstrip: real (top) vs Frenet-imagined (bottom), every 10 steps
    real, fimg = sample
    cols = list(range(0, K + 1, 10))
    strip = np.zeros((64 * 2 + 4, 64 * len(cols)), np.float32)
    for j, k in enumerate(cols):
        strip[:64, j*64:(j+1)*64] = real[k]
        strip[68:, j*64:(j+1)*64] = np.clip(fimg[k], 0, 1)
    Image.fromarray((strip * 255).astype("uint8")).resize((64*len(cols)*3, (64*2+4)*3), Image.NEAREST)\
        .save("figures/fig_frenet_filmstrip.png")
    print("saved figures/fig_frenet_image_mse.png + fig_frenet_filmstrip.png")


if __name__ == "__main__":
    main()
