"""Decisive comparison for the pure-world-model switch: feed PERCEIVED kappa into the
Frenet dynamics rollout and measure 100-step xy drift, against the privileged map oracle.

Rows (identical val windows, GT init, real metres):
  oracle      kappa(s) looked up from the KNOWN map every step      (privileged upper bound)
  GT-profile  TRUE kappa preview at t0 + structured shift           (ceiling of perception path)
  finetune    perceived kappa (RoadContextEncoder, backbone tuned)  (pure world model)
  frozen      perceived kappa (frozen v6 backbone + linear head)    (pure world model)

oracle->GT-profile gap  = cost of using a single t0 preview instead of the full map.
GT-profile->perceived   = cost of imperfect perception.
Run from piwm/:  <py311 or py> src/.. ; here: python scripts/compare_kappa_rollout.py
"""
import os, sys, glob
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models.frenet_dynamics import FrenetDynamics
from models.road_perception import RoadContextEncoder
from lane_utils import LANE_FRAME_STACK as FS
from utils import load_checkpoint

K = 100
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
FRENET_DIR = os.path.join(ROOT, "Data_Donkeycar_frenet")
META = os.path.join(FRENET_DIR, "_meta")

_tr = np.load(os.path.join(META, "track.npz"))
_cen, _nrm = _tr["centers"], _tr["normals"]
_L, _ds, _M = float(_tr["total_len"]), float(_tr["grid_ds"]), len(_cen)
def sd2xy(s, d):
    i = (np.mod(s, _L) / _ds).astype(int) % _M
    return _cen[i] + d[..., None] * _nrm[i]


def val_split(files, val_frac=0.10, seed=0):
    """Same episode split as train_frenet / train_kappa_perception."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(files))
    nval = max(1, int(round(len(files) * val_frac)))
    return set(perm[:nval].tolist())


@torch.no_grad()
def rollout_map(dyn, z0, acts):
    z = torch.tensor(z0).unsqueeze(0)
    out = [z0.copy()]
    for k in range(K):
        z = dyn(z, torch.tensor(acts[k]).unsqueeze(0))
        out.append(z[0].numpy())
    return np.array(out)


@torch.no_grad()
def rollout_prof(dyn, z0, acts, prof, offsets):
    F = dyn.rollout_perceived(torch.tensor(z0), torch.tensor(acts),
                              torch.tensor(prof), offsets)
    return F.numpy()


def xy_err(pred, gt):
    return np.linalg.norm(sd2xy(pred[:, 0], pred[:, 1]) - sd2xy(gt[:, 0], gt[:, 1]), axis=-1)


def load_dyn(path):
    d = FrenetDynamics(os.path.join(META, "track.npz"), os.path.join(META, "stats.npz"))
    d.load_state_dict(load_checkpoint(path)["dynamics"]); d.eval()
    return d


def main():
    offsets = np.load(os.path.join(META, "stats.npz"))["kappa_offsets"].astype(np.float32)
    # two dynamics: stage-1 (trained with map kappa) and stage-2 (finetuned on the
    # perceived/preview regime). Each evaluated against the same kappa sources.
    dyns = {"map-trained": load_dyn("checkpoints/frenet/dyn_k16.tar"),
            "profile-tuned": load_dyn("checkpoints/frenet/dyn_k16_profile.tar")}

    encs = {}
    for tag, path in [("scratch", "checkpoints/frenet/kappa_scratch.tar"),
                      ("finetune", "checkpoints/frenet/kappa_finetune.tar"),
                      ("frozen", "checkpoints/frenet/kappa_frozen.tar")]:
        m = RoadContextEncoder(n_offsets=len(offsets), freeze_backbone=True)
        m.load_state_dict(load_checkpoint(path)["model"]); m.eval()
        encs[tag] = m

    files = sorted(f for f in glob.glob(os.path.join(FRENET_DIR, "*.npz")) if "_meta" not in f)
    val_eps = val_split(files)
    print(f"val segs: {sorted(val_eps)}")

    # rows: (dynamics, kappa-source)
    rows = [("map-trained", "oracle"), ("map-trained", "GT-profile"),
            ("map-trained", "scratch"), ("map-trained", "finetune"), ("map-trained", "frozen"),
            ("profile-tuned", "oracle"), ("profile-tuned", "GT-profile"),
            ("profile-tuned", "scratch"), ("profile-tuned", "finetune"), ("profile-tuned", "frozen")]
    res = {r: [] for r in rows}
    nwin = 0
    for ep in sorted(val_eps):
        d = np.load(files[ep])
        st, ac, imgs, kp = d["state"], d["action"], d["imgs"].astype(np.float32), d["kappa_profile"]
        T = min(len(st), len(ac) + 1, len(imgs))
        for t0 in range(FS - 1, T - K - 1, 4):
            z0 = st[t0].astype(np.float32); acts = ac[t0:t0 + K].astype(np.float32)
            gt = st[t0:t0 + K + 1]
            stack = torch.tensor(imgs[t0 - FS + 1:t0 + 1]).unsqueeze(0)
            perc = {}
            for tag, m in encs.items():
                with torch.no_grad():
                    perc[tag] = m(stack)[0].numpy().astype(np.float32)
            for dtag, dyn in dyns.items():
                res[(dtag, "oracle")].append(xy_err(rollout_map(dyn, z0, acts), gt))
                res[(dtag, "GT-profile")].append(xy_err(rollout_prof(dyn, z0, acts, kp[t0].astype(np.float32), offsets), gt))
                for etag in encs:
                    res[(dtag, etag)].append(xy_err(rollout_prof(dyn, z0, acts, perc[etag], offsets), gt))
            nwin += 1

    print(f"windows: {nwin}\n")
    print(f"{'dynamics':<14}{'kappa source':<12} {'xy@25':>8} {'xy@50':>8} {'xy@100 med':>11} {'xy@100 mean':>12}")
    print("-" * 70)
    prev = None
    for (dtag, ktag) in rows:
        if dtag != prev and prev is not None: print()
        prev = dtag
        E = np.array(res[(dtag, ktag)])
        print(f"{dtag:<14}{ktag:<12} {E[:,25].mean():>7.3f}m {E[:,50].mean():>7.3f}m "
              f"{np.median(E[:,100]):>10.3f}m {E[:,100].mean():>11.3f}m")
    # Fair-retrain baselines (batch 128/64), same val windows and metric.
    # These SUPERSEDE the earlier under-trained numbers (V2P ~0.44 / DVBF ~0.80),
    # which flattered us; source of truth is src/eval_frenet_vs_baselines.py.
    print("\nref baselines (map-free, same metric): V2P 0.383m  GOKU 0.478m  DVBF 0.610m  @xy100")


if __name__ == "__main__":
    main()
