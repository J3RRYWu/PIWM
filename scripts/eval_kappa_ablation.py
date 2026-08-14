"""Where the curvature comes from: the kappa-source ablation, per fold.

The paper's Table 2 asks what the rollout loses as the curvature signal degrades
from a privileged map lookup down to a camera-only estimate. It was measured on the
single split, from the same checkpoint family the main table turned out not to
reproduce, so it could not be shown beside 5-fold numbers. This redoes it per fold.

Rows, in decreasing privilege, all sharing that fold's dynamics (checkpoints/cv/dyn_f{f}):
  map oracle    kappa read from the surveyed track at every step   (privileged)
  GT profile    the true curvature preview, perceived-style        (perfect perception)
  scratch       RoadContextEncoder trained from scratch            checkpoints/cv/kappa_f{f}
  v6 finetune   same, warm-started from the v6 vision backbone     kappa_finetune_f{f}
  v6 frozen     same, backbone frozen                              kappa_frozen_f{f}
  VQ+Transformer  per-frame CNN -> VQ-512 -> Transformer           kappa_vqformer_f{f}

Two things the table has to disclose, both of which cut AGAINST the conclusion
rather than for it, so the reading is conservative:
  * the v6 rows warm-start from an encoder trained on the legacy split, which saw
    episodes these folds hold out. That leak favours them.
  * the VQ+Transformer row is the configuration the conference version found
    strongest on fully observable benchmarks, given equal capacity and four times
    the epoch budget here.

    <py311> scripts/eval_kappa_ablation.py
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)

import argparse
import glob
import time
import numpy as np
import torch

import eval_seeds_table as est
from folds import fold_split

from config import DATA_DIR
from lane_utils import LANE_FRAME_STACK as FS
from train_piwm_lane_v5 import SeqLaneDataset
from models.road_perception_vqformer import RoadContextVQFormer
from models.road_perception_ae import (ExtrinsicVisionVAE, ExtrinsicRoadEncoder,
                                       IntrinsicRoadEncoder)
from baselines.road_seq_baselines import LSTMRoadEncoder, TransformerRoadEncoder
from utils import load_checkpoint

K, STRIDE = est.K, est.STRIDE
OFFSETS = est.OFFSETS
REPORT = _os.path.join("reports", "matrix", "kappa_ablation.md")

# row key -> (checkpoint template, label). Missing checkpoints are reported and
# skipped, so this runs on whatever subset has been trained.
ENCODERS = [
    # the directly supervised regressors this repository ships
    ("scratch",  "checkpoints/cv/kappa_f{f}.tar",           "camera, scratch"),
    ("finetune", "checkpoints/cv/kappa_finetune_f{f}.tar",  "camera, v6 warm-start"),
    ("frozen",   "checkpoints/cv/kappa_frozen_f{f}.tar",    "camera, v6 frozen"),
    ("vqformer", "checkpoints/cv/kappa_vqformer_f{f}.tar",  "camera, VQ+Transformer"),
    # the conference version's autoencoding design space, implemented for this task
    ("extrinsic", "checkpoints/cv/enc_extrinsic_f{f}.tar",  "camera, extrinsic (conf.)"),
    ("intrinsic_l1000",   "checkpoints/cv/enc_intrinsicl1000_f{f}.tar",
     "camera, intrinsic (lam 1e3)"),
    ("intrinsic_l10000",  "checkpoints/cv/enc_intrinsicl10000_f{f}.tar",
     "camera, intrinsic (lam 1e4)"),
    ("intrinsic_l100000", "checkpoints/cv/enc_intrinsicl100000_f{f}.tar",
     "camera, intrinsic (lam 1e5)"),
    # the non-physical sequence benchmarks the conference version had
    ("lstm",        "checkpoints/cv/enc_lstm_f{f}.tar",        "camera, LSTM"),
    ("transformer", "checkpoints/cv/enc_transformer_f{f}.tar", "camera, Transformer"),
]
ROWS = ["map-oracle", "gt-profile"] + [k for k, _, _ in ENCODERS]
LABEL = dict([("map-oracle", "map lookup (privileged)"),
              ("gt-profile", "true preview (perfect perception)")]
             + [(k, lab) for k, _, lab in ENCODERS])


def _load_enc(tag, path):
    """Build the right class for a row, then load. Every one of these exposes
    forward(stack) -> (B, n_offsets), which is what makes them interchangeable in
    the rollout below."""
    n = len(OFFSETS)
    if tag == "vqformer":
        m = RoadContextVQFormer(n_offsets=n, frame_stack=FS)
    elif tag == "extrinsic":
        # the checkpoint carries the stage-1 weights as a submodule, so a freshly
        # constructed VAE is only a shape template here
        m = ExtrinsicRoadEncoder(ExtrinsicVisionVAE(frame_stack=FS),
                                 n_offsets=n, frame_stack=FS)
    elif tag.startswith("intrinsic"):
        m = IntrinsicRoadEncoder(n_offsets=n, frame_stack=FS)
    elif tag == "lstm":
        m = LSTMRoadEncoder(n_offsets=n, frame_stack=FS)
    elif tag == "transformer":
        m = TransformerRoadEncoder(n_offsets=n, frame_stack=FS)
    else:
        m = est.RoadContextEncoder(n_offsets=n, freeze_backbone=True)
    m.load_state_dict(load_checkpoint(path)["model"]); m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--nfolds", type=int, default=5)
    a = ap.parse_args()

    t0 = time.time()
    lines = []
    def log(s=""):
        print(s, flush=True); lines.append(s)

    base = SeqLaneDataset(DATA_DIR, seq_len=2)
    fr_files = sorted([f for f in glob.glob(_os.path.join(est.FRENET_DIR, "*.npz"))
                       if "_meta" not in f])

    per_fold, per_fold_rmse = {}, {}
    for f in a.folds:
        dyn_ck = f"checkpoints/cv/dyn_f{f}.tar"
        if not _os.path.exists(dyn_ck):
            log(f"fold {f}: {dyn_ck} missing -- skipped"); continue
        fr = est._load_frenet(dyn_ck)
        encs = {}
        for tag, tmpl, _ in ENCODERS:
            p = tmpl.format(f=f)
            if _os.path.exists(p):
                encs[tag] = _load_enc(tag, p)
            else:
                log(f"  ! missing {p}")
        _, val_eps = fold_split(len(base.imgs_list), f, a.nfolds)

        res = {r: [] for r in ROWS}
        rmse = {t: [] for t in encs}          # per-offset squared error, camera rows only
        for ep in sorted(val_eps):
            frd = np.load(fr_files[ep])
            fst, fac, fim = frd["state"], frd["action"], frd["imgs"].astype(np.float32)
            kap_true_all = frd["kappa_profile"]
            T = len(fst)
            if T < K + 1 or T < FS:
                continue
            for t in range(FS - 1, T - K - 1, STRIDE):
                t = int(t)
                gt = fst[t:t + K + 1]
                z0 = torch.tensor(fst[t]); acts = torch.tensor(fac[t:t + K])
                gt_xy = est.sd2xy_nn(gt[:, 0], gt[:, 1])
                stack = torch.tensor(fim[t - FS + 1:t + 1], dtype=torch.float32).unsqueeze(0)
                kap_t = kap_true_all[t].astype(np.float32)

                with torch.no_grad():
                    # privileged: kappa from the surveyed track at EVERY step
                    p = fr(z0.unsqueeze(0), acts[0].unsqueeze(0))
                    zz = z0.unsqueeze(0); out = [z0.numpy()]
                    for k in range(K):
                        zz = fr(zz, acts[k].unsqueeze(0))
                        out.append(zz[0].numpy())
                    o = np.array(out)
                    res["map-oracle"].append(np.linalg.norm(
                        est.sd2xy_nn(o[:, 0], o[:, 1]) - gt_xy, axis=-1))
                    # perfect perception: the TRUE preview, read the same way
                    o = fr.rollout_perceived(z0, acts, torch.tensor(kap_t), OFFSETS).numpy()
                    res["gt-profile"].append(np.linalg.norm(
                        est.sd2xy_nn(o[:, 0], o[:, 1]) - gt_xy, axis=-1))
                    # camera rows
                    for tag, m in encs.items():
                        prof = m(stack)[0].numpy().astype(np.float32)
                        rmse[tag].append((prof - kap_t) ** 2)
                        o = fr.rollout_perceived(z0, acts, torch.tensor(prof), OFFSETS).numpy()
                        res[tag].append(np.linalg.norm(
                            est.sd2xy_nn(o[:, 0], o[:, 1]) - gt_xy, axis=-1))

        n = {r: len(v) for r, v in res.items() if v}
        if len(set(n.values())) > 1:
            raise RuntimeError(f"fold {f}: rows saw different window counts {n}")
        per_fold[f] = {r: np.array(v)[:, [25, 50, 100]] for r, v in res.items() if v}
        per_fold_rmse[f] = {t: float(np.sqrt(np.mean(np.array(v)))) for t, v in rmse.items()}
        log(f"fold {f}: {list(n.values())[0]} windows, {len(per_fold[f])} rows "
            f"({time.time()-t0:.0f}s)")

    got = sorted(per_fold)
    if not got:
        log("nothing to evaluate"); return

    def cell(row, col):
        v = [per_fold[f][row][:, col].mean() for f in got if row in per_fold[f]]
        if not v:
            return None
        v = np.array(v)
        return v.mean(), (v.max() - v.min()) / 2 if len(v) > 1 else 0.0

    log("\n" + "=" * 78)
    log(f"KAPPA SOURCE ABLATION  E_xy (m), mean +/- half-range over {len(got)} folds")
    log("=" * 78)
    log(f"{'curvature source':<34} {'RMSE_k':>9} {'@25':>13} {'@50':>13} {'@100':>13}")
    log("-" * 78)
    for r in ROWS:
        c = [cell(r, j) for j in range(3)]
        if c[0] is None:
            continue
        rk = [per_fold_rmse[f][r] for f in got if r in per_fold_rmse[f]]
        rk = f"{np.mean(rk):.4f}" if rk else "--"
        cells = [f"{m:.3f}+/-{s:.3f}" for m, s in c]
        log(f"{LABEL[r]:<34} {rk:>9} {cells[0]:>13} {cells[1]:>13} {cells[2]:>13}")

    log("\nRMSE_k is the encoder's curvature error in 1/m, averaged over the preview "
        "offsets;\nthe privileged rows have none because they do not estimate curvature.")
    log("\nPAIRED within fold, @100, relative to the scratch camera encoder:")
    for r in ROWS:
        if r == "scratch" or r not in per_fold[got[0]]:
            continue
        d = [(per_fold[f][r][:, 2] - per_fold[f]["scratch"][:, 2]).mean() for f in got]
        log(f"  {LABEL[r]:<34} {np.mean(d):+.3f}m   per fold "
            f"[{', '.join(f'{x:+.3f}' for x in d)}]")
    log("\nCAVEATS. The v6 rows warm-start from an encoder trained on the legacy split, "
        "which\nsaw episodes these folds hold out: that leak favours them. The "
        "VQ+Transformer row is\nthe conference version's strongest configuration, at "
        "matched capacity and 4x the epochs.")

    _os.makedirs(_os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("# kappa-source ablation, 5-fold\n\n```\n" + "\n".join(lines) + "\n```\n")
    print(f"\nsaved -> {REPORT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
