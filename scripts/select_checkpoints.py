"""Re-select the baselines' checkpoints by the REPORTED metric, post hoc.

THE ASYMMETRY THIS FIXES  (HANDOFF §0.3)
Our Frenet dynamics keeps the epoch with the best 100-step xy error -- the number
the paper reports. The baselines keep the epoch with the best K=32 normalised
composite validation loss, which barely predicts it: goku_obs and v2p are the same
architecture with the same 58,220 parameters and val losses of 0.300 vs 0.290, yet
score 0.561 vs 0.384 at @100. So the two sides were selected by different rules and
the baselines got the worse-correlated one.

`--snapshot-every 5` kept every fifth epoch of the 5-fold matrix, so the criterion
can be changed without retraining anything: this script scores each snapshot with
the reported metric and reports what the table looks like under either rule.

WHAT THIS DOES NOT DO
It does not make the comparison unbiased. Selecting the epoch by the metric you then
report is selection on the evaluation set, and it flatters whoever does it -- which
is precisely why doing it for ours but not for the baselines was unfair. Applying it
to both makes them SYMMETRIC, not clean. If the paper reports these numbers it must
say the epoch was chosen this way for every model.

Selection runs on a coarser window stride than the final tables (choosing an epoch
does not need their precision); the chosen checkpoints should then be measured by
scripts/eval_folds_table.py at full density.

    <py311> scripts/select_checkpoints.py                 # delta=0, folds 0-4
    <py311> scripts/select_checkpoints.py --deltas 0 0.05 0.10
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)

import argparse
import glob
import json
import re
import time
import numpy as np
import torch

import eval_seeds_table as est
from folds import fold_split

from config import DATA_DIR, PHYSICS_STD_REL
from lane_utils import LANE_FRAME_STACK as FS
from donkey_config import DONKEY_SPATIAL_SCALE as SCALE
from train_piwm_lane_v5 import SeqLaneDataset
from baselines.shared_dynamics_lane import (DynamicsGOKULane, DynamicsDVBFLane,
                                            DynamicsVid2ParamLane)
from models.encoder_lane import PhysicsEncoderLane
from utils import load_checkpoint

K = est.K
VARIANTS = {"dvbf": (DynamicsDVBFLane, "dvbf_lane_donkey"),
            "goku": (DynamicsGOKULane, "goku_lane_donkey"),
            "v2p": (DynamicsVid2ParamLane, "v2p_lane_donkey"),
            "goku_obs": (lambda: DynamicsGOKULane(theta_dim=DynamicsVid2ParamLane.THETA_DIM),
                         "goku_obs_lane_donkey")}
OBS_CONDITIONED = {"v2p", "goku_obs"}
OUT = _os.path.join("reports", "matrix", "selected_checkpoints.json")


def dtag(d):
    return "" if d == 0 else f"_d{int(round(d * 100))}"


def fold_windows(base, fr_files, fold, nfolds, stride, enc):
    """Every evaluation window of one fold, stacked so a whole snapshot can be
    rolled out in one batched pass. `obs` does not depend on the snapshot, so it is
    computed once here rather than 12 times."""
    _, val_eps = fold_split(len(base.imgs_list), fold, nfolds)
    S, A, O = [], [], []
    for ep in sorted(val_eps):
        phys = base.phys_list[ep]; acts31 = base.acts_list[ep]
        wpw = base.wp_world_list[ep]; imgs = base.imgs_list[ep]
        T = min(len(phys), len(np.load(fr_files[ep])["state"]))
        if T < K + 1 or T < FS:
            continue
        for t0 in range(FS - 1, T - K - 1, stride):
            t0 = int(t0)
            S.append(est.build_state31(phys, wpw, t0, K + 1))
            A.append(acts31[t0:t0 + K])
            with torch.no_grad():
                stack = np.stack([imgs[t0 - FS + 1 + j] for j in range(FS)], 0)[None]
                O.append(enc(torch.tensor(stack, dtype=torch.float32))[0].numpy())
    return (np.stack(S).astype(np.float32), np.stack(A).astype(np.float32),
            torch.tensor(np.stack(O)).unsqueeze(1))


@torch.no_grad()
def score(model, S, A, obs, obs_conditioned):
    """Mean E_xy@100 in real metres over the whole fold, one batched rollout."""
    z = torch.tensor(S[:, 0])
    acts = torch.tensor(A)
    theta = model.infer_theta(obs)[0] if obs_conditioned else None
    for k in range(K):
        if theta is not None:
            z = model.step(z, acts[:, k], theta)
        else:
            out = model(z, acts[:, k])
            z = out[0] if isinstance(out, tuple) else out
    e = np.linalg.norm((z[:, :2].numpy() - S[:, K, :2]) * PHYSICS_STD_REL[:2] / SCALE, axis=-1)
    return float(e.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--nfolds", type=int, default=5)
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.0])
    ap.add_argument("--variants", nargs="+", default=["dvbf", "goku", "v2p", "goku_obs"])
    ap.add_argument("--stride", type=int, default=8,
                    help="window stride for SELECTION only; the tables use 4")
    a = ap.parse_args()

    t0 = time.time()
    lines = []
    def log(s=""):
        print(s, flush=True); lines.append(s)

    base = SeqLaneDataset(DATA_DIR, seq_len=2)
    fr_files = sorted([f for f in glob.glob(_os.path.join(est.FRENET_DIR, "*.npz"))
                       if "_meta" not in f])
    enc = PhysicsEncoderLane()
    enc.load_state_dict(load_checkpoint("checkpoints/piwm_lane_v6_donkey/ae.tar")["encoder"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False

    chosen, rows = {}, []
    log(f"{'run':<26} {'val_loss pick':>14} {'metric pick':>13} {'epoch':>6} {'gain':>8}")
    log("-" * 72)
    for f in a.folds:
        S, A, obs = fold_windows(base, fr_files, f, a.nfolds, a.stride, enc)
        for v in a.variants:
            cls, dirname = VARIANTS[v]
            for d in a.deltas:
                run = f"{dirname}_f{f}{dtag(d)}"
                dd = _os.path.join("checkpoints", run)
                best = _os.path.join(dd, "best.tar")
                if not _os.path.exists(best):
                    continue
                snaps = sorted(glob.glob(_os.path.join(dd, "ep*.tar")),
                               key=lambda p: int(re.search(r"ep(\d+)\.tar$", p).group(1)))
                m = cls()
                m.load_state_dict(load_checkpoint(best)["model"]); m.eval()
                s_best = score(m, S, A, obs, v in OBS_CONDITIONED)
                if not snaps:
                    log(f"{run:<26} {s_best:>13.3f}m {'(no snapshots)':>13}")
                    continue
                scores = []
                for p in snaps:
                    ck = load_checkpoint(p)
                    m.load_state_dict(ck["model"]); m.eval()
                    scores.append((score(m, S, A, obs, v in OBS_CONDITIONED),
                                   int(re.search(r"ep(\d+)\.tar$", p).group(1)), p))
                # best.tar belongs in the pool: it is just another epoch, and ours picks
                # its epoch over ALL of them while the snapshots are only every 5th. Leaving
                # it out would hand the baseline a coarser grid than we give ourselves.
                s_new, ep_new, p_new = min(scores + [(s_best, -1, best)])
                # the val-loss pick is the honest reference even if a snapshot beats it
                chosen[run] = dict(best_tar=best, val_loss_score=s_best,
                                   metric_tar=p_new, metric_score=s_new, epoch=ep_new,
                                   all_scores=[(e, s) for s, e, _ in scores])
                rows.append((v, run, s_best, s_new))
                log(f"{run:<26} {s_best:>13.3f}m {s_new:>12.3f}m "
                    f"{('best' if ep_new < 0 else str(ep_new)):>6} "
                    f"{s_new - s_best:>+7.3f}m")

    if not rows:
        log("\nno runs with snapshots found -- was the matrix run with --snapshot-every?")
        return

    log("\n" + "=" * 72)
    log("effect on the table: mean E_xy@100 over folds, by selection rule")
    log("=" * 72)
    log(f"{'model':<12} {'val_loss (current)':>20} {'reported metric':>18} {'shift':>9}")
    log("-" * 72)
    for v in a.variants:
        r = [x for x in rows if x[0] == v]
        if not r:
            continue
        b = float(np.mean([x[2] for x in r])); n = float(np.mean([x[3] for x in r]))
        log(f"{v:<12} {b:>19.3f}m {n:>17.3f}m {n - b:>+8.3f}m")
    log("\n(selection stride "
        f"{a.stride}, so these are not the table's numbers -- rerun "
        "eval_folds_table.py on the chosen files for those)")
    log("SYMMETRY, NOT CLEANLINESS: ours already picks its epoch this way. Applying it "
        "to the baselines removes the asymmetry but leaves BOTH sides selected on the "
        "evaluation windows, which the paper must state.")

    _os.makedirs(_os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(chosen, fh, indent=1)
    with open(OUT.replace(".json", ".md"), "w", encoding="utf-8") as fh:
        fh.write("# post-hoc checkpoint selection\n\n```\n" + "\n".join(lines) + "\n```\n")
    print(f"\nsaved -> {OUT}\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
