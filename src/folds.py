"""Episode-level train/val partitioning, shared by every training script.

WHY THIS IS ITS OWN MODULE
`train_frenet.py`, `train_kappa_perception.py` and `train_baselines_donkey.py`
each grew their own copy of the same eight lines. That is survivable for a single
fixed split, but under k-fold it is a silent-leakage bug waiting to happen: if the
kappa encoder's fold 2 is not EXACTLY the dynamics' fold 2, the encoder has seen
the dynamics' validation episodes and every number downstream is inflated. So the
partition lives in one numpy-only module the three scripts can all import without
dragging in config/dataset machinery (the two Frenet scripts only put `src/` on
the path and never call `donkey_config.patch_globals()`).

The episode INDEX is the shared key: `Data_Donkeycar_frenet/*.npz` sorted and
`SeqLaneDataset.imgs_list` are index-aligned (that alignment is what
`eval_frenet_vs_baselines.py` already relies on via `fr_files[ep]`).
"""
import numpy as np


def fold_split(n_eps, fold=-1, nfolds=5, val_frac=0.10, seed=0):
    """Return (train_eps, val_eps) as sets of episode indices.

    fold < 0 (THE DEFAULT) reproduces the legacy single hold-out bit-for-bit:
    the same permutation, the same `round(n_eps * val_frac)` validation episodes.
    Every checkpoint and every number produced before 2026-08-12 depends on that,
    so this path must never change -- `test_legacy_unchanged()` below pins it.

    fold >= 0 carves that SAME permutation into `nfolds` contiguous blocks, so
    the folds are disjoint, cover every episode exactly once, and are identical
    across the three training scripts.

    NOTE the fold hold-out is 1/nfolds of the data (20% at nfolds=5), NOT
    val_frac (10%). Per-fold validation sets are therefore about twice the size
    of the legacy one, so per-fold window counts will not match the 201 windows
    the single-split tables report. That is inherent to k-fold, not a bug.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_eps)
    if fold < 0:
        n_val = max(1, int(round(n_eps * val_frac)))
        return set(perm[n_val:].tolist()), set(perm[:n_val].tolist())
    if not (0 <= fold < nfolds):
        raise ValueError(f"fold {fold} out of range for nfolds={nfolds}")
    if nfolds > n_eps:
        raise ValueError(f"nfolds={nfolds} exceeds n_eps={n_eps}")
    bounds = np.linspace(0, n_eps, nfolds + 1).round().astype(int)
    lo, hi = int(bounds[fold]), int(bounds[fold + 1])
    val = perm[lo:hi]
    train = np.concatenate([perm[:lo], perm[hi:]])
    return set(train.tolist()), set(val.tolist())


def describe(n_eps, fold=-1, nfolds=5, val_frac=0.10, seed=0):
    """One-line provenance string for training logs."""
    tr, va = fold_split(n_eps, fold, nfolds, val_frac, seed)
    tag = "legacy holdout" if fold < 0 else f"fold {fold}/{nfolds}"
    return f"{tag}: {len(tr)} train / {len(va)} val eps, val={sorted(va)}"


def _selftest(n_eps=71):
    # the legacy split is pinned to the episodes the paper's tables were measured on
    _, va = fold_split(n_eps)
    assert sorted(va) == [5, 8, 30, 50, 64, 69, 70], sorted(va)
    # folds partition the episodes exactly once, with no overlap
    for k in (3, 5, 7):
        seen = []
        for f in range(k):
            tr, va = fold_split(n_eps, fold=f, nfolds=k)
            assert not (tr & va), f"leak in fold {f}/{k}"
            assert len(tr) + len(va) == n_eps
            seen += sorted(va)
        assert sorted(seen) == list(range(n_eps)), f"nfolds={k} is not a partition"
    print(f"folds selftest OK (n_eps={n_eps})")
    print(" ", describe(n_eps))
    for f in range(5):
        print(" ", describe(n_eps, fold=f, nfolds=5))


if __name__ == "__main__":
    _selftest()
