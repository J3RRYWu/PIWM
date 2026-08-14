# Code review — k-fold matrix / encoder-variants round

Reviewed 2026-08-14. Scope: `src/folds.py`, `src/models/road_perception_ae.py`,
`src/baselines/road_seq_baselines.py`, `src/train/train_kappa_perception.py`,
`src/train/train_frenet.py`, `src/train/train_baselines_donkey.py`,
`scripts/run_matrix.py`, `scripts/eval_folds_table.py`, `scripts/eval_seeds_table.py`,
`scripts/eval_kappa_ablation.py`, `scripts/select_checkpoints.py`,
`scripts/sindyc_fold_curves.py`, `scripts/fig_main_cv.py`, `scripts/fig_delta_cv.py`.
Every suspected issue was traced to its call path; claims below marked VERIFIED were
confirmed by running code or reading the actual dependency, not by pattern-matching.

Counts: **0 critical, 4 major, 6 minor.**

---

## Major

### M1. Evaluators sample theta stochastically with no seed — tables are not run-reproducible
- Where: `scripts/eval_seeds_table.py:193`, `scripts/eval_folds_table.py:184`,
  `scripts/select_checkpoints.py:95`, all via `infer_theta` in
  `src/baselines/shared_dynamics_lane.py:117` / `:169` (`theta = mu + exp(0.5*lv) * randn`).
- Failure scenario: every V2P / goku_obs cell (rows, paired bootstraps, and the
  epoch picked by `select_checkpoints.py`) changes on every rerun of the evaluator,
  because theta is drawn from the posterior and **no script calls `torch.manual_seed`**.
  With a trained fold-0 V2P checkpoint the posterior std is ~0.017–0.041 against
  mu ~0.1–0.8 (VERIFIED by loading `checkpoints/v2p_lane_donkey_f0/best.tar`), so the
  per-window rollout is genuinely stochastic; window-averaging shrinks it, but the
  mean shift across reruns is nonzero and `select_checkpoints` can flip between
  near-tied epochs on a single draw. This breaks the project rule that every paper
  number traces to a checkpoint — the checkpoint alone does not determine the number.
  Note this is inherited verbatim from `src/eval_frenet_vs_baselines.py:126` (the
  pinned 08-08 harness), so it is a pre-existing convention, not a new regression.
- Fix: add `torch.manual_seed(0)` at the top of each evaluator's `main()` (preserves
  the sampled-theta convention while making runs bit-reproducible), or switch eval to
  `theta = mu` — but the latter breaks comparability with the pinned 08-08 numbers,
  so seeding is the safe change. `select_checkpoints.score()` should additionally
  reuse one fixed theta draw (or mu) across snapshots of the same run so the epoch
  choice compares dynamics, not noise draws.

### M2. `run_matrix.py --force` races dependent jobs against their dependency's rewrite
- Where: `scripts/run_matrix.py:320` (`todo` includes everything under `--force`) plus
  `:354` (`done_ok` seeded from `out` existence over ALL jobs, not just non-forced ones).
- Failure scenario: `run_matrix.py --preset encoder_variants --force` re-queues both
  `vae_f0` and `extrinsic_f0`; since the OLD `enc_vae_f0.tar` still exists,
  `extrinsic_f0` is `_ready()` immediately and launches in parallel with the `vae_f0`
  retrain, which rewrites that same file at every improving epoch
  (`train_kappa_perception.py:173`). Stage 2 then loads a stale stage-1 — or a
  torn file mid-`torch.save`, crashing — and its "forced rerun" silently reports
  numbers for the old VAE. Reachable only with `--force` on a preset with `needs`.
- Fix: when `--force` is set, initialize `done_ok` to
  `{j["name"] for j in jobs if j not in todo and (ROOT/j["out"]).exists()}` (i.e. an
  artifact only satisfies a dependency if that job is NOT being rerun).

### M3. Resume-by-checkpoint accepts a partial checkpoint as complete
- Where: `scripts/run_matrix.py:320` (skip if `out` exists), against trainers that
  write `best.tar` from the first improving epoch onward
  (`train_frenet.py:203-206`, `train_kappa_perception.py:290-298` and `:170-174`,
  `train_baselines_donkey.py:230-233`).
- Failure scenario: a training job killed without the parent observing its exit
  (power loss; second Ctrl-C detaches children which are later killed; OOM-killed
  worker on a rented box) leaves an epoch-2 `best.tar`. Re-running the matrix counts
  it as done forever, and the undertrained model flows into the published tables with
  no warning — the DIRTY state at `:394` only exists while the parent is alive to see
  the exit code. This is exactly the "resume path mistakes partial output for
  complete" class; today nothing distinguishes a finished 60-epoch run from a
  2-epoch stub.
- Fix: have trainers write a terminal marker (e.g. `save + ".done"` or an
  `"epochs_run"` field written once after the final epoch) and make `run_matrix`'s
  skip test require it; or at minimum log a scan of `reports/matrix/logs/*.log` tails
  on resume and warn for outputs whose log never reached the final epoch line.

### M4. `eval_kappa_ablation.py` paired section crashes on a partially trained matrix
- Where: `scripts/eval_kappa_ablation.py:208-212`.
- Failure scenario: the guard `r not in per_fold[got[0]]` checks only the FIRST fold,
  then `d = [(per_fold[f][r][:, 2] - per_fold[f]["scratch"][:, 2]).mean() for f in got]`
  indexes every fold unconditionally — KeyError as soon as a row (or `scratch`
  itself) exists in fold `got[0]` but is missing in another fold, which the script's
  own contract invites ("runs on whatever subset has been trained", missing
  checkpoints skipped per fold at `:129`). The whole report dies after minutes of
  rollouts, and nothing is written.
- Fix: `folds_ok = [f for f in got if r in per_fold[f] and "scratch" in per_fold[f]]`
  and compute/print over those (as `eval_folds_table.py:400-401` already does).

## Minor

- **m1** `scripts/eval_folds_table.py:265-267` — in `leak_audit`, if either subset has
  no usable checkpoints `eval_one_fold` returns `None` and `row not in subsets[t]`
  raises TypeError. Guard `if not all(subsets.values()): return` after `:255`.
- **m2** `scripts/sindyc_fold_curves.py:99` — uses `T = len(phys)` where
  `eval_folds_table.py:150` uses `T = min(len(phys), len(fst))`; if a frenet file
  were ever shorter than its prep episode the SINDYc row would walk extra windows
  and the "comparable cell-for-cell" claim would silently break. VERIFIED benign on
  current data (all 71 episode lengths identical between
  `Data_Donkeycar_prep` and `Data_Donkeycar_frenet`), so this is a robustness fix:
  add the same `min()`.
- **m3** `src/train/train_frenet.py:168-170` — `half_sup` is computed from the range
  of ALL episodes (train+val) while `train_baselines_donkey.py:177` computes it from
  train windows only. Effect is a scalar range from held-out data (negligible
  leakage) but the two scripts define "the same delta" on slightly different `|X_i|`;
  worth aligning (use train episodes only) and worth a sentence if the delta sweep is
  described as identical across models.
- **m4** `src/train/train_kappa_perception.py:293-295` — extrinsic/intrinsic
  checkpoints do not record `head`, `latent_dim`, or `visual_dim`;
  `eval_kappa_ablation._load_enc:88-91` reconstructs with defaults. Any non-default
  training (`--head transformer`, `--latent-dim != 128`) produces a checkpoint the
  evaluator cannot load (loud `load_state_dict` mismatch, not silent misload — which
  is why this is minor). Store those fields in the checkpoint dict and read them back.
- **m5** `scripts/fig_main_cv.py:103` crashes (`steps=None`) if every series is
  missing from the cache; `scripts/fig_delta_cv.py:44,53` KeyErrors if any fold in
  the cache lacks `ours-a_d5/_d10/V2P` (cache written while the matrix was
  incomplete). Both are plotting scripts run interactively, fail loudly.
- **m6** `scripts/run_matrix.py:204-210` — `jobs_sindyc` loops `for d in deltas` but
  never uses `d`; with any future multi-delta call it would emit duplicate
  name/output jobs that race on the same `model.pkl`. Drop the loop or fold `d` into
  the suffix.

## Checked and found sound (the load-bearing claims)

- **Legacy split bit-identity** (`src/folds.py:39-41`): VERIFIED against all three
  pre-refactor copies via git (`c68d8ae~1`: `donkey_dataset.split_segments`,
  `train_frenet.split`, `train_kappa_perception.split`) — same
  `default_rng(seed).permutation`, same `max(1, int(round(n*val_frac)))`, same
  slice order. Selftest run: legacy val = [5, 8, 30, 50, 64, 69, 70] for n=71, and
  folds partition exactly (no leak, full cover) for k=3,5,7. The leak-audit premise
  (legacy val ⊂ fold-0 val, same permutation prefix) also VERIFIED by running it.
- **Variance matching** (`train_frenet.py:139-140`): bias ~ U[-h,h] gives h²/3;
  resid = mean of 50 U[-h,h] gives h²/150; sum = 51h²/150. `_GAUSS_SCALE =
  sqrt(51/150)` is exactly right, and `gaussian_supervision_noise` applies it per-dim
  to the same `half` vector the delta noise uses. Broadcasting in both noise fns
  ((b,t,d)×(d,) and (b,t,d,50)×(d,1)) checked shape-by-shape; correct.
- **Fold plumbing**: all three trainers and all four evaluators reach the split
  exclusively through `folds.fold_split` (train_kappa `:104`, train_frenet `:64`,
  train_baselines via `split_segments:179`, eval scripts directly); fold f means the
  same episode set everywhere, so encoder-f/dynamics-f/baseline-f are mutually clean.
- **Window pairing**: `eval_folds_table.py:210-220` and `eval_kappa_ablation.py:171-173`
  enforce equal window counts per row within a fold and raise otherwise; every row
  appends unconditionally per window when its model exists, so the guard is
  sufficient. `eval_seeds_table` appends every row unconditionally — inherently paired.
- **Checkpoint→class mapping**: `select_checkpoints.VARIANTS` dirnames match the
  training save paths (`train_baselines_donkey.py:243,248,336,340,352,361`) and
  `eval_folds_table`'s `run = f"{v.lower()}_lane_donkey_f{fold}{dtag(d)}"`; goku_obs
  is constructed with `theta_dim=8` in both evaluators so a goku_obs checkpoint can
  never be loaded into the theta-free class (it would fail loudly anyway).
  `metric_tar` key written at `select_checkpoints.py:162` matches the read at
  `eval_folds_table.py:123`.
- **SINDYc SUFFIX/out_dir** (`train_baselines_donkey.py:404`): per-fold dirs, no
  overwrite across folds; matches `run_matrix` job outputs and
  `sindyc_fold_curves` read paths. Divergence guard freezes-then-clamps and cannot
  resurrect a dead trajectory; the exception path leaves `z` at its last finite value
  before overwriting with the clamp sentinel — correct.
- **SNAP_EVERY**: snapshots land in the same directory as `best.tar` for both the NN
  and obs-conditioned paths; module-level globals are assigned in `__main__` before
  any training function runs (they are read at call time), so `--fold/--suffix/
  --snapshot-every` all take effect.
- **run_matrix scheduling (non-force)**: needs/done_ok/failed logic cannot deadlock —
  jobs waiting on a running dep are re-polled, jobs waiting on a failed dep are
  removed with SKIP, dirty deps intentionally satisfy dependents (documented tradeoff);
  venv fallback order (.venv → bin → `sys.executable`) is sane.
- **VAE/encoder classes**: frozen-stage handling (`ExtrinsicRoadEncoder.train/:177`,
  grad gating `:184`), last-frame-only reconstruction+KL consistency, and
  `IntrinsicRoadEncoder`'s deterministic z_p all do what the docstrings claim;
  `evaluate()` sees exactly the interpretable output for every arch.
