# PIWM — Physics-Informed World Model for Image-Controlled Racing Car

Image-to-future-image prediction for the `CarRacing-v0` DQN-controlled racing car, built on top of the [`hsai-predictor`](https://github.com/hsai-lab/hsai-predictor) dataset. The encoder lifts stacked frames to physical state, a **differentiable physics dynamics model** propagates the state forward, and a decoder renders the predicted frame. Progressive releases (v1 → v6) move from a black-box MLP to a resample-based lane-augmented bicycle model.

## Pipeline

```
frames_{t-2..t} ──► Encoder ──► physical state s_t ──► Dynamics(s_t, a_t) ──► s_{t+1} ──► Decoder ──► image_{t+1}
                                     │                         ▲
                                     └── lane waypoints ───────┘  (v5, v6)
```

- **State** 11-D: `[x, y, yaw, vx, vy, ω, w0, w1, w2, w3, steer]`
- **Encoder target** 9-D (`x, y` integrated by dynamics)
- **Lane augment** (v5/v6) 20-D: 10 waypoints × (x, y) sampled at fixed s-offsets

## Repository layout

```
piwm/
├── config.py                         # device, paths, normalization stats, hparams
├── data/dataset.py                   # RacingCarDataset (autoencoder / dynamics modes)
├── utils.py                          # checkpoint + normalization helpers
├── track_utils.py                    # CarRacing track geometry
├── lane_utils.py                     # lane waypoint sampling + stats
├── lane_preprocess.py                # offline lane feature extraction
├── relative_coords.py                # relative-pose frame conversion
│
├── models/
│   ├── encoder.py / encoder_lane.py
│   ├── decoder.py / decoder_lane.py
│   ├── dynamics.py                   # v1 MLP dynamics
│   ├── dynamics_bicycle.py           # v2 analytical bicycle
│   ├── dynamics_bicycle_v2/v3/v4.py  # ablations (no slip / no drag / full)
│   ├── dynamics_lane_v5.py           # + lane waypoints (rigid-body propagate)
│   ├── dynamics_lane_v6.py           # + resample-based lane propagation
│   ├── localization*.py              # frame→(x,y) localization variants
│   ├── baseline.py                   # black-box baseline
│   └── piwm_v3.py
│
├── baselines/
│   ├── dvbf.py, goku.py, vid2param.py
│   ├── sindyc.py / sindyc_lane.py    # SINDy-C
│   └── shared_dynamics.py / shared_dynamics_lane.py
│
├── train_autoencoder.py              # Phase 1: frames + physics supervision
├── train_dynamics.py                 # Phase 2: single-step dynamics
├── train_dynamics_multistep.py       # Phase 2: k-step rollout
├── train_localization*.py            # (x,y) localization variants
├── train_piwm_rel.py                 # v2-rel (relative coords)
├── train_piwm_bicycle.py             # v4 bicycle (full)
├── train_piwm_bicycle_ablation.py    # v2/v3/v4 ablations
├── train_piwm_lane_v5.py             # v5 lane (3-stage: AE → dyn → e2e)
├── train_piwm_lane_v6.py             # v6 resample-based (reuses v5 AE)
├── train_piwm_e2e.py
├── train_piwm_v3.py
├── train_baseline.py
├── train_shared_baselines.py / _lane.py   # GOKU/DVBF/V2P with shared backbone
│
├── compare_*.py                      # numerical comparison scripts
├── visualize_*.py / animate_*.py / plot_*.py
├── evaluate_by_time.py
└── make_report_pdf.py / make_report_v2.py  # auto-generated PDF reports
```

## Version history (dynamics)

| Tag | Model | Key idea |
|-----|-------|----------|
| v1  | `dynamics.py` MLP | pure black-box residual |
| v2  | `dynamics_bicycle.py` | analytical bicycle + learned residual |
| v3  | `dynamics_bicycle_v3.py` | drops tire slip angle |
| v4  | `dynamics_bicycle_v4.py` | full model (used in later lane variants) |
| v5  | `dynamics_lane_v5.py` | + 20-D lane waypoints, rigid-body body-frame transform |
| v6  | `dynamics_lane_v6.py` | + resample to fixed s-offset (fixes v5 semantic drift) |

## Data

Data is **not** included. It is generated via the upstream project:

- Repo: [andywu0913/OpenAI-GYM-CarRacing-DQN](https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN) (controller) + [hsai-lab/hsai-predictor](https://github.com/hsai-lab/hsai-predictor) (collection pipeline)
- Expected path (see `config.py`): `../hsai-predictor-main/hsai-predictor-main/RacingCar/data/train/controller_5/*.npz`
- Each `.npz`: `imgs, position, yaw, velocity, angular_velocity, wheel_omega, steering_angle, action`

Normalization statistics (`PHYSICS_MEAN/STD`, `PHYSICS_MEAN_REL/STD_REL`) are already computed and hard-coded in `config.py` (≈85K frames, controller_5).

## Usage

```bash
# Stage 1 — autoencoder
python train_autoencoder.py

# Stage 2+3 — lane-augmented v6 (reuses v5 AE checkpoint)
python train_piwm_lane_v5.py        # produces checkpoints/piwm_lane_v5/ae.tar
python train_piwm_lane_v6.py        # produces checkpoints/piwm_lane_v6/{dyn,best}.tar

# Baselines (shared backbone, fair comparison)
python train_shared_baselines_lane.py

# Compare everything
python compare_all_with_v5.py
python make_report_v2.py            # writes report_v2_{cn,en}.pdf
```

Hyperparameters live in `config.py` and at the top of each `train_*.py`.

## Requirements

PyTorch, NumPy, matplotlib, tqdm, reportlab (PDF), Pillow. No `requirements.txt` is shipped — use the same environment as `hsai-predictor` (`conda env piwm`).

## Notes

- `checkpoints/`, `*.log`, `vis/`, and report PDFs are excluded from version control (see `.gitignore`). Re-generate them by re-running the training / comparison scripts.
- `USE_RELATIVE = True` (config.py) switches normalization to pose-relative coordinates — the default from v2-rel onwards.
