"""Donkeycar-specific constants for adapting PIWM-v6 without architecture change.

The encoder/decoder/dynamics modules read these constants by name from `config`
and `lane_utils` at module-import time, so the donkey training script must patch
`config.*` and `lane_utils.*` *before* importing the model modules.

Track scale: ~10.45 m closed loop (vs CarRacing ~600 unit track). Max speed
~1 m/s (vs ~70 unit/s). DT 1/22 s (vs 1/50). All stats here are computed by
`donkey_prep.py` and cached in `Data_Donkeycar_prep/stats.npz`.

Lane sampling: Δs=0.5 m (vs 5 unit in CarRacing) for 10 samples covering
[-0.5, 4.0] m — fits inside one lap and gives a 4-s lookahead at top speed.
"""
import os
import numpy as np

DONKEY_FPS = 22.0
DONKEY_DT  = 1.0 / DONKEY_FPS

# Real donkey wheelbase in metres. Used to invert bicycle kinematics
#     omega = (v_forward / L) * tan(steer)
# to recover the *physical* steering angle (radians) from observed yaw rate
# and speed. This is what PIWM's 11-dim phys vector expects in slot [10] (see
# dynamics_bicycle_v4: alpha_f = steer - atan2(...)); donkey only logs the
# normalised actuator command, which is what `action[:, 0]` already carries.
DONKEY_WHEELBASE_PHYS = 0.165
DONKEY_STEER_PHYS_MAX = 0.6   # rad, ~34 deg — physical limit of the servo

# BicycleDynamicsV4's learnable-parameter bounds (mass 500-2000 kg, cornering
# stiffness 5e4-5e5 N/rad, wheelbase 1-4 m) are sized for CarRacing-scale
# vehicles. A real donkey is ~50x smaller in every linear dim, so we upscale
# spatial quantities by this factor *only* in the saved dataset. The training
# pipeline then operates in CarRacing units; converting predictions back to
# physical donkey meters is just division by this factor.
DONKEY_SPATIAL_SCALE = 50.0

# Lane sampling — 10 forward-only samples at 0.5 m spacing on the real donkey
# track. CarRacing was top-down so it had a behind-the-car waypoint (s=-5);
# donkey is FIRST-PERSON, so the camera can't observe anything behind. Make
# every waypoint correspond to a region the encoder can actually see.
_DONKEY_LANE_DS_PHYS = 0.5   # metres on the real donkey track
DONKEY_LANE_S_SAMPLES_PHYS = np.arange(0, 10, dtype=np.float32) * _DONKEY_LANE_DS_PHYS
DONKEY_LANE_S_SAMPLES = DONKEY_LANE_S_SAMPLES_PHYS * DONKEY_SPATIAL_SCALE

# Where the preprocessed donkey data lives.
DONKEY_RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "Data_Donkeycar")
DONKEY_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data_Donkeycar_prep")
# Stats file lives in a subdir so SeqLaneDataset's `*.npz` glob on DATA_DIR
# doesn't accidentally pick it up.
DONKEY_STATS_FILE = os.path.join(DONKEY_DATA_DIR, "_meta", "stats.npz")


# ---- Fallback / placeholder stats (overwritten by load_donkey_stats) ----
# 11-dim physics in REL coords: [x, y, yaw, vx, vy, omega, w0..w3, steer]
DONKEY_PHYSICS_MEAN_REL = np.zeros(11, dtype=np.float32)
DONKEY_PHYSICS_STD_REL  = np.ones(11, dtype=np.float32)
DONKEY_LANE_MEAN = np.zeros(20, dtype=np.float32)
DONKEY_LANE_STD  = np.ones(20, dtype=np.float32)


def load_donkey_stats():
    """Load cached normalization stats; returns dict or None if not yet computed."""
    if not os.path.exists(DONKEY_STATS_FILE):
        return None
    s = np.load(DONKEY_STATS_FILE)
    return {
        "physics_mean_rel": s["physics_mean_rel"].astype(np.float32),
        "physics_std_rel":  s["physics_std_rel"].astype(np.float32),
        "lane_mean":        s["lane_mean"].astype(np.float32),
        "lane_std":         s["lane_std"].astype(np.float32),
    }


def patch_globals():
    """Patch `config.*` and `lane_utils.*` in-place with donkey values.

    MUST be called BEFORE importing any of `models.dynamics_*`,
    `models.encoder_lane`, `models.decoder_lane`, `relative_coords`, or
    `train.train_piwm_lane_v5` — because those bind these names at import time.
    """
    import config, lane_utils

    s = load_donkey_stats()
    if s is not None:
        global DONKEY_PHYSICS_MEAN_REL, DONKEY_PHYSICS_STD_REL
        global DONKEY_LANE_MEAN, DONKEY_LANE_STD
        DONKEY_PHYSICS_MEAN_REL = s["physics_mean_rel"]
        DONKEY_PHYSICS_STD_REL  = s["physics_std_rel"]
        DONKEY_LANE_MEAN        = s["lane_mean"]
        DONKEY_LANE_STD         = s["lane_std"]

    config.DT               = DONKEY_DT
    config.PHYSICS_MEAN_REL = DONKEY_PHYSICS_MEAN_REL
    config.PHYSICS_STD_REL  = DONKEY_PHYSICS_STD_REL
    config.DATA_DIR         = DONKEY_DATA_DIR
    config.ENCODER_MEAN_REL = DONKEY_PHYSICS_MEAN_REL[[2, 3, 4, 5, 6, 7, 8, 9, 10]]
    config.ENCODER_STD_REL  = DONKEY_PHYSICS_STD_REL[[2, 3, 4, 5, 6, 7, 8, 9, 10]]

    lane_utils.LANE_S_SAMPLES = DONKEY_LANE_S_SAMPLES
    lane_utils.LANE_MEAN      = DONKEY_LANE_MEAN
    lane_utils.LANE_STD       = DONKEY_LANE_STD
