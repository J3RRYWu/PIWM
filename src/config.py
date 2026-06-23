import torch
import numpy as np

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
DATA_DIR = "../hsai-predictor-main/hsai-predictor-main/RacingCar/data/train/controller_5/"
IMG_SIZE = 64
FRAME_STACK = 3  # number of consecutive frames for encoder input

# Full state: 11 physical dimensions (used by dynamics)
# [x, y, yaw, vx, vy, omega, w0, w1, w2, w3, steer]
FULL_STATE_DIM = 11
FULL_STATE_NAMES = ["x", "y", "yaw", "vx", "vy", "omega", "w0", "w1", "w2", "w3", "steer"]

# Encoder output: 9 observable dimensions (no x, y — position is integrated by dynamics)
# [yaw, vx, vy, omega, w0, w1, w2, w3, steer]
ENCODER_DIM = 9
ENCODER_NAMES = ["yaw", "vx", "vy", "omega", "w0", "w1", "w2", "w3", "steer"]
# Indices into full 11-dim state for the 9 encoder dims
ENCODER_STATE_INDICES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# Indices for position (maintained by dynamics integration)
POSITION_INDICES = [0, 1]

# Normalization stats (computed from ~85K frames, controller_5 / trial_500)
# Full 11-dim: [x, y, yaw, vx, vy, omega, w0, w1, w2, w3, steer]
PHYSICS_MEAN = np.array([
    66.5948, 1.9087, 3.6736, -6.9395, 3.9824,
    0.3946, 106.5288, 107.9091, 113.9854, 115.4350, 0.0427
], dtype=np.float32)

PHYSICS_STD = np.array([
    113.6750, 71.3678, 2.8062, 49.0433, 38.5598,
    1.6314, 35.8798, 35.8862, 35.0840, 35.2595, 0.1685
], dtype=np.float32)

# Relative-coordinate normalization (initial pose at origin, initial yaw=0)
# Computed via relative_coords.compute_relative_stats with seq_len=100
# [x_rel, y_rel, yaw_rel, vx_rel, vy_rel, omega, w0, w1, w2, w3, steer]
PHYSICS_MEAN_REL = np.array([
    -10.87, 46.28, 0.40, -14.55, 40.24,
    0.39, 108.19, 109.59, 114.94, 116.42, 0.04
], dtype=np.float32)

PHYSICS_STD_REL = np.array([
    30.82, 34.28, 0.97, 34.49, 32.22,
    1.65, 35.69, 35.67, 35.57, 35.76, 0.17
], dtype=np.float32)

# Encoder-only (9-dim) normalization
ENCODER_MEAN = PHYSICS_MEAN[ENCODER_STATE_INDICES]
ENCODER_STD = PHYSICS_STD[ENCODER_STATE_INDICES]
ENCODER_MEAN_REL = PHYSICS_MEAN_REL[ENCODER_STATE_INDICES]
ENCODER_STD_REL = PHYSICS_STD_REL[ENCODER_STATE_INDICES]

# Switch to use relative coords
USE_RELATIVE = True

# Phase 1: Autoencoder training
AE_BATCH_SIZE = 64
AE_LR = 1e-3
AE_EPOCHS = 50
LAMBDA_PHYSICS = 10.0  # weight for physics supervision loss
AE_SAVE_DIR = "checkpoints/autoencoder/"

# Phase 2: Dynamics training
DYN_BATCH_SIZE = 128
DYN_LR = 1e-3
DYN_EPOCHS = 100
DT = 1.0 / 50.0  # simulation timestep (50 FPS)
LAMBDA_RESIDUAL = 0.01  # weight for residual regularization
DYN_SAVE_DIR = "checkpoints/dynamics/"
