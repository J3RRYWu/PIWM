import os
import torch
import numpy as np
from config import PHYSICS_MEAN, PHYSICS_STD, DEVICE


def normalize_physics(state):
    """Normalize physics state using precomputed mean/std."""
    mean = torch.tensor(PHYSICS_MEAN, device=state.device, dtype=state.dtype)
    std = torch.tensor(PHYSICS_STD, device=state.device, dtype=state.dtype)
    return (state - mean) / std


def denormalize_physics(state_norm):
    """Denormalize physics state back to original scale."""
    mean = torch.tensor(PHYSICS_MEAN, device=state_norm.device, dtype=state_norm.dtype)
    std = torch.tensor(PHYSICS_STD, device=state_norm.device, dtype=state_norm.dtype)
    return state_norm * std + mean


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path):
    return torch.load(path, map_location=DEVICE)
