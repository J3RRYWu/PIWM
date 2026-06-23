import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from config import (
    PHYSICS_MEAN, PHYSICS_STD, FRAME_STACK,
    ENCODER_MEAN, ENCODER_STD, ENCODER_STATE_INDICES,
)


class RacingCarDataset(Dataset):
    """Dataset for PIWM training.

    Modes:
    - "autoencoder": returns (frames, encoder_physics_9d, target_image)
      encoder_physics is 9-dim (no x,y) for encoder supervision
    - "dynamics": returns (full_state_11d_t, action_t, full_state_11d_t+1)
      full state includes x,y for dynamics integration
    """

    def __init__(self, data_dir, mode="autoencoder"):
        self.mode = mode

        npz_files = sorted(glob.glob(f"{data_dir}/*.npz"))

        all_imgs = []
        all_physics = []  # full 11-dim
        all_actions = []
        episode_lengths = []

        for f in npz_files:
            try:
                d = np.load(f, allow_pickle=True)
                imgs = d["imgs"].astype(np.float32)
                if imgs.max() > 1.0:
                    imgs = imgs / 255.0

                n = len(imgs)
                if n < FRAME_STACK + 1:
                    continue

                pos = d["position"].astype(np.float32)
                yaw = d["yaw"].astype(np.float32)
                vel = d["velocity"].astype(np.float32)
                omega = d["angular_velocity"].astype(np.float32)
                wheel = d["wheel_omega"].astype(np.float32)
                steer = d["steering_angle"].astype(np.float32)

                # Full 11-dim: [x, y, yaw, vx, vy, omega, w0, w1, w2, w3, steer]
                physics = np.column_stack([pos, yaw, vel, omega, wheel, steer])
                actions = d["action"].astype(np.float32)

                all_imgs.append(imgs)
                all_physics.append(physics)
                all_actions.append(actions)
                episode_lengths.append(n)
            except Exception as e:
                print(f"Skipping {f}: {e}")

        # Build valid indices
        self.indices = []
        for ep_idx, ep_len in enumerate(episode_lengths):
            max_frame = ep_len - 1 if mode == "dynamics" else ep_len
            for t in range(FRAME_STACK - 1, max_frame - 1):
                self.indices.append((ep_idx, t))

        self.all_imgs = all_imgs
        self.all_physics = all_physics
        self.all_actions = all_actions

        print(f"Loaded {len(episode_lengths)} episodes, {sum(episode_lengths)} frames, {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, t = self.indices[idx]

        if self.mode == "autoencoder":
            # Stack FRAME_STACK consecutive frames ending at t
            frames = np.stack([
                self.all_imgs[ep_idx][t - FRAME_STACK + 1 + i]
                for i in range(FRAME_STACK)
            ], axis=0)  # (3, 64, 64)

            current_img = self.all_imgs[ep_idx][t]  # (64, 64)

            # 9-dim encoder target (no x, y)
            full_state = self.all_physics[ep_idx][t]
            encoder_state = full_state[ENCODER_STATE_INDICES]
            encoder_state_norm = (encoder_state - ENCODER_MEAN) / ENCODER_STD

            return (
                torch.tensor(frames, dtype=torch.float32),
                torch.tensor(encoder_state_norm, dtype=torch.float32),
                torch.tensor(current_img, dtype=torch.float32).unsqueeze(0),  # (1, 64, 64)
            )

        elif self.mode == "dynamics":
            # Full 11-dim normalized state for dynamics
            state_t = self.all_physics[ep_idx][t]
            state_t1 = self.all_physics[ep_idx][t + 1]
            state_t_norm = (state_t - PHYSICS_MEAN) / PHYSICS_STD
            state_t1_norm = (state_t1 - PHYSICS_MEAN) / PHYSICS_STD
            action_t = self.all_actions[ep_idx][t]

            return (
                torch.tensor(state_t_norm, dtype=torch.float32),
                torch.tensor(action_t, dtype=torch.float32),
                torch.tensor(state_t1_norm, dtype=torch.float32),
            )
