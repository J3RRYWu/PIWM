"""Profile where NN-baseline training time goes: image-laden DataLoader vs the
K=32 GPU rollout. Run from piwm/ with py311."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "train"))
import donkey_config; donkey_config.patch_globals()
import torch
from config import DEVICE, DATA_DIR
from donkey_dataset import make_donkey_loaders
from baselines.shared_dynamics_lane import DynamicsGOKULane

K, B = 32, 384
trl, _, _ = make_donkey_loaders(DATA_DIR, seq_len=K + 1, batch_size=B,
                                val_frac=0.10, flip_aug=True, seed=0)

# 1) pure data iteration (image collation included) — 15 batches
t = time.time(); nb = 0
for s0, fi, fp, fa, wn in trl:
    nb += 1
    if nb >= 15: break
t_data = (time.time() - t) / nb
print(f"[data]   {t_data*1000:.1f} ms/batch  (loads+collates s0{tuple(s0.shape)} fi{tuple(fi.shape)} -- both UNUSED by GOKU/DVBF)")

# 2) data + transfer of the 3 tensors actually used
t = time.time(); nb = 0
for s0, fi, fp, fa, wn in trl:
    fp = fp.to(DEVICE); fa = fa.to(DEVICE); wn = wn.to(DEVICE)
    nb += 1
    if nb >= 15: break
t_data2 = (time.time() - t) / nb
print(f"[data+xfer] {t_data2*1000:.1f} ms/batch")

# 3) rollout-only compute (fixed batch already on GPU), 15 reps
model = DynamicsGOKULane().to(DEVICE)
s0, fi, fp, fa, wn = next(iter(trl))
fp = fp.to(DEVICE); fa = fa.to(DEVICE); wn = wn.to(DEVICE)
torch.cuda.synchronize() if DEVICE.type == "cuda" else None
t = time.time(); reps = 15
for _ in range(reps):
    z = torch.cat([fp[:, 0], wn[:, 0]], dim=-1)
    loss = 0
    for k in range(K):
        z = model(z, fa[:, k])
        gt = torch.cat([fp[:, k + 1], wn[:, k + 1]], dim=-1)
        loss = loss + ((z - gt) ** 2).mean()
    loss.backward()
torch.cuda.synchronize() if DEVICE.type == "cuda" else None
t_roll = (time.time() - t) / reps
print(f"[rollout] {t_roll*1000:.1f} ms/batch  (K={K} steps fwd+bwd, GOKU {sum(p.numel() for p in model.parameters())} params)")
print(f"\n=> data is {t_data/t_roll:.1f}x the rollout compute. "
      f"Eliminating unused-image loading should cut ~{t_data*1000:.0f} ms/batch.")
