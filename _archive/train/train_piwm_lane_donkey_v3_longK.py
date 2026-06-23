"""Retrain PIWM-v3 dynamics with a longer rollout horizon (K=32) so the model
is stable when extrapolated to 100-step rollouts. Same architecture, same
data, only the rollout length during stage-2 training changes.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import donkey_config; donkey_config.patch_globals()

import os
import torch
from tqdm import tqdm
from config import DEVICE, DATA_DIR
from models.dynamics_lane_donkey_v3 import LaneAugmentedDonkeyV3
from utils import save_checkpoint
from donkey_dataset import make_donkey_loaders


SAVE_DIR    = "checkpoints/piwm_lane_donkey_v3_longK/"
ROLLOUT_K   = 32                              # 4x longer training horizon
DYN_EPOCHS  = 40                              # fewer epochs (each iter 4x cost)
LAMBDA_LANE = 2.0


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"PIWM-v3 K={ROLLOUT_K} long-horizon retrain")
    tr_loader, val_loader, _ = make_donkey_loaders(
        DATA_DIR, seq_len=ROLLOUT_K + 1, batch_size=64,
        val_frac=0.10, flip_aug=True, seed=0)

    dyn = LaneAugmentedDonkeyV3().to(DEVICE)
    n_params = sum(p.numel() for p in dyn.parameters())
    print(f"  params: {n_params:,}  K={ROLLOUT_K}")
    opt = torch.optim.Adam(dyn.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')

    for epoch in range(DYN_EPOCHS):
        dyn.train()
        tc, tl, nb = 0, 0, 0
        for stack0, fi, fp, fa, wn in tqdm(tr_loader, desc=f"Dyn {epoch+1}/{DYN_EPOCHS}"):
            fp = fp.to(DEVICE); fa = fa.to(DEVICE); wn = wn.to(DEVICE)
            z = torch.cat([fp[:, 0], wn[:, 0]], dim=-1)
            ls_car, ls_lane = 0, 0
            for k in range(ROLLOUT_K):
                z_next, _ = dyn(z, fa[:, k])
                gt = torch.cat([fp[:, k + 1], wn[:, k + 1]], dim=-1)
                ls_car  = ls_car  + ((z_next[:, :11] - gt[:, :11]) ** 2).mean()
                ls_lane = ls_lane + ((z_next[:, 11:] - gt[:, 11:]) ** 2).mean()
                z = z_next
            ls_car /= ROLLOUT_K; ls_lane /= ROLLOUT_K
            loss = ls_car + LAMBDA_LANE * ls_lane
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(dyn.parameters(), 5.0)
            opt.step()
            tc += ls_car.item(); tl += ls_lane.item(); nb += 1
        tc /= nb; tl /= nb

        dyn.eval()
        vc, vl, vn = 0, 0, 0
        with torch.no_grad():
            for stack0, fi, fp, fa, wn in val_loader:
                fp = fp.to(DEVICE); fa = fa.to(DEVICE); wn = wn.to(DEVICE)
                z = torch.cat([fp[:, 0], wn[:, 0]], dim=-1)
                ls_car, ls_lane = 0, 0
                for k in range(ROLLOUT_K):
                    z, _ = dyn(z, fa[:, k])
                    gt = torch.cat([fp[:, k + 1], wn[:, k + 1]], dim=-1)
                    ls_car  = ls_car  + ((z[:, :11] - gt[:, :11]) ** 2).mean()
                    ls_lane = ls_lane + ((z[:, 11:] - gt[:, 11:]) ** 2).mean()
                ls_car /= ROLLOUT_K; ls_lane /= ROLLOUT_K
                vc += ls_car.item(); vl += ls_lane.item(); vn += 1
        vc /= vn; vl /= vn
        vt = vc + LAMBDA_LANE * vl
        sch.step(vt)
        print(f"  Train car={tc:.5f} lane={tl:.5f}  Val car={vc:.5f} lane={vl:.5f}")
        if vt < best:
            best = vt
            save_checkpoint({'dynamics': dyn.state_dict(), 'val_loss': vt,
                             'val_car': vc, 'val_lane': vl},
                            os.path.join(SAVE_DIR, 'dyn.tar'))
    print(f"PIWM-v3 K=32 best: {best:.5f}")


if __name__ == "__main__":
    main()
