"""Retrain ALL 5 models (PIWM-v3 + 4 baselines) with K=32 rollout horizon
on donkey. For a fair long-horizon (100-step) comparison every method needs
the SAME training schedule. SINDYc is unchanged (no rollout horizon).

Time on CPU: each model ~30-45 min × 4 = ~3 hours total.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import donkey_config; donkey_config.patch_globals()

import os, argparse, time
import torch
from tqdm import tqdm
from config import DEVICE, DATA_DIR, ENCODER_DIM as CAR_ENCODER_DIM
from models.dynamics_lane_donkey_v3 import LaneAugmentedDonkeyV3
from baselines.shared_dynamics_lane import (DynamicsDVBFLane, DynamicsGOKULane,
                                            DynamicsVid2ParamLane)
from models.encoder_lane import PhysicsEncoderLane
from utils import save_checkpoint, load_checkpoint
from donkey_dataset import make_donkey_loaders


ROLLOUT_K   = 32
DYN_EPOCHS  = 40
BATCH       = 64
LR          = 1e-3
LAMBDA_LANE = 2.0
GRAD_CLIP   = 5.0
SEED        = 0
ENC_CK = "checkpoints/piwm_lane_v6_donkey/ae.tar"


def _train_loop(model, tr_loader, val_loader, name, save_path,
                theta_provider=None):
    print("=" * 60); print(f"Train {name} (K={ROLLOUT_K})"); print("=" * 60)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')
    t0 = time.time()
    for epoch in range(DYN_EPOCHS):
        model.train()
        tc, tl, nb = 0, 0, 0
        for stack0, fi, fp, fa, wn in tqdm(tr_loader, desc=f"{name} {epoch+1}/{DYN_EPOCHS}"):
            stack0 = stack0.to(DEVICE); fp = fp.to(DEVICE); fa = fa.to(DEVICE); wn = wn.to(DEVICE)
            theta = None
            if theta_provider is not None:
                theta = theta_provider(stack0)
            z = torch.cat([fp[:, 0], wn[:, 0]], dim=-1)
            ls_car, ls_lane = 0, 0
            for k in range(ROLLOUT_K):
                if theta is not None:
                    z = model.step(z, fa[:, k], theta)
                else:
                    out = model(z, fa[:, k])
                    z = out[0] if isinstance(out, tuple) else out
                gt = torch.cat([fp[:, k + 1], wn[:, k + 1]], dim=-1)
                ls_car  = ls_car  + ((z[:, :11] - gt[:, :11]) ** 2).mean()
                ls_lane = ls_lane + ((z[:, 11:] - gt[:, 11:]) ** 2).mean()
            ls_car /= ROLLOUT_K; ls_lane /= ROLLOUT_K
            loss = ls_car + LAMBDA_LANE * ls_lane
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tc += ls_car.item(); tl += ls_lane.item(); nb += 1
        tc /= nb; tl /= nb

        model.eval()
        vc, vl, vn = 0, 0, 0
        with torch.no_grad():
            for stack0, fi, fp, fa, wn in val_loader:
                stack0 = stack0.to(DEVICE); fp = fp.to(DEVICE); fa = fa.to(DEVICE); wn = wn.to(DEVICE)
                theta = None
                if theta_provider is not None:
                    theta = theta_provider(stack0)
                z = torch.cat([fp[:, 0], wn[:, 0]], dim=-1)
                ls_car, ls_lane = 0, 0
                for k in range(ROLLOUT_K):
                    if theta is not None:
                        z = model.step(z, fa[:, k], theta)
                    else:
                        out = model(z, fa[:, k])
                        z = out[0] if isinstance(out, tuple) else out
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
            ck = {'val_loss': vt, 'val_car': vc, 'val_lane': vl}
            # Save under 'model' (baselines) AND 'dynamics' (PIWM) for compat.
            ck['model'] = model.state_dict()
            ck['dynamics'] = model.state_dict()
            save_checkpoint(ck, save_path)
    print(f"{name} best: {best:.5f}  ({time.time()-t0:.0f}s)")


def main():
    tr_loader, val_loader, _ = make_donkey_loaders(
        DATA_DIR, seq_len=ROLLOUT_K + 1, batch_size=BATCH,
        val_frac=0.10, flip_aug=True, seed=SEED)

    # PIWM-v3
    os.makedirs("checkpoints/piwm_lane_donkey_v3_longK", exist_ok=True)
    _train_loop(LaneAugmentedDonkeyV3().to(DEVICE), tr_loader, val_loader,
                "PIWM-v3-K32", "checkpoints/piwm_lane_donkey_v3_longK/dyn.tar")

    # DVBF
    os.makedirs("checkpoints/dvbf_lane_donkey_longK", exist_ok=True)
    _train_loop(DynamicsDVBFLane().to(DEVICE), tr_loader, val_loader,
                "DVBF-K32", "checkpoints/dvbf_lane_donkey_longK/best.tar")

    # GOKU
    os.makedirs("checkpoints/goku_lane_donkey_longK", exist_ok=True)
    _train_loop(DynamicsGOKULane().to(DEVICE), tr_loader, val_loader,
                "GOKU-K32", "checkpoints/goku_lane_donkey_longK/best.tar")

    # V2P (needs frozen encoder)
    enc = PhysicsEncoderLane().to(DEVICE)
    enc.load_state_dict(load_checkpoint(ENC_CK)['encoder']); enc.eval()
    for p in enc.parameters(): p.requires_grad = False

    def theta_provider(stack0):
        with torch.no_grad():
            obs = enc(stack0).unsqueeze(1)
        v2p_model = theta_provider.model
        theta, _, _ = v2p_model.infer_theta(obs)
        return theta

    v2p_model = DynamicsVid2ParamLane().to(DEVICE)
    theta_provider.model = v2p_model
    os.makedirs("checkpoints/v2p_lane_donkey_longK", exist_ok=True)
    _train_loop(v2p_model, tr_loader, val_loader, "V2P-K32",
                "checkpoints/v2p_lane_donkey_longK/best.tar",
                theta_provider=theta_provider)


if __name__ == "__main__":
    main()
