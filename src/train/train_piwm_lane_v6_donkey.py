"""Train PIWM-v6 on donkeycar data — same architecture, swapped constants.

Architecture is UNCHANGED. Only data-side constants are patched at import time:
  config.DT, config.PHYSICS_MEAN_REL, config.PHYSICS_STD_REL, config.DATA_DIR
  lane_utils.LANE_S_SAMPLES, lane_utils.LANE_MEAN, lane_utils.LANE_STD

This must happen BEFORE importing the model / dataset modules, which bind
those names at import time. Order is enforced below.

Three stages (same as v5/v6):
  Stage 1 — AE on lane-augmented encoder/decoder
  Stage 2 — Dynamics rollout
  Stage 3 — E2E fine-tune

Output: checkpoints/piwm_lane_v6_donkey/{ae.tar, dyn.tar, best.tar}
"""

# --- repo root on sys.path ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end shim ---

# CRITICAL: patch globals BEFORE importing models/dataset.
import donkey_config
donkey_config.patch_globals()

import os, argparse
import torch
from tqdm import tqdm

# Now-safe imports (config / lane_utils are already patched).
from config import (DEVICE, DATA_DIR, LAMBDA_RESIDUAL,
                    ENCODER_DIM as CAR_ENCODER_DIM)
from models.encoder_lane import PhysicsEncoderLane
from models.decoder_lane import PhysicsDecoderLane
# KINEMATIC variant — low-speed real donkey has no tire slip; V4's dynamic
# bicycle (slip angles, learnable mass/cornering stiffness) is the wrong
# physical prior and its parameter bounds are CarRacing-scale.
from models.dynamics_lane_v6_kin import LaneAugmentedV6Kin as LaneAugmentedV6
from utils import save_checkpoint, load_checkpoint
from donkey_dataset import make_donkey_loaders


SAVE_DIR = "checkpoints/piwm_lane_v6_donkey/"
ROLLOUT_K  = 8
AE_EPOCHS  = 40
DYN_EPOCHS = 60
E2E_EPOCHS = 20
# Front-view images can't be reconstructed faithfully from 29-dim latent
# (sky / walls / lighting are unrelated to lane geometry). Drop from CR's
# weight of 10 so the encoder doesn't waste capacity on non-lane content.
LAMBDA_IMG  = 1.0
LAMBDA_CAR  = 1.0
LAMBDA_LANE = 2.0


# ================================================================
# Stage 1: AE with car+lane supervision
# ================================================================
def stage1_ae():
    print("=" * 60); print("Stage 1 [v6/donkey]: AE training"); print("=" * 60)
    tr_loader, val_loader, _ = make_donkey_loaders(
        DATA_DIR, seq_len=ROLLOUT_K + 1, batch_size=64,
        val_frac=0.10, flip_aug=True)

    enc = PhysicsEncoderLane().to(DEVICE)
    dec = PhysicsDecoderLane().to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')

    for epoch in range(AE_EPOCHS):
        enc.train(); dec.train()
        ti, tc, tl, nb = 0, 0, 0, 0
        for stack0, fut_imgs, fut_phys, fut_acts, wp_norm in tqdm(
                tr_loader, desc=f"AE {epoch+1}/{AE_EPOCHS}"):
            stack0 = stack0.to(DEVICE); fut_imgs = fut_imgs.to(DEVICE)
            fut_phys = fut_phys.to(DEVICE); wp_norm = wp_norm.to(DEVICE)
            z_obs = enc(stack0)
            rec = dec(z_obs)
            gt_car  = fut_phys[:, 0, 2:11]
            gt_lane = wp_norm[:, 0]
            img_l  = ((rec - fut_imgs[:, 0]) ** 2).mean()
            car_l  = ((z_obs[:, :CAR_ENCODER_DIM]  - gt_car)  ** 2).mean()
            lane_l = ((z_obs[:, CAR_ENCODER_DIM:] - gt_lane) ** 2).mean()
            loss = LAMBDA_IMG * img_l + LAMBDA_CAR * car_l + LAMBDA_LANE * lane_l
            opt.zero_grad(); loss.backward(); opt.step()
            ti += img_l.item(); tc += car_l.item(); tl += lane_l.item(); nb += 1
        ti /= nb; tc /= nb; tl /= nb

        enc.eval(); dec.eval()
        vi, vc, vl, vn = 0, 0, 0, 0
        with torch.no_grad():
            for stack0, fut_imgs, fut_phys, fut_acts, wp_norm in val_loader:
                stack0 = stack0.to(DEVICE); fut_imgs = fut_imgs.to(DEVICE)
                fut_phys = fut_phys.to(DEVICE); wp_norm = wp_norm.to(DEVICE)
                z_obs = enc(stack0); rec = dec(z_obs)
                vi += ((rec - fut_imgs[:, 0]) ** 2).mean().item()
                vc += ((z_obs[:, :CAR_ENCODER_DIM] - fut_phys[:, 0, 2:11]) ** 2).mean().item()
                vl += ((z_obs[:, CAR_ENCODER_DIM:] - wp_norm[:, 0]) ** 2).mean().item()
                vn += 1
        vi /= vn; vc /= vn; vl /= vn
        vt = LAMBDA_IMG * vi + LAMBDA_CAR * vc + LAMBDA_LANE * vl
        sch.step(vt)
        print(f"  Train img={ti:.5f} car={tc:.5f} lane={tl:.5f}"
              f"  Val img={vi:.5f} car={vc:.5f} lane={vl:.5f}")
        if vt < best:
            best = vt
            save_checkpoint({'encoder': enc.state_dict(),
                             'decoder': dec.state_dict(),
                             'val_loss': vt},
                            os.path.join(SAVE_DIR, 'ae.tar'))
    print(f"AE best: {best:.5f}")


# ================================================================
# Stage 2: Dynamics rollout
# ================================================================
def stage2_dyn():
    print("\n" + "=" * 60); print("Stage 2 [v6/donkey]: Dynamics rollout"); print("=" * 60)
    tr_loader, val_loader, _ = make_donkey_loaders(
        DATA_DIR, seq_len=ROLLOUT_K + 1, batch_size=128,
        val_frac=0.10, flip_aug=True)

    dyn = LaneAugmentedV6().to(DEVICE)
    opt = torch.optim.Adam(dyn.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')

    for epoch in range(DYN_EPOCHS):
        dyn.train()
        tc, tl, nb = 0, 0, 0
        for stack0, fut_imgs, fut_phys, fut_acts, wp_norm in tqdm(
                tr_loader, desc=f"Dyn {epoch+1}/{DYN_EPOCHS}"):
            fut_phys = fut_phys.to(DEVICE); fut_acts = fut_acts.to(DEVICE)
            wp_norm = wp_norm.to(DEVICE)
            z = torch.cat([fut_phys[:, 0], wp_norm[:, 0]], dim=-1)
            ls_car, ls_lane, lres = 0, 0, 0
            for k in range(ROLLOUT_K):
                z_next, car_res = dyn(z, fut_acts[:, k])
                gt = torch.cat([fut_phys[:, k + 1], wp_norm[:, k + 1]], dim=-1)
                ls_car  = ls_car  + ((z_next[:, :11] - gt[:, :11]) ** 2).mean()
                ls_lane = ls_lane + ((z_next[:, 11:] - gt[:, 11:]) ** 2).mean()
                lres = lres + (car_res ** 2).mean()
                z = z_next
            ls_car /= ROLLOUT_K; ls_lane /= ROLLOUT_K; lres /= ROLLOUT_K
            loss = ls_car + LAMBDA_LANE * ls_lane + LAMBDA_RESIDUAL * lres
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(dyn.parameters(), 5.0)
            opt.step()
            tc += ls_car.item(); tl += ls_lane.item(); nb += 1
        tc /= nb; tl /= nb

        dyn.eval()
        vc, vl, vn = 0, 0, 0
        with torch.no_grad():
            for stack0, fut_imgs, fut_phys, fut_acts, wp_norm in val_loader:
                fut_phys = fut_phys.to(DEVICE); fut_acts = fut_acts.to(DEVICE)
                wp_norm = wp_norm.to(DEVICE)
                z = torch.cat([fut_phys[:, 0], wp_norm[:, 0]], dim=-1)
                ls_car, ls_lane = 0, 0
                for k in range(ROLLOUT_K):
                    z, _ = dyn(z, fut_acts[:, k])
                    gt = torch.cat([fut_phys[:, k + 1], wp_norm[:, k + 1]], dim=-1)
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
            save_checkpoint({'dynamics': dyn.state_dict(), 'val_loss': vt},
                            os.path.join(SAVE_DIR, 'dyn.tar'))
    print(f"Dyn best: {best:.5f}")


# ================================================================
# Stage 3: E2E fine-tune
# ================================================================
def stage3_e2e():
    print("\n" + "=" * 60); print("Stage 3 [v6/donkey]: E2E"); print("=" * 60)
    tr_loader, val_loader, _ = make_donkey_loaders(
        DATA_DIR, seq_len=ROLLOUT_K + 1, batch_size=32,
        val_frac=0.10, flip_aug=True)

    enc = PhysicsEncoderLane().to(DEVICE)
    dec = PhysicsDecoderLane().to(DEVICE)
    dyn = LaneAugmentedV6().to(DEVICE)
    ae_ck = load_checkpoint(os.path.join(SAVE_DIR, 'ae.tar'))
    enc.load_state_dict(ae_ck['encoder']); dec.load_state_dict(ae_ck['decoder'])
    dy_ck = load_checkpoint(os.path.join(SAVE_DIR, 'dyn.tar'))
    dyn.load_state_dict(dy_ck['dynamics'])

    params = list(enc.parameters()) + list(dec.parameters()) + list(dyn.parameters())
    opt = torch.optim.Adam(params, lr=3e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')

    for epoch in range(E2E_EPOCHS):
        enc.train(); dec.train(); dyn.train()
        ti, tc, tl, nb = 0, 0, 0, 0
        for stack0, fut_imgs, fut_phys, fut_acts, wp_norm in tqdm(
                tr_loader, desc=f"E2E {epoch+1}/{E2E_EPOCHS}"):
            stack0 = stack0.to(DEVICE); fut_imgs = fut_imgs.to(DEVICE)
            fut_phys = fut_phys.to(DEVICE); fut_acts = fut_acts.to(DEVICE)
            wp_norm = wp_norm.to(DEVICE)
            B = stack0.size(0)
            z_obs = enc(stack0)
            z = torch.zeros(B, 31, device=DEVICE)
            z[:, 0:2]  = fut_phys[:, 0, 0:2]
            z[:, 2:11] = z_obs[:, :CAR_ENCODER_DIM]
            z[:, 11:]  = z_obs[:, CAR_ENCODER_DIM:]
            img_l, car_l, lane_l = 0, 0, 0
            r0 = dec(torch.cat([z[:, 2:11], z[:, 11:]], dim=-1))
            img_l  += ((r0 - fut_imgs[:, 0]) ** 2).mean()
            car_l  += ((z[:, 2:11] - fut_phys[:, 0, 2:11]) ** 2).mean()
            lane_l += ((z[:, 11:]  - wp_norm[:, 0])         ** 2).mean()
            for k in range(ROLLOUT_K):
                z, _ = dyn(z, fut_acts[:, k])
                rec = dec(torch.cat([z[:, 2:11], z[:, 11:]], dim=-1))
                img_l  += ((rec - fut_imgs[:, k + 1]) ** 2).mean()
                car_l  += ((z[:, 2:11] - fut_phys[:, k + 1, 2:11]) ** 2).mean()
                lane_l += ((z[:, 11:]  - wp_norm[:, k + 1])        ** 2).mean()
            img_l  /= (ROLLOUT_K + 1); car_l /= (ROLLOUT_K + 1); lane_l /= (ROLLOUT_K + 1)
            loss = LAMBDA_IMG * img_l + LAMBDA_CAR * car_l + LAMBDA_LANE * lane_l
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            ti += img_l.item(); tc += car_l.item(); tl += lane_l.item(); nb += 1
        ti /= nb; tc /= nb; tl /= nb

        enc.eval(); dec.eval(); dyn.eval()
        vi, vc, vl, vn = 0, 0, 0, 0
        with torch.no_grad():
            for stack0, fut_imgs, fut_phys, fut_acts, wp_norm in val_loader:
                stack0 = stack0.to(DEVICE); fut_imgs = fut_imgs.to(DEVICE)
                fut_phys = fut_phys.to(DEVICE); fut_acts = fut_acts.to(DEVICE)
                wp_norm = wp_norm.to(DEVICE)
                B = stack0.size(0)
                z_obs = enc(stack0)
                z = torch.zeros(B, 31, device=DEVICE)
                z[:, 0:2]  = fut_phys[:, 0, 0:2]
                z[:, 2:11] = z_obs[:, :CAR_ENCODER_DIM]
                z[:, 11:]  = z_obs[:, CAR_ENCODER_DIM:]
                il, cl, ll = 0, 0, 0
                r0 = dec(torch.cat([z[:, 2:11], z[:, 11:]], dim=-1))
                il += ((r0 - fut_imgs[:, 0]) ** 2).mean().item()
                cl += ((z[:, 2:11] - fut_phys[:, 0, 2:11]) ** 2).mean().item()
                ll += ((z[:, 11:]  - wp_norm[:, 0]) ** 2).mean().item()
                for k in range(ROLLOUT_K):
                    z, _ = dyn(z, fut_acts[:, k])
                    rec = dec(torch.cat([z[:, 2:11], z[:, 11:]], dim=-1))
                    il += ((rec - fut_imgs[:, k + 1]) ** 2).mean().item()
                    cl += ((z[:, 2:11] - fut_phys[:, k + 1, 2:11]) ** 2).mean().item()
                    ll += ((z[:, 11:]  - wp_norm[:, k + 1]) ** 2).mean().item()
                il /= (ROLLOUT_K + 1); cl /= (ROLLOUT_K + 1); ll /= (ROLLOUT_K + 1)
                vi += il; vc += cl; vl += ll; vn += 1
        vi /= vn; vc /= vn; vl /= vn
        vt = LAMBDA_IMG * vi + LAMBDA_CAR * vc + LAMBDA_LANE * vl
        sch.step(vt)
        print(f"  Train img={ti:.5f} car={tc:.5f} lane={tl:.5f}"
              f"  Val img={vi:.5f} car={vc:.5f} lane={vl:.5f}")
        if vt < best:
            best = vt
            save_checkpoint({'encoder': enc.state_dict(),
                             'decoder': dec.state_dict(),
                             'dynamics': dyn.state_dict(),
                             'val_loss': vt},
                            os.path.join(SAVE_DIR, 'best.tar'))
    print(f"E2E best: {best:.5f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["ae", "dyn", "e2e", "all"])
    args = p.parse_args()
    os.makedirs(SAVE_DIR, exist_ok=True)
    if args.stage in ("ae",  "all"): stage1_ae()
    if args.stage in ("dyn", "all"): stage2_dyn()
    if args.stage in ("e2e", "all"): stage3_e2e()
