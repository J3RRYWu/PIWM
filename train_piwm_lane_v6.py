"""Train PIWM-v6 (resample-based lane propagation).

Reuses v5's Stage 1 AE checkpoint (same encoder/decoder), only retrains:
  Stage 2: Dynamics rollout (v6 dynamics)
  Stage 3: E2E fine-tune

Output: checkpoints/piwm_lane_v6/{dyn.tar, best.tar}
"""
import os, argparse
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config import (DEVICE, DATA_DIR, LAMBDA_RESIDUAL,
                    ENCODER_DIM as CAR_ENCODER_DIM)
from lane_utils import LANE_DIM
from models.encoder_lane import PhysicsEncoderLane
from models.decoder_lane import PhysicsDecoderLane
from models.dynamics_lane_v6 import LaneAugmentedV6
from utils import save_checkpoint, load_checkpoint
from train_piwm_lane_v5 import SeqLaneDataset

SAVE_DIR = "checkpoints/piwm_lane_v6/"
V5_AE_PATH = "checkpoints/piwm_lane_v5/ae.tar"
ROLLOUT_K = 8
DYN_EPOCHS = 60
E2E_EPOCHS = 20
LAMBDA_IMG = 10.0
LAMBDA_CAR = 1.0
LAMBDA_LANE = 0.5


def stage2_dyn():
    print("=" * 60); print("Stage 2 [v6]: Resample-based lane dynamics rollout"); print("=" * 60)
    ds = SeqLaneDataset(DATA_DIR, seq_len=ROLLOUT_K + 1)
    n_val = int(len(ds) * 0.1)
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=128, shuffle=False, drop_last=True)

    dyn = LaneAugmentedV6().to(DEVICE)
    opt = torch.optim.Adam(dyn.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')

    for epoch in range(DYN_EPOCHS):
        dyn.train(); tc, tl_lane, nb = 0, 0, 0
        for stack0, fut_imgs, fut_phys, fut_acts, wp_norm in tqdm(
                tr_loader, desc=f"v6 Dyn {epoch+1}/{DYN_EPOCHS}"):
            fut_phys = fut_phys.to(DEVICE); fut_acts = fut_acts.to(DEVICE)
            wp_norm = wp_norm.to(DEVICE)
            z0 = torch.cat([fut_phys[:, 0], wp_norm[:, 0]], dim=-1)
            z = z0
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
            torch.nn.utils.clip_grad_norm_(dyn.parameters(), 5.0); opt.step()
            tc += ls_car.item(); tl_lane += ls_lane.item(); nb += 1
        tc /= nb; tl_lane /= nb

        dyn.eval(); vc, vl_lane, vn = 0, 0, 0
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
                vc += ls_car.item(); vl_lane += ls_lane.item(); vn += 1
        vc /= vn; vl_lane /= vn
        vt = vc + LAMBDA_LANE * vl_lane
        sch.step(vt)
        print(f"  Train car={tc:.5f} lane={tl_lane:.5f}  Val car={vc:.5f} lane={vl_lane:.5f}")
        if vt < best:
            best = vt
            save_checkpoint({'dynamics': dyn.state_dict(), 'val_loss': vt},
                            os.path.join(SAVE_DIR, 'dyn.tar'))
    print(f"v6 Dyn best: {best:.5f}")


def stage3_e2e():
    print("\n" + "=" * 60); print("Stage 3 [v6]: E2E fine-tune"); print("=" * 60)
    ds = SeqLaneDataset(DATA_DIR, seq_len=ROLLOUT_K + 1)
    n_val = int(len(ds) * 0.1)
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=32, shuffle=False, drop_last=True)

    enc = PhysicsEncoderLane().to(DEVICE)
    dec = PhysicsDecoderLane().to(DEVICE)
    dyn = LaneAugmentedV6().to(DEVICE)
    ae_ck = load_checkpoint(V5_AE_PATH)
    enc.load_state_dict(ae_ck['encoder']); dec.load_state_dict(ae_ck['decoder'])
    dy_ck = load_checkpoint(os.path.join(SAVE_DIR, 'dyn.tar'))
    dyn.load_state_dict(dy_ck['dynamics'])

    params = list(enc.parameters()) + list(dec.parameters()) + list(dyn.parameters())
    opt = torch.optim.Adam(params, lr=3e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')

    for epoch in range(E2E_EPOCHS):
        enc.train(); dec.train(); dyn.train()
        ti, tc, tl_lane, nb = 0, 0, 0, 0
        for stack0, fut_imgs, fut_phys, fut_acts, wp_norm in tqdm(
                tr_loader, desc=f"v6 E2E {epoch+1}/{E2E_EPOCHS}"):
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
            torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
            ti += img_l.item(); tc += car_l.item(); tl_lane += lane_l.item(); nb += 1
        ti /= nb; tc /= nb; tl_lane /= nb

        enc.eval(); dec.eval(); dyn.eval()
        vi, vc, vl_lane, vn = 0, 0, 0, 0
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
                vi += il; vc += cl; vl_lane += ll; vn += 1
        vi /= vn; vc /= vn; vl_lane /= vn
        vt = LAMBDA_IMG * vi + LAMBDA_CAR * vc + LAMBDA_LANE * vl_lane
        sch.step(vt)
        print(f"  Train img={ti:.5f} car={tc:.5f} lane={tl_lane:.5f}"
              f"  Val img={vi:.5f} car={vc:.5f} lane={vl_lane:.5f}")
        if vt < best:
            best = vt
            save_checkpoint({'encoder': enc.state_dict(),
                             'decoder': dec.state_dict(),
                             'dynamics': dyn.state_dict(),
                             'val_loss': vt},
                            os.path.join(SAVE_DIR, 'best.tar'))
    print(f"v6 E2E best: {best:.5f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["dyn", "e2e", "all"])
    args = p.parse_args()
    os.makedirs(SAVE_DIR, exist_ok=True)
    if args.stage in ("dyn", "all"): stage2_dyn()
    if args.stage in ("e2e", "all"): stage3_e2e()
