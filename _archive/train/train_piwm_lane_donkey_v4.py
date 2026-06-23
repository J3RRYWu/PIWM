"""Train PIWM-donkey-v4 (damped recurrence + bounded resample), with ablation
flags so v3-equivalent baselines run through the IDENTICAL code path / data /
seed / schedule — the only way to attribute any delta to Change 1/2/3.

Args:
  --K            rollout horizon (8 or 32)
  --omega-weight car-loss weight on omega dim (Change 3; 1.0=off, 2.0=on)
  --damp/--no-damp   Change 1 (default on)
  --hold/--no-hold   Change 2 (default on)
  --lr           Adam lr (default 1e-3; use 3e-4 for K=32)
  --grad-clip    grad-norm clip (default 5.0; use 1.0 for K=32)
  --warmstart    path to a dyn.tar to initialize from (curriculum for K=32)
  --epochs       override (default 60 for K=8, 40 for K=32)
  --tag          save-dir suffix

Save: checkpoints/piwm_lane_donkey_v4_<auto-tag>/dyn.tar
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import donkey_config; donkey_config.patch_globals()

import os, argparse
import torch
from tqdm import tqdm
from config import DEVICE, DATA_DIR
from models.dynamics_lane_donkey_v4 import LaneAugmentedDonkeyV4
from utils import save_checkpoint, load_checkpoint
from donkey_dataset import make_donkey_loaders


LAMBDA_LANE = 2.0


def run(K, omega_weight, damp, hold, lr, grad_clip, epochs, warmstart, tag):
    variant = ("v4" if (damp and hold) else
               f"d{int(damp)}h{int(hold)}")
    save_dir = f"checkpoints/piwm_lane_donkey_v4_{variant}_K{K}_w{omega_weight:g}{tag}/"
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 64)
    print(f"v4 trainer | variant={variant} damp={damp} hold={hold} | K={K} "
          f"omega_w={omega_weight} lr={lr} clip={grad_clip} epochs={epochs}")
    print(f"  warmstart={warmstart}")
    print(f"  save -> {save_dir}")
    print("=" * 64)

    tr_loader, val_loader, _ = make_donkey_loaders(
        DATA_DIR, seq_len=K + 1, batch_size=(128 if K <= 8 else 64),
        val_frac=0.10, flip_aug=True, seed=0)

    dyn = LaneAugmentedDonkeyV4(damp=damp, hold=hold).to(DEVICE)
    if warmstart:
        sd = load_checkpoint(warmstart)['dynamics']
        dyn.load_state_dict(sd)
        print(f"  warm-started from {warmstart}")
    print(f"  params: {sum(p.numel() for p in dyn.parameters()):,}")

    car_w = torch.ones(11, device=DEVICE)
    car_w[5] = float(omega_weight)        # omega only (NOT delta)

    opt = torch.optim.Adam(dyn.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float('inf')

    for epoch in range(epochs):
        dyn.train()
        tc, tl, nb = 0, 0, 0
        for stack0, fi, fp, fa, wn in tqdm(tr_loader, desc=f"{variant} K{K} {epoch+1}/{epochs}"):
            fp = fp.to(DEVICE); fa = fa.to(DEVICE); wn = wn.to(DEVICE)
            z = torch.cat([fp[:, 0], wn[:, 0]], dim=-1)
            ls_car, ls_lane = 0, 0
            for k in range(K):
                z_next, _ = dyn(z, fa[:, k])
                gt = torch.cat([fp[:, k + 1], wn[:, k + 1]], dim=-1)
                ls_car  = ls_car  + (car_w * (z_next[:, :11] - gt[:, :11]) ** 2).mean()
                ls_lane = ls_lane + ((z_next[:, 11:] - gt[:, 11:]) ** 2).mean()
                z = z_next
            ls_car /= K; ls_lane /= K
            loss = ls_car + LAMBDA_LANE * ls_lane
            if not torch.isfinite(loss):
                print("  !! non-finite loss, skipping batch")
                opt.zero_grad(); continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(dyn.parameters(), grad_clip)
            opt.step()
            tc += ls_car.item(); tl += ls_lane.item(); nb += 1
        tc /= max(nb, 1); tl /= max(nb, 1)

        dyn.eval()
        vc, vl, vn = 0, 0, 0
        with torch.no_grad():
            for stack0, fi, fp, fa, wn in val_loader:
                fp = fp.to(DEVICE); fa = fa.to(DEVICE); wn = wn.to(DEVICE)
                z = torch.cat([fp[:, 0], wn[:, 0]], dim=-1)
                ls_car, ls_lane = 0, 0
                for k in range(K):
                    z, _ = dyn(z, fa[:, k])
                    gt = torch.cat([fp[:, k + 1], wn[:, k + 1]], dim=-1)
                    ls_car  = ls_car  + ((z[:, :11] - gt[:, :11]) ** 2).mean()
                    ls_lane = ls_lane + ((z[:, 11:] - gt[:, 11:]) ** 2).mean()
                ls_car /= K; ls_lane /= K
                vc += ls_car.item(); vl += ls_lane.item(); vn += 1
        vc /= max(vn, 1); vl /= max(vn, 1)
        vt = vc + LAMBDA_LANE * vl
        sch.step(vt)
        g = torch.sigmoid(dyn.decay).detach().cpu().numpy()
        print(f"  Train car={tc:.5f} lane={tl:.5f}  Val car={vc:.5f} lane={vl:.5f} "
              f"comp={vt:.5f}  g=[{g[0]:.3f},{g[1]:.3f},{g[2]:.3f}]")
        if vt < best:
            best = vt
            save_checkpoint({'dynamics': dyn.state_dict(), 'val_loss': vt,
                             'val_car': vc, 'val_lane': vl, 'K': K,
                             'omega_weight': omega_weight, 'damp': damp, 'hold': hold},
                            os.path.join(save_dir, 'dyn.tar'))
    print(f"[{variant} K={K} w={omega_weight}] best comp: {best:.5f}")
    return best


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--omega-weight", type=float, default=1.0)
    p.add_argument("--damp", dest="damp", action="store_true", default=True)
    p.add_argument("--no-damp", dest="damp", action="store_false")
    p.add_argument("--hold", dest="hold", action="store_true", default=True)
    p.add_argument("--no-hold", dest="hold", action="store_false")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--warmstart", type=str, default="")
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()
    epochs = args.epochs if args.epochs > 0 else (60 if args.K <= 8 else 40)
    run(args.K, args.omega_weight, args.damp, args.hold, args.lr,
        args.grad_clip, epochs, args.warmstart, args.tag)
