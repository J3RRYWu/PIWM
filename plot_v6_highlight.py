"""Clean v6-focused comparison plots.

Generates two plots:
  1. vis/v6_vs_piwm.png     — v6 vs PIWM family (linear / bicycle / v4 / v5)
  2. vis/v6_vs_baselines.png — v6 vs all non-PIWM baselines (SINDYc/GOKU/DVBF/V2P
                                and lane variants)

Both highlight v6 with thick purple line. Uses the data already computed in
compare_all_with_v5.py — we recompute here to keep it standalone.
"""
import os, glob, pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from compare_all_with_v5 import (
    load_episodes, load_shared_ae, patch_dyn_rel,
    eval_shared, eval_sindyc, eval_v5, eval_lane_baseline,
    MAX_STEPS, DEVICE, COLORS, LW, LS,
)
from config import ENCODER_DIM as CAR_ENCODER_DIM
from lane_utils import LANE_DIM, LANE_MEAN, LANE_STD
from models.encoder_lane import PhysicsEncoderLane
from models.decoder_lane import PhysicsDecoderLane
from models.dynamics import PhysicsDynamics
from models.dynamics_bicycle import BicycleDynamics
from models.dynamics_bicycle_v4 import BicycleDynamicsV4
from models.dynamics_lane_v5 import LaneAugmentedV5
from models.dynamics_lane_v6 import LaneAugmentedV6
from baselines.shared_dynamics import DynamicsDVBF, DynamicsGOKU, DynamicsVid2Param
from baselines.shared_dynamics_lane import DynamicsDVBFLane, DynamicsGOKULane, DynamicsVid2ParamLane
from utils import load_checkpoint


def plot_panels(state_results, image_results, save, title, ylim_state=None):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ['x_rel MSE', 'y_rel MSE', 'yaw_rel MSE', 'Image MSE']
    # 3 state panels
    for ax, title_s, idx in zip(axes[:3], titles[:3], [0, 1, 2]):
        for name, sm in state_results.items():
            steps, means = [], []
            for k in range(len(sm)):
                if len(sm[k]) >= 3:
                    steps.append(k); means.append(np.array(sm[k])[:, idx].mean())
            ax.plot(steps, means,
                    color=COLORS.get(name, 'gray'),
                    linewidth=LW.get(name, 2.0),
                    linestyle=LS.get(name, '-'),
                    label=name)
        ax.set_xlabel('Step'); ax.set_ylabel(title_s)
        ax.set_title(title_s); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
        # Autoscale via 95th percentile
        vals = []
        for sm in state_results.values():
            for k in range(len(sm)):
                if len(sm[k]) >= 3:
                    vals.append(np.array(sm[k])[:, idx].mean())
        if vals:
            y_top = np.percentile(vals, 95) * 1.3
            ax.set_ylim(0, max(y_top, 0.01))
    # Image panel
    ax = axes[3]
    for name, img in image_results.items():
        steps, means = [], []
        for k in range(len(img)):
            if len(img[k]) >= 3:
                steps.append(k); means.append(np.mean(img[k]))
        ax.plot(steps, means,
                color=COLORS.get(name, 'gray'),
                linewidth=LW.get(name, 2.0),
                linestyle=LS.get(name, '-'), label=name)
    ax.set_xlabel('Step'); ax.set_ylabel('Image MSE')
    ax.set_title('Image MSE'); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    ax.set_ylim(0, 0.03)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(); fig.savefig(save, dpi=150); plt.close(fig)
    print(f"Saved {save}")


if __name__ == "__main__":
    os.makedirs("vis", exist_ok=True)
    eps = load_episodes()
    enc, dec = load_shared_ae()
    v5_ae = load_checkpoint("checkpoints/piwm_lane_v5/best.tar")
    enc5 = PhysicsEncoderLane().to(DEVICE); enc5.load_state_dict(v5_ae['encoder']); enc5.eval()
    dec5 = PhysicsDecoderLane().to(DEVICE); dec5.load_state_dict(v5_ae['decoder']); dec5.eval()
    v6_ck = load_checkpoint("checkpoints/piwm_lane_v6/best.tar")
    enc6 = PhysicsEncoderLane().to(DEVICE); enc6.load_state_dict(v6_ck['encoder']); enc6.eval()
    dec6 = PhysicsDecoderLane().to(DEVICE); dec6.load_state_dict(v6_ck['decoder']); dec6.eval()

    all_state, all_img = {}, {}

    def _add_shared(name, dyn_fn, needs_theta=False, theta_fn=None):
        img, sm = eval_shared(eps, enc, dec, dyn_fn, needs_theta=needs_theta, theta_fn=theta_fn)
        all_state[name] = sm; all_img[name] = img

    def _add_lane(name, dyn, needs_theta=False, theta_fn=None, enc_use=None, dec_use=None):
        img, sm, _ = eval_lane_baseline(eps, enc_use or enc5, dec_use or dec5, dyn,
                                         needs_theta=needs_theta, theta_fn=theta_fn)
        all_state[name] = sm; all_img[name] = img

    # PIWM-linear
    print("[1] PIWM-linear")
    dyn = PhysicsDynamics().to(DEVICE)
    dyn.load_state_dict(load_checkpoint("checkpoints/piwm_v2_rel/best.tar")['dynamics'])
    patch_dyn_rel(dyn); dyn.eval()
    _add_shared('PIWM-linear', lambda z, a: dyn(z, a))

    # PIWM-bicycle v1
    print("[2] PIWM-bicycle")
    dyn_b = BicycleDynamics().to(DEVICE)
    dyn_b.load_state_dict(load_checkpoint("checkpoints/piwm_bike/best.tar")['dynamics'])
    dyn_b.eval()
    _add_shared('PIWM-bicycle', lambda z, a: dyn_b(z, a))

    # PIWM-v4
    print("[3] PIWM-bicycle-v4")
    dyn4 = BicycleDynamicsV4().to(DEVICE)
    dyn4.load_state_dict(load_checkpoint("checkpoints/piwm_bike_v4/best.tar")['dynamics'])
    dyn4.eval()
    _add_shared('PIWM-bicycle-v4', lambda z, a: dyn4(z, a))

    # SINDYc
    print("[4] SINDYc")
    img, sm = eval_sindyc(eps, enc, dec)
    all_state['SINDYc'] = sm; all_img['SINDYc'] = img

    # GOKU
    print("[5] GOKU")
    dyn_g = DynamicsGOKU().to(DEVICE)
    dyn_g.load_state_dict(load_checkpoint("checkpoints/shared_goku/best.tar")['model'])
    dyn_g.eval()
    _add_shared('GOKU-net', lambda z, a: dyn_g(z, a))

    # DVBF
    print("[6] DVBF")
    dyn_d = DynamicsDVBF().to(DEVICE)
    dyn_d.load_state_dict(load_checkpoint("checkpoints/shared_dvbf/best.tar")['model'])
    dyn_d.eval()
    _add_shared('DVBF', lambda z, a: dyn_d(z, a))

    # V2P
    print("[7] V2P")
    dyn_v = DynamicsVid2Param().to(DEVICE)
    dyn_v.load_state_dict(load_checkpoint("checkpoints/shared_v2p/best.tar")['model'])
    dyn_v.eval()
    _add_shared('Vid2Param',
                dyn_fn=lambda z, a, t: dyn_v.step(z, a, t),
                needs_theta=True, theta_fn=dyn_v.infer_theta)

    # v5
    print("[8] v5")
    dyn5 = LaneAugmentedV5().to(DEVICE)
    dyn5.load_state_dict(v5_ae['dynamics']); dyn5.eval()
    img, sm, _ = eval_v5(eps, enc5, dec5, dyn5)
    all_state['PIWM-lane-v5'] = sm; all_img['PIWM-lane-v5'] = img

    # v6
    print("[9] v6")
    dyn6 = LaneAugmentedV6().to(DEVICE)
    dyn6.load_state_dict(v6_ck['dynamics']); dyn6.eval()
    img, sm, _ = eval_v5(eps, enc6, dec6, dyn6)
    all_state['PIWM-lane-v6'] = sm; all_img['PIWM-lane-v6'] = img

    # Lane baselines
    for name, cls, ckpt_path, needs_theta in [
        ('DVBF-lane', DynamicsDVBFLane, "checkpoints/shared_dvbf_lane/best.tar", False),
        ('GOKU-lane', DynamicsGOKULane, "checkpoints/shared_goku_lane/best.tar", False),
        ('V2P-lane',  DynamicsVid2ParamLane, "checkpoints/shared_v2p_lane/best.tar", True),
    ]:
        print(f"[*] {name}")
        b = cls().to(DEVICE)
        b.load_state_dict(load_checkpoint(ckpt_path)['model']); b.eval()
        theta_fn = b.infer_theta if needs_theta else None
        img, sm, _ = eval_lane_baseline(eps, enc5, dec5, b,
                                         needs_theta=needs_theta, theta_fn=theta_fn)
        all_state[name] = sm; all_img[name] = img

    # -------- PLOT 1: v6 vs PIWM family --------
    keep = ['PIWM-linear', 'PIWM-bicycle', 'PIWM-bicycle-v4',
            'PIWM-lane-v5', 'PIWM-lane-v6']
    sub_state = {n: all_state[n] for n in keep if n in all_state}
    sub_img = {n: all_img[n] for n in keep if n in all_img}
    plot_panels(sub_state, sub_img, "vis/v6_vs_piwm.png",
                "PIWM-lane-v6 vs PIWM family")

    # -------- PLOT 2: v6 vs all non-PIWM baselines --------
    keep = ['PIWM-lane-v6',
            'SINDYc', 'GOKU-net', 'DVBF', 'Vid2Param',
            'DVBF-lane', 'GOKU-lane', 'V2P-lane']
    sub_state = {n: all_state[n] for n in keep if n in all_state}
    sub_img = {n: all_img[n] for n in keep if n in all_img}
    plot_panels(sub_state, sub_img, "vis/v6_vs_baselines.png",
                "PIWM-lane-v6 vs baselines (non-PIWM)")

    # -------- PLOT 3: v6 vs ALL 11 others (condensed) --------
    sub_state = all_state; sub_img = all_img
    plot_panels(sub_state, sub_img, "vis/v6_vs_all.png",
                "PIWM-lane-v6 vs everything")
