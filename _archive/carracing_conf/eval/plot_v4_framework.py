"""Full framework diagram for PIWM-bicycle-v4.

Shows the COMPLETE pipeline (no ellipsis):
  Image stack (3 frames)   ->  PhysicsEncoder (CNN)  ->  9-dim observable
  Initial (x_0, y_0) = (0, 0) in relative-body frame
  Assemble 11-dim z_0
  For each rollout step:
    z_t + a_t -> denormalize -> body frame -> slip angles -> tire forces
              -> Newton's 2nd law -> Euler integrate -> body->world + position
              -> wheel/steer update -> masked residual -> normalize -> z_{t+1}
  PhysicsDecoder (optional): z_t[2:11] -> reconstructed image

Output: vis/v4_framework.png
"""

# --- auto-added: make repo root importable when run as a script ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end shim ---


import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

COLOR_IMG    = '#c7dcff'   # image / decoder output
COLOR_CNN    = '#d6c7ff'   # encoder (CNN)
COLOR_INIT   = '#ffe6c7'   # initial state construction
COLOR_XFORM  = '#fff4e0'   # denorm / frame transform
COLOR_PHYS   = '#d4f5d4'   # analytical physics (bicycle)
COLOR_LEARN  = '#ffe0e0'   # learned MLPs
COLOR_PARAM  = '#ffd6c7'   # physical params
COLOR_RES    = '#f0e0ff'   # residual (masked)
COLOR_STATE  = '#e1f5ff'   # state tensor

EDGE = '#333'


def rbox(ax, x, y, w, h, label, color, fs=9, weight='normal'):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.2, edgecolor=EDGE, facecolor=color, zorder=2,
    ))
    ax.text(x + w / 2, y + h / 2, label,
            ha='center', va='center', fontsize=fs, color='#000', weight=weight, zorder=3)


def arr(ax, x0, y0, x1, y1, txt=None, rad=0.0, color='#555', fs=7.5, style='-|>',
        offset=(0, 0), mut=12):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, mutation_scale=mut,
        connectionstyle=f"arc3,rad={rad}",
        linewidth=1.2, color=color, zorder=1,
    ))
    if txt:
        ax.text((x0 + x1) / 2 + offset[0], (y0 + y1) / 2 + offset[1], txt,
                ha='center', va='center', fontsize=fs, color=color, style='italic',
                bbox=dict(boxstyle='round,pad=0.14', fc='white', ec='none', alpha=0.85),
                zorder=3)


def plot():
    fig, ax = plt.subplots(figsize=(19, 11))
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 11.5)
    ax.axis('off')

    # ---- Title ----
    ax.text(9.5, 11.0, 'PIWM-bicycle-v4: Full Framework',
            ha='center', fontsize=17, weight='bold')
    ax.text(9.5, 10.6,
            'from raw image stack (3 frames) to dynamic bicycle rollout',
            ha='center', fontsize=11.5, color='#555', style='italic')

    # ================================================================
    # STAGE A: Perception (image -> 9-dim observable via PhysicsEncoder)
    # ================================================================
    ax.text(2.3, 10.0, 'A. Perception (one-time, t = 0)',
            fontsize=11.5, weight='bold', color='#0050aa')

    # 3-frame image stack
    rbox(ax, 0.2, 8.5, 1.9, 1.2,
         r'3-frame image stack' + '\n' + r'$I_{t-2}, I_{t-1}, I_t$' + '\n' +
         r'$(3 \times 64 \times 64)$',
         COLOR_IMG, fs=9)

    # PhysicsEncoder
    rbox(ax, 2.5, 8.5, 1.9, 1.2,
         r'PhysicsEncoder' + '\n' + r'(4-layer CNN + MLP)' + '\n' +
         r'frozen / fine-tuned',
         COLOR_CNN, fs=9, weight='bold')
    arr(ax, 2.1, 9.1, 2.5, 9.1)

    # 9-dim observable (what encoder outputs)
    rbox(ax, 4.8, 8.5, 2.6, 1.2,
         r'$z_{\mathrm{obs}} \in \mathbb{R}^9$' + '\n' +
         r'$[\psi, v_x, v_y, \omega,$' + '\n' + r'$\; w_0, w_1, w_2, w_3, \delta]$',
         COLOR_STATE, fs=9)
    arr(ax, 4.4, 9.1, 4.8, 9.1, 'observable physics (no x, y)',
        offset=(0, 0.4), fs=7.5)

    # Initial (x, y) from relative-frame definition
    rbox(ax, 4.8, 7.1, 2.6, 1.0,
         r'$(x_0, y_0) = (0, 0)$' + '\n' +
         r'by definition of body-centric' + '\n' +
         r'relative coordinates',
         COLOR_INIT, fs=8.5)

    # Merge into 11-dim z_0
    rbox(ax, 7.9, 7.8, 2.6, 1.8,
         r'$z_0 \in \mathbb{R}^{11}$' + '\n\n' +
         r'$x_0, y_0$   (= 0, 0)' + '\n' +
         r'$\psi, v_x, v_y, \omega$' + '\n' +
         r'$w_{0..3}, \delta$  (encoder)',
         COLOR_STATE, fs=9, weight='bold')
    arr(ax, 7.4, 9.1, 7.9, 9.1)   # from z_obs
    arr(ax, 7.4, 7.6, 7.9, 8.2)   # from (x0, y0)

    # Normalize
    rbox(ax, 10.9, 7.9, 2.2, 1.6,
         r'Normalize' + '\n' + r'$z_0^{\mathrm{norm}} = \dfrac{z_0 - \mu}{\sigma}$',
         COLOR_XFORM, fs=9.5)
    arr(ax, 10.5, 8.7, 10.9, 8.7)

    # Action input (for rollout)
    rbox(ax, 13.5, 8.5, 2.0, 1.1,
         r'Action' + '\n' + r'$a_t \in \mathbb{R}^3$' + '\n' +
         r'$[\delta_{\mathrm{cmd}},\mathrm{gas},\mathrm{brake}]$',
         COLOR_STATE, fs=9)

    # ================================================================
    # STAGE B: Dynamic bicycle step (per rollout step k -> k+1)
    # ================================================================
    ax.text(3.0, 6.4, 'B. Dynamic bicycle step  (repeat for $k = 0,\\dots,K{-}1$)',
            fontsize=11.5, weight='bold', color='#0050aa')

    # z_k^norm + a_k as inputs
    rbox(ax, 0.2, 5.1, 2.0, 0.9,
         r'$z_k^{\mathrm{norm}},\; a_k$' + '\n' + r'(step input)',
         COLOR_STATE, fs=9, weight='bold')

    # Denormalize
    rbox(ax, 2.7, 5.1, 1.8, 0.9,
         r'Denormalize' + '\n' + r'$z_k = \sigma z_k^{\mathrm{norm}} + \mu$',
         COLOR_XFORM, fs=9)
    arr(ax, 2.2, 5.55, 2.7, 5.55)

    # World -> Body frame
    rbox(ax, 5.0, 5.1, 2.4, 0.9,
         r'World $\to$ Body frame' + '\n' +
         r'$v_x^b{=}v_x c + v_y s,\; v_y^b{=}-v_x s + v_y c$',
         COLOR_XFORM, fs=8)
    arr(ax, 4.5, 5.55, 5.0, 5.55)

    # Slip angles
    rbox(ax, 7.8, 5.05, 3.0, 1.0,
         r'Slip angles' + '\n' +
         r'$\alpha_f = \delta - \arctan\frac{v_y^b + L_f \omega}{v_x^b}$' + '\n' +
         r'$\alpha_r = -\arctan\frac{v_y^b - L_r \omega}{v_x^b}$',
         COLOR_PHYS, fs=8)
    arr(ax, 7.4, 5.55, 7.8, 5.55)

    # Tire forces
    rbox(ax, 11.2, 5.1, 2.6, 0.9,
         r'Tire forces' + '\n' +
         r'$F_{yf} = -C_f \alpha_f,\; F_{yr} = -C_r \alpha_r$',
         COLOR_PHYS, fs=8.5)
    arr(ax, 10.8, 5.55, 11.2, 5.55)

    # Learned physical params box
    rbox(ax, 7.8, 3.8, 6.0, 0.9,
         r'Learnable bounded physical params:   '
         r'$m{=}1680\,\mathrm{kg}$,  $I_z{=}4338\,\mathrm{kg\cdot m^2}$,  '
         r'$C_f{=}237\mathrm{k},\; C_r{=}233\mathrm{k}$,  '
         r'$L_f{=}2.28,\; L_r{=}2.22$',
         COLOR_PARAM, fs=8.5, weight='bold')
    arr(ax, 9.3, 4.7, 9.3, 5.05, color='#aa5555')
    arr(ax, 12.5, 4.7, 12.5, 5.1, color='#aa5555')

    # Newton 2nd law: body-frame accels
    rbox(ax, 14.1, 4.8, 4.6, 1.6,
         r"Newton's 2nd law (body frame)" + '\n\n' +
         r'$\dot v_x^b = a_{\mathrm{fwd}} - \frac{F_{yf}\sin\delta}{m} + v_y^b\omega$' + '\n' +
         r'$\dot v_y^b = \frac{F_{yf}\cos\delta + F_{yr}}{m} - v_x^b\omega$' + '\n' +
         r'$\dot\omega = \frac{L_f F_{yf}\cos\delta - L_r F_{yr}}{I_z}$',
         COLOR_PHYS, fs=8.5)
    arr(ax, 13.8, 5.55, 14.1, 5.6)

    # accel_net (learned a_fwd)
    rbox(ax, 14.1, 3.5, 2.4, 0.9,
         r'$a_{\mathrm{fwd}} = $ accel_net$([z, a])$',
         COLOR_LEARN, fs=9)
    arr(ax, 15.3, 4.4, 15.3, 4.8, color='#aa5555')

    # action feed to accel_net
    arr(ax, 1.2, 5.1, 1.2, 4.0, color='#888')
    arr(ax, 1.2, 4.0, 14.1, 4.0, color='#888',
        txt='input $[z_k, a_k]$ to learned nets', offset=(0.0, 0.16), fs=7.5)

    # Euler integrate
    rbox(ax, 14.1, 2.1, 4.6, 1.2,
         r'Euler integrate  ($\Delta t = 1/50$ s)' + '\n' +
         r'$v_{x,\mathrm{next}}^b = v_x^b + \dot v_x^b \Delta t$' + '\n' +
         r'$\omega_{\mathrm{next}} = \omega + \dot\omega \Delta t$',
         COLOR_PHYS, fs=8)
    arr(ax, 16.4, 4.8, 16.4, 3.3)

    # Body -> World + Position
    rbox(ax, 14.1, 0.7, 4.6, 1.1,
         r'Body $\to$ World  +  Position' + '\n' +
         r'$v_{x,\mathrm{next}} = v_{x,\mathrm{next}}^b c - v_{y,\mathrm{next}}^b s$' + '\n' +
         r'$x_{\mathrm{next}}{=}x + v_x \Delta t,\; \psi_{\mathrm{next}}{=}\psi + \omega \Delta t$',
         COLOR_PHYS, fs=8)
    arr(ax, 16.4, 2.1, 16.4, 1.8)

    # wheel_net
    rbox(ax, 10.5, 2.1, 3.0, 0.9,
         r'wheel_net$([z,a])$:  $\Delta w_{0..3}, \Delta\delta$',
         COLOR_LEARN, fs=8.5)
    arr(ax, 13.5, 2.55, 14.1, 2.6, rad=-0.05)
    arr(ax, 13.5, 2.55, 14.1, 1.3, rad=-0.12, color='#888')

    # residual_net (masked)
    rbox(ax, 10.5, 0.6, 3.0, 1.0,
         r'residual_net$([z,a]) \in \mathbb{R}^{11}$' + '\n' +
         r'with mask:  $r[0] = r[1] = 0$' + '\n' +
         r'(no $x,y$ modification)',
         COLOR_RES, fs=8.5)
    arr(ax, 13.5, 1.1, 14.1, 1.1, color='#8844aa')

    # z_{k+1} output + re-normalize
    rbox(ax, 7.2, 0.4, 2.8, 1.3,
         r'$z_{k+1} = z_{\mathrm{phys}} + r_{\mathrm{masked}}$' + '\n' +
         r'then  $z_{k+1}^{\mathrm{norm}} = \frac{z_{k+1} - \mu}{\sigma}$',
         COLOR_XFORM, fs=9, weight='bold')
    arr(ax, 14.1, 1.1, 10.0, 1.1, rad=-0.1)
    arr(ax, 10.5, 1.1, 10.0, 1.1, color='#8844aa')   # residual into assemble

    # Recurrence back to step input
    arr(ax, 7.2, 1.1, 1.2, 1.1, rad=-0.18, color='#0077bb')
    arr(ax, 1.2, 1.1, 1.2, 5.1, rad=0.0, color='#0077bb',
        txt='next step  $k \\to k+1$', offset=(0.7, 0.0), fs=8)

    # ================================================================
    # STAGE C: Image reconstruction (optional, during training)
    # ================================================================
    ax.text(16.8, 10.0, 'C. Reconstruction', fontsize=11.5, weight='bold', color='#0050aa')

    rbox(ax, 16.3, 8.5, 2.5, 1.2,
         r'PhysicsDecoder' + '\n' + r'(MLP + Deconv)' + '\n' +
         r'on $z_t[2:11]$',
         COLOR_CNN, fs=9)
    rbox(ax, 16.3, 7.1, 2.5, 1.2,
         r'reconstructed' + '\n' + r'$\hat I_t \in \mathbb{R}^{64 \times 64}$',
         COLOR_IMG, fs=9)
    arr(ax, 17.55, 8.5, 17.55, 8.3)
    # From z_0^norm (at the end of stage A) OR from any z_k during rollout
    arr(ax, 13.1, 8.7, 16.3, 9.1, rad=0.08, color='#0050aa',
        txt='$z_0^{\\mathrm{norm}}$ obs dims', offset=(-0.3, 0.45), fs=7.5)
    arr(ax, 10.0, 1.0, 16.3, 8.5, rad=-0.35, color='#0050aa',
        txt='any $z_k^{\\mathrm{norm}}$ during rollout', offset=(1.5, -0.2), fs=7.5)

    # ================================================================
    # LEGEND
    # ================================================================
    legend_x, legend_y = 0.2, 3.2
    items = [
        (COLOR_IMG,   'Image (raw / reconstructed)'),
        (COLOR_CNN,   'CNN-based (encoder / decoder)'),
        (COLOR_INIT,  'Initial condition (relative coords)'),
        (COLOR_STATE, 'State / action tensors'),
        (COLOR_XFORM, 'Normalization / frame transform'),
        (COLOR_PHYS,  'Analytical physics (bicycle)'),
        (COLOR_LEARN, 'Learned MLPs'),
        (COLOR_PARAM, 'Learned physical params (bounded)'),
        (COLOR_RES,   'Residual correction (masked)'),
    ]
    for i, (c, lbl) in enumerate(items):
        y = legend_y - 0.30 * i
        ax.add_patch(Rectangle((legend_x, y), 0.28, 0.20, facecolor=c, edgecolor=EDGE, lw=1))
        ax.text(legend_x + 0.37, y + 0.10, lbl, fontsize=8.5, va='center')

    plt.tight_layout()
    out = "vis/v4_framework.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    os.makedirs("vis", exist_ok=True)
    plot()
