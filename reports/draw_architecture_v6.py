"""Publication-quality architecture figure for PIWM-v6.

Outputs reports/figures/architecture_v6.{png,pdf}.
"""

# --- auto-added: make repo root importable when run as a script ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end shim ---

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D


# ---------- style ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.6,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Color palette — muted, paper-friendly
C_FRAME  = "#E8EEF7"
C_NN     = "#CFE0F4"
C_LATENT = "#FFF2CC"
C_CAR    = "#F8D7C9"
C_LANE   = "#D9E8D2"
C_DYN    = "#F4E6F2"
C_PHYS   = "#FBE9C8"
C_OPS    = "#FAFAFA"
C_BRIDGE = "#FFF6DA"
EDGE     = "#2C3E50"
ARROW    = "#34495E"
ACCENT   = "#9C2A2A"
DASH     = "#9B7B2E"


def rbox(ax, xy, w, h, text, fc="white", ec=EDGE, fontsize=8.5, lw=0.7,
         weight="normal", style="round,pad=0.02,rounding_size=0.04",
         text_color="black", va="center", linespacing=1.3):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h, boxstyle=style,
                         linewidth=lw, edgecolor=ec, facecolor=fc,
                         joinstyle="round")
    ax.add_patch(box)
    ty = y + h / 2 if va == "center" else (y + h - 0.20 if va == "top" else y + 0.20)
    ax.text(x + w / 2, ty, text, ha="center", va=va,
            fontsize=fontsize, fontweight=weight, color=text_color,
            linespacing=linespacing)
    return (x, y, w, h)


def arrow(ax, p_from, p_to, color=ARROW, lw=1.0, style="-|>",
          mutation=12, rad=0.0, ls="-"):
    a = FancyArrowPatch(p_from, p_to, arrowstyle=style,
                        mutation_scale=mutation, color=color,
                        linewidth=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}",
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def right_mid(b):  return (b[0] + b[2],     b[1] + b[3] / 2)
def left_mid(b):   return (b[0],            b[1] + b[3] / 2)
def top_mid(b):    return (b[0] + b[2] / 2, b[1] + b[3])
def bot_mid(b):    return (b[0] + b[2] / 2, b[1])


# ---------- figure ----------
fig = plt.figure(figsize=(15.5, 12.0))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.45], hspace=0.10)

# ============================================================
# Panel (a): high-level pipeline
# ============================================================
axA = fig.add_subplot(gs[0, 0])
axA.set_xlim(0, 16); axA.set_ylim(0, 5.0); axA.axis("off")
axA.text(0.10, 4.65, "(a)  PIWM-v6 inference pipeline",
         fontsize=12.5, fontweight="bold")

# stacked input frames
for i, dx in enumerate([0, 0.07, 0.14, 0.21]):
    rbox(axA, (0.40 + dx, 2.20 - dx * 0.6), 1.05, 1.05,
         "" if i < 3 else r"$I_{t-2{:}t}$",
         fc=C_FRAME, lw=0.6, fontsize=11,
         style="round,pad=0.01,rounding_size=0.03")
axA.text(0.97, 1.78, r"$15\!\times\!64\!\times\!64$",
         ha="center", fontsize=9, color="#444", style="italic")

# encoder
b_enc = rbox(axA, (2.40, 1.85), 2.10, 1.65,
             "Encoder\n(CNN + FC)", fc=C_NN, fontsize=11, weight="bold")

# latent split
b_zcar  = rbox(axA, (5.20, 2.65), 1.95, 0.85,
               r"$z^{\,\mathrm{car}}_t \in \mathbb{R}^{9}$",
               fc=C_CAR, fontsize=11)
b_zlane = rbox(axA, (5.20, 1.65), 1.95, 0.85,
               r"$z^{\,\mathrm{lane}}_t \in \mathbb{R}^{20}$",
               fc=C_LANE, fontsize=11)
axA.text(6.175, 3.78, r"latent  $z_t \in \mathbb{R}^{29}$",
         ha="center", fontsize=9.5, style="italic", color="#444")

# action
b_act = rbox(axA, (8.05, 0.55), 1.20, 0.70,
             r"$a_t \in \mathbb{R}^{3}$", fc="#EFEFEF", fontsize=11)

# dynamics block
b_dyn = rbox(axA, (7.85, 1.65), 3.05, 1.85,
             "V6 Dynamics\n(bicycle + lane resample)",
             fc=C_DYN, fontsize=11.5, weight="bold")

# next latent
b_next = rbox(axA, (11.30, 1.85), 1.85, 1.65,
              r"$z_{t+1}$" + "\n" + r"$\in \mathbb{R}^{29}$",
              fc=C_LATENT, fontsize=11.5, weight="bold")

# decoder
b_dec = rbox(axA, (13.55, 1.85), 1.95, 1.65,
             "Decoder\n(FC + Deconv)", fc=C_NN, fontsize=11, weight="bold")

# predicted frame
rbox(axA, (15.65, 2.20), 1.05, 1.05, r"$\hat I_{t+1}$",
     fc=C_FRAME, fontsize=12, weight="bold")

# arrows in (a)
arrow(axA, (1.75, 2.65), left_mid(b_enc))
arrow(axA, right_mid(b_enc), left_mid(b_zcar))
arrow(axA, right_mid(b_enc), left_mid(b_zlane))
arrow(axA, right_mid(b_zcar),  (b_dyn[0], b_dyn[1] + b_dyn[3] - 0.45))
arrow(axA, right_mid(b_zlane), (b_dyn[0], b_dyn[1] + 0.45))
arrow(axA, top_mid(b_act),     (b_dyn[0] + b_dyn[2] / 2, b_dyn[1]))
arrow(axA, right_mid(b_dyn),   left_mid(b_next))
arrow(axA, right_mid(b_next),  left_mid(b_dec))
arrow(axA, right_mid(b_dec),   (15.65, 2.72))

# rollout loop arrow
loop_from = (b_next[0] + b_next[2] / 2, b_next[1])
loop_to   = (b_dyn[0] + b_dyn[2] / 2, b_dyn[1])
a_loop = FancyArrowPatch(loop_from, loop_to, arrowstyle="-|>",
                         mutation_scale=12, color="#888",
                         linewidth=1.0, linestyle=(0, (4, 3)),
                         connectionstyle="arc3,rad=0.50",
                         shrinkA=4, shrinkB=4)
axA.add_patch(a_loop)
axA.text(10.65, 0.65, "auto-regressive\nrollout",
         ha="center", fontsize=9, color="#666", style="italic")


# ============================================================
# Panel (b): zoom-in on V6 dynamics
# ============================================================
axB = fig.add_subplot(gs[1, 0])
axB.set_xlim(0, 16); axB.set_ylim(0, 12.5); axB.axis("off")
axB.text(0.10, 12.10,
         "(b)  Inside the V6 dynamics block — analytic dynamic-bicycle"
         " for the car branch, resample-based propagation for the lane branch",
         fontsize=12.5, fontweight="bold")

# subtle outer frame
frame = FancyBboxPatch((0.15, 0.45), 15.70, 11.10,
                       boxstyle="round,pad=0.02,rounding_size=0.10",
                       linewidth=0.9, edgecolor="#888", facecolor=C_DYN,
                       alpha=0.30)
axB.add_patch(frame)
# faint vertical divider
axB.plot([7.85, 7.85], [0.70, 11.40], color="#BBBBBB",
         linestyle=(0, (2, 3)), linewidth=0.8)

# ---------------- LEFT: Car branch ----------------
axB.text(3.85, 11.50, "Car branch", fontsize=12.5,
         fontweight="bold", color="#222", ha="center")

b_zc_in = rbox(axB, (1.55, 10.30), 4.70, 0.85,
               r"$z^{\,\mathrm{car}}_t \in \mathbb{R}^{11}\ \ \oplus\ \ "
               r"a_t \in \mathbb{R}^{3}$",
               fc=C_CAR, fontsize=12)

# Bicycle V4 — header outside the box for clarity
axB.text(3.85, 9.85, "Dynamic Bicycle (V4)",
         ha="center", fontsize=11.5, fontweight="bold")

b_bv = rbox(axB, (0.55, 5.30), 6.60, 4.20, "", fc="white", fontsize=10)

eq = (
    r"$\alpha_f = \delta - \arctan\!\dfrac{v_y + L_f\,\omega}{v_x}$"
    "\n"
    r"$\alpha_r = -\arctan\!\dfrac{v_y - L_r\,\omega}{v_x}$"
    "\n"
    r"$F_{yf} = -C_f\,\alpha_f,\quad F_{yr} = -C_r\,\alpha_r$"
    "\n"
    r"$\dot v_x = a - \dfrac{F_{yf}\sin\delta}{m} + v_y\,\omega$"
    "\n"
    r"$\dot v_y = \dfrac{F_{yf}\cos\delta + F_{yr}}{m} - v_x\,\omega$"
    "\n"
    r"$\dot \omega = \dfrac{L_f F_{yf}\cos\delta - L_r F_{yr}}{I_z}$"
)
axB.text(b_bv[0] + 2.10, b_bv[1] + b_bv[3] / 2, eq,
         ha="center", va="center", fontsize=8.8, color="#222",
         linespacing=1.45)

# learnable physics params (compact 4 lines so the box height is safely respected)
b_param = rbox(axB, (b_bv[0] + 4.55, b_bv[1] + 0.65), 1.85, 2.30,
               "learnable\nphysical params\n"
               r"$m,\ I_z$" + "\n" +
               r"$C_f,\ C_r$" + "\n" +
               r"$L_f,\ L_r$",
               fc=C_PHYS, fontsize=9.0, lw=0.7, linespacing=1.55)

# Euler integrate band
b_euler = rbox(axB, (0.55, 4.55), 6.60, 0.95,
               "Euler integrate  $\\Rightarrow$  "
               r"$v_x^{t+1},\ v_y^{t+1},\ \omega^{t+1},\ \psi^{t+1},"
               r"\ x^{t+1},\ y^{t+1}$"
               "\n"
               r"$+\ $ wheel-MLP $\to\ w_{0..3}^{t+1},\ \delta^{t+1}$",
               fc=C_OPS, fontsize=9.5, linespacing=1.6)

# masked MLP residual
b_resc = rbox(axB, (0.55, 3.20), 6.60, 0.95,
              r"$+\ \mathrm{MLP}_{\mathrm{res}}([z^{\,\mathrm{car}}_t,\,a_t])$"
              "\n"
              r"(residual masked on $x,\,y$)",
              fc=C_OPS, fontsize=9.5, linespacing=1.6)

# Output
b_zc_out = rbox(axB, (1.55, 1.55), 4.70, 1.05,
                r"$z^{\,\mathrm{car}}_{t+1} \in \mathbb{R}^{11}$",
                fc=C_CAR, fontsize=12.5, weight="bold")

# Car arrows
arrow(axB, bot_mid(b_zc_in),
      (b_bv[0] + b_bv[2] / 2, b_bv[1] + b_bv[3]))
arrow(axB, (b_bv[0] + b_bv[2] / 2, b_bv[1]),
      (b_euler[0] + b_euler[2] / 2, b_euler[1] + b_euler[3]))
arrow(axB, (b_euler[0] + b_euler[2] / 2, b_euler[1]),
      (b_resc[0] + b_resc[2] / 2, b_resc[1] + b_resc[3]))
arrow(axB, (b_resc[0] + b_resc[2] / 2, b_resc[1]),
      top_mid(b_zc_out))

# ---------------- RIGHT: Lane branch ----------------
axB.text(11.85, 11.50,
         "Lane branch — 10 waypoints at fixed $s$-offsets",
         fontsize=12.5, fontweight="bold", color="#222", ha="center")

b_zl_in = rbox(axB, (9.55, 10.30), 4.95, 0.85,
               r"$z^{\,\mathrm{lane}}_t \in \mathbb{R}^{20}$"
               r" $=$ $\{w_i\}_{i=0}^{9},\ w_i \in \mathbb{R}^2$",
               fc=C_LANE, fontsize=11.5)

# body-frame motion bridge
b_motion = rbox(axB, (8.05, 8.05), 2.45, 2.10,
                "body-frame\nmotion\n"
                r"$\Delta x_b = (v_x c{+}v_y s)\Delta t$" + "\n" +
                r"$\Delta y_b = (-v_x s{+}v_y c)\Delta t$" + "\n" +
                r"$\Delta \psi  = \omega\,\Delta t$",
                fc=C_BRIDGE, fontsize=8.8, linespacing=1.5)

# Step 1: Rigid-body inverse transform
b_rig = rbox(axB, (10.85, 8.05), 4.65, 2.10,
             "Step 1 · Rigid-body inverse transform\n"
             r"shift each $w_i$ by $(-\Delta x_b,\,-\Delta y_b)$"
             "\n"
             r"then rotate by $-\Delta\psi$",
             fc="white", fontsize=9.6, weight="bold", linespacing=1.6)

# Step 2: RESAMPLE (highlighted) — full width on right side
b_res = (8.05, 5.45, 7.45, 2.05)
box_res = FancyBboxPatch(
    (b_res[0], b_res[1]), b_res[2], b_res[3],
    boxstyle="round,pad=0.03,rounding_size=0.06",
    linewidth=1.8, edgecolor=ACCENT, facecolor="#FFE3E3", joinstyle="round")
axB.add_patch(box_res)
axB.text(b_res[0] + b_res[2] / 2, b_res[1] + b_res[3] - 0.30,
         "Step 2 · RESAMPLE to fixed $s$    (V6 key fix)",
         ha="center", fontsize=12, fontweight="bold", color=ACCENT)
axB.text(b_res[0] + b_res[2] / 2, b_res[1] + b_res[3] / 2 - 0.15,
         r"$\alpha = \Delta y_b / \Delta s,\ \ \Delta s = 5$",
         ha="center", fontsize=10.5, color="#1a1a1a")
axB.text(b_res[0] + b_res[2] / 2, b_res[1] + 0.40,
         r"slot $i < N{-}1:\ \ \widetilde w_i = (1-\alpha)\,w'_i + "
         r"\alpha\,w'_{i+1}$",
         ha="center", fontsize=10.5, color="#1a1a1a")

# Step 3: tangent extrapolation
b_ext = rbox(axB, (8.05, 3.20), 3.55, 1.65,
             "Step 3 · Front-slot\ntangent extrapolation\n"
             r"$\widetilde w_{N{-}1} = w'_{N{-}1}$"
             r"$+\,\alpha\,(w'_{N{-}1}\!-\!w'_{N{-}2})$",
             fc="white", fontsize=9.2, weight="normal", linespacing=1.7)

# Step 4: small learned residual
b_lres = rbox(axB, (11.95, 3.20), 3.55, 1.65,
              r"Step 4 · $0.1 \cdot$ MLP-residual"
              "\n"
              r"input: $[z^{\,\mathrm{car}}_{t+1},\ \widetilde w,\ a_t]$",
              fc="white", fontsize=9.2, weight="normal", linespacing=1.7)

# Output
b_zl_out = rbox(axB, (9.55, 1.55), 4.95, 1.05,
                r"$z^{\,\mathrm{lane}}_{t+1} \in \mathbb{R}^{20}$",
                fc=C_LANE, fontsize=12.5, weight="bold")

# Lane arrows
arrow(axB, bot_mid(b_zl_in), top_mid(b_rig))
arrow(axB, bot_mid(b_rig),   (b_res[0] + 5.20, b_res[1] + b_res[3]))
arrow(axB, (b_res[0] + 1.85, b_res[1]), top_mid(b_ext))
arrow(axB, right_mid(b_ext), left_mid(b_lres))
arrow(axB, bot_mid(b_lres),  (b_zl_out[0] + b_zl_out[2] - 0.95,
                              b_zl_out[1] + b_zl_out[3]))

# body-motion feeds (dashed orange)
arrow(axB, right_mid(b_motion),
      (b_rig[0], b_rig[1] + b_rig[3] / 2),
      ls=(0, (3, 2)), color=DASH)
arrow(axB, bot_mid(b_motion),
      (b_res[0] + 0.30, b_res[1] + b_res[3] - 0.30),
      ls=(0, (3, 2)), color=DASH, rad=0.0)

# car-state feed into MLP residual (dashed)
arrow(axB, right_mid(b_zc_out),
      (b_lres[0], b_lres[1] + 0.50),
      ls=(0, (3, 2)), color=DASH, rad=-0.18)

# bicycle output -> body motion (dashed)
arrow(axB, (b_bv[0] + b_bv[2] - 0.10, b_bv[1] + 0.20),
      (b_motion[0] + 0.30, b_motion[1]),
      ls=(0, (3, 2)), color=DASH, rad=-0.10)


# ----- legend -----
leg_handles = [
    Line2D([0], [0], color=ARROW, lw=1.4, label="data flow"),
    Line2D([0], [0], color=DASH, lw=1.4, linestyle=(0, (3, 2)),
           label=r"body-frame $\Delta$ feed"),
    Rectangle((0, 0), 1, 1, fc=C_CAR,    ec=EDGE, lw=0.6, label="car state"),
    Rectangle((0, 0), 1, 1, fc=C_LANE,   ec=EDGE, lw=0.6, label="lane state"),
    Rectangle((0, 0), 1, 1, fc=C_PHYS,   ec=EDGE, lw=0.6,
              label="learnable physical params"),
    Rectangle((0, 0), 1, 1, fc="#FFE3E3", ec=ACCENT, lw=1.4,
              label="V6's key contribution"),
]
fig.legend(handles=leg_handles, loc="lower center",
           bbox_to_anchor=(0.5, 0.005),
           ncol=6, frameon=False, fontsize=10,
           handlelength=1.8, columnspacing=2.4, handletextpad=0.7)


# ----- save -----
out_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(out_dir, exist_ok=True)
png_path = os.path.join(out_dir, "architecture_v6.png")
pdf_path = os.path.join(out_dir, "architecture_v6.pdf")
fig.savefig(png_path, bbox_inches="tight", dpi=300, facecolor="white")
fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
print(f"wrote {png_path}")
print(f"wrote {pdf_path}")
