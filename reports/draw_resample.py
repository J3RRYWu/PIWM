"""Visualize the lane-resample step with concrete numbers
(Δy_b = 1.5, Δs = 5, α = 0.3).

Three panels in body frame:
  (a) at time t          : slot i is at body-y = s_i
  (b) after Step 1 rigid : slot i is at body-y = s_i − Δy_b   (drifted)
  (c) after Step 2 resam.: slot i is back at body-y = s_i      (corrected)

Bottom strip shows the linear-interpolation formula with the actual
weighted sum that recovers slot 1 (s_1 = 0) from slot 1 (drifted to
−1.5) and slot 2 (drifted to 3.5).
"""

# --- auto-added: make repo root importable when run as a script ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end shim ---

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, FancyBboxPatch
from matplotlib.lines import Line2D


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.6,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------- numerical example ----------
S = np.array([-5, 0, 5, 10, 15, 20, 25, 30, 35, 40], dtype=float)   # s_i
N = len(S)
DS = 5.0
DY_B = 1.5
ALPHA = DY_B / DS                                                    # = 0.30


def lane_curve_x(s):
    """A gentle synthetic centerline x_b(s) so panels look like a real lane."""
    return 2.0 * np.sin(s / 12.0) - 0.4


# state at time t: every slot exactly at body-y = s_i
x_t = lane_curve_x(S)
y_t = S.copy()

# Step 1 (rigid, with Δx_b = 0 and Δψ = 0 for visualisation clarity):
x_r = x_t.copy()
y_r = y_t - DY_B

# Step 2 (resample): linear interp slot i ← (slot i, slot i+1); front extrapolates
x_R = np.zeros_like(x_t); y_R = np.zeros_like(y_t)
for i in range(N - 1):
    x_R[i] = (1 - ALPHA) * x_r[i] + ALPHA * x_r[i + 1]
    y_R[i] = (1 - ALPHA) * y_r[i] + ALPHA * y_r[i + 1]
tang_x = x_r[-1] - x_r[-2]; tang_y = y_r[-1] - y_r[-2]
x_R[-1] = x_r[-1] + ALPHA * tang_x
y_R[-1] = y_r[-1] + ALPHA * tang_y


# ---------- styles ----------
C_LANE_BG  = "#D9E8D2"
C_CENTER   = "#B28A2A"
C_CAR      = "#C24B4B"
C_T0       = "#1F6FB2"     # state at time t
C_RIGID    = "#E89B25"     # after rigid transform (drifted)
C_RESAMP   = "#9C2A2A"     # after resample (corrected)
C_GUIDE    = "#888888"


def car_triangle(pos, fwd, length=2.4, width=1.4):
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    side = np.array([-fwd[1], fwd[0]])
    tip   = pos + fwd * length * 0.6
    backL = pos - fwd * length * 0.4 + side * width / 2
    backR = pos - fwd * length * 0.4 - side * width / 2
    return np.stack([tip, backL, backR], axis=0)


def style_panel(ax, title, subtitle=""):
    ax.axhline(0, color="#CCCCCC", lw=0.5, zorder=0)
    ax.axvline(0, color="#CCCCCC", lw=0.5, zorder=0)
    ax.grid(True, color="#F0F0F0", lw=0.4, zorder=0)
    ax.set_xlabel(r"$x_b$ (lateral)", fontsize=10)
    ax.set_ylabel(r"$y_b$ (forward)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=22)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes,
                fontsize=9.5, style="italic", color="#444")
    ax.tick_params(labelsize=8.5)
    for s in S:
        ax.axhline(s, color="#EEEEEE", lw=0.4, zorder=0)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-10, 46)
    ax.add_patch(Polygon(car_triangle(np.array([0., 0.]), np.array([0., 1.]),
                                       length=2.6, width=1.6),
                         facecolor=C_CAR, edgecolor="black", linewidth=0.7,
                         zorder=8))


# ---------- figure ----------
fig = plt.figure(figsize=(15.2, 8.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.18], hspace=0.10, wspace=0.32)

# (a) at time t
axA = fig.add_subplot(gs[0, 0])
style_panel(axA, "(a)  At time $t$",
            subtitle=r"every slot $i$ already at $y_b = s_i$")
axA.scatter(x_t, y_t, s=60, c=C_T0, edgecolor="white",
            linewidths=0.9, zorder=7, label=r"$w_i$ (slot $i$, target met)")
for s, x, y in zip(S, x_t, y_t):
    axA.annotate(f"{int(s):+d}" if s != 0 else "  0",
                 xy=(x, y), xytext=(x + 0.4, y),
                 fontsize=8.5, color="#222", va="center",
                 ha="left")

# (b) after Step 1 rigid (drift)
axB = fig.add_subplot(gs[0, 1])
style_panel(axB, "(b)  After Step 1 — rigid transform",
            subtitle=r"slot $i$ drifts to $y_b = s_i - \Delta y_b$")
# Faint outline of where they used to be
axB.scatter(x_t, y_t, s=46, facecolors="none",
            edgecolors=C_T0, linewidths=0.9, zorder=4,
            label=r"old position $w_i$")
# After-rigid points
axB.scatter(x_r, y_r, s=60, c=C_RIGID, edgecolor="white",
            linewidths=0.9, zorder=7,
            label=r"$w'_i$ after rigid")
# arrows showing the downward shift
for i in range(N):
    axB.add_patch(FancyArrowPatch(
        (x_t[i], y_t[i]), (x_r[i], y_r[i]),
        arrowstyle="-|>", mutation_scale=8, color=C_RIGID, lw=0.9,
        shrinkA=4, shrinkB=4, zorder=6))
# Δy_b annotation
axB.annotate(r"$\Delta y_b = 1.5$",
             xy=(x_t[3] + 0.05, (y_t[3] + y_r[3]) / 2),
             xytext=(2.1, (y_t[3] + y_r[3]) / 2),
             fontsize=10, color=C_RIGID, style="italic",
             arrowprops=dict(arrowstyle="-", color=C_RIGID, lw=0.5),
             va="center")

# (c) after Step 2 resample (corrected)
axC = fig.add_subplot(gs[0, 2])
style_panel(axC, "(c)  After Step 2 — resample",
            subtitle=r"slot $i$ back at $y_b = s_i$")
# Faint outlines of after-rigid for context
axC.scatter(x_r, y_r, s=46, facecolors="none",
            edgecolors=C_RIGID, linewidths=0.9, zorder=4,
            label=r"$w'_i$ after rigid")
# Resampled points
axC.scatter(x_R, y_R, s=70, c=C_RESAMP, edgecolor="white",
            linewidths=1.0, zorder=8, marker="o",
            label=r"$\widetilde w_i$ resampled (target met)")
# interp arrows: slot i ← (1−α) w'_i + α w'_{i+1}
for i in range(N - 1):
    # source position is on the segment between w'_i and w'_{i+1};
    # show the interpolation as a curve from w'_i toward w'_{i+1},
    # ending at the resampled point.
    axC.add_patch(FancyArrowPatch(
        (x_r[i], y_r[i]), (x_R[i], y_R[i]),
        arrowstyle="-|>", mutation_scale=7, color=C_RESAMP, lw=0.9,
        shrinkA=3, shrinkB=3, zorder=6))
# front-slot extrapolation arrow (different style)
axC.add_patch(FancyArrowPatch(
    (x_r[-1], y_r[-1]), (x_R[-1], y_R[-1]),
    arrowstyle="-|>", mutation_scale=7, color=C_RESAMP, lw=1.1,
    linestyle=(0, (3, 2)),
    shrinkA=3, shrinkB=3, zorder=6))
axC.annotate("front slot:\nextrapolate along\ntangent of last two",
             xy=(x_R[-1], y_R[-1]),
             xytext=(1.8, 35.0),
             fontsize=8.5, color=C_RESAMP,
             arrowprops=dict(arrowstyle="-", color=C_RESAMP, lw=0.5))
axC.text(-4.30, 44.5,
         r"$\alpha = \Delta y_b/\Delta s = 0.3,\ \ "
         r"\widetilde w_i = (1-\alpha)\,w'_i + \alpha\,w'_{i+1}$",
         fontsize=9, color=C_RESAMP, style="italic", va="top")

# Highlight one slot with the worked number example (slot 1, target s_1 = 0).
# In panel (c): ringed marker + a small callout box.
axC.scatter(x_R[1], y_R[1], s=180, facecolors="none",
            edgecolors=C_RESAMP, linewidths=1.4, zorder=9)
axC.annotate(
    "worked example  (slot $1$, target $s_1\\!=\\!0$):\n"
    r"$y_b = (1-\alpha)\!\cdot\!(-1.5) + \alpha\!\cdot\!3.5$"
    "\n"
    r"$\quad= 0.7\!\cdot\!(-1.5) + 0.3\!\cdot\!3.5 = 0.0\ \ \mathrm{(=s_1)}$",
    xy=(x_R[1], y_R[1]),
    xytext=(-4.30, 16.5),
    fontsize=8.4, color=C_RESAMP,
    arrowprops=dict(arrowstyle="-", color=C_RESAMP, lw=0.5),
    bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
              edgecolor=C_RESAMP, linewidth=0.7),
    ha="left", va="center")


# ---------- bottom strip: legend / formula ----------
axL = fig.add_subplot(gs[1, :])
axL.set_xlim(0, 16); axL.set_ylim(0, 1); axL.axis("off")

# left half: legend
handles = [
    Line2D([0], [0], marker="o", color="white", markerfacecolor=C_T0,
           markersize=9, label=r"$w_i$ at time $t$ (target met)"),
    Line2D([0], [0], marker="o", color="white", markerfacecolor=C_RIGID,
           markersize=9, label=r"$w'_i$ after rigid (drifted)"),
    Line2D([0], [0], marker="o", color="white", markerfacecolor=C_RESAMP,
           markersize=9, label=r"$\widetilde w_i$ after resample (target met)"),
]
axL.legend(handles=handles, loc="center left", bbox_to_anchor=(0.0, 0.5),
           ncol=3, frameon=False, fontsize=10,
           handlelength=1.6, columnspacing=2.0, handletextpad=0.6)

axL.text(16.0, 0.5,
         r"parameters here:  $\Delta s = 5$,  $\Delta y_b = 1.5$,  "
         r"$\alpha = \Delta y_b/\Delta s = 0.3$,  $N = 10$",
         ha="right", va="center", fontsize=9.5, style="italic",
         color="#444")


# ---------- save ----------
out_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(out_dir, exist_ok=True)
png = os.path.join(out_dir, "lane_resample.png")
pdf = os.path.join(out_dir, "lane_resample.pdf")
fig.savefig(png, bbox_inches="tight", dpi=300, facecolor="white")
fig.savefig(pdf, bbox_inches="tight", facecolor="white")
print(f"wrote {png}")
print(f"wrote {pdf}")
