"""Publication-quality figure: how the 10 lane waypoints are sampled.

Outputs reports/figures/lane_waypoints.{png,pdf}.

Three panels:
  (a) World view — track centerline (built from CarRacing road tile centers),
      car at its current pose, 10 points on the centerline sampled at
      arc-length offsets s_car + s_i, with s_i = -5, 0, 5, …, 40.
  (b) Body view — those same 10 points after transforming into the car's
      body frame (forward = body+y, lateral = body+x).  This is the 20-D
      vector that the encoder is supervised against and that the dynamics
      block propagates.
  (c) A short algorithm strip listing the three preprocessing steps.
"""

# --- auto-added: make repo root importable when run as a script ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end shim ---

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch
from matplotlib.lines import Line2D

from lane_utils import LANE_S_SAMPLES, world_to_body


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

C_TRACK    = "#E8E2D2"   # asphalt fill
C_EDGE     = "#5A5A5A"
C_CENTER   = "#B28A2A"   # centerline (dashed yellow, like a real road)
C_CAR      = "#C24B4B"
C_WP       = "#1F6FB2"
C_WP_BACK  = "#9AAEC0"   # the s=-5 waypoint (behind the car)
C_BODY_AX  = "#2C3E50"
ACCENT     = "#9C2A2A"


# ---------- centerline (synthetic, but uses real arc-length math) ----------
def build_centerline(L=85.0, n=4000):
    """Return centerline points and cumulative arc-length array."""
    t = np.linspace(0, L, n)
    x = t
    y = 6.0 * np.sin(t / 14.0) + 1.6 * np.sin(t / 4.5 + 0.3)
    pts = np.column_stack([x, y])
    diffs = np.diff(pts, axis=0)
    ds = np.sqrt((diffs ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    return pts, s


CL_PTS, CL_S = build_centerline()
TOTAL_S = CL_S[-1]


def sample_at_s(s_target):
    """Return (xy, tangent unit vector) on the centerline at arc length s_target."""
    s_target = np.clip(s_target, CL_S[0] + 1e-3, CL_S[-1] - 1e-3)
    idx = np.searchsorted(CL_S, s_target)
    idx = np.clip(idx, 1, len(CL_S) - 1)
    a = (s_target - CL_S[idx - 1]) / max(CL_S[idx] - CL_S[idx - 1], 1e-9)
    p = CL_PTS[idx - 1] * (1 - a) + CL_PTS[idx] * a
    tang = CL_PTS[idx] - CL_PTS[idx - 1]
    tang /= np.linalg.norm(tang) + 1e-12
    return p, tang


def normal_of(tang):
    """Unit normal (rotate +90deg) of a 2-D tangent."""
    return np.array([-tang[1], tang[0]])


# ---------- choose a representative car pose ----------
S_CAR = 32.0
car_pos, car_tang = sample_at_s(S_CAR)
# In PIWM convention, body+y = forward; world forward direction = (-sin yaw, cos yaw).
yaw = float(np.arctan2(-car_tang[0], car_tang[1]))

# Sample 10 waypoints at car_s + s_offset on centerline.
wp_world = np.stack([sample_at_s(S_CAR + s)[0] for s in LANE_S_SAMPLES], axis=0)
wp_body = world_to_body(wp_world, car_pos, yaw)


# ---------- figure ----------
fig = plt.figure(figsize=(14.5, 6.6))
gs = fig.add_gridspec(
    2, 2,
    width_ratios=[1.65, 1.0],
    height_ratios=[1.0, 0.18],
    wspace=0.18, hspace=0.05,
)

# =============================================================
# (a) world view
# =============================================================
axA = fig.add_subplot(gs[0, 0])
axA.set_aspect("equal"); axA.axis("off")

# Track band: centerline ± half-width
HALF_W = 4.0
tangents = np.diff(CL_PTS, axis=0, prepend=CL_PTS[0:1])
norms = np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12
tangents /= norms
normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=-1)
left  = CL_PTS + HALF_W * normals
right = CL_PTS - HALF_W * normals

# Asphalt fill
band = np.concatenate([left, right[::-1]], axis=0)
axA.add_patch(Polygon(band, closed=True, facecolor=C_TRACK,
                      edgecolor=C_EDGE, linewidth=1.0, joinstyle="round"))
# White edge stripes
axA.plot(left[:, 0],  left[:, 1],  color="white", lw=1.4, zorder=2)
axA.plot(right[:, 0], right[:, 1], color="white", lw=1.4, zorder=2)
# Dashed yellow centerline (visual aid only — geometry uses the underlying CL_PTS)
axA.plot(CL_PTS[:, 0], CL_PTS[:, 1], color=C_CENTER,
         lw=1.2, linestyle=(0, (6, 4)), zorder=3, alpha=0.9)

# Car triangle, pointing along its forward direction
def car_triangle(pos, fwd, length=2.4, width=1.4):
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    side = np.array([-fwd[1], fwd[0]])
    tip   = pos + fwd * length * 0.6
    backL = pos - fwd * length * 0.4 + side * width / 2
    backR = pos - fwd * length * 0.4 - side * width / 2
    return np.stack([tip, backL, backR], axis=0)


axA.add_patch(Polygon(car_triangle(car_pos, car_tang),
                      facecolor=C_CAR, edgecolor="black", linewidth=0.8,
                      zorder=6))

# Body-frame axes attached to car
ax_len = 4.2
ax_fwd = car_pos + car_tang * ax_len
ax_lat_dir = np.array([car_tang[1], -car_tang[0]])  # body +x = right of forward
ax_lat = car_pos + ax_lat_dir * ax_len * 0.7
axA.add_patch(FancyArrowPatch(car_pos, ax_fwd, arrowstyle="-|>",
                              mutation_scale=10, color=C_BODY_AX, lw=1.2,
                              zorder=7))
axA.add_patch(FancyArrowPatch(car_pos, ax_lat, arrowstyle="-|>",
                              mutation_scale=10, color=C_BODY_AX, lw=1.2,
                              zorder=7))
axA.text(*(ax_fwd + car_tang * 0.6), r"$\mathrm{body}\!+\!y$ (forward)",
         color=C_BODY_AX, fontsize=9.5, ha="left", va="center")
axA.text(*(ax_lat + ax_lat_dir * 0.6), r"$\mathrm{body}\!+\!x$",
         color=C_BODY_AX, fontsize=9.5, ha="left", va="center")

# 10 waypoint markers
for i, (s_off, p) in enumerate(zip(LANE_S_SAMPLES, wp_world)):
    color = C_WP_BACK if s_off < 0 else C_WP
    axA.add_patch(plt.Circle(p, 0.55, facecolor=color, edgecolor="white",
                             linewidth=1.0, zorder=8))
    if s_off in (-5.0, 0.0, 20.0, 40.0):
        # label near the dot, perpendicular to centerline
        _, t_here = sample_at_s(S_CAR + s_off)
        n_here = normal_of(t_here)
        lp = p + n_here * 2.6
        axA.text(lp[0], lp[1], f"$s\\!=\\!{int(s_off)}$",
                 fontsize=9, ha="center", va="center", color="#333")

# Arc-length axis arrow along the centerline near the start
p0, t0 = sample_at_s(S_CAR + LANE_S_SAMPLES[0])
p1, _  = sample_at_s(S_CAR + LANE_S_SAMPLES[-1])
axA.text(p0[0] - 1.8, p0[1] + 4.6, "centerline arc-length $s$",
         fontsize=9.5, color="#555", style="italic")

# car position label
axA.annotate(r"car at $s_{\rm car}$",
             xy=car_pos, xytext=(car_pos[0] - 4.0, car_pos[1] - 6.0),
             fontsize=10, color=C_CAR, ha="center",
             arrowprops=dict(arrowstyle="-", color=C_CAR, lw=0.7))

axA.set_title("(a)  World view — sample centerline at "
              "$\\{s_{\\rm car}+s_i\\}_{i=0}^{9},\\ \\ "
              "s_i\\in\\{-5,0,5,\\dots,40\\}$",
              fontsize=11.5, fontweight="bold", loc="left", pad=8)
# Set view bounds with a bit of margin around relevant region
xmin = wp_world[:, 0].min() - 6
xmax = wp_world[:, 0].max() + 6
ymin = wp_world[:, 1].min() - 7
ymax = wp_world[:, 1].max() + 7
axA.set_xlim(xmin, xmax); axA.set_ylim(ymin, ymax)


# =============================================================
# (b) body view
# =============================================================
axB = fig.add_subplot(gs[0, 1])
axB.set_aspect("equal")

# Body-frame grid + axes
axB.axhline(0, color="#CCCCCC", lw=0.6, zorder=1)
axB.axvline(0, color="#CCCCCC", lw=0.6, zorder=1)
axB.grid(True, color="#EFEFEF", lw=0.4, zorder=0)

# Car at origin (small triangle, body+y up)
fwd_b = np.array([0.0, 1.0])
axB.add_patch(Polygon(car_triangle(np.array([0.0, 0.0]), fwd_b,
                                    length=2.4, width=1.4),
                      facecolor=C_CAR, edgecolor="black", linewidth=0.8,
                      zorder=6))

# Plot the 10 body-frame waypoints, connect with a thin line to suggest lane shape
order = np.argsort(LANE_S_SAMPLES)  # already sorted
axB.plot(wp_body[order, 0], wp_body[order, 1],
         color=C_WP, lw=1.0, alpha=0.5, zorder=5)
for i, (s_off, p) in enumerate(zip(LANE_S_SAMPLES, wp_body)):
    color = C_WP_BACK if s_off < 0 else C_WP
    axB.scatter(p[0], p[1], s=42, c=color, edgecolor="white",
                linewidths=0.9, zorder=7)
    if s_off in (-5.0, 0.0, 20.0, 40.0):
        axB.annotate(f"$s\\!=\\!{int(s_off)}$",
                     xy=(p[0], p[1]),
                     xytext=(p[0] + 2.4, p[1]),
                     fontsize=9, color="#333",
                     arrowprops=dict(arrowstyle="-", color="#888", lw=0.5,
                                     shrinkA=2, shrinkB=2))

axB.set_xlabel(r"$x_b$  (lateral)", fontsize=10)
axB.set_ylabel(r"$y_b$  (forward)", fontsize=10)
axB.set_title("(b)  Body view — the 20-D lane vector\n"
              r"$z^{\,\mathrm{lane}} = "
              r"(x_b^0, y_b^0, x_b^1, y_b^1, \dots, x_b^9, y_b^9)$",
              fontsize=11, fontweight="bold", loc="left", pad=8)
axB.tick_params(labelsize=8.5)
# Make sure 0 is visible and add a bit of margin
xb_min, xb_max = wp_body[:, 0].min(), wp_body[:, 0].max()
yb_min, yb_max = wp_body[:, 1].min(), wp_body[:, 1].max()
axB.set_xlim(min(xb_min, -3) - 4, max(xb_max, 3) + 7)
axB.set_ylim(min(yb_min, -3) - 3, yb_max + 5)


# =============================================================
# (c) algorithm strip
# =============================================================
axC = fig.add_subplot(gs[1, :])
axC.set_xlim(0, 16); axC.set_ylim(0, 1); axC.axis("off")

# Three sequential boxes describing the algorithm
def step(ax, x, w, head, body):
    fc = "#FAFAFA"
    ax.add_patch(FancyBboxPatch(
        (x, 0.05), w, 0.90,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=0.8, edgecolor="#888", facecolor=fc))
    ax.text(x + w / 2, 0.72, head, ha="center", va="center",
            fontsize=10.5, fontweight="bold")
    ax.text(x + w / 2, 0.32, body, ha="center", va="center",
            fontsize=9.5)


step(axC, 0.10, 5.10,
     "Step 1 · Project car to centerline",
     r"$s_{\rm car} = \mathrm{xy\_to\_sd}(\mathrm{car\_pos})$"
     r"     (Frenet $s$-coordinate)")
step(axC, 5.45, 5.10,
     "Step 2 · Sample 10 fixed offsets",
     r"$w_i^{\,\rm world} = \mathrm{sd\_to\_xy}(s_{\rm car}+s_i,\,d{=}0),"
     r"\ \ s_i\in\{-5,0,5,\dots,40\}$")
step(axC, 10.80, 5.10,
     "Step 3 · World → body frame",
     r"$w_i = R(-\mathrm{yaw})\cdot(w_i^{\,\rm world}-\mathrm{car\_pos})$")

# small arrows between the three boxes
for x in (5.20, 10.55):
    axC.add_patch(FancyArrowPatch(
        (x, 0.50), (x + 0.25, 0.50),
        arrowstyle="-|>", mutation_scale=10, color="#888", lw=1.0))


# =============================================================
# Outer title and figure-level legend
# =============================================================
fig.suptitle("Lane waypoint extraction — built once per episode by "
             "`scripts/lane_preprocess.py`",
             fontsize=12.5, fontweight="bold", y=0.99)

leg_handles = [
    Line2D([0], [0], marker="o", color="white",
           markerfacecolor=C_WP, markersize=8,
           label=r"waypoints ahead of car ($s_i\geq 0$)"),
    Line2D([0], [0], marker="o", color="white",
           markerfacecolor=C_WP_BACK, markersize=8,
           label=r"waypoint behind car ($s_i=-5$)"),
    Line2D([0], [0], color=C_CENTER, lw=1.2, linestyle=(0, (6, 4)),
           label="track centerline"),
    Line2D([0], [0], color=C_CAR, marker="^", markersize=8,
           markerfacecolor=C_CAR, lw=0,
           label=r"car (pose $=$ pos $+$ yaw)"),
]
fig.legend(handles=leg_handles, loc="lower center",
           bbox_to_anchor=(0.5, -0.01),
           ncol=4, frameon=False, fontsize=9.5,
           handlelength=2.0, columnspacing=1.8, handletextpad=0.6)


# ----- save -----
out_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(out_dir, exist_ok=True)
png_path = os.path.join(out_dir, "lane_waypoints.png")
pdf_path = os.path.join(out_dir, "lane_waypoints.pdf")
fig.savefig(png_path, bbox_inches="tight", dpi=300, facecolor="white")
fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
print(f"wrote {png_path}")
print(f"wrote {pdf_path}")
