"""Single source of truth for the look of every figure that goes into the paper.

Both src/paper_figures.py and scripts/fig_delta_supervision.py import this, so the
result figures cannot drift apart in font, size or palette. They previously did:
fig_delta_supervision.py used bare matplotlib defaults while paper_figures.py set
its own 9pt sans-serif style, so Figure 6 and Figure 5 rendered in different
typefaces at different sizes on facing pages.

Two things matter for a figure that has to sit next to 12pt Latin Modern body text:

1. TYPESET THE LABELS WITH LATEX (``text.usetex``). The figure text is then the
   same font as the paper rather than a lookalike, and maths in labels matches the
   maths in the body.

2. DRAW AT THE FINAL SIZE. A figure authored 3.5 in wide and included at
   ``width=\\linewidth`` (5.4 in) is magnified 1.55x, so a declared 10pt label
   lands on the page at ~15pt -- larger than the 12pt body text it is meant to
   sit under. Authoring at exactly the text width makes declared pt equal
   rendered pt. Use FIG_W below as the figure width and include the result at
   ``width=\\linewidth``.

Requires a LaTeX installation reachable from PATH (MiKTeX on this machine); see
README.md. If LaTeX is unavailable, call apply(usetex=False) to fall back to a
metric-similar serif -- the figures will still be self-consistent, just not
identical to the body font.
"""
import matplotlib as mpl

#: \textwidth of the elsarticle preprint/12pt class, in inches (390 TeX pt).
#: Measured, not guessed: \the\textwidth in that class reports 390.0pt.
FIG_W = 390.0 / 72.27

#: Colour-blind-safe palette shared by every result figure.
COL = {"Frenet": "#3B33A0", "FrenetOr": "#8A83D8", "GOKU": "#2E9E4F",
       "V2P": "#E8820C", "DVBF": "#C0457B", "SINDYc": "#C0392B"}

LBL = {"Frenet": "PIWM-Frenet (ours)", "FrenetOr": r"PIWM-Frenet (oracle $\kappa$)",
       "GOKU": "GOKU", "V2P": "Vid2Param", "DVBF": "DVBF", "SINDYc": "SINDYc"}


def apply(usetex=True):
    """Install the publication style. Call once, before creating any figure."""
    mpl.rcParams.update({
        # --- type ---------------------------------------------------------
        "text.usetex": usetex,
        "font.family": "serif",
        # only consulted when usetex is False; Latin Modern first if installed
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        # Match the document's font so figure maths and body maths agree.
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}",

        # --- sizes: drawn at final scale, so these are the ON-PAGE sizes ---
        # Body text is 12pt and captions ~10pt, so 9-10pt here reads as
        # subordinate to the caption without being cramped.
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        # --- furniture ----------------------------------------------------
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.8,
        "lines.markersize": 4,
        # An opaque, borderless legend patch. Frameless looks cleaner in the
        # abstract, but in fig_main the diverging SINDYc curve shoots straight
        # through the legend text; a white backing keeps the labels readable
        # without adding a visible box.
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.facecolor": "white",
        "legend.edgecolor": "none",
        "legend.borderpad": 0.3,
        "legend.handlelength": 1.8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,

        # --- output -------------------------------------------------------
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
