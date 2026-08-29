#!/usr/bin/env python3
"""One publication figure style for the whole paper."""
from __future__ import annotations
import os
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Agg")

MM = 1 / 25.4
W1, W15, W2 = 90 * MM, 140 * MM, 190 * MM

C = ["#0072B2",
     "#D55E00",
     "#009E73",
     "#7A4FA3",
     "#B8860B"]
INK = "#1a1a1a"
INK2 = "#4d4d4d"
MUTED = "#8c8c8c"
GRID = "#d9d9d9"
BAND = "#c9c9c9"

DASH = [(0, ()), (0, (5, 1.6)), (0, (1.4, 1.4)), (0, (6, 1.5, 1.3, 1.5)),
        (0, (3, 1.3, 1.3, 1.3, 1.3, 1.3))]
MARK = ["o", "s", "^", "D", "v"]


L_REDUCED = "Reduced formulation"
L_COMPLETE = "Complete formulation"
L_MEAS = "Measured (NASA database)"
L_SURR = "NeuralFoil surrogate"
L_RANS = "RANS ($k$\u2013$\\omega$ SST)"
L_GEOM = "$\\alpha^*=\\alpha$ (geometric)"
L_EFF = "$\\alpha^*=C_L/C_{L\\alpha}$ (lift-effective)"


def use():
    plt.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 400,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 7.2,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "xtick.minor.size": 1.4, "ytick.minor.size": 1.4,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.45,
        "grid.alpha": 1.0, "axes.axisbelow": True,
        "lines.linewidth": 1.2, "lines.markersize": 3.4,
        "lines.markeredgewidth": 0.6,
        "legend.frameon": True, "legend.framealpha": 0.92,
        "legend.edgecolor": GRID, "legend.borderpad": 0.35,
        "legend.handlelength": 2.1, "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.28, "legend.columnspacing": 1.1,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def series(i: int) -> dict:
    return dict(color=C[i % len(C)], linestyle=DASH[i % len(DASH)],
                marker=MARK[i % len(MARK)])


def panel(ax, label: str, dx: float = -0.135, dy: float = 1.045, **kw):
    ax.text(dx, dy, label, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="bottom", ha="left", color=INK, **kw)


def annotate(ax, text, xy, xytext, color=INK, arrow=True, fontsize=7.0, **kw):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fontsize, color=color,
                ha=kw.pop("ha", "left"), va=kw.pop("va", "center"),
                arrowprops=dict(arrowstyle="-", lw=0.55, color=color,
                                shrinkA=1.5, shrinkB=2.5) if arrow else None,
                **kw)


def save(fig, name: str, outdir: str = "figs"):
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    paper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_AST")
    src = os.path.join(outdir, f"{name}.pdf")
    if os.path.isdir(paper) and os.path.abspath(outdir) != os.path.abspath(paper):
        shutil.copy2(src, os.path.join(paper, f"{name}.pdf"))
        paths.append(os.path.join(paper, f"{name}.pdf"))
    print("wrote " + "  ".join(paths))
    return paths
