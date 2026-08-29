#!/usr/bin/env python3
"""Bound on the aircraft-level benefit."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

fs.use()


def delta_total(f, dL):
    f = np.asarray(f, float)[..., None] if np.ndim(f) else f
    return -10 * np.log10(1 - f * (1 - 10 ** (-np.asarray(dL, float) / 10)))


def build(dL_paper=8.0, dL_achieved=0.16, dL_corrected=None):
    f = np.linspace(0.0, 1.0, 400)
    levels = [2.0, 4.0, 6.0, 8.0, 12.0]

    fig = plt.figure(figsize=(fs.W15, 0.44 * fs.W15))
    gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.085, right=0.995,
                          bottom=0.165, top=0.86)
    ax = [fig.add_subplot(gs[0, i]) for i in range(2)]

    for i, dL in enumerate(levels):
        ax[0].plot(f, delta_total(f, dL), color=fs.C[i % 5], ls=fs.DASH[i % 5],
                   lw=1.25, label=f"$\\Delta L={dL:g}$ dB")
    ax[0].axvspan(0.0, 0.25, color="#f0f0f0", lw=0, zorder=0)
    ax[0].text(0.02, 0.32,
               "Transport on approach:\nslat and flap dominate,\nTE self-noise minor",
               transform=ax[0].get_xaxis_transform(), fontsize=6.4, ha="left",
               va="top", color=fs.INK2, linespacing=1.35,
               bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.85))
    ax[0].axvline(1.0, color=fs.INK2, lw=0.6, ls=(0, (2.5, 1.6)))
    ax[0].text(0.985, 0.05, "Isolated lifting surface  ",
               transform=ax[0].get_xaxis_transform(), fontsize=6.4, ha="right",
               va="bottom", color=fs.INK2, rotation=90)
    ax[0].set_xlabel("Energy fraction of wing TE self-noise in the total, $f$")
    ax[0].set_ylabel("Upper bound on total reduction, $\\Delta_{\\rm tot}$  [dB]")
    ax[0].set_title("Bound versus source share", pad=3)
    ax[0].set_xlim(0, 1.02); ax[0].set_ylim(0, 12.4)
    ax[0].legend(loc="upper left", fontsize=6.6)
    fs.panel(ax[0], "(a)", dx=-0.185, dy=1.075)

    dL = np.linspace(0, 14, 400)
    for i, ff in enumerate([0.05, 0.10, 0.20, 0.50, 0.95]):
        ax[1].plot(dL, delta_total(ff, dL), color=fs.C[i % 5], ls=fs.DASH[i % 5],
                   lw=1.25, label=f"$f={ff:.2f}$")
    ax[1].plot(dL, dL, color=fs.MUTED, lw=0.8, ls=(0, (1.2, 1.4)), zorder=1)
    ax[1].text(12.4, 12.7, "$f=1$", fontsize=6.5, color=fs.MUTED,
               ha="right", va="bottom", rotation=41)

    F_REF = 0.10
    sat = float(10 * np.log10(1.0 / (1.0 - F_REF)))
    ax[1].axhline(sat, color=fs.INK2, lw=0.7, ls=(0, (2.5, 1.6)), zorder=2)
    y_ach = float(delta_total(F_REF, dL_achieved))
    y_pap = float(delta_total(F_REF, dL_paper))
    for dLv, yv in ((dL_achieved, y_ach), (dL_paper, y_pap)):
        ax[1].plot([dLv], [yv], marker="*", ms=7.5, color=fs.INK,
                   mec="white", mew=0.5, zorder=6)

    axi = ax[1].inset_axes([0.585, 0.195, 0.405, 0.285], zorder=9)
    axi.set_facecolor("white")
    axi.patch.set_alpha(1.0)
    for i, ff in enumerate([0.05, 0.10, 0.20]):
        axi.plot(dL, delta_total(ff, dL), color=fs.C[i % 5], ls=fs.DASH[i % 5],
                 lw=1.1)
    axi.axhline(sat, color=fs.INK2, lw=0.7, ls=(0, (2.5, 1.6)), zorder=2)
    axi.text(13.9, sat + 0.012, f"ceiling, {sat:.2f} dB", fontsize=5.9,
             color=fs.INK2, ha="right", va="bottom", zorder=8)
    for dLv, yv, tx, ty, ha, va in ((dL_achieved, y_ach, 0.75, 0.055, "left", "center"),
                                    (dL_paper, y_pap, 8.7, 0.30, "left", "top")):
        axi.plot([dLv], [yv], marker="*", ms=6.5, color=fs.INK, mec="white",
                 mew=0.5, zorder=6)
        axi.text(tx, ty, f"{yv:.2f} dB", fontsize=5.9, color=fs.INK, ha=ha,
                 va=va, zorder=7, clip_on=True,
                 bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.9))
    axi.set_xlim(0, 14.4); axi.set_ylim(0, 0.60)
    axi.tick_params(labelsize=5.8, length=2.0, pad=1.4)
    axi.set_yticks([0.0, 0.2, 0.4])
    axi.set_title("Detail: the starred cases", fontsize=6.2, pad=2)
    for sp in axi.spines.values():
        sp.set_linewidth(0.6); sp.set_visible(True)
    rect, lines = ax[1].indicate_inset_zoom(axi, edgecolor=fs.INK2, lw=0.6,
                                            alpha=0.65)
    for ln in lines:
        ln.set_visible(False)
    ax[1].set_xlabel("Reduction of the TE self-noise component, $\\Delta L$  [dB]")
    ax[1].set_ylabel("Upper bound on total reduction, $\\Delta_{\\rm tot}$  [dB]")
    ax[1].set_title("Bound versus component reduction", pad=3)
    ax[1].set_xlim(0, 14.4); ax[1].set_ylim(0, 14.4)
    ax[1].legend(loc="upper left", fontsize=6.6)
    fs.panel(ax[1], "(b)", dx=-0.185, dy=1.075)

    fs.save(fig, "fig_system_level")

    print("Upper bound on Delta EPNL [dB]")
    print(f"{'f':>6s} " + " ".join(f"{d:>7.0f} dB" for d in levels))
    for ff in (0.05, 0.10, 0.20, 0.35, 0.50, 0.95):
        print(f"{ff:6.2f} " + " ".join(f"{float(delta_total(ff, d)):10.2f}"
                                       for d in levels))


if __name__ == "__main__":
    build()
