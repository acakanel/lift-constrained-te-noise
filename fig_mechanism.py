#!/usr/bin/env python3
"""The source of the lift-preserving level reduction."""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import figstyle as fs
import bpm_noise_v2 as v2

fs.use()
CHORD = 1.0
CLA = 0.1143


def manifold(U=55.0, CLreq=1.30, deltas=np.arange(-10, 10.1, 2.0),
             cache="manifold_path.csv"):
    if os.path.exists(cache):
        return pd.read_csv(cache)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from aero_dataset import run_case
    rows = []
    for d in deltas:
        f = lambda a: run_case(a, d, d, U)[0] - CLreq
        try:
            if f(-4.0) * f(14.0) > 0:
                continue
            a = brentq(f, -4.0, 14.0, xtol=1e-3)
        except Exception:
            continue
        CL, CM, ds, dp = run_case(a, d, d, U)
        rows.append(dict(delta=d, alpha=a, CL=CL, CM=CM, dstar_s=ds, dstar_p=dp))
    c = pd.DataFrame(rows)
    c.to_csv(cache, index=False)
    return c


def decompose(row, U, alpha_star):
    tot, comp, info = v2.spl_third_octave(
        row.dstar_s, row.dstar_p, U, alpha_star, CHORD, return_components=True)
    band = lambda x: 10 * np.log10(np.sum(10 ** (np.asarray(x, float) / 10)) + 1e-300)
    return dict(total=band(tot), p=band(comp["p"]), s=band(comp["s"]),
                a=band(comp["alpha"]), stalled=info["stalled"])


def build(U=55.0, CLreq=1.30):
    M = manifold(U, CLreq)
    M["alpha_eff"] = np.clip(np.abs(M.CL) / CLA, 0, 25)

    geo = pd.DataFrame([decompose(r, U, r.alpha) for r in M.itertuples()])
    eff = pd.DataFrame([decompose(r, U, r.alpha_eff) for r in M.itertuples()])

    fig = plt.figure(figsize=(fs.W2, 0.315 * fs.W2))
    gs = fig.add_gridspec(1, 3, wspace=0.34, width_ratios=[1, 1.14, 1.02],
                          left=0.058, right=0.982, bottom=0.155, top=0.845)
    ax = [fig.add_subplot(gs[0, i]) for i in range(3)]

    ax[0].plot(M.alpha, 1e3 * M.dstar_s, color=fs.C[0], ls=fs.DASH[0],
               marker=fs.MARK[0], ms=3.4, mfc=fs.C[0], mec="white", mew=0.45,
               label="Suction side, $\\delta^*_s$")
    ax[0].plot(M.alpha, 1e3 * M.dstar_p, color=fs.C[2], ls=fs.DASH[2],
               marker=fs.MARK[2], ms=3.4, mfc="white", mec=fs.C[2], mew=0.8,
               label="Pressure side, $\\delta^*_p$")
    ax[0].set_xlabel("Trimmed incidence, $\\alpha$  [deg]")
    ax[0].set_ylabel("Displacement thickness at TE  [mm]")
    ax[0].set_yscale("log")
    ax[0].set_title("Boundary-layer state on the manifold", pad=3)
    ax[0].legend(loc="upper right", fontsize=6.6)
    fs.panel(ax[0], "(a)", dx=-0.21, dy=1.08)
    lo, hi = M.iloc[M.alpha.idxmin()], M.iloc[M.alpha.idxmax()]
    ax[0].annotate(f"Flaps down, lower\nincidence:\n$\\delta^*_s$ {100*(lo.dstar_s/hi.dstar_s-1):+.0f}%",
                   xy=(lo.alpha, 1e3 * lo.dstar_s),
                   xytext=(0.97, 0.72), textcoords="axes fraction",
                   fontsize=6.5, ha="right", va="top", color=fs.INK,
                   linespacing=1.35,
                   bbox=dict(fc="white", ec=fs.GRID, lw=0.45, pad=2.0, alpha=0.94),
                   arrowprops=dict(arrowstyle="-", lw=0.55, color=fs.INK2,
                                   shrinkA=1.5, shrinkB=2.5))

    sw = v2.alpha_switch(U / 340.46)
    stalled = geo["stalled"].values.astype(bool)
    if stalled.any():
        ax[1].axvspan(sw, M.alpha.max() + 1, color="#f2f2f2", lw=0, zorder=0)
    for j, (k, lab) in enumerate([("p", "Pressure-side term, $\\mathrm{SPL}_p$"),
                                  ("s", "Suction-side term, $\\mathrm{SPL}_s$"),
                                  ("a", "Separation term, $\\mathrm{SPL}_\\alpha$")]):
        y = np.asarray(geo[k], float)
        y = np.where(np.isfinite(y) & (y > -100), y, np.nan)
        ax[1].plot(M.alpha, y, color=fs.C[j], ls=fs.DASH[j], marker=fs.MARK[j],
                   ms=3.2, mfc="white" if j else fs.C[0], mec=fs.C[j], mew=0.7,
                   label=lab)
    ax[1].plot(M.alpha, geo["total"], color=fs.INK, ls=(0, ()), lw=1.5,
               marker="", label="Total")
    ax[1].set_xlabel("Trimmed incidence, $\\alpha$  [deg]")
    ax[1].set_ylabel("Overall SPL  [dB]")
    ax[1].set_title("Source-term decomposition", pad=3)
    ax[1].set_ylim(48, 92)
    ax[1].set_xlim(M.alpha.min() - 0.6, M.alpha.max() + 0.6)
    ax[1].legend(loc="lower center", fontsize=6.2, ncol=2,
                 framealpha=0.95, handlelength=1.8, columnspacing=1.1)
    fs.panel(ax[1], "(b)", dx=-0.20, dy=1.08)
    if M.alpha.min() < sw < M.alpha.max():
        ax[1].axvline(sw, color=fs.INK2, lw=0.7, ls=(0, (2.5, 1.6)))
        ax[1].text(M.alpha.max() + 0.45, 91.2,
                   "Separated branch:\n$\\mathrm{SPL}_p,\\mathrm{SPL}_s$ off",
                   fontsize=6.3, color=fs.INK2, va="top", ha="right",
                   linespacing=1.3)
    d_tot = geo["total"].max() - geo["total"].min()
    d_alp = np.nanmax(geo["a"]) - np.nanmin(geo["a"])
    ax[1].text(0.035, 0.965,
               f"Change along the manifold\nTotal            {d_tot:5.1f} dB\n"
               f"Separation term  {d_alp:5.1f} dB",
               transform=ax[1].transAxes, fontsize=6.4, ha="left", va="top",
               linespacing=1.35,
               bbox=dict(fc="white", ec=fs.GRID, lw=0.45, pad=2.2, alpha=0.94))

    ax[2].plot(M.alpha, geo["total"] - geo["total"].min(), color=fs.C[0],
               ls=fs.DASH[0], marker=fs.MARK[0], ms=3.4, mfc=fs.C[0],
               mec="white", mew=0.45,
               label=fs.L_GEOM)
    ax[2].plot(M.alpha, eff["total"] - eff["total"].min(), color=fs.C[1],
               ls=fs.DASH[1], marker=fs.MARK[1], ms=3.6, mfc="white", mew=0.8,
               label=fs.L_EFF)
    ax[2].set_xlabel("Trimmed incidence, $\\alpha$  [deg]")
    ax[2].set_ylabel("Level above the manifold minimum  [dB]")
    ax[2].set_title("Sensitivity to the incidence parameter", pad=3)
    ax[2].legend(loc="upper left", fontsize=6.3)
    ax[2].set_xlim(M.alpha.min() - 0.6, M.alpha.max() + 0.6)
    ax[2].set_ylim(-0.4, 11.0)
    fs.panel(ax[2], "(c)", dx=-0.20, dy=1.08)
    ax[2].text(0.97, 0.05,
               f"Spread at fixed lift\nGeometric   {geo['total'].max()-geo['total'].min():4.1f} dB\n"
               f"Lift-effective  {eff['total'].max()-eff['total'].min():4.1f} dB",
               transform=ax[2].transAxes, fontsize=6.5, ha="right", va="bottom",
               linespacing=1.35,
               bbox=dict(fc="white", ec=fs.GRID, lw=0.45, pad=2.2, alpha=0.94))

    fs.save(fig, "fig_mechanism")

    out = M.copy()
    out["SPL_geom"] = geo["total"].values
    out["SPL_eff"] = eff["total"].values
    for k in ("p", "s", "a"):
        out[f"SPL_{k}_geom"] = geo[k].values
    out.to_csv("mechanism_manifold.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nfixed-lift spread: geometric {geo['total'].max()-geo['total'].min():.2f} dB, "
          f"lift-effective {eff['total'].max()-eff['total'].min():.2f} dB")
    print(f"delta*_s at the quiet end vs the loud end: "
          f"{100*(lo.dstar_s/hi.dstar_s-1):+.0f}%")


if __name__ == "__main__":
    build()
