#!/usr/bin/env python3
"""The constant-lift manifold in resolved flow."""
from __future__ import annotations
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

import figstyle as fs
import ransdata
import bpm_noise_v2 as v2

fs.use()
CHORD, U_REF = 1.0, 55.0


def split(sweep_csv=None):
    d = ransdata.sweep() if sweep_csv is None else pd.read_csv(sweep_csv)
    d = d[(d.U == U_REF) & d.CL.notna()].copy()
    clean = d[(d.d1 == 0) & (d.d2 == 0)].sort_values("alpha").reset_index(drop=True)
    man = d[d.CL > 1.30].sort_values("alpha").reset_index(drop=True)
    man = man.drop_duplicates(subset=["d1", "d2"], keep="first").reset_index(drop=True)
    return clean, man


def equivalent_incidence(clean):
    c = clean.sort_values("H_s")
    H, A = c.H_s.values, c.alpha.values
    return (lambda h: float(np.interp(h, H, A))), float(H.max())


def build(sweep_csv=None):
    clean, M = split(sweep_csv)
    if len(M) < 3:
        print(f"only {len(M)} manifold points -- rerun when more are available")
        return
    a_of_H, H_valid = equivalent_incidence(clean)

    M["alpha_eq"] = [a_of_H(h) for h in M.H_s]
    M["in_range"] = M.H_s <= H_valid * 1.10
    M["SPL"] = [v2.overall_spl(r.dstar_s, r.dstar_p, U_REF, max(r.alpha_eq, 0.0),
                               CHORD)
                if np.isfinite(r.dstar_s) and np.isfinite(r.dstar_p)
                and np.isfinite(r.alpha_eq) else np.nan
                for r in M.itertuples()]
    cl_sorted = clean.sort_values("CL")
    M["dstar_s_ref"] = np.interp(M.CL.values, cl_sorted.CL.values,
                                 cl_sorted.dstar_s.values)
    M["scale"] = 10 * np.log10(M.dstar_s.values / M.dstar_s_ref.values)
    inside = ((M.CL >= cl_sorted.CL.min()) & (M.CL <= cl_sorted.CL.max())).all()
    if not inside:
        print("WARNING: a manifold lift lies outside the clean-section sweep; "
              "its reference is an endpoint, not an interpolant")
    M["label"] = [f"({int(r.d1):+d}, {int(r.d2):+d})" for r in M.itertuples()]

    fig = plt.figure(figsize=(fs.W2, 0.325 * fs.W2))
    gs = fig.add_gridspec(1, 3, wspace=0.34, width_ratios=[1, 1, 1.06],
                          left=0.058, right=0.982, bottom=0.16, top=0.84)
    ax = [fig.add_subplot(gs[0, i]) for i in range(3)]

    def tag(a, xs, ys, dx=0, dy=6):
        for x, y, lab in zip(xs, ys, M.label):
            a.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points",
                       fontsize=5.9, ha="center", va="bottom", color=fs.INK2)

    ax[0].plot(M.alpha, 1e3 * M.dstar_s, ls=(0, (1.4, 1.4)), lw=0.8, color=fs.C[0],
               marker=fs.MARK[0], ms=4.4, mfc=fs.C[0], mec="white", mew=0.5,
               label="Suction side, $\\delta^*_s$")
    ax[0].plot(M.alpha, 1e3 * M.dstar_p, ls=(0, (1.4, 1.4)), lw=0.8, color=fs.C[2],
               marker=fs.MARK[2], ms=4.0, mfc="white", mec=fs.C[2], mew=0.9,
               label="Pressure side, $\\delta^*_p$")
    tag(ax[0], M.alpha, 1e3 * M.dstar_s)
    ax[0].set_xlabel("Trimmed incidence, $\\alpha$  [deg]")
    ax[0].set_ylabel("Displacement thickness at TE  [mm]")
    ax[0].set_title("Boundary-layer thickness at fixed lift", pad=3)
    ax[0].legend(loc="upper right", fontsize=6.6)
    fs.panel(ax[0], "(a)", dx=-0.21, dy=1.08)
    ref = M[(M.d1 == 0) & (M.d2 == 0)]
    if len(ref):
        r0 = float(ref.dstar_s.iloc[0])
        worst = M.loc[M.dstar_s.idxmax()]
        ax[0].annotate(f"$\\times${worst.dstar_s/r0:.1f} the flaps-neutral value",
                       xy=(worst.alpha, 1e3 * worst.dstar_s), xycoords="data",
                       xytext=(0.97, 0.30), textcoords="axes fraction",
                       fontsize=6.4, ha="right", va="center", color=fs.INK,
                       bbox=dict(fc="white", ec=fs.GRID, lw=0.45, pad=2.0,
                                 alpha=0.94),
                       arrowprops=dict(arrowstyle="-", lw=0.55, color=fs.INK2,
                                       shrinkA=2.0, shrinkB=3.0))

    ax[1].axhspan(clean.H_s.min() - 0.05, H_valid, color="#eef2f6", lw=0, zorder=0)
    ax[1].text(0.98, H_valid - 0.06, "Range covered by the clean-section sweep  ",
               transform=ax[1].get_yaxis_transform(), fontsize=6.2, color=fs.INK2,
               ha="right", va="top")
    ax[1].axhline(2.4, color=fs.C[1], lw=0.7, ls=(0, (2.5, 1.6)))
    ax[1].text(0.02, 2.46, " Separation onset", transform=ax[1].get_yaxis_transform(),
               fontsize=6.3, color=fs.C[1], ha="left", va="bottom")
    ax[1].plot(M.alpha, M.H_s, ls=(0, (1.4, 1.4)), lw=0.8, color=fs.C[0],
               marker=fs.MARK[0], ms=4.4, mfc=fs.C[0], mec="white", mew=0.5,
               label="Suction side, $H_s$")
    ax[1].plot(M.alpha, M.H_p, ls=(0, (1.4, 1.4)), lw=0.8, color=fs.C[2],
               marker=fs.MARK[2], ms=4.0, mfc="white", mec=fs.C[2], mew=0.9,
               label="Pressure side, $H_p$")
    tag(ax[1], M.alpha, M.H_s)
    ax[1].set_xlabel("Trimmed incidence, $\\alpha$  [deg]")
    ax[1].set_ylabel("Shape factor at TE, $H$")
    ax[1].set_title("Separation state at fixed lift", pad=3)
    ax[1].legend(loc="center right", fontsize=6.6)
    fs.panel(ax[1], "(b)", dx=-0.20, dy=1.08)

    ok = M[M.in_range]
    dL_base = M.scale.values
    base = float(M.loc[(M.d1 == 0) & (M.d2 == 0), "SPL"].iloc[0])

    ax[2].axhline(0, color=fs.INK2, lw=0.6)
    ax[2].plot(M.alpha, dL_base, ls=(0, (1.4, 1.4)), lw=0.8, color=fs.C[0],
               marker=fs.MARK[0], ms=4.4, mfc=fs.C[0], mec="white", mew=0.5,
               label="_nolegend_")
    tag(ax[2], M.alpha, dL_base)
    ax[2].set_xlabel("Trimmed incidence, $\\alpha$  [deg]")
    ax[2].set_ylabel("Level relative to flaps neutral,\n"
                     "$10\\log_{10}(\\delta^*_s/\\delta^*_{s,0})$  [dB]")
    ax[2].set_title("Achievable lift-preserving change", pad=3)
    fs.panel(ax[2], "(c)", dx=-0.20, dy=1.08)
    n_sep = int(np.sum(~np.isfinite(dL_base)))
    worst = float(np.nanmax(dL_base))
    msg = f"Every deflection is louder at\nfixed lift; largest {worst:+.1f} dB."
    if n_sep:
        msg += (f"\n{n_sep} separated: no $\\delta^*_s$ exists.")
    ax[2].text(0.97, 0.97, msg,
               transform=ax[2].transAxes, fontsize=6.4, ha="right", va="top",
               linespacing=1.35,
               bbox=dict(fc="white", ec=fs.GRID, lw=0.45, pad=2.2, alpha=0.94))
    for r in M.itertuples():
        if not np.isfinite(getattr(r, "scale", np.nan)):
            ax[2].axvline(r.alpha, color=fs.INK2, lw=0.7, ls=(0, (1.2, 1.6)),
                          zorder=1)
            ax[2].annotate(f"{r.label}\nseparated", xy=(r.alpha, 1.0),
                           xycoords=("data", "axes fraction"),
                           xytext=(3, -12), textcoords="offset points",
                           fontsize=6.0, ha="left", va="top",
                           color=fs.INK2, linespacing=1.25)

    for a in ax:
        lo, hi = a.get_ylim()
        a.set_ylim(lo, hi + 0.16 * (hi - lo))
    fs.save(fig, "fig_manifold_rans")
    M.to_csv("manifold_rans.csv", index=False)
    cols = ["label", "alpha", "CL", "dstar_s", "dstar_s_ref", "scale", "H_s",
            "alpha_eq", "in_range", "SPL"]
    print(M[cols].to_string(index=False))
    print(f"\nclean-section sweep covers H_s up to {H_valid:.2f}")
    d0 = float(M.loc[(M.d1 == 0) & (M.d2 == 0), "dstar_s"].iloc[0])
    base = float(M.loc[(M.d1 == 0) & (M.d2 == 0), "SPL"].iloc[0])
    ok = M[M.in_range]
    print(f"within the calibrated range: best "
          f"{(ok.SPL - base).min():+.2f} dB relative to flaps neutral")
    print("model-independent scale term 10log10(dstar_s/dstar_s0):")
    for r in M.itertuples():
        print(f"   {r.label:11s} {r.scale:+6.2f} dB   "
              f"H_s {r.H_s:.2f}   {'calibrated' if r.in_range else 'outside range'}")
    print(f"thickest trailing-edge boundary layer: {M.loc[M.dstar_s.idxmax(),'label']} "
          f"at {1e3*M.dstar_s.max():.1f} mm, H_s = {M.H_s.max():.2f}")


if __name__ == "__main__":
    build()
