#!/usr/bin/env python3
"""Fidelity of the aerodynamic input to the aeroacoustic model."""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

import figstyle as fs
import ransdata
from bpm_bl import dstar0_over_c, ratio_suction, ratio_pressure

fs.use()
CHORD, NU = 1.0, 1.5e-5


def neuralfoil_sweep(alphas, U=55.0, d1=0.0, d2=0.0, cache="nf_sweep.csv"):
    if os.path.exists(cache):
        c = pd.read_csv(cache)
        if set(np.round(alphas, 3)) <= set(np.round(c.alpha, 3)):
            return c
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from aero_dataset import run_case
    rows = []
    for a in alphas:
        CL, CM, ds, dp = run_case(float(a), d1, d2, U)
        rows.append(dict(alpha=a, CL=CL, CM=CM, dstar_s=ds, dstar_p=dp))
    c = pd.DataFrame(rows)
    c.to_csv(cache, index=False)
    return c


def build(sweep_csv=None, U=55.0):
    R = ransdata.sweep() if sweep_csv is None else pd.read_csv(sweep_csv)
    R = R[(R.d1 == 0) & (R.d2 == 0)].sort_values("alpha").reset_index(drop=True)
    N = neuralfoil_sweep(R.alpha.values, U=U).sort_values("alpha").reset_index(drop=True)

    Rec = U * CHORD / NU
    a_fine = np.linspace(0, R.alpha.max(), 120)
    d0_u = float(dstar0_over_c(Rec, tripped=False)) * CHORD
    corr_u = d0_u * ratio_suction(a_fine, tripped=False)

    fig = plt.figure(figsize=(fs.W2, 0.315 * fs.W2))
    gs = fig.add_gridspec(1, 3, wspace=0.33, width_ratios=[1, 1.12, 1.12],
                          left=0.058, right=0.995, bottom=0.155, top=0.845)
    ax = [fig.add_subplot(gs[0, i]) for i in range(3)]

    ax[0].plot(N.alpha, N.CL, color=fs.C[1], ls=fs.DASH[1], marker=fs.MARK[1],
               ms=3.6, mfc="white", mew=0.8, label=fs.L_SURR)
    ax[0].plot(R.alpha, R.CL, color=fs.C[0], ls=fs.DASH[0], marker=fs.MARK[0],
               ms=3.4, mfc=fs.C[0], mec="white", mew=0.45, label="RANS ($k$\u2013$\\omega$ SST)")
    ax[0].set_xlabel("Angle of attack, $\\alpha$  [deg]")
    ax[0].set_ylabel("Lift coefficient, $C_L$")
    ax[0].set_title("Lift coefficient", pad=3)
    ax[0].legend(loc="upper left")
    fs.panel(ax[0], "(a)", dx=-0.22, dy=1.08)
    err = 100 * np.abs(np.interp(R.alpha, N.alpha, N.CL) - R.CL) / np.maximum(np.abs(R.CL), 1e-6)
    ax[0].text(0.97, 0.05, f"Mean discrepancy {np.nanmean(err[R.alpha > 1]):.1f}%",
               transform=ax[0].transAxes, fontsize=6.6, ha="right", va="bottom",
               bbox=dict(fc="white", ec=fs.GRID, lw=0.45, pad=2.2, alpha=0.94))

    ax[1].plot(N.alpha, 1e3 * N.dstar_s, color=fs.C[1], ls=fs.DASH[1],
               marker=fs.MARK[1], ms=3.6, mfc="white", mew=0.8,
               label="NeuralFoil, suction side")
    ax[1].plot(N.alpha, 1e3 * N.dstar_p, color=fs.C[1], ls=fs.DASH[2],
               marker=fs.MARK[1], ms=3.0, mfc=fs.C[1], mec="white", mew=0.4,
               alpha=0.8, label="NeuralFoil, pressure side")
    ax[1].plot(R.alpha, 1e3 * R.dstar_s, color=fs.C[0], ls=fs.DASH[0],
               marker=fs.MARK[0], ms=3.4, mfc=fs.C[0], mec="white", mew=0.45,
               label="RANS, suction side")
    ax[1].plot(R.alpha, 1e3 * R.dstar_p, color=fs.C[0], ls=fs.DASH[3],
               marker=fs.MARK[2], ms=3.2, mfc="white", mec=fs.C[0], mew=0.7,
               label="RANS, pressure side")
    ax[1].set_xlabel("Angle of attack, $\\alpha$  [deg]")
    ax[1].set_ylabel("Displacement thickness at TE, $\\delta^*$  [mm]")
    ax[1].set_title("Displacement thickness at the trailing edge", pad=3)
    ax[1].legend(loc="upper left", fontsize=6.5)
    fs.panel(ax[1], "(b)", dx=-0.20, dy=1.08)

    nf_fac = float(N.dstar_s.iloc[-1] / N.dstar_s.iloc[0])
    ra_fac = float(R.dstar_s.iloc[-1] / R.dstar_s.iloc[0])
    a_lo, a_hi = float(R.alpha.min()), float(R.alpha.max())
    ax[1].text(0.045, 0.615,
               f"Suction side, ${a_lo:g}^\\circ\\!\\to\\!{a_hi:g}^\\circ$\n"
               f"NeuralFoil  $\\times{nf_fac:.1f}$\n"
               f"RANS         $\\times{ra_fac:.1f}$",
               transform=ax[1].transAxes, fontsize=6.5, ha="left", va="top",
               linespacing=1.35,
               bbox=dict(fc="white", ec=fs.GRID, lw=0.45, pad=2.2, alpha=0.94))

    import bpm_noise_v2 as v2
    spl_r = [v2.overall_spl(r.dstar_s_, r.dstar_p_, U, r.alpha_, CHORD)
             for r in (pd.DataFrame(dict(alpha_=R.alpha, dstar_s_=R.dstar_s,
                                         dstar_p_=R.dstar_p)).itertuples())]
    spl_n = [v2.overall_spl(r.dstar_s_, r.dstar_p_, U, r.alpha_, CHORD)
             for r in (pd.DataFrame(dict(alpha_=N.alpha, dstar_s_=N.dstar_s,
                                         dstar_p_=N.dstar_p)).itertuples())]
    spl_r, spl_n = np.array(spl_r), np.array(spl_n)

    ax[2].plot(N.alpha, spl_n - spl_n[0], color=fs.C[1], ls=fs.DASH[1],
               marker=fs.MARK[1], ms=3.6, mfc="white", mew=0.8,
               label=fs.L_SURR)
    ax[2].plot(R.alpha, spl_r - spl_r[0], color=fs.C[0], ls=fs.DASH[0],
               marker=fs.MARK[0], ms=3.4, mfc=fs.C[0], mec="white", mew=0.45,
               label="RANS ($k$\u2013$\\omega$ SST)")
    ax[2].set_xlabel("Angle of attack, $\\alpha$  [deg]")
    ax[2].set_ylabel("Level rise from $\\alpha=0$, $\\Delta$OASPL  [dB]")
    ax[2].set_title("Predicted overall level", pad=3)
    ax[2].legend(loc="lower right", fontsize=6.5)
    fs.panel(ax[2], "(c)", dx=-0.20, dy=1.08)

    amax = float(R.alpha.max())
    gap = (spl_n[-1] - spl_n[0]) - (spl_r[-1] - spl_r[0])
    ax[2].annotate("", xy=(amax, spl_n[-1] - spl_n[0]), xytext=(amax, spl_r[-1] - spl_r[0]),
                   arrowprops=dict(arrowstyle="<->", lw=0.6, color=fs.INK2,
                                   shrinkA=1.5, shrinkB=1.5))
    ax[2].text(amax - 0.35, 0.5 * ((spl_n[-1] - spl_n[0]) + (spl_r[-1] - spl_r[0])),
               f"{gap:.1f} dB", fontsize=6.8, ha="right", va="center", color=fs.INK)
    ax[2].text(0.035, 0.955,
               f"Level rise, $0^\\circ\\!\\to\\!{amax:g}^\\circ$\n"
               f"NeuralFoil  {spl_n[-1]-spl_n[0]:5.1f} dB\n"
               f"RANS        {spl_r[-1]-spl_r[0]:5.1f} dB",
               transform=ax[2].transAxes, fontsize=6.5, ha="left", va="top",
               linespacing=1.35,
               bbox=dict(fc="white", ec=fs.GRID, lw=0.45, pad=2.2, alpha=0.94))

    fs.save(fig, "fig_fidelity_ladder")

    out = R[["alpha", "CL", "CM", "dstar_s", "dstar_p", "H_s", "H_p"]].copy()
    out.columns = ["alpha", "CL_rans", "CM_rans", "dstar_s_rans", "dstar_p_rans",
                   "H_s_rans", "H_p_rans"]
    out["CL_nf"] = np.interp(out.alpha, N.alpha, N.CL)
    out["dstar_s_nf"] = np.interp(out.alpha, N.alpha, N.dstar_s)
    out["dstar_p_nf"] = np.interp(out.alpha, N.alpha, N.dstar_p)
    out.to_csv("fidelity_comparison.csv", index=False)
    print(out.to_string(index=False))
    nr = float(N.dstar_s.iloc[-1] / N.dstar_s.iloc[0])
    rr = float(R.dstar_s.iloc[-1] / R.dstar_s.iloc[0])
    print(f"\ndelta*_s sensitivity 0 -> {amax:g} deg:  NeuralFoil x{nr:.2f}   RANS x{rr:.2f}")
    print(f"noise rise 0 -> {amax:g} deg:  NeuralFoil {spl_n[-1]-spl_n[0]:.1f} dB   "
          f"RANS {spl_r[-1]-spl_r[0]:.1f} dB   (difference {gap:.1f} dB)")


if __name__ == "__main__":
    build()
