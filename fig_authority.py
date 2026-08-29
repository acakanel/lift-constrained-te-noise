#!/usr/bin/env python3
"""The lift-preserving trailing-edge noise authority of the section, measured in resolved flow at two lift levels."""
from __future__ import annotations
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

import figstyle as fs
import ransdata

fs.use()
U_REF = 55.0
L_MOD = "Moderate lift, $C_L\\approx0.71$"
L_HI = "High lift, $C_L\\approx1.34$"
PATHS = [(L_MOD, 0.55, 0.85),
         (L_HI, 1.30, 1.40)]


def paths(sweep_csv=None):
    d = ransdata.sweep() if sweep_csv is None else pd.read_csv(sweep_csv)
    d = d[(d.U == U_REF) & d.CL.notna() & (d.d2 == 0)]
    out = []
    for label, lo, hi in PATHS:
        m = d[(d.CL > lo) & (d.CL < hi)].copy()
        target = 0.5 * (lo + hi)
        m["err"] = (m.CL - target).abs()
        m = m.sort_values("err").drop_duplicates("d1", keep="first").sort_values("d1")
        if len(m) >= 2:
            out.append((label, m.reset_index(drop=True)))
    return out


def _neutral(sweep_csv):
    d = ransdata.sweep() if sweep_csv is None else pd.read_csv(sweep_csv)
    d = d[(d.U == U_REF) & d.CL.notna() & (d.d2 == 0) & (d.d1 == 0)]
    return d.sort_values("CL").reset_index(drop=True)


GEO = {L_MOD: 0.102,
       L_HI: 0.389}
SYM_FLOOR = 0.094


def _numerical_terms():
    import re
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "paper_AST", "conv_numbers.tex")
    m = {}
    if os.path.exists(p):
        for k, v in re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(p).read()):
            m[k] = v
    hi = float(m.get("cvStepWorst", m.get("cvStepBC", SYM_FLOOR)))
    mo = float(m["cvMStepWorst"]) if "cvMStepWorst" in m else SYM_FLOOR
    return {L_HI: hi, L_MOD: mo}


_NUM = _numerical_terms()
UNC = {k: float(np.hypot(_NUM[k], GEO[k])) for k in GEO}


def build(sweep_csv=None):
    P = paths(sweep_csv)
    if not P:
        print("no complete deflection path available yet")
        return

    fig = plt.figure(figsize=(fs.W2, 0.325 * fs.W2))
    gs = fig.add_gridspec(1, 3, wspace=0.33, width_ratios=[1, 1, 1.04],
                          left=0.058, right=0.982, bottom=0.16, top=0.84)
    ax = [fig.add_subplot(gs[0, i]) for i in range(3)]

    base = _neutral(sweep_csv)
    H_valid = float(base.H_s.max())
    rows = []
    for i, (label, m) in enumerate(P):
        ref = np.interp(m.CL.values, base.CL.values, base.dstar_s.values)
        dL = 10 * np.log10(m.dstar_s.values / ref)
        st = dict(color=fs.C[i], ls=fs.DASH[i], marker=fs.MARK[i],
                  ms=4.4, mfc=fs.C[i] if i == 0 else "white",
                  mec="white" if i == 0 else fs.C[i], mew=0.5 if i == 0 else 1.0)
        ax[0].plot(m.d1, 1e3 * m.dstar_s, label=label, **st)
        ax[1].plot(m.d1, m.H_s, label=label, **st)
        unc = np.where(m.d1.values == 0.0, 0.0, UNC[label])
        nz = m.d1.values != 0.0
        ax[2].fill_between(m.d1.values[nz], (dL - unc)[nz], (dL + unc)[nz],
                           color=fs.C[i], alpha=0.13, lw=0, zorder=1)
        ax[2].errorbar(m.d1.values[nz], dL[nz], yerr=unc[nz], fmt="none",
                       ecolor=fs.C[i], elinewidth=0.8, capsize=2.4,
                       capthick=0.8, zorder=4, alpha=0.85)
        ax[2].plot(m.d1, dL, label=label, zorder=5, **st)
        for r, y, rf in zip(m.itertuples(), dL, ref):
            rows.append(dict(path=label, d1=r.d1, alpha=r.alpha, CL=r.CL,
                             dstar_s=r.dstar_s, H_s=r.H_s,
                             dstar_s_ref_matched=rf, dL=y))

    ax[0].set_xlabel("First-segment deflection, $\\delta_1$  [deg]")
    ax[0].set_ylabel("Displacement thickness at TE, $\\delta^*_s$  [mm]")
    ax[0].set_title("Boundary-layer thickness", pad=3)
    ax[0].legend(loc="upper left", fontsize=6.5)
    fs.panel(ax[0], "(a)", dx=-0.21, dy=1.08)

    ax[1].axhspan(1.4, H_valid, color="#eef2f6", lw=0, zorder=0)
    ax[1].text(0.98, H_valid - 0.02, "Calibrated range  ",
               transform=ax[1].get_yaxis_transform(), fontsize=6.2,
               color=fs.INK2, ha="right", va="top")
    ax[1].axhline(2.4, color=fs.C[1], lw=0.7, ls=(0, (2.5, 1.6)))
    ax[1].text(0.02, 2.44, " Separation onset", transform=ax[1].get_yaxis_transform(),
               fontsize=6.3, color=fs.C[1], ha="left", va="bottom")
    ax[1].set_xlabel("First-segment deflection, $\\delta_1$  [deg]")
    ax[1].set_ylabel("Shape factor at TE, $H_s$")
    ax[1].set_title("Separation state", pad=3)
    ax[1].legend(loc="upper left", fontsize=6.5)
    fs.panel(ax[1], "(b)", dx=-0.20, dy=1.08)

    ax[2].axhline(0, color=fs.INK2, lw=0.6)
    ax[2].set_xlabel("First-segment deflection, $\\delta_1$  [deg]")
    ax[2].set_ylabel("Level change, $10\\log_{10}(\\delta^*_s/\\delta^*_{s,0})$  [dB]")
    ax[2].set_title("Lift-preserving level change", pad=3)
    ax[2].legend(loc="upper left", fontsize=6.5)
    fs.panel(ax[2], "(c)", dx=-0.20, dy=1.08)

    R = pd.DataFrame(rows)
    lab = R[(R.d1 == 5.0)]
    for _, r in lab.iterrows():
        up = r.dL > 0
        tx, ty, ha = ((0.15, UNC[r.path] + 0.30, "left") if up
                      else (0.15, -UNC[r.path] - 0.30, "left"))
        note = ("resolves its sign" if abs(r.dL) > UNC[r.path]
                else "does not resolve its sign")
        ax[2].annotate(f"$\\delta_1=+5^\\circ$: {r.dL:+.2f} $\\pm$ "
                       f"{UNC[r.path]:.2f} dB\n{note}",
                       xy=(r.d1, r.dL), xycoords="data",
                       xytext=(tx, ty), textcoords="data",
                       fontsize=6.6, ha=ha, va="bottom" if up else "top",
                       color=fs.INK, linespacing=1.35,
                       bbox=dict(fc="white", ec=fs.GRID, lw=0.45, pad=2.0,
                                 alpha=0.94),
                       arrowprops=dict(arrowstyle="-", lw=0.55, color=fs.INK2,
                                       shrinkA=2.0, shrinkB=3.0))
    best = R.loc[R.dL.idxmin()]

    for a in ax:
        lo, hi = a.get_ylim()
        a.set_ylim(lo, hi + 0.14 * (hi - lo))
    lo, hi = ax[2].get_ylim()
    ax[2].set_ylim(min(lo, -max(UNC.values()) - 0.85), hi)
    R.to_csv("authority.csv", index=False)
    fs.save(fig, "fig_authority")
    print(R.to_string(index=False))
    print(f"\nbest lift-preserving change: {best.dL:+.2f} dB at "
          f"delta_1 = {best.d1:+.0f} deg ({best.path})")
    for label, g in R.groupby("path", sort=False):
        print(f"  {label}: best {g.dL.min():+.3f} dB, worst {g.dL.max():+.3f} dB, "
              f"lift spread {g.CL.max()-g.CL.min():.4f}")
    lo = R[R.d1 == 5].dL
    print(f"  at delta_1=+5 the two lift levels agree to "
          f"{abs(lo.max()-lo.min()):.3f} dB")


if __name__ == "__main__":
    build()
