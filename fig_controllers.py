#!/usr/bin/env python3
"""The constraint layer on the dynamic plant."""
from __future__ import annotations
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import band as _band, pandas as pd
import matplotlib.pyplot as plt

import figstyle as fs

fs.use()
BAND = _band.EPS
ORDER = ["SG-FTSMC", "BF-STA", "CLF-QP", "MPC"]
NAMES = {"SG-FTSMC": "Fixed-time super-twisting",
         "BF-STA": "Barrier-function adaptive",
         "CLF-QP": "Control-Lyapunov-function QP",
         "MPC": "Model predictive"}


def build(csv="controller_comparison.csv"):
    d = pd.read_csv(csv)
    d["op"] = [f"$U={u:.0f}$, $C_L={c:.2f}$" for u, c in zip(d.U, d.CLreq)]
    ops = list(dict.fromkeys(d.op))
    import pandas as _pd
    df = _pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "descent_fraction.csv"))
    key = ["U", "CLreq", "controller"]
    d = d.merge(df[key + ["available_descent_dB", "fraction_realised_pct"]],
                on=key, how="left")
    missing = d.fraction_realised_pct.isna()
    if missing.any():
        raise SystemExit(
            "descent_fraction.csv has no row for: "
            + ", ".join(f"{r.U:.0f}/{r.CLreq:.2f}/{r.controller}"
                        for r in d[missing].itertuples())
            + "\n  Run descent_fraction.py first.")
    d["available"] = d.available_descent_dB
    d["frac"] = d.fraction_realised_pct
    print("available descent at each operating point (from descent_fraction.csv):")
    for (u, cl), g in d.groupby(["U", "CLreq"]):
        print(f"  U={u:.0f}, C_L={cl:.2f}: {g.available.iloc[0]:.2f} dB")

    fig = plt.figure(figsize=(fs.W2, 0.33 * fs.W2))
    gs = fig.add_gridspec(1, 2, wspace=0.22, width_ratios=[1, 1],
                          left=0.062, right=0.985, bottom=0.17, top=0.83)
    ax = [fig.add_subplot(gs[0, i]) for i in range(2)]

    x = np.arange(len(ops))
    w = 0.19
    for k, ctl in enumerate(ORDER):
        for panel, col in ((0, "frac"), (1, "lift_max")):
            for j, cond in enumerate(("nominal", "model_err_25pct")):
                sub = d[(d.controller == ctl) & (d.condition == cond)]
                v = [sub[sub.op == o][col].values for o in ops]
                v = [float(a[0]) if len(a) else np.nan for a in v]
                xpos = x + (k - 1.5) * w + (j - 0.5) * w * 0.42
                ax[panel].bar(xpos, v, w * 0.42, color=fs.C[k],
                              alpha=1.0 if j == 0 else 0.45,
                              edgecolor="white", linewidth=0.4,
                              label=NAMES[ctl] if (panel == 0 and j == 0) else None,
                              zorder=3)
                if panel == 1:
                    r = [sub[sub.op == o]["lift_RMS"].values for o in ops]
                    r = [float(a[0]) if len(a) else np.nan for a in r]
                    for xi, ri in zip(xpos, r):
                        ax[1].plot([xi - 0.20 * w, xi + 0.20 * w], [ri, ri],
                                   color=fs.INK, lw=0.7, solid_capstyle="butt",
                                   zorder=6)

    lo = float(np.floor(d["frac"].min() / 5.0) * 5.0) - 5.0
    ax[0].set_ylim(lo, 102)
    ax[0].axhline(lo, color=fs.INK2, lw=0.8, zorder=5)
    ax[0].text(0.988, 0.035, f"axis starts at {lo:.0f}%",
               transform=ax[0].transAxes, fontsize=6.0, ha="right", va="bottom",
               color=fs.INK2)
    ax[0].axhline(100, color=fs.INK2, lw=0.6, zorder=2)
    ax[0].set_ylabel("Fraction realised of the\navailable descent  [%]")
    ax[0].set_title("The descent is controller-agnostic", pad=3)
    fs.panel(ax[0], "(a)", dx=-0.12, dy=1.06)

    ax[1].axhline(BAND, color=fs.C[1], lw=0.9, ls=(0, (2.5, 1.6)), zorder=4)
    ax[1].text(len(ops) - 0.45, BAND * 1.04,
               f"Band, $\\varepsilon={BAND:g}$ (certified for the barrier law)  ",
               fontsize=6.4, color=fs.C[1], ha="right", va="bottom")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("Settled lift error,  $|s|$")
    ax[1].set_title("The constraint is held in every case", pad=3)
    ax[1].text(0.015, 0.95, "Bars: settled maximum.   Ticks: settled RMS.",
               transform=ax[1].transAxes, fontsize=6.4, ha="left", va="top",
               color=fs.INK2)
    fs.panel(ax[1], "(b)", dx=-0.13, dy=1.06)

    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(ops, fontsize=6.8)
        a.grid(axis="x", visible=False)
    ax[0].legend(loc="lower center", bbox_to_anchor=(1.06, -0.30), ncol=4,
                 fontsize=6.5, frameon=False)
    ax[0].text(0.015, 0.045,
               "Solid: nominal.   Faded: 25% lift-model error.",
               transform=ax[0].transAxes, fontsize=6.4, ha="left", va="bottom",
               color=fs.INK2)

    fs.save(fig, "fig_controllers")

    print(d.groupby(["op", "controller"])[["frac", "lift_RMS"]].mean().to_string())
    within = d.groupby(["op", "condition"]).frac.agg(lambda x: x.max()-x.min())
    print(f"\nrealised fraction, by operating point: "
          f"{d.groupby('op').frac.mean().min():.0f} % to "
          f"{d.groupby('op').frac.mean().max():.0f} %")
    print(f"spread across the four controllers at a fixed operating point and "
          f"condition: at most {within.max():.2f} percentage points")
    print(f"largest settled lift-error RMS: {d.lift_RMS.max():.4f} "
          f"({100*d.lift_RMS.max()/BAND:.0f}% of the certified band)")
    print(f"all within band: {bool((d.lift_RMS < BAND).all())}")


if __name__ == "__main__":
    build()
