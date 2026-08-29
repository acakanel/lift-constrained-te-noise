#!/usr/bin/env python3
"""Deployment figure, built from what the driver measured."""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import figstyle as fs                                       # noqa: E402

fs.use()
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import band as _band
BAND = _band.EPS
import json as _json
_res = _json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                     "pil_results.json")))
FID_REF = float(_res.get("closed_loop", {}).get("settled_lift_band", BAND))
DT = 0.01


def load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        raise SystemExit(f"{name} not found; run pil_host.py first")
    return p


res = json.load(open(load("pil_results.json")))
pil = np.loadtxt(load("pil_loop.csv"), delimiter=",", skiprows=1)
ref = np.loadtxt(load("ref_loop.csv"), delimiter=",", skiprows=1)
cl = np.loadtxt(load("pil_closedloop.csv"), delimiter=",", skiprows=1)

fig = plt.figure(figsize=(fs.W2, 0.34 * fs.W2))
gs = fig.add_gridspec(1, 3, wspace=0.32, left=0.062, right=0.988,
                      bottom=0.185, top=0.865)
ax = [fig.add_subplot(gs[0, k]) for k in range(3)]

b = res.get("bench", {})
clk = b.get("clk", 0)
if clk:
    us = lambda c: 1e6 * c / clk
    names = ["Inner step", "Constraint model\n(timed in isolation)",
             "Confidence gate", "Set-point map"]
    inner = us(b["inner"][1]); gpv = us(b["gp"][1])
    sig = us(b["sigma"][1]); mp = us(b["map_mean"])
    hi = [us(b["inner"][2]), us(b["gp"][2]), us(b["sigma"][2]), mp]
    vals = [inner, gpv, sig, mp]
    cols = [fs.C[0], fs.C[0], fs.C[2], fs.C[3]]
    hat = ["", "//", "", ""]
    y = np.arange(len(names))[::-1]
    lo = 10 ** np.floor(np.log10(min(v for v in vals if v > 0))) / 2.0
    for k, (v, c, h, vmax) in enumerate(zip(vals, cols, hat, hi)):
        ax[0].barh(y[k], max(v - lo, 0.0), left=lo, height=0.62, color=c,
                   hatch=h, edgecolor="white", linewidth=0.6, zorder=3)
        ax[0].plot([vmax], [y[k]], marker="|", ms=5.5, color=fs.INK, zorder=5)
        if max(v, vmax) > 0.12 * 1e6 * DT:
            ax[0].text(v * 0.92, y[k], f"{v:.0f} $\\mu$s", va="center",
                       ha="right", fontsize=6.4, color="white", zorder=6)
        else:
            ax[0].text(max(v, vmax) * 1.10, y[k], f"{v:.0f} $\\mu$s",
                       va="center", fontsize=6.4, color=fs.INK)
    ax[0].axvline(1e6 * DT, color=fs.C[1], lw=1.0, ls=(0, (2.5, 1.6)), zorder=4)
    ax[0].text(1e6 * DT * 0.78, 0.5 * (y[0] + y[-1]), "10 ms step budget",
               rotation=90, ha="right", va="center", fontsize=6.4,
               color=fs.C[1])
    ax[0].set_xscale("log")
    ax[0].set_xlim(lo, 2.2 * 1e6 * DT)
    ax[0].set_ylim(y[-1] - 0.95, y[0] + 0.55)
    ax[0].set_yticks(y); ax[0].set_yticklabels(names, fontsize=6.4)
    ax[0].set_xlabel("Execution time  [$\\mu$s]")
    ax[0].set_title(f"Cortex-M7 at {clk/1e6:.0f} MHz", pad=3)
    ax[0].text(0.5, 0.015,
               f"Inner step uses {100*inner/1e6/DT:.1f} % of the budget",
               transform=ax[0].transAxes, fontsize=6.4, va="bottom",
               ha="center",
               bbox=dict(fc="white", ec=fs.GRID, lw=0.45, pad=2.0, alpha=0.94))
else:
    ax[0].text(0.5, 0.5, "Timing requires the target;\nthis run used the"
               "\nworkstation stand-in", ha="center", va="center",
               fontsize=7.2, color=fs.INK2, transform=ax[0].transAxes,
               linespacing=1.5)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    for sp in ax[0].spines.values():
        sp.set_visible(False)
    ax[0].grid(False)
    ax[0].set_title("Execution time", pad=3)

lab = ["Steady high-lift point", "Speed and lift transient"]
du_max = 0.0
for sc in (0, 1):
    A = pil[pil[:, 0] == sc]; B = ref[ref[:, 0] == sc]
    m = min(len(A), len(B))
    du_max = max(du_max, float(np.abs(A[:m, 3:6] - B[:m, 3:6]).max()))
    ds = np.abs(A[:m, 7] - B[:m, 7])
    ax[1].semilogy(A[:m, 2], np.maximum(ds, 1e-9), lw=1.0, color=fs.C[sc],
                   ls=fs.DASH[sc], zorder=3, label=lab[sc])
ax[1].axhline(FID_REF, color=fs.C[1], lw=0.9, ls=(0, (1.4, 1.4)), zorder=2)
ax[1].text(0.98, FID_REF * 1.5,
           f"Largest lift excursion of the\ndeployed loop, {FID_REF:.4f}",
           transform=ax[1].get_yaxis_transform(), ha="right", fontsize=6.4,
           color=fs.C[1])
ax[1].set_xlabel("Time  [s]")
ax[1].set_ylabel("Lift departure from the\ndouble-precision reference,  $|\\Delta s|$")
ax[1].set_title("Single precision on the target", pad=3)
ax[1].legend(loc="lower left", fontsize=6.4)
ax[1].set_ylim(1e-7, max(FID_REF * 6.0, 1e-1))
ax[1].text(0.02, 0.78, f"Surface angles depart by at most {du_max:.2f}$^\\circ$",
           transform=ax[1].transAxes, fontsize=6.2, color=fs.INK2,
           ha="left", va="top",
           bbox=dict(fc="white", ec="none", pad=1.4, alpha=0.90))

t = cl[:, 1]
ax[2].axhspan(0, BAND, color="#eef2f6", lw=0, zorder=0)
ax[2].plot(t, cl[:, 6], lw=1.0, color=fs.C[0], zorder=3, label="Lift error")
ax[2].axhline(BAND, color=fs.C[1], lw=0.9, ls=(0, (2.5, 1.6)), zorder=4,
              label=f"Band, $\\varepsilon={BAND:g}$")
ax[2].set_xlabel("Time  [s]")
ax[2].set_ylabel("Lift error,  $|s|$")
ax[2].set_ylim(0, BAND * 1.12)
a2 = ax[2].twinx()
a2.plot(t, cl[:, 7], lw=1.0, color=fs.C[2], ls=fs.DASH[1], zorder=3,
        label="Objective")
a2.set_ylabel("Objective  [dB]", color=fs.C[2])
a2.tick_params(axis="y", colors=fs.C[2])
a2.set_ylim(cl[:, 7].min() - 0.25, cl[:, 7].max() + 0.35)
ax[2].set_title("Closed loop across the link", pad=3)
h1, l1 = ax[2].get_legend_handles_labels()
h2, l2 = a2.get_legend_handles_labels()
ax[2].legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.4,
             framealpha=0.94)

mc = res.get("monte_carlo")
note = (f"{mc['n']} trials: descent {mc['descent_mean']:.2f} dB,\n"
        f"worst lift error {mc['band_max']:.3f}"
        if mc else
        f"Descent {res['closed_loop']['descent_dB']:.2f} dB,\n"
        f"worst lift error {res['closed_loop']['settled_lift_band']:.3f}")
ax[2].text(0.97, 0.52, note, transform=ax[2].transAxes, fontsize=6.4,
           va="center", ha="right", linespacing=1.35,
           bbox=dict(fc="white", ec=fs.GRID, lw=0.45, pad=2.2, alpha=0.94))

for k, tag in enumerate(["(a)", "(b)", "(c)"]):
    fs.panel(ax[k], tag, dx=-0.14, dy=1.07)

fs.save(fig, "fig_deployment")
plt.close(fig)
print("saved figs/fig_deployment.pdf and .png")
