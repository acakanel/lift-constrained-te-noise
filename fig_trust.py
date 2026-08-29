#!/usr/bin/env python3
"""The trust diagnostic: does the predictive spread know where the model is wrong?"""
from __future__ import annotations
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from surrogate import _RBFGP, _FEATS, DATASET, SPL_COL
import figstyle as fs

HERE = os.path.dirname(os.path.abspath(__file__))
fs.use()

d = pd.read_csv(os.path.join(HERE, DATASET)).dropna().reset_index(drop=True)
X = d[_FEATS].values.astype(float)
yCL = d["CL"].values.astype(float)
ySPL = d[SPL_COL].values.astype(float)

U_LEVELS = sorted(d["U"].unique())
U_HELD = float(U_LEVELS[-1])
U_IN = float(U_LEVELS[-2])

tCL = _RBFGP([2.5, 1.5, 1.5, 2.0], 1e-3).fit(X, yCL)
tSPL = _RBFGP([2.5, 1.5, 1.5, 1.5], 1e-3).fit(X, ySPL)
mask = X[:, 3] < U_HELD - 1e-9
rCL = _RBFGP([2.5, 1.5, 1.5, 2.0], 1e-3).fit(X[mask], yCL[mask])
rSPL = _RBFGP([2.5, 1.5, 1.5, 1.5], 1e-3).fit(X[mask], ySPL[mask])

LO = np.array([0.0, -10.0, -10.0])
HI = np.array([12.0, 10.0, 10.0])
RATE = np.array([10.0, 60.0, 60.0])
DT = 0.01


rng = np.random.RandomState(3)
CFG = np.column_stack([rng.uniform(2.0, 11.0, 48),
                       rng.uniform(-9.0, 9.0, 48),
                       rng.uniform(-9.0, 9.0, 48)])
U_SWEEP = np.linspace(U_LEVELS[0], U_HELD + 5.0, 31)

spread = np.zeros_like(U_SWEEP)
jerr = np.zeros_like(U_SWEEP)
for i, U in enumerate(U_SWEEP):
    P = np.column_stack([CFG, np.full(len(CFG), U)])
    spread[i] = float(np.mean(rSPL.std(P)))
    e = []
    for p in P:
        jt = np.asarray(tCL.grad(p))[:3]
        jr = np.asarray(rCL.grad(p))[:3]
        e.append(np.linalg.norm(jr - jt) / (np.linalg.norm(jt) + 1e-12))
    jerr[i] = 100.0 * np.mean(e)


rng2 = np.random.RandomState(11)
cal_sd, cal_err, cal_in = [], [], []
for U in np.concatenate([np.linspace(U_LEVELS[0], U_IN, 7),
                         np.linspace(U_IN + 2.0, U_HELD + 5.0, 7)]):
    Q = np.column_stack([rng2.uniform(1.0, 11.5, 36),
                         rng2.uniform(-9.5, 9.5, 36),
                         rng2.uniform(-9.5, 9.5, 36),
                         np.full(36, U)])
    sd = rSPL.std(Q)
    err = np.abs(rSPL.predict(Q) - tSPL.predict(Q))
    cal_sd.append(sd); cal_err.append(err)
    cal_in.append(np.full(36, U <= U_IN + 1e-9))
cal_sd = np.concatenate(cal_sd)
cal_err = np.concatenate(cal_err)
cal_in = np.concatenate(cal_in)

fig = plt.figure(figsize=(fs.W15, 0.42 * fs.W15))
gs = fig.add_gridspec(1, 2, wspace=0.42, left=0.085, right=0.915,
                      bottom=0.20, top=0.86)
ax = [fig.add_subplot(gs[0, k]) for k in range(2)]

ax[0].axvspan(U_IN, U_SWEEP[-1], color="#f4eef2", lw=0, zorder=0)
ax[0].axvline(U_IN, color=fs.INK2, lw=0.7, ls=(0, (1.4, 1.4)), zorder=2)
ax[0].text(U_SWEEP[-1] - 0.5, 0.60, "Beyond the training data", transform=
           ax[0].get_xaxis_transform(), va="top", ha="right", fontsize=6.4,
           color=fs.INK2)
ax[0].text(U_IN - 0.4, 0.60, "trained\nhere", transform=
           ax[0].get_xaxis_transform(), va="top", ha="right", fontsize=6.0,
           color=fs.INK2, linespacing=1.25)
ax[0].axvline(U_HELD, color=fs.INK2, lw=0.5, ls=(0, (0.8, 1.6)), zorder=2)
ax[0].text(U_HELD, 0.03, "withheld\nslice", transform=
           ax[0].get_xaxis_transform(), va="bottom", ha="center", fontsize=6.0,
           color=fs.INK2, linespacing=1.25)
l1, = ax[0].plot(U_SWEEP, spread, color=fs.C[0], ls=fs.DASH[0], lw=1.2,
                 zorder=3, label="Predictive spread")
ax[0].set_xlabel("Free-stream speed  [m/s]")
ax[0].set_ylabel("Predictive spread  [dB]", color=fs.C[0])
ax[0].tick_params(axis="y", colors=fs.C[0])
a2 = ax[0].twinx()
l2, = a2.plot(U_SWEEP, jerr, color=fs.C[1], ls=fs.DASH[1], lw=1.2, zorder=3,
              label="True lift-Jacobian error")
a2.set_ylabel("Lift-Jacobian error  [%]", color=fs.C[1])
a2.tick_params(axis="y", colors=fs.C[1])
a2.grid(False)
ax[0].set_title("The model knows where it is wrong", pad=3)
ax[0].legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper left",
             fontsize=6.4)

lim = [3e-3, max(cal_sd.max(), cal_err.max()) * 1.6]
ax[1].plot(lim, lim, color=fs.INK2, lw=0.7, ls=(0, (1.4, 1.4)), zorder=2)
ax[1].text(lim[1] * 0.92, lim[1] * 0.92, "1:1", fontsize=6.2, color=fs.INK2,
           ha="right", va="bottom", rotation=45)
ax[1].scatter(cal_sd[cal_in], cal_err[cal_in], s=5, color=fs.C[0], alpha=0.55,
              linewidths=0, zorder=3, label="Inside the training envelope")
ax[1].scatter(cal_sd[~cal_in], cal_err[~cal_in], s=5, color=fs.C[1], alpha=0.55,
              linewidths=0, zorder=3, label="Beyond it")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlim(*lim); ax[1].set_ylim(*lim)
ax[1].set_xlabel("Predictive spread  [dB]")
ax[1].set_ylabel("Actual prediction error  [dB]")
ax[1].set_title("And by how much", pad=3)
ax[1].legend(loc="upper left", fontsize=6.4, markerscale=1.8)

for k, tag in enumerate(["(a)", "(b)"]):
    fs.panel(ax[k], tag, dx=-0.19, dy=1.07)

fs.save(fig, "fig_trust")
plt.close(fig)
print(f"spread: {spread[0]:.3f} dB at U={U_SWEEP[0]:.0f} -> "
      f"{spread[-1]:.3f} dB at U={U_SWEEP[-1]:.0f}")
print(f"Jacobian error: {jerr[0]:.1f} % -> {jerr[-1]:.1f} %")
i70 = int(np.argmin(np.abs(U_SWEEP - U_HELD)))
print(f"at the withheld slice U={U_HELD:.0f}: spread {spread[i70]:.2f} dB, "
      f"Jacobian error {jerr[i70]:.0f} %")
print(f"calibration: inside the envelope, spread {np.median(cal_sd[cal_in]):.3f} dB "
      f"vs error {np.median(cal_err[cal_in]):.3f} dB; beyond it, "
      f"{np.median(cal_sd[~cal_in]):.2f} vs {np.median(cal_err[~cal_in]):.2f} dB")
print("saved figs/fig_trust.pdf and .png")

import csv as _csv
with open(os.path.join(HERE, "trust_diagnostic.csv"), "w", newline="") as _f:
    _w = _csv.writer(_f)
    _w.writerow(["quantity", "value", "unit"])
    _w.writerow(["spread_at_lowest_U", f"{spread[0]:.4f}", "dB"])
    _w.writerow(["jacobian_error_at_lowest_U", f"{jerr[0]:.3f}", "percent"])
    _w.writerow(["spread_at_withheld_slice", f"{spread[i70]:.4f}", "dB"])
    _w.writerow(["jacobian_error_at_withheld_slice", f"{jerr[i70]:.2f}", "percent"])
    _w.writerow(["calibration_spread_inside", f"{np.median(cal_sd[cal_in]):.4f}", "dB"])
    _w.writerow(["calibration_error_inside", f"{np.median(cal_err[cal_in]):.4f}", "dB"])
    _w.writerow(["calibration_spread_outside", f"{np.median(cal_sd[~cal_in]):.4f}", "dB"])
    _w.writerow(["calibration_error_outside", f"{np.median(cal_err[~cal_in]):.4f}", "dB"])
    _w.writerow(["U_withheld", f"{U_HELD:.1f}", "m/s"])
print("wrote trust_diagnostic.csv")
