#!/usr/bin/env python3
"""The self-noise model against the NASA airfoil self-noise database, on absolute levels."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from bpm_noise import spl_third_octave

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)
d = pd.read_csv(os.path.join(HERE, "NASA_selfnoise.csv"))
plt.rcParams.update({"font.size": 11})

ALPHA_MAX = 12.5
d = d[d["alpha"] <= ALPHA_MAX]
groups = [(k, g) for k, g in d.groupby(["alpha", "c", "U_infinity", "delta"]) if len(g) >= 4]
corrs, npts = [], []
pred_pts, meas_pts = [], []
for (alpha, c, U, delta), g in groups:
    g = g.sort_values("f")
    f = g["f"].values.astype(float); meas = g["SSPL"].values.astype(float)
    pred = np.asarray(spl_third_octave(delta, delta, U, alpha, c, freqs=f))
    ok = np.isfinite(pred) & np.isfinite(meas) & (pred > 0) & (pred < 200)
    if ok.sum() < 4: continue
    p, m = pred[ok], meas[ok]
    corrs.append(np.corrcoef(p, m)[0, 1]); npts.append(ok.sum())
    pred_pts.append(p); meas_pts.append(m)
corrs = np.array(corrs)
shape_rmse = []
for (alpha, c, U, delta), g in groups:
    g = g.sort_values("f"); f = g["f"].values.astype(float); m = g["SSPL"].values.astype(float)
    p = np.asarray(spl_third_octave(delta, delta, U, alpha, c, freqs=f))
    ok = np.isfinite(p) & (p > 0) & (p < 200)
    if ok.sum() < 4: continue
    shape_rmse.append(np.sqrt(np.mean(((p[ok]-p[ok].mean())-(m[ok]-m[ok].mean()))**2)))
shape_rmse = np.array(shape_rmse)
print(f"attached-flow conditions (alpha<= {ALPHA_MAX}): {len(corrs)} spectra, {len(meas_pts)} points")
print(f"within-spectrum correlation: median {np.median(corrs):.3f}, 25th {np.percentile(corrs,25):.3f}")
print(f"shape RMSE (level-normalised): median {np.median(shape_rmse):.2f} dB")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
sel = [g for g in groups if len(g[1]) >= 8]
pick = [sel[0], sel[len(sel)//3], sel[2*len(sel)//3], sel[-1]]
colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
for (key, g), col in zip(pick, colors):
    alpha, c, U, delta = key; g = g.sort_values("f")
    f = g["f"].values.astype(float); meas = g["SSPL"].values.astype(float)
    pred = np.asarray(spl_third_octave(delta, delta, U, alpha, c, freqs=f))
    pred = pred - np.nanmean(pred) + np.nanmean(meas)
    ax[0].semilogx(f, meas, "o", color=col, ms=4,
                   label=f"$\\alpha$={alpha:.1f}$^\\circ$, $U$={U:.0f} m/s")
    ax[0].semilogx(f, pred, "-", color=col, lw=1.8)
ax[0].set_xlabel("Frequency (Hz)"); ax[0].set_ylabel("1/3-octave SPL (dB, level-normalised)")
ax[0].set_title("Spectral shape: BPM (lines) vs NASA (markers)")
ax[0].legend(fontsize=8, loc="lower left", ncol=2); ax[0].grid(alpha=0.3, which="both")

ax[1].hist(corrs, bins=np.linspace(0.4, 1.0, 25), color="#1f77b4", alpha=0.85, edgecolor="k", lw=0.4)
ax[1].axvline(np.median(corrs), color="k", ls="--", lw=1.4, label=f"Median $r$={np.median(corrs):.2f}")
ax[1].set_xlabel("Within-spectrum correlation $r$"); ax[1].set_ylabel("Number of conditions")
ax[1].set_title("Per-condition spectral-shape agreement"); ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
fig.suptitle(f"Experimental validation of the BPM model against the NASA database "
             f"(attached flow, {len(corrs)} conditions)")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_bpm_validation.png", dpi=200); plt.close(fig)
print("  saved figs/fig_bpm_validation.png")
