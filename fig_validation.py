#!/usr/bin/env python3
"""Experimental validation of the trailing-edge self-noise model against the NASA airfoil self-noise database."""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import figstyle as fs
import bpm_noise as v1
import bpm_noise_v2 as v2
from bpm_bl import dstar_p_from_dstar_s, infer_tripped

fs.use()

NU, C0 = 1.5e-5, 340.46
SPAN, RE_OBS = 0.4572, 1.22
ALPHA_MAX = 12.5


def load():
    d = pd.read_csv("NASA_selfnoise.csv")
    return d[d.alpha <= ALPHA_MAX].copy()


def repeatability(d: pd.DataFrame) -> float:
    x = d.copy()
    x["Rec"] = x.U_infinity * x.c / NU
    M = x.U_infinity / C0
    St1 = 0.02 * M ** -0.6
    Sts = x.f * x.delta / x.U_infinity
    x["b_st"] = np.round(np.log10(Sts / St1) / 0.06)
    x["b_re"] = np.round(np.log10(x.Rec) / 0.12)
    x["b_a"] = np.round(x.alpha / 3.0)
    g = x.groupby(["b_st", "b_re", "b_a"])["SSPL"]
    n, s = g.count(), g.std()
    return float(s[n >= 3].median())


def predict(model, ds, dp, U, alpha, c, freqs):
    raw = np.asarray(model.spl_third_octave(ds, dp, U, alpha, c, span=SPAN,
                                            r=RE_OBS, freqs=freqs), float)
    return raw - 10 * np.log10((U / C0) ** 5 * ds * SPAN / RE_OBS ** 2)


def per_condition(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (alpha, c, U, ds), g in d.groupby(["alpha", "c", "U_infinity", "delta"]):
        if len(g) < 5:
            continue
        g = g.sort_values("f")
        f = g.f.values.astype(float); meas = g.SSPL.values.astype(float)
        trip = bool(infer_tripped(ds, alpha, U * c / NU, c))
        dp_new = float(dstar_p_from_dstar_s(ds, alpha, trip))
        old = predict(v1, ds, ds, U, alpha, c, f)
        new = predict(v2, ds, dp_new, U, alpha, c, f)
        rows.append(dict(alpha=alpha, c=c, U=U, dstar_s=ds, n=len(g),
                         alpha_switch=float(v2.alpha_switch(U / C0)),
                         err_old=float(np.mean(old - meas)),
                         err_new=float(np.mean(new - meas)),
                         rmse_old=float(np.sqrt(np.mean((old - meas) ** 2))),
                         rmse_new=float(np.sqrt(np.mean((new - meas) ** 2)))))
    return pd.DataFrame(rows)


def spectrum(d, alpha_target, prefer_U=None):
    s = d[np.isclose(d.alpha, alpha_target)]
    keys = sorted({(cc, UU, dd) for cc, UU, dd in
                   zip(s.c, s.U_infinity, s.delta)},
                  key=lambda k: -(k[1] * k[0]))
    c, U, ds = keys[0]
    g = s[(s.c == c) & (s.U_infinity == U)].sort_values("f")
    f, meas = g.f.values.astype(float), g.SSPL.values.astype(float)
    trip = bool(infer_tripped(ds, alpha_target, U * c / NU, c))
    dp_new = float(dstar_p_from_dstar_s(ds, alpha_target, trip))
    return dict(alpha=alpha_target, c=c, U=U, f=f, meas=meas,
                old=predict(v1, ds, ds, U, alpha_target, c, f),
                new=predict(v2, ds, dp_new, U, alpha_target, c, f))


def build():
    d = load()
    sig = repeatability(d)
    R = per_condition(d)

    fig = plt.figure(figsize=(fs.W2, 0.315 * fs.W2))
    gs = fig.add_gridspec(1, 3, wspace=0.34, width_ratios=[1, 1, 1.24],
                          left=0.062, right=0.995, bottom=0.225, top=0.815)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    for ax, a_t, tag in zip(axes[:2], (6.7, 12.3), ("(a)", "(b)")):
        S = spectrum(d, a_t)
        ax.fill_between(S["f"], S["meas"] - sig, S["meas"] + sig,
                        color=fs.BAND, alpha=0.5, lw=0, zorder=1)
        ax.plot(S["f"], S["old"], color=fs.C[1], ls=fs.DASH[1], lw=1.25,
                zorder=3, label=fs.L_REDUCED)
        ax.plot(S["f"], S["new"], color=fs.C[0], ls=fs.DASH[0], lw=1.4,
                zorder=5, label=fs.L_COMPLETE)
        ax.errorbar(S["f"], S["meas"], yerr=sig, fmt="o", ms=3.0, mfc="white",
                    mec=fs.INK, mew=0.7, ecolor=fs.INK2, elinewidth=0.55,
                    capsize=1.5, capthick=0.55, zorder=6, ls="none",
                    label=fs.L_MEAS + ", $\\pm1\\sigma$")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([200, 500, 1000, 2000, 5000]))
        ax.set_xlabel("Frequency, $f$  [Hz]")
        _rec = S["U"] * S["c"] / NU
        ax.set_title(f"$\\alpha^*={a_t:g}^\\circ$,  $U={S['U']:.1f}$ m/s\n"
                     f"$c={1e3*S['c']:.1f}$ mm,  $Re_c={_rec/1e6:.2f}\\times10^6$",
                     pad=3, fontsize=6.6, linespacing=1.25)
        fs.panel(ax, tag, dx=-0.20, dy=1.10)
        lo = min(S["meas"].min(), S["new"].min()) - 3
        hi = max(S["meas"].max(), S["new"].max(), S["old"].max()) + 3
        ax.set_ylim(lo, hi + 0.34 * (hi - lo))
        e_old = np.mean(S["old"] - S["meas"]); e_new = np.mean(S["new"] - S["meas"])
        ax.text(0.035, 0.035,
                f"Mean level error\nReduced    {e_old:+.1f} dB\nComplete  {e_new:+.1f} dB",
                transform=ax.transAxes, fontsize=6.5, va="bottom", ha="left",
                color=fs.INK, linespacing=1.35,
                bbox=dict(fc="white", ec=fs.GRID, lw=0.45, pad=2.2, alpha=0.94))
    axes[0].set_ylabel("Scaled 1/3-octave SPL  [dB]")
    axes[1].legend(loc="upper right", ncol=1, fontsize=6.5)

    ax = axes[2]
    a_lo = float(R.alpha_switch.min())
    a_hi = float(R.alpha_switch.max())
    YLO, YHI = -30.0, 11.0
    ax.axvspan(a_hi, 13.2, color="#f2f2f2", lw=0, zorder=0)
    ax.axvspan(a_lo, a_hi, color="#f8f8f8", lw=0, zorder=0)
    ax.axhspan(-sig, sig, color=fs.BAND, alpha=0.5, lw=0, zorder=1)
    ax.axhline(0, color=fs.INK2, lw=0.6, zorder=2)
    for a_sw in (a_lo, a_hi):
        ax.axvline(a_sw, color=fs.INK2, lw=0.7, ls=(0, (2.5, 1.6)), zorder=2)
    ax.annotate(f"$(\\alpha^*)_0$ over the test speeds, "
                f"{a_lo:.1f}\u2013{a_hi:.1f}$^\\circ$",
                xy=(0.5 * (a_lo + a_hi), 0.0), xycoords=("data", "axes fraction"),
                xytext=(0, -34), textcoords="offset points",
                fontsize=6.2, color=fs.INK2, ha="center", va="top",
                annotation_clip=False)

    below = R.err_old < YLO
    ax.plot(R.alpha[~below], R.err_old[~below], ls="none", marker=fs.MARK[1],
            ms=3.6, mfc="none", mec=fs.C[1], mew=0.85, zorder=3,
            label=fs.L_REDUCED)
    ax.plot(R.alpha[below], np.full(below.sum(), YLO + 0.9), ls="none",
            marker="v", ms=3.4, mfc="none", mec=fs.C[1], mew=0.85, zorder=3)
    ax.plot(R.alpha, R.err_new, ls="none", marker=fs.MARK[0], ms=3.4,
            mfc=fs.C[0], mec="white", mew=0.45, zorder=4,
            label=fs.L_COMPLETE)

    ax.set_xlim(-0.6, 13.2); ax.set_ylim(YLO, YHI)
    ax.set_xlabel("Effective angle of attack, $\\alpha^*$  [deg]")
    ax.set_ylabel("Level error, predicted $-$ measured  [dB]")
    ax.set_title("Level error across the database  ($n=88$)", pad=3)
    fs.panel(ax, "(c)", dx=-0.185, dy=1.10)

    ax.text(0.985, 0.975, "Separated-flow branch", transform=ax.transAxes,
            fontsize=6.4, color=fs.INK2, va="top", ha="right")
    if below.sum():
        ax.text(11.7, YLO + 1.4, f"{int(below.sum())} points off scale\n(to $-91$ dB)",
                fontsize=6.2, color=fs.C[1], va="bottom", ha="right",
                linespacing=1.3)

    hi = R[R.alpha > R.alpha_switch]
    ax.text(0.035, 0.16,
            f"Mean error above $(\\alpha^*)_0$   ($n={len(hi)}$)\n"
            f"Reduced    {hi.err_old.mean():+.1f} dB\n"
            f"Complete  {hi.err_new.mean():+.1f} dB",
            transform=ax.transAxes, fontsize=6.5, va="bottom", ha="left",
            color=fs.INK, linespacing=1.35,
            bbox=dict(fc="white", ec=fs.GRID, lw=0.45, pad=2.2, alpha=0.94))
    ax.legend(loc="upper left", ncol=1, fontsize=6.5)

    fs.save(fig, "fig_bpm_validation")

    print(f"\ndatabase repeatability (1 sigma) = {sig:.2f} dB")
    print(f"mean level error   reduced  {R.err_old.mean():+.2f} dB   "
          f"complete {R.err_new.mean():+.2f} dB")
    print(f"median band RMSE   reduced  {R.rmse_old.median():.2f} dB   "
          f"complete {R.rmse_new.median():.2f} dB")
    print(f"  (that is a median over conditions; the mean is "
          f"{R.rmse_new.mean():.2f} dB and the pooled value "
          f"{np.sqrt(np.mean(R.rmse_new.values ** 2)):.2f} dB)")
    print(f"switching angle    {a_lo:.2f} to {a_hi:.2f} deg over the test speeds")
    print(f"above (alpha*)_0   reduced  {hi.err_old.mean():+.2f} dB   "
          f"complete {hi.err_new.mean():+.2f} dB   (n={len(hi)})")
    R.to_csv("validation_per_condition.csv", index=False)


if __name__ == "__main__":
    build()
