#!/usr/bin/env python3
"""Publication figures for the control-framework sections."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import band as _band, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from surrogate import AeroSurrogate, _RBFGP, _FEATS
import control3 as c

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
S = AeroSurrogate.from_csv()
import figstyle as fs
import gains as _g
fs.use()


def fig_surrogate():
    from surrogate import DATASET
    d = pd.read_csv(os.path.join(HERE, DATASET)).dropna().reset_index(drop=True)
    from surrogate import SPL_COL
    X = d[_FEATS].values.astype(float)
    fig = plt.figure(figsize=(fs.W15, 0.34 * fs.W15))
    gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.065, right=0.995,
                          bottom=0.20, top=0.86)
    ax = [fig.add_subplot(gs[0, i]) for i in range(3)]
    panels = [("CL", AeroSurrogate.LS_CL, "Lift coefficient, $C_L$",
               "$C_L$", "", "(a)"),
              (SPL_COL, AeroSurrogate.LS_SPL, "Overall level",
               "overall level", " dB", "(b)")]
    if "CM" in d.columns and hasattr(AeroSurrogate, "LS_CM"):
        panels.append(("CM", AeroSurrogate.LS_CM, "Pitching moment, $C_M$",
                       "$C_M$", "", "(c)"))
    for j, (col, ls, lab, short, unit, tag) in enumerate(panels):
        y = d[col].values.astype(float)
        idx = np.arange(len(X)); np.random.RandomState(0).shuffle(idx)
        pred = np.zeros(len(X))
        for f in np.array_split(idx, 5):
            tr = np.setdiff1d(idx, f)
            pred[f] = _RBFGP(ls, 1e-3).fit(X[tr], y[tr]).predict(X[f])
        r2 = 1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)
        rmse = np.sqrt(np.mean((pred - y) ** 2))
        lo, hi = y.min(), y.max()
        pad = 0.04 * (hi - lo)
        ax[j].plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=fs.INK2,
                   lw=0.7, ls=(0, (2.5, 1.6)), zorder=2)
        ax[j].scatter(y, pred, s=7, alpha=0.55, color=fs.C[j], edgecolor="none",
                      zorder=3)
        ax[j].set_xlim(lo - pad, hi + pad); ax[j].set_ylim(lo - pad, hi + pad)
        ax[j].set_aspect("equal", adjustable="box")
        u = f"  [{unit.strip()}]" if unit.strip() else ""
        ax[j].set_xlabel(f"Computed {short}{u}")
        ax[j].set_ylabel(f"Predicted {short}{u}")
        ax[j].set_title(lab, pad=3)
        fs.panel(ax[j], tag, dx=-0.24, dy=1.05)
        ax[j].text(0.96, 0.06,
                   f"$R^2={r2:.4f}$\nRMSE $={rmse:.3g}${unit}",
                   transform=ax[j].transAxes, fontsize=6.6, ha="right",
                   va="bottom", linespacing=1.35,
                   bbox=dict(fc="white", ec=fs.GRID, lw=0.45, pad=2.2, alpha=0.94))
    fs.save(fig, "fig1_surrogate")


def fig_nullspace(log):
    U, CLreq = 55.0, 1.30
    ag = np.linspace(0, 12, 61); dg = np.linspace(-10, 10, 61)
    best_spl = []
    D1, D2 = np.meshgrid(dg, dg)
    flat = np.column_stack([D1.ravel(), D2.ravel()])
    for a in ag:
        X = np.column_stack([np.full(len(flat), a), flat, np.full(len(flat), U)])
        CL = S._cl.predict(X); SPL = S._spl.predict(X)
        m = np.abs(CL - CLreq) < 0.015
        best_spl.append(SPL[m].min() if np.any(m) else np.nan)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(ag, best_spl, "-", lw=2.2, color="#1f77b4",
            label="Min-noise frontier at $C_L$=1.30 (per incidence)")
    ax.plot(log["alpha"], log["SPL"], "-", lw=1.6, color="#d62728", alpha=0.9,
            label="SG-FTSMC trajectory")
    ax.scatter([log["alpha"][0]], [log["SPL"][0]], c="k", zorder=5, label="Start (loud)")
    ax.scatter([log["alpha"][-1]], [log["SPL"][-1]], c="#2ca02c", zorder=5, marker="*",
               s=160, label="Converged (quiet)")
    ax.set_xlabel("Wing incidence $\\alpha$ (deg)")
    ax.set_ylabel("Overall SPL (dB)")
    ax.set_title("Trailing-edge noise on the constant-lift manifold ($U$=55 m/s)")
    ax.annotate(f"{log['SPL'][0]-log['SPL'][-1]:.1f} dB",
                xy=(2.6, log["SPL"][-1]+0.3), xytext=(5.2, log["SPL"][0]-3.0),
                arrowprops=dict(arrowstyle="<->", color="gray"), color="gray")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_nullspace.png", dpi=200); plt.close(fig)
    print("  fig2_nullspace.png")


def fig_timehist(Lp, Ln, Le):
    fig = plt.figure(figsize=(fs.W15, 0.62 * fs.W15))
    gs = fig.add_gridspec(2, 1, hspace=0.12, left=0.13, right=0.985,
                          bottom=0.10, top=0.93)
    ax = [fig.add_subplot(gs[0, 0]), None]
    ax[1] = fig.add_subplot(gs[1, 0], sharex=ax[0])
    series = [(Lp, 0, "Learning-assisted, constraint-guaranteed"),
              (Ln, 1, "Local gradient, no global set-point"),
              (Le, 2, "Extremum seeking, model-free")]
    for L, k, lab in series:
        w = dict(lw=1.3, alpha=1.0) if k < 2 else dict(lw=0.8, alpha=0.55)
        ax[0].plot(L["t"], L["SPL"], color=fs.C[k], ls=fs.DASH[k], label=lab, **w)
        ax[1].plot(L["t"], L["viol"], color=fs.C[k], ls=fs.DASH[k], **w)
    ax[0].set_ylabel("Objective, overall level  [dB]")
    ax[0].set_title("Lift-constrained descent  ($U=55$ m/s, $C_L=1.30$)", pad=3)
    ax[0].legend(loc="upper right", fontsize=6.6)
    ax[0].tick_params(labelbottom=False)
    fs.panel(ax[0], "(a)", dx=-0.105, dy=1.03)

    ax[1].axhspan(0, _band.EPS, color="#eef2f6", lw=0, zorder=0)
    ax[1].axhline(_band.EPS, color=fs.C[1], lw=0.8, ls=(0, (2.5, 1.6)), zorder=2)
    ax[1].text(0.015, _band.EPS * 1.06, "  " + _band.label(False),
               transform=ax[1].get_yaxis_transform(), fontsize=6.4,
               color=fs.C[1], ha="left", va="bottom")
    ax[1].set_ylabel("Lift error,  $|C_L-C_{L,\\mathrm{req}}|$")
    ax[1].set_xlabel("Time  [s]")
    ax[1].set_ylim(-0.01, 0.35)
    fs.panel(ax[1], "(b)", dx=-0.105, dy=1.03)
    fs.save(fig, "fig3_timehist")


def fig_gust():
    def prop_gust(T=8.0, gamma=3.0, k1=_g.K1, k2=_g.K2, mu=_g.MU, conf_c=3.0):
        n = int(T/c.DT); u = np.array([2.1, 10., 10.]); w = 0.
        t = np.zeros(n); viol = np.zeros(n); spl = np.zeros(n); cl = np.zeros(n)
        for i in range(n):
            ti = i*c.DT; U = 55.0; CLreq = 1.30
            gust = 0.20 if ti >= 4.0 else 0.0
            CLm = c._CL(u, U) + gust; s = CLm - CLreq
            J = c._jac_CL(u, U); Jp = J/(J@J+1e-9)
            phi1 = np.sqrt(abs(s))*np.sign(s)+mu*abs(s)**1.5*np.sign(s)
            phi2 = 0.5*np.sign(s)+1.5*mu*abs(s)*np.sign(s)+1.5*mu*mu*abs(s)**2*np.sign(s)
            w += -k2*phi2*c.DT
            P = np.eye(3)-np.outer(Jp, J); g = c._grad_SPL(u, U)
            ge = gamma/(1+conf_c*c._SPLstd(u, U))
            us = c.quiet_setpoint(U, CLreq)
            u = c.clip_pos(u+c.clip_rate(Jp*(-k1*phi1+w)-ge*(P@(u-us)))*c.DT)
            t[i] = ti; viol[i] = s; spl[i] = c._SPL(u, U); cl[i] = CLm
        return t, viol, spl, cl
    t, viol, spl, cl = prop_gust()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(t, cl, color="#d62728", lw=1.8, label="Measured $C_L$ (with gust)")
    ax.axhline(1.30, color="k", lw=1, ls="--", label="Required $C_L$")
    ax.axvline(4.0, color="gray", lw=1, ls=":", alpha=0.7)
    ax.text(4.05, 1.45, "Gust +0.20", fontsize=9, color="gray")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("$C_L$")
    ax.set_title("Fixed-time SMC rejects a lift gust while holding the quiet config")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_gust.png", dpi=200); plt.close(fig)
    print("  fig4_gust.png")


if __name__ == "__main__":
    U1 = lambda t: 55.0; C1 = lambda t: 1.30
    print("running controllers for figures...")
    Lp = c.proposed(U1, C1, T=20.0)
    Ln = c.naive_grad(U1, C1, T=20.0)
    Le = c.esc(U1, C1, T=20.0)
    print("plotting:")
    fig_surrogate()
    fig_nullspace(Lp)
    fig_timehist(Lp, Ln, Le)
    fig_gust()
    print("\nSummary:")
    print(f"  baseline(loud) SPL = {Lp['SPL'][0]:.2f} dB")
    print(f"  SG-FTSMC settled   = {np.mean(Lp['SPL'][-400:]):.2f} dB  "
          f"(-{Lp['SPL'][0]-np.mean(Lp['SPL'][-400:]):.2f} dB, lift held |viol|<{np.max(Lp['viol'][200:]):.3f})")
    print(f"  NAIVE settled      = {np.mean(Ln['SPL'][-400:]):.2f} dB (local-min stall)")
    print(f"  ESC settled        = {np.mean(Le['SPL'][-400:]):.2f} dB, lift |viol|={np.mean(Le['viol'][-400:]):.3f} (unsafe)")
    print("[OK] figures in ./figs/")
