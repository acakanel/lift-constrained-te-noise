#!/usr/bin/env python3
"""Empirical evidence for the fixed-time lift-constraint convergence."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import figstyle as _fs; _fs.use()
import control3 as c
from surrogate import AeroSurrogate
import gains as _g

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True); S = c.S

U, CLreq, DT = 55.0, 1.30, 0.002


def settle(u0, fixed_time, tol=1e-3, T=3.0, k1=_g.K1, k2=_g.K2, mu=_g.MU):
    u = np.array(u0, float); w = 0.0; n = int(T / DT)
    s0 = c._CL(u, U) - CLreq
    last = np.nan
    for i in range(n):
        s = c._CL(u, U) - CLreq
        if abs(s) >= tol:
            last = i * DT
        J = c._jac_CL(u, U); Jp = J / (J @ J + 1e-9)
        if fixed_time:
            phi1, phi2 = c.phi(s, mu)
        else:
            phi1 = np.sqrt(abs(s))*np.sign(s); phi2 = 0.5*np.sign(s)
        w += -k2*phi2*DT
        u = c.clip_pos(u + c.clip_rate(Jp*(-k1*phi1 + w))*DT)
    if np.isnan(last):
        return 0.0, s0
    if last >= T - 2*DT:
        return np.nan, s0
    return last, s0


def find_u0(target_s):
    a = np.linspace(0, 12, 200)
    cl = S._cl.predict(np.column_stack([a, np.zeros(200), np.zeros(200), np.full(200, U)]))
    a0 = a[np.argmin(np.abs((cl - CLreq) - target_s))]
    return [a0, 0.0, 0.0]


def surface_settle(s0, fixed, tol=1e-3, T=40.0, dt=2e-4,
                   k1=_g.K1, k2=_g.K2, mu=_g.MU):
    s, w = float(s0), 0.0
    n = int(T / dt); last = np.nan
    for i in range(n):
        if abs(s) >= tol:
            last = i * dt
        a = abs(s); sg = np.sign(s)
        if fixed:
            p1 = (np.sqrt(a) + mu * a ** 1.5) * sg
            p2 = (0.5 + 1.5 * mu * a + 1.5 * mu * mu * a * a) * sg
        else:
            p1 = np.sqrt(a) * sg
            p2 = 0.5 * sg
        sd = -k1 * p1 + w
        w += -k2 * p2 * dt
        s += sd * dt
    return last


if __name__ == "__main__":
    DFIX = 5.0
    alphas = np.linspace(0.0, 12.0, 21)
    sF, tF, sC, tC = [], [], [], []
    for a0 in alphas:
        u0 = [a0, DFIX, DFIX]
        t1, s0 = settle(u0, True)
        t2, _ = settle(u0, False)
        if not np.isnan(t1): sF.append(s0); tF.append(t1)
        if not np.isnan(t2): sC.append(s0); tC.append(t2)
    order = np.argsort(sF); sF = np.asarray(sF)[order]; tF = np.asarray(tF)[order]
    order = np.argsort(sC); sC = np.asarray(sC)[order]; tC = np.asarray(tC)[order]

    S0 = np.logspace(np.log10(0.05), 3.0, 22)
    wideC = np.array([surface_settle(v, False) for v in S0])
    wideF = np.array([surface_settle(v, True) for v in S0])

    fig = plt.figure(figsize=(_fs.W2, 0.34 * _fs.W2))
    gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.075, right=0.985,
                          bottom=0.185, top=0.865)
    ax = [fig.add_subplot(gs[0, k]) for k in range(2)]

    ax[0].plot(sC, tC, color=_fs.C[1], ls=_fs.DASH[1], marker=_fs.MARK[1],
               lw=1.1, zorder=3, label="Classic super-twisting")
    ax[0].plot(sF, tF, color=_fs.C[0], ls=_fs.DASH[0], marker=_fs.MARK[0],
               lw=1.1, zorder=4, label="Fixed-time super-twisting")
    ax[0].set_xlabel("Initial lift error,  $s(0)=C_L-C_{L,\\mathrm{req}}$")
    ax[0].set_ylabel("Settling time to $|s|<10^{-3}$  [s]")
    ax[0].set_title("Within reach of the section, the two agree", pad=3)
    ax[0].legend(loc="upper center", fontsize=6.6)
    ax[0].set_ylim(0, max(max(tC), max(tF)) * 1.35)
    gap = float(np.max(np.abs(np.interp(sC, sF, tF) - tC)))
    ax[0].text(0.02, 0.03, f"Largest difference {1e3*gap:.0f} ms across\nthe whole "
               f"reachable range", transform=ax[0].transAxes, fontsize=6.4,
               ha="left", va="bottom", linespacing=1.35,
               bbox=dict(fc="white", ec=_fs.GRID, lw=0.45, pad=2.2, alpha=0.94))

    ax[1].axvspan(S0[0], abs(sC).max(), color="#eef2f6", lw=0, zorder=0)
    ax[1].text(abs(sC).max() * 0.92, 0.42, "Reachable by\nthe section",
               transform=ax[1].get_xaxis_transform(), ha="right", va="center",
               fontsize=6.4, color=_fs.INK2, linespacing=1.3, zorder=6,
               bbox=dict(fc="white", ec="none", pad=1.6, alpha=0.85))
    ax[1].loglog(S0, wideC, color=_fs.C[1], ls=_fs.DASH[1], lw=1.2, zorder=3,
                 label="Classic super-twisting")
    ax[1].loglog(S0, wideF, color=_fs.C[0], ls=_fs.DASH[0], lw=1.2, zorder=4,
                 label="Fixed-time super-twisting")
    ax[1].axhline(wideF[-1], color=_fs.INK2, lw=0.8, ls=(0, (1.4, 1.4)), zorder=5)
    ax[1].text(S0[0] * 1.2, wideF[-1] * 1.08,
               f"Uniform bound, {wideF[-1]:.2f} s", fontsize=6.4,
               color=_fs.INK2)
    ax[1].set_xlabel("Initial constraint error,  $|s(0)|$")
    ax[1].set_ylabel("Settling time  [s]")
    ax[1].set_title("Beyond it, only one of them stops growing", pad=3)
    ax[1].legend(loc="upper left", fontsize=6.6)

    _fs.panel(ax[0], "(a)", dx=-0.15, dy=1.07)
    _fs.panel(ax[1], "(b)", dx=-0.15, dy=1.07)
    _fs.save(fig, "fig13_fixedtime"); plt.close(fig)

    import os as _os
    from writeonce import write_if_changed
    V = {"stGapMs": f"{1e3*gap:.0f}",
         "stBound": f"{wideF[-1]:.2f}",
         "stClassicHi": f"{wideC[-1]:.2f}",
         "stSmax": f"{S0[-1]:.0f}"}
    lines = ["% st_numbers.tex -- generated by fixedtime.py. Do not edit.", ""]
    lines += ["\\newcommand{\\%s}{%s}" % (k, v) for k, v in V.items()]
    write_if_changed(_os.path.join(HERE, "paper_AST", "st_numbers.tex"),
                     "\n".join(lines) + "\n")
    print(f"reachable range: largest difference {1e3*gap:.1f} ms")
    print(f"wide range: classic {wideC[0]:.2f}->{wideC[-1]:.2f} s, "
          f"fixed-time {wideF[0]:.2f}->{wideF[-1]:.2f} s "
          f"at |s0|={S0[-1]:.0f}")
    print(f"fixed-time settling bound  ~ {max(tF):.3f} s over |s0| up to {max(np.abs(sF)):.2f}")
    print(f"classic STA settling range   {min(tC):.3f}..{max(tC):.3f} s (grows with |s0|)")
    print("  saved figs/fig13_fixedtime.png")
