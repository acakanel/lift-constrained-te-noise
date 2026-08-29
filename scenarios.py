#!/usr/bin/env python3
"""Fault, saturation and turbulence scenarios on the dynamic plant."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import band as _band
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import figstyle as _fs; _fs.use()
import control3 as c

from surrogate import AeroSurrogate
import gains as _g

MU = _g.MU

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
S = AeroSurrogate.from_csv()

AG = np.linspace(0, 12, 25); DG = np.linspace(-10, 10, 17)
GRID = np.array([[a, d1, d2] for a in AG for d1 in DG for d2 in DG])


def feasible_span(U, CLreq, tol=0.02):
    X = np.column_stack([GRID, np.full(len(GRID), U)])
    CL = S._cl.predict(X); SPL = S._spl.predict(X)
    m = np.abs(CL - CLreq) < tol
    if np.count_nonzero(m) < 3: return np.nan, np.nan
    return SPL[m].min(), SPL[m].max()


def envelope_map():
    Us = np.linspace(45, 70, 6); CLs = np.linspace(0.8, 1.5, 8)
    R = np.full((len(CLs), len(Us)), np.nan)
    for i, CL in enumerate(CLs):
        for j, U in enumerate(Us):
            lo, hi = feasible_span(U, CL)
            if not np.isnan(lo): R[i, j] = hi - lo
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    im = ax.imshow(R, origin="lower", aspect="auto", cmap="viridis",
                   extent=[Us[0], Us[-1], CLs[0], CLs[-1]])
    cb = fig.colorbar(im, ax=ax); cb.set_label("Achievable noise reduction at fixed lift (dB)")
    ax.set_xlabel("Approach speed $U$ (m/s)"); ax.set_ylabel("Required lift $C_{L,req}$")
    ax.set_title("Lift-preserving noise-reduction authority over the flight envelope")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig6_envelope.png", dpi=200); plt.close(fig)
    good = R[~np.isnan(R)]
    print(f"A. envelope map: reduction {np.nanmin(R):.1f}-{np.nanmax(R):.1f} dB "
          f"(median {np.median(good):.1f}); largest at high lift. saved fig6_envelope.png")


def approach():
    T = 16.0
    U = lambda t: 72.0 - (72.0-48.0)*min(t, 12.0)/12.0
    CL = lambda t: 0.90 + (1.40-0.90)*min(t, 12.0)/12.0
    log = c.proposed(U, CL, T=T, u0=[6, 0, 0], conf_c=0.0)
    n = len(log["t"]); base_spl = np.zeros(n); a = np.linspace(0, 12, 60)
    for i, t in enumerate(log["t"]):
        Xg = np.column_stack([a, np.zeros(60), np.zeros(60), np.full(60, U(t))])
        cl = S._cl.predict(Xg); ai = a[np.argmin(np.abs(cl - CL(t)))]
        base_spl[i] = S.SPL(ai, 0, 0, U(t))
    dSPL = np.mean(base_spl - log["SPL"])
    fig, ax = plt.subplots(2, 1, figsize=(7.4, 5.6), sharex=True)
    ax[0].plot(log["t"], base_spl, color="#7f7f7f", lw=1.6, label="Baseline (flaps neutral, incidence trims lift)")
    ax[0].plot(log["t"], log["SPL"], color="#d62728", lw=1.9, label="SG-FTSMC (coordinated)")
    ax[0].set_ylabel("Overall SPL (dB)"); ax[0].legend(fontsize=9, loc="upper right")
    ax[0].set_title("Realistic landing approach (speed 72$\\to$48 m/s, lift 0.90$\\to$1.40)")
    ax[1].plot(log["t"], log["viol"], color="#d62728", lw=1.6, label="SG-FTSMC lift error")
    ax[1].axhline(0.05, color="k", ls=":", lw=0.8)
    ax[1].set_ylabel("$|C_L-C_{L,req}|$"); ax[1].set_xlabel("Time (s)"); ax[1].set_ylim(0, 0.2)
    ax[1].legend(fontsize=9, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig7_approach.png", dpi=200); plt.close(fig)
    print(f"B. approach: mean noise reduction over glideslope = {dSPL:.2f} dB, "
          f"max lift error = {np.max(log['viol'][50:]):.3f}. saved fig7_approach.png")


def _run_2dof(U, CLreq, stuck2, T=14.0):
    DT = c.DT; n = int(T/DT); u = np.array([9., 10., stuck2]); w = 0.0
    D1G = np.array([[a, d1] for a in np.linspace(0, 12, 25) for d1 in np.linspace(-10, 10, 21)])
    t = np.zeros(n); spl = np.zeros(n); viol = np.zeros(n)
    Xg = np.column_stack([D1G[:, 0], D1G[:, 1], np.full(len(D1G), stuck2), np.full(len(D1G), U)])
    CLg = S._cl.predict(Xg); SPLg = S._spl.predict(Xg)
    mm = np.abs(CLg - CLreq) < 0.04
    v_star = D1G[np.where(mm)[0][np.argmin(SPLg[mm])]] if mm.any() else u[:2].copy()
    for i in range(n):
        CL = c._CL(u, U); s = CL - CLreq
        g = S._cl.grad([u[0], u[1], stuck2, U]); J2 = np.array([g[0], g[1]])
        Jp = J2/(J2@J2+1e-9); P = np.eye(2)-np.outer(Jp, J2)
        phi1, phi2 = c.phi(s, MU)
        w += -_g.K2*phi2*DT
        dv = np.clip(Jp*(-_g.K1*phi1+w) - 3.0*(P@(u[:2]-v_star)),
                     -c.RATE[:2], c.RATE[:2])
        u[:2] = np.minimum(np.maximum(u[:2]+dv*DT, c.LO[:2]), c.HI[:2]); u[2] = stuck2
        t[i] = i*DT; spl[i] = c._SPL(u, U); viol[i] = abs(s)
    return t, spl, viol


def jam():
    U, CLreq = 55.0, 1.30
    full = float(c._SPL(c.quiet_setpoint(U, CLreq), U))
    results = {}
    for stuck in [0.0, -10.0]:
        t, spl, viol = _run_2dof(U, CLreq, stuck)
        results[stuck] = (t, spl, viol)
        print(f"C. jam: flap-2 dead at {stuck:+.0f} deg -> settled SPL={np.mean(spl[-200:]):.2f} dB "
              f"(vs {full:.1f} healthy), lift held to {np.max(viol[100:]):.3f}")
    fig = plt.figure(figsize=(_fs.W15, 0.62 * _fs.W15))
    gs = fig.add_gridspec(2, 1, hspace=0.16, left=0.115, right=0.985,
                          bottom=0.115, top=0.93)
    ax = [fig.add_subplot(gs[0, 0])]
    ax.append(fig.add_subplot(gs[1, 0], sharex=ax[0]))

    ax[0].axhline(full, color=_fs.C[2], lw=1.0, ls=(0, (2.5, 1.6)), zorder=4,
                  label=f"Healthy three-surface optimum, {full:.1f} dB")
    for k, stuck in enumerate([0.0, -10.0]):
        t, spl, viol = results[stuck]
        ax[0].plot(t, spl, color=_fs.C[k], ls=_fs.DASH[k], lw=1.2, zorder=3,
                   label=f"Flap 2 frozen at ${stuck:+.0f}^{{\\circ}}$ "
                         f"(two surfaces left)")
        ax[1].plot(t, viol, color=_fs.C[k], ls=_fs.DASH[k], lw=1.2, zorder=3)
    ax[0].set_ylabel("Overall level  [dB]")
    ax[0].legend(loc="upper right", fontsize=6.6)
    pen = max(np.mean(results[k][1][-200:]) - full for k in results)
    ax[0].set_title(f"One surface lost: lift is still held, and the reduction "
                    f"is given up by at most {pen:.2f} dB", pad=3)
    ax[0].tick_params(labelbottom=False)

    ax[1].axhspan(0, _band.EPS, color="#eef2f6", lw=0, zorder=0)
    ax[1].axhline(_band.EPS, color=_fs.C[1], lw=0.9, ls=(0, (1.4, 1.4)),
                  zorder=4, label=_band.label(False))
    ax[1].set_ylim(0, 0.15)
    ax[1].set_ylabel("Lift error,  $|C_L-C_{L,\\mathrm{req}}|$")
    ax[1].set_xlabel("Time  [s]")
    ax[1].legend(loc="upper right", fontsize=6.6)
    _fs.panel(ax[0], "(a)", dx=-0.10, dy=1.04)
    _fs.panel(ax[1], "(b)", dx=-0.10, dy=1.04)
    _fs.save(fig, "fig8_jam"); plt.close(fig)
    print("   saved fig8_jam.png")


def turbulence(seed=0):
    T = 16.0; DT = c.DT; n = int(T/DT); U = 55.0; CLreq = 1.30
    rng = np.random.RandomState(seed)
    dist = np.zeros(n); tau = 0.4
    for i in range(1, n):
        dist[i] = dist[i-1] + DT*(-dist[i-1]/tau) + np.sqrt(2*DT/tau)*0.06*rng.randn()
    u = np.array([2.1, 10., 10.]); w = 0.0
    t = np.zeros(n); viol = np.zeros(n); spl = np.zeros(n); res = np.zeros(n)
    ustar = c.quiet_setpoint(U, CLreq)
    for i in range(n):
        CL = c._CL(u, U) + dist[i]; s = CL - CLreq
        J = c._jac_CL(u, U); Jp = J/(J@J+1e-9); P = np.eye(3)-np.outer(Jp, J)
        phi1, phi2 = c.phi(s, MU)
        w += -_g.K2*phi2*DT
        u = c.clip_pos(u + c.clip_rate(Jp*(-_g.K1*phi1+w) - 3.0*(P@(u-ustar)))*DT)
        t[i] = i*DT; viol[i] = abs(s); res[i] = s; spl[i] = c._SPL(u, U)
    rms = np.sqrt(np.mean(viol[int(0.2*n):]**2))
    fig = plt.figure(figsize=(_fs.W15, 0.40 * _fs.W15))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.19, top=0.88)
    ax.plot(t, dist, color=_fs.INK2, lw=0.8, zorder=2,
            label="Turbulent lift disturbance")
    ax.plot(t, res, color=_fs.C[0], lw=1.2, zorder=3,
            label="Residual lift error under control")
    ax.set_xlabel("Time  [s]")
    ax.set_ylabel("$\\Delta C_L$")
    ax.set_title("Continuous turbulence rejection", pad=3)
    ax.legend(loc="upper right", fontsize=6.6)
    ax.axvspan(0, 0.2 * T, color="#f4eef2", lw=0, zorder=0)
    ax.text(0.2 * T - 0.05, 0.97, "Start-up, excluded from the RMS",
            transform=ax.get_xaxis_transform(),
            ha="right", va="top", fontsize=6.4, color=_fs.INK2)
    ax.text(0.985, 0.045, f"Lift-error RMS {rms:.3f}, "
            f"{np.std(dist)/rms:.1f}$\\times$ attenuation",
            transform=ax.transAxes, fontsize=6.6, ha="right", va="bottom",
            bbox=dict(fc="white", ec=_fs.GRID, lw=0.45, pad=2.2, alpha=0.94))
    _fs.panel(ax, "(c)", dx=-0.10, dy=1.04)
    _fs.save(fig, "fig9_turbulence"); plt.close(fig)
    print(f"D. turbulence: disturbance std={np.std(dist):.3f} -> residual lift-error RMS={rms:.3f} "
          f"({np.std(dist)/rms:.1f}x attenuation). saved fig9_turbulence.png")


if __name__ == "__main__":
    envelope_map(); approach(); jam(); turbulence()
    print("\n[OK] scenarios done.")
