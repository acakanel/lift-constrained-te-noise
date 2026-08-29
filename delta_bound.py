"""A number for Delta, the bound on the matched-disturbance rate."""
import os
import argparse
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import compare as C
import plant as P

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "delta_bound.csv")
OUT_TEX = os.path.join(HERE, "paper_AST", "delta_numbers.tex")

C.set_operating_point(55.0, 1.30)
_RATE0 = P.RATE.copy()
_SIGMA0 = P.SIGMA

GUST_THRESH = 0.0562


def gust_derivatives(g, t):
    ph = g.w * t + g.ph
    return (float(g.A * np.sum(g.w * np.cos(ph))),
            float(-g.A * np.sum(g.w ** 2 * np.sin(ph))))


def instrumented_bfasta(model, T=16.0, dt=0.005, gust=None, sensor=0.0, seed=0,
                        eps=0.12, gamma=3.0, budget=0.05, k_reach=6.0):
    rng = np.random.RandomState(seed)
    pl = P.Plant(C.LOUD.copy(), C.U, dt)
    n = int(T / dt)
    w = 0.0
    ustar = C.setpoint(model, budget)
    K0 = 0.5
    K = K0
    reached = False
    c1, c2 = 1.5, 1.1
    keys = ("t", "viol", "gdd", "ebar", "lag", "K", "nullrate", "jang",
            "ratefrac")
    log = {k: np.zeros(n) for k in keys}
    for i in range(n):
        t = i * dt
        d = pl.d.copy()
        g = gust(t) if gust else 0.0
        truelift = pl.CL() + g
        meas = truelift + (rng.normal(0, sensor) if sensor else 0.0)
        s = meas - C.CLreq
        if not reached and abs(s) < 0.95 * eps:
            reached = True
        elif reached and abs(s) >= eps:
            reached = False
        if reached:
            K = (eps * abs(s)) / (eps - min(abs(s), 0.999 * eps)) + K0
        else:
            K += k_reach * dt
        k1 = c1 * np.sqrt(K)
        k2 = c2 * K
        J = C.jacCL(model, d)
        Jp = J / (J @ J + 1e-9)
        Pn = np.eye(3) - np.outer(Jp, J)
        phi1 = np.sqrt(abs(s)) * np.sign(s)
        w += -k2 * np.sign(s) * dt
        ddes = Jp * (-k1 * phi1 + w) - gamma * (Pn @ (d - ustar))
        log["ratefrac"][i] = float(np.max(np.abs(ddes) / P.RATE))
        da_before = pl.da.copy()
        pl.step(d + ddes / P.SIGMA)
        ddot = (pl.d - d) / dt

        Jt = np.asarray(P.tCL.grad([d[0], d[1], d[2], C.U])[:3], float)
        denom = float(Jt @ Jp)
        log["ebar"][i] = abs(1.0 / denom - 1.0) if abs(denom) > 1e-9 else np.nan

        Jt_a = np.asarray(P.tCL.grad(
            [da_before[0], da_before[1], da_before[2], C.U])[:3], float)
        log["lag"][i] = float(Jt_a @ ((pl.d - da_before) / pl.tau) - Jt @ ddot)

        zeta = -gamma * (Pn @ (d - ustar))
        log["nullrate"][i] = abs(float(Jt @ zeta))
        cosang = float(Jt @ J) / (np.linalg.norm(Jt) * np.linalg.norm(J) + 1e-12)
        log["jang"][i] = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

        _, gdd = gust_derivatives(gust, t) if gust else (0.0, 0.0)
        log["t"][i] = t
        log["viol"][i] = abs(truelift - C.CLreq)
        log["gdd"][i] = gdd
        log["K"][i] = K
    return log


def trial(seed, eps=0.12):
    rng = np.random.RandomState(1_000_000 + seed)
    b = rng.normal(0, 0.12)
    model = C.BiasedModel(C.Sur, b)
    sensor = rng.uniform(0.002, 0.006)
    rate_scale = rng.uniform(0.7, 1.3)
    P.RATE = _RATE0 * rate_scale
    sigma_scale = rng.uniform(0.7, 1.4)
    P.SIGMA = _SIGMA0 * sigma_scale
    g = P.VonKarmanGust(rng, amp=rng.uniform(0.03, 0.07))
    g.w = 2 * np.pi * np.exp(rng.uniform(np.log(0.2), np.log(1.5), len(g.w)))
    try:
        L = instrumented_bfasta(model, gust=g, sensor=sensor, seed=seed, eps=eps)
        ref = C.run_bfasta(model, gust=g, sensor=sensor, seed=seed, eps=eps)
    finally:
        P.RATE = _RATE0.copy()
        P.SIGMA = _SIGMA0
    assert np.allclose(L["viol"], ref["viol"], atol=1e-12), (
        "the instrumented loop has drifted from compare.run_bfasta")
    return dict(seed=seed,
                gust_amp=float(g.A * np.sqrt(len(g.w))),
                lift_bias=float(b), rate_scale=float(rate_scale),
                sigma_scale=float(sigma_scale),
                delta=float(np.max(np.abs(L["gdd"]))),
                ebar=float(np.nanmax(L["ebar"])),
                lag=float(np.max(np.abs(L["lag"]))),
                nullrate=float(np.max(L["nullrate"])),
                rate_max=float(np.max(L["ratefrac"])),
                rate_binds_pct=float(100.0 * np.mean(
                    L["ratefrac"][int(0.6 * len(L["ratefrac"])):] > 1.0)),
                band_lost=bool(np.max(L["viol"][int(0.6 * len(L["viol"])):]) >= eps),
                jang=float(np.max(L["jang"])),
                K_max=float(L["K"].max()))


def main(n, reuse=False):
    if reuse and os.path.exists(OUT_CSV):
        d = pd.read_csv(OUT_CSV)
        print(f"reusing {OUT_CSV} ({len(d)} trials)")
    else:
        d = pd.DataFrame([trial(k) for k in range(n)])
        d.to_csv(OUT_CSV, index=False)

    dmax = float(d.delta.max())
    sub = d[d.gust_amp < GUST_THRESH]
    dsub = float(sub.delta.max()) if len(sub) else float("nan")
    ebar = float(d.ebar.max())

    import gains as _g
    k2_fixed = _g.K2
    k1_fixed = _g.K1
    k2_bf = 1.1 * float(d.K_max.max())
    need = 2 * dmax / (1 - ebar)

    def k1_required(g, D):
        return float(np.sqrt(4 * D * g / (g - D))) if g > D else float("inf")

    g_fixed = k2_fixed / 2.0
    k1_need = k1_required(g_fixed, dmax)
    scale = 1.0 / (1.0 + ebar)
    k1_need_eff = k1_required(g_fixed * scale, dmax)

    c1, c2 = _g.C1, _g.C2
    lev_k1, lev_g = c1 * np.sqrt(dmax), c2 * dmax

    ALT_K1, ALT_K2 = 6.0, 20.0
    alt_need = k1_required(ALT_K2 / 2.0, dmax)
    alt_need_eff = k1_required(ALT_K2 / 2.0 * scale, dmax)

    import band as _b
    eps_band = _b.EPS
    K0_floor = _g.K0
    eta_max = eps_band ** 2 / (2 * dmax - K0_floor + eps_band)

    V = dict(
        dbN=len(d),
        dbDelta=dmax, dbDeltaSub=dsub, dbDeltaMean=float(d.delta.mean()),
        dbEbar=ebar, dbLag=float(d.lag.max()),
        dbTwoDelta=2 * dmax, dbNeed=need,
        dbKtwoFixed=k2_fixed, dbKtwoBF=k2_bf,
        dbMarginFixed=k2_fixed / need, dbMarginBF=k2_bf / need,
        dbCorrGust=float(np.corrcoef(d.gust_amp, d.delta)[0, 1]),
        dbKoneFixed=k1_fixed, dbKoneNeed=k1_need,
        dbKoneMargin=k1_fixed / k1_need,
        dbKoneMarginEff=(k1_fixed * scale) / k1_need_eff,
        dbKoneNeedEff=k1_need_eff,
        dbKoneAlt=ALT_K1, dbKtwoAlt=ALT_K2,
        dbKoneNeedAlt=alt_need, dbKoneNeedAltEff=alt_need_eff,
        dbEtaMax=eta_max, dbTwoDeltaBar=2 * dmax,
        dbLevMarginBF=(k2_bf / c2 / 2.0) / dmax,
        dbLevKone=lev_k1, dbLevG=lev_g,
        dbLevMarginKone=k1_fixed / lev_k1,
        dbLevMarginG=g_fixed / lev_g,
        dbLevMarginKoneEff=k1_fixed * scale / lev_k1,
        dbLevMarginGEff=g_fixed * scale / lev_g,
        dbJAngle=float(d.jang.max()) if "jang" in d else float("nan"),
        dbNullRate=float(d.nullrate.max()) if "nullrate" in d else float("nan"),
        dbNullOverDelta=(float(d.nullrate.max()) / dmax
                         if "nullrate" in d else float("nan")),
    )
    if "rate_binds_pct" in d:
        held = d[~d.band_lost] if "band_lost" in d else d
        lost = d[d.band_lost] if "band_lost" in d else d.iloc[0:0]
        V["dbRateBindsHeld"] = float(held.rate_binds_pct.mean())
        V["dbRateBindsLost"] = (float(lost.rate_binds_pct.mean())
                                if len(lost) else float("nan"))
        V["dbNLost"] = int(len(lost))
        V["dbRateMaxHeld"] = float(held.rate_max.max())
    lines = ["% delta_numbers.tex -- generated by delta_bound.py. Do not edit.",
             f"% {len(d)} instrumented trials of the campaign's own class", ""]
    for k, v in V.items():
        lines.append("\\newcommand{\\%s}{%s}" % (
            k, int(v) if k in ("dbN", "dbNLost") else f"{v:.3g}"))
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    open(OUT_TEX, "w").write("\n".join(lines) + "\n")

    print(f"{len(d)} instrumented trials\n")
    print(f"  Delta = max |rho-dot| over the exogenous disturbance: {dmax:.3f}")
    print(f"    mean over trials {d.delta.mean():.3f}, "
          f"correlation with gust amplitude {V['dbCorrGust']:+.3f}")
    print(f"    over the certified class (gust < {GUST_THRESH}): {dsub:.3f} "
          f"({len(sub)} trials)")
    print(f"  e_bar, realised Jacobian error: {ebar:.3f}")
    print(f"  lag residual magnitude: {d.lag.max():.3f} (a level, not a rate)\n")
    print(f"  condition k2(1-e_bar) > 2*Delta, i.e. k2 > {need:.2f}")
    print(f"    fixed-time law, k2 = {k2_fixed:.1f}       margin "
          f"x{k2_fixed/need:.2f}  "
          f"{'SATISFIED' if k2_fixed > need else 'VIOLATED'}")
    print(f"    barrier law: not subject to this condition; its gain adapts, and "
          f"over these trials k2 = 1.1*K peaked at {k2_bf:.1f}")
    print(f"\n  condition k1^2 > 4*Delta*g/(g-Delta), g = k2/2 = {g_fixed:.1f}")
    print(f"    requires k1 > {k1_need:.2f};  deployed k1 = {k1_fixed:.1f}  "
          f"{'SATISFIED' if k1_fixed > k1_need else 'NOT SATISFIED'} "
          f"(short by x{k1_need/k1_fixed:.2f})")
    print(f"    read through prop:rob (both gains scaled by 1/(1+e_bar)): "
          f"requires k1 > {k1_need_eff:.2f} against {k1_fixed*scale:.2f}")
    print(f"\n  Levant tuning k1 > {c1}*sqrt(Delta) = {lev_k1:.2f}, "
          f"g > {c2}*Delta = {lev_g:.2f}")
    print(f"    deployed k1 = {k1_fixed:.1f} (x{k1_fixed/lev_k1:.2f}), "
          f"g = {g_fixed:.1f} (x{g_fixed/lev_g:.2f})")
    print(f"    with the gain margin: k1 (x{k1_fixed*scale/lev_k1:.2f}), "
          f"g (x{g_fixed*scale/lev_g:.2f})")
    if "rate_binds_pct" in d:
        print(f"\n  rate limit binding in the settled window: "
              f"{V['dbRateBindsHeld']:.2f} % of steps where the band held, "
              f"{V['dbRateBindsLost']:.2f} % where it was lost "
              f"({V['dbNLost']} trials)")
    if "jang" in d:
        print(f"\n  direction error between the surrogate and true Jacobians: "
              f"{d.jang.max():.2f} deg at worst")
        print(f"    lift rate it injects through the projector (lem:null's zero): "
              f"{d.nullrate.max():.4f} per s, {d.nullrate.max()/dmax:.3f} of Delta")
    print(f"\nwrote {OUT_CSV} and {OUT_TEX}")
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--reuse", action="store_true",
                    help="re-derive the macros from delta_bound.csv")
    a = ap.parse_args()
    main(a.n, a.reuse)
