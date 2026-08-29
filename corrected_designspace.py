#!/usr/bin/env python3
"""The design space with the aerodynamic input corrected against the Reynolds-averaged solutions."""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

import bpm_noise_v2 as v2

CHORD, NU = 1.0, 1.5e-5


def design_matrix(df):
    return np.column_stack([np.ones(len(df)), df.alpha.values,
                            df.d1.values + df.d2.values])


def fit_linear(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def loo_error(X, y):
    e = []
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        if m.sum() <= X.shape[1]:
            continue
        b = fit_linear(X[m], y[m])
        e.append(y[i] - X[i] @ b)
    return float(np.sqrt(np.mean(np.square(e)))) if e else np.nan


def surrogate_at(alpha, d1, d2, U, cache={}):
    key = (round(alpha, 3), d1, d2, U)
    if key not in cache:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from aero_dataset import run_case
        cache[key] = run_case(alpha, d1, d2, U)
    return cache[key]


def build_corrections(sweep_csv=None, U=55.0):
    if sweep_csv is None:
        import ransdata
        R = ransdata.sweep(U)
    else:
        R = pd.read_csv(sweep_csv)
        R = R[R.U == U]
    R = R[R.CL.notna() & R.dstar_s.notna() & R.dstar_p.notna()].copy()
    if len(R) < 5:
        raise SystemExit(f"only {len(R)} Reynolds-averaged points -- not enough to fit")

    rows = []
    for r in R.itertuples():
        CL, CM, ds, dp = surrogate_at(r.alpha, r.d1, r.d2, U)
        rows.append(dict(alpha=r.alpha, d1=r.d1, d2=r.d2,
                         ds_r=r.dstar_s, dp_r=r.dstar_p, H_r=r.H_s,
                         ds_s=ds, dp_s=dp, CL_r=r.CL, CL_s=CL))
    D = pd.DataFrame(rows)
    X = design_matrix(D)

    out = {}
    for tag, yr, ys in [("s", D.ds_r, D.ds_s), ("p", D.dp_r, D.dp_s)]:
        y = np.log(yr.values / ys.values)
        out[f"beta_{tag}"] = fit_linear(X, y)
        out[f"loo_{tag}"] = loo_error(X, y)
    out["beta_H"] = fit_linear(X, D.H_r.values)
    out["loo_H"] = loo_error(X, D.H_r.values)
    out["data"] = D
    out["support"] = dict(
        alpha=(float(D.alpha.min()), float(D.alpha.max())),
        dsum=(float((D.d1 + D.d2).min()), float((D.d1 + D.d2).max())),
        n=len(D))

    print(f"correction fitted on {len(D)} Reynolds-averaged points at U = {U:g} m/s")
    print(f"  ln(delta*_s ratio):  a = {np.round(out['beta_s'], 4)}   "
          f"LOO RMS {out['loo_s']:.3f} (= {8.686*out['loo_s']:.2f} dB in level)")
    print(f"  ln(delta*_p ratio):  a = {np.round(out['beta_p'], 4)}   "
          f"LOO RMS {out['loo_p']:.3f}")
    print(f"  H_s:                 b = {np.round(out['beta_H'], 4)}   "
          f"LOO RMS {out['loo_H']:.3f}")
    return out


def h_to_alpha_map(sweep_csv=None, U=55.0):
    if sweep_csv is None:
        import ransdata
        R = ransdata.sweep(U)
    else:
        R = pd.read_csv(sweep_csv)
        R = R[R.U == U]
    ref = R[(R.d1 == 0) & (R.d2 == 0) & R.H_s.notna()].sort_values("H_s")
    return ref.H_s.values, ref.alpha.values


def apply(dataset_csv="dataset.csv", sweep_csv=None, U_fit=55.0,
          out_csv="dataset_corrected.csv"):
    C = build_corrections(sweep_csv, U_fit)
    Hs, As = h_to_alpha_map(sweep_csv, U_fit)

    d = pd.read_csv(dataset_csv).dropna().copy()
    dsum = d.delta1.values + d.delta2.values
    X = np.column_stack([np.ones(len(d)), d.alpha.values, dsum])
    d["dstar_s_c"] = d.dstar_s.values * np.exp(X @ C["beta_s"])
    d["dstar_p_c"] = d.dstar_p.values * np.exp(X @ C["beta_p"])
    d["H_s_c"] = X @ C["beta_H"]

    sa, sd = C["support"]["alpha"], C["support"]["dsum"]
    in_box = ((d.alpha.values >= sa[0]) & (d.alpha.values <= sa[1]) &
              (dsum >= sd[0]) & (dsum <= sd[1]))
    physical = d.H_s_c.values >= 1.0
    in_map = (d.H_s_c.values >= Hs.min()) & (d.H_s_c.values <= Hs.max())

    ok = in_box & physical & in_map
    aeq = np.full(len(d), np.nan)
    aeq[ok] = np.interp(d.H_s_c.values[ok], Hs, As)
    d["alpha_eq"] = aeq
    d["corr_ok"] = ok
    d["corr_reason"] = np.where(
        ok, "",
        np.where(~in_box, "outside the fitted design box",
                 np.where(~physical, "shape factor below unity",
                          "outside the equivalent-incidence map")))

    d["SPL_c"] = [v2.overall_spl(r.dstar_s_c, r.dstar_p_c, r.U, r.alpha_eq, CHORD)
                  if np.isfinite(r.alpha_eq) else np.nan for r in d.itertuples()]
    d["SPLA_c"] = [v2.oaspl_a_weighted(r.dstar_s_c, r.dstar_p_c, r.U, r.alpha_eq,
                                       CHORD)
                   if np.isfinite(r.alpha_eq) else np.nan for r in d.itertuples()]
    d.to_csv(out_csv, index=False)

    n = len(d)
    print(f"\ncorrection support: alpha in [{sa[0]:g}, {sa[1]:g}] deg, "
          f"d1+d2 in [{sd[0]:g}, {sd[1]:g}] deg, from {C['support']['n']} points")
    print(f"design box:         alpha in [{d.alpha.min():g}, {d.alpha.max():g}], "
          f"d1+d2 in [{dsum.min():g}, {dsum.max():g}]")
    print(f"\nwrote {out_csv}: {int(ok.sum())} of {n} rows "
          f"({100*ok.mean():.0f} %) carry a corrected level")
    for reason in ("outside the fitted design box", "shape factor below unity",
                   "outside the equivalent-incidence map"):
        k = int((d.corr_reason == reason).sum())
        if k:
            print(f"  {k:4d} refused: {reason}")
    if ok.any():
        print(f"  surviving rows: SPL {d.SPL_c.min():.1f}..{d.SPL_c.max():.1f} dB, "
              f"equivalent incidence "
              f"{np.nanmin(aeq):.1f}..{np.nanmax(aeq):.1f} deg")
    if ok.mean() < 0.5:
        print("\nLess than half the design box is inside the correction's domain.")
        print("This chain cannot support a design-space claim. It is kept because")
        print("it was run and because it is why the authority question is settled")
        print("in resolved flow instead; no result may be taken from it.")
    return d, C


def fixed_lift_spread(d, points=((40, 1.10), (55, 1.30), (70, 1.50)), tol=0.01):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    n0 = len(d)
    d = d[d.SPL_c.notna() & d.SPLA_c.notna()].copy()
    print(f"\ndesign-space fit uses {len(d)} of {n0} rows "
          f"({100*len(d)/n0:.0f} % of the box)")
    if len(d) < 0.5 * n0:
        print("The surrogate would be fitted on a small and irregular subset of "
              "the design box and then queried across all of it. The spreads "
              "below are reported for completeness and are not usable.")
    X = d[["alpha", "delta1", "delta2", "U"]].values.astype(float)
    mu, sd = X.mean(0), X.std(0)
    Xs = (X - mu) / sd
    ker = (ConstantKernel(1.0) * RBF([1.5] * 4, length_scale_bounds=(0.3, 20.0))
           + WhiteKernel(1e-6, noise_level_bounds=(1e-10, 1e-3)))
    gcl = GaussianProcessRegressor(ker, normalize_y=True, alpha=1e-8).fit(Xs, d.CL.values)
    gs = {k: GaussianProcessRegressor(ker, normalize_y=True, alpha=1e-8)
          .fit(Xs, d[k].values) for k in ("SPL_c", "SPLA_c")}

    g = np.meshgrid(np.linspace(0, 12, 25), np.linspace(-10, 10, 21),
                    np.linspace(-10, 10, 21), indexing="ij")
    print(f"\n{'U':>4s} {'CLreq':>6s} {'n':>5s} {'spread dB':>10s} {'spread dBA':>11s}"
          f"  quietest (alpha, d1, d2)")
    rows = []
    for U, CL in points:
        P = np.column_stack([g[0].ravel(), g[1].ravel(), g[2].ravel(),
                             np.full(g[0].size, float(U))])
        Ps = (P - mu) / sd
        m = np.abs(gcl.predict(Ps) - CL) < tol
        if m.sum() < 5:
            print(f"{U:4g} {CL:6.2f} {m.sum():5d}   (too few feasible points)")
            continue
        s = gs["SPL_c"].predict(Ps[m]); sa = gs["SPLA_c"].predict(Ps[m])
        q = P[m][np.argmin(s)]
        print(f"{U:4g} {CL:6.2f} {m.sum():5d} {s.max()-s.min():10.2f} "
              f"{sa.max()-sa.min():11.2f}   "
              f"({q[0]:.1f}, {q[1]:+.1f}, {q[2]:+.1f})")
        rows.append(dict(U=U, CL=CL, n=int(m.sum()), spread=float(s.max()-s.min()),
                         spread_A=float(sa.max()-sa.min()),
                         alpha_q=q[0], d1_q=q[1], d2_q=q[2]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d, C = apply()
    res = fixed_lift_spread(d)
    res.to_csv("designspace_corrected.csv", index=False)
    print("\nwrote designspace_corrected.csv")
