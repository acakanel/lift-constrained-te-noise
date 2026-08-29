#!/usr/bin/env python3
"""The measurements behind the boundary-layer-edge criterion of the convergence appendix."""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import post
import edge_study as E

OUT = os.path.join(HERE, "edge_evidence.csv")
RAYS = (0.06, 0.12, 0.25)


def rows_for(case, label, path):
    out = []
    for L in RAYS:
        y, ut, ueb, p0 = E.raw_profile(path, ray_scale=L)
        rec = dict(case=case, label=label, ray_c=L,
                   ray_mm=1e3 * float(y[-1]),
                   reversed_frac=float(np.mean(ut[:max(1, len(ut) // 2)] < 0)),
                   u_e_bernoulli=ueb, u_max=float(np.abs(ut).max()))
        for name, fn in (("bernoulli", E.edge_bernoulli), ("shear", E.edge_shear),
                         ("max99", E.edge_max99), ("totalp", E.edge_totalp)):
            e, ue, ok = fn(y, ut, ueb, p0) if name == "totalp" else fn(y, ut, ueb)
            ds, th, H = E.integrate(y, ut, ue, e)
            rec[f"{name}_edge_mm"] = 1e3 * float(y[e])
            rec[f"{name}_dstar_mm"] = 1e3 * ds
            rec[f"{name}_H"] = H
            rec[f"{name}_at_ray_end"] = bool(e >= len(y) - 2)
        out.append(rec)
    return out


def main():
    rows = []
    for case, label in E.CASES:
        path = os.path.join(HERE, "resweep", case)
        if not os.path.isdir(path):
            path = os.path.join(HERE, "conv", case)
        if not os.path.isdir(path):
            print(f"{case:24s} not found"); continue
        rows += rows_for(case, label, path)
        print(f"{case:24s} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)

    print(f"\nwrote {os.path.basename(OUT)}\n")
    print("level change from lengthening the sampling ray "
          f"{RAYS[0]:.2f}c -> {RAYS[1]:.2f}c, Bernoulli criterion")
    print(f"  {'case':24s} {'d* short':>10s} {'d* long':>10s} {'change dB':>10s}")
    for case, g in d.groupby("case", sort=False):
        a = g[g.ray_c == RAYS[0]].bernoulli_dstar_mm.iloc[0]
        b = g[g.ray_c == RAYS[1]].bernoulli_dstar_mm.iloc[0]
        print(f"  {case:24s} {a:10.4f} {b:10.4f} {10*np.log10(b/a):+10.4f}")

    print("\nthickness against the ray length, Bernoulli criterion: a case with no"
          "\nedge integrates to the end of the ray, so the two are equal")
    print(f"  {'case':24s} {'ray mm':>9s} {'d* mm':>9s} {'ratio':>7s} {'rev.':>6s}")
    for _, r in d[d.ray_c == RAYS[0]].iterrows():
        print(f"  {r.case:24s} {r.ray_mm:9.3f} {r.bernoulli_dstar_mm:9.3f} "
              f"{r.bernoulli_dstar_mm/r.ray_mm:7.4f} {r.reversed_frac:6.3f}")

    print("\ncriteria compared on the cases that have an edge, at "
          f"{RAYS[0]:.2f}c")
    print(f"  {'case':24s} {'bernoulli':>10s} {'shear':>10s} {'max99':>10s} "
          f"{'totalp':>10s}   [d* mm]")
    for _, r in d[d.ray_c == RAYS[0]].iterrows():
        print(f"  {r.case:24s} {r.bernoulli_dstar_mm:10.3f} {r.shear_dstar_mm:10.3f} "
              f"{r.max99_dstar_mm:10.3f} {r.totalp_dstar_mm:10.3f}")

    def pick(tag, col):
        m = d[(d.case == tag) & (d.ray_c == RAYS[0])]
        return float(m[col].iloc[0]) if len(m) else np.nan
    for crit in ("bernoulli", "shear"):
        num = pick("d+5_+0_a8.58_L0", f"{crit}_dstar_mm")
        den = pick("d+0_+0_a11.83_L0", f"{crit}_dstar_mm")
        if np.isfinite(num) and np.isfinite(den):
            print(f"\nhigh-lift scale term, {crit:9s}: "
                  f"{10*np.log10(num/den):+.4f} dB")


if __name__ == "__main__":
    main()
