#!/usr/bin/env python3
"""The lift-preserving authority, at matched lift."""
from __future__ import annotations
import numpy as np, pandas as pd
import ransdata

U_REF = 55.0
LEVELS = [("Moderate lift", 0.70, 0.55, 0.85),
          ("High lift", 1.33, 1.20, 1.40)]


def main():
    print(ransdata.banner())
    d = ransdata.sweep(U_REF)
    d = d[d.CL.notna()].copy()
    d["cfg"] = list(zip(d.d1, d.d2))

    base = d[(d.d1 == 0) & (d.d2 == 0) & d.dstar_s.notna()].sort_values("CL")
    print(f"flaps-neutral sweep: {len(base)} runs, "
          f"C_L {base.CL.min():.3f} to {base.CL.max():.3f}")
    print("  C_L strictly increasing (required): "
          f"{bool(np.all(np.diff(base.CL.values) > 0))}")

    for name, target, lo, hi in LEVELS:
        print(f"\n=== {name}, nominal C_L = {target:.2f} ===")
        sel = d[(d.CL > lo) & (d.CL < hi)].copy()
        sel["err"] = (sel.CL - target).abs()
        sel = sel.sort_values("err").drop_duplicates("cfg", keep="first")
        sel = sel.sort_values(["d1", "d2"])

        print(f"  {'d1':>5s} {'d2':>5s} {'alpha':>7s} {'C_L':>8s} {'d*_s':>8s} "
              f"{'H_s':>6s} {'d*_s(0) at':>11s} {'scale':>8s} {'naive':>8s}")
        print(f"  {'':>5s} {'':>5s} {'[deg]':>7s} {'':>8s} {'[mm]':>8s} "
              f"{'':>6s} {'same C_L':>11s} {'[dB]':>8s} {'[dB]':>8s}")
        ref_row = sel[(sel.d1 == 0) & (sel.d2 == 0)]
        if ref_row.empty:
            print("  no flaps-neutral run on this path")
            continue
        ref_naive = float(ref_row.dstar_s.iloc[0])

        for _, r in sel.iterrows():
            if not np.isfinite(r.dstar_s):
                print(f"  {r.d1:+5.0f} {r.d2:+5.0f} {r.alpha:7.2f} {r.CL:8.4f} "
                      f"{'--':>8s} {'--':>6s} {'--':>11s} {'--':>8s} {'--':>8s}"
                      f"   WITHDRAWN: separated, no boundary-layer edge")
                continue
            if not (base.CL.min() <= r.CL <= base.CL.max()):
                matched = np.nan
            else:
                matched = float(np.interp(r.CL, base.CL.values,
                                          base.dstar_s.values))
            scale = 10 * np.log10(r.dstar_s / matched) if np.isfinite(matched) \
                else np.nan
            naive = 10 * np.log10(r.dstar_s / ref_naive)
            print(f"  {r.d1:+5.0f} {r.d2:+5.0f} {r.alpha:7.2f} {r.CL:8.4f} "
                  f"{1e3*r.dstar_s:8.4f} {r.H_s:6.3f} {1e3*matched:11.4f} "
                  f"{scale:+8.3f} {naive:+8.3f}")

        dcl = sel.CL.max() - sel.CL.min()
        print(f"  lift spread along this path: {dcl:.4f} "
              f"({100*dcl/target:.1f} % of the nominal)")


if __name__ == "__main__":
    main()
