#!/usr/bin/env python3
"""Regenerate the constant-lift manifold table from the resolved-flow sweep."""
from __future__ import annotations
import numpy as np, pandas as pd
import ransdata

U_REF = 55.0
CL_TARGET = 1.33
ROWS = [(5.0, 0.0), (5.0, 5.0), (0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
STATE = {(5.0, 0.0): "attached", (5.0, 5.0): "thickening",
         (0.0, 0.0): "attached", (10.0, 0.0): "separating",
         (10.0, 10.0): "separated"}


def load():
    d = ransdata.sweep(U_REF)
    return d[d.CL.notna()].copy()


def main():
    print(ransdata.banner())
    d = load()
    base = d[(d.d1 == 0) & (d.d2 == 0) & d.dstar_s.notna()].sort_values("CL")
    lo, hi = float(base.CL.min()), float(base.CL.max())
    print(f"flaps-neutral sweep: {len(base)} runs, C_L in [{lo:.4f}, {hi:.4f}]")
    xmono = bool(np.all(np.diff(base.CL.values) > 0))
    ymono = bool(np.all(np.diff(base.dstar_s.values) > 0))
    print(f"  C_L strictly increasing along the sweep (required by the "
          f"interpolation): {xmono}")
    print(f"  delta*_s monotone in C_L (a property of the flow, not a "
          f"requirement): {ymono}")
    if not xmono:
        raise SystemExit("the flaps-neutral sweep is not monotone in C_L; "
                         "np.interp would return silent nonsense")
    b = base.tail(2)
    slope = float(np.diff(b.dstar_s.values)[0] / np.diff(b.CL.values)[0])
    print(f"  d(delta*_s)/dC_L at the top of the sweep: "
          f"{1e3*slope:.2f} mm per unit C_L")

    out, tex = [], []
    for d1, d2 in ROWS:
        s = d[(d.d1 == d1) & (d.d2 == d2)]
        if s.empty:
            print(f"  no run for ({d1:+g},{d2:+g})")
            continue
        r = s.iloc[int((s.CL - CL_TARGET).abs().values.argmin())]
        if not np.isfinite(r.dstar_s):
            rev = float(r.get("reversed_s", np.nan))
            out.append(dict(d1=d1, d2=d2, alpha=float(r.alpha), CL=float(r.CL),
                            dstar_s=np.nan, H_s=np.nan, ref=np.nan,
                            ratio=np.nan, scale=np.nan, inside=None,
                            clamp_dB=np.nan, resolved=False,
                            reversed_s=rev))
            deg = "^\\circ"
            second = f"+{d2:.0f}{deg}" if d2 else f"{d2:.0f}"
            tex.append(f"\\(+{d1:.0f}{deg},\\,{second}\\) & {r.alpha:.2f} & "
                       f"{r.CL:.3f} & --- & --- & --- & "
                       f"{STATE[(d1, d2)]} \\\\")
            continue
        inside = lo <= r.CL <= hi
        ref = float(np.interp(r.CL, base.CL.values, base.dstar_s.values))
        scale = 10 * np.log10(r.dstar_s / ref)
        excess = 0.0 if inside else float(r.CL - hi)
        clamp_dB = 0.0 if inside else abs(
            10 * np.log10(r.dstar_s / (ref + slope * excess)) - scale)
        out.append(dict(d1=d1, d2=d2, alpha=float(r.alpha), CL=float(r.CL),
                        dstar_s=float(r.dstar_s), H_s=float(r.H_s),
                        ref=ref, ratio=float(r.dstar_s / ref),
                        scale=float(scale), inside=inside,
                        clamp_dB=clamp_dB, resolved=True,
                        reversed_s=float(r.get("reversed_s", np.nan))))
        deg = "^\\circ"
        eol = " \\\\"
        tag = "\\(0\\) (ref.)" if (d1 == 0 and d2 == 0) else f"\\({scale:+.2f}\\)"
        second = f"+{d2:.0f}{deg}" if d2 else f"{d2:.0f}"
        lead = f"\\(+{d1:.0f}{deg},\\,{second}\\)"
        tex.append(f"{lead} & {r.alpha:.2f} & {r.CL:.3f} & "
                   f"{1e3*r.dstar_s:.2f} & {r.H_s:.2f} & "
                   f"{tag} & {STATE[(d1, d2)]}{eol}")

    print(f"\n  {'d1,d2':>9s} {'alpha':>6s} {'C_L':>7s} {'d*_s':>7s} {'H_s':>5s} "
          f"{'ref':>7s} {'ratio':>6s} {'scale':>7s}  lift")
    for r in out:
        if not r["resolved"]:
            print(f"  {r['d1']:+4.0f},{r['d2']:+4.0f} {r['alpha']:6.2f} "
                  f"{r['CL']:7.4f} {'--':>7s} {'--':>5s} {'--':>7s} "
                  f"{'--':>6s} {'--':>7s}  WITHDRAWN: separated trailing edge, "
                  f"reversed over {r['reversed_s']:.0%} of the sampling ray")
            continue
        note = "in sweep" if r["inside"] else \
            f"ABOVE sweep by {r['CL']-hi:.4f} (worth {r['clamp_dB']:.3f} dB)"
        print(f"  {r['d1']:+4.0f},{r['d2']:+4.0f} {r['alpha']:6.2f} "
              f"{r['CL']:7.4f} {1e3*r['dstar_s']:7.3f} {r['H_s']:5.2f} "
              f"{1e3*r['ref']:7.3f} {r['ratio']:6.3f} {r['scale']:+7.3f}  {note}")

    sep = [r for r in out if (r["d1"], r["d2"]) == (10.0, 10.0)]
    if sep and sep[0]["resolved"]:
        s = sep[0]
        print(f"\nthe most deflected configuration is {s['ratio']:.2f} times "
              f"thicker than the flaps-neutral solution at the same lift, "
              f"a scale term of {s['scale']:+.2f} dB")
        print("(the text must quote these two, not a ratio taken against the "
              "flaps-neutral run at its own different lift)")
    else:
        print("\nthe most deflected configuration has no resolved boundary-layer "
              "thickness, so no ratio and no scale term can be quoted for it.")
        print("Any sentence in the text that gives a thickness ratio or a level "
              "for it has to be withdrawn, not re-estimated: the quantity was "
              "never measured, it was the length of the sampling ray.")

    print("\n--- LaTeX table body ---")
    for line in tex:
        print(line)

    pd.DataFrame(out).to_csv("manifold_table.csv", index=False)
    print("\nwrote manifold_table.csv")


if __name__ == "__main__":
    main()
