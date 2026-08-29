#!/usr/bin/env python3
"""How much of the headline number is the hinge fairing?"""
from __future__ import annotations
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import geometry, convergence as C

OUT = os.path.join(HERE, "fillet")
JSON = os.path.join(OUT, "fillet.json")

PAIR = [dict(tag="neutral", d1=0.0, d2=0.0, alpha=11.83),
        dict(tag="deflect", d1=5.0, d2=0.0, alpha=8.58)]
RADII = [0.0, 0.008, 0.012]


def surface_shift(d1, d2, r):
    a = np.asarray(geometry.section(d1, d2, fillet_radius=0.0)["coords"], float)
    b = np.asarray(geometry.section(d1, d2, fillet_radius=r)["coords"], float)
    if a.shape != b.shape:
        return float("nan")
    return float(np.abs(a - b).max())


def main():
    os.makedirs(OUT, exist_ok=True)
    done = json.load(open(JSON)) if os.path.exists(JSON) else []
    seen = {(r["tag"], r["fillet"]) for r in done}

    print("surface displacement from the fairing, in chords")
    for cfg in PAIR:
        for r in RADII:
            print(f"  {cfg['tag']:8s} r={r:.3f}: "
                  f"{surface_shift(cfg['d1'], cfg['d2'], r):.3e}")
    print()

    for r in RADII:
        for cfg in PAIR:
            if (cfg["tag"], r) in seen:
                continue
            print(f"=== {cfg['tag']}  fillet={r:.3f} ===", flush=True)
            C.FILLET = r
            c = dict(cfg, tag=f"{cfg['tag']}_f{r:.3f}")
            try:
                res = C.build_one(c, C.LEVELS[0], OUT)
                res["fillet"] = r
                res["tag"] = cfg["tag"]
                done.append(res)
                ms, es = C.settled(res, "dstar_s")
                print(f"  CL={res['CL']:+.5f}  d*_s={1e3*ms:.4f}+-{1e3*es:.4f} mm",
                      flush=True)
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            json.dump(done, open(JSON, "w"), indent=1)
    report(done)


def report(done):
    levels = sorted({r.get("level_tag", "high") for r in done})
    for lv in levels:
        sub = [r for r in done if r.get("level_tag", "high") == lv]
        if len({r["tag"] for r in sub}) < 2:
            continue
        print(f"\n{'=' * 62}\n  {lv} lift")
        _report_one(sub)
    return


def lift_slope(cl):
    import csv
    p = os.path.join(HERE, "rans_sweep_converged.csv")
    if not os.path.exists(p):
        return float("nan")
    rows = [r for r in csv.DictReader(open(p))
            if float(r["d1"]) == 0.0 and float(r["d2"]) == 0.0]
    if len(rows) < 2:
        return float("nan")
    rows.sort(key=lambda r: float(r["CL"]))
    x = np.array([float(r["CL"]) for r in rows])
    y = np.array([float(r["dstar_s"]) for r in rows])
    i = int(np.clip(np.searchsorted(x, cl), 1, len(x) - 1))
    return float((y[i] - y[i - 1]) / (x[i] - x[i - 1]))


def _report_one(done):
    by = {(r["tag"], r["fillet"]): r for r in done}
    print("\n" + "=" * 62)
    print(f"  {'fillet':>8s} {'C_L n':>9s} {'C_L d':>9s} {'d*_s n':>9s} "
          f"{'d*_s d':>9s} {'scale':>8s} {'dC_L':>8s} {'=dB':>7s}")
    base = None
    for r in sorted({k[1] for k in by}):
        if ("neutral", r) not in by or ("deflect", r) not in by:
            continue
        n, d = by[("neutral", r)], by[("deflect", r)]
        mn, _ = C.settled(n, "dstar_s")
        md, _ = C.settled(d, "dstar_s")
        sc = 10 * np.log10(md / mn)
        if base is None:
            base = sc
        dcl = d["CL"] - n["CL"]
        sl = lift_slope(0.5 * (d["CL"] + n["CL"]))
        mis = 10 * np.log10((md - sl * dcl) / mn) - sc if np.isfinite(sl) else float("nan")
        print(f"  {r:8.3f} {n['CL']:9.5f} {d['CL']:9.5f} {1e3*mn:9.4f} "
              f"{1e3*md:9.4f} {sc:+8.4f} {dcl:+8.5f} {mis:+7.4f}")
    vals = [10 * np.log10(C.settled(by[("deflect", r)], "dstar_s")[0] /
                          C.settled(by[("neutral", r)], "dstar_s")[0])
            for r in sorted({k[1] for k in by})
            if ("neutral", r) in by and ("deflect", r) in by]
    if len(vals) > 1:
        print(f"\n  spread across fairing radii: {max(vals)-min(vals):.4f} dB")
        print("  This belongs in the error budget. If it is comparable to the")
        print("  0.09 dB floor the appendix already carries, say so and move on;")
        print("  if it is larger, the fairing has to be fixed and everything")
        print("  re-solved.")


if __name__ == "__main__":
    main()
