#!/usr/bin/env python3
"""Re-solve every configuration in the sweep at the converged budget."""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import convergence as C

OUT = os.path.join(HERE, "resweep")
JSON = os.path.join(OUT, "resweep.json")
CSV = os.path.join(HERE, "rans_sweep_converged.csv")
LEVEL = C.LEVELS[0]


def configurations():
    d = pd.read_csv(os.path.join(HERE, "rans_sweep.csv"))
    rows = [dict(tag=f"d{int(r.d1):+d}_{int(r.d2):+d}_a{r.alpha:g}",
                 d1=float(r.d1), d2=float(r.d2), alpha=float(r.alpha))
            for r in d.itertuples()]
    have = {(r["d1"], r["d2"], round(r["alpha"], 2)) for r in rows}
    extra = [
        (0.0, 0.0, 10.50), (0.0, 0.0, 11.00),
        (0.0, 0.0, 11.50), (0.0, 0.0, 11.83),
        (0.0, 0.0, 4.50), (0.0, 0.0, 7.50),
        (5.0, 5.0, 0.77), (10.0, 0.0, -0.08), (10.0, 10.0, -3.63),
    ]
    for d1, d2, a in extra:
        if (d1, d2, round(a, 2)) in have:
            continue
        rows.append(dict(tag=f"d{int(d1):+d}_{int(d2):+d}_a{a:g}",
                         d1=d1, d2=d2, alpha=a))
    return rows


def solve_all():
    os.makedirs(OUT, exist_ok=True)
    done = json.load(open(JSON)) if os.path.exists(JSON) else []
    seen = {r["tag"] for r in done}
    cfgs = configurations()
    print(f"{len(cfgs)} configurations, {len(seen)} already solved\n", flush=True)
    for cfg in cfgs:
        if cfg["tag"] in seen:
            continue
        print(f"=== {cfg['tag']}  (d1={cfg['d1']:+g}, d2={cfg['d2']:+g}, "
              f"alpha={cfg['alpha']:g}) ===", flush=True)
        try:
            r = C.build_one(cfg, LEVEL, OUT)
            done.append(r)
            ms, es = C.settled(r, "dstar_s")
            print(f"  CL={r['CL']:+.5f}  d*_s={1e3*ms:.4f}+-{1e3*es:.4f} mm  "
                  f"H_s={r['H_s']:.4f}  {r['wall_s']:.0f}s", flush=True)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
        json.dump(done, open(JSON, "w"), indent=1)
    return done


def write_csv(done):
    rows = []
    for r in done:
        ms, es = C.settled(r, "dstar_s")
        mp, ep = C.settled(r, "dstar_p")
        hs, _ = C.settled(r, "H_s")
        hp, _ = C.settled(r, "H_p")
        hs = r["H_s"] if hs is None else hs
        hp = r["H_p"] if hp is None else hp
        rows.append(dict(d1=r["d1"], d2=r["d2"], U=C.U, alpha=r["alpha"],
                         CL=r["CL"], CD=r["CD"], CM=r["CM"],
                         dstar_s=ms, dstar_s_band=es,
                         dstar_p=mp, dstar_p_band=ep,
                         H_s=hs, H_p=hp,
                         theta_s=r["theta_s"], theta_p=r["theta_p"],
                         bl_ok_s=r.get("bl_ok_s"), bl_ok_p=r.get("bl_ok_p"),
                         reversed_s=r.get("reversed_s"),
                         reversed_p=r.get("reversed_p"),
                         drift_s=C.drift(r, "dstar_s"),
                         cells=r["cells"], iterations=r.get("iterations"),
                         case=os.path.join("resweep", f"{r['tag']}")))
    d = pd.DataFrame(rows).sort_values(["d1", "d2", "alpha"])
    d.to_csv(CSV, index=False)
    json.dump(done, open(os.path.join(HERE, "resweep_summary.json"), "w"),
              indent=1)
    print(f"\nwrote {os.path.basename(CSV)}: {len(d)} configurations")
    return d


def report(done):
    new = write_csv(done)
    old = pd.read_csv(os.path.join(HERE, "rans_sweep.csv"))
    print("\nwhat changed, against the 3000-iteration sweep")
    print(f"  {'d1':>4s} {'d2':>4s} {'alpha':>6s} {'C_L old':>9s} {'C_L new':>9s} "
          f"{'dC_L %':>7s} {'d*_s old':>9s} {'d*_s new':>9s} {'dd*_s %':>8s} "
          f"{'in dB':>7s}")
    for _, n in new.iterrows():
        m = old[(old.d1 == n.d1) & (old.d2 == n.d2) &
                (np.abs(old.alpha - n.alpha) < 1e-6)]
        if m.empty:
            print(f"  {n.d1:+4.0f} {n.d2:+4.0f} {n.alpha:6.2f} "
                  f"{'':>9s} {n.CL:9.4f} {'':>7s} {'':>9s} "
                  f"{1e3*n.dstar_s:9.4f}   (new)")
            continue
        o = m.iloc[0]
        dcl = 100 * (n.CL - o.CL) / abs(o.CL) if o.CL else np.nan
        dds = 100 * (n.dstar_s - o.dstar_s) / o.dstar_s
        db = 10 * np.log10(n.dstar_s / o.dstar_s)
        print(f"  {n.d1:+4.0f} {n.d2:+4.0f} {n.alpha:6.2f} {o.CL:9.4f} "
              f"{n.CL:9.4f} {dcl:+7.2f} {1e3*o.dstar_s:9.4f} "
              f"{1e3*n.dstar_s:9.4f} {dds:+8.2f} {db:+7.3f}")
    print("\n  A large change in the last column with a small one in dC_L is the")
    print("  signature this study was built to find: the forces had settled and")
    print("  the boundary layer had not.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report(json.load(open(JSON)))
        return
    report(solve_all())


if __name__ == "__main__":
    main()
