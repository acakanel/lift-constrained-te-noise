#!/usr/bin/env python3
"""What actually sets the achievable lift-error band."""
from __future__ import annotations
import os, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import compare as C, plant as P

HERE = os.path.dirname(os.path.abspath(__file__))
_RATE0 = P.RATE.copy()
SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)


def main(seeds=3):
    C.set_operating_point(55.0, 1.30)
    rows = []
    for sc in SCALES:
        for name, run in C.CONTROLLERS.items():
            v = []
            for k in range(seeds):
                P.RATE = _RATE0 * sc
                try:
                    L = run(C.Sur, C._turb(k), k)
                finally:
                    P.RATE = _RATE0.copy()
                ss = slice(int(0.6 * len(L["t"])), None)
                v.append(float(np.sqrt(np.mean(L["viol"][ss] ** 2))))
            rows.append(dict(rate_scale=sc, alpha_rate=_RATE0[0] * sc,
                             flap_rate=_RATE0[1] * sc, controller=name,
                             lift_RMS=float(np.mean(v)),
                             lift_RMS_sd=float(np.std(v))))
            print(f"  rate x{sc:<4g}  {name:10s} settled lift RMS "
                  f"{np.mean(v):.4f}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(HERE, "rate_limit_sweep.csv"), index=False)

    print("\n" + "=" * 60)
    print(f"{'rate':>8s} {'alpha deg/s':>12s} {'mean RMS':>10s} "
          f"{'spread across controllers':>26s}")
    for sc, g in d.groupby("rate_scale"):
        print(f"  x{sc:<6g} {g.alpha_rate.iloc[0]:11.1f} "
              f"{g.lift_RMS.mean():10.4f} "
              f"{g.lift_RMS.max()-g.lift_RMS.min():26.4f}")
    across_rate = d.groupby("controller").lift_RMS.agg(lambda x: x.max()-x.min())
    across_ctrl = d.groupby("rate_scale").lift_RMS.agg(lambda x: x.max()-x.min())
    print(f"\nspread across rate limits, worst controller : "
          f"{across_rate.max():.4f}")
    print(f"spread across controllers, worst rate limit : "
          f"{across_ctrl.max():.4f}")
    print(f"ratio: {across_rate.max()/max(across_ctrl.max(), 1e-12):.1f}")
    print("\nwrote rate_limit_sweep.csv")
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    main(ap.parse_args().seeds)
