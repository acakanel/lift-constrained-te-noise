#!/usr/bin/env python3
"""The validation statistics, stated as what they are."""
from __future__ import annotations
import numpy as np, pandas as pd

import bpm_noise as v1
import bpm_noise_v2 as v2
from bpm_bl import dstar_p_from_dstar_s, infer_tripped

NU, C0 = 1.5e-5, 340.46
SPAN, RE_OBS = 0.4572, 1.22
ALPHA_CAP = 12.5


def predict(model, ds, dp, U, alpha, c, freqs):
    raw = np.asarray(model.spl_third_octave(ds, dp, U, alpha, c, span=SPAN,
                                            r=RE_OBS, freqs=freqs), float)
    return raw - 10 * np.log10((U / C0) ** 5 * ds * SPAN / RE_OBS ** 2)


def per_condition(d):
    rows = []
    for (alpha, c, U, ds), g in d.groupby(["alpha", "c", "U_infinity", "delta"]):
        if len(g) < 5:
            continue
        g = g.sort_values("f")
        f, meas = g.f.values.astype(float), g.SSPL.values.astype(float)
        trip = bool(infer_tripped(ds, alpha, U * c / NU, c))
        dp = float(dstar_p_from_dstar_s(ds, alpha, trip))
        old = predict(v1, ds, ds, U, alpha, c, f)
        new = predict(v2, ds, dp, U, alpha, c, f)
        rows.append(dict(alpha=alpha, c=c, U=U, dstar_s=ds, n=len(g),
                         alpha_switch=float(v2.alpha_switch(U / C0)),
                         err_old=float(np.mean(old - meas)),
                         err_new=float(np.mean(new - meas)),
                         rmse_old=float(np.sqrt(np.mean((old - meas) ** 2))),
                         rmse_new=float(np.sqrt(np.mean((new - meas) ** 2)))))
    return pd.DataFrame(rows)


def block(name, R):
    e, r = R.err_new.values, R.rmse_new.values
    eo, ro = R.err_old.values, R.rmse_old.values
    print(f"\n{name}  (n = {len(R)})")
    print(f"  complete formulation")
    print(f"    mean level error        {e.mean():+7.2f} dB   "
          f"(median {np.median(e):+.2f}, sd {e.std(ddof=1):.2f})")
    print(f"    per-condition band RMSE  median {np.median(r):6.2f} dB   "
          f"mean {r.mean():.2f}   pooled {np.sqrt(np.mean(r ** 2)):.2f}   "
          f"IQR [{np.percentile(r, 25):.2f}, {np.percentile(r, 75):.2f}]   "
          f"max {r.max():.2f}")
    print(f"    conditions within 2 dB in level  {100*np.mean(np.abs(e)<=2):.0f}%"
          f"   within 3 dB  {100*np.mean(np.abs(e)<=3):.0f}%")
    print(f"  reduced formulation")
    print(f"    mean level error        {eo.mean():+7.2f} dB")
    print(f"    per-condition band RMSE  median {np.median(ro):6.2f} dB   "
          f"mean {ro.mean():.2f}")


def main():
    d = pd.read_csv("NASA_selfnoise.csv")
    print(f"database: {len(d)} band levels, "
          f"{d.groupby(['alpha','c','U_infinity','delta']).ngroups} conditions, "
          f"alpha {d.alpha.min():.1f} to {d.alpha.max():.1f} deg")

    R_all = per_condition(d)
    R_cap = R_all[R_all.alpha <= ALPHA_CAP]
    below = R_cap[R_cap.alpha <= R_cap.alpha_switch]
    above = R_cap[R_cap.alpha > R_cap.alpha_switch]

    block(f"whole database", R_all)
    block(f"at or below the incidence cap of {ALPHA_CAP} deg "
          f"(the population the paper's figures use)", R_cap)
    block("of those, below the separated-flow switch (alpha*)_0", below)
    block("of those, above the separated-flow switch (alpha*)_0", above)

    print(f"\nconditions excluded by the cap: {len(R_all) - len(R_cap)}, "
          f"at alpha {sorted(set(R_all.alpha[R_all.alpha > ALPHA_CAP]))}")
    R_all.to_csv("validation_all_conditions.csv", index=False)
    print("wrote validation_all_conditions.csv")


if __name__ == "__main__":
    main()
