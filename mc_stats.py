"""Every number Section 8.1 quotes, computed from the campaign file."""
import os
import numpy as np

from writeonce import write_if_changed, frame_if_changed
import pandas as pd
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "mc_results.csv")
OUT = os.path.join(HERE, "paper_AST", "mc_numbers.tex")


def _eps():
    src = open(os.path.join(HERE, "mc_robustness.py")).read()
    for line in src.splitlines():
        if line.startswith("EPS = "):
            return float(line.split("=")[1].split("#")[0])
    raise RuntimeError("EPS not found in mc_robustness.py")


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return 100 * lo, 100 * hi


def bootstrap_ci(x, n_boot=20000, seed=7):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(x), (n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def gust_threshold(g, w, b, eps):
    exceed = (w >= eps) | (b >= eps)
    if not exceed.any():
        return None
    return float(g[exceed].min())


def _check_provenance():
    import json
    import hashlib
    p = os.path.join(HERE, "mc_provenance.json")
    if not os.path.exists(p):
        return ["no mc_provenance.json: this campaign predates the check, so "
                "which sources produced it cannot be established"]
    prov = json.load(open(p))
    bad = []
    for f, want in prov.get("sources", {}).items():
        q = os.path.join(HERE, f)
        got = (hashlib.sha256(open(q, "rb").read()).hexdigest()[:16]
               if os.path.exists(q) else "missing")
        if got != want:
            bad.append(f"{f}: campaign ran {want}, on disk now {got}")
    if not bad:
        return []

    return _reproduces(bad)


def _reproduces(bad, k=12, tol=1e-9):
    import numpy as _np
    import pandas as _pd
    try:
        import mc_robustness as M
        d = _pd.read_csv(CSV)
        if "seed" not in d.columns:
            return bad + ["and the campaign file records no seeds, so it "
                          "cannot be spot-checked"]
        rng = _np.random.RandomState(0)
        seeds = sorted(rng.choice(d.seed.values, size=min(k, len(d)),
                                  replace=False).tolist())
        cols = ["noise_reduction_dB", "settled_lift_band", "whole_run_lift_max"]
        by = d.set_index("seed")
        worst, where = 0.0, None
        for sd in seeds:
            got = M.trial(int(sd))
            for c in cols:
                e = abs(float(got[c]) - float(by.loc[sd, c]))
                if e > worst:
                    worst, where = e, (sd, c)
    except Exception as e:                       # noqa: BLE001
        return bad + [f"and the spot-check could not run: {type(e).__name__}: {e}"]
    if worst <= tol:
        print(f"  [note] sources have changed since the campaign, but {len(seeds)} "
              f"stored trials reproduce exactly (worst difference {worst:.1e}); "
              f"the campaign stands.")
        return []
    return bad + [f"and a spot-check of {len(seeds)} stored trials does NOT "
                  f"reproduce: seed {where[0]} differs by {worst:.3e} in "
                  f"{where[1]}"]


def main(strict=True):
    stale = _check_provenance()
    if stale:
        msg = ("the campaign file was not produced by the sources now on disk:\n  "
               + "\n  ".join(stale) + "\n  re-run mc_robustness.py")
        if strict:
            raise SystemExit("mc_stats: " + msg)
        print("  [warning] " + msg)
    eps = _eps()
    d = pd.read_csv(CSV)
    n = len(d)
    b = d.settled_lift_band.values
    w = d.whole_run_lift_max.values
    g = d.gust_amp.values
    r = d.noise_reduction_dB.values

    s_in, w_in = int((b < eps).sum()), int((w < eps).sum())
    s_ex, w_ex = n - s_in, n - w_in
    start_in = int(d.started_inside_band.sum())

    corr = {c: float(np.corrcoef(d[c].values, w)[0, 1])
            for c in ["gust_amp", "lift_bias", "rate_scale",
                      "sigma_scale", "sensor_noise"]}
    corr["abs_lift_bias"] = float(np.corrcoef(np.abs(d.lift_bias.values), w)[0, 1])

    gt = gust_threshold(g, w, b, eps)
    below = g < gt
    n_below, n_above = int(below.sum()), int((~below).sum())
    wmax_below = float(w[below].max())
    s_ex_above = int((b[~below] >= eps).sum())
    w_ex_above = int((w[~below] >= eps).sum())

    dec = g >= np.percentile(g, 90)
    w_ex_top = 100 * float(np.mean(w[dec] >= eps))

    s_lo, s_hi = clopper_pearson(s_ex, n)
    w_lo, w_hi = clopper_pearson(w_ex, n)
    r_lo, r_hi = bootstrap_ci(r)

    V = dict(
        mcN=n, mcEps=eps,
        mcSettledIn=s_in, mcWholeIn=w_in,
        mcSettledEx=s_ex, mcWholeEx=w_ex,
        mcSettledExPct=100 * s_ex / n, mcWholeExPct=100 * w_ex / n,
        mcSettledExLo=s_lo, mcSettledExHi=s_hi,
        mcWholeExLo=w_lo, mcWholeExHi=w_hi,
        mcSettledMax=float(b.max()), mcWholeMax=float(w.max()),
        mcSettledMaxRel=float(b.max()) / eps, mcWholeMaxRel=float(w.max()) / eps,
        mcStartIn=start_in, mcStartOut=n - start_in,
        mcCorrGust=corr["gust_amp"], mcCorrBias=corr["lift_bias"],
        mcCorrRate=corr["rate_scale"], mcCorrSigma=corr["sigma_scale"],
        mcCorrNoise=corr["sensor_noise"],
        mcGustLo=float(g.min()), mcGustHi=float(g.max()),
        mcGustThresh=gt,
        mcNBelow=n_below, mcNAbove=n_above,
        mcWholeMaxBelow=wmax_below, mcWholeMaxBelowRel=wmax_below / eps,
        mcSettledExAbove=s_ex_above, mcWholeExAbove=w_ex_above,
        mcSettledExAbovePct=100 * s_ex_above / n_above,
        mcWholeExAbovePct=100 * w_ex_above / n_above,
        mcWholeExTopDecPct=w_ex_top,
        mcDescentMean=float(r.mean()), mcDescentLo=float(r_lo),
        mcDescentHi=float(r_hi), mcDescentPfive=float(np.percentile(r, 5)),
        mcDescentMin=float(r.min()), mcDescentMax=float(r.max()),
        mcDescentNeg=int((r < 0).sum()),
        mcDescentNegPct=100 * float(np.mean(r < 0)),
        mcDescentCIHalf=(r_hi - r_lo) / 2,
        mcCorrDescentBias=float(np.corrcoef(d.lift_bias.values, r)[0, 1]),
        mcCorrDescentNext=abs(max(
            (float(np.corrcoef(d[c].values, r)[0, 1])
             for c in ["gust_amp", "rate_scale", "sigma_scale", "sensor_noise"]),
            key=abs)),
        mcRuleOfThreeBelow=300.0 / int((g < gust_threshold(g, w, b, eps)).sum()),
    )

    V["mcWorstBias"] = float(d.lift_bias.values[int(np.argmin(r))])
    neg = r < 0
    if neg.any():
        V["mcNegBiasLo"] = float(d.lift_bias.values[neg].min())
        V["mcNegBiasHi"] = float(d.lift_bias.values[neg].max())
    else:
        V["mcNegBiasLo"] = V["mcNegBiasHi"] = None

    ints = {"mcN", "mcSettledIn", "mcWholeIn", "mcSettledEx", "mcWholeEx",
            "mcStartIn", "mcStartOut", "mcNBelow", "mcNAbove",
            "mcSettledExAbove", "mcWholeExAbove", "mcDescentNeg"}

    def fmt(k, v):
        if v is None:
            return "\\textbf{[no such trial]}"
        if k in ints:
            return str(int(v))
        if k in ("mcEps", "mcGustLo", "mcGustHi"):
            return f"{v:.4g}"
        if k == "mcGustThresh":
            return f"{v:.3f}"
        if k == "mcCorrDescentNext":
            return f"{v:.2f}"
        if k.startswith("mcCorr"):
            return f"{v:+.2f}"
        if k == "mcDescentCIHalf":
            return f"{v:.3f}"
        if k in ("mcNegBiasLo", "mcNegBiasHi", "mcWorstBias"):
            return f"{v:+.3f}"
        if "Max" in k and "Rel" not in k and not k.startswith("mcDescent"):
            return f"{v:.4f}"
        if k in ("mcSettledExAbovePct", "mcWholeExAbovePct"):
            return f"{v:.1f}"
        if k == "mcWholeExTopDecPct":
            return f"{v:.0f}"
        return f"{v:.2f}"

    lines = ["% mc_numbers.tex -- generated by mc_stats.py. Do not edit by hand.",
             "% Every value is read from mc_results.csv; editing this file",
             "% silently decouples Section 8.1 from the campaign it reports.",
             f"% campaign: N={n}, epsilon={eps}", ""]
    for k, v in V.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{fmt(k, v)}}}")
    write_if_changed(OUT, "\n".join(lines) + "\n")

    print(f"campaign N={n}, epsilon={eps}\n")
    print(f"  settled inside  {s_in}/{n}  ({100*s_in/n:.2f}%), max {b.max():.4f} "
          f"= {b.max()/eps:.2f} eps")
    print(f"  whole-run inside {w_in}/{n}  ({100*w_in/n:.2f}%), max {w.max():.4f} "
          f"= {w.max()/eps:.2f} eps")
    print(f"  started inside   {start_in}/{n}")
    print("\n  correlation of whole-run max with:")
    for k, v in sorted(corr.items(), key=lambda kv: -abs(kv[1])):
        print(f"    {k:<14s} {v:+.3f}")
    print(f"\n  disturbance amplitude spans {g.min():.3f}-{g.max():.3f}")
    print(f"  smallest amplitude with any exceedance: {gt:.4f}")
    print(f"    below it: {n_below} trials, 0 exceedances, worst |s| {wmax_below:.4f} "
          f"= {wmax_below/eps:.2f} eps")
    print(f"    above it: {n_above} trials, settled {s_ex_above} "
          f"({100*s_ex_above/n_above:.1f}%), whole {w_ex_above} "
          f"({100*w_ex_above/n_above:.1f}%)")
    print(f"    top decile of amplitude: whole-run exceeded in {w_ex_top:.0f}%")
    print(f"\n  settled exceedance rate {100*s_ex/n:.2f}% "
          f"[{s_lo:.2f}, {s_hi:.2f}] %  (Clopper-Pearson)")
    print(f"  whole-run exceedance rate {100*w_ex/n:.2f}% "
          f"[{w_lo:.2f}, {w_hi:.2f}] %")
    print(f"\n  descent mean {r.mean():.2f} dB [{r_lo:.2f}, {r_hi:.2f}], "
          f"5th pct {np.percentile(r,5):.2f}, min {r.min():.2f}, max {r.max():.2f}")
    print(f"  negative-descent trials {int((r<0).sum())} "
          f"({100*np.mean(r<0):.1f}%)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
