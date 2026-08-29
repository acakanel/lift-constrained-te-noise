#!/usr/bin/env python3
"""Monte-Carlo robustness of the barrier-function adaptive law on the dynamic plant."""
import os, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import compare as C, plant as P

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
C.set_operating_point(55.0, 1.30)
LOUD_SPL = C.loud_level(55.0)
_RATE0 = P.RATE.copy()
_SIGMA0 = P.SIGMA
EPS = 0.12


SOURCES = ("mc_robustness.py", "compare.py", "control3.py", "plant.py",
           "surrogate.py")


def source_hashes():
    import hashlib
    h = {}
    for f in SOURCES:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            h[f] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    return h


_HASHES_AT_IMPORT = source_hashes()


def _write_provenance(N):
    import json
    json.dump(dict(N=N, eps=EPS, sources=_HASHES_AT_IMPORT),
              open(os.path.join(HERE, "mc_provenance.json"), "w"), indent=2)


def trial(seed, rng=None):
    if rng is None:
        rng = np.random.RandomState(1_000_000 + seed)
    b = rng.normal(0, 0.12)
    model = C.BiasedModel(C.Sur, b)
    sensor = rng.uniform(0.002, 0.006)
    rate_scale = rng.uniform(0.7, 1.3)
    P.RATE = _RATE0*rate_scale
    sigma_scale = rng.uniform(0.7, 1.4)
    P.SIGMA = _SIGMA0*sigma_scale
    g = P.VonKarmanGust(rng, amp=rng.uniform(0.03, 0.07))
    g.w = 2*np.pi*np.exp(rng.uniform(np.log(0.2), np.log(1.5), len(g.w)))
    try:
        L = C.run_bfasta(model, gust=g, sensor=sensor, seed=seed, eps=EPS)
    finally:
        P.RATE = _RATE0.copy()
        P.SIGMA = _SIGMA0
    ss = slice(int(0.6*len(L["t"])), None)
    assert abs(L["spl"][0] - LOUD_SPL) < 1e-3, (
        f"loud reference {LOUD_SPL:.4f} is not the level this trial starts from "
        f"({L['spl'][0]:.4f})")
    reduction = LOUD_SPL - np.mean(L["spl"][ss])
    settle_band = np.max(L["viol"][ss])
    whole_run = float(np.max(L["viol"]))
    inside0 = bool(L["viol"][0] < EPS)
    return dict(noise_reduction_dB=reduction, settled_lift_band=settle_band,
                whole_run_lift_max=whole_run, started_inside_band=inside0,
                lift_bias=b, sensor_noise=sensor, gust_amp=float(g.A*np.sqrt(len(g.w))),
                rate_scale=float(rate_scale),
                sigma_scale=float(sigma_scale))


def band_sweep(radii=(0.06, 0.08, 0.10, 0.12, 0.15, 0.20), n=2000):
    global EPS
    if n < 500:
        print(f"  [warning] n={n}: too few trials to resolve a sub-percent "
              f"exceedance rate. Debug only -- do not read the percentages.")
    keep = EPS
    print(f"{'eps':>6s} {'settled max':>12s} {'whole max':>11s} "
          f"{'settled in':>11s} {'whole in':>9s}")
    rows = []
    try:
        for e in radii:
            EPS = e
            r = [trial(k) for k in range(n)]
            b = np.array([x["settled_lift_band"] for x in r])
            w = np.array([x["whole_run_lift_max"] for x in r])
            rows.append(dict(eps=e, settled_max=b.max(), whole_max=w.max(),
                             settled_inside=float(np.mean(b < e)),
                             whole_inside=float(np.mean(w < e))))
            print(f"{e:6.2f} {b.max():12.4f} {w.max():11.4f} "
                  f"{100*np.mean(b<e):10.0f} % {100*np.mean(w<e):8.0f} %")
    finally:
        EPS = keep
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(HERE, "band_feasibility.csv"), index=False)
    print("\nwrote band_feasibility.csv")
    return d


def _chunk_seeds(seeds):
    out = []
    for k in seeds:
        r = trial(k)
        r["seed"] = k
        out.append(r)
    return out


CKPT = os.path.join(HERE, "mc_partial.csv")


def _resume():
    if not os.path.exists(CKPT):
        return {}
    try:
        d = pd.read_csv(CKPT)
        return {int(s): r for s, r in zip(d.seed, d.to_dict("records"))}
    except Exception:
        return {}


def _bank(rows):
    df = pd.DataFrame(rows).sort_values("seed")
    tmp = CKPT + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, CKPT)


def plot(D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle as _fs
    _fs.use()

    red = D.noise_reduction_dB.values
    band = D.settled_lift_band.values
    whole = D.whole_run_lift_max.values

    fig = plt.figure(figsize=(_fs.W15, 0.44 * _fs.W15))
    gs = fig.add_gridspec(1, 2, wspace=0.28, left=0.085, right=0.99,
                          bottom=0.18, top=0.87)
    ax = [fig.add_subplot(gs[0, k]) for k in range(2)]

    bias = D.lift_bias.values
    ax[0].scatter(bias, red, s=5.0, color=_fs.C[0], alpha=0.28,
                  linewidths=0, zorder=3, rasterized=True)
    order = np.argsort(bias)
    bs, rs = bias[order], red[order]
    k = max(25, len(bs) // 40)
    mid = np.array([np.median(rs[max(0, i-k):i+k]) for i in range(len(bs))])
    ax[0].plot(bs, mid, color=_fs.INK, lw=1.2, zorder=5, label="Running median")
    ax[0].axhline(red.mean(), color=_fs.INK2, lw=0.8, ls=(0, (2.5, 1.6)),
                  zorder=4, label=f"Mean, {red.mean():.2f} dB")
    ax[0].set_xlabel("Lift-model bias, $b$")
    ax[0].set_ylabel("Objective descent  [dB]")
    ax[0].set_title(f"Descent follows the model error ($r={np.corrcoef(bias, red)[0,1]:+.2f}$)",
                    pad=3)
    ax[0].legend(loc="lower right", fontsize=6.5)
    ax[0].set_ylim(0, max(red.max(), 0.1) * 1.10)
    _fs.panel(ax[0], "(a)", dx=-0.17, dy=1.05)

    ax[1].axvspan(0, EPS, color="#eef2f6", lw=0, zorder=0)
    hi = max(EPS, float(whole.max()))*1.04
    bins = np.linspace(0, hi, 30)
    ax[1].hist(whole, bins=bins, color=_fs.C[2], alpha=0.85,
               edgecolor="white", linewidth=0.4, zorder=3,
               label=f"Whole run ({int(np.sum(whole >= EPS))} of {len(whole)} outside)")
    ax[1].hist(band, bins=bins, color=_fs.C[0], alpha=0.55,
               edgecolor="white", linewidth=0.4, zorder=4,
               label=f"Settled window ({int(np.sum(band >= EPS))} of {len(band)} outside)")
    ax[1].axvline(EPS, color=_fs.C[1], lw=1.0, ls=(0, (2.5, 1.6)), zorder=5,
                  label=f"Band, $\\varepsilon={EPS}$")
    ax[1].set_xlim(0, hi)
    ax[1].set_xlabel("Maximum lift error,  $|s|$")
    ax[1].set_ylabel("Trials")
    ax[1].set_title("The constraint does not", pad=3)
    ax[1].legend(loc="upper right", fontsize=6.0, framealpha=0.95)
    _fs.panel(ax[1], "(b)", dx=-0.17, dy=1.05)
    ax[1].annotate(f"worst {whole.max():.3f}",
                   xy=(whole.max(), 2.0), xycoords="data",
                   xytext=(-4, 26), textcoords="offset points",
                   fontsize=6.5, ha="right", va="bottom", color=_fs.INK,
                   arrowprops=dict(arrowstyle="-", lw=0.55, color=_fs.INK2,
                                   shrinkA=1.5, shrinkB=1.5))
    _fs.save(fig, "fig10_montecarlo"); plt.close(fig)
    return os.path.join(FIG, "fig10_montecarlo.pdf")


def main(N, jobs=1):
    done = _resume()
    if done:
        print(f"resuming: {len(done)} of {N} trials already banked")
    if jobs > 1:
        import multiprocessing as mp
        todo = [k for k in range(N) if k not in done]
        if not todo:
            out = list(done.values())
        else:
            edges = np.linspace(0, len(todo), jobs + 1).astype(int)
            chunks = [todo[a:b] for a, b in zip(edges[:-1], edges[1:]) if b > a]
            with mp.Pool(jobs) as pool:
                parts = pool.map(_chunk_seeds, chunks)
            out = list(done.values()) + [r for p in parts for r in p]
        _bank(out)
    else:
        out = list(done.values())
        for k in range(N):
            if k in done:
                continue
            r = trial(k); r["seed"] = k; out.append(r)
            if len(out) % 25 == 0:
                _bank(out); print(f"  {len(out)}/{N} banked", flush=True)
        _bank(out)
    D = pd.DataFrame(out)
    red = D.noise_reduction_dB.values; band = D.settled_lift_band.values
    whole = D.whole_run_lift_max.values; start_in = D.started_inside_band.values
    D.to_csv(os.path.join(HERE, "mc_results.csv"), index=False)
    _write_provenance(N)
    print(f"\nMonte Carlo (N={N}, dynamic plant: actuator + circulation lag,")
    print(f"  barrier-function adaptive super-twisting, epsilon = {EPS}):")
    print(f"  noise reduction : mean {red.mean():.2f} dB, 5th pct {np.percentile(red,5):.2f}, min {red.min():.2f}")
    print(f"  settled lift band: mean {band.mean():.4f}, 95th pct {np.percentile(band,95):.4f}, max {band.max():.4f}")
    print(f"  whole-run max |s|: mean {whole.mean():.4f}, 95th pct {np.percentile(whole,95):.4f}, max {whole.max():.4f}")
    print(f"  P(reduction>5 dB) = {np.mean(red>5)*100:.0f}%")
    print(f"  P(settled band < {EPS}) = {np.mean(band<EPS)*100:.0f}%   "
          f"P(whole run < {EPS}) = {np.mean(whole<EPS)*100:.0f}%")
    print(f"  trials starting already inside the band: "
          f"{int(start_in.sum())}/{N} (the reaching phase is only exercised by "
          f"the remainder)")
    try:
        plot(D)
        print("  saved figs/fig10_montecarlo.png and mc_results.csv")
    except Exception as e:
        print("  (figure skipped:", e, ")")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=1,
                    help="processes; results are identical for any value")
    ap.add_argument("--band-sweep", action="store_true",
                    help="measure where the certified band stops being feasible")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the figure from mc_results.csv without "
                         "re-running the campaign")
    a = ap.parse_args()
    if a.plot_only:
        d = pd.read_csv(os.path.join(HERE, "mc_results.csv"))
        print(f"redrawing from {len(d)} stored trials")
        print("  ->", plot(d))
    elif a.band_sweep:
        band_sweep()
    else:
        main(a.n, a.jobs)
