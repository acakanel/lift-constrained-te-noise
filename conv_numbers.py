"""The grid-convergence appendix, as macros, from the solutions themselves."""
import os
import json
import numpy as np

from writeonce import write_if_changed, frame_if_changed

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "rans", "conv", "convergence.json")
OUT = os.path.join(HERE, "paper_AST", "conv_numbers.tex")
LEVELS = ["L0", "L1", "L2"]
TAG = {"L0": "A", "L1": "B", "L2": "C"}


def load():
    return {(r["tag"], r["level"]): r for r in json.load(open(SRC))}


def lift_sensitivity(R, level):
    ns = [R[(t, level)] for t in ("neutral", "neutral_lm", "neutral_lo")
          if (t, level) in R]
    if len(ns) < 2:
        return None, len(ns)
    cl = np.array([r["CL"] for r in ns])
    ds = np.array([1e3 * r["dstar_s"] for r in ns])
    return float(np.polyfit(cl, ds, 1)[0]), len(ns)


def band(rec, frac=0.5):
    h = [1e3 * x["dstar_s"] for x in rec["history"]]
    tail = h[int((1 - frac) * len(h)):]
    return 0.5 * (max(tail) - min(tail))


def moderate_sensitivity():
    import json as _json
    p = os.path.join(HERE, "rans", "resweep_summary.json")
    if not os.path.exists(p):
        return None
    rs = {r["tag"]: r for r in _json.load(open(p))}
    lo, hi = rs.get("d+0_+0_a6"), rs.get("d+0_+0_a6.27")
    if not (lo and hi):
        return None
    return ((1e3 * hi["dstar_s"] - 1e3 * lo["dstar_s"])
            / (hi["CL"] - lo["CL"]))


def moderate(R, V):
    slope = moderate_sensitivity()
    rows = []
    for lvl in LEVELS:
        if ("mdeflect", lvl) not in R or ("mneutral", lvl) not in R:
            continue
        a, b = R[("mdeflect", lvl)], R[("mneutral", lvl)]
        if not (a.get("dstar_s") and b.get("dstar_s")):
            continue
        da, db = 1e3 * a["dstar_s"], 1e3 * b["dstar_s"]
        dcl = a["CL"] - b["CL"]
        dbc = db + dcl * slope if slope else db
        scale = 10 * np.log10(da / dbc)
        t = TAG[lvl]
        V[f"cvM{t}Def"] = f"{da:.3f}"
        V[f"cvM{t}Neu"] = f"{db:.3f}"
        V[f"cvM{t}Scale"] = f"{scale:+.3f}"
        V[f"cvM{t}BandDef"] = f"{band(a):.3f}"
        V[f"cvM{t}BandNeu"] = f"{band(b):.3f}"
        V[f"cvM{t}HsDef"] = f"{a['H_s']:.3f}"
        V[f"cvM{t}HsNeu"] = f"{b['H_s']:.3f}"
        V[f"cvM{t}Cells"] = f"{a['cells']:,}".replace(",", "\\,")
        V[f"cvM{t}Corr"] = f"{scale - 10 * np.log10(da / db):+.3f}"
        V[f"cvM{t}dCL"] = f"{dcl:+.4f}"
        rows.append((lvl, scale))
    if slope:
        V["cvMSens"] = f"{slope:+.2f}"
    V["cvMLevels"] = str(len(rows))
    for (l0, s0), (l1, s1) in zip(rows, rows[1:]):
        V[f"cvMStep{TAG[l0]}{TAG[l1]}"] = f"{abs(s1 - s0):.3f}"
    if len(rows) >= 2:
        signs = {"+" if s > 0 else "-" for _, s in rows}
        V["cvMSignStable"] = "yes" if len(signs) == 1 else "no"
    vals = [s for _, s in rows]
    if len(vals) >= 3:
        diffs = np.diff(vals)
        if diffs[-1] != 0:
            ratio = diffs[0] / diffs[1]
            with np.errstate(all="ignore"):
                pp = np.log(abs(ratio)) / np.log(np.sqrt(2.0))
            V["cvMDiffRatio"] = f"{abs(ratio):.2f}"
            V["cvMOrder"] = f"{pp:+.1f}"
            V["cvMConverging"] = "yes" if abs(ratio) > 1 else "no"
            if abs(ratio) > 1:
                V["cvMRichardson"] = f"{vals[-1] + diffs[-1] / (2 ** (pp / 2) - 1):+.3f}"
    steps = [abs(b - a) for a, b in zip(vals, vals[1:])]
    if steps:
        V["cvMStepWorst"] = f"{max(steps):.3f}"
    return rows


def main():
    R = load()
    V, rows = {}, []
    fallback = lift_sensitivity(R, "L1")[0]

    for lvl in LEVELS:
        if ("deflect", lvl) not in R or ("neutral_lm", lvl) not in R:
            continue
        a, b = R[("deflect", lvl)], R[("neutral_lm", lvl)]
        da, db = 1e3 * a["dstar_s"], 1e3 * b["dstar_s"]
        s, n = lift_sensitivity(R, lvl)
        if s is None:
            s, n = fallback, 1
        dbc = db + (a["CL"] - b["CL"]) * s
        scale = 10 * np.log10(da / dbc)
        t = TAG[lvl]
        V[f"cv{t}Corr"] = f"{scale - 10 * np.log10(da / db):+.3f}"
        V[f"cv{t}dCL"] = f"{a['CL'] - b['CL']:+.4f}"
        V[f"cv{t}Cells"] = f"{a['cells']:,}".replace(",", "\\,")
        V[f"cv{t}Def"] = f"{da:.3f}"
        V[f"cv{t}Neu"] = f"{db:.3f}"
        V[f"cv{t}Scale"] = f"{scale:+.3f}"
        V[f"cv{t}BandDef"] = f"{band(a):.3f}"
        V[f"cv{t}BandNeu"] = f"{band(b):.3f}"
        V[f"cv{t}HsDef"] = f"{a['H_s']:.3f}"
        V[f"cv{t}HsNeu"] = f"{b['H_s']:.3f}"
        V[f"cv{t}Neutrals"] = str(n)
        V[f"cv{t}Iters"] = str(max(x["it"] for x in a["history"]))
        rows.append((lvl, scale))

    for (l0, s0), (l1, s1) in zip(rows, rows[1:]):
        V[f"cvStep{TAG[l0]}{TAG[l1]}"] = f"{abs(s1 - s0):.3f}"
    hi_steps = [abs(b - a) for (_, a), (_, b) in zip(rows, rows[1:])]
    if hi_steps:
        V["cvStepWorst"] = f"{max(hi_steps):.3f}"

    vals = [s for _, s in rows]
    diffs = np.diff(vals)
    V["cvLevels"] = str(len(rows))
    counts = [int(V[f"cv{TAG[l]}Cells"].replace("\\,", "")) for l, _ in rows]
    for (t0, c0), (t1, c1) in zip(zip([TAG[l] for l, _ in rows], counts),
                                  list(zip([TAG[l] for l, _ in rows], counts))[1:]):
        V[f"cvRatio{t0}{t1}"] = f"{c1 / c0:.2f}"
    V["cvSens"] = f"{fallback:+.2f}"
    if len(diffs) >= 2 and diffs[-1] != 0:
        ratio = diffs[0] / diffs[1]
        V["cvDiffRatio"] = f"{abs(ratio):.3f}"
        V["cvGrowth"] = f"{abs(1.0 / ratio):.0f}" if ratio else "n/a"
        with np.errstate(all="ignore"):
            p = np.log(abs(ratio)) / np.log(np.sqrt(2.0))
        V["cvOrder"] = f"{p:+.1f}"
        V["cvConverging"] = "yes" if abs(ratio) > 1 else "no"
    dsq = [float(V[f"cv{TAG[l]}Def"]) for l, _ in rows]
    V["cvThickMonotone"] = "yes" if (all(np.diff(dsq) > 0) or
                                     all(np.diff(dsq) < 0)) else "no"

    if ("sym", "L1") in R:
        r = R[("sym", "L1")]
        V["cvSymFloor"] = f"{abs(10*np.log10(r['dstar_s']/r['dstar_p'])):.3f}"

    mrows = moderate(R, V)

    lines = ["% conv_numbers.tex -- generated by conv_numbers.py. Do not edit.", ""]
    lines += ["\\newcommand{\\%s}{%s}" % (k, v) for k, v in V.items()]
    write_if_changed(OUT, "\n".join(lines) + "\n")

    print(f"{'level':<6}{'cells':>9}{'d*_def':>9}{'d*_neu':>9}{'scale dB':>10}"
          f"{'H_s def':>9}{'H_s neu':>9}")
    for lvl, scale in rows:
        t = TAG[lvl]
        print(f"{lvl:<6}{V[f'cv{t}Cells'].replace(chr(92)+',', ','):>9}"
              f"{V[f'cv{t}Def']:>9}{V[f'cv{t}Neu']:>9}{V[f'cv{t}Scale']:>10}"
              f"{V[f'cv{t}HsDef']:>9}{V[f'cv{t}HsNeu']:>9}")
    for k, v in V.items():
        if k.startswith("cvStep"):
            print(f"  change {k[6]} -> {k[7]}: {v} dB")
    print(f"  successive differences shrink: {V.get('cvConverging','n/a')} "
          f"(ratio {V.get('cvDiffRatio','n/a')}, apparent order "
          f"{V.get('cvOrder','n/a')})")
    print(f"  thicknesses monotone under refinement: {V['cvThickMonotone']}")
    if mrows:
        print("\nmoderate-lift pair:")
        for lvl, sc in mrows:
            t = TAG[lvl]
            print(f"  {lvl}  d*_def {V[f'cvM{t}Def']}  d*_neu {V[f'cvM{t}Neu']}  "
                  f"scale {V[f'cvM{t}Scale']} dB")
        for k, v in V.items():
            if k.startswith("cvMStep"):
                print(f"  step {k[7:]}: {v} dB")
        print(f"  sign stable across levels: {V.get('cvMSignStable','n/a')}")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
