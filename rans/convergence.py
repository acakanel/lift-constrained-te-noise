#!/usr/bin/env python3
"""Grid and iteration convergence of the quantity the paper claims."""
from __future__ import annotations
import os, re, sys, json, time, glob, shutil, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import geometry, mesh, case as case_mod, post

OUT = os.path.join(HERE, "conv")
U = 55.0
FILLET = 0.008
WRITE_EVERY = 2000
N_ITER = 40000
NP = 2

_G0, _N0 = 1.14, 45
_BL_T = (_G0 ** _N0 - 1) / (_G0 - 1)


def _growth(n_bl: int) -> float:
    lo, hi = 1.0 + 1e-9, _G0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (mid ** n_bl - 1) / (mid - 1) > _BL_T:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


LEVELS = [
    dict(name="L0", h=1.00, n_bl=45, n_iter=20000),
    dict(name="L1", h=1.0 / np.sqrt(2), n_bl=64, n_iter=20000),
    dict(name="L2", h=0.50, n_bl=90, n_iter=40000),
]
for _l in LEVELS:
    _l["bl_growth"] = _growth(_l["n_bl"])

CASES = [
    dict(tag="neutral", d1=0.0, d2=0.0, alpha=12.00),
    dict(tag="deflect", d1=5.0, d2=0.0, alpha=8.58),
    dict(tag="sym",     d1=0.0, d2=0.0, alpha=0.00),
    dict(tag="neutral_lo", d1=0.0, d2=0.0, alpha=11.50),
    dict(tag="neutral_lm", d1=0.0, d2=0.0, alpha=11.83),
    dict(tag="mdeflect", d1=5.0, d2=0.0, alpha=2.89),
    dict(tag="mneutral", d1=0.0, d2=0.0, alpha=6.00),
]

TIGHT = "residualControl { p 1e-6; U 1e-7; k 1e-7; omega 1e-7; }"


def cells_from_checkmesh(cdir: str):
    p = os.path.join(cdir, "log.checkMesh")
    if not os.path.exists(p):
        return None
    m = re.search(r"^\s*cells:\s+(\d+)", open(p, errors="ignore").read(), re.M)
    return int(m.group(1)) if m else None


def parallel_times(cdir: str):
    out = []
    for t in glob.glob(os.path.join(cdir, "processor0", "*")):
        b = os.path.basename(t)
        if not os.path.isdir(t):
            continue
        try:
            v = int(b)
        except ValueError:
            continue
        if v > 0:
            out.append(v)
    return sorted(out)


def solver_state(cdir: str) -> dict:
    p = os.path.join(cdir, "log.simpleFoam")
    if not os.path.exists(p):
        return dict(converged=None)
    t = open(p, errors="ignore").read()
    conv = "SIMPLE solution converged in" in t
    m = re.search(r"SIMPLE solution converged in (\d+) iterations", t)
    it_conv = int(m.group(1)) if m else None
    n_it = t.count("\nTime = ") + _count_times(
        os.path.join(cdir, "log.stage1"))
    res = {}
    for f in ("p", "Ux", "Uy", "k", "omega"):
        hits = re.findall(rf"Solving for {f}, Initial residual = ([0-9.eE+-]+)", t)
        if hits:
            res[f] = float(hits[-1])
    return dict(converged=bool(conv), converged_at=it_conv,
                iterations=n_it, final_residuals=res)


class at_time:
    def __init__(self, t):
        self.t = float(t)

    def __enter__(self):
        import pyvista as pv
        self.saved = post._latest_vtk
        target = self.t

        def picker(case):
            dirs = [d for d in glob.glob(os.path.join(case, "VTK", "*_*"))
                    if os.path.isdir(d)]
            for d in dirs:
                if abs(float(d.rsplit("_", 1)[1]) - target) < 1e-6:
                    return (pv.read(os.path.join(d, "internal.vtu")),
                            pv.read(os.path.join(d, "boundary", "airfoil.vtp")))
            raise FileNotFoundError(f"no VTK export at t={target:g} in {case}")

        post._latest_vtk = picker
        return self

    def __exit__(self, *a):
        post._latest_vtk = self.saved
        return False


def written_times(cdir: str):
    return sorted(int(x) for x in os.listdir(cdir)
                  if x.isdigit() and int(x) > 0)


def force_history(cdir: str):
    base = os.path.join(cdir, "postProcessing", "forceCoeffs")
    if not os.path.isdir(base):
        return []
    t = sorted(os.listdir(base))[-1]
    cand = [x for x in os.listdir(os.path.join(base, t)) if x.endswith(".dat")]
    if not cand:
        return []
    hdr, rows = None, []
    for line in open(os.path.join(base, t, cand[0]), errors="ignore"):
        if line.startswith("#"):
            hdr = line
        else:
            rows.append(line.split())
    if not rows or hdr is None:
        return []
    cols = hdr.replace("#", "").split()
    if "Cl" not in cols:
        return []
    i, j = cols.index("Time"), cols.index("Cl")
    return [[float(r[i]), float(r[j])] for r in rows]


def _count_times(path: str) -> int:
    if not os.path.exists(path):
        return 0
    return open(path, errors="ignore").read().count("\nTime = ")


PSOLVER = """    p { solver GAMG; tolerance 1e-7; relTol 0.05; smoother DICGaussSeidel;
        nPreSweeps 0; nPostSweeps 2; cacheAgglomeration off;
        agglomerator faceAreaPair; nCellsInCoarsestLevel 200; mergeLevels 1; }"""


def solve(d: str, lvl=None) -> str:
    par = f"mpirun --allow-run-as-root -np {NP}"

    started = parallel_times(d) or written_times(d)
    if started:
        cd = os.path.join(d, "system", "controlDict")
        s = open(cd).read()
        s = re.sub(r"startFrom\s+\w+;", "startFrom       latestTime;", s)
        open(cd, "w").write(s)
        print(f"    resuming from iteration {max(started)}", flush=True)

    if not parallel_times(d):
        case_mod._sh("decomposePar -force > log.decomposePar 2>&1", d)
    case_mod._sh(f"{par} simpleFoam -parallel >> log.simpleFoam 2>&1",
                 d, timeout=72000)
    t = open(os.path.join(d, "log.simpleFoam"), errors="ignore").read()
    n = t.count("\nTime = ")
    want = lvl["n_iter"] if lvl else N_ITER
    if "SIMPLE solution converged in" not in t and n < 0.95 * want:
        raise RuntimeError(f"solver stopped at iteration {n} of {want}")
    return "single-stage"


def build_one(cfg, lvl, root, prepare_only=False):
    d = os.path.join(root, f"{cfg['tag']}_{lvl['name']}")
    done = os.path.join(d, "DONE.json")
    if os.path.exists(done):
        return json.load(open(done))

    resumable = (os.path.isdir(os.path.join(d, "constant", "polyMesh"))
                 and bool(written_times(d) or parallel_times(d)))
    if not resumable:
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    if resumable:
        n = len(written_times(d) or parallel_times(d))
        print(f"    {os.path.basename(d)}: mesh and {n} written "
              f"time(s) already present, resuming", flush=True)
        return _finish_one(cfg, lvl, d, resumed=True)

    sec = geometry.section(cfg["d1"], cfg["d2"], fillet_radius=FILLET)
    geo = os.path.join(d, "m.geo")
    msh = os.path.join(d, "m.msh")
    minfo = mesh.write_geo(sec, geo, U,
                           n_bl=lvl["n_bl"],
                           bl_growth=lvl["bl_growth"],
                           le_size=1.5e-3 * lvl["h"],
                           te_size=6.0e-4 * lvl["h"],
                           mid_size=5.0e-3 * lvl["h"],
                           ff_size=6.0 * lvl["h"])
    g = mesh.run_gmsh(geo, msh)
    if not g["ok"]:
        raise RuntimeError(f"gmsh failed:\n{g['log']}")

    case_mod.build(d, U, cfg["alpha"], n_iter=lvl["n_iter"], robust=True)
    fv = os.path.join(d, "system", "fvSolution")
    s = open(fv).read()
    s = re.sub(r"residualControl\s*\{[^}]*\}", TIGHT, s)
    s = re.sub(r"    p \{.*?\n(?=    \")", PSOLVER + "\n", s, flags=re.S)
    open(fv, "w").write(s)

    cd = os.path.join(d, "system", "controlDict")
    s = open(cd).read()
    s = re.sub(r"functions\s*\{.*\}\s*$", "functions { }\n", s, flags=re.S)
    s = re.sub(r"writeInterval\s+%d;" % lvl["n_iter"],
               "writeInterval   %d;" % WRITE_EVERY, s)
    s = re.sub(r"purgeWrite\s+\d+;", "purgeWrite      0;", s)
    s = s.replace("writeFormat     binary;", "writeFormat     ascii;")
    open(cd, "w").write(s)

    case_mod._sh(f"gmshToFoam {os.path.basename(msh)} > log.gmshToFoam 2>&1", d)
    case_mod.fix_boundary(d)
    case_mod._sh("checkMesh -constant > log.checkMesh 2>&1", d)
    if prepare_only:
        json.dump(dict(tag=cfg["tag"], level=lvl["name"], alpha=cfg["alpha"],
                       d1=cfg["d1"], d2=cfg["d2"], fillet=FILLET,
                       cells=cells_from_checkmesh(d), prepared=True),
                  open(os.path.join(d, "PREPARED.json"), "w"), indent=1)
        return d
    return _finish_one(cfg, lvl, d, minfo=minfo)


def _finish_one(cfg, lvl, d, minfo=None, resumed=False):
    minfo = minfo or {}
    done = os.path.join(d, "DONE.json")
    t0 = time.time()
    init = solve(d, lvl)
    case_mod._sh("reconstructPar > log.rec 2>&1", d)
    case_mod._sh("foamToVTK > log.vtk 2>&1", d)
    wall = time.time() - t0
    for p in glob.glob(os.path.join(d, "processor*")):
        shutil.rmtree(p, ignore_errors=True)

    st = solver_state(d)
    hist = []
    for it in written_times(d):
        try:
            with at_time(it):
                e = post.extract(d, U, cfg["alpha"])
            hist.append(dict(it=it, **{k: e.get(k) for k in
                                       ("CL", "dstar_s", "dstar_p",
                                        "H_s", "H_p")}))
        except Exception as exc:
            hist.append(dict(it=it, error=f"{type(exc).__name__}: {exc}"))

    final = post.extract(d, U, cfg["alpha"])
    out = dict(tag=cfg["tag"], level=lvl["name"], h=lvl["h"], n_bl=lvl["n_bl"],
               alpha=cfg["alpha"], d1=cfg["d1"], d2=cfg["d2"],
               cells=cells_from_checkmesh(d), init=init,
               y1=minfo.get("y1"), bl_thickness=minfo.get("bl_thickness"),
               wall_s=round(wall, 1), history=hist, cl_history=force_history(d),
               **st,
               **{k: final.get(k) for k in
                  ("CL", "CD", "CM", "dstar_s", "dstar_p", "H_s", "H_p",
                   "theta_s", "theta_p")})
    if final.get("dstar_p"):
        out["sym_error_dB"] = float(
            10 * np.log10(final["dstar_s"] / final["dstar_p"]))
    json.dump(out, open(done, "w"), indent=1)
    return out


def gci(f, r=np.sqrt(2.0)):
    f3, f2, f1 = f
    e21, e32 = f1 - f2, f2 - f3
    if abs(e21) < 1e-12 or abs(e32) < 1e-12:
        return dict(note="differences below round-off")
    s = e21 / e32
    if s <= 0:
        return dict(order=None, note="oscillatory convergence", ratio=float(s))
    p = np.log(abs(e32 / e21)) / np.log(r)
    ext = f1 + e21 / (r ** p - 1.0)
    band = 1.25 * abs(e21 / (r ** p - 1.0))
    return dict(order=float(p), extrapolated=float(ext), gci=float(band),
                ratio=float(s))


def settled(r, key, n=5):
    hist = r.get("history", [])
    if not hist:
        return None, None
    last = max(x["it"] for x in hist if isinstance(x.get("it"), (int, float)))
    fin = r.get(key)
    if fin is None or not np.isfinite(fin):
        return float("nan"), float("nan")
    h = [x for x in hist if isinstance(x.get(key), (int, float))
         and np.isfinite(x[key])]
    if not h:
        return None, None
    tail = [x for x in h if x["it"] >= 0.5 * last][-n:]
    if not tail:
        return None, None
    t = [x[key] for x in tail]
    if len(tail) < 2:
        return float(t[-1]), float("nan")
    return float(np.mean(t)), float(0.5 * (max(t) - min(t)))


def drift(r, key, frac=0.5):
    hist = r.get("history", [])
    if not hist:
        return float("nan")
    last = max(x["it"] for x in hist if isinstance(x.get("it"), (int, float)))
    h = [x for x in hist if isinstance(x.get(key), (int, float))
         and np.isfinite(x[key]) and x["it"] >= frac * last]
    if len(h) < 2:
        return float("nan")
    m, _ = settled(r, key)
    if m is None or not np.isfinite(m) or m == 0:
        return float("nan")
    return float((h[-1][key] - h[0][key]) / abs(m))


def report(results):
    by = {(r["tag"], r["level"]): r for r in results}
    lv = [l["name"] for l in LEVELS if any(k[1] == l["name"] for k in by)]
    print("\n" + "=" * 74)
    for tag in ("sym", "neutral_lo", "neutral_lm", "neutral", "deflect"):
        rs = [by[(tag, n)] for n in lv if (tag, n) in by]
        if not rs:
            continue
        print(f"\n{tag}")
        for r in rs:
            ms, es = settled(r, "dstar_s")
            mp, ep = settled(r, "dstar_p")
            print(f"  {r['level']}  cells={r['cells']:>7d}  CL={r['CL']:+.5f}  "
                  f"d*_s={1e3*ms:.4f}+-{1e3*es:.4f} mm  "
                  f"d*_p={1e3*mp:.4f}+-{1e3*ep:.4f} mm  "
                  f"conv={r['converged']}  {r['wall_s']:.0f}s")
        if tag == "sym":
            print("  symmetry error, which is exactly 0.0000 dB for this section:")
            for r in rs:
                ms, es = settled(r, "dstar_s")
                mp, ep = settled(r, "dstar_p")
                e = 10 * np.log10(ms / mp)
                band = 10 * np.log10((ms + es) / max(mp - ep, 1e-12)) - e
                print(f"    {r['level']}  {e:+.4f} dB  "
                      f"(iteration band +-{band:.4f})")
        if len(rs) == 3:
            g = gci([settled(r, "dstar_s")[0] for r in rs])
            print(f"  GCI on d*_s: {json.dumps(g)}")

    lv2 = [n for n in lv if ("neutral", n) in by and ("deflect", n) in by]
    if lv2:
        print("\nscale term 10 log10( d*_s(deflect) / d*_s(neutral at the same lift) )")
        vals = []
        for n in lv2:
            D = by[("deflect", n)]
            md, ed = settled(D, "dstar_s")
            pts = sorted((by[(t, n)]["CL"], settled(by[(t, n)], "dstar_s")[0],
                          settled(by[(t, n)], "dstar_s")[1])
                         for t in ("neutral", "neutral_lo", "neutral_lm")
                         if (t, n) in by)
            cl = np.array([p[0] for p in pts])
            ds = np.array([p[1] for p in pts])
            en = max(p[2] for p in pts)
            naive = 10 * np.log10(md / ds[-1])
            if len(pts) >= 2 and cl.min() <= D["CL"] <= cl.max():
                ref = float(np.interp(D["CL"], cl, ds))
                how = f"matched at C_L={D['CL']:.4f} from {len(pts)} neutral runs"
            else:
                ref = float(ds[np.argmin(np.abs(cl - D["CL"]))])
                how = (f"NOT matched: C_L={D['CL']:.4f} outside the neutral "
                       f"range [{cl.min():.4f}, {cl.max():.4f}]")
            v = 10 * np.log10(md / ref)
            band = 10 * np.log10((md + ed) / max(ref - en, 1e-12)) - v
            vals.append(v)
            print(f"  {n}  {v:+.4f} dB  (iteration band +-{band:.4f}; "
                  f"naive {naive:+.4f})")
            print(f"      {how}")
        if len(vals) == 3:
            g = gci(vals)
            print(f"  GCI on the scale term: {json.dumps(g)}")
        print("  No level change smaller than the larger of that index and the "
              "symmetry error above is a result.")


def init_check(results):
    old_path = os.path.join(OUT, "uniform_start_L0.json")
    if not os.path.exists(old_path):
        raise SystemExit("no uniform-start reference in conv/")
    old = {(r["tag"], r["level"]): r for r in json.load(open(old_path))}
    new = {(r["tag"], r["level"]): r for r in results}
    keys = sorted(set(old) & set(new))
    if not keys:
        raise SystemExit("nothing solved both ways yet")

    print("\nSame mesh, same schemes, same relaxation, same residual targets,")
    print("same iteration budget. Only the pressure solver differs.\n")
    print(f"  {'case':12s} {'solver':12s} {'C_L':>10s} {'d*_s [mm]':>11s} "
          f"{'H_s':>8s}")
    worst = 0.0
    for k in keys:
        for lab, r in (("GAMG/GS", old[k]), ("GAMG/DIC-GS", new[k])):
            print(f"  {k[0]+' '+k[1]:12s} {lab:12s} {r['CL']:10.5f} "
                  f"{1e3*r['dstar_s']:11.4f} {r['H_s']:8.4f}")
        dv = 10 * np.log10(new[k]["dstar_s"] / old[k]["dstar_s"])
        worst = max(worst, abs(dv))
        print(f"  {'':12s} {'difference':12s} "
              f"{new[k]['CL']-old[k]['CL']:+10.5f} "
              f"{1e3*(new[k]['dstar_s']-old[k]['dstar_s']):+11.4f} "
              f"{new[k]['H_s']-old[k]['H_s']:+8.4f}   "
              f"({dv:+.4f} dB in the scale term)")
    print(f"\n  largest difference attributable to the linear solver: "
          f"{worst:.4f} dB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--only", default=None, choices=["symmetry", "scale"])
    ap.add_argument("--cases", nargs="*", default=None,
                    help="run only these tags (default: all)")
    ap.add_argument("--report", action="store_true",
                    help="re-print the summary from conv/convergence.json")
    ap.add_argument("--init-check", action="store_true",
                    help="solve one coarse case from a potential-flow field and "
                         "compare it with the uniform-start answer")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jpath = os.path.join(OUT, "convergence.json")

    if a.report:
        report(json.load(open(jpath)))
        return

    if a.init_check:
        init_check(json.load(open(jpath)))
        return

    lv = [LEVELS[i] for i in a.levels]
    if a.only == "symmetry":
        cs = [c for c in CASES if c["tag"] == "sym"]
    elif a.only == "scale":
        cs = [c for c in CASES if c["tag"] != "sym"]
    else:
        cs = CASES
    if a.cases:
        want = set(a.cases)
        unknown = want - {c["tag"] for c in CASES}
        if unknown:
            raise SystemExit(f"unknown case tag(s): {sorted(unknown)}; "
                             f"known: {[c['tag'] for c in CASES]}")
        cs = [c for c in cs if c["tag"] in want]

    results = json.load(open(jpath)) if os.path.exists(jpath) else []
    seen = {(r["tag"], r["level"]) for r in results}
    for lvl in lv:
        for cfg in cs:
            if (cfg["tag"], lvl["name"]) in seen:
                continue
            print(f"\n=== {cfg['tag']} {lvl['name']} "
                  f"(h={lvl['h']:.3f}, n_bl={lvl['n_bl']}, "
                  f"growth={lvl['bl_growth']:.5f}) ===", flush=True)
            try:
                r = build_one(cfg, lvl, OUT)
                results.append(r)
                print(f"  cells={r['cells']}  CL={r['CL']:+.5f}  "
                      f"d*_s={1e3*r['dstar_s']:.4f} mm  H_s={r['H_s']:.3f}  "
                      f"conv={r['converged']}  {r['wall_s']:.0f}s", flush=True)
                if cfg["tag"] == "sym":
                    print(f"  symmetry error {r['sym_error_dB']:+.4f} dB",
                          flush=True)
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            json.dump(results, open(jpath, "w"), indent=1)
    report(results)
    summary = os.path.join(HERE, "convergence_summary.json")
    json.dump(results, open(summary, "w"), indent=1)
    print(f"\nwrote conv/convergence.json and "
          f"{os.path.basename(summary)}")


if __name__ == "__main__":
    main()
