#!/usr/bin/env python3
"""Incidence sweep on one mesh by freestream rotation, for the undeflected section."""
from __future__ import annotations
import os, re, sys, glob, shutil, subprocess, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geometry, mesh, case, post

FOAM = "/usr/share/openfoam/etc/bashrc"
OUT = "rans_sweep.csv"


def _sh(cmd, cwd, timeout=7200):
    return subprocess.run(["bash", "-lc", f"source {FOAM} >/dev/null 2>&1; {cmd}"],
                          cwd=cwd, capture_output=True, text=True, timeout=timeout)


FILLET = 0.008


def build_mesh(tag, d1, d2, U):
    d = f"mesh_{tag}"
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(f"{d}/case.msh"):
        r_fillet = 0.0 if (d1 == 0 and d2 == 0) else FILLET
        sec = geometry.section(d1, d2, fillet_radius=r_fillet)
        mesh.write_geo(sec, f"{d}/case.geo", U)
        r = mesh.run_gmsh(f"{d}/case.geo", f"{d}/case.msh")
        if not r["ok"]:
            raise RuntimeError(r["log"])
    return d


def mesh_quality(cdir):
    f = os.path.join(cdir, "log.checkMesh")
    if not os.path.exists(f):
        return {}
    t = open(f, errors="ignore").read()
    out = {}
    for key, pat in [("cells", r"cells:\s+(\d+)"),
                     ("nonorth_max", r"non-orthogonality Max:\s+([\d.]+)"),
                     ("skew_max", r"Max skewness = ([\d.]+)")]:
        m = re.search(pat, t)
        if m:
            out[key] = float(m.group(1))
    out["mesh_ok"] = "Mesh OK" in t
    return out


def solved(cdir):
    vt = [x for x in glob.glob(os.path.join(cdir, "VTK", "*_*")) if os.path.isdir(x)]
    return any(float(x.rsplit("_", 1)[1]) > 0 for x in vt)


def _attempt(cdir, mdir, U, alpha, n_iter, np_proc, robust):
    shutil.rmtree(cdir, ignore_errors=True)
    os.makedirs(cdir)
    shutil.copy(f"{mdir}/case.msh", f"{cdir}/case.msh")

    case.build(cdir, U, alpha, n_iter=n_iter, robust=robust)
    s = open(f"{cdir}/system/controlDict").read()
    s = re.sub(r"functions\s*\{.*\}\s*$", "functions { }\n", s, flags=re.S)
    s = s.replace("writeFormat     binary;", "writeFormat     ascii;")
    open(f"{cdir}/system/controlDict", "w").write(s)

    _sh("gmshToFoam case.msh > log.gmshToFoam 2>&1", cdir)
    case.fix_boundary(cdir)
    _sh("checkMesh -constant > log.checkMesh 2>&1", cdir)
    if os.environ.get("USE_POTENTIAL", "0") == "1":
        _sh("potentialFoam -writePhi > log.potentialFoam 2>&1", cdir)
    _sh("decomposePar -force > log.decomposePar 2>&1", cdir)
    t0 = time.time()
    _sh(f"mpirun --allow-run-as-root -np {np_proc} simpleFoam -parallel > log.simpleFoam 2>&1",
        cdir, timeout=7200)
    _sh("reconstructPar -latestTime > log.rec 2>&1", cdir)
    _sh("foamToVTK -latestTime > log.vtk 2>&1", cdir)
    return time.time() - t0


def run_one(tag, d1, d2, U, alpha, n_iter=3000, np_proc=2):
    cdir = f"run_{tag}_a{alpha:g}"
    mdir = build_mesh(tag, d1, d2, U)

    deflected = not (d1 == 0 and d2 == 0)
    wall = _attempt(cdir, mdir, U, alpha, n_iter if not deflected else 5000,
                    np_proc, robust=deflected)
    used = "robust" if deflected else "standard"
    if not solved(cdir):
        wall += _attempt(cdir, mdir, U, alpha, 6000, np_proc, robust=True)
        used = "robust (retry)"
    if not solved(cdir):
        raise RuntimeError("no converged field written")

    log = open(f"{cdir}/log.simpleFoam", errors="ignore").read()
    n_it = log.count("\nTime = ")
    r = post.extract(cdir, U, alpha)

    import pyvista as _pv
    _, _patch = post._latest_vtk(cdir)
    p_max = float(np.abs(np.asarray(_patch.cell_data["p"], float)).max())
    q = 0.5 * U ** 2
    r["p_max_over_q"] = p_max / q
    r["valid"] = bool(abs(r.get("CL", 9e9)) < 3.0 and p_max < 5.0 * q)
    r.update(dict(tag=tag, d1=d1, d2=d2, U=U, alpha=alpha, iters=n_it,
                  converged="SIMPLE solution converged" in log, numerics=used,
                  wall_s=round(wall, 1), case=cdir), **mesh_quality(cdir))
    return r


if __name__ == "__main__":
    a = sys.argv[1:]
    d1, d2, U = (float(a[0]), float(a[1]), float(a[2])) if len(a) >= 3 else (0., 0., 55.)
    alphas = [float(x) for x in a[3:]] or [0., 3., 6., 9., 12.]
    tag = f"d{d1:+.0f}_{d2:+.0f}_U{U:.0f}".replace("+", "p").replace("-", "m")

    done = set()
    if os.path.exists(OUT):
        try:
            prev = pd.read_csv(OUT)
            done = {(r.tag, r.alpha) for r in prev.itertuples()}
        except Exception as e:
            print(f"warning: could not read {OUT} ({e}); continuing without resume",
                  flush=True)
    for al in alphas:
        if (tag, al) in done:
            print(f"skip {tag} a={al}"); continue
        try:
            r = run_one(tag, d1, d2, U, al)
        except Exception as e:
            r = dict(tag=tag, d1=d1, d2=d2, U=U, alpha=al, error=str(e)[:200])
        row = pd.DataFrame([r])
        if os.path.exists(OUT):
            cols = list(pd.read_csv(OUT, nrows=0).columns)
            for c in row.columns:
                if c not in cols:
                    cols.append(c)
            if cols != list(pd.read_csv(OUT, nrows=0).columns):
                prev = pd.read_csv(OUT).reindex(columns=cols)
                prev.to_csv(OUT, index=False)
            row.reindex(columns=cols).to_csv(OUT, mode="a", index=False, header=False)
        else:
            row.to_csv(OUT, index=False)
        print(f"{tag} a={al}: CL={r.get('CL')} dstar_s={r.get('dstar_s')} "
              f"H_s={r.get('H_s')} conv={r.get('converged')} {r.get('wall_s')}s",
              flush=True)
