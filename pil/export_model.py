#!/usr/bin/env python3
"""Emit the deployed model as C."""
from __future__ import annotations
import os, sys, argparse, textwrap
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

U_GRID = np.linspace(40.0, 70.0, 41)
CL_GRID = np.linspace(0.60, 1.60, 81)

A_LO, A_HI, D_LIM = 0.0, 12.0, 10.0

AG = np.linspace(A_LO, A_HI, 25)
DG = np.linspace(-D_LIM, D_LIM, 15)
BOX = np.array([[a, d1, d2] for a in AG for d1 in DG for d2 in DG])


def carray(name, values, per_line=8, fmt="%.9ef", ctype="float"):
    v = np.asarray(values, float).ravel()
    body = []
    for i in range(0, len(v), per_line):
        body.append("    " + ", ".join(fmt % x for x in v[i:i + per_line]) + ",")
    return (f"const {ctype} {name}[{len(v)}] = {{\n" + "\n".join(body) + "\n};\n")


def search(gcl, gspl, U, CLreq, tol=0.02, gcm=None, budget=0.05):
    X = np.column_stack([BOX, np.full(len(BOX), U)])
    cl = gcl.predict(X)
    spl = gspl.predict(X)
    lift = np.abs(cl - CLreq) < tol
    if not lift.any():
        lift = np.abs(cl - CLreq) < 3 * tol
    if not lift.any():
        return None, None
    if gcm is not None:
        cm = gcm.predict(X)
        idx0 = np.where(lift)[0]
        ref = cm[idx0[np.argmax(spl[idx0])]]
        ok = np.abs(cm - ref) <= budget
    else:
        ok = np.ones(len(BOX), bool)
    m = ok & (np.abs(cl - CLreq) < tol)
    if not m.any():
        m = ok & (np.abs(cl - CLreq) < 3 * tol)
    if not m.any():
        idx = np.where(ok)[0]
        best = BOX[idx[int(np.argmin(np.abs(cl[idx] - CLreq)))]]
        return best, float(spl[idx][int(np.argmin(np.abs(cl[idx] - CLreq)))])
    idx = np.where(m)[0]
    best = BOX[idx[np.argmin(spl[idx])]]
    return best, float(spl[idx].min())


def bilinear(table, U, CLreq):
    iu = np.clip(np.searchsorted(U_GRID, U) - 1, 0, len(U_GRID) - 2)
    ic = np.clip(np.searchsorted(CL_GRID, CLreq) - 1, 0, len(CL_GRID) - 2)
    tu = np.clip((U - U_GRID[iu]) / (U_GRID[iu + 1] - U_GRID[iu]), 0, 1)
    tc = np.clip((CLreq - CL_GRID[ic]) / (CL_GRID[ic + 1] - CL_GRID[ic]), 0, 1)
    iu = min(iu + 1, len(U_GRID) - 1) if tu > 0.5 else iu
    ic = min(ic + 1, len(CL_GRID) - 1) if tc > 0.5 else ic
    return table[iu, ic]


def build(dataset="dataset.csv", spl_col="SPL"):
    os.environ["AERO_DATASET"] = dataset
    os.environ["AERO_SPL_COL"] = spl_col
    from surrogate import AeroSurrogate            # noqa: E402
    S = AeroSurrogate.from_csv()
    gcl, gspl = S._cl, S._spl
    n, d = gcl.Xn.shape
    assert d == 4
    assert np.allclose(gcl.Xn, gspl.Xn), "the two models must share their inputs"

    table = np.zeros((len(U_GRID), len(CL_GRID), 3), float)
    gate = np.zeros((len(U_GRID), len(CL_GRID)), float)
    gcm = S._cm
    for i, U in enumerate(U_GRID):
        X = np.column_stack([BOX, np.full(len(BOX), U)])
        cl = gcl.predict(X); spl = gspl.predict(X)
        cm = gcm.predict(X) if gcm is not None else None
        for j, CLreq in enumerate(CL_GRID):
            lift = np.abs(cl - CLreq) < 0.02
            if not lift.any():
                lift = np.abs(cl - CLreq) < 0.06
            if not lift.any():
                table[i, j] = [0.5 * (A_LO + A_HI), 0.0, 0.0]
                gate[i, j] = 1.0
                continue
            if cm is not None:
                idx0 = np.where(lift)[0]
                ref = cm[idx0[np.argmax(spl[idx0])]]
                ok = np.abs(cm - ref) <= 0.05
            else:
                ok = np.ones(len(BOX), bool)
            m = ok & (np.abs(cl - CLreq) < 0.02)
            if not m.any():
                m = ok & (np.abs(cl - CLreq) < 0.06)
            if not m.any():
                idx = np.where(ok)[0]
                k = int(np.argmin(np.abs(cl[idx] - CLreq)))
                table[i, j] = BOX[idx[k]]
                gate[i, j] = float(gspl.std([[*BOX[idx[k]], U]])[0])
                continue
            idx = np.where(m)[0]
            best = BOX[idx[np.argmin(spl[idx])]]
            table[i, j] = best
            gate[i, j] = float(gspl.std([[best[0], best[1], best[2], U]])[0])

    for i, U in enumerate(U_GRID):
        for j, CLreq in enumerate(CL_GRID):
            e, _ = search(gcl, gspl, U, CLreq, gcm=gcm)
            if e is not None and not np.allclose(e, table[i, j]):
                raise SystemExit(
                    f"table and search() disagree at U={U}, CL={CLreq}: "
                    f"{table[i, j]} vs {e}. They have drifted apart again.")

    rng = np.random.RandomState(11)
    du, dspl = [], []
    for _ in range(120):
        U = rng.uniform(U_GRID[0], U_GRID[-1])
        CLreq = rng.uniform(CL_GRID[0], CL_GRID[-1])
        exact, spl_exact = search(gcl, gspl, U, CLreq, gcm=gcm)
        if exact is None:
            continue
        approx = bilinear(table, U, CLreq)
        du.append(np.linalg.norm(approx - exact))
        spl_approx = float(gspl.predict([[approx[0], approx[1], approx[2], U]])[0])
        dspl.append(spl_approx - spl_exact)
    du = np.asarray(du); dspl = np.asarray(dspl)

    ntri = n * (n + 1) // 2
    hdr = textwrap.dedent(f"""\
        /* gp_model.h -- generated by pil/export_model.py; do not edit.
         *
         * Retained training set : {n} points, {d} inputs, shared by both models
         * Set-point map         : {len(U_GRID)} x {len(CL_GRID)} operating points
         * Everything is const and single precision.
         */
        #ifndef GP_MODEL_H
        #define GP_MODEL_H

        #define GP_N        {n}
        #define GP_D        {d}
        #define GP_NTRI     {ntri}          /* n(n+1)/2, packed lower triangle */
        #define SP_NU       {len(U_GRID)}
        #define SP_NCL      {len(CL_GRID)}

        #define CTRL_A_LO   {A_LO:.6f}f
        #define CTRL_A_HI   {A_HI:.6f}f
        #define CTRL_D_LIM  {D_LIM:.6f}f

        /* shared standardisation and retained inputs */
        extern const float gp_in_mu[GP_D];
        extern const float gp_in_sd[GP_D];
        extern const float gp_xn[GP_N * GP_D];   /* normalised, row major */

        /* lift model: value and Jacobian, evaluated every inner step */
        extern const float gp_cl_inv_ls2[GP_D];  /* 1 / lengthscale^2      */
        extern const float gp_cl_alpha[GP_N];
        extern const float gp_cl_ym, gp_cl_ys;

        /* acoustic model: value, gradient and predictive spread */
        extern const float gp_spl_inv_ls2[GP_D];
        extern const float gp_spl_alpha[GP_N];
        extern const float gp_spl_ym, gp_spl_ys;
        extern const float gp_spl_chol[GP_NTRI]; /* lower triangle, row major */

        /* deployed set-point map */
        extern const float sp_u_grid[SP_NU];
        extern const float sp_cl_grid[SP_NCL];
        extern const float sp_table[SP_NU * SP_NCL * 3];  /* alpha, delta1, delta2 */
        extern const float sp_gate[SP_NU * SP_NCL];       /* predictive spread */

        #endif /* GP_MODEL_H */
        """)
    open(os.path.join(HERE, "gp_model.h"), "w").write(hdr)

    tril = np.concatenate([gspl.L[i, :i + 1] for i in range(n)])
    assert tril.size == ntri

    src = ['/* gp_model.c -- generated by pil/export_model.py; do not edit. */',
           '#include "gp_model.h"', '']
    src.append(carray("gp_in_mu", gcl.mu))
    src.append(carray("gp_in_sd", gcl.sd))
    src.append(carray("gp_xn", gcl.Xn))
    src.append(carray("gp_cl_inv_ls2", 1.0 / np.asarray(gcl.ls, float) ** 2))
    src.append(carray("gp_cl_alpha", gcl.a))
    src.append(f"const float gp_cl_ym = {gcl.ym:.9e}f;\n"
               f"const float gp_cl_ys = {gcl.ys:.9e}f;\n")
    src.append(carray("gp_spl_inv_ls2", 1.0 / np.asarray(gspl.ls, float) ** 2))
    src.append(carray("gp_spl_alpha", gspl.a))
    src.append(f"const float gp_spl_ym = {gspl.ym:.9e}f;\n"
               f"const float gp_spl_ys = {gspl.ys:.9e}f;\n")
    src.append(carray("gp_spl_chol", tril))
    src.append(carray("sp_u_grid", U_GRID))
    src.append(carray("sp_cl_grid", CL_GRID))
    src.append(carray("sp_table", table))
    src.append(carray("sp_gate", gate))
    open(os.path.join(HERE, "gp_model.c"), "w").write("\n".join(src))
    np.savez(os.path.join(HERE, "deployed_map.npz"),
             u_grid=U_GRID, cl_grid=CL_GRID, table=table, gate=gate)

    rng = np.random.RandomState(7)
    pts = np.column_stack([
        rng.uniform(A_LO, A_HI, 64),
        rng.uniform(-D_LIM, D_LIM, 64),
        rng.uniform(-D_LIM, D_LIM, 64),
        rng.uniform(40.0, 70.0, 64)])
    ref = []
    for p in pts:
        ref.append([gcl.predict([p])[0], *gcl.grad(p),
                    gspl.predict([p])[0], *gspl.grad(p),
                    gspl.std([p])[0]])
    ref = np.asarray(ref)
    np.savetxt(os.path.join(HERE, "reference_vectors.csv"),
               np.column_stack([pts, ref]), delimiter=",",
               header=("alpha,delta1,delta2,U,"
                       "CL,dCL_dalpha,dCL_dd1,dCL_dd2,dCL_dU,"
                       "SPL,dSPL_dalpha,dSPL_dd1,dSPL_dd2,dSPL_dU,sigma"),
               comments="")

    shared = 4 * (2 * d + n * d)
    cl_only = 4 * (d + n + 2)
    spl_only = 4 * (d + n + 2 + ntri)
    tab = 4 * (len(U_GRID) + len(CL_GRID) + table.size + gate.size)
    print(f"emitted gp_model.[ch] from {dataset}:{spl_col}")
    print(f"  retained training set : {n} points x {d} inputs")
    print(f"  shared inputs         : {shared/1024:8.1f} kB")
    print(f"  lift model            : {cl_only/1024:8.1f} kB")
    print(f"  acoustic model + factor:{spl_only/1024:8.1f} kB")
    print(f"  set-point map         : {tab/1024:8.1f} kB  "
          f"({len(U_GRID)}x{len(CL_GRID)} points)")
    print(f"  total deployed        : {(shared+cl_only+spl_only+tab)/1024:8.1f} kB")
    print(f"  set-point incidence spans {table[..., 0].min():.1f}-{table[..., 0].max():.1f} deg")
    print(f"  table vs exact search, off-node ({du.size} points):")
    print(f"    set-point offset  : mean {du.mean():.3f} deg, max {du.max():.3f} deg")
    print(f"    objective penalty : mean {dspl.mean():+.3f} dB, max {dspl.max():+.3f} dB")
    print("  wrote reference_vectors.csv (64 points, double precision)")

    import json
    json.dump(dict(n=int(n), d=int(d), nu=len(U_GRID), ncl=len(CL_GRID),
                   table_du_mean=float(du.mean()), table_du_max=float(du.max()),
                   table_dspl_mean=float(dspl.mean()),
                   table_dspl_max=float(dspl.max()),
                   bytes_shared=int(shared), bytes_cl=int(cl_only),
                   bytes_spl=int(spl_only), bytes_table=int(tab)),
              open(os.path.join(HERE, "export_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset.csv")
    ap.add_argument("--spl-col", default="SPL")
    a = ap.parse_args()
    build(a.dataset, a.spl_col)
