#!/usr/bin/env python3
"""Gmsh .geo writer for a 2-D OpenFOAM case, one cell in span."""
from __future__ import annotations
import os, subprocess
import numpy as np
from geometry import section, first_cell_height


def write_geo(sec: dict, path: str, U: float, R: float = 60.0,
              n_bl: int = 45, bl_growth: float = 1.14,
              le_size: float = 1.5e-3, te_size: float = 6e-4,
              mid_size: float = 5e-3, ff_size: float = 6.0,
              span: float = 0.05, yplus: float = 1.0) -> dict:
    c = np.asarray(sec["coords"], float)
    keep = np.r_[True, np.linalg.norm(np.diff(c, axis=0), axis=1) > 1e-9]
    c = c[keep]
    if np.allclose(c[0], c[-1]):
        c = c[:-1]
    n = len(c)

    y1 = first_cell_height(U, yplus=yplus)
    bl_t = y1 * (bl_growth ** n_bl - 1) / (bl_growth - 1)

    x = c[:, 0]
    size = np.where(x < 0.03, le_size, np.where(x > 0.93, te_size, mid_size))

    L = [f"// airfoil d1={sec['d1']} d2={sec['d2']}  U={U} m/s",
         f"// y1={y1:.4e} m (y+={yplus}), BL thickness={bl_t:.4e} m, {n_bl} layers",
         "lcFF = %g;" % ff_size, ""]
    for i, ((xi, yi), si) in enumerate(zip(c, size), start=1):
        L.append(f"Point({i}) = {{{xi:.10f}, {yi:.10f}, 0, {si:.6g}}};")
    for i in range(1, n):
        L.append(f"Line({i}) = {{{i}, {i+1}}};")
    L.append(f"Line({n}) = {{{n}, 1}};")
    L.append("Curve Loop(1) = {" + ",".join(str(i) for i in range(1, n + 1)) + "};")

    L += ["", f"Point(9001) = {{0.5, 0, 0, lcFF}};",
          f"Point(9002) = {{{0.5-R:.6f}, 0, 0, lcFF}};",
          f"Point(9003) = {{0.5, {R:.6f}, 0, lcFF}};",
          f"Point(9004) = {{{0.5+R:.6f}, 0, 0, lcFF}};",
          f"Point(9005) = {{0.5, {-R:.6f}, 0, lcFF}};",
          "Circle(9001) = {9002, 9001, 9003};",
          "Circle(9002) = {9003, 9001, 9004};",
          "Circle(9003) = {9004, 9001, 9005};",
          "Circle(9004) = {9005, 9001, 9002};",
          "Curve Loop(2) = {9001,9002,9003,9004};",
          "Plane Surface(1) = {2, 1};", ""]

    L += ["Field[1] = BoundaryLayer;",
          "Field[1].CurvesList = {" + ",".join(str(i) for i in range(1, n + 1)) + "};",
          "Field[1].PointsList = {1};",
          f"Field[1].Size = {y1:.6e};",
          f"Field[1].Ratio = {bl_growth};",
          f"Field[1].Thickness = {bl_t:.6e};",
          f"Field[1].SizeFar = {mid_size:.6e};",
          "Field[1].Quads = 1;",
          "BoundaryLayer Field = 1;", "",
          "Mesh.Algorithm = 5;",
          "Mesh.RecombineAll = 1;",
          "Mesh.RecombinationAlgorithm = 1;",
          "Mesh.ElementOrder = 1;", ""]

    L += [f"ex[] = Extrude {{0, 0, {span}}} {{ Surface{{1}}; Layers{{1}}; Recombine; }};",
          'Physical Volume("internal") = {ex[1]};',
          'Physical Surface("back") = {1};',
          'Physical Surface("front") = {ex[0]};',
          'Physical Surface("farfield") = {ex[2], ex[3], ex[4], ex[5]};',
          'Physical Surface("airfoil") = {' +
          ",".join(f"ex[{i}]" for i in range(6, 6 + n)) + "};", ""]

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    open(path, "w").write("\n".join(L))
    return dict(path=path, n_surface_points=n, y1=y1, bl_thickness=bl_t)


def run_gmsh(geo: str, msh: str, timeout: int = 900) -> dict:
    r = subprocess.run(["gmsh", "-3", "-format", "msh2", "-o", msh, geo],
                       capture_output=True, text=True, timeout=timeout)
    tail = (r.stdout + r.stderr).strip().splitlines()[-12:]
    return dict(ok=r.returncode == 0 and os.path.exists(msh),
                returncode=r.returncode, log="\n".join(tail))


if __name__ == "__main__":
    import sys
    d1, d2, U = (float(v) for v in (sys.argv[1:4] or [0, 0, 55]))
    sec = section(d1, d2, fillet_radius=0.0)
    info = write_geo(sec, "case.geo", U)
    print(info)
    print(run_gmsh("case.geo", "case.msh"))
