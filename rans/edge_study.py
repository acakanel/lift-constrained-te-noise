#!/usr/bin/env python3
"""Choose a boundary-layer edge criterion that terminates."""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pyvista as pv
import post

CASES = [
    ("d+0_+0_a6.27_L0",  "neutral, moderate lift"),
    ("d+0_+0_a11.83_L0", "neutral, high lift (headline reference)"),
    ("d+5_+0_a2.89_L0",  "+5, moderate lift"),
    ("d+5_+0_a8.58_L0",  "+5, high lift (headline)"),
    ("d+5_+5_a6.42_L0",  "+5+5, high lift"),
    ("d+10_+0_a5.49_L0", "+10, high lift"),
    ("d+10_+10_a2.08_L0",   "+10+10, high lift  [ray-bound]"),
    ("d+10_+10_a-3.63_L0",  "+10+10, moderate   [ray-bound]"),
    ("sym_L0",           "symmetric, zero incidence"),
]


def raw_profile(case, U=55.0, side="suction", x_station=0.98,
                n_samples=1200, ray_scale=0.25):
    internal, patch = post._latest_vtk(case)
    poly = post._surface_polyline(patch)
    ile, ite = np.argmin(poly[:, 0]), np.argmax(poly[:, 0])
    LE, TE = poly[ile], poly[ite]
    t = TE - LE; t = t / np.linalg.norm(t)
    nrm = np.array([-t[1], t[0]])
    sign = 1.0 if side == "suction" else -1.0
    sall = (poly - LE) @ nrm
    half = poly[(sall * sign) > 0]
    xh = (half - LE) @ t
    pt = half[np.argmin(np.abs(xh - ((TE - LE) @ t) * x_station))]

    d = np.linalg.norm(poly - pt, axis=1)
    near = poly[np.argsort(d)[:7]]
    tt = np.linalg.svd(near - near.mean(0))[2][0]
    nn = np.array([-tt[1], tt[0]])
    if np.sign(nn @ nrm) != sign:
        nn = -nn

    z0 = float(patch.points[:, 2].min()) + 0.5 * (
        patch.points[:, 2].max() - patch.points[:, 2].min())
    L = ray_scale
    frac = np.geomspace(1e-6 / L, 1.0, n_samples)
    pts = np.array([pt[0], pt[1], z0])[None, :] + np.outer(
        frac * L, np.array([nn[0], nn[1], 0.0]))
    samp = pv.PolyData(pts).sample(internal)
    u = np.asarray(samp["U"], float)
    pk = np.asarray(samp["p"], float)
    y = frac * L
    valid = np.asarray(samp["vtkValidPointMask"], bool)
    ok = np.isfinite(u).all(1) & np.isfinite(pk) & valid & (y > 0)
    u, y, pk = u[ok], y[ok], pk[ok]
    ut = u @ np.array([tt[0], tt[1], 0.0])
    if np.median(ut[-50:]) < 0:
        ut = -ut
    pcen = patch.cell_centers().points
    iw = int(np.argmin(np.linalg.norm(pcen[:, :2] - pt, axis=1)))
    p_wall = float(np.asarray(patch.cell_data["p"], float)[iw])
    ue_bern = float(np.sqrt(max(U ** 2 - 2.0 * p_wall, 1e-9)))
    p0 = pk + 0.5 * np.sum(u ** 2, axis=1)
    return y, ut, ue_bern, p0


def integrate(y, ut, ue, edge):
    yy = y[:edge + 1]
    uu = np.clip(ut[:edge + 1] / ue, -2.0, 1.0)
    ds = float(np.trapezoid(1.0 - uu, yy))
    th = float(np.trapezoid(uu * (1.0 - uu), yy))
    return ds, th, (ds / th if th > 0 else np.nan)


def edge_bernoulli(y, ut, ue_bern):
    above = np.where(np.abs(ut) >= 0.99 * ue_bern)[0]
    if not len(above):
        return len(ut) - 1, ue_bern, False
    return int(above[0]), ue_bern, True


def edge_shear(y, ut, ue_bern, tau=0.01):
    s = np.abs(np.gradient(ut, y))
    smax = s.max()
    hot = np.where(s > tau * smax)[0]
    e = int(hot[-1]) if len(hot) else len(ut) - 1
    e = min(e, len(ut) - 1)
    return e, float(ut[e]), True


def edge_max99(y, ut, ue_bern):
    um = np.abs(ut).max()
    above = np.where(np.abs(ut) >= 0.99 * um)[0]
    e = int(above[0]) if len(above) else len(ut) - 1
    return e, float(np.abs(ut[e])), True


def edge_totalp(y, ut, ue_bern, p0, tol=0.005, U=55.0):
    q = 0.5 * U ** 2
    p0inf = float(np.median(p0[-40:]))
    deficit = (p0inf - p0) / q
    inside = np.where(deficit > tol)[0]
    e = int(inside[-1]) if len(inside) else 0
    e = min(max(e, 5), len(ut) - 1)
    return e, float(abs(ut[e])), True


def main():
    print(f"{'case':24s} {'method':12s} {'edge mm':>8s} {'u_e':>7s} "
          f"{'d* mm':>8s} {'H':>6s}  note")
    for case, lab in CASES:
        path = os.path.join(HERE, "resweep", case)
        if not os.path.isdir(path):
            path = os.path.join(HERE, "conv", case)
        if not os.path.isdir(path):
            print(f"{case:24s} -- not found")
            continue
        y, ut, ueb, p0 = raw_profile(path)
        rev = float(ut.min())
        print(f"\n{case:24s} {lab}")
        print(f"{'':24s} u_e(Bernoulli)={ueb:.2f}  "
              f"max|u_t|={np.abs(ut).max():.2f}  min u_t={rev:+.2f}"
              f"{'   <- REVERSED FLOW' if rev < -1 else ''}")
        for name, fn in (("bernoulli", edge_bernoulli), ("shear", edge_shear),
                         ("max99", edge_max99), ("totalp", edge_totalp)):
            e, ue, ok = fn(y, ut, ueb, p0) if name == "totalp" else fn(y, ut, ueb)
            ds, th, H = integrate(y, ut, ue, e)
            note = "" if ok else "FELL BACK TO RAY END"
            if e >= len(y) - 2:
                note = (note + " at ray end").strip()
            print(f"{'':24s} {name:12s} {1e3*y[e]:8.3f} {ue:7.2f} "
                  f"{1e3*ds:8.4f} {H:6.3f}  {note}")


if __name__ == "__main__":
    main()
