#!/usr/bin/env python3
"""Extract C_L, C_M and the trailing-edge boundary-layer state from a converged solution."""
from __future__ import annotations
import os, glob
import numpy as np
import pyvista as pv

NU = 1.5e-5


def _latest_vtk(case: str):
    dirs = sorted([d for d in glob.glob(os.path.join(case, "VTK", "*_*")) if os.path.isdir(d)],
                  key=lambda p: float(p.rsplit("_", 1)[1]))
    if not dirs:
        raise FileNotFoundError(f"no VTK export in {case}")
    d = dirs[-1]
    return (pv.read(os.path.join(d, "internal.vtu")),
            pv.read(os.path.join(d, "boundary", "airfoil.vtp")))


def coefficients(case: str, U: float, alpha_deg: float, chord: float = 1.0,
                 span: float = 0.05, x_ref: float = 0.25) -> dict:
    internal, patch = _latest_vtk(case)
    patch = patch.compute_normals(cell_normals=True, point_normals=False,
                                  auto_orient_normals=False)
    pc = patch.cell_centers().points
    area = patch.compute_cell_sizes(length=False, area=True,
                                    volume=False)["Area"]
    n = np.asarray(patch.cell_data["Normals"], float)
    centroid = pc.mean(0)
    flux = float(np.einsum("ij,ij->i", pc - centroid, n) @ area)
    if flux < 0:
        n = -n
    p = np.asarray(patch.cell_data["p"], float)

    Fp = -(p[:, None] * n) * area[:, None]

    probe = internal.find_closest_cell(pc + n * 0.0, return_closest_point=False)
    Uc = np.asarray(internal.cell_data["U"], float)[probe]
    nutc = np.asarray(internal.cell_data["nut"], float)[probe]
    cc = internal.cell_centers().points[probe]
    d1 = np.maximum(np.abs(np.einsum("ij,ij->i", cc - pc, n)), 1e-9)
    Ut = Uc - np.einsum("ij,ij->i", Uc, n)[:, None] * n
    mag = np.linalg.norm(Ut, axis=1) + 1e-30
    tau = (NU + nutc) * mag / d1
    Fv = (tau / mag)[:, None] * Ut * area[:, None]

    F = (Fp + Fv).sum(0)
    r = pc - np.array([x_ref, 0.0, 0.0])
    Mz = np.cross(r, Fp + Fv)[:, 2].sum()

    a = np.radians(alpha_deg)
    lift = -F[0] * np.sin(a) + F[1] * np.cos(a)
    drag = F[0] * np.cos(a) + F[1] * np.sin(a)
    q = 0.5 * U ** 2 * chord * span
    return dict(CL=float(lift / q), CD=float(drag / q),
                CM=float(Mz / (q * chord)),
                Fp_frac=float(np.linalg.norm(Fp.sum(0)) /
                              max(np.linalg.norm(F), 1e-30)))


def _surface_polyline(patch) -> np.ndarray:
    pts = patch.points
    z0 = float(pts[:, 2].min())
    tol = max(1e-9, 1e-6 * (pts[:, 2].max() - z0 + 1e-12))
    sel = pts[np.abs(pts[:, 2] - z0) < tol][:, :2]
    sel = np.unique(np.round(sel, 10), axis=0)
    c = sel.mean(0)
    ang = np.arctan2(sel[:, 1] - c[1], sel[:, 0] - c[0])
    return sel[np.argsort(ang)]


def bl_profile(case: str, U: float, side: str, x_station: float = 0.98,
               n_samples: int = 500, ray_scale: float = 0.06) -> dict:
    internal, patch = _latest_vtk(case)
    poly = _surface_polyline(patch)
    ile = np.argmin(poly[:, 0]); ite = np.argmax(poly[:, 0])
    LE, TE = poly[ile], poly[ite]
    t = TE - LE; t = t / np.linalg.norm(t)
    nrm = np.array([-t[1], t[0]])
    s = (poly - LE) @ nrm
    sign = 1.0 if side == "suction" else -1.0
    sall = (poly - LE) @ nrm
    xall = (poly - LE) @ t
    half = poly[(sall * sign) > 0]
    if len(half) < 3:
        half = poly
    xh = (half - LE) @ t
    pt = half[np.argmin(np.abs(xh - ((TE - LE) @ t) * x_station))]

    d = np.linalg.norm(poly - pt, axis=1)
    near = poly[np.argsort(d)[:7]]
    tt = np.linalg.svd(near - near.mean(0))[2][0]
    nn = np.array([-tt[1], tt[0]])
    if np.sign(nn @ nrm) != sign:
        nn = -nn

    z0 = float(patch.points[:, 2].min()) + 0.5 * (patch.points[:, 2].max() - patch.points[:, 2].min())
    L = ray_scale * 1.0
    a3 = np.array([pt[0], pt[1], z0])
    b3 = a3 + np.array([nn[0], nn[1], 0.0]) * L
    y1 = 1e-6
    frac = np.geomspace(y1 / L, 1.0, n_samples)
    pts = a3[None, :] + np.outer(frac * L, np.array([nn[0], nn[1], 0.0]))
    cloud = pv.PolyData(pts)
    samp = cloud.sample(internal)
    u = np.asarray(samp["U"], float)
    dist = frac * L
    valid = np.asarray(samp["vtkValidPointMask"], bool) if "vtkValidPointMask" in samp.array_names else np.ones(len(u), bool)
    ok = np.isfinite(u).all(1) & valid & (dist > 0)
    u, dist = u[ok], dist[ok]
    if len(dist) < 20:
        return dict(side=side, ok=False)

    uts = u @ np.array([tt[0], tt[1], 0.0])
    if np.median(uts[-max(10, len(uts) // 20):]) < 0:
        uts = -uts
    ut = np.abs(uts)
    reversed_frac = float(np.mean(uts[:max(1, len(uts) // 2)] < 0.0))

    pcen = patch.cell_centers().points
    iw = int(np.argmin(np.linalg.norm(pcen[:, :2] - pt, axis=1)))
    p_wall = float(np.asarray(patch.cell_data["p"], float)[iw])
    ue = float(np.sqrt(max(U ** 2 - 2.0 * p_wall, 1e-9)))

    above = np.where(ut >= 0.99 * ue)[0]
    if not len(above) or int(above[0]) < 5:
        return dict(side=side, ok=False,
                    reason=("no boundary-layer edge: the profile never reaches "
                            "0.99 u_e within the sampled ray, which is the "
                            "signature of a separated trailing edge"),
                    ue=float(ue), u_max=float(ut.max()),
                    reversed_fraction=reversed_frac,
                    x=float(pt[0]), y=float(pt[1]))
    edge = int(above[0])
    y = dist[:edge + 1]
    uu = np.clip(ut[:edge + 1] / ue, 0.0, 1.0)
    dstar = float(np.trapezoid(1.0 - uu, y))
    theta = float(np.trapezoid(uu * (1.0 - uu), y))
    return dict(side=side, ok=True, dstar=dstar, theta=theta,
                H=float(dstar / theta) if theta > 0 else np.nan,
                ue=float(ue), delta=float(y[-1]), reversed_fraction=reversed_frac,
                x=float(pt[0]), y=float(pt[1]))


def extract(case: str, U: float, alpha_deg: float, **kw) -> dict:
    out = coefficients(case, U, alpha_deg)
    for side in ("suction", "pressure"):
        r = bl_profile(case, U, side, **kw)
        tag = "s" if side == "suction" else "p"
        for k in ("dstar", "theta", "H", "ue", "delta"):
            out[f"{k}_{tag}"] = r.get(k, np.nan)
        out[f"bl_ok_{tag}"] = bool(r.get("ok", False))
        out[f"reversed_{tag}"] = float(r.get("reversed_fraction", np.nan))
        if not r.get("ok", False):
            out[f"bl_reason_{tag}"] = r.get("reason", "unknown")
    return out


if __name__ == "__main__":
    import sys, json
    c = sys.argv[1] if len(sys.argv) > 1 else "c000"
    U = float(sys.argv[2]) if len(sys.argv) > 2 else 55.0
    a = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
    print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                      for k, v in extract(c, U, a).items()}, indent=2))
