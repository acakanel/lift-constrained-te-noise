#!/usr/bin/env python3
"""Does the confidence gate earn its keep?"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from surrogate import _RBFGP, _FEATS
import control3 as c
import gains as _g

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "dataset.csv")).dropna().reset_index(drop=True)
X = d[_FEATS].values.astype(float); yCL = d["CL"].values.astype(float); ySPL = d["SPL"].values.astype(float)

tCL = _RBFGP([2.5, 1.5, 1.5, 2.0], 1e-3).fit(X, yCL)
tSPL = _RBFGP([2.5, 1.5, 1.5, 1.5], 1e-3).fit(X, ySPL)
mask = (d["U"] < 70).values
rCL = _RBFGP([2.5, 1.5, 1.5, 2.0], 1e-3).fit(X[mask], yCL[mask])
rSPL = _RBFGP([2.5, 1.5, 1.5, 1.5], 1e-3).fit(X[mask], ySPL[mask])

_AG = np.linspace(0, 12, 25); _DG = np.linspace(-10, 10, 15)
_GRID = np.array([[a, d1, d2] for a in _AG for d1 in _DG for d2 in _DG])
def setpoint(U, CLreq, tol=0.03):
    Xg = np.column_stack([_GRID, np.full(len(_GRID), U)])
    CL = rCL.predict(Xg); SPL = rSPL.predict(Xg)
    m = np.abs(CL - CLreq) < tol
    if not np.any(m): m = np.abs(CL - CLreq) < 3*tol
    idx = np.where(m)[0]; return _GRID[idx[np.argmin(SPL[idx])]].copy()

DT = 0.01; RATE = c.RATE
def run(U, CLreq, conf_c, T=12.0, u0=(9., 10., -8.), gamma=3.0,
        k1=_g.K1, k2=_g.K2, mu=_g.MU):
    u = np.array(u0, float); w = 0.0; ustar = setpoint(U, CLreq)
    n = int(T/DT); trueviol = np.zeros(n); truespl = np.zeros(n)
    for i in range(n):
        CLtrue = float(tCL.predict([[u[0], u[1], u[2], U]])[0])
        s = CLtrue - CLreq
        J = np.array(rCL.grad([u[0], u[1], u[2], U])[:3])
        Jp = J/(J@J + 1e-9); P = np.eye(3) - np.outer(Jp, J)
        std = float(rSPL.std([[u[0], u[1], u[2], U]])[0])
        ge = gamma/(1.0 + conf_c*std)
        phi1 = np.sqrt(abs(s))*np.sign(s) + mu*abs(s)**1.5*np.sign(s)
        phi2 = 0.5*np.sign(s) + 1.5*mu*abs(s)*np.sign(s) + 1.5*mu*mu*abs(s)**2*np.sign(s)
        w += -k2*phi2*DT
        u = c.clip_pos(u + c.clip_rate(Jp*(-k1*phi1 + w) - ge*(P@(u - ustar)))*DT)
        trueviol[i] = abs(s); truespl[i] = float(tSPL.predict([[u[0], u[1], u[2], U]])[0])
    settle = slice(int(0.6*n), None)
    return (np.max(trueviol), np.mean(truespl[settle]),
            float(np.max(trueviol[settle])),
            float(np.sqrt(np.mean(trueviol[settle] ** 2))))


if __name__ == "__main__":
    print("TRUE-plant metrics (controller uses a surrogate blind to U=70):\n")
    print(f"{'condition':26s}| gate | whole-run max | settled max | settled RMS "
          f"| settled TRUE SPL")
    res = {}
    for U, tag in [(55, "in-domain U=55"), (70, "EXTRAPOLATION U=70")]:
        for conf_c, gate in [(0.0, "OFF"), (3.0, "ON ")]:
            mv, spl, smax, srms = run(U, 1.30, conf_c)
            res[(U, gate.strip())] = (mv, spl, smax, srms)
            flag = "  <-- unsafe" if mv > 0.15 else ""
            print(f"{tag:26s}|  {gate} | {mv:13.4f} | {smax:11.5f} | "
                  f"{srms:11.5f} | {spl:12.2f} dB{flag}")
    print()
    for U, tag in [(55, "In-domain"), (70, "In extrapolation")]:
        d = abs(res[(U, "OFF")][3] - res[(U, "ON")][3])
        print(f"  {tag}: gating changes the SETTLED lift-error RMS by "
              f"{d:.2e}")

    print()
    for U, tag in [(55, "In-domain"), (70, "In extrapolation")]:
        off = res[(U, "OFF")][0]
        on = res[(U, "ON")][0]
        if off <= 0:
            continue
        change = (off - on) / off
        if abs(change) < 0.05:
            verdict = "the gate makes no measurable difference"
        elif change > 0:
            verdict = f"the gate cuts the true lift excursion by {100*change:.0f} %"
        else:
            verdict = f"the gate widens the true lift excursion by {-100*change:.0f} %"
        print(f"  {tag} (U={U}): {off:.3f} -> {on:.3f}; {verdict}.")
    print()
    print("  The lift constraint is held by feedback on the measurement, not by")
    print("  the gate, so the gate is not what keeps the excursion bounded in")
    print("  either case. Its role is as an inside-envelope monitor.")

    import csv as _csv, os as _os
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                       "confidence_gate.csv")
    with open(_p, "w", newline="") as _f:
        _w = _csv.writer(_f)
        _w.writerow(["U", "gate", "whole_run_max", "settled_max", "settled_rms",
                     "settled_SPL_dB"])
        for (U, gate), v in sorted(res.items()):
            _w.writerow([U, gate, f"{v[0]:.6f}", f"{v[2]:.6f}", f"{v[3]:.8f}",
                         f"{v[1]:.4f}"])
    print(f"wrote {_os.path.basename(_p)}")
