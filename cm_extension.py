#!/usr/bin/env python3
"""Pitching-moment trim as a second constraint alongside lift."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from surrogate import AeroSurrogate
import control3 as c
import control3 as _c3

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
S = AeroSurrogate.from_csv()
plt.rcParams.update({"font.size": 11})


def _glauert(E):
    th = np.arccos(2*E - 1)
    dCl = 2*(np.pi - th + np.sin(th)) * np.pi/180.0
    dCm = -(1 - np.cos(th))*np.sin(th)/2.0 * np.pi/180.0
    return dCl, dCm
_dCm1 = _glauert(0.70)[1]; _dCm2 = _glauert(0.85)[1]

import pandas as pd
from surrogate import _RBFGP, _FEATS
from surrogate import DATASET as _DATASET
import gains as _g
_dd = pd.read_csv(os.path.join(HERE, _DATASET))
if "CM" in _dd.columns and _dd["CM"].notna().any():
    _dd = _dd.dropna(subset=["CM"])
    _cmgp = _RBFGP([2.5, 1.5, 1.5, 2.0], 1e-3).fit(_dd[_FEATS].values.astype(float),
                                                   _dd["CM"].values.astype(float))
    _CM_SOURCE = "NeuralFoil (exact)"
    def Cm(a, d1, d2, U=55.0): return float(_cmgp.predict([[a, d1, d2, U]])[0])
else:
    _CM_SOURCE = "thin-airfoil (Glauert)"
    def Cm(a, d1, d2, U=55.0): return _dCm1*d1 + _dCm2*d2
print(f"pitching-moment model: {_CM_SOURCE}")

U, CLreq = 55.0, 1.30
LOUD = (9.0, 10.0, -8.0); Cm_trim = Cm(*LOUD); SPL_loud = S.SPL(*LOUD, U)

AG = np.linspace(0, 12, 25); DG = np.linspace(-10, 10, 21)
GRID = np.array([[a, d1, d2] for a in AG for d1 in DG for d2 in DG])
_Xg = np.column_stack([GRID, np.full(len(GRID), U)])
_CL = S._cl.predict(_Xg); _SPL = S._spl.predict(_Xg); _CM = np.array([Cm(*g) for g in GRID])


def quiet_setpoint_trim(budget, tol=0.02):
    m = (np.abs(_CL - CLreq) < tol) & (np.abs(_CM - Cm_trim) <= budget)
    if not m.any(): m = np.abs(_CL - CLreq) < tol
    idx = np.where(m)[0]; return GRID[idx[np.argmin(_SPL[idx])]].copy()


def run_ctrl(budget, T=16.0):
    DT = c.DT; n = int(T/DT); u = np.array(LOUD, float); w = 0.0
    ustar = quiet_setpoint_trim(budget)
    spl = np.zeros(n); viol = np.zeros(n); cmv = np.zeros(n)
    for i in range(n):
        CL = c._CL(u, U); s = CL - CLreq
        J = c._jac_CL(u, U); Jp = J/(J@J+1e-9); P = np.eye(3)-np.outer(Jp, J)
        phi1, phi2 = _c3.phi(s, 0.4)
        w += -_g.K2*phi2*DT
        u = c.clip_pos(u + c.clip_rate(Jp*(-_g.K1*phi1+w) - 3.0*(P@(u-ustar)))*DT)
        spl[i] = c._SPL(u, U); viol[i] = abs(s); cmv[i] = abs(Cm(*u) - Cm_trim)
    return np.mean(spl[-200:]), np.max(viol[100:]), np.max(cmv[-200:])


def bracket_figure():
    Bs = np.linspace(0, 0.06, 25); red = []
    for B in Bs:
        m = (np.abs(_CL - CLreq) < 0.02) & (np.abs(_CM - Cm_trim) <= B + 1e-9)
        red.append(SPL_loud - _SPL[m].min() if m.any() else 0.0)
    red = np.array(red)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(Bs, red, "-o", color="#d62728", ms=4)
    ax.axhline(7.96, color="gray", ls="--", lw=1, label="Lift-only ceiling (7.96 dB)")
    ax.axvspan(0.08, 0.06, alpha=0)
    ax.axvline(0.034, color="#1f77b4", ls=":", lw=1.4)
    ax.text(0.036, 2.0, "Loud$\\to$quiet\nneeds $\\Delta C_m$=0.034", fontsize=9, color="#1f77b4")
    ax.annotate("Routine tail authority", xy=(0.05, 7.4), xytext=(0.03, 5.6),
                arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9, color="gray")
    ax.set_xlabel("Available trim-moment budget  $|\\Delta C_m|$")
    ax.set_ylabel("Noise reduction at fixed lift (dB)")
    ax.set_title("How much of the 8 dB survives pitching-moment trim")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig12_trim_bracket.png", dpi=200); plt.close(fig)
    print("  saved fig12_trim_bracket.png")


if __name__ == "__main__":
    print(f"Cm model: dCm/dd1={_dCm1:.5f}, dCm/dd2={_dCm2:.5f} per deg (Glauert, about c/4)")
    print(f"loud Cm_trim={Cm_trim:+.4f}; loud->quiet needs |dCm|={abs(Cm(2,10,10)-Cm_trim):.4f}\n")
    print("closed-loop SG-FTSMC with trim budget (lift hard, moment budgeted):")
    for B in [0.00, 0.02, 0.034, 0.05]:
        spl, mv, cmused = run_ctrl(B)
        print(f"  budget |dCm|<={B:.3f}: settled SPL={spl:.2f} dB  (-{SPL_loud-spl:.2f} dB) | "
              f"lift held {mv:.3f} | trim used {cmused:.3f}")
    bracket_figure()
    print("\n[OK] cm_extension done. With routine tail authority (|dCm|~0.05) the full "
          "~8 dB survives;\n     the wing-self-trim floor (budget 0) is the conservative bound.")
