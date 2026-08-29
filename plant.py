#!/usr/bin/env python3
"""Dynamic wing-section plant with first-order actuator dynamics."""
import os, numpy as np, pandas as pd
from surrogate import _RBFGP, _FEATS, AeroSurrogate, DATASET, SPL_COL

HERE = os.path.dirname(os.path.abspath(__file__))
SIGMA = 1.0/0.0495
CHORD = 1.0
AERO_TAU_COEF = 2.0
RATE = np.array([10.0, 60.0, 60.0])
LO = np.array([0.0, -10.0, -10.0]); HI = np.array([12.0, 10.0, 10.0])

_d = pd.read_csv(os.path.join(HERE, DATASET)).dropna()
_X = _d[_FEATS].values.astype(float)
tCL = _RBFGP([2.5, 1.5, 1.5, 2.0], 1e-3).fit(_X, _d["CL"].values.astype(float))

_tDS = _RBFGP([2.5, 1.5, 1.5, 2.0], 1e-3).fit(
    _X, np.log(_d["dstar_s"].values.astype(float)))
_tDP = _RBFGP([2.5, 1.5, 1.5, 2.0], 1e-3).fit(
    _X, np.log(_d["dstar_p"].values.astype(float)))


class _PhysicalSPL:
    def __init__(self):
        import bpm_noise_v2 as _v2
        self._v2 = _v2

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        ds = np.exp(_tDS.predict(X))
        dp = np.exp(_tDP.predict(X))
        return np.array([self._v2.overall_spl(a, b, x[3], x[0], CHORD)
                         for a, b, x in zip(ds, dp, X)])


tSPL = _PhysicalSPL()


def sat_pos(d): return np.minimum(np.maximum(d, LO), HI)
def sat_rate(dd): return np.clip(dd, -RATE, RATE)


class VonKarmanGust:
    def __init__(self, rng, amp=0.06, n=12, f_lo=0.2, f_hi=4.0):
        self.w = 2*np.pi*np.exp(rng.uniform(np.log(f_lo), np.log(f_hi), n))
        self.ph = rng.uniform(0, 2*np.pi, n); self.A = amp/np.sqrt(n)
    def __call__(self, t): return self.A*np.sum(np.sin(self.w*t + self.ph))


class Plant:
    def __init__(self, delta0, U=55.0, dt=0.005):
        self.d = np.array(delta0, float)
        self.da = np.array(delta0, float)
        self.U = U; self.dt = dt
        self.tau = AERO_TAU_COEF*CHORD/U
    def step(self, u_cmd):
        u_cmd = sat_pos(np.asarray(u_cmd, float))
        dd = sat_rate(SIGMA*(u_cmd - self.d))
        self.d = sat_pos(self.d + dd*self.dt)
        self.da = self.da + (self.dt/self.tau)*(self.d - self.da)
        return self.d.copy()
    def CL(self, d=None):
        d = self.da if d is None else d
        return float(tCL.predict([[d[0], d[1], d[2], self.U]])[0])
    def SPL(self, d=None):
        d = self.da if d is None else d
        return float(tSPL.predict([[d[0], d[1], d[2], self.U]])[0])
