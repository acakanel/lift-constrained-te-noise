#!/usr/bin/env python3
"""Double-precision reference for the deployed control law."""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _root():
    tried = [HERE, os.path.dirname(HERE)]
    for d in tried:
        if os.path.exists(os.path.join(d, "surrogate.py")):
            return d
    raise SystemExit(
        "cannot find surrogate.py in " + " or ".join(tried) +
        "\n  The board pack must be self-contained: it needs surrogate.py, "
        "plant.py,\n  control3.py, compare.py, bpm_noise_v2.py and dataset.csv "
        "beside it.")


ROOT = _root()
sys.path.insert(0, ROOT)
os.environ.setdefault("AERO_DATASET", "dataset.csv")
os.environ.setdefault("AERO_SPL_COL", "SPL")

from surrogate import AeroSurrogate                       # noqa: E402

def _from_header(name, default):
    import os, re
    h = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ctrl.h")
    m = re.search(r"#define\s+%s\s+([0-9.eE+-]+)f?" % name, open(h).read())
    return float(m.group(1)) if m else default

DT = _from_header("CTRL_DT", 0.01)
K1 = _from_header("CTRL_K1", 10.0)
K2 = _from_header("CTRL_K2", 40.0)
MU = _from_header("CTRL_MU", 0.4)
GAMMA = _from_header("CTRL_GAMMA", 3.0)
CONF_C = _from_header("CTRL_CONF_C", 3.0)
SP_DT = _from_header("CTRL_SETPOINT_DT", 0.2)
SP_EVERY = int(round(SP_DT / DT))
LO = np.array([0.0, -10.0, -10.0])
HI = np.array([12.0, 10.0, 10.0])
RATE = np.array([10.0, 60.0, 60.0])

F_POS_SAT, F_RATE_SAT, F_SP_CLAMP, F_LOW_CONF, F_NONFINITE, F_ILLCOND = (
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20)
USE_MEAS_CL, USE_MEAS_D = 1, 2

_S = AeroSurrogate.from_csv()
_M = np.load(os.path.join(HERE, "deployed_map.npz"))
UG, CG, TAB, GATE = _M["u_grid"], _M["cl_grid"], _M["table"], _M["gate"]


def sp_lookup(U, CLreq):
    oor = (U < UG[0] or U > UG[-1] or CLreq < CG[0] or CLreq > CG[-1])
    iu = int(np.clip(np.searchsorted(UG, U) - 1, 0, len(UG) - 2))
    ic = int(np.clip(np.searchsorted(CG, CLreq) - 1, 0, len(CG) - 2))
    tu = float(np.clip((U - UG[iu]) / (UG[iu + 1] - UG[iu]), 0, 1))
    tc = float(np.clip((CLreq - CG[ic]) / (CG[ic + 1] - CG[ic]), 0, 1))
    w = [(1 - tu) * (1 - tc), (1 - tu) * tc, tu * (1 - tc), tu * tc]
    us = (w[0] * TAB[iu, ic] + w[1] * TAB[iu, ic + 1]
          + w[2] * TAB[iu + 1, ic] + w[3] * TAB[iu + 1, ic + 1])
    return np.clip(us, LO, HI), oor


class Local:
    def init(self, u0):
        self.u = np.clip(np.array(u0, float), LO, HI)
        self.ustar = self.u.copy()
        self.w = 0.0
        self.gamma_eff = GAMMA
        self.k = 0
        return self.u.copy()

    def step(self, U, CLreq, CLmeas, d, Up, CLp, preview, mode):
        flags = 0
        if not np.isfinite(U) or not np.isfinite(CLreq):
            flags |= F_NONFINITE
            U = np.clip(U if np.isfinite(U) else 55.0, 20.0, 120.0)
            CLreq = CLreq if np.isfinite(CLreq) else 0.0
        d = np.array(d, float) if (mode & USE_MEAS_D) else self.u.copy()
        bad = ~np.isfinite(d)
        if bad.any():
            d[bad] = self.u[bad]; flags |= F_NONFINITE
        d = np.clip(d, LO, HI)

        if self.k % SP_EVERY == 0:
            self.ustar, oor = sp_lookup(Up, CLp)
            if oor:
                flags |= F_SP_CLAMP
            sigma = _S.SPL_std(d[0], d[1], d[2], U)
            if not np.isfinite(sigma) or sigma < 0:
                sigma = 1.0; flags |= F_NONFINITE
            self.gamma_eff = GAMMA / (1.0 + CONF_C * sigma)
            if self.gamma_eff < 0.5 * GAMMA:
                flags |= F_LOW_CONF

        CL = _S.CL(d[0], d[1], d[2], U)
        g = _S._cl.grad([d[0], d[1], d[2], U])
        J = np.array([g[0], g[1], g[2]])
        if mode & USE_MEAS_CL:
            if np.isfinite(CLmeas):
                CL = CLmeas
            else:
                flags |= F_NONFINITE
        s = CL - CLreq

        JJ = J @ J + 1e-9
        if JJ < 1e-6:
            flags |= F_ILLCOND
        Jp = J / JJ
        a_s = abs(s); sw = np.sign(s)
        phi1 = (np.sqrt(a_s) + MU * a_s ** 1.5) * sw
        phi2 = (0.5 + 2.0 * MU * a_s + 1.5 * MU * MU * a_s ** 2) * sw
        ff = (CLp - CLreq) / max(preview, 1e-6)
        if not np.isfinite(ff):
            ff = 0.0; flags |= F_NONFINITE
        rs = -K1 * phi1 + self.w + ff
        self.w += -K2 * phi2 * DT

        e = d - self.ustar
        ud = Jp * rs - self.gamma_eff * (e - Jp * (J @ e))
        bad = ~np.isfinite(ud)
        if bad.any():
            ud[bad] = 0.0; flags |= F_NONFINITE
        if np.any(np.abs(ud) > RATE):
            flags |= F_RATE_SAT
        ud = np.clip(ud, -RATE, RATE)
        un = d + ud * DT
        if np.any(un < LO) or np.any(un > HI):
            flags |= F_POS_SAT
        self.u = np.clip(un, LO, HI)
        self.k += 1
        return dict(rate=ud, u=self.u.copy(), CL=CL, s=s,
                    sigma=(GAMMA / self.gamma_eff - 1.0) / CONF_C, flags=flags)


def sched(sc, t):
    if sc == 0:
        return 55.0, 1.30
    r = min(t, 3.0) / 3.0
    return 70.0 - 25.0 * r, 1.00 + 0.45 * r


SCEN = [dict(T=20.0, u0=[9.0, 10.0, -8.0], preview=0.4),
        dict(T=6.0, u0=[5.0, 0.0, 0.0], preview=0.8)]


def kinematic(ctl):
    rows = []
    for sc, cfg in enumerate(SCEN):
        ctl.init(cfg["u0"])
        n = int(cfg["T"] / DT + 0.5)
        for i in range(n):
            t = i * DT
            U, CL = sched(sc, t)
            Up, CLp = sched(sc, t + cfg["preview"])
            r = ctl.step(U, CL, 0.0, (0.0, 0.0, 0.0), Up, CLp, cfg["preview"], 0)
            rows.append([sc, i, t, *r["u"], r["CL"], r["s"], r["sigma"], r["flags"]])
    return np.asarray(rows)


U_OP, CL_OP = 55.0, 1.30


def loud_start():
    import compare as C
    C.set_operating_point(U_OP, CL_OP)
    return C.LOUD.copy(), C.loud_level(U_OP)


def dynamic(ctl, seed=0, T=16.0, gust_amp=0.05, sensor=0.004, rate_scale=1.0,
            lift_bias=0.0, d0=None):
    import plant as P
    rng = np.random.RandomState(seed)
    if d0 is None:
        d0 = loud_start()[0]
    d0 = np.array(d0, float)
    rate0 = P.RATE.copy()
    P.RATE = rate0 * rate_scale
    try:
        pl = P.Plant(d0.copy(), U_OP, DT)
        g = P.VonKarmanGust(rng, amp=gust_amp, f_lo=0.2, f_hi=1.5)
        ctl.init(d0)
        rows = []
        for i in range(int(T / DT + 0.5)):
            t = i * DT
            lift = pl.CL() * (1.0 + lift_bias) + g(t)
            meas = lift + rng.normal(0, sensor)
            r = ctl.step(U_OP, CL_OP, meas, pl.d, U_OP, CL_OP, 0.4,
                         USE_MEAS_CL | USE_MEAS_D)
            pl.step(pl.d + r["rate"] / P.SIGMA)
            rows.append([i, t, *pl.d, lift, abs(lift - CL_OP), pl.SPL(),
                         r["flags"]])
    finally:
        P.RATE = rate0
    return np.asarray(rows)


def draw(rng):
    return dict(sensor=float(rng.uniform(0.002, 0.006)),
                rate_scale=float(rng.uniform(0.7, 1.3)),
                gust_amp=float(rng.uniform(0.03, 0.07)))


if __name__ == "__main__":
    rows = kinematic(Local())
    np.savetxt(os.path.join(HERE, "ref_loop.csv"), rows, delimiter=",",
               header="scenario,i,t,alpha,delta1,delta2,CL,s,sigma,flags",
               comments="")
    for sc in (0, 1):
        r = rows[rows[:, 0] == sc]
        print(f"scenario {sc}: settled at alpha={r[-1,3]:.3f} "
              f"d1={r[-1,4]:.3f} d2={r[-1,5]:.3f}, |s|={abs(r[-1,7]):.2e}")
    d0, loud = loud_start()
    d = dynamic(Local(), d0=d0)
    np.savetxt(os.path.join(HERE, "ref_closedloop.csv"), d, delimiter=",",
               header="i,t,alpha,delta1,delta2,CL,viol,SPL,flags", comments="")
    ss = d[int(0.6 * len(d)):]
    print(f"dynamic plant: loud start {np.round(d0,2)} at {loud:.2f} dB, "
          f"settled {ss[:,7].mean():.2f} dB "
          f"(descent {loud - ss[:,7].mean():.2f} dB), "
          f"settled lift error {ss[:,6].max():.4f}")
    print("wrote ref_loop.csv, ref_closedloop.csv")
