#!/usr/bin/env python3
"""The four constraint laws head to head on the dynamic plant: SG-FTSMC, BF-STA, CLF-QP and MPC."""
import os, numpy as np, pandas as pd
from surrogate import AeroSurrogate, _RBFGP, _FEATS
import plant as P
import control3 as C3
import gains as _g

HERE = os.path.dirname(os.path.abspath(__file__))
Sur = AeroSurrogate.from_csv()

def Cm(d, u=None):
    return Sur.CM(d[0], d[1], d[2], U if u is None else u)

U, CLreq = 55.0, 1.30
LOUD = None; Cm_trim = None
AG = np.linspace(0, 12, 25); DG = np.linspace(-10, 10, 15)
GRID = np.array([[a, d1, d2] for a in AG for d1 in DG for d2 in DG])
_XG = np.column_stack([GRID, np.full(len(GRID), U)])


_CM_GRID = Sur._cm.predict(_XG)


def set_operating_point(Uop, CLop):
    global U, CLreq, LOUD, Cm_trim, _XG, _CM_GRID
    U = float(Uop); CLreq = float(CLop)
    _XG = np.column_stack([GRID, np.full(len(GRID), U)])
    _CM_GRID = Sur._cm.predict(_XG)
    CLg = Sur._cl.predict(_XG); SPLg = Sur._spl.predict(_XG)
    feas = np.abs(CLg - CLreq) < 0.02
    if not feas.any(): feas = np.abs(CLg - CLreq) < 0.06
    LOUD = GRID[np.where(feas)[0][np.argmax(SPLg[feas])]].copy()
    Cm_trim = Cm(LOUD)
    return LOUD.copy()


set_operating_point(U, CLreq)


def loud_level(Uop=None):
    return float(P.tSPL.predict([[LOUD[0], LOUD[1], LOUD[2],
                                  U if Uop is None else Uop]])[0])


def setpoint(model, budget=0.05, tol=0.02):
    CL = model._cl.predict(_XG); SPL = model._spl.predict(_XG)
    ok = np.abs(_CM_GRID - Cm_trim) <= budget
    m = ok & (np.abs(CL-CLreq) < tol)
    if not m.any(): m = ok & (np.abs(CL-CLreq) < 3*tol)
    if not m.any():
        idx = np.where(ok)[0]
        return GRID[idx[int(np.argmin(np.abs(CL[idx]-CLreq)))]].copy()
    idx = np.where(m)[0]; return GRID[idx[np.argmin(SPL[idx])]].copy()


def jacCL(model, d): g = model._cl.grad([d[0], d[1], d[2], U]); return np.array([g[0], g[1], g[2]])


def run_sgftsmc(model, T=16.0, dt=0.005, gust=None, sensor=0.0, seed=0,
                k1=_g.K1, k2=_g.K2, mu=_g.MU, gamma=3.0, budget=0.05):
    rng = np.random.RandomState(seed)
    pl = P.Plant(LOUD.copy(), U, dt); n = int(T/dt); w = 0.0
    ustar = setpoint(model, budget)
    log = {k: np.zeros(n) for k in ("t", "spl", "viol")}
    for i in range(n):
        t = i*dt; d = pl.d
        g = gust(t) if gust else 0.0
        truelift = pl.CL() + g
        spl_now = pl.SPL()
        meas = truelift + (rng.normal(0, sensor) if sensor else 0.0)
        s = meas - CLreq
        J = jacCL(model, d); Jp = J/(J@J+1e-9); Pn = np.eye(3)-np.outer(Jp, J)
        phi1, phi2 = C3.phi(s, mu)
        w += -k2*phi2*dt
        ddes = Jp*(-k1*phi1+w) - gamma*(Pn@(d-ustar))
        u_cmd = d + ddes/P.SIGMA
        pl.step(u_cmd)
        log["t"][i] = t; log["spl"][i] = spl_now; log["viol"][i] = abs(truelift-CLreq)
    return log


def run_clfqp(model, T=16.0, dt=0.005, gust=None, sensor=0.0, seed=0,
              lam=8.0, gamma=3.0, budget=0.05):
    rng = np.random.RandomState(seed)
    pl = P.Plant(LOUD.copy(), U, dt); n = int(T/dt)
    ustar = setpoint(model, budget)
    log = {k: np.zeros(n) for k in ("t", "spl", "viol")}
    for i in range(n):
        t = i*dt; d = pl.d
        g = gust(t) if gust else 0.0
        truelift = pl.CL() + g
        spl_now = pl.SPL()
        meas = truelift + (rng.normal(0, sensor) if sensor else 0.0)
        s = meas - CLreq
        J = jacCL(model, d); Jp = J/(J@J+1e-9); Pn = np.eye(3)-np.outer(Jp, J)
        r_des = -gamma*(Pn@(d-ustar))
        a = s*J; b = -lam*s*s
        r = r_des.copy()
        if a@r > b:
            r = r_des - ((a@r_des - b)/(a@a+1e-12))*a
        r = np.clip(r, -P.RATE, P.RATE)
        u_cmd = d + r/P.SIGMA
        pl.step(u_cmd)
        log["t"][i] = t; log["spl"][i] = spl_now; log["viol"][i] = abs(truelift-CLreq)
    return log


def run_bfasta(model, T=16.0, dt=0.005, gust=None, sensor=0.0, seed=0,
               eps=0.03, gamma=3.0, budget=0.05, k_reach=6.0):
    rng = np.random.RandomState(seed)
    pl = P.Plant(LOUD.copy(), U, dt); n = int(T/dt); w = 0.0
    ustar = setpoint(model, budget)
    K0 = 0.5
    K = K0; reached = False; c1, c2 = 1.5, 1.1
    log = {k: np.zeros(n) for k in ("t", "spl", "viol", "K")}
    for i in range(n):
        t = i*dt; d = pl.d
        g = gust(t) if gust else 0.0
        truelift = pl.CL() + g
        spl_now = pl.SPL()
        meas = truelift + (rng.normal(0, sensor) if sensor else 0.0)
        s = meas - CLreq
        if not reached and abs(s) < 0.95*eps:
            reached = True
        elif reached and abs(s) >= eps:
            reached = False
        if reached:
            K = (eps*abs(s))/(eps - min(abs(s), 0.999*eps)) + K0
        else:
            K += k_reach*dt
        k1 = c1*np.sqrt(K); k2 = c2*K
        J = jacCL(model, d); Jp = J/(J@J+1e-9); Pn = np.eye(3)-np.outer(Jp, J)
        phi1 = np.sqrt(abs(s))*np.sign(s)
        w += -k2*np.sign(s)*dt
        ddes = Jp*(-k1*phi1 + w) - gamma*(Pn@(d-ustar))
        pl.step(d + ddes/P.SIGMA)
        log["t"][i] = t; log["spl"][i] = spl_now; log["viol"][i] = abs(truelift-CLreq); log["K"][i] = K
    return log


def run_mpc(model, T=16.0, dt=0.005, gust=None, sensor=0.0, seed=0,
            H=15, q=60.0, sref=2.0, rho=0.3, budget=0.05, iters=25, lr=0.15):
    rng = np.random.RandomState(seed)
    pl = P.Plant(LOUD.copy(), U, dt); n = int(T/dt)
    ustar = setpoint(model, budget)
    A = 1.0 - P.SIGMA*dt; B = P.SIGMA*dt
    Apow = A**np.arange(1, H+1)
    M = np.zeros((H, H))
    for k in range(H):
        for j in range(k+1):
            M[k, j] = A**(k-j)*B
    Useq = np.tile(pl.d, (H, 1))
    log = {k: np.zeros(n) for k in ("t", "spl", "viol")}
    for i in range(n):
        t = i*dt; d0 = pl.d.copy()
        g = gust(t) if gust else 0.0
        truelift = pl.CL() + g
        spl_now = pl.SPL()
        meas = truelift + (rng.normal(0, sensor) if sensor else 0.0)
        e0 = meas - CLreq; J = jacCL(model, d0)
        Pn = np.eye(3) - np.outer(J/(J@J+1e-9), J)
        Useq = np.clip(Useq, P.LO, P.HI)
        for _ in range(iters):
            delta = Apow[:, None]*d0[None, :] + M @ Useq
            lift_err = e0 + (delta - d0[None, :]) @ J
            track = (delta - ustar[None, :]) @ Pn.T
            gdelta = 2*q*lift_err[:, None]*J[None, :] + 2*sref*track
            gU = M.T @ gdelta
            gU[1:] += 2*rho*(Useq[1:]-Useq[:-1]); gU[:-1] += -2*rho*(Useq[1:]-Useq[:-1])
            Useq = np.clip(Useq - lr*gU, P.LO, P.HI)
        u_cmd = Useq[0].copy(); r = P.SIGMA*(u_cmd - d0)
        a = e0*J; b = -8.0*e0*e0
        if a@r > b: r = r - ((a@r - b)/(a@a+1e-12))*a
        u_cmd = d0 + r/P.SIGMA
        pl.step(u_cmd)
        Useq = np.vstack([Useq[1:], Useq[-1]])
        log["t"][i] = t; log["spl"][i] = spl_now; log["viol"][i] = abs(truelift-CLreq)
    return log


class BiasedModel:
    def __init__(self, base, b): self._b = b; self._cl = self._CL(base, b); self._spl = base._spl
    class _CL:
        def __init__(self, base, b): self._c = base._cl; self._b = b
        def predict(self, X): return self._c.predict(X)*(1+self._b)
        def grad(self, x): return self._c.grad(x)*(1+self._b)


def summ(name, L):
    ss = slice(int(0.6*len(L["t"])), None)
    spl = np.mean(L["spl"][ss]); sv = np.mean(L["viol"][ss])
    rms = np.sqrt(np.mean(L["viol"][ss]**2)); mx = np.max(L["viol"][int(0.2*len(L['t'])):])
    print(f"  {name:16s} settled SPL={spl:6.2f} | settled lift-err={sv:.4f} | RMS={rms:.4f} | max={mx:.4f}")
    return spl, sv, rms, mx


def _turb(seed):
    r = np.random.RandomState(seed); v = P.VonKarmanGust(r, 0.05)
    v.w = 2*np.pi*np.exp(np.random.RandomState(seed).uniform(np.log(0.2), np.log(1.5), 12))
    return v


CONTROLLERS = {
    "SG-FTSMC":  lambda mk, g, s: run_sgftsmc(mk, gust=g, sensor=0.003, seed=s),
    "BF-STA":    lambda mk, g, s: run_bfasta(mk, gust=g, sensor=0.003, seed=s, eps=0.12),
    "CLF-QP":    lambda mk, g, s: run_clfqp(mk, gust=g, sensor=0.003, seed=s, lam=40),
    "MPC":       lambda mk, g, s: run_mpc(mk, gust=g, sensor=0.003, seed=s, q=300, lr=0.03),
}


OPS = [(45.0, 1.10), (55.0, 1.30), (70.0, 1.50)]


def full_comparison(nseeds=5, ops=OPS):
    import warnings; warnings.filterwarnings("ignore")
    bmod = BiasedModel(Sur, 0.25)
    rows = []
    for (Uop, CLop) in ops:
        set_operating_point(Uop, CLop)
        loud_spl = loud_level(Uop)
        for cond, mk in [("nominal", Sur), ("model_err_25pct", bmod)]:
            for name, fn in CONTROLLERS.items():
                spl, rms, mx, whole = [], [], [], []
                for sd in range(nseeds):
                    L = fn(mk, _turb(sd), sd); ss = slice(int(0.6*len(L["t"])), None)
                    assert abs(L["spl"][0] - loud_spl) < 1e-3, (
                        f"loud reference {loud_spl:.4f} is not the level this run "
                        f"starts from ({L['spl'][0]:.4f})")
                    spl.append(np.mean(L["spl"][ss])); rms.append(np.sqrt(np.mean(L["viol"][ss]**2)))
                    mx.append(np.max(L["viol"][ss])); whole.append(np.max(L["viol"]))
                rows.append(dict(U=Uop, CLreq=CLop, condition=cond, controller=name,
                                 loud_SPL=loud_spl, SPL=np.mean(spl),
                                 reduction=loud_spl-np.mean(spl),
                                 lift_RMS=np.mean(rms), lift_RMS_std=np.std(rms),
                                 lift_max=np.max(mx), lift_max_mean=np.mean(mx),
                                 whole_max=np.max(whole)))
                print(f"  U={Uop:.0f} CL={CLop:.2f} {cond:16s} {name:9s} SPL={np.mean(spl):.2f} "
                      f"(-{loud_spl-np.mean(spl):.2f} dB) | lift RMS={np.mean(rms):.4f} "
                      f"settled max={np.max(mx):.4f}")
    df = pd.DataFrame(rows); df.to_csv(os.path.join(HERE, "controller_comparison.csv"), index=False)
    _plot_multi(df, ops)
    return df


def _plot_multi(df, ops):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 11})
        names = list(CONTROLLERS); nom = df[df.condition == "nominal"]
        x = np.arange(len(ops)); wdt = 0.2
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
        for k, nm in enumerate(names):
            red = [nom[(nom.U == U) & (nom.controller == nm)].reduction.values[0] for U, _ in ops]
            rms = [nom[(nom.U == U) & (nom.controller == nm)].lift_RMS.values[0] for U, _ in ops]
            ax[0].bar(x+(k-1.5)*wdt, red, wdt, label=nm)
            ax[1].bar(x+(k-1.5)*wdt, rms, wdt, label=nm)
        labs = [f"U={U:.0f}\n$C_L$={C:.2f}" for U, C in ops]
        ax[0].set_ylabel("Noise reduction (dB)"); ax[0].set_title("Reduction is controller-agnostic at every operating point")
        ax[1].set_ylabel("Lift error RMS"); ax[1].set_title("Lift held within safe band at every operating point")
        import band as _bd
        ax[1].axhline(_bd.EPS, color="k", ls=":", lw=1, label="safety band")
        for a in ax:
            a.set_xticks(x); a.set_xticklabels(labs); a.legend(fontsize=8, ncol=2); a.grid(alpha=0.3, axis="y")
        fig.suptitle("Controller-agnostic across the flight envelope (dynamic plant, actuator + circulation lag)")
        fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "fig14_controller_agnostic.png"), dpi=200)
        plt.close(fig); print("  saved figs/fig14_controller_agnostic.png + controller_comparison.csv")
    except Exception as e:
        print("  (figure skipped:", e, ")")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=5)
    n = ap.parse_args().n
    print(f"Four-controller comparison, 3 operating points, dynamic plant (nseeds={n}):")
    full_comparison(n)
