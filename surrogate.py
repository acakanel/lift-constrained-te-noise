#!/usr/bin/env python3
"""Differentiable Gaussian-process surrogate for (alpha, delta1, delta2, U) -> (C_L, C_M, SPL)."""
import os, numpy as np, pandas as pd

_FEATS = ["alpha", "delta1", "delta2", "U"]

DATASET = os.environ.get("AERO_DATASET", "dataset.csv")
SPL_COL = os.environ.get("AERO_SPL_COL", "SPL")

if "corrected" in os.path.basename(DATASET) and SPL_COL == "SPL":
    raise SystemExit(
        f"{DATASET} carries both SPL and SPL_c; AERO_SPL_COL is still 'SPL', "
        f"which would train on the uncorrected target from the corrected file. "
        f"Set AERO_SPL_COL=SPL_c, or use dataset.csv.")


class _RBFGP:
    def __init__(self, lengthscales, noise=1e-3, sig=1.0):
        self.ls = np.asarray(lengthscales, float)
        self.noise = noise
        self.sig = sig

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-12
        self.Xn = (X - self.mu) / self.sd
        self.ym = y.mean(); self.ys = y.std() + 1e-12
        yn = (y - self.ym) / self.ys
        K = self._k(self.Xn, self.Xn) + self.noise * np.eye(len(X))
        self.L = np.linalg.cholesky(K)
        self.a = np.linalg.solve(self.L.T, np.linalg.solve(self.L, yn))
        return self

    def _k(self, A, B):
        d = (A[:, None, :] - B[None, :, :]) / self.ls
        return self.sig * np.exp(-0.5 * np.sum(d ** 2, -1))

    def predict(self, X):
        Xn = (np.atleast_2d(X) - self.mu) / self.sd
        return self._k(Xn, self.Xn) @ self.a * self.ys + self.ym

    def grad(self, x):
        x = np.asarray(x, float).ravel()
        z = (x - self.mu) / self.sd
        diff = z[None, :] - self.Xn
        kv = self.sig * np.exp(-0.5 * np.sum((diff / self.ls) ** 2, 1))
        dk = kv[:, None] * (-(diff) / self.ls ** 2) / self.sd
        return self.ys * (self.a @ dk)

    def std(self, X):
        Xn = (np.atleast_2d(X) - self.mu) / self.sd
        ks = self._k(Xn, self.Xn)
        v = np.linalg.solve(self.L, ks.T)
        var = self.sig - np.sum(v ** 2, 0)
        return np.sqrt(np.clip(var, 0, None)) * self.ys


class AeroSurrogate:
    LS_CL = [2.5, 1.5, 1.5, 2.5]
    LS_SPL = [2.5, 1.5, 1.5, 1.5]
    LS_CM = [2.5, 1.5, 1.5, 2.5]

    def __init__(self, gp_cl, gp_spl, gp_cm=None):
        self._cl = gp_cl; self._spl = gp_spl; self._cm = gp_cm

    @classmethod
    def from_csv(cls, path=None, noise=1e-3):
        path = path or DATASET
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        d = pd.read_csv(path)
        need = _FEATS + ["CL", SPL_COL]
        n0 = len(d)
        d = d.dropna(subset=need)
        if len(d) < n0:
            print(f"surrogate: {len(d)} of {n0} rows in "
                  f"{os.path.basename(path)} carry {SPL_COL}; "
                  f"{n0-len(d)} dropped")
        if len(d) < 0.5 * n0:
            raise SystemExit(
                f"only {100*len(d)/n0:.0f} % of {os.path.basename(path)} carries "
                f"a usable {SPL_COL}. A surrogate fitted on that covers a "
                f"different design box than the one it is queried over.")
        X = d[_FEATS].values.astype(float)
        gp_cl = _RBFGP(cls.LS_CL, noise).fit(X, d["CL"].values.astype(float))
        gp_spl = _RBFGP(cls.LS_SPL, noise).fit(X, d[SPL_COL].values.astype(float))
        gp_cm = None
        if "CM" in d.columns and d["CM"].notna().all():
            gp_cm = _RBFGP(cls.LS_CM, noise).fit(X, d["CM"].values.astype(float))
        return cls(gp_cl, gp_spl, gp_cm)

    def CL(self, a, d1, d2, U):  return float(self._cl.predict([[a, d1, d2, U]])[0])
    def SPL(self, a, d1, d2, U): return float(self._spl.predict([[a, d1, d2, U]])[0])
    def SPL_std(self, a, d1, d2, U): return float(self._spl.std([[a, d1, d2, U]])[0])
    def CM(self, a, d1, d2, U):
        if self._cm is None:
            raise RuntimeError("no C_M surrogate: the dataset carries no CM column")
        return float(self._cm.predict([[a, d1, d2, U]])[0])

    def CL_grad_delta(self, a, d1, d2, U):
        g = self._cl.grad([a, d1, d2, U]); return float(g[1]), float(g[2])
    def SPL_grad_delta(self, a, d1, d2, U):
        g = self._spl.grad([a, d1, d2, U]); return float(g[1]), float(g[2])


class UVecAdapter:
    def __init__(self, S):
        self.S = S
    def CL(self, alpha, u, U):  return self.S.CL(alpha, u[0], u[1], U)
    def SPL(self, alpha, u, U): return self.S.SPL(alpha, u[0], u[1], U)
    def CL_grad_u(self, alpha, u, U):
        return np.array(self.S.CL_grad_delta(alpha, u[0], u[1], U))
    def SPL_grad_u(self, alpha, u, U):
        return np.array(self.S.SPL_grad_delta(alpha, u[0], u[1], U))
    def SPL_std(self, alpha, u, U): return self.S.SPL_std(alpha, u[0], u[1], U)


if __name__ == "__main__":
    from itertools import product
    here = os.path.dirname(os.path.abspath(__file__))
    d = pd.read_csv(os.path.join(here, "dataset.csv")).dropna().reset_index(drop=True)
    X = d[_FEATS].values.astype(float)

    def cv(col, ls):
        y = d[col].values.astype(float)
        idx = np.arange(len(X)); np.random.RandomState(0).shuffle(idx)
        folds = np.array_split(idx, 5); num = den = 0.0; se = []
        for f in folds:
            tr = np.setdiff1d(idx, f)
            g = _RBFGP(ls, 1e-3).fit(X[tr], y[tr]); p = g.predict(X[f])
            num += np.sum((p - y[f]) ** 2); den += np.sum((y[f] - y.mean()) ** 2)
            se.append(np.sqrt(np.mean((p - y[f]) ** 2)))
        print(f"  {col:3s}: 5-fold R2={1-num/den:.4f}  RMSE={np.mean(se):.4f}")

    print("Cross-validated accuracy:")
    cv("CL", AeroSurrogate.LS_CL); cv("SPL", AeroSurrogate.LS_SPL)

    S = AeroSurrogate.from_csv()
    a, d1, d2, U = 6.0, 3.0, -2.0, 55.0
    h = 1e-3
    g_an = S.SPL_grad_delta(a, d1, d2, U)
    g_fd = ((S.SPL(a, d1 + h, d2, U) - S.SPL(a, d1 - h, d2, U)) / (2 * h),
            (S.SPL(a, d1, d2 + h, U) - S.SPL(a, d1, d2 - h, U)) / (2 * h))
    print("\nGradient check (SPL wrt delta1,delta2) at (a=6,d1=3,d2=-2,U=55):")
    print(f"  analytic     = ({g_an[0]:+.5f}, {g_an[1]:+.5f}) dB/deg")
    print(f"  finite-diff  = ({g_fd[0]:+.5f}, {g_fd[1]:+.5f}) dB/deg")
    err = max(abs(g_an[0]-g_fd[0]), abs(g_an[1]-g_fd[1]))
    print(f"  max abs err  = {err:.2e}  -> {'OK' if err<1e-3 else 'CHECK'}")
    print(f"\n  SPL predictive std here = {S.SPL_std(a,d1,d2,U):.2f} dB (confidence signal)")
    print("[OK] surrogate ready -> next: python proto_control.py")
