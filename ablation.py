#!/usr/bin/env python3
"""Surrogate-family comparison under repeated cross-validation."""
from __future__ import annotations
import os, sys, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from surrogate import _RBFGP, _FEATS                      # noqa: E402

R_REPEATS, K_FOLDS, N_BOOT = 5, 5, 4000


class Poly3:
    smooth, uncert, label = True, False, "Polynomial (cubic)"

    def __init__(self, lam=1e-3):
        self.lam = lam

    def _feat(self, X):
        Xs = (X - self.mu) / self.sd
        n = len(Xs)
        cols = [np.ones(n)]
        terms = [(i,) for i in range(4)]
        terms += [(i, j) for i in range(4) for j in range(i, 4)]
        terms += [(i, j, k) for i in range(4) for j in range(i, 4) for k in range(j, 4)]
        for t in terms:
            c = np.ones(n)
            for idx in t:
                c = c * Xs[:, idx]
            cols.append(c)
        return np.column_stack(cols)

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        F = self._feat(X)
        self.w = np.linalg.solve(F.T @ F + self.lam * np.eye(F.shape[1]), F.T @ y)
        return self

    def predict(self, X):
        return self._feat(np.atleast_2d(X)) @ self.w


class KNN:
    smooth, uncert, label = False, False, "$k$-nearest neighbours"

    def __init__(self, k=7):
        self.k = k

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        self.Xn, self.y = (X - self.mu) / self.sd, y
        return self

    def predict(self, X):
        Xn = (np.atleast_2d(X) - self.mu) / self.sd
        out = []
        for x in Xn:
            dd = np.sqrt(np.sum((self.Xn - x) ** 2, 1))
            i = np.argsort(dd)[:self.k]
            w = 1 / (dd[i] + 1e-6)
            out.append(np.sum(w * self.y[i]) / np.sum(w))
        return np.array(out)


class GPnp:
    smooth, uncert, label = True, True, "Gaussian process"

    def __init__(self, ls=(2.5, 1.5, 1.5, 2.0)):
        self.ls = ls

    def fit(self, X, y):
        self.m = _RBFGP(self.ls, 1e-3).fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)


def build_methods():
    M = {"Gaussian process": GPnp, "Polynomial (cubic)": Poly3,
         "$k$-nearest neighbours": KNN}
    try:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        def wrap(est, smooth, uncert, label):
            class W:
                def __init__(s):
                    s.smooth, s.uncert, s.label = smooth, uncert, label

                def fit(s, X, y):
                    s.p = make_pipeline(StandardScaler(), est())
                    s.p.fit(X, y)
                    return s

                def predict(s, X):
                    return s.p.predict(np.atleast_2d(X))
            return W

        M["Random forest"] = wrap(
            lambda: RandomForestRegressor(300, random_state=0), False, False, "Random forest")
        M["Gradient boosting"] = wrap(
            lambda: GradientBoostingRegressor(random_state=0), False, False, "Gradient boosting")
        M["Multilayer perceptron"] = wrap(
            lambda: MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=0), True, False,
            "Multilayer perceptron")
    except Exception:
        print("[info] scikit-learn unavailable; comparing the NumPy families only")
    try:
        from xgboost import XGBRegressor
        M["Gradient boosting (XGBoost)"] = wrap(
            lambda: XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                                 random_state=0), False, False, "Gradient boosting (XGBoost)")
    except Exception:
        pass
    return M


def folds(n, k, seed):
    idx = np.arange(n)
    np.random.RandomState(seed).shuffle(idx)
    return np.array_split(idx, k)


def cv_scores(fn, X, y, repeats=R_REPEATS, k=K_FOLDS):
    r2, rmse = [], []
    for rep in range(repeats):
        for f in folds(len(X), k, seed=rep):
            tr = np.setdiff1d(np.arange(len(X)), f)
            p = np.asarray(fn().fit(X[tr], y[tr]).predict(X[f])).ravel()
            ss_res = np.sum((p - y[f]) ** 2)
            ss_tot = np.sum((y[f] - y[f].mean()) ** 2)
            r2.append(1 - ss_res / ss_tot)
            rmse.append(np.sqrt(np.mean((p - y[f]) ** 2)))
    return np.array(r2), np.array(rmse)


def boot_ci(v, n_boot=N_BOOT, seed=0):
    rng = np.random.RandomState(seed)
    m = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    lo, hi = boot_ci(d)
    sd = d.std(ddof=1)
    return float(d.mean()), lo, hi, float(d.mean() / sd) if sd > 0 else np.nan


def roughness(fn, X, y):
    m = fn().fit(X, y)
    n = 1000
    line = np.linspace(-10, 10, n)
    Xl = np.column_stack([np.full(n, 6.0), line, np.zeros(n), np.full(n, 55.0)])
    f = np.asarray(m.predict(Xl)).ravel()
    return float(np.sqrt(np.mean(np.diff(f, 2) ** 2)))


def run(dataset="dataset.csv", spl_col="SPL", out="ablation.csv"):
    d = pd.read_csv(os.path.join(HERE, dataset)).dropna().reset_index(drop=True)
    X = d[_FEATS].values.astype(float)
    yCL, ySPL = d["CL"].values.astype(float), d[spl_col].values.astype(float)
    M = build_methods()

    print(f"dataset {dataset}: {len(d)} cases; "
          f"{R_REPEATS}x{K_FOLDS}-fold cross-validation, "
          f"{R_REPEATS*K_FOLDS} folds per score\n")
    store, rows = {}, []
    for name, fn in M.items():
        r2c, rmc = cv_scores(fn, X, yCL)
        r2s, rms = cv_scores(fn, X, ySPL)
        store[name] = dict(rms=rms, r2s=r2s)
        inst = fn()
        rows.append(dict(
            method=name,
            CL_R2=r2c.mean(), CL_R2_lo=boot_ci(r2c)[0], CL_R2_hi=boot_ci(r2c)[1],
            CL_RMSE=rmc.mean(),
            SPL_R2=r2s.mean(), SPL_R2_lo=boot_ci(r2s)[0], SPL_R2_hi=boot_ci(r2s)[1],
            SPL_RMSE=rms.mean(), SPL_RMSE_lo=boot_ci(rms)[0], SPL_RMSE_hi=boot_ci(rms)[1],
            roughness=roughness(fn, X, yCL),
            smooth=getattr(inst, "smooth", False),
            uncertainty=getattr(inst, "uncert", False)))
        r = rows[-1]
        print(f"{name:28s} SPL RMSE {r['SPL_RMSE']:5.2f} dB "
              f"[{r['SPL_RMSE_lo']:.2f}, {r['SPL_RMSE_hi']:.2f}]   "
              f"C_L R2 {r['CL_R2']:.4f}   roughness {r['roughness']:.1e}   "
              f"{'smooth' if r['smooth'] else 'non-smooth':10s} "
              f"{'uncertainty' if r['uncertainty'] else '--'}")

    ref = "Gaussian process"
    print(f"\npaired difference in SPL RMSE against the {ref.lower()} "
          f"(positive means worse), over matched folds")
    pair = {}
    for name in M:
        if name == ref:
            continue
        m, lo, hi, dz = paired(store[name]["rms"], store[ref]["rms"])
        sig = "" if lo <= 0 <= hi else "  *"
        pair[name] = (m, lo, hi, dz)
        print(f"  {name:28s} {m:+6.3f} dB  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"d = {dz:+.2f}{sig}")
    for r in rows:
        m, lo, hi, dz = pair.get(r["method"], (0.0, 0.0, 0.0, 0.0))
        r["paired_diff_dB"] = m
        r["paired_diff_lo"] = lo
        r["paired_diff_hi"] = hi
        r["effect_size_dz"] = dz

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, out), index=False)
    print(f"\nwrote {out}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset.csv")
    ap.add_argument("--spl-col", default="SPL")
    ap.add_argument("--out", default="ablation.csv")
    a = ap.parse_args()
    run(a.dataset, a.spl_col, a.out)
