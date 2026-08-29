#!/usr/bin/env python3
"""One place that decides which set of resolved-flow solutions is in use."""
from __future__ import annotations
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERGED = os.path.join(HERE, "rans", "rans_sweep_converged.csv")
LEGACY = os.path.join(HERE, "rans", "rans_sweep.csv")


def path() -> str:
    env = os.environ.get("RANS_SWEEP")
    if env:
        if env == "legacy":
            return LEGACY
        if env == "converged":
            return CONVERGED
        return env
    return CONVERGED if os.path.exists(CONVERGED) else LEGACY


def sweep(U: float | None = None) -> pd.DataFrame:
    p = path()
    d = pd.read_csv(p)
    if U is not None:
        d = d[d.U == U].copy()
    d.attrs["source"] = os.path.relpath(p, HERE)
    d.attrs["converged"] = os.path.abspath(p) == os.path.abspath(CONVERGED)
    return d


def banner() -> str:
    p = path()
    tag = "converged" if os.path.abspath(p) == os.path.abspath(CONVERGED) \
        else "LEGACY, under-converged"
    line = f"resolved-flow solutions: {os.path.relpath(p, HERE)}  [{tag}]"
    try:
        d = pd.read_csv(p)
        if "bl_ok_s" in d.columns:
            bad = d[~d.bl_ok_s.astype(bool)]
            if len(bad):
                line += (f"\n  {len(bad)} of {len(d)} configurations carry no "
                         f"suction-side boundary-layer solution (separated "
                         f"trailing edge; the edge detector refuses):")
                for _, r in bad.iterrows():
                    line += (f"\n    d1={r.d1:+g} d2={r.d2:+g} "
                             f"alpha={r.alpha:+g}  C_L={r.CL:.4f}  "
                             f"reversed fraction {r.reversed_s:.3f}")
    except Exception:
        pass
    return line


if __name__ == "__main__":
    d = sweep()
    print(banner())
    print(f"{len(d)} configurations")
    print(d.head().to_string(index=False))
