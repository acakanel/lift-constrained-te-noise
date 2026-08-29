#!/usr/bin/env python3
"""Re-derive every quantity from the solutions already on disk, without re-solving anything."""
from __future__ import annotations
import os, sys, json, glob, shutil, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import convergence as C
import post


def times_in(case_dir):
    out = []
    for d in glob.glob(os.path.join(case_dir, "VTK", "*_*")):
        if not os.path.isdir(d):
            continue
        try:
            t = int(d.rsplit("_", 1)[1])
        except ValueError:
            continue
        if t > 0:
            out.append(t)
    return sorted(out)


def redo(rec, case_dir):
    alpha = rec["alpha"]
    hist = []
    for t in times_in(case_dir):
        try:
            with C.at_time(t):
                e = post.extract(case_dir, C.U, alpha)
            hist.append(dict(it=t, **{k: e.get(k) for k in
                                      ("CL", "dstar_s", "dstar_p",
                                       "H_s", "H_p")}))
        except Exception as exc:
            hist.append(dict(it=t, error=f"{type(exc).__name__}: {exc}"))
    final = post.extract(case_dir, C.U, alpha)
    new = dict(rec)
    new["history"] = hist
    for k in ("CL", "CD", "CM", "dstar_s", "dstar_p", "H_s", "H_p",
              "theta_s", "theta_p"):
        new[k] = final.get(k)
    for k in ("bl_ok_s", "bl_ok_p", "reversed_s", "reversed_p",
              "bl_reason_s", "bl_reason_p"):
        if k in final:
            new[k] = final[k]
    if final.get("dstar_p") and final.get("dstar_s"):
        new["sym_error_dB"] = float(10 * np.log10(final["dstar_s"] /
                                                  final["dstar_p"]))
    else:
        new.pop("sym_error_dB", None)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="resweep")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    root = os.path.join(HERE, a.dir)
    jpath = a.json or os.path.join(root, f"{a.dir}.json")
    recs = json.load(open(jpath))
    shutil.copy2(jpath, jpath + ".pre_edgefix")
    print(f"kept the previous records as {os.path.basename(jpath)}.pre_edgefix\n")

    print(f"{'case':22s} {'d*_s old':>10s} {'d*_s new':>10s} {'change':>9s}  status")
    out = []
    for rec in recs:
        d = os.path.join(root, f"{rec['tag']}_{rec['level']}")
        if not os.path.isdir(d):
            print(f"{rec['tag']:22s}  solution directory missing, kept as is")
            out.append(rec); continue
        old_ds = rec.get("dstar_s")
        new = redo(rec, d)
        out.append(new)
        nd = new.get("dstar_s")
        if nd and old_ds:
            print(f"{rec['tag']:22s} {1e3*old_ds:10.4f} {1e3*nd:10.4f} "
                  f"{10*np.log10(nd/old_ds):+9.4f}  ok")
        elif old_ds:
            print(f"{rec['tag']:22s} {1e3*old_ds:10.4f} {'--':>10s} {'--':>9s}  "
                  f"REFUSED (separated; no boundary-layer edge)")
        else:
            print(f"{rec['tag']:22s} {'--':>10s} {'--':>10s} {'--':>9s}  refused")
    json.dump(out, open(jpath, "w"), indent=1)
    print(f"\nrewrote {os.path.relpath(jpath, HERE)}")

    import resweep
    resweep.write_csv(out)
    print(f"rewrote {os.path.basename(resweep.CSV)} from those records")
    return out


if __name__ == "__main__":
    main()
