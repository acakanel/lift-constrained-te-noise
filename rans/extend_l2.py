"""Continue the finest level to the iteration budget its boundary-layer integrals need."""
import os
import re
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "conv")
JSON = os.path.join(OUT, "convergence.json")


def main(target, level):
    if not os.path.isdir(OUT):
        raise SystemExit(f"{OUT} does not exist")
    touched = []
    for name in sorted(os.listdir(OUT)):
        d = os.path.join(OUT, name)
        if not (os.path.isdir(d) and name.endswith("_" + level)):
            continue
        cd = os.path.join(d, "system", "controlDict")
        if not os.path.exists(cd):
            continue
        s = open(cd).read()
        m = re.search(r"endTime\s+(\d+)\s*;", s)
        if not m:
            raise SystemExit(f"no endTime in {cd}")
        was = int(m.group(1))
        if was >= target:
            print(f"  {name}: endTime already {was}, left alone")
            continue
        open(cd, "w").write(re.sub(r"endTime\s+\d+\s*;",
                                   f"endTime {target};", s))
        done = os.path.join(d, "DONE.json")
        if os.path.exists(done):
            os.remove(done)
        touched.append(name)
        print(f"  {name}: endTime {was} -> {target}, DONE marker cleared")

    if not touched:
        print("  nothing to extend")
        return

    if os.path.exists(JSON):
        rows = json.load(open(JSON))
        keep = [r for r in rows if r["level"] != level]
        drop = [r for r in rows if r["level"] == level]
        if drop:
            side = os.path.join(OUT, f"convergence_{level}_at20k.json")
            json.dump(drop, open(side, "w"), indent=1)
            json.dump(keep, open(JSON, "w"), indent=1)
            print(f"  moved {len(drop)} {level} record(s) to "
                  f"{os.path.basename(side)}")

    print(f"\n  now re-run:  python3 convergence.py --levels 2 "
          f"--cases deflect neutral_lm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", type=int, default=40000)
    ap.add_argument("--level", default="L2")
    main(ap.parse_args().to, ap.parse_args().level)
