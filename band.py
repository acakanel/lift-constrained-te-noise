"""The certified band, in one place."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _read(start=None):
    here = start or HERE
    for d in (here, os.path.dirname(here)):
        p = os.path.join(d, "mc_robustness.py")
        if os.path.exists(p):
            for line in open(p):
                if line.startswith("EPS = "):
                    return float(line.split("=")[1].split("#")[0])
    raise RuntimeError(
        "mc_robustness.py not found beside " + here +
        " -- the band cannot be read, and guessing it is how it went wrong "
        "the first time")


EPS = _read()


def label(certified=True):
    return (f"Certified band, $\\varepsilon={EPS:g}$" if certified
            else f"Band, $\\varepsilon={EPS:g}$")
