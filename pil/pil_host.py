#!/usr/bin/env python3
"""Processor-in-the-loop driver."""
from __future__ import annotations
import os, sys, json, struct, argparse, subprocess, time
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

def _band():
    for d in (HERE, os.path.dirname(HERE)):
        p = os.path.join(d, "mc_robustness.py")
        if os.path.exists(p):
            for line in open(p):
                if line.startswith("EPS = "):
                    return float(line.split("=")[1].split("#")[0]), "mc_robustness.py"
    return 0.12, "this file's fallback -- mc_robustness.py not found beside the pack"


BAND, BAND_SRC = _band()

os.environ.setdefault("AERO_DATASET", "dataset.csv")
os.environ.setdefault("AERO_SPL_COL", "SPL")

REQ_SYNC = b"\xA5\x5A"
RSP_SYNC = b"\x5A\xA5"
CMD_PING, CMD_INIT, CMD_STEP, CMD_BENCH, CMD_FAULT, CMD_ERR = 1, 2, 3, 4, 5, 0x7F
USE_MEAS_CL, USE_MEAS_D = 1, 2
FAULTS = ["non-finite lift measurement", "non-finite speed",
          "operating point off the map", "lift demand beyond authority",
          "surface frozen at its limit"]
FLAG_NAMES = [(0x01, "POS_SAT"), (0x02, "RATE_SAT"), (0x04, "SP_CLAMP"),
              (0x08, "LOW_CONF"), (0x10, "NONFINITE"), (0x20, "ILLCOND")]


def crc16(b: bytes) -> int:
    c = 0xFFFF
    for x in b:
        c ^= x << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x1021) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
    return c


def flagstr(f):
    s = [n for m, n in FLAG_NAMES if f & m]
    return "|".join(s) if s else "-"


class Link:
    def __init__(self, port=None, baud=921600, sim=False):
        self.sim = sim
        self.retries = 0
        if sim:
            exe = os.path.join(HERE, "sim_target")
            if not os.path.exists(exe):
                raise SystemExit("sim_target not built; run make")
            self.p = subprocess.Popen([exe], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE)
            self.w, self.r = self.p.stdin, self.p.stdout
        else:
            import serial                                   # noqa: F401
            self.ser = serial.Serial(port, baud, timeout=2.0)
            time.sleep(0.2)
            self.ser.reset_input_buffer()

    def _write(self, b):
        if self.sim:
            self.w.write(b); self.w.flush()
        else:
            self.ser.write(b)

    def _read(self, n):
        if self.sim:
            out = b""
            while len(out) < n:
                k = self.r.read(n - len(out))
                if not k:
                    raise IOError("target closed the link")
                out += k
            return out
        out = self.ser.read(n)
        if len(out) != n:
            raise IOError(f"short read: wanted {n}, got {len(out)}")
        return out

    def xfer(self, cmd, payload=b"", tries=3):
        body = bytes([cmd, len(payload)]) + payload
        frame = REQ_SYNC + body + struct.pack("<H", crc16(body))
        last = None
        for attempt in range(tries):
            try:
                if not self.sim:
                    self.ser.reset_input_buffer()
                self._write(frame)
                while True:
                    b = self._read(1)
                    if b != RSP_SYNC[0:1]:
                        continue
                    if self._read(1) == RSP_SYNC[1:2]:
                        break
                head = self._read(2)
                rcmd, n = head[0], head[1]
                pay = self._read(n) if n else b""
                crc = struct.unpack("<H", self._read(2))[0]
                if crc16(head + pay) != crc:
                    raise IOError("bad CRC in reply")
                if rcmd == CMD_ERR:
                    raise IOError(f"target error {struct.unpack('<I', pay)[0]}")
                if attempt:
                    self.retries += 1
                return pay
            except IOError as e:
                last = e
                if self.sim:
                    raise
        raise IOError(f"link failed after {tries} attempts: {last}")

    def close(self):
        if self.sim:
            self.p.terminate()
        else:
            self.ser.close()


def ping(L):
    r = struct.unpack("<9I", L.xfer(CMD_PING))
    b = r[8]
    opt = ("-O0 (unoptimised)" if not (b & 1) else
           "-Os" if (b & 2) else "-O1/-O2/-O3")
    return dict(magic=r[0], gp_n=r[1], sp_nu=r[2], sp_ncl=r[3], clk=r[4],
                icache=bool(r[5]), dcache=bool(r[6]), float_bytes=r[7],
                optimisation=opt, gcc_major=(b >> 8) & 0xFF,
                optimised=bool(b & 1))


def init(L, u0):
    return np.array(struct.unpack("<3f", L.xfer(CMD_INIT, struct.pack("<3f", *u0))))


def step(L, U, CLreq, CLmeas, d, Up, CLp, preview, mode):
    pay = struct.pack("<10fI", U, CLreq, CLmeas, d[0], d[1], d[2],
                      Up, CLp, preview, 0.0, mode)
    r = L.xfer(CMD_STEP, pay)
    f = struct.unpack("<9f", r[:36])
    u32 = struct.unpack("<4I", r[36:52])
    return dict(rate=np.array(f[0:3]), u=np.array(f[3:6]), CL=f[6], s=f[7],
                sigma=f[8], flags=u32[0], c_step=u32[1], c_gp=u32[2],
                c_outer=u32[3])


def bench(L, n=400):
    r = struct.unpack("<16I", L.xfer(CMD_BENCH, struct.pack("<I", n)))
    return dict(n=r[0], clk=r[1],
                inner=(r[2], r[3], r[4]), outer=(r[5], r[6], r[7]),
                gp=(r[8], r[9], r[10]), sigma=(r[11], r[12], r[13]),
                map_mean=r[14], cache=r[15])


def fault(L, kind, nstep=200):
    r = L.xfer(CMD_FAULT, struct.pack("<2I", kind, nstep))
    f = struct.unpack("<7f", r[:28])
    u = struct.unpack("<4I", r[28:44])
    return dict(rate=np.array(f[0:3]), u=np.array(f[3:6]), s_max=f[6],
                flags=u[0], kind=u[1], ok=bool(u[2]), nstep=u[3])


import ref_loop as R                                       # noqa: E402

DT = R.DT


class Remote:
    def __init__(self, link):
        self.L = link
        self.cyc = []

    def init(self, u0):
        return init(self.L, u0)

    def step(self, U, CLreq, CLmeas, d, Up, CLp, preview, mode):
        r = step(self.L, U, CLreq, CLmeas, d, Up, CLp, preview, mode)
        self.cyc.append((r["c_step"], r["c_gp"], r["c_outer"]))
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--bench-n", type=int, default=400)
    ap.add_argument("--mc", type=int, default=20,
                    help="robustness trials across the link; 0 to skip")
    a = ap.parse_args()
    if not a.sim and not a.port:
        ap.error("give --port, or --sim for the workstation stand-in")

    L = Link(a.port, a.baud, a.sim)
    res = {}
    try:
        info = ping(L)
        res["target"] = info
        print(f"target  (band {BAND:g}, from {BAND_SRC})")
        print(f"  retained training set {info['gp_n']} points, "
              f"map {info['sp_nu']}x{info['sp_ncl']}, "
              f"float {info['float_bytes']*8} bit")

        import ref_loop as _R
        hm = _R._M["table"].shape
        if (info["sp_nu"], info["sp_ncl"]) != (hm[0], hm[1]):
            raise SystemExit(
                f"\n  STOP: the board carries a {info['sp_nu']}x{info['sp_ncl']} "
                f"set-point map and deployed_map.npz here is {hm[0]}x{hm[1]}.\n"
                f"  Every fidelity and twin figure below would be comparing two "
                f"different tables.\n"
                f"  Whichever is behind, fix it before running: re-export, or "
                f"reflash, or copy the\n  current deployed_map.npz into this "
                f"directory.")
        if info["clk"]:
            print(f"  core clock {info['clk']/1e6:.0f} MHz, "
                  f"I-cache {'on' if info['icache'] else 'off'}, "
                  f"D-cache {'on' if info['dcache'] else 'off'}")
            print(f"  built with gcc {info['gcc_major']}, {info['optimisation']}")
            if not info["optimised"]:
                print("  WARNING: an unoptimised build. The timing below is not "
                      "the deployed\n           figure -- rebuild with -O2 "
                      "before quoting it.")
        else:
            print("  workstation stand-in: timings are not meaningful")

        b = bench(L, a.bench_n)
        res["bench"] = b
        if b["clk"]:
            f = b["clk"]
            print(f"\ntiming over {b['n']} evaluations "
                  f"(budget {1e3*DT:.0f} ms per step)")
            def line(name, tup):
                lo, me, hi = tup
                print(f"  {name:26s} {me:8d} cycles  {1e6*me/f:7.1f} us  "
                      f"{100.0*me/f/DT:5.1f} %   [{lo}, {hi}]")
            line("inner step", b["inner"])
            line("step with outer refresh", b["outer"])
            line("constraint model alone", b["gp"])
            line("predictive spread alone", b["sigma"])
            print(f"  {'set-point map alone':26s} {b['map_mean']:8d} cycles  "
                  f"{1e6*b['map_mean']/f:7.1f} us")

        T = Remote(L)
        rows = R.kinematic(T)
        np.savetxt(os.path.join(HERE, "pil_loop.csv"), rows, delimiter=",",
                   header="scenario,i,t,alpha,delta1,delta2,CL,s,sigma,flags",
                   comments="")
        print("  regenerating the double-precision reference", flush=True)
        ref = R.kinematic(R.Local())
        np.savetxt(os.path.join(HERE, "ref_loop.csv"), ref, delimiter=",",
                   header="scenario,i,t,alpha,delta1,delta2,CL,s,sigma,flags",
                   comments="")
        fid = {}
        print("\nnumerical fidelity against the double-precision reference")
        for sc in (0, 1):
            A = rows[rows[:, 0] == sc]
            B = ref[ref[:, 0] == sc]
            m = min(len(A), len(B))
            du = np.abs(A[:m, 3:6] - B[:m, 3:6])
            ds = np.abs(A[:m, 7] - B[:m, 7])
            fid[f"scenario{sc}"] = dict(max_du_deg=float(du.max()),
                                        final_du_deg=float(du[-1].max()),
                                        max_ds=float(ds.max()),
                                        band_pct=float(100 * ds.max() / BAND))
            print(f"  scenario {sc}: max |du| {du.max():.2e} deg, "
                  f"settled |du| {du[-1].max():.2e} deg, "
                  f"max |ds| {ds.max():.2e} "
                  f"({100 * ds.max() / BAND:.2f} % of the band {BAND:g})")
        res["fidelity"] = fid

        d0, loud = R.loud_start()
        print("\nclosed loop across the link, dynamic plant with actuator and")
        print("circulation lag, turbulence and sensor noise")
        tr = R.dynamic(T, seed=0, T=10.0, d0=d0, **R.draw(np.random.RandomState(1)))
        np.savetxt(os.path.join(HERE, "pil_closedloop.csv"), tr, delimiter=",",
                   header="i,t,alpha,delta1,delta2,CL,viol,SPL,flags",
                   comments="")
        tl = R.dynamic(R.Local(), seed=0, T=10.0, d0=d0,
                       **R.draw(np.random.RandomState(1)))
        dtraj = np.abs(tr[:, 2:5] - tl[:, 2:5]).max()
        ss = tr[int(0.6 * len(tr)):]
        res["closed_loop"] = dict(
            loud_start=[float(x) for x in d0], loud_level=float(loud),
            settled_objective=float(ss[:, 7].mean()),
            descent_dB=float(loud - ss[:, 7].mean()),
            settled_lift_band=float(ss[:, 6].max()),
            max_trajectory_deviation_deg=float(dtraj))
        print(f"  loud start {np.round(d0, 2)} at {loud:.2f} dB")
        print(f"  objective descent {loud - ss[:,7].mean():.2f} dB, "
              f"settled lift error {ss[:,6].max():.4f} (band {BAND:g})")
        print(f"  trajectory deviation from the host-side twin "
              f"{dtraj:.2e} deg")

        if a.mc > 0:
            print(f"\nrobustness across the link, {a.mc} trials")
            rng = np.random.RandomState(1)
            band = np.zeros(a.mc); desc = np.zeros(a.mc)
            for k in range(a.mc):
                d = R.dynamic(T, seed=k, T=8.0, d0=d0, **R.draw(rng))
                q = d[int(0.6 * len(d)):]
                band[k] = q[:, 6].max(); desc[k] = loud - q[:, 7].mean()
                print(f"    trial {k+1:3d}/{a.mc}  descent {desc[k]:5.2f} dB  "
                      f"band {band[k]:.4f}")
            res["monte_carlo"] = dict(
                n=a.mc, descent_mean=float(desc.mean()),
                descent_min=float(desc.min()), descent_max=float(desc.max()),
                band_mean=float(band.mean()), band_max=float(band.max()),
                inside_band=int(np.sum(band < BAND)))
            print(f"  objective descent {desc.mean():.2f} dB "
                  f"[{desc.min():.2f}, {desc.max():.2f}]")
            print(f"  settled lift error, worst of {a.mc} trials "
                  f"{band.max():.4f}; inside the certified band in "
                  f"{int(np.sum(band < BAND))} of {a.mc}")

        print("\nfaults and saturation")
        fr = []
        print(f"  {'injection':32s} {'raised':36s} {'settled |s|':>11s}   state")
        for k, name in enumerate(FAULTS):
            r = fault(L, k, 200)
            fr.append(dict(kind=name, flags=int(r["flags"]),
                           flag_names=flagstr(r["flags"]),
                           settled_s=float(r["s_max"]), bounded=r["ok"],
                           u=[float(x) for x in r["u"]]))
            print(f"  {name:32s} {flagstr(r['flags']):36s} "
                  f"{r['s_max']:11.2e}   "
                  f"{'finite and inside the box' if r['ok'] else 'UNBOUNDED'}")
        res["faults"] = fr

        if L.retries:
            print(f"\nlink: {L.retries} frame(s) retried")
        res["link_retries"] = L.retries

        import re as _re
        _h = open(os.path.join(HERE, "ctrl.h")).read()
        res["firmware"] = {
            k: float(_re.search(r"#define\s+%s\s+([0-9.eE+-]+)f?" % k,
                                _h).group(1))
            for k in ("CTRL_K1", "CTRL_K2", "CTRL_MU", "CTRL_GAMMA", "CTRL_DT")
        }

        name = "pil_sim_results.json" if a.sim else "pil_results.json"
        with open(os.path.join(HERE, name), "w") as f:
            json.dump(res, f, indent=2, default=float)
        print(f"\nwrote pil_loop.csv, pil_closedloop.csv, {name}")
        if a.sim:
            print("  (stand-in run; the board's timings in pil_results.json are "
                  "untouched)")
    finally:
        L.close()


if __name__ == "__main__":
    main()
