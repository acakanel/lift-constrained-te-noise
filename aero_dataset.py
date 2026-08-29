#!/usr/bin/env python3
"""Build the (alpha, delta1, delta2, U) -> (C_L, C_M, SPL) dataset from NeuralFoil and the BPM self-noise model."""
import os, argparse, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from bpm_noise import overall_spl

CHORD = 1.0
NU    = 1.5e-5
HINGE1, HINGE2 = 0.70, 0.85
NACA  = "0012"


def _airfoil(d1, d2):
    import aerosandbox as asb
    af = asb.Airfoil(f"naca{NACA}")
    af = af.add_control_surface(deflection=d1, hinge_point_x=HINGE1)
    af = af.add_control_surface(deflection=d2, hinge_point_x=HINGE2)
    return af


def _extract_dstar(aero):
    def te_val(prefix):
        keys=[k for k in aero if k.startswith(prefix)]
        if not keys: return None
        idx=max(int(k[len(prefix):]) for k in keys)
        return float(np.ravel(aero[f"{prefix}{idx}"])[0])
    th_u=te_val("upper_bl_theta_"); H_u=te_val("upper_bl_H_")
    th_l=te_val("lower_bl_theta_"); H_l=te_val("lower_bl_H_")
    if None in (th_u,H_u,th_l,H_l): return None, None
    return th_u*H_u, th_l*H_l


def run_case(alpha, d1, d2, U, model_size="large", dump=False):
    import neuralfoil  # noqa
    af = _airfoil(d1, d2)
    Re = U*CHORD/NU; mach = U/340.46
    aero = af.get_aero_from_neuralfoil(alpha=alpha, Re=Re, mach=mach, model_size=model_size)
    aero = {k: (np.asarray(v)) for k,v in aero.items()}
    if dump:
        print("Available NeuralFoil fields:")
        for k in aero:
            v=aero[k]; print(f"   {k:22s} shape={np.shape(v)}  sample={np.ravel(v)[:1]}")
    CL = float(np.ravel(aero["CL"])[0])
    CM = float(np.ravel(aero["CM"])[0])
    ds, dp = _extract_dstar(aero)
    if ds is not None: ds *= CHORD
    if dp is not None: dp *= CHORD
    return CL, CM, ds, dp


def one(alpha=6.0, d1=5.0, d2=0.0, U=55.0):
    print(f"[single case] alpha={alpha} d1={d1} d2={d2} U={U} (Re={U*CHORD/NU:.0f})\n")
    CL, CM, ds, dp = run_case(alpha, d1, d2, U, dump=True)
    print(f"\n  C_L = {CL:.4f}   C_M = {CM:.4f}")
    print(f"  dstar_suction = {ds}   dstar_pressure = {dp}")
    if ds and dp:
        print(f"  SPL (BPM) = {overall_spl(ds,dp,U,alpha,CHORD):.1f} dB")
        print("  [OK] full chain works -> run: python aero_dataset.py")
    else:
        print("  [CHECK] couldn't find dstar fields — paste the 'Available fields' list")
        print("          above and I'll map the exact field names.")


def sweep():
    ALPHAS=[0,3,6,9,12]; D1=[-10,-5,0,5,10]; D2=[-10,-5,0,5,10]; US=[40,55,70]
    combos=list(itertools.product(ALPHAS,D1,D2,US)); out="dataset.csv"
    done=set()
    have_file = os.path.exists(out) and os.path.getsize(out) > 0
    if have_file:
        try:
            done={tuple(r) for r in pd.read_csv(out)[["alpha","delta1","delta2","U"]].values}
            print(f"resuming: {len(done)} done")
        except Exception:
            have_file=False; done=set()
            print("existing dataset.csv was empty/unreadable — starting fresh")
    need_header = not have_file
    n=len(combos)
    for i,(a,d1,d2,U) in enumerate(combos):
        if (a,d1,d2,U) in done: continue
        try: CL,CM,ds,dp=run_case(a,d1,d2,U)
        except Exception as e: CL=CM=ds=dp=None
        spl=overall_spl(ds,dp,U,a,CHORD) if (CL is not None and ds and dp) else np.nan
        row=pd.DataFrame([dict(alpha=a,delta1=d1,delta2=d2,U=U,CL=CL,CM=CM,dstar_s=ds,dstar_p=dp,SPL=spl)])
        row.to_csv(out, mode="a", header=need_header, index=False)
        need_header=False
        if (i+1)%20==0 or i==n-1: print(f"  {i+1}/{n}  (a={a} d1={d1} d2={d2} U={U} CL={CL} SPL={spl})")
    df=pd.read_csv(out).dropna()
    print(f"\nDONE -> {out}: {len(df)} valid cases | CL {df.CL.min():.2f}..{df.CL.max():.2f} | "
          f"SPL {df.SPL.min():.1f}..{df.SPL.max():.1f} dB")


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--single",action="store_true")
    one() if ap.parse_args().single else sweep()
