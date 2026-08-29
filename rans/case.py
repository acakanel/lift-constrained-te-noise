#!/usr/bin/env python3
"""Build and run a 2-D steady RANS case (simpleFoam, k-omega SST) and extract the trailing-edge state."""
from __future__ import annotations
import os, shutil, subprocess, math, re
import numpy as np

FOAM_BASHRC = "/usr/share/openfoam/etc/bashrc"
NU = 1.5e-5


def _sh(cmd: str, cwd: str, timeout: int = 7200) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-lc", f"source {FOAM_BASHRC} >/dev/null 2>&1; {cmd}"],
                          cwd=cwd, capture_output=True, text=True, timeout=timeout)


HEAD = """FoamFile
{{
    version 2.0; format ascii; class {cls}; {extra}object {obj};
}}
"""


def _w(path: str, cls: str, obj: str, body: str, extra: str = ""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(HEAD.format(cls=cls, obj=obj, extra=extra))
        f.write(body)


def build(case: str, U: float, alpha_deg: float, n_iter: int = 3000,
          turb_intensity: float = 0.001, nut_ratio: float = 10.0,
          chord: float = 1.0, robust: bool = False):
    a = math.radians(alpha_deg)
    Uvec = (U * math.cos(a), U * math.sin(a), 0.0)
    k = 1.5 * (turb_intensity * U) ** 2
    omega = k / (nut_ratio * NU)
    nut = k / omega

    def field(obj, dims, internal, wall, far, cls="volScalarField"):
        return f"""
dimensions      {dims};
internalField   uniform {internal};
boundaryField
{{
    airfoil  {{ {wall} }}
    farfield {{ {far} }}
    front    {{ type empty; }}
    back     {{ type empty; }}
}}
"""
    _w(f"{case}/0/U", "volVectorField", "U", field(
        "U", "[0 1 -1 0 0 0 0]", f"({Uvec[0]:.6f} {Uvec[1]:.6f} 0)",
        "type noSlip;",
        f"type freestreamVelocity; freestreamValue uniform ({Uvec[0]:.6f} {Uvec[1]:.6f} 0);"))
    _w(f"{case}/0/p", "volScalarField", "p", field(
        "p", "[0 2 -2 0 0 0 0]", "0", "type zeroGradient;",
        "type freestreamPressure; freestreamValue uniform 0;"))
    _w(f"{case}/0/k", "volScalarField", "k", field(
        "k", "[0 2 -2 0 0 0 0]", f"{k:.6e}",
        f"type kqRWallFunction; value uniform {k:.6e};",
        f"type freestream; freestreamValue uniform {k:.6e};"))
    _w(f"{case}/0/omega", "volScalarField", "omega", field(
        "omega", "[0 0 -1 0 0 0 0]", f"{omega:.6e}",
        f"type omegaWallFunction; value uniform {omega:.6e};",
        f"type freestream; freestreamValue uniform {omega:.6e};"))
    _w(f"{case}/0/Phi", "volScalarField", "Phi", """
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    airfoil  { type zeroGradient; }
    farfield { type fixedValue; value uniform 0; }
    front    { type empty; }
    back     { type empty; }
}
""")
    _w(f"{case}/0/nut", "volScalarField", "nut", field(
        "nut", "[0 2 -1 0 0 0 0]", f"{nut:.6e}",
        "type nutUSpaldingWallFunction; value uniform 0;",
        "type calculated; value uniform 0;"))

    _w(f"{case}/constant/transportProperties", "dictionary", "transportProperties",
       f"\ntransportModel Newtonian;\nnu [0 2 -1 0 0 0 0] {NU};\n")
    _w(f"{case}/constant/turbulenceProperties", "dictionary", "turbulenceProperties",
       "\nsimulationType RAS;\nRAS { RASModel kOmegaSST; turbulence on; printCoeffs on; }\n")

    _w(f"{case}/system/controlDict", "dictionary", "controlDict", f"""
application     simpleFoam;
startFrom       startTime;  startTime 0;
stopAt          endTime;    endTime {n_iter};
deltaT          1;
writeControl    runTime;    writeInterval {n_iter};
purgeWrite      1;
writeFormat     binary;     writePrecision 8; writeCompression off;
timeFormat      general;    timePrecision 6; runTimeModifiable false;

functions
{{
    forceCoeffs
    {{
        type forceCoeffs; libs (forces); writeControl timeStep; writeInterval 50;
        patches (airfoil); rho rhoInf; rhoInf 1; log no;
        CofR (0.25 0 0); liftDir ({-math.sin(a):.8f} {math.cos(a):.8f} 0);
        dragDir ({math.cos(a):.8f} {math.sin(a):.8f} 0); pitchAxis (0 0 1);
        magUInf {U}; lRef {chord}; Aref {chord * 0.05};
    }}
    yPlus {{ type yPlus; libs (fieldFunctionObjects); writeControl onEnd; log no; }}
    residuals
    {{
        type solverInfo; libs (utilityFunctionObjects); writeControl timeStep;
        fields (U p k omega);
    }}
}}
""")
    _w(f"{case}/system/fvSchemes", "dictionary", "fvSchemes", """
ddtSchemes      { default steadyState; }
gradSchemes     { default cellLimited Gauss linear 1; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited corrected 0.33; }
interpolationSchemes { default linear; }
snGradSchemes   { default limited corrected 0.33; }
wallDist        { method meshWave; }
""")
    _w(f"{case}/system/fvSolution", "dictionary", "fvSolution",
       FVSOL_ROBUST if robust else FVSOL_STD)
    _w(f"{case}/system/decomposeParDict", "dictionary", "decomposeParDict",
       "\nnumberOfSubdomains 2;\nmethod hierarchical;\ncoeffs { n (2 1 1); order xyz; }\n")
    return dict(case=case, U=U, alpha=alpha_deg, Uvec=Uvec, k=k, omega=omega)


FVSOL_STD = """
solvers
{
    Phi { solver GAMG; tolerance 1e-6; relTol 0.01;
          smoother GaussSeidel; }
    p { solver GAMG; tolerance 1e-8; relTol 0.01; smoother GaussSeidel; }
    "(U|k|omega)" { solver smoothSolver; smoother symGaussSeidel;
                    tolerance 1e-9; relTol 0.01; nSweeps 2; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent yes;
    residualControl { p 1e-6; U 1e-7; k 1e-7; omega 1e-7; }
}
relaxationFactors { equations { U 0.9; ".*" 0.9; } }
potentialFlow { nNonOrthogonalCorrectors 5; }
"""

FVSOL_ROBUST = """
solvers
{
    Phi { solver GAMG; tolerance 1e-6; relTol 0.01;
          smoother GaussSeidel; }
    p { solver GAMG; tolerance 1e-7; relTol 0.05; smoother GaussSeidel;
        nPreSweeps 0; nPostSweeps 2; cacheAgglomeration on;
        agglomerator faceAreaPair; nCellsInCoarsestLevel 40; mergeLevels 1; }
    "(U|k|omega)" { solver smoothSolver; smoother symGaussSeidel;
                    tolerance 1e-8; relTol 0.1; nSweeps 2; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 2;
    consistent no;
    residualControl { p 1e-5; U 1e-6; k 1e-6; omega 1e-6; }
}
relaxationFactors
{
    fields    { p 0.3; }
    equations { U 0.7; k 0.7; omega 0.7; }
}
potentialFlow { nNonOrthogonalCorrectors 5; }
"""

BOUNDARY_FIX = {"front": "empty", "back": "empty", "airfoil": "wall", "farfield": "patch"}


def fix_boundary(case: str):
    p = f"{case}/constant/polyMesh/boundary"
    s = open(p).read()
    for name, typ in BOUNDARY_FIX.items():
        s = re.sub(rf"(\b{name}\s*\{{[^}}]*?type\s+)\w+;", rf"\g<1>{typ};", s, flags=re.S)
        if typ == "empty":
            s = re.sub(rf"(\b{name}\s*\{{[^}}]*?)physicalType\s+\w+;", r"\1", s, flags=re.S)
    open(p, "w").write(s)


def run(case: str, msh: str, np_proc: int = 2, timeout: int = 7200) -> dict:
    logs = {}
    r = _sh(f"gmshToFoam {msh} > log.gmshToFoam 2>&1", case); logs["gmshToFoam"] = r.returncode
    fix_boundary(case)
    r = _sh("checkMesh -constant > log.checkMesh 2>&1", case); logs["checkMesh"] = r.returncode
    if np_proc > 1:
        _sh("decomposePar -force > log.decomposePar 2>&1", case)
        r = _sh(f"mpirun --allow-run-as-root -np {np_proc} simpleFoam -parallel > log.simpleFoam 2>&1",
                case, timeout=timeout)
        _sh("reconstructPar -latestTime > log.reconstructPar 2>&1", case)
    else:
        r = _sh("simpleFoam > log.simpleFoam 2>&1", case, timeout=timeout)
    logs["simpleFoam"] = r.returncode
    return logs


def read_forces(case: str) -> dict:
    base = os.path.join(case, "postProcessing", "forceCoeffs")
    if not os.path.isdir(base):
        return {}
    t = sorted(os.listdir(base))[-1]
    f = os.path.join(base, t, "coefficient.dat")
    if not os.path.exists(f):
        cand = [x for x in os.listdir(os.path.join(base, t)) if x.endswith(".dat")]
        if not cand:
            return {}
        f = os.path.join(base, t, cand[0])
    hdr, rows = None, []
    for line in open(f):
        if line.startswith("#"):
            hdr = line
        else:
            rows.append(line.split())
    if not rows:
        return {}
    cols = hdr.replace("#", "").split()
    last = np.array(rows[-1], float)
    tail = np.array([np.array(r, float) for r in rows[-50:]], float)
    out = {c: float(v) for c, v in zip(cols, last)}
    out["_Cl_std_last50"] = float(tail[:, cols.index("Cl")].std()) if "Cl" in cols else np.nan
    return out
