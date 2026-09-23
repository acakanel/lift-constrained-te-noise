# Lift-constrained trailing-edge noise control of a wing section

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22924822.svg)](https://doi.org/10.5281/zenodo.22924822)

Code and data for the manuscript

> **Lift-Constrained Trailing-Edge Noise Control of a Wing Section: Achievable
> Authority and the Role of Aeroacoustic Model Form**
> Ahmet Çakanel, *Aerospace Science and Technology* (accepted, 2026).

Every number, figure and table in the paper is produced by something in this
repository, from the data also in it. Nothing here is a demonstration written
afterwards.

The study asks how much trailing-edge self-noise a two-segment section can shed
while holding the lift the flight condition demands, and answers it in resolved
flow rather than through a semi-empirical chain. The answer is negative: the
authority is absent at both lift conditions examined. A constraint-control
framework that holds lift to a prescribed band under unknown bounded
disturbances is reported alongside it, and is independent of that result.

---

## What is here

| directory | contents |
|---|---|
| *(root)* | the acoustic model, the aerodynamic surrogate, the four control laws, the dynamic plant, the Monte-Carlo campaign, and the scripts that produce every figure and macro |
| `rans/` | the Reynolds-averaged rung: geometry, meshing, case setup, post-processing, the sweeps and the grid-convergence study |
| `pil/` | the processor-in-the-loop package: the deployed law in C, the exporter, the host driver and the double-precision reference |

Data files sit beside the code that reads them. `dataset.csv` is the 375-case
training set; `rans/rans_sweep_converged.csv` and `rans/conv/convergence.json`
are the resolved-flow solutions and the convergence study; `mc_results.csv` is
the 2000-trial campaign; `pil/pil_results.json` is the board run.

## Requirements

Python 3.11 with `numpy`, `scipy`, `pandas`, `matplotlib`. Additionally:

- `neuralfoil` and `aerosandbox` — only to rebuild `dataset.csv` from scratch
- `pyvista` — only to re-post-process Reynolds-averaged solutions
- `pyserial` — only to drive the board
- `xgboost` — only for one row of the surrogate-family comparison
- OpenFOAM v1912 and Gmsh — only to re-solve the flow
- `arm-none-eabi-gcc` — only to cross-compile the firmware

None of the optional packages is needed to reproduce a number from the data
already here.

## Reproducing the paper

Run from the repository root. Scripts that emit LaTeX macros write them into
`paper_AST/`, which is created on demand.

### The acoustic model and its comparison with measurement (§3)

| script | produces |
|---|---|
| `bpm_noise.py`, `bpm_noise_v2.py` | the reduced and complete Brooks–Pope–Marcolini formulations |
| `bpm_validation.py` | the comparison against the NASA database on absolute levels |
| `validation_stats.py` | the level errors, band RMSE and within-cell scatter |
| `fig_validation.py` | Figure 2 |

### The aerodynamic surrogate (§3.4–3.5)

| script | produces |
|---|---|
| `aero_dataset.py` | rebuilds `dataset.csv` |
| `surrogate.py` | the Gaussian-process fit used by every controller |
| `ablation.py` | the surrogate-family comparison, repeated cross-validation |
| `fig_fidelity.py` | Figure 3 |

### Achievable authority in resolved flow (§4)

| script | produces |
|---|---|
| `rans/sweep_alpha.py`, `rans/resweep.py` | the incidence sweep and the constant-lift manifold |
| `rans/convergence.py` | the three-level grid study, both lift conditions |
| `rans/fillet_sensitivity.py` | the hinge-fairing geometric uncertainty |
| `rans/edge_study.py`, `rans/edge_evidence.py` | the boundary-layer edge criterion |
| `conv_numbers.py` | every macro in the convergence appendix |
| `authority_recompute.py`, `manifold_table.py` | the manifold table |
| `fig_authority.py`, `fig_mechanism.py` | Figures 5 and 6 |

The resolved-flow solutions are already in `rans/`; the scripts above re-derive
the paper's numbers from them without re-solving. Re-solving needs OpenFOAM and
several days of wall time.

### Control (§5–§7)

| script | produces |
|---|---|
| `gains.py`, `band.py` | the super-twisting gains and the certified band, read by everything |
| `plant.py` | the dynamic plant: first-order actuators, circulation lag, von Kármán turbulence |
| `compare.py` | the four laws — SG-FTSMC, barrier-adaptive, CLF-QP, MPC |
| `control3.py`, `fixedtime.py` | the constraint law and the fixed-time evidence |
| `descent_fraction.py` | the fraction of the available descent each law realises |
| `rate_limit_sweep.py`, `rate_numbers.py` | what sets the achievable band |
| `gamma_sweep.py` | the null-space gain plateau |
| `jacobian_direction.py` | a Jacobian error in direction |
| `feas_test.py` | the rate test behind the band-feasibility assumption |
| `delta_bound.py` | the disturbance-rate bound, and the gain conditions checked against it |
| `barrier_clip.py` | how often the barrier gain's edge guard binds |
| `confidence_stress.py`, `fig_trust.py` | the trust diagnostic |
| `mc_robustness.py`, `mc_stats.py` | the 2000-trial campaign and every number quoted from it |
| `figures.py`, `scenarios.py`, `fig_controllers.py` | Figures 7–11 |

### Aircraft-level bound (§8.2)

`fig_system_level.py` — the energy identity and Figure 15.

## Execution on flight hardware

The constraint layer and the confidence gate run on an STM32F767ZI
(Arm Cortex-M7, 216 MHz). The set-point search does not: it scans 5625 box
points against 375 training points for two outputs, so it is precomputed over
the flight envelope and deployed as a **41 × 81** table in `(U, C_L,req)` read
at its nearest node. That table is the only approximation deployment
introduces. It costs **+0.018 dB** of the objective on average and **+1.30 dB**
at worst, and it cannot affect lift: it enters only through the null-space
projector, while lift is enforced by feedback on the measurement.

| file | role |
|---|---|
| `pil/export_model.py` | emits `gp_model.[ch]`, `deployed_map.npz`, `reference_vectors.csv` |
| `pil/ctrl.[ch]` | the deployed law: single precision, no allocation, `<math.h>` only |
| `pil/pil_core.[ch]`, `pil/pil_proto.h` | framing, CRC-16/CCITT-FALSE, command dispatch |
| `pil/pil_target.c` | board entry point and the DWT cycle counter |
| `pil/sim_target.c` | the same firmware built for the host, so the driver runs without a board |
| `pil/host_test.c` | single- against double-precision agreement at 64 reference points |
| `pil/ref_loop.py` | the double-precision twin of the deployed law |
| `pil/pil_host.py` | the driver: timing, precision, closed loop, faults |

Without hardware:

```
cd pil
make model       # regenerate the deployed model and the reference vectors
make verify      # single vs double precision at 64 reference points
make sim         # the whole driver against the host stand-in
make footprint   # cross-compile for Cortex-M7 and report the deployed size
```

`make sim` exercises the protocol, the scenarios, the plant coupling and the
fault battery. The only quantity it cannot produce is time on the target.

With hardware (NUCLEO-F767ZI):

1. Create an STM32CubeIDE project for the board at its default 216 MHz clock.
2. Enable **USART3** — the one wired to the ST-Link virtual COM port — at
   **921600 baud, 8N1**, no flow control.
3. Add `ctrl.c`, `ctrl.h`, `gp_model.c`, `gp_model.h`, `pil_core.c`,
   `pil_core.h`, `pil_proto.h`, `pil_target.c` to the project, and call
   `pil_server(&huart3)` from `main()` after the UART is initialised.
4. Build, flash, then run `python3 pil_host.py --port <device>`.

The driver refuses to proceed unless the set-point table compiled into the
firmware has the dimensions the exporter last wrote, so a board running a
superseded model is detected rather than measured.

### Everything else

| script | role |
|---|---|
| `figstyle.py` | one figure style for the whole paper: sizes, palette, line styles |
| `writeonce.py` | writes a generated file only when its contents would change |
| `ransdata.py` | the single place that decides which set of resolved-flow solutions is in use |
| `bpm_bl.py` | the boundary-layer thickness correlations the acoustic model needs |
| `cm_extension.py` | the pitching-moment budget as a second constraint |
| `corrected_designspace.py` | the correction attempt described in `NOTES_corrected_chain.md` |
| `make_schematic.py` | the wing-section schematic |
| `fig_manifold_rans.py` | the resolved-flow manifold, drawn |
| `rans/geometry.py`, `rans/mesh.py`, `rans/case.py` | the section outline, the Gmsh writer and the OpenFOAM case builder |
| `rans/post.py` | C_L, C_M and the trailing-edge state, extracted in Python from the VTK export |
| `rans/repost_converged.py` | re-derives every quantity from the stored solutions without re-solving |
| `rans/extend_l2.py` | continues the finest level to the iteration budget it needs |
| `pil/make_tex.py`, `pil/fig_pil.py` | the deployment macros and Figure 13 |

## A negative result that is kept

`NOTES_corrected_chain.md` documents an attempt to correct the fast aerodynamic
model against the Reynolds-averaged solutions and apply the correction to the
training set. It does not work, and the file sets out why: the fit is supported
on nineteen points all at non-negative deflection and is asked about a design
box four times as wide, so only 84 of 375 cases survive the domain guard. No
number in the manuscript comes from it. It is kept because it is the reason the
authority question is settled by solving the flow rather than by correcting a
surrogate.

## Citing this

Please cite the paper:

> A. Çakanel, Lift-constrained trailing-edge noise control of a wing section:
> achievable authority and the role of aeroacoustic model form, *Aerospace
> Science and Technology* (2026), accepted.

This repository is archived on Zenodo. Cite the paper for the work; cite the
archive only if you are citing the code or data specifically:

> A. Çakanel, Lift-constrained trailing-edge noise control of a wing section:
> code and data, Zenodo, https://doi.org/10.5281/zenodo.22924822

That is the *concept* DOI: it always resolves to the newest release. Each
release also has its own DOI if you need to pin an exact version.

## Licence

Released under the MIT Licence — see `LICENSE`. The NASA airfoil self-noise
measurements in `NASA_selfnoise.csv` are from NASA Reference Publication 1218
(Brooks, Pope & Marcolini, 1989) and are in the public domain.
