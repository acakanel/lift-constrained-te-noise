# The Reynolds-averaged correction of the surrogate dataset

`corrected_designspace.py` builds a correction that maps the NeuralFoil
trailing-edge boundary-layer state onto the Reynolds-averaged one, applies it to
the 375-case dataset, and rescans the constant-lift manifold with the corrected
levels. It produces `dataset_corrected.csv` and `designspace_corrected.csv`.

**No number in the manuscript comes from this correction.** Every result uses
either the raw dataset (`dataset.csv`, the surrogate, the ablation, the
design-space spread, the controllers, the processor-in-the-loop package) or the
Reynolds-averaged solutions directly (`rans/`, the authority, the manifold
table). The correction is kept because it was run, because it is why the
authority question is settled in resolved flow rather than on the semi-empirical
chain, and because deleting a negative result to tidy a repository is not a
defensible thing to do. It is not kept because it works. It does not.

Which dataset is in use is set explicitly, never by default. `surrogate.py`
reads `dataset.csv` and the `SPL` column unless `AERO_DATASET` and
`AERO_SPL_COL` say otherwise, and it refuses to start if it is pointed at the
corrected file without also being told to use the corrected column: that
combination trains on the *uncorrected* target inside the corrected file, runs
without error, and reproduces the uncorrected accuracy figure exactly.

## Why it does not work

The correction fits three linear models in \((1, \alpha, \delta_1+\delta_2)\) on
the Reynolds-averaged points available at \(U = 55\) m/s: two on the logarithm of
the displacement-thickness ratio, one on the suction-side shape factor \(H_s\).
The first two are ratios and stay bounded. The third is not.

The decisive number is not any of the fit's own diagnostics. It is the mismatch
between the box the fit was made on and the box it is asked about:

| | \(\alpha\) | \(\delta_1+\delta_2\) |
|---|---|---|
| support of the fit (19 Reynolds-averaged points) | \([-0.1,\ 12]^\circ\) | \([0,\ 10]^\circ\) |
| design box it is applied over (375 cases) | \([0,\ 12]^\circ\) | \([-20,\ +20]^\circ\) |

Every solved case carries non-negative deflections, because those are the cases
the authority study needed; the design box runs to \(-20^\circ\). A linear model
carries no information outside the points it was fitted on, so more than half the
design box is not corrected by this fit, it is asserted about by it. With the
domain guard in place:

| outcome | rows, of 375 |
|---|---|
| carry a corrected level | **84 (22 %)** |
| refused: outside the fitted design box | 195 |
| refused: outside the equivalent-incidence map | 96 |
| corrected \(H_s < 1\), which no boundary layer can have | 72 |

The equivalent incidence is obtained by inverting the clean-section
\(H_s\to\alpha\) map, which is tabulated only over the solved incidences, so a
corrected shape factor outside \([1.49,\ 1.85]\) has nowhere to land.
The domain guard is what makes those refusals visible. `numpy.interp` does not
extrapolate --- it returns the endpoint, silently --- so without the guard the
refused rows do not appear as refusals at all. They appear as a plausible column
of angles piled up on the two endpoints, and a `numpy.clip` to
\([0,25]^\circ\) around the call can never fire, because the map spans
\([0,12]^\circ\) and the interpolation has already clamped inside that.
Extrapolating off the same map instead moves the
fixed-lift level spread at \(U=55\) m/s from 4.0 dB to 18.7 dB and moves the
quietest design point at two of the three conditions. Neither figure is
trustworthy; that they differ by a factor of four is the point.

Two readings are available and both should be stated. The narrow one is that the
correction is under-determined: nineteen points, all on one side of the
deflection axis, cannot support a three-parameter fit extrapolated over a box
four times as wide, and a bounded parameterisation --- \(\log(H_s-1)\), or a fit
on the ratio rather than on \(H_s\) itself --- together with solved cases at
negative deflection would be the repair. The broad one is that the failure is the
same failure the paper reports: the quantity the semi-empirical chain needs is
the separation state at the trailing edge, and neither a fast aerodynamic model
nor a cheap correction of one delivers it over a flapped design box. Solving the
flow is not a refinement of that approach. It is the alternative to it.

## If you want to run it anyway

```
python3 corrected_designspace.py
```

It prints the fit coefficients, their leave-one-out errors, the support of the
fit against the design box, and how many rows each ground of refusal removes,
then writes both CSV files. Read the support table before reading anything else.
