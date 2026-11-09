# Fixed-dictionary BSPF versus FC-Gram at 1e-8

The front-free dictionary experiment beats the retuned local FC-Gram solver on
the 1D analytical Becker shock: 1.359 seconds versus 2.171 seconds by median,
with maximum primitive errors 1.12e-9 and 6.68e-9. This is a 1.598-times speedup
on this benchmark, not a demonstrated 3D speedup or a general superiority claim.

Follow-up: the [multiple-block viscous shock-tube test](dictionary_shock_tube.md)
does not reproduce this advantage. Its larger BSPF configuration reaches
5.07e-7 in 273.71 seconds, versus FC-Gram at 5.51e-10 in 15.62 seconds.

The successful experiment uses one selected block of compatible splines.
It is the 1D position/scale component of the proposed directional-dictionary
approach, not an implementation of a 3D shearlet frame. Selection uses a
state-approximation residual, not a certified PDE-residual error estimator.

## Problem, metric and timing

The full conservative compressible Navier–Stokes equations are evolved with
mu = 0.001, Pr = 0.75, gamma = 1.4, shock speed 0.5 and final time 0.05.
The analytical profile provides initial and open-flow endpoint data. After
initialization, the enrichment selector receives only the numerical field.
There is no exact interior feedback, artificial viscosity, filtering, or FVM
blend. The far-field endpoint values are constant to numerical precision.

The reported error is the largest absolute error over density, velocity,
pressure and temperature at 64,001 common points on [0,1].

| Method | Independent coefficients / physical samples per field | Final error | Median seconds | Range, seven runs |
|---|---:|---:|---:|---:|
| BSPF + fixed spline-block dictionary | 25 + 45 = 70 | 1.12322e-9 | 1.35872 | 1.32128–1.42887 |
| Retuned FC-Gram | 2120 | 6.67993e-9 | 2.17123 | 1.98064–3.08058 |

FC also computes 25 continuation values, giving FFT length 2145. The ratio of
independent coefficients is 30.29, not a measured memory ratio. Both methods
meet the same error threshold; their errors are not identical.

Runs alternate method order, reversing the order on alternating repetitions.
Each timed call includes fresh setup, initialization, selection/adaptation,
integration and dense analytical error assessment. File writing, imports and
one-time JIT preparation are excluded. BLAS thread counts are one; no other
assistant numerical benchmark/test job ran concurrently. FC timings have
visible variation, so the result should be treated as a local measurement.
Every measured BSPF run was faster than every measured FC run in this batch.

## Fixed dictionary and automatic selection

The candidate dictionary is fixed before examining the initial data:

- Candidate positions are integer multiples of 1/512.
- Candidate scales are 1/64, 1/128 and 1/256.
- Blocks whose entire spline support is not interior are excluded.
- There are 1295 eligible blocks. Each contains a spline step plus compact
  degree-13 B-splines on a predetermined graded knot pattern.
- The selected configuration has 38 grading intervals, core scale 0.25,
  support width 8 and core padding 0.5, as in the earlier spline construction.

Only the active block's full Galerkin operators are built. The selection
matrix stores 1295 step signatures at 512 quadrature samples; the work of
constructing and applying this matrix is included in the timings.

For each step signature s_j, subtract its weighted projection onto the
homogeneous BSPF background to obtain h_j. The screening score is

```text
score_j^2 = sum_fields |h_j^T W (U - endpoint_lift)|^2 / (h_j^T W h_j).
```

This is a normalized approximation-residual correlation. It uses neither
gradient peaks nor a fitted shock location, width, speed or trajectory.
The best-scoring location is screened at each prescribed scale. Initially,
the algorithm chooses the coarsest scale that passes a full-block projection
check. It rejected scale 1/64 with conservative fitting discrepancy 6.51e-5,
then accepted scale 1/128 with discrepancy 2.13e-9.

Every 0.0025 time units, selection is repeated using the computed state.
The current scale is retained unless transfer fails; neighboring dictionary
locations and finer scales are available as fallback candidates. The transfer
check uses an absolute conservative-field tolerance of 1e-7. This tolerance
is a selection safeguard, not a guarantee of 1e-8 PDE error. Actual analytical
errors and refinement controls determine acceptance of the experiment.

Blocks have fixed labels and fixed shapes. Selecting different labels changes
the representation, but no physical shock object or interface is propagated.
The current code activates one block at a time: multiple simultaneous regions
and curved surfaces require an extension of this prototype.

The uniform 25-point BSPF background stays fixed. Only the active enrichment
and its integration rule change. Time integration uses 533 quadrature points,
local Gauss order 7, background order 6, and the existing fused Galerkin/RK4
kernel with a consistently assembled mass inverse. Dense compilation of this
small 1D operator is part of the measured implementation advantage.

## Controls and unsuccessful configurations

| Control | Maximum analytical error | Profile change from selected run |
|---|---:|---:|
| Half CFL | 1.17288e-9 | 1.34e-10 |
| Local quadrature 16, background 12 | 1.12908e-9 | 7.08e-11 |
| 2048 selection samples instead of 512 | 1.48823e-9 | 2.34e-9 |
| Half selection interval | 1.10865e-9 | 6.01e-11 |
| Initial center 0.417 | 2.69371e-9 | Different physical problem |
| Initial center 0.317 | 2.59176e-9 | Different physical problem |
| Reverse speed, -0.5 | 1.90411e-9 | Different physical problem |
| Freeze initial dictionary block | 2.13543e-1 | Ablation fails |

An additional diagnostic samples all four primitive fields on 8193 points at
21 times, including initialization. Its largest sampled error is 3.08428e-9.
The largest conservative remap discrepancy is 7.13e-9. No stage was rejected;
density and pressure remain positive. These are observed diagnostics, not
proofs of positivity, conservation or nonlinear stability.

Dense-grid final conservative integral discrepancies against the analytical
profile are (-3.90e-12, -1.20e-11, -2.78e-11) for BSPF and
(-9.46e-11, -2.55e-10, -5.88e-10) for FC. These are solution-integral errors
in an open-flow problem, not a claim that the domain totals remain constant.

The initial selection interval 0.005 passed the main case but failed a
shifted-case transfer check. Reducing it to 0.0025 resolved that failure.
The earlier 36-interval single-scale trial missed the final threshold at
1.23e-8. A 37-interval trial failed a transfer check. The failed individual-
spline and multilevel-difference experiments remain separate prototypes;
their conditioning and global selection costs motivated block selection.

Freezing the initial block fails badly, so this is not merely an accurate
initial fit followed by successful evolution without adaptation.

## Giving FC the same implementation opportunities

The comparison first reused the earlier 2162-point FC setup. Before accepting
that result, FC received a guarded constant-endpoint fast path, matching the
assumption used by the BSPF compiled kernel, and compiled wave-speed bounds.
The general FC endpoint path remains available and is tested for agreement.
The fast path rejects endpoint data that are not stationary to tolerance.

We then screened 18 nearby FFT lengths. The fastest passing single-run result
was 2120 physical points plus 25 continuation values. Smaller candidates were
retained in the record, including 2075 points at error 1.42e-8 and 2087 points
at 1.06e-8. The search is finite, not a proof of globally optimal FC tuning.
The selected FC result also passes time-step halving (error 6.67697e-9).

The reported 1.598-times result uses the improved FC path, not the earlier,
slower endpoint implementation or its larger headline speedup.

## Meaning for a 3D method

The proposed extension replaces position/scale labels with position, normal
direction, normal thickness and tangential extent. A block can combine a
one-dimensional spline profile along its normal with tangential windows and
polynomials. Selection would act on a fixed directional dictionary, retaining
the FFT background and physical NS fluxes without explicit surface tracking.

That extension is mathematically definable, but its cost and stability are
unestablished. The current implementation does not solve the problems of
multiple overlapping blocks, orientation search, curved shock intersections,
boundary-attached layers or scalable 3D mass/projection operations. In
particular, the dense 1D operator compilation cannot simply be scaled to 3D.
The present result establishes a useful 1D baseline and a successful discrete
dictionary-selection mechanism, not completion of the 3D research program.

## Artifacts and reproduction

- [Timing records and controls](../../benchmarks/traveling_shock/dictionary_fc_final/summary.json)
- [Comparison plot](../../benchmarks/traveling_shock/dictionary_fc_final/comparison.png)
- [PDF plot](../../benchmarks/traveling_shock/dictionary_fc_final/comparison.pdf)
- [FC tuning records](../../benchmarks/traveling_shock/dictionary_splines/fc_fast_search.json)
- [Dictionary solver](../../examples/pde/dictionary_spline_shock.py)
- [Benchmark driver](../../bench/compare_dictionary_fc.py)
- [Dictionary regression tests](../../tests/python/test_dictionary_spline_shock.py)

```sh
export PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 bench/compare_dictionary_fc.py --repeats 7 --fc-n 2120 --interval .0025 --quadrature 7 --outer-quadrature 6 --output benchmarks/traveling_shock/dictionary_fc_final
MPLCONFIGDIR=/tmp/bspf-mpl python3 bench/plot_dictionary_fc.py
python3 -m pytest tests/python --ignore=tests/python/test_accuracy.py --ignore=tests/python/test_parity.py --ignore=tests/python/test_examples.py -q
```

The test command excludes the two legacy modules requiring unavailable
`pybspf` and the long-running example demonstration module. New checks cover
the dictionary's fixed labels and background orthogonality, isolation of
analytical interior data after initialization, absence of detector calls,
and equivalence of the FC constant-boundary and general paths.

Validation: 188 tests passed. The generated comparison plot was visually
checked, local artifact links verified, and `git diff --check` passed.
