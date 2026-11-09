# Fixed spline-block dictionary on the viscous shock tube

This experiment transfers the dictionary enrichment from the single analytical
traveling shock to the transient, tanh-smoothed Sod shock tube. It uses several
simultaneous blocks and no prescribed front positions, widths or trajectories.
The analytical traveling-shock speedup is not evidence of a speedup here.

**Result:** the larger tube configuration reaches 5.07e-7 discrepancy with
at most 361 coefficients per field, but needs 273.71 seconds. FC-Gram reaches
5.51e-10 in 15.62 seconds. This multiple-block prototype does not reproduce
the analytical single-shock advantage and has not met the 1e-8 target here.

## Problem and reference

The domain is [0,1], final time 0.1, gamma 1.4, constant dynamic viscosity
1e-4, Prandtl number 0.72 and initial smoothing width 0.01. Thermal conductivity
is mu gamma / ((gamma - 1) Pr). The equations, initial condition and rigid,
insulated walls match the [FC tube experiment](fc_shock_tube_comparison.md).
Density and energy evolve at the walls; wall momentum and mass/energy fluxes
are zero. No filter, artificial viscosity or finite-volume blend is used.

Errors below mean maximum absolute discrepancies over density, velocity,
pressure and temperature on 64,001 common points against the stored 6119-point
FC solution. This is a numerical reference, not an analytical solution or
certified error bound. The existing 4071-to-6119 comparison is 5.51e-10, and
halving the 4071-point time step changes its solution by 5.39e-10.

## Extension from the analytical experiment

For conservative state U, the enriched trial space is

```text
V_A = V_BSPF + span{ step_j - x, compact B-splines_j : j in A }.
M_A dU_A/dt = D_A^T W F(U_A, D_A U_A),
F = (m, m u + p - tau, u(E+p) - u tau - kappa T_x),
tau = (4/3) mu u_x.
```

The momentum equation uses the mass-orthogonal constraint enforcing zero wall
momentum. The weak boundary mass and energy fluxes are zero. The unconstrained
mass inverse is used for density and energy, unlike the stationary endpoint
treatment in the analytical traveling-shock experiment. RK4 uses the existing
physical advective/diffusive stability estimate, including the enriched
stiffness radius. A constant correction preserves the previous numerical total
mass and energy during each remap; it does not reset them to reference data.

The uniform BSPF background is fixed. The candidate block labels are fixed
before evolution: centers are multiples of 1/1024 and scales are selected from
a specified dyadic list. Each block contains a smooth spline step and compact
degree-13 B-splines with 38 grading intervals and core padding 1. The common
shape is predetermined. Only fully interior blocks are eligible. Changing the
active labels changes basis functions and integration points, not the physical
background mesh. The current small 1D operators are compiled as dense matrices;
this is not a scalable 3D implementation.

Every 0.001 time units, a selector fits the concatenated target

```text
Y = [ U_h, 0.001 R(U_h) ],
R(U_h) = -d_x F(U_h, d_x U_h).
```

The strong rate R is evaluated analytically from the numerical U, U_x and U_xx,
including the physical viscous and heat-conduction terms. No reference solution
is supplied. Using the state alone would always accept the old space because
U_h already lies in it; the rate columns test whether it can represent the next
direction of evolution. This is still a heuristic selection criterion, not a
certified evolution-error estimator.

Selection first attempts to retain the current space. If that fit fails, it
greedily rebuilds a block set: step-signature correlations screen one candidate
location per scale, full-block fits choose among those candidates, and the
process repeats up to the block budget. The check uses 2048 fixed Gauss probes.
Near-dependent enrichment directions are removed by the existing Gram
eigenvalue threshold. Failed tolerance checks at the block limit are recorded
as cap hits and evolution continues, so the requested tolerance must not be
read as an achieved error bound. Global rebuilding and step-only screening
are limitations of this first multiple-block selection strategy.

## Full-time results

| Configuration | Peak coefficients / physical samples per field | Maximum discrepancy | Seconds | Block-cap hits |
|---|---:|---:|---:|---:|
| BSPF 65, up to 3 blocks, fit tolerance 1e-6 | 254 | 3.64862e-4 | 256.12* | 64 / 100 |
| BSPF 129, up to 4 blocks, fit tolerance 1e-7 | 361 | 5.07303e-7 | 273.71 | 32 / 100 |
| FC-Gram 999, FFT 1024 | 999 | 3.40886e-4 | 0.471 | — |
| FC-Gram 2023, FFT 2048 | 2023 | 9.00584e-7 | 2.706 | — |
| FC-Gram 4071, FFT 4096 | 4071 | 5.50795e-10 | 15.623 | — |

The 65-background dictionary includes scales 1/32 through 1/512. The
129-background run uses 1/32 through 1/256; this removes the finest candidates
that were expensive in the first trial. Background size, block budget,
tolerance and scale list all change, so these are two screening configurations,
not a controlled convergence-rate study. Neither BSPF result qualifies for a
time-to-1e-8 comparison. Both the larger BSPF run and FC 2023 pass 1e-6 on this
reference comparison; their measured times differ by about 101 times. This is
a comparison of these particular implementations, not optimal method costs.

The larger BSPF run has component discrepancies (rho,u,p,T) of
(2.249e-7, 1.194e-7, 6.529e-8, 5.073e-7). It uses 8959 time steps, no rejected
stages, and up to 3520 integration points despite its 361 coefficients. Its
minimum density and pressure are 0.1249999646 and 0.0999999670. Relative final
mass and energy drifts are -2.18e-14 and -4.44e-16, measured with the Galerkin
quadrature. These are observed balances and positivity, not general stability
or monotonicity guarantees.

The largest temperature discrepancy occurs at x = 0.5973125, in the contact
region, and the largest density discrepancy at x = 0.578390625. Velocity's
largest discrepancy occurs at x = 0.739046875 and pressure's at the left wall,
so small global oscillations also remain. The final profile plot alone hides
these differences; the logarithmic error panel makes them visible.

The first configuration's maximum remap change is 2.72e-4; the larger one's
is 1.92e-7 across 50 remaps. The maximum requested state/scaled-rate residual in the larger run
is 1.22e-6, above its nominal 1e-7 target. Continuing at the block cap therefore
matters even though the final discrepancy happens to pass 1e-6.

The 129-background runtime consists of 1.66 seconds initialization, 134.88
seconds selection/remapping, and 137.09 seconds integration, plus final output
and bookkeeping. The numerical state is compact, but the current greedy
algorithm repeatedly builds and tests full overlapping block spaces, and each
explicit step applies dense operators on thousands of integration points.
FC 4071 uses 10083 steps, close to the larger BSPF run's 8959, yet its entire
solve is faster than either BSPF cost component. The failed transfer cannot
be explained by time-step count alone.

All reported times include fresh setup, evolution and dense output and exclude
imports, one-time Numba preparation, file writing and subsequent reference
comparison. The larger BSPF and fresh FC runs were serial with single-thread
BLAS. These are individual warm measurements, not repeated medians.
*The first BSPF run overlapped the validation suite and a brief profiler
sample; its timing is exploratory and is not used for controlled speed ratios.*

The experiment supports automatic representation of this multiple-wave case
to approximately 1e-6 with a small coefficient count. It does not establish an
advantage over FC-Gram. The next useful work is to retain compatible active
blocks and update their operators incrementally, improve block screening
beyond a single step signature, and control the state/rate approximation
budget. Simply increasing the block count makes this dense prototype more
expensive. None of these prospective changes is credited as a measured gain.

![Profiles, discrepancies and adaptive budget](../../benchmarks/shock_tube/dictionary/comparison.png)

[Machine-readable comparison](../../benchmarks/shock_tube/dictionary/summary.json)
and [PDF figure](../../benchmarks/shock_tube/dictionary/comparison.pdf).

## Validation and reproduction

The regression checks verify the strong rate against an independently
differenced physical flux, spline derivatives, equilibrium preservation,
evolving wall density, zero wall momentum, mass/energy balance and a short
simulation with the front detector disabled. The broader suite passed 192
tests, excluding the legacy accuracy/parity modules requiring unavailable
`pybspf` and the long-running demonstration module.

```sh
export PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 bench/bench_dictionary_shock_tube.py --tag n65_tol6
python3 bench/bench_dictionary_shock_tube.py --n 129 --max-blocks 4 --tolerance 1e-7 --widths .03125 .015625 .0078125 .00390625 --tag n129_b4_tol7
python3 bench/bench_fc_shock_tube.py --n 999 2023 4071 --tag _dictionary
python3 bench/summarize_dictionary_shock_tube.py
MPLCONFIGDIR=/tmp/bspf-mpl python3 bench/plot_dictionary_shock_tube.py n65_tol6 n129_b4_tol7
python3 -m pytest tests/python --ignore=tests/python/test_accuracy.py --ignore=tests/python/test_parity.py --ignore=tests/python/test_examples.py -q
```

- [Solver](../../examples/pde/dictionary_shock_tube.py)
- [Benchmark driver](../../bench/bench_dictionary_shock_tube.py)
- [Regression tests](../../tests/python/test_dictionary_shock_tube.py)
- [Results directory](../../benchmarks/shock_tube/dictionary)
