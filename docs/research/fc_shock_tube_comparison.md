# FC-Gram on the viscous shock tube

FC-Gram transfers to the full transient shock tube without front detection or
front-specific enrichment. At 4071 physical samples its solution differs from
the 6119-sample solution by 5.51e-10. Halving its time step changes the result
by 5.39e-10. This supports numerical convergence below 1e-8 for this case;
it does not certify the error against the exact PDE solution.

## Problem and implementation

We evolve the compressible Navier–Stokes equations on [0, 1] to t = 0.1, with
gamma = 1.4, mu = 1e-4, Pr = 0.72, and kappa = mu gamma / ((gamma - 1) Pr).
The initial condition is the existing tanh-smoothed Sod state, width 0.01.
This is a physically viscous, initially smoothed shock tube, not an inviscid
discontinuous Riemann problem.

For U = (rho, m, E), the conservative total flux is

```text
F = (m, m u + p - tau, u(E + p) - u tau - kappa T_x)
u = m/rho, p = (gamma - 1)(E - m u/2), T = p/rho
tau = (4/3) mu u_x
U_t = -D_FC F(U, D_FC U).
```

The implementation uses the existing FC-Gram continuation matrices with five
matching points per end and 25 continuation samples, FFT derivatives, and RK4.
The continuation samples are computed values, not additional independent
unknowns. Derivatives remove a constant before continuation to preserve uniform
states exactly. There is no filter, artificial viscosity, or enrichment.

At each rigid insulated wall, momentum is zero and mass/energy fluxes are zero.
Boundary density and energy evolve; they are not frozen to their initial values.
This is our local FC spatial adaptation, not a reproduction of the original
authors' complete Navier–Stokes solver. Waves remain far from the walls over
the tested interval; reflected-wave and long-time behavior have not been tested.

## Resolution and time-step checks

The discrepancy is the largest absolute difference over rho, u, p and T on
64001 common points, compared with the 6119-point FC result. It combines spatial
and temporal differences and is not a relative error.

| Physical samples | FFT samples | Peak discrepancy | Measured seconds |
|---:|---:|---:|---:|
| 487 | 512 | 1.07718e-2 | 0.203 |
| 999 | 1024 | 3.40886e-4 | 0.456 |
| 2023 | 2048 | 9.00584e-7 | 5.457, 2.370 |
| 4071 | 4096 | 5.50795e-10 | 22.127, 16.143 |
| 6119 | 6144 | Numerical reference | 51.184 |

Timings are individual serial warm runs, including setup, integration and dense
output, excluding imports, Numba warmup and file writing. The two repeated
resolutions show substantial timing variability; these are not medians or a
controlled speedup measurement. In particular, the 2023-point half-CFL run
took 4.644 seconds, less than its original full-CFL run despite twice the steps.

Halving CFL from 1 to 0.5 changes the 2023-point solution by 1.01310e-7 and the
4071-point solution by 5.39003e-10. The latter uses 20165 steps versus 10083 for
the full-CFL run. The 6119-point run uses 22062 steps. Time-step refinement
therefore supports the high-resolution result independently of the spatial
resolution change, though both checks share the same spatial formulation.

At 4071 points, minimum density and pressure remain approximately 0.125 and 0.1.
All runs completed without a rejected stage. Final relative mass and energy
drifts at 4071 points are -5.80e-13 and -4.68e-13. These measured balances are
not a general discrete conservation or positivity theorem.

![FC shock-tube profiles and convergence](../../benchmarks/shock_tube/fc_gram/comparison.png)

## Implication for BSPF comparisons

Both fine FC solutions differ from the old stored 1025-node BSPF reference by
about 2.75e-4, predominantly in velocity. This discrepancy is much larger than
the FC refinement differences. The old reference cannot serve as a certified
1e-8 target; an independently converged reference is still desirable to rule
out common formulation errors in FC self-convergence.

Coarse FC does exhibit ringing, which decreases sharply with resolution in
this smooth viscous case. FC succeeds here through global resolution without
adaptive front representation. The previous 70-coefficient graded BSPF win on
a single traveling shock does not establish a speed advantage on this problem.
The subsequent [multiple-block dictionary test](dictionary_shock_tube.md)
reaches 5.07e-7 with at most 361 BSPF coefficients per field but takes 273.71
seconds. It does not reproduce the single-shock speed advantage or meet 1e-8.
No shock-tube BSPF/FC speedup is claimed.

## Artifacts and reproduction

- [Solver](../../examples/pde/fc_shock_tube.py)
- [Sweep script](../../bench/bench_fc_shock_tube.py)
- [Machine-readable results](../../benchmarks/shock_tube/fc_gram/summary.json)
- [Plot PDF](../../benchmarks/shock_tube/fc_gram/comparison.pdf)
- [Regression tests](../../tests/python/test_fc_shock_tube.py)

The sweep uses the already downloaded FC matrices and the stored shock-tube
snapshot; see `bench/fetch_fc_data.py` if the matrices are absent.

```sh
export PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 bench/bench_fc_shock_tube.py --n 487 999 2023 4071 6119
python3 bench/bench_fc_shock_tube.py --n 2023 4071 --tag _repeat
python3 bench/bench_fc_shock_tube.py --n 2023 4071 --cfl .5 --tag _half_cfl
MPLCONFIGDIR=/tmp/bspf-mpl python3 bench/plot_fc_shock_tube.py
python3 -m pytest tests/python --ignore=tests/python/test_accuracy.py --ignore=tests/python/test_parity.py --ignore=tests/python/test_examples.py -q
```

Validation: 181 tests passed. The three new tests check exact uniform
equilibrium, evolving wall density/energy, and short-time refinement with mass
and energy balance. The command excludes legacy accuracy/parity tests requiring
the unavailable `pybspf` package and the long-running example demonstrations.
