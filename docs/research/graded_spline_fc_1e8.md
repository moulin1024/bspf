# Graded spline BSPF versus FC-Gram at 1e-8

A later [fixed-dictionary experiment](dictionary_spline_fc_1e8.md) replaces
the front detector with discrete block selection and retunes the FC baseline.
It reports a separate seven-pair comparison; do not mix its timings with this
earlier detector-driven experiment.

Transfer check: a subsequent [shock-tube snapshot audit](graded_spline_transfer_audit.md)
finds that the 70-coefficient single-front configuration does not transfer
unchanged. Its discrepancy from the stored lower-viscosity shock-tube profile
is about 0.066. That audit is a projection check, not a new PDE simulation or
a certified 1e-8 reference comparison.

Research result, 7 November 2026. The spline-focused implementation closes the earlier runtime gap and beats the optimized local FC-Gram baseline on the smooth 1D Becker traveling shock. This is a measured result for the implementations and problem below, not a general superiority or novelty claim.

## Matched-threshold result

Both methods solve the full conservative compressible Navier–Stokes equations with physical viscosity and heat conduction: mu = 0.001, Pr = 0.75, gamma = 1.4, compression ratio 2, speed 0.5 and final time 0.05. Exact data supplies initialization, endpoint conditions and error assessment; there is no exact-solution forcing in the interior.

The error is the largest absolute error among density, velocity, pressure and temperature on 64,001 common points over [0,1].

| Method | Independent coefficients / physical samples per field | Computed FC continuation values | Peak error | Median seconds | Range, five runs |
|---|---:|---:|---:|---:|---:|
| BSPF + graded degree-13 physical B-splines | 25 + 45 = 70 | 0 | 2.92918e-9 | 1.04796 | 1.04190–1.07459 |
| Optimized FC-Gram | 2162 | 25 | 5.48061e-9 | 1.75826 | 1.71077–1.85143 |

BSPF is **1.678 times faster** by median wall time and uses **30.9 times fewer independent coefficients**. Both pass 1e-8; they do not have identical errors. Coefficient counts alone do not establish a memory ratio.

Runs alternate method order, reversing the order on alternating repetitions. Each run includes fresh geometry/operator setup, initialization, all adaptation and integration, and dense error evaluation. File output, imports and one-time JIT preparation are excluded. The recorded cached-session preparation took 0.284 seconds; this is not a cold-compilation benchmark. All timed runs were serial, with BLAS thread counts set to one and no concurrent test suite.

FC also receives the fused physical-flux kernel, CFL tuning to 1.0, a search over FFT-friendly sizes, and an unfiltered control. The final FC comparison uses **no derivative filter**. The 2162 physical points plus 25 continuation points give an FFT length of 2187. The first 4097-point FC result and the earlier CFL-0.7 timings are not used as the competing timing target.

Artifacts: [raw records and timing summary](../../benchmarks/traveling_shock/graded_spline_fc_1e8/summary.json), [plot](../../benchmarks/traveling_shock/graded_spline_fc_1e8/comparison.png), [PDF](../../benchmarks/traveling_shock/graded_spline_fc_1e8/comparison.pdf).

## The spline advantage being used

The representation can be written in unconditioned coordinates as

\[
U_h(x)=\Phi(x)b+\sigma(x;\kappa)a_0+
\sum_{j\in I}B_{j,13}(x;\kappa)a_j.
\]

Here Phi is the original BSPF base on 25 fixed uniform points. The additional functions are ordinary physical-coordinate B-splines on graded knots, not tanh or exact-shock basis functions. The function sigma is a spline step with its linear endpoint lift removed. Its control coefficients are zero on one side and one on the other; consequently its derivative is localized and its endpoint values vanish after lifting. It carries the change between the left and right states, while the compact splines correct the layer shape.

This avoids spending Fourier modes on the thin layer. The final Fourier sample residual, after subtracting the fitted base spline component, has maximum magnitudes approximately 2.60e-9, 5.84e-9 and 5.20e-9 in the three conservative fields. Thus almost all sharp-front structure in this particular example is carried by splines. A pure-spline ablation would be required to establish that the Fourier component is essential here; these measurements do not establish a uniquely BSPF approximation advantage over every other spline method.

The density/pressure-gradient detector uses the numerical solution on the existing 4001-point probe grid. Its smoothing spacing is 1/48. It supplies the center c and width w; neither the exact trajectory nor the exact interior profile is used during adaptation. The grading starts from c + 0.25 w sinh(z), with z spanning plus/minus asinh(32). A uniformly spaced core extending 0.5 w on each side is inserted, so the layer can move between updates without immediately leaving the finely represented region. Outer knots are wider spaced. The selected configuration uses 38 grading intervals before inserting the core, producing 45 enrichment directions including the spline step.

Adaptation occurs every 0.005 time units. The old numerical field is evaluated on the new quadrature and projected into the new space. The largest recorded conservative remap perturbation is 6.63e-10. The uniform solution grid stays fixed: basis knots and integration points adapt, with no local finite-volume/finite-difference solution mesh refinement.

An initially accurate graded basis without the padded core was insufficient: one 56-interval configuration had about 2e-9 initial projection error but approximately 6e-6 final evolution error. The front moved into the more widely spaced knots between updates. This failure motivated the padded core; initial approximation accuracy alone was not accepted as evidence of a successful simulation.

## Integration and implementation changes

Gauss integration is split at enrichment knots and base spline knots. The background uses structured FFT Gauss evaluation; extra integration is restricted to the spline region. The selected local/background orders are both 8, with 632 quadrature points in the final space. A higher-order quadrature control is reported below.

Every changed quadrature receives a consistent base mass matrix, represented for solves by a low-rank interior correction and boundary Schur complement. The existing conditioned enrichment and enriched Schur solve are retained. For the small 70-coefficient space, value/gradient maps and the constrained weak flux-to-rate map are then precompiled after each adaptation. A Numba kernel fuses the physical NS flux and classical RK4 stages. The PDE remains Galerkin NS: there is no finite-volume blend, artificial viscosity, state filtering or extra damping.

The compiled kernel explicitly requires stationary prescribed endpoint values to numerical tolerance. That holds for this far-field traveling-shock benchmark. The general Python RHS continues to support time-dependent endpoint data; the fast kernel rejects unsupported endpoint data rather than silently freezing them.

Dense output evaluates the short Fourier remainder by a chirp-z transform on uniform probe grids, or direct short Fourier sums on nonuniform grids. Physical B-splines are evaluated directly from their combined coefficients, avoiding a large dense diagnostic basis matrix.

## Controls, convergence and ablations

Full-profile changes are measured over all four primitive fields on the common grid:

| Control | Peak exact-solution error | Maximum change from selected run |
|---|---:|---:|
| BSPF, halved CFL | 2.91528e-9 | 6.61e-11 |
| BSPF, local order 16 / background order 12 | 2.93521e-9 | 3.62e-11 |
| FC, halved CFL | 5.43832e-9 | 1.72e-10 |

The final conservative integral errors against the exact profile are approximately 5.26e-13, 5.73e-14 and 9.88e-13. These are open-domain solution-integral errors, not a claim of constant mass or energy in the domain.

The neighboring BSPF resolutions include failures as well as passing cases:

| Grading intervals before core insertion | Final coefficients | Peak error | Seconds, single run |
|---|---:|---:|---:|
| 34 | 64 | 4.61930e-8 | 0.715 |
| 36 | 67 | 1.22893e-8 | 0.905 |
| 38 | 70 | 2.92918e-9 | 1.048 median |
| 40 | 73 | 1.15735e-9 | 1.229 |

The FC search also retains threshold misses: 2135 physical points with filtering disabled give 1.00611e-8, which is a failure, whereas the selected 2162-point case passes. The finite search does not establish a globally optimal configuration for either method.

At the same 70-coefficient spline space and quadrature:

| BSPF execution path | Seconds, single run | Peak error |
|---|---:|---:|
| Structured evaluation/projection and Schur solves in Python driver | 2.922 | 2.93081e-9 |
| Precompiled small operators, Python RK4 | 2.094 | 2.93680e-9 |
| Precompiled operators, fused RK4 | 1.048 median | 2.92918e-9 |

The structured path still takes longer than the tuned FC implementation. The measured win therefore combines the improved spline representation with compilation of the small operators and time loop; it must not be attributed to the approximation space alone. The structured and Python-precompiled profiles differ from the fused result by at most 2.33e-11 and 5.04e-11 respectively.

The mapped Jacobi/tanh experiment used earlier during this investigation is not the winning method in this table. It remains a separate experimental baseline. The reported winner uses graded physical B-splines and a spline step.

## Reproduction and validation

```sh
python bench/fetch_fc_data.py
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python bench/bench_graded_spline_fc.py --repeats 5
MPLCONFIGDIR=/tmp/bspf-mpl python bench/plot_graded_spline_fc.py
```

Implementation: [graded spline space and local quadrature](../../examples/pde/graded_spline_shock.py), [quadrature/mass utilities](../../examples/pde/fft_quadrature_shock.py), [fused Galerkin RK4](../../examples/pde/compiled_galerkin_step.py), [benchmark](../../bench/bench_graded_spline_fc.py).

The focused Python suite passes **178 tests**. New checks cover FFT evaluation against the native BSPF operator, weighted adjointness, quadrature-consistent mass inversion, direct Fourier evaluation, compiled RHS/step equivalence, constant-state remapping between moving spline spaces, spline derivatives and boundary lifts, and rejection of invalid fast-kernel states. The command retains the prior exclusions for two legacy modules requiring unavailable pybspf and the long-running demonstration module:

```sh
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -m pytest tests/python --ignore=tests/python/test_accuracy.py --ignore=tests/python/test_parity.py --ignore=tests/python/test_examples.py -q
```

This result covers one interior smooth viscous front in 1D. Multiple fronts, front interaction, boundary-attached layers, different viscosity regimes and 3D are not established. In particular, dense compilation of a small 1D operator is not a demonstrated scalable 3D algorithm. The global solver defaults are unchanged; the benchmark explicitly selects the new spline construction and execution path.
