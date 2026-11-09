# Viscous shock comparison with enriched spectral Galerkin and Fourier continuation

Latest follow-up: [graded physical-spline enrichment at 1e-8](graded_spline_fc_1e8.md) reaches 2.93e-9 in 1.048 seconds median, versus optimized FC's 5.48e-9 in 1.758 seconds. The original configuration and measurements below remain valid; the new representation and execution path change the runtime ranking for this benchmark.

Follow-up: the [quadrature and cost investigation](bspf_fc_cost_gap.md) reduces the BSPF run from 89.6 to 21.6 seconds at 7.08e-7 error using Gauss order 12. The original measurements below remain recorded, but the gap to the best configuration in that follow-up is 8.6 times, not 35.7 times. FC still wins runtime on this benchmark.

This experiment adds two literature-based baselines to the existing compressible Navier–Stokes Becker traveling-shock study. All runs solve the full conservative mass, momentum and energy equations with physical viscosity and heat conduction: mu = 0.001, Pr = 0.75, gamma = 1.4, compression ratio 2, speed 0.5 and final time 0.05. The exact solution is used for initialization, endpoint data and error evaluation, not interior forcing.

## What was implemented

**Legendre ESG-I adaptation.** The base is a global polynomial Galerkin space with two endpoint lifts and homogeneous Shen functions P_j minus P_(j+2). It uses the enriched-space block formulation of Chen–Shen ESG-I, with an L2-conditioned base and Schur mass inversion. For a controlled comparison, it receives the same compact degree-13 spline dictionary, spacing h/4, patch halfwidth 12h, computed-solution detector and reconditioning policy as enriched BSPF. Chen–Shen's published applications concern weakly singular problems; this is our NS adaptation of their framework, not their published shock solver. ESG-II's coefficient-decay approximation is not implemented. [Chen and Shen, Section 2](https://www.math.purdue.edu/~shen7/pub/She.S19.pdf)

**FC-Gram.** The physical grid is uniform. Each derivative extends the data with published continuation matrices, performs an FFT, multiplies by the wave number and restricts the inverse transform to the physical grid. The numeric matrices have five matching points and 25 continuation points. The strong conservative flux divergence includes physical viscous stress and thermal conduction. RK4 and exact endpoint values are used. The default derivative filter is exp[-16 log(10) (k/k_Nyquist)^100]; an unfiltered control is also run. This implements FC-Gram spatial differentiation in our NS driver, not the full multidomain Albin–Bruno code or its original time integrator. [Albin and Bruno](https://www.math.k-state.edu/~albin/pubs/albin_bruno_2011_sfc.pdf), [published matrix construction](https://github.com/neuraloperator/FC_PINO/tree/main/FC_Gram_Construction).

The original oscarbruno/FC data files store MATLAB symbolic objects. The usable numeric matrices were obtained from the FC-PINO repository's FC_Gram_Construction directory, attributed there to Daniel Leibovici and Valentin Duruisseaux. They are continuation data only: no neural network is used in these simulations.

## Data provenance and reproduction

[fetch_fc_data.py](../../bench/fetch_fc_data.py) downloads the external numeric matrix and verifies SHA-256:

`0fe304836d87dadf0b09d452ea97445ceefeb46adcc4b173edaae09a1fe7668c`

The data lives under the ignored benchmark directory; it is not silently bundled as original project source. [prior_art_traveling_shock.py](../../examples/pde/prior_art_traveling_shock.py) contains both new baselines, and [compare_prior_art.py](../../bench/compare_prior_art.py) runs individual cases. The existing solver's default remains BSPF; a base-factory parameter enables the controlled polynomial comparison.

```sh
python bench/fetch_fc_data.py
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python bench/compare_prior_art.py --method fc --n 2049
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python bench/compare_prior_art.py --method esg --n 129
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python bench/compare_prior_art.py --method bspf --n 193
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python bench/compare_prior_art.py --method bspf --n 417 --quadrature 64
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python bench/compare_prior_art.py --method fc --n 1793
MPLCONFIGDIR=/tmp/bspf-mpl python bench/plot_prior_art_comparison.py
```

For FC, use `--cfl .35 --tag _half_cfl` for the time-step control and `--filter-order 0 --tag _unfiltered` for the no-filter control. For Galerkin quadrature sensitivity, use `--quadrature 64 --tag _double_quadrature`.

## Measurement protocol

The error is the maximum absolute error over density, velocity, pressure and temperature on 64,001 common physical-domain points. Galerkin solutions are evaluated directly. FC output is Fourier-resampled to at least 262,144 extended-domain samples and then interpolated to the common probe grid; its physical-grid values remain the evolved unknowns. Additional continuation points are disclosed separately from physical samples.

Each reported timing is a single serial end-to-end measurement, including operator setup, initialization, integration and final error evaluation, excluding file output and matrix download. There are no repeated-run uncertainty estimates. Separate standalone test processes are kept out of the timing sweeps. Earlier WENO/DG measurements remain historical reference results rather than new measurements in this experiment.

The polynomial implementation uses dense evaluation and projection. Its wall time therefore combines the polynomial explicit stability restriction and an implementation-specific operator cost. It is not evidence that every optimized ESG implementation, or an implicit one, would take the same time. Similarly, FC's published filtering is disclosed rather than silently forcing it to use BSPF's stabilization choices.

## Results and limits

The numerical table and sensitivity measurements are generated from completed files in [comparison.json](../../benchmarks/traveling_shock/prior_art/comparison.json). The [comparison plot](../../benchmarks/traveling_shock/prior_art/comparison.png) and [PDF](../../benchmarks/traveling_shock/prior_art/comparison.pdf) show error against coefficient count and wall time.

At equal base size 129, the Legendre adaptation finishes with 231 coefficients, error 0.00736237 and 6,799 RK4 steps in 47.36 seconds. Enriched BSPF finishes with 223 coefficients, error 0.00731294 and 270 steps in 2.53 seconds. The raw spline selection rules are identical, but the retained rank can differ because the global spaces differ. The measured time ratio is about 18.7; the step ratio is about 25.2. This supports an advantage over this explicit global-polynomial implementation, not an advantage over every ESG method.

The completed Legendre sweep uses base sizes 65, 97 and 129. An exploratory 193-base run was stopped after more than 5,500 steps, before final time, because of its high cost; it is excluded from the completed-result table and no final error or runtime is extrapolated. No Legendre result at the 1e-6 threshold has been established.

The first FC sweep already reverses the earlier runtime implication: FC at 2049 physical points reaches 2.78e-8 error in approximately 3.20 seconds. Halving CFL leaves its peak error unchanged to the displayed precision, and disabling the derivative filter gives 1.96e-8. Thus its accuracy in this benchmark is not explained by artificial shock broadening from that filter. The 4097-point result reaches about 3.41e-9; increasing resolution does not establish unrestricted convergence below the continuation/roundoff floor.

At 1537 FC points, error is 1.40e-6, slightly above the earlier 1e-6 target. Passing and failing resolutions must both be shown. We do not infer a globally optimal FC resolution from this finite sweep.

The fresh same-problem threshold comparison is:

| Method | Coefficients or physical samples per field | Additional FC samples | Peak error | Seconds |
|---|---:|---:|---:|---:|
| Enriched BSPF | 530 | 0 | 7.10060e-7 | 89.614 |
| FC-Gram | 1793 | 25 | 2.08013e-7 | 2.510 |

Both pass 1e-6; the errors are not identical. FC uses about 3.38 times as many independent physical samples but is about 35.7 times faster in these measurements. The extra 25 continuation values are computed from the boundary data, not independently evolved unknowns. Coefficient counts do not establish a memory advantage. The BSPF threshold run uses quadrature 64, whereas the lower-resolution BSPF/Legendre sweep uses 32.

Therefore the demonstrated edge is over this explicit enriched polynomial baseline, primarily through fewer time steps. A runtime edge over FC is contradicted by this benchmark. The actual scientific positioning must retain both findings.

The sensitivity checks compare full primitive profiles on the common probe grid:

| Control | Maximum change from baseline |
|---|---:|
| FC 1793, half time step | 1.60e-10 |
| FC 2049, half time step | 6.08e-11 |
| FC 2049, no derivative filter | 2.97e-8 |
| Legendre 129, double quadrature | 8.93e-9 |

The unfiltered FC control has a smaller exact-solution error, 1.96e-8, despite its profile difference from the filtered run. Doubling Legendre quadrature leaves its 0.00736 error essentially unchanged. These controls do not explain away FC's measured advantage.

The comparison establishes performance of these particular implementations on a single smooth viscous front. It does not establish a ranking for interacting shocks, boundary layers, inviscid discontinuities or complex geometry. In particular, any conclusion that enriched BSPF has a general runtime edge over Fourier-continuation prior art must be withdrawn for this benchmark if the present measurements are retained.

## Validation

The focused Python regression suite passes **171 tests**, including polynomial mass fitting and differentiation, enriched Legendre evolution, FC differentiation convergence and FC traveling-shock refinement. The command excludes the two legacy modules requiring unavailable `pybspf` and the long-running demonstration module:

```sh
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -m pytest tests/python --ignore=tests/python/test_accuracy.py --ignore=tests/python/test_parity.py --ignore=tests/python/test_examples.py -q
```

FC tests used the downloaded, checksum-verified numeric matrices; without those data they explicitly skip. The degree-24 polynomial mass-fit derivative test allows 3e-10 absolute roundoff near endpoints: the measured field-fit error is 2.91e-13 and differentiation amplifies it to 1.04e-10. A separate exactly represented constant verifies a zero derivative to 1e-14. The generated plot was visually checked and the report's local artifact links were verified.
