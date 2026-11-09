# Closing the BSPF–FC cost gap

Completed follow-up: [graded physical B-splines at 1e-8](graded_spline_fc_1e8.md) close the gap and reach a 1.68-times median speedup over an equivalently tuned local FC baseline. The measurements below document the earlier quadrature-only stage.

Follow-up to the [prior-art solver comparison](prior_art_solver_comparison.md), 7 November 2026. These measurements use the same smooth Becker shock, mu = 0.001, final time 0.05, CFL 0.7, 417 BSPF base coefficients and the same adaptive degree-13 spline enrichment. Peak error is measured over all four primitive variables at 64,001 points. No additional damping or solution mesh refinement is introduced.

## First measured improvement

The original BSPF comparison used composite Gauss order 64 in every base interval. Reducing this order changes the quadrature approximation, not the trial dictionary, detector or time integrator. It must therefore be checked against the exact solution rather than assumed safe.

| Gauss order | End-to-end seconds | Steps | Worst primitive error | Passes 1e-6 |
|---|---:|---:|---:|---|
| 64, original comparison | 89.614 | 1776 | 7.10060e-7 | Yes |
| 32 | 45.058 | 1776 | 7.10040e-7 | Yes |
| 16 | 25.474 | 1776 | 7.09050e-7 | Yes |
| 12 | 21.642 | 1776 | 7.08127e-7 | Yes |
| 8 | 17.044 | 1905 | 7.48518e-6 | No |

Order 12 is 4.14 times faster than the original configuration and retains the target accuracy in this test. FC-Gram's recorded 2.510-second run is still 8.62 times faster. The original 35.7 ratio should not be used as the gap to the best tested BSPF quadrature configuration. These are individual serial measurements, not repeated timing distributions. Lower order has not been validated for every other problem, so solver defaults are unchanged.

The full-run summaries are in [fc_gap](../../benchmarks/traveling_shock/fc_gap). Reproduce a full run with, for example:

```sh
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python bench/compare_prior_art.py --method bspf --n 417 --quadrature 12 --tag _q12
```

## What the code and profile show

BSPF already uses fewer RK4 steps than FC (1776 versus 2023). The gap is predominantly cost per step and setup/diagnostics, not the explicit time-step restriction. Composite Gauss order 64 entails 26,624 flux samples for 417 base unknowns; order 12 uses 4992 samples. FC uses 1793 physical samples and 25 computed continuation values. The native base mass correction rank is 15 at both Gauss orders.

The native BSPF `evaluate` and `project` kernels loop over Gauss offsets and execute a separate FFT for each offset. They are already compiled C++; simply moving the Python time loop into compiled code cannot remove that quadrature work.

A separate cProfile run to time 0.005 at order 64 takes 11.280 seconds under profiling. Base evaluation accounts for 4.737 seconds and base projection 2.205 seconds. The enriched mass solve takes only 0.088 seconds cumulatively. Dense output evaluation takes 1.620 seconds. These categories are diagnostic, and the short run overweights setup/output relative to a full integration. Profiling times are not substituted into the unprofiled benchmark table.

An order-16 short profile takes 4.697 seconds: base evaluation 1.274 seconds, base projection 0.599 seconds, enriched mass solves 0.088 seconds, and dense output 1.842 seconds. Thus the mass solve remains secondary after reducing quadrature. Profile dumps and text tables are saved alongside the full-run summaries.

## Priority changes

1. **Use the lowest validated quadrature order.** The completed sweep already recovers 4.1 times in wall time. Validate both final error and remapping/conditioning when changing this parameter; order 8 demonstrates that unrestricted reduction fails.
2. **Replace many shifted FFTs with a small number of oversampled uniform FFTs for nonlinear evaluation and projection.** Preserve BSPF trial/test functions and the structured mass inverse. This requires a new, accurate integration treatment for the nonperiodic boundary contribution and enrichment products; simply substituting trapezoidal weights would change the discrete operator without a convergence justification. Use the existing Gauss operator as the reference and test quadrature convergence.
3. **Separate global flux work from compact-support corrections.** With raw spline matrix S and conditioning transform T, write b = c - C T a and z = T a, so U = Phi b + S z. For the total physical flux G(U,U_x), use the exact identity G(U,U_x) = G(U_b,U_b,x) + [G(U_b+S z,U_b,x+S_x z)-G(U_b,U_b,x)]. The bracket vanishes outside raw spline support, even though the residualized enrichment functions are global. Evaluate that correction only on its support and retain global projection of its contribution. This changes quadrature work, not the solution mesh; stability, cancellation and quadrature error still need validation. Savings depend on how much of the domain the enrichment covers.
4. **Reuse transforms and sampled states, and accelerate diagnostics.** Share the base forward transform between value and gradient evaluation, reuse the accepted state's samples in the next CFL estimate, and evaluate uniform diagnostic grids with structured transforms instead of dense basis matrices. These reduce overhead but do not by themselves remove the dominant quadrature factor.

Mass-inverse tuning is a low priority based on the measured profile. An implicit integrator also does not directly address a comparison where BSPF already takes fewer steps. The remaining speedup is an engineering and quadrature research target, not an established performance claim. FC remains a strong baseline for this smooth, single-front, rectangular-domain problem.
