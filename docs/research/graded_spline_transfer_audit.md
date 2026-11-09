# Transfer audit: graded spline BSPF on a shock-tube snapshot

The [1e-8 traveling-shock win](graded_spline_fc_1e8.md) does not establish transfer
to a transient shock tube. The physical B-spline construction and Galerkin
machinery are reusable, but the selected single-front configuration is not
adequate unchanged.

## Executed check

The audit projects the existing 1025-node numerical shock-tube reference at
t = 0.1 into the new graded spline space. This stored case uses mu = 1e-4,
Pr = 0.72 and a tanh-smoothed initial Sod jump of width 0.01. It differs from
the traveling-shock case in viscosity as well as wave structure. This is a
transfer stress test, not a matched-viscosity performance comparison.

The existing detector finds features at x = 0.67425, 0.404 and 0.58875, in
descending sensor strength. The current factory accepts one detected front,
so the test places its 45 enrichment functions at the strongest feature.
The remaining parameters match the graded construction: degree 13, 38 grading
intervals before core insertion, core scale 0.25, support width 8, core padding
0.5, and quadrature order 8.

| BSPF base coefficients | Total coefficients per field | Largest primitive discrepancy from stored snapshot |
|---:|---:|---:|
| 25 | 70 | 6.64311e-2 |
| 65 | 110 | 6.99587e-4 |
| 129 | 174 | 8.79976e-5 |

These are errors of the straightforward quadrature-based L2 fit, assessed at
the stored 2001 probe points. They are not a minimax approximation bound. Cubic
interpolation supplies values on the fitting quadrature. No new PDE evolution
or runtime comparison was performed. The stored reference is not certified
to 1e-8, so these numbers must not be presented as exact-solution errors.
Nevertheless, the large 70-coefficient discrepancy demonstrates that the
successful traveling-shock configuration cannot simply be reused unchanged.

[Machine-readable audit](../../benchmarks/shock_tube/graded_transfer_audit.json)
and [profiles](../../benchmarks/shock_tube/graded_transfer_audit.npz) are saved.
Reproduce with:

```sh
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python bench/audit_graded_shock_transfer.py
```

## What transfers and what remains to be established

The local spline step, compact corrections, consistent quadrature/mass solve,
and physical NS flux kernel are reusable components. A shock tube requires
separate treatment of the shock, contact layer and rarefaction structure,
including their different speeds and widths. The current graded factory
explicitly rejects multiple centers; simply stacking overlapping dictionaries
without conditioning and integration changes is not an implemented solution.

The traveling-shock fast path constrains all conservative endpoint values.
The existing closed shock-tube solver instead constrains wall momentum and
uses natural mass/energy flux conditions. Reusing the common constrained-rate
matrix would silently change that boundary treatment. The time-stepping kernel
can be adapted, but the boundary operators need to match the intended problem.

A credible transfer study should implement multiple detected spline regions,
handle overlapping supports and remapping, retain enough background modes for
the rarefaction, and adapt the knot padding/update frequency to observed front
motion. It should then evolve the actual shock tube, check positivity and
mass/energy balance, establish an independently converged viscous reference,
and compare FC at the same initial smoothing, viscosity, boundaries, error norm
and threshold. The 1.68-times speedup and the 70-coefficient count remain claims
about the traveling-shock benchmark only.

The subsequent [FC shock-tube evolution study](fc_shock_tube_comparison.md)
establishes a numerical FC baseline for this transient problem: 4071 and 6119
physical samples agree within 5.51e-10, with a time-step-halving change of
5.39e-10. Their discrepancy from the stored BSPF snapshot is about 2.75e-4,
so the stored snapshot should not be used as an exact high-accuracy reference.
