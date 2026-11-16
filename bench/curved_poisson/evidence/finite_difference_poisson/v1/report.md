# Native fourth-order finite-difference Poisson baseline

Problem: -Δu=f on the same nonconformal four-spline Coons quadrilateral, exact nonzero Dirichlet data on all four edges. The hard MMS is the unchanged saved seed 20261115 field with shell energy k^(-5/3), shells 4–128.

## Implementation and timing

C++ float64 local fourth-order strong-form mapped Laplacian, with both metric drift terms and the mixed derivative. Five-point centered first/second derivatives; fourth-order one-sided first derivatives and six-point second-derivative closures at the first/last interior rows. Mixed derivatives are successive local first derivatives. No dense global matrix or dense solve.

Right-preconditioned BiCGSTAB uses an FFTW DST-I inverse of a constant-coefficient second-order reference-grid Poisson operator, with mean mapped diagonal coefficients. Cross and drift terms remain in the full operator. Stencil application is O(N), preconditioning O(N log N), storage O(N), for N interior unknowns. Single-thread FFTW; no threaded plan.

Geometry preparation, analytic sampling, FFTW plan construction and preconditioner preparation are outside timing. Boundary lifting, iterative solve, true residual verification and boundary solution assembly are inside timing. All cases start at zero interior correction; exact interior solution is used only for validation. Forcing is sampled analytically at native FD nodes and only interior forcing enters the solve. BSPF and Nektar use direct analytic forcing at their own quadrature nodes. This is the same analytic-input policy, not identical sample counts.

One solve per configuration, no warmups/repeats. BSPF and Nektar timings are reused from the previous campaign; these are individual timing observations, not confidence intervals. Nektar factorization is prepared outside its timer. Gray crosses denote failed audits, including the archived BSPF n=96 point.

## Independent accuracy checks

Off-grid solution reconstruction uses local degree-seven Lagrange interpolation for error integration only; it never enters the FD solve. Relative physical L2 error is independently integrated on Gauss grids of 640 and 768 points per axis for the hard MMS. Agreement must be within 1% plus 1e-11. Smooth qualification uses 100/140-point rules. Native true relative residual must be below 2e-12. The residual check validates the discrete equation; the MMS and refinement test validate consistency.

Smooth qualification: u=sin(3x+0.3) cos(4y-0.2)+0.2x+0.1y; f=25 sin(3x+0.3) cos(4y-0.2).

| Nodes/axis | Relative L2 | Observed order |
|---:|---:|---:|
| 17 | 7.268280e-05 | — |
| 33 | 4.916230e-06 | 3.886 |
| 65 | 3.021237e-07 | 4.024 |
| 129 | 1.870787e-08 | 4.013 |

## Hard turbulence MMS

| Nodes/axis | Unknowns | Solve ms | Iterations | Relative L2 | Status |
|---:|---:|---:|---:|---:|---|
| 65 | 3969 | 4.387 | 19 | 3.569264e+01 | validated |
| 129 | 16129 | 26.987 | 24 | 2.486740e-01 | validated |
| 257 | 65025 | 92.540 | 23 | 1.170104e-02 | validated |
| 513 | 261121 | 397.013 | 25 | 5.964023e-04 | validated |
| 1025 | 1046529 | 2205.602 | 26 | 3.791299e-05 | validated |

## Fastest tested configuration reaching each target

| Relative L2 target | BSPF ms | Nektar++ ms | Native FD4 ms |
|---:|---:|---:|---:|
| 1e-02 | 200.958 | 24.948 | 397.013 |
| 1e-03 | 328.509 | 151.256 | 397.013 |
| 1e-04 | 328.509 | 151.256 | 2205.602 |
| 1e-05 | 328.509 | 151.256 | not reached |
| 1e-06 | 328.509 | 685.890 | not reached |
| 1e-08 | 421.599 | 685.890 | not reached |

“Not reached” means not reached by the tested ladder. The native FD4 solver is an in-house reference, not an established upstream solver or a claim about the best possible finite-difference implementation. Coarse hard-field grids can alias the forcing severely; their large errors remain visible. These runs do not establish universal method superiority.
