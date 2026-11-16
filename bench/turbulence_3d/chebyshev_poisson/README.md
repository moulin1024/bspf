# Global Chebyshev Galerkin CUDA Poisson comparison

Current Gaussian MMS campaign: [dependencies, builds, validation and reproduction](../REPRODUCING.md). Historical measurements below retain their original configurations.

This solver uses one global tensor-product polynomial space P_(n-1) in each
reference direction, represented by cardinal polynomials at Chebyshev-Lobatto
nodes. Thus n=145 means degree 144 in each direction, not degree 8. It is a
pure polynomial reference for the existing Fourier-enriched BSPF space.

The problem is unchanged: `-Delta_X phi=f`, `phi=u_x` of the saved nonperiodic
1,856-mode turbulence MMS, on the curved box with map amplitudes
`(0.08,-0.06,0.07)`. All six faces use exact inhomogeneous Dirichlet samples.
No opposite faces are identified.

## Discretization and shared implementation

`axis.py` constructs the inverse Chebyshev Vandermonde using a DCT-I and
analytically differentiates its Chebyshev coefficients. The nodal representation
provides the boundary trace coordinates required by the native solver. The
physical weak form is the ordinary unweighted Galerkin form

`a(v,u) = integral grad_s(v)^T det(J) J^-1 J^-T grad_s(u) ds`.

Chebyshev polynomials specify the approximation space; this does **not** replace
the volume measure with the singular Chebyshev orthogonality weight. The same
polynomial space could equivalently be expressed in a Legendre basis. This is
Galerkin integration, not strong-form Chebyshev collocation.

The driver reuses `../bspf_poisson/run.py` through an explicit basis selection,
and the same native CUDA executable. Both methods therefore use:

* Full curved metric, including all mixed-gradient terms.
* FP64 matrix-free operator and CG, tolerance 1e-12; independently checked residuals.
* FP32 separable inverse from the generalized 1D stiffness/mass eigensystem.
* Reflection splitting and shared gradient/transpose contractions.
* Gauss-Legendre quadrature q=max(96,ceil(1.5n)); polynomial guard minimum 48.
* One warmup and three measured zero-initial-guess solves, with synchronized
  wall time for RHS projection, boundary lift, solve and recovery. Setup,
  analytic sampling, file I/O and accuracy audits are excluded.
* Independent physical L2 error integration at max(96,q+8) and eight more points
  per direction, using the physical Jacobian determinant.

## Reproduction

Build the existing native backend once with
`bash bench/turbulence_3d/bspf_poisson/build.sh`. On this machine use the CuPy
CUDA 12 environment; the driver selects CUDA 13 libraries for the subprocess:

```sh
export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/math_libs/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/lib64
python -m pytest -q tests/python/test_chebyshev_poisson.py tests/python/test_cuda_poisson3d.py
python bench/turbulence_3d/chebyshev_poisson/run.py --n 17 --polynomial --check
python bench/turbulence_3d/chebyshev_poisson/run.py --n 17 --check
python bench/turbulence_3d/chebyshev_poisson/run.py --n 33 65 97 129 145
python bench/turbulence_3d/fc_poisson/report.py
```

Evidence is saved in `../evidence/chebyshev_cuda3d_rtx3070/nN_qQ/`, including
input bundles, basis coefficients, solutions, setup metadata, binary hashes,
native logs, repeated timings, residuals and error audits. The main comparison
retains accuracy versus DOFs and repeated full-solve wall time versus accuracy.

## Validation

The five CPU tests cover cardinal interpolation, reflection identities,
reproduction and differentiation of independently evaluated Legendre
polynomials through degree n-1, ordinary mass integrals, and the first five
Dirichlet Laplacian eigenvalues. High-order second-derivative tests use a
matrix-norm-scaled roundoff allowance near zero: endpoint derivatives grow as
n^4. The weak solver only uses first derivatives. All five CPU tests and the
18 existing CUDA regression tests pass.

The n=17 curved polynomial guard has physical relative L2 error 2.43e-13.
The n=17 turbulence run also checks the native operator and FP32 preconditioner
against independent NumPy tensor contractions.

## Measured convergence

| Nodes/axis | Polynomial degree | Full warm solve (s) | Relative physical L2 |
|---:|---:|---:|---:|
| 17 | 16 | 0.05318 | 4.287e-01 |
| 33 | 32 | 0.07160 | 1.552e-01 |
| 65 | 64 | 0.14714 | 2.980e-05 |
| 97 | 96 | 0.53894 | 2.064e-12 |
| 129 | 128 | 1.45210 | 2.433e-12 |
| 145 | 144 | 2.11095 | 2.550e-12 |

All turbulence cases have zero nodal boundary error and true residual below 1e-12.
Results above the resolution threshold are limited by the solve tolerance and
floating-point effects; increasing n further does not improve this error floor.

At n=65, increasing operator quadrature from q=98 to q=114 changes the solution
by 7.45e-14 relatively and the measured error by 1.24e-12 relatively.
The 2.98e-5 error therefore persists under overintegration. Results are recorded
in `../evidence/chebyshev_cuda3d_rtx3070/quadrature_validation.json`.
