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

## Leonardo optimized CUDA 13 run

Chebyshev uses the same optimized native backend as BSPF: tiled cyclic
transposes, shared gradient contractions, and eight reflection sectors with
fused metric application. Sector eligibility is checked against basis and
metric symmetries; unsupported inputs retain the general operator. The
Chebyshev discretization and direct Gauss-point forcing are unchanged.

Submit the polynomial validation, 17--257 sweep and alternating same-GPU
257-cubed comparison with:

```sh
sbatch bench/turbulence_3d/leonardo/chebyshev_cuda13.slurm
```

This uses `leonardo/env_cuda13.sh`, CUDA 13.0.2 and cuBLAS 13.1.0.3. The
`performant` FP64 emulation strategy allows cuBLAS to choose native kernels;
it does not establish that emulated kernels were used. The FP64 operator and
CG tolerance remain unchanged, with the existing FP32 preconditioner.

The CPU identities now cover 16, 17, 65, 145, 193 and 257 nodes (seven tests
including the spectrum check). Derivative checks allow degree- and
matrix-norm-scaled roundoff for cancellation near zero. Cardinal interpolation,
Legendre reproduction and mass integration retain their original tolerances.

Job **64626778** completed on one Leonardo A100. The fixed-input 257-cubed
comparison alternates three invocations of each executable, each with one
warmup and three measured solves. The pooled median improved from **1.385736 s
to 0.635640 s (2.180x)**; both took 18 CG iterations. The relative solution
difference was 5.44e-14, the operator-probe difference 1.48e-12, and the
preconditioner probe was bitwise identical. Independent physical L2 audits at
394 and 402 Gauss points per axis both gave 3.0681e-12. This compares the frozen
preoptimization executable with the optimized CUDA 13 executable; it does not
isolate a CUDA-version or emulation speedup.

| Nodes/axis | Optimized warm solve (s) | Relative physical L2 |
|---:|---:|---:|
| 17 | 0.024949 | 4.186e-01 |
| 33 | 0.025118 | 1.516e-01 |
| 65 | 0.026847 | 2.910e-05 |
| 97 | 0.045172 | 1.983e-12 |
| 129 | 0.082038 | 2.319e-12 |
| 193 | 0.236040 | 2.683e-12 |
| 257 | 0.635040 | 3.068e-12 |

All seven sweep cases have zero nodal boundary error and true residual below
1e-12. The polynomial guard has L2 error 2.4343e-13 and an independent CPU
operator discrepancy of 1.89e-15. Results and comparison details are in
[`../evidence/chebyshev_cuda13_64626778/`](../evidence/chebyshev_cuda13_64626778/),
with updated PNG/PDF plots in
[`../evidence/comparison_leonardo_to257/`](../evidence/comparison_leonardo_to257/).

## Historical reproduction on RTX 3070

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
