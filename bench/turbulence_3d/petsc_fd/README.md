# PETSc CUDA eighth-order finite-difference Poisson benchmark

Current Gaussian MMS campaign: [dependencies, builds, validation and reproduction](../REPRODUCING.md). Historical measurements below retain their original configurations.

This solver uses the same physical, nonperiodic curved box and continuous
plane-wave turbulence MMS as the BSPF, MFEM, and NekRS experiments. It uses
uniform reference coordinates, exact physical Dirichlet data on all six faces,
and interior unknowns only. A grid of n³ total nodes has (n-2)³ unknowns.

For the map x=X(s), let J=dX/ds, G=J^{-1}J^{-T}, and
b_i=-(J^{-1})_{ia} G_{jk} d²X_a/(ds_j ds_k). The full physical operator is

    -Delta_x u = -sum_ij G_ij D_i D_j u - sum_i b_i D_i u.

Diagonal second derivatives use their own FD stencil, rather than applying
D1 twice. First derivatives have nine-point, eighth-order formulas. Second
derivatives have nine-point symmetric interior formulas and ten-point shifted
boundary formulas, exact through degree nine. Mixed derivatives use tensor
products of D1. Rational moment equations generate the weights; conversion to
FP64 occurs after solving them exactly. Geometry derivatives are analytic.
Boundary closures use in-domain points only, without periodic wrapping or
manufactured ghost values. The boundary lifting is subtracted from the RHS.

CUDA kernels apply the operator without assembling its 3D matrix. PETSc
FGMRES runs in FP64 with CUDA vectors. One-sided closures and the strong-form
mapped operator are nonsymmetric; CG is therefore not used. The auxiliary
preconditioner matrix is a seven-point reference-grid Laplacian, scaled by the
mean diagonal metric. It is not an eighth-order approximation and does not
replace the high-order system operator.

Two GPU preconditioners are available:

- `--pc fp32` (default): a PETSc shell PC uses the isolated FP32 Hypre BoomerAMG
  bridge from the MFEM experiment, with GPU FP64/FP32 conversions. Both casts
  and synchronization are included in solve timings. The bridge uses the same
  AMG settings as the MFEM FP32 experiment.
- `--pc hypre`: PETSc's native PCHYPRE BoomerAMG, using FP64 GPU Hypre and a
  `seqaijcusparse` auxiliary matrix. This also validates the PETSc/Hypre build.

The Krylov tolerance is 1e-11, and every solve additionally checks its true
FP64 residual. Timings contain one initial solve plus three measured warm
solves; the report uses the warm median. Setup, MMS sampling and error audits
are separate. This is a single-rank benchmark, not a distributed FD solver.

## Build

The local build uses PETSc 3.26.0, CUDA 13.2 / SM86, Open MPI from NVHPC 26.5,
and the existing FP64 GPU Hypre 3.1.0 installation. PETSc is static because
the existing Hypre archive is not position-independent. That does not change
its GPU execution. Build the MFEM GPU/Hypre and FP32 bridge prerequisites first
if their `/tmp` installations are missing.

```bash
bash bench/turbulence_3d/petsc_fd/build_petsc.sh
bash bench/turbulence_3d/petsc_fd/build_solver.sh
```

PETSc's [CUDA/Hypre installation documentation](https://petsc.org/release/manualpages/PC/PCHYPRE/)
describes native GPU preconditioning with CUDA vectors and cuSPARSE matrices.
The build records `PETSC_HAVE_CUDA`, `PETSC_HAVE_HYPRE`, and
`PETSC_HAVE_HYPRE_DEVICE` in its generated configuration.

## Run and validate

Sampling uses the same CuPy/CUDA 12 environment as the other campaigns;
the Python driver supplies a separate CUDA 13 environment to the native binary.

```bash
export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/math_libs/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/lib64
python -m pytest -q tests/python/test_fd8_poisson.py
python bench/turbulence_3d/petsc_fd/run.py --field polynomial --n 17 --check --pc hypre
python bench/turbulence_3d/petsc_fd/run.py --field polynomial --n 17 --check
python bench/turbulence_3d/petsc_fd/run.py --field smooth --n 17 25 33 49 65
python bench/turbulence_3d/petsc_fd/run.py --field turbulence --n 17 33 65 97 129 --audit
```

The polynomial guard is phi=1+x²+y²+z², f=-6. The smooth convergence field is
sin(2pi x)sin(2pi y)sin(2pi z)+0.2cos(pi(3x+2y-z)), with its analytic forcing.
Turbulence uses the unchanged `nonperiodic_curved_shell32/modes.npz` dataset.

`relative_nodal_l2` is a Jacobian-weighted nodal error with trapezoidal weights.
It is explicitly distinct from the physical L2 error used in the solver
comparison. `--audit` additionally reconstructs the FD field with local
nine-point interpolation and integrates the physical error at two separate
Gauss orders. Only independently audited results should enter a cross-solver
accuracy plot. All boundary values are imposed exactly, and the native output
includes a random-probe application for the independent CPU/GPU check.
