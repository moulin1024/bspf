# NekRS CUDA Poisson comparison

Current Gaussian MMS campaign: [dependencies, builds, validation and reproduction](../REPRODUCING.md). Historical measurements below retain their original configurations.

This case solves the same scalar MMS as `../mfem_poisson`: `phi = u_x`,
`-Laplacian(phi) = f`, with analytic nonzero Dirichlet values on all six
nonperiodic faces of `CurvedBox`. It reads the existing 1,856-mode dataset
`../evidence/nonperiodic_curved_shell32/modes.npz`; it does not interpolate the
saved HDF5 grid or substitute a periodic field.

`poisson.udf` applies the exact Q2 physical mapping to Nek's reference GLL
nodes, then constructs a standalone scalar elliptic solver after geometry
updates. The initial vector contains the exact boundary values and zero
interior values; Nek's solver forms the lifted residual and preserves the
boundary values. All repetitions reset that vector. The diffusion operator,
polynomial multigrid, and HYPRE coarse solve run on CUDA.

The degree is 8 (9 local nodes per axis). For `e` elements per axis,
unique global DOFs are `(8*e+1)^3` and unconstrained DOFs are `(8*e-1)^3`.
The harness verifies both using Nek's global interface-node IDs, counting zero-ID unshared interior nodes individually. It also checks
physical coordinates, tensor node ordering, exact boundary values, and an
independently assembled residual. Physical L2 errors use the same independent
CuPy tensor interpolation/integration code as the MFEM campaign, at two Gauss
orders. Off-node native C++ interpolation probes check the exported field.
A polynomial guard uses `phi = 1 + x² + y² + z²`, `f = -6`.

## Build and run on this machine

```bash
bash bench/turbulence_3d/nekrs_poisson/build.sh
export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/math_libs/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/lib64
python bench/turbulence_3d/nekrs_poisson/run.py --elements 2 --polynomial
python bench/turbulence_3d/nekrs_poisson/run.py --elements 2 4 8 12 16
python bench/turbulence_3d/nekrs_poisson/compare.py
python bench/turbulence_3d/nekrs_poisson/report.py
```

GPU access is required. CuPy uses the existing CUDA 12 runtime; the native
NekRS subprocess gets an isolated CUDA 13.2/OpenMPI environment. Builds,
installation, and OCCA kernel cache live under `/tmp/bspf-nekrs-*`. Build
requires network access for the pinned source and Nek5000 dependencies.
The script pins NekRS v26.0, targets SM86, and adds two explicit Thrust
includes required by bundled HYPRE with CUDA 13. It also fixes a missing
SEMFEM kernel registration when SEMFEM is used only on the p-MG coarse grid. It disables fast-math flags
for OCCA kernels, unused backends, ADIOS, and the separate FP32 executable.
The main executable and outer solve use FP64; stock NekRS uses FP32 for its
HYPRE preconditioner. Build provenance and patches accompany results.

## Interpretation of the comparison

Matching DOFs, geometry, data, and PDE does **not** imply identical matrices.
NekRS uses collocated 9-point GLL quadrature; the saved MFEM run uses
11-point Gauss quadrature. Their preconditioners also differ: NekRS uses
p-multigrid with GPU BoomerAMG on its coarse SEMFEM grid, while MFEM uses
LORSolver with FP64 GPU BoomerAMG on the full low-order refined grid.

The plot separates linear solve and full warm RHS/boundary/solve wall time.
NekRS's RHS weighting is on GPU; the saved MFEM p8 RHS assembly is on CPU.
Both exclude source sampling, file I/O, error integration, and setup from
warm timings. Each records a first solve and three reset warm repetitions;
plots use medians. Preconditioner construction times are saved separately.
NekRS launch/JIT time is not comparable to a warm linear solve.
`report.json` also records each native initialization time, including
NekRS kernel selection/tuning, separately from elliptic setup and solves.
The independent residual is normalized by the assembled forcing on free
DOFs; the Krylov tolerance uses the initial lifted residual.

Results are under `../evidence/nekrs_rtx3070/`. The initial cap was 16 elements
per axis. The requested 145-node extension raises the allowed maximum to 18
elements per axis; run it with `--elements 18`.

The one-element Dirichlet case passes geometry/DOF checks but stock NekRS
v26.0 fails in `ogsSetup` on an empty masked gather structure. Its failure
log is retained under `e1_p8_polynomial`; the default sweep begins at 2³
elements and compares only matching completed MFEM meshes.

The successful 2³-element polynomial guard has relative physical L2 error
4.44e-13, exact boundary values, and off-node interpolation agreement to
2.7e-15. The residual normalized by the small polynomial forcing is
1.72e-10; the Krylov residual reduction is 7.9e-13, consistent with its
1e-12 initial-residual tolerance. Turbulence residuals are recorded per case.

## RTX 3070 results

Completed matching meshes: 2³, 4³, 8³, 12³, 16³ elements, all p=8.
At 16³ both solvers have 2,146,689 unique DOFs (2,048,383 free).

| Metric at 16³ | NekRS p-MG/GPU AMG | MFEM LOR/GPU AMG |
|---|---:|---:|
| Relative physical L2 error | 1.4417e-5 | 1.3859e-5 |
| Warm linear solve, median | 0.12130 s | 0.78387 s |
| Warm RHS/boundary/solve, median | 0.12174 s | 10.58019 s |
| Iterations | 16 | 40 |
| Elliptic/preconditioner construction | 8.5633 s | 1.4259 s |

NekRS's warm linear solve is 6.46× faster in this configuration, with 4.02%
higher L2 error. The full warm difference also reflects MFEM's host RHS
assembly. This is **not** a first-run speedup: NekRS separately spent 29.37 s
in native initialization/tuning, before elliptic/preconditioner setup.
The last refinement gives observed L2 order 8.41; the final independent
residual is 1.70e-12 and exact Dirichlet values are preserved.

[Accuracy/wall-time figure](../evidence/nekrs_rtx3070/accuracy_walltime.png),
[PDF](../evidence/nekrs_rtx3070/accuracy_walltime.pdf),
[comparison CSV](../evidence/nekrs_rtx3070/comparison.csv),
[full report](../evidence/nekrs_rtx3070/report.json).
