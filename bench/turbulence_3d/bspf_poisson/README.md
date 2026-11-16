# CUDA BSPF Galerkin Poisson in 3D

Current Gaussian MMS campaign: [dependencies, builds, validation and reproduction](../REPRODUCING.md). Historical measurements below retain their original configurations.

Native reusable C++ API: `include/bspf/cuda_poisson.hpp`; implementation:
`src/cuda/poisson3d.cu`. The root CMake CUDA build exposes `bspf_cuda_poisson`
and links it into `bspf_cuda`. The standalone build below avoids unrelated
flow/Python dependencies.

This extends the independent single-span B-spline/Fourier space from the
committed 2D Galerkin Poisson benchmark (43b7ed8), **not** the older fitted
cardinal `Line` space. The harness uses degree 8 plus paired Fourier modes,
period 1.5, and symmetric QR nodes. Degree 8 permits odd sizes 17,33,65,97,129,
matching the previous SEM global DOF counts exactly; the original 2D benchmark
used degree 7 and even sizes. Both include a nonperiodic polynomial/spline
part, with no periodic identification of opposite faces.

For the same curved-box turbulence MMS, the weak operator is
`sum_ab D_a^T [W det(J) (J^-1 J^-T)_ab] D_b`, using the full six-component
metric. Each D_a is a tensor product of one derivative and two value tables.
Native CUDA permutations and cuBLAS contractions apply it without assembling
a global 3D stiffness matrix. Memory is O(q³+n³+nq), work O(nq³+n³q) per
application; this initial implementation uses dense 1D contractions, not the
FFT acceleration of the other BSPF axis implementation.

The outer PCG, vectors, metric, RHS integration, boundary lifting, dot products,
and independently recomputed residual are FP64. The preconditioner uses the
1D generalized eigenproblem `K_II V = M_II V Lambda`, with a positive mean-metric
separable inverse. Its six tensor contractions and spectral division are
FP32. TF32 is disabled explicitly. Residuals are scaled before conversion to
FP32, and results converted back to FP64. Eigenproblems are prepared in FP64.
True-residual replacement restarts CG, including when its recursive residual
first passes tolerance. A final check uses the full recovered solution and
original quadrature operator. No analytic MMS values enter CG or stopping.

All six faces receive exact nonzero Dirichlet nodal samples. The boundary lift has zero
interior; each repeated solve resets the correction to zero. Source sampling,
setup and error audits are outside warm timings. RHS quadrature, lifting, CG,
and recovery are included in `wall_s`; `cg_s` isolates the iteration. Input
transfers are separate from solve timing.

## Reproduce on this RTX 3070 machine

```bash
bash bench/turbulence_3d/bspf_poisson/build.sh
export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/math_libs/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/lib64
python bench/turbulence_3d/bspf_poisson/run.py --n 17 --polynomial --check
python bench/turbulence_3d/bspf_poisson/run.py --n 17 33 65 97 129
python -m pytest -q tests/python/test_cuda_poisson3d.py
```

Default quadrature is `q = max(96, ceil(1.5*n))` for the turbulence MMS;
the polynomial guard uses a minimum of 48 instead. Thus n=65,97,129 use
q=98,146,194. An explicit `--q` overrides this default.
`report.py` regenerates the original 2n-quadrature campaign from its saved
results; `report_optimization.py` regenerates the quadrature comparison.

Native CUDA uses toolkit 13.2, SM86, no fast-math. CuPy sampling/audits use the
existing CUDA 12 installation in a separate process environment. The binary
is `/tmp/bspf-poisson3d-cuda-build/bspf_poisson3d`. GPU execution requires
access to the host GPU outside the restricted sandbox.

The CPU reference check uses NumPy tensor contractions for every curved-metric
cross term and an FP64 reference tensor preconditioner. Error audits evaluate
the BSPF basis directly at two separate higher Gauss orders. This is not GLL
interpolation of the nonuniform BSPF nodal values. The unchanged analytic
1,856-mode MMS is evaluated at physical coordinates; the HDF5 grid is not
interpolated. Output JSON records the dataset and native binary hashes.

Tests additionally check the zero-load solve and reject indefinite metrics,
nonpositive preconditioner spectra, and invalid boundary lifts. This version
supports one mapped tensor patch and Dirichlet conditions on all faces; it
does not yet provide multipatch assembly or Neumann null-space handling.

## Measured validation and convergence

The following table records the original `q=max(96,2*n)` campaign, before
the default changed. Results with the current default are in the optimization study.

Seven GPU regression tests pass. At n=17 the polynomial guard has relative
L2 error 1.92e-13. CUDA/CPU operator agreement is 8.97e-17 relative; the FP32
preconditioner differs from its FP64 reference by 2.11e-7. At n=65 increasing
operator quadrature from 130 to 146 changes the solution by 9.62e-14 relative.

| Nodes per axis | Global DOFs | Relative L2 error | Warm CG (s) | Iterations |
|---:|---:|---:|---:|---:|
| 17 | 4,913 | 4.222e-1 | 0.0479 | 16 |
| 33 | 35,937 | 1.286e-1 | 0.0867 | 18 |
| 65 | 274,625 | 2.737e-12 | 0.3583 | 18 |
| 97 | 912,673 | 1.937e-12 | 1.5648 | 18 |
| 129 | 2,146,689 | 2.634e-12 | 4.7065 | 18 |

The n=129 solver owns 1.99 GB of device buffers (this excludes CUDA/cuBLAS
runtime allocations). Its full warm RHS/lift/CG/recovery time is 4.989 s.
For this smooth finite-mode MMS the accuracy reaches the current solve-tolerance
floor at n=65. Higher resolutions increase the dense-contraction cost without
improving accuracy. This is not a claim of the same efficiency for nonsmooth
fields, complex multipatch domains, or a different frequency spectrum.

[Accuracy/wall-time comparison](../evidence/bspf_cuda3d_rtx3070/accuracy_walltime.png),
[PDF](../evidence/bspf_cuda3d_rtx3070/accuracy_walltime.pdf),
[full report](../evidence/bspf_cuda3d_rtx3070/report.json).
The MFEM and NekRS curves have matching DOF counts, but different trial spaces,
node locations and integration rules; read the accuracy and time together.

The [optimization study](OPTIMIZATION.md) profiles the CUDA kernels, tests a
direct tensor layout, and validates reduced quadrature with 1.79–1.89× faster
warm CG at n=65–129 on this MMS. These validated quadrature orders are now
the benchmark default; `--q` remains available for accuracy studies.

[Optimized BSPF vs MFEM and NekRS plot](../evidence/bspf_cuda3d_optimization/optimized_solver_comparison.png)
([PDF](../evidence/bspf_cuda3d_optimization/optimized_solver_comparison.pdf),
[CSV](../evidence/bspf_cuda3d_optimization/optimized_solver_comparison.csv)) uses
current-default quadrature and matching DOFs. Regenerate it with
`python bench/turbulence_3d/bspf_poisson/report_optimized_comparison.py`.
The n=17/33 points reuse unchanged q=96 measurements; n=65/97/129 use the
validated optimized campaign. Full warm MFEM timing includes host RHS assembly.
The current comparison uses MFEM's newly measured FP32 LOR-AMG preconditioner
with FP64 operator/CG. The previous plot is preserved as
`optimized_solver_comparison_mfem_fp64.png` in the same evidence directory.

## Reflection-split contractions

The CUDA operator now detects reflection symmetry of the supplied 1D value and
first-derivative matrices. When present, it uses orthonormal paired sums and
differences to reduce each FP64 contraction to two smaller GEMMs. The value
matrix preserves even/odd parity; the derivative exchanges it. Transposed
contractions use the same blocks, preserving the Galerkin adjoint structure.
The full spatially varying metric is still applied pointwise between forward
and transpose transforms; geometry itself need not have reflection symmetry.

This preserves degree 8, Fourier enrichment, quadrature and arithmetic precision.
Matrices are checked against reflected copies with a relative max-entry bound
of 1e-12; larger defects automatically use the original path. Tiny discarded
cross-parity components are covered by the CPU-reference and solution audits.
Set `BSPF_POISSON_PARITY=0` to force the original contractions for comparison.
The native JSON reports `parity_optimized` and includes extra buffers in
`device_bytes`. Splitting adds two q³ FP64 workspaces and the small block tables.
It applies to the FP64 operator, RHS transform and boundary lift; the FP32
preconditioner is unchanged.

Current measurements and solution comparisons are in
`../evidence/bspf_cuda3d_parity/optimization.json`; regenerate the plot with
`python bench/turbulence_3d/bspf_poisson/report_parity.py`. The solver comparison
scripts prefer these independently audited results when available. Original
measurements remain in their existing evidence directories.

## Shared gradient contractions

When reflection splitting is active, the default operator also shares the x
interpolation prefix of the y and z gradients. The transpose reverses this
operation graph: those two contributions are added before the final x
interpolation transpose. This reduces the gradient-plus-divergence calculation
from 18 to 16 one-dimensional contractions, preserving the Galerkin adjoint.
The full metric, degree 8, quadrature, FP64 CG and FP32 preconditioner remain
unchanged. One additional n²q FP64 buffer holds the shared intermediate.

Set `BSPF_POISSON_SHARED=0` to compare against the previous reflection-split
operator; native JSON records the actual `shared_contractions` flag. Sharing
falls back when parity is unavailable or the experimental direct layout is
compiled. To reproduce the alternating timing/error campaign from saved inputs:

```sh
python bench/turbulence_3d/bspf_poisson/benchmark_shared.py
```

This reads the previous `bspf_cuda3d_parity/nN_qQ/input.bin` bundles. Each size
gets two alternating off/on pairs, with one warmup and three measured solves
per invocation, followed by independent physical-error quadratures. Its result
uses the median of six measured solves per mode. Evidence is saved under
`../evidence/bspf_cuda3d_shared/`; comparison plots prefer these results.

## Fused parity conversion and tensor rotation

Parity packing/unpacking now incorporates the tensor-axis permutations. This
removes separate full-array rotation passes without changing the Galerkin
matrix, contraction count, quadrature, precision, or workspace allocation.
It is enabled by default whenever parity splitting is available; set
`BSPF_POISSON_FUSED_LAYOUT=0` for the previous path. Native JSON records the
actual choice as `fused_tensor_layout`.

Run `python bench/turbulence_3d/bspf_poisson/benchmark_fused_layout.py` to compare
both paths on the saved Gaussian MMS inputs at 65³, 97³, and 129³. This measures
repeated full solves, including RHS/lifting work, with three alternating trials
and nine warm samples per mode. See [OPTIMIZATION.md](OPTIMIZATION.md) for results.
