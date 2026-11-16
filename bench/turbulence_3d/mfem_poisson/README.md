# MFEM/libCEED curved 3D Poisson benchmark

Current Gaussian MMS campaign: [dependencies, builds, validation and reproduction](../REPRODUCING.md). Historical measurements below retain their original configurations.

Solves `-Δφ = f` with `φ = u_x` from the nonperiodic shell-32 turbulence MMS and exact nonhomogeneous Dirichlet data on all six curved faces. This tests one scalar Poisson component, not a coupled incompressible-flow solve. The Q2 geometry is identical to `mms.CurvedBox`; forcing and boundary values are evaluated analytically from `modes.npz` on the GPU, independently of the saved 97³ field grid.

The native adapter uses stock MFEM 4.9 H1 Gauss–Lobatto elements, partial assembly, libCEED `/gpu/cuda/gen`, and FP64 CG (relative tolerance 1e-12). Preconditioners include Jacobi and GPU LOR-AMG; the current comparison uses FP32 Hypre LOR-AMG. Operator/load quadrature uses `p+3` Gauss points per axis.

The current adapter caches physical Jacobian determinants during fixed-mesh setup and feeds them to MFEM's existing GPU tensor load-assembly kernel. This avoids the generic geometry kernel's `q^3` thread launch, which previously forced degree-8 / 11-point runs onto CPU RHS assembly. The load kernel itself launches only `q^2` threads and supports the existing 11- and 13-point rules. RHS assembly still runs on every solve; the assembled load vector is not cached. Boundary initialization uses a device-to-device copy of the prescribed nodal boundary vector. Operator quadrature, FP64 outer CG and FP32 AMG are unchanged. Geometry must be rebuilt if the mesh changes.

Results report `rhs_assembly: mfem_device_cached_geometry`, per-repeat `rhs_samples_seconds`, and `boundary_initialization: device_copy`. On small meshes the adapter directly compares the GPU assembled RHS with stock host assembly and rejects discrepancies above 2e-12. Historical host-fallback timings below are retained as evidence of the earlier implementation. Updated Gaussian-campaign results are in `../evidence/gaussian_mms_rtx3070/mfem_gpu_rhs`; the comparison plot selects this directory when available.

Wall times synchronize CUDA and include load assembly, Dirichlet elimination, CG, and solution recovery. They exclude geometry/operator setup, analytic sampling, file exchange, and error integration. First-solve times include lazy kernel compilation; warm times are medians of three measured repetitions after one additional warmup. The plot shows warm full solves, warm GPU CG alone, and first solves separately. These are single-machine measurements on a display-attached RTX 3070, not statistically isolated hardware comparisons.

Errors integrate the physical-domain L² norm using independent Gauss rules at two orders. The auditor reconstructs the element polynomials on the GPU from MFEM's lexicographic local coefficients; separate native point probes verify ordering. The runner requires agreement of the two error quadratures, checks the geometry against the analytic map, checks the true algebraic residual, and first reproduces `1+x²+y²+z²` on a curved mesh. By default the last turbulence case is repeated with higher operator/load quadrature. `--quadrature-check-case e:p` selects a different case; the extension to 24³ retains the completed 16³ operator-quadrature check. Every case still has two independent error-integration orders.

## Build and run in this workspace

Requires a CUDA/libCEED-enabled MFEM installation and CuPy, NumPy, matplotlib. Existing library builds are in `/tmp` and may need rebuilding after reboot. The local machine has separate CUDA installations for CuPy (12.6) and MFEM (13.2); `run.py` explicitly sets the native subprocess library path.

```bash
CPATH=/home/moulin/Workspace/nvidia/hpc_sdk/Linux_x86_64/26.5/math_libs/include \
cmake -S bench/turbulence_3d/mfem_poisson -B /tmp/bspf-mfem-poisson3d \
  -DMFEM_DIR=/tmp/bspf-mfem-ceed86 \
  -DCUDAToolkit_ROOT=/home/moulin/Workspace/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2 \
  -DCMAKE_BUILD_TYPE=Release
CPATH=/home/moulin/Workspace/nvidia/hpc_sdk/Linux_x86_64/26.5/math_libs/include \
cmake --build /tmp/bspf-mfem-poisson3d -j2
CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6 \
LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/math_libs/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/lib64 \
python bench/turbulence_3d/mfem_poisson/run.py \
  --output bench/turbulence_3d/evidence/mfem_poisson_rtx3070
```

Use a new output directory for each run, or `--resume` to reuse completed cases from the same dataset and configuration. `--cases` selects comma-separated `elements_per_axis:degree` pairs. Raw coordinate/sample files are removed after a successful audited solve; local coefficients, native solutions, logs, JSON results, and PNG/PDF plots are retained.

## Backend limits observed

The initial degree-8 device load assembly failed in MFEM's `Det3D<0,0,false>` kernel with `cudaErrorInvalidValue` at 11 quadrature points per direction. The adapter now uses the host RHS path for that configuration. A degree-12 / 15-point run subsequently failed inside libCEED's `CeedRunKernelDimShared_Cuda` with `CUDA_ERROR_INVALID_VALUE`; the exact launch-resource cause has not been diagnosed. Its log is retained under `e8_p12/solve.log`, and it is excluded from accuracy/timing results. The completed default campaign therefore refines degrees 4 and 8.

## RTX 3070 measurements

| Elements | Degree | Unknowns | Relative L² error | Warm solve (s) | GPU CG (s) |
|---|---:|---:|---:|---:|---:|
| 4³ | 4 | 4,913 | 6.234e-01 | 0.0191 | 0.01883 |
| 8³ | 4 | 35,937 | 1.238e-01 | 0.03458 | 0.03394 |
| 12³ | 4 | 117,649 | 3.053e-02 | 0.1234 | 0.1219 |
| 4³ | 8 | 35,937 | 1.374e-01 | 0.1896 | 0.04319 |
| 8³ | 8 | 274,625 | 3.545e-03 | 1.599 | 0.3765 |
| 12³ | 8 | 912,673 | 1.507e-04 | 5.85 | 1.846 |
| 16³ | 8 | 2,146,689 | 1.386e-05 | 15.45 | 5.814 |
| 24³ | 8 | 7,189,057 | 4.053e-07 | 59.78 | 28.75 |

[Plot](../evidence/mfem_poisson_rtx3070/accuracy_walltime.png) · [PDF](../evidence/mfem_poisson_rtx3070/accuracy_walltime.pdf) · [CSV](../evidence/mfem_poisson_rtx3070/results.csv) · [JSON](../evidence/mfem_poisson_rtx3070/report.json).

The finest degree-8 refinement has observed L² convergence order about 8.7, approaching the nominal smooth-solution order of 9. Coarser meshes do not yet resolve the highest-frequency modes. This is a correctness and baseline performance study; Jacobi-preconditioned CG is not an optimized multigrid solver.

Increasing the 16³ degree-8 operator/load quadrature from 11 to 13 points per direction changed the relative L² error by 1.456e-08 fractionally. The curved polynomial reproduction test reached 2.220e-13 relative error; all measured true algebraic residuals were below 3e-13.

## Extension through 24³ elements

With the same CUDA environment as above, run:

```bash
python bench/turbulence_3d/mfem_poisson/run.py \
  --output bench/turbulence_3d/evidence/mfem_poisson_rtx3070 --resume \
  --cases 4:4,8:4,12:4,4:8,8:8,12:8,16:8,24:8 \
  --quadrature-check-case 16:8
```

The native adapter records total device memory in use at selected stages, including other processes. These observations are not a continuously measured peak. Completed native solves can be recovered after an interrupted audit when their configuration manifest matches. The 24³ run's sampling timer was not preserved when the controller was stopped at the revised endpoint; it is explicitly null, while the saved solver timings are complete. Geometry was rechecked from the saved coordinates during recovery.

## Preconditioner scope

The original campaign is a Jacobi baseline built with serial MFEM. The separate MPI/CUDA/HYPRE build below adds the LOR-AMG comparison. MFEM 4.9's `miniapps/solvers/plor_solvers.cpp` demonstrates `LORSolver<HypreBoomerAMG>` with a partial-assembly high-order operator. The new build uses CUDA-enabled HYPRE and MPI-enabled MFEM with one MPI rank. Low-order-refined AMG avoids full high-order matrix assembly, but still adds an auxiliary sparse matrix and multigrid hierarchy. Its setup cost, memory, iteration count and solve time are reported separately from the saved Jacobi baseline.

The 24³ case converged in 1,568 CG iterations, with true relative residual 2.079e-14. Its two independent error quadratures gave relative L² errors 4.0533922588e-07 and 4.0533920024e-07. The observed order from 16³ to 24³ was 8.711. Maximum observed total device use was 5.581 GB (including other processes), compared with 2.226 GB at startup. The requested endpoint was revised to 24³; no 32³ case was launched.

## CUDA HYPRE + LORSolver build

`build_hypre_gpu.sh` builds a separate FP64 stack under `/tmp`: HYPRE 3.1.0 with CUDA 13.2, architecture 86, MPI, and device memory (unified memory disabled); then MFEM 4.9 with MPI, CUDA and libCEED. It copies HYPRE sources to `/tmp` because HYPRE generates files in its source tree. The original serial MFEM build and Jacobi evidence are retained.

```bash
bash bench/turbulence_3d/mfem_poisson/build_hypre_gpu.sh
```

The MPI adapter requires one rank. `--preconditioner lor-amg` selects MFEM's `LORSolver<HypreBoomerAMG>`, with MFEM's GPU AMG defaults and a fixed auxiliary operator across repeated solves. It explicitly builds the AMG hierarchy before timing repeated solves and reports LOR assembly, AMG setup, total preconditioner setup, and total native setup separately. `hypre_gpu` must be true for a CUDA LOR-AMG run.

With the same CuPy CUDA environment used above:

```bash
python bench/turbulence_3d/mfem_poisson/run.py \
  --binary /tmp/bspf-mfem-poisson3d-hypre/mfem_poisson3d \
  --preconditioner lor-amg --cases 1:8,2:8,4:8,8:8,12:8,16:8 \
  --quadrature-check-case 16:8 \
  --output bench/turbulence_3d/evidence/mfem_lor_amg_rtx3070
python bench/turbulence_3d/mfem_poisson/compare_preconditioners.py \
  bench/turbulence_3d/evidence/mfem_poisson_rtx3070 \
  bench/turbulence_3d/evidence/mfem_lor_amg_rtx3070
```

The high-order diffusion operator remains partial-assembled through libCEED. The degree-8 RHS still uses the host assembly fallback documented above. The comparison checks dataset identity, matching resolution/quadrature and consistent independently integrated solution errors; setup-plus-first-solve timings include AMG construction.

For the static operator, a small `ParLORDiscretization` subclass releases the protected batched assembly cache after stock MFEM has built the auxiliary matrix. `LORSolver<HypreBoomerAMG>` retains that matrix; no discretization or AMG kernels are replaced. This avoids retaining element assembly buffers throughout the solve. A solver wrapper keeps CG operator updates from retargeting the auxiliary preconditioner to the high-order operator.

### Degree-8 curved-geometry GPU correction

MFEM 4.9's generic `QuadratureInterpolator::Values3D` CUDA fallback allocates temporary storage using `MAX_INTERP_1D=8`. The Q2 geometry-to-degree-8 LOR interpolation requires 9 points per direction, and MFEM's default registration list lacks the `(dimension=3, layout=byVDIM, vdim=3, D1D=3, Q1D=9)` specialization. The initial run produced incorrect LOR vertex coordinates and nonfinite matrix entries. `lor_qinterp.cu` registers MFEM's existing template at those exact dimensions, with correctly sized static storage. It does not change the interpolation formula. The adapter checks vertex coordinates against physical element transformations before using AMG and rejects discrepancies above 2e-13.

The HYPRE source copy also includes two pre-existing local header additions (`thrust/iterator/reverse_iterator.h` and `thrust/pair.h`); the patch and build provenance are saved with the LOR-AMG evidence.

The LOR-AMG campaign starts at one curved element, with a degree-8 polynomial reproduction check on that minimum mesh. The 1³ and 2³ turbulence cases are severely underresolved, including their forcing quadrature; they are smoke tests and coarse points, not the asymptotic convergence regime. The completed sequence is `--cases 1:8,2:8,4:8,8:8,12:8,16:8`, capped at 16³ elements at the user’s request.

### Measured GPU LOR-AMG results

| Elements | Unknowns | Relative L² error | CG iterations | GPU CG (s) | Warm solve (s) | Preconditioner setup (s) |
|---|---:|---:|---:|---:|---:|---:|
| 1³ | 729 | 5.741e+00 | 20 | 0.005993 | 0.008513 | 0.04211 |
| 2³ | 4,913 | 1.651e+00 | 30 | 0.01433 | 0.03351 | 0.04392 |
| 4³ | 35,937 | 1.374e-01 | 37 | 0.02685 | 0.1772 | 0.06078 |
| 8³ | 274,625 | 3.545e-03 | 39 | 0.117 | 1.356 | 0.2063 |
| 12³ | 912,673 | 1.507e-04 | 40 | 0.3529 | 4.483 | 0.6362 |
| 16³ | 2,146,689 | 1.386e-05 | 40 | 0.7839 | 10.58 | 1.426 |

[AMG accuracy–walltime plot](../evidence/mfem_lor_amg_rtx3070/accuracy_walltime.png) · [Jacobi comparison](../evidence/mfem_lor_amg_rtx3070/preconditioner_comparison.png) · [Comparison data](../evidence/mfem_lor_amg_rtx3070/comparison.json) · [Report](../evidence/mfem_lor_amg_rtx3070/report.json).

At 16³, GPU CG is 7.42× faster than the saved Jacobi baseline, while assembly-plus-solve is 1.46× faster. The one-element polynomial guard passed at 7.353e-13 relative error. All AMG runs report `hypre_gpu=true`; each solution passed independent error quadrature, geometry checks, and a true residual check. The report references the existing 16³ Jacobi 11-vs-13-point operator-quadrature check for the identical high-order operator; it does not claim an additional AMG operator-quadrature run.

A 24³ attempt ran out of device memory during auxiliary LOR matrix assembly, before AMG setup or CG. It is excluded from the plotted measurements and retained as a failure log. The requested endpoint was then reduced to 16³. The completed measurements use MFEM’s default GPU BoomerAMG options (no aggressive coarsening); no larger-memory tuning is included.

## FP32 GPU LOR-AMG with FP64 outer CG

`build_hypre_fp32.sh` builds a private, static, position-independent FP32 HYPRE
3.1.0 and a separate benchmark executable under `/tmp/bspf-mfem-poisson3d-fp32`.
It reuses the existing FP64 MFEM/libCEED stack. Select `--preconditioner
lor-amg-fp32` with that executable. The original `lor-amg` option remains FP64.

The adapter uses `LORSolver<FloatHypreBoomerAMG>` and the same MFEM GPU AMG
settings: PMIS coarsening (8), no aggressive levels, extended+i interpolation
(6), Pmax=4, strength threshold 0.25, one l1-Jacobi sweep (18), at most 25
levels, and one V-cycle per application. The LOR matrix is assembled in FP64,
then copied to an FP32 HYPRE matrix during setup. Hierarchy setup and cycles
use FP32. The partial-assembly operator, outer CG, and true residual check
remain FP64. Residual and correction casts run on CUDA; synchronization and
conversion costs are included in linear-solve timing. Setup includes the matrix
conversion and upload.

A shared bridge hides all symbols from its private HYPRE archive and binds
internal calls locally. Only `bspf_amg32_*` C functions cross the ABI boundary;
HYPRE objects never cross between FP64 and FP32 libraries. Static assertions
and a runtime scalar-size check require four-byte HYPRE scalars. This adapter
supports the benchmark's single MPI rank and GPU backend only.

```bash
bash bench/turbulence_3d/mfem_poisson/build_hypre_fp32.sh
# Use the CuPy CUDA 12 environment shown above.
python bench/turbulence_3d/mfem_poisson/run.py \
  --binary /tmp/bspf-mfem-poisson3d-fp32/mfem_poisson3d \
  --preconditioner lor-amg-fp32 --cases 2:8,4:8,8:8,12:8,16:8 \
  --output bench/turbulence_3d/evidence/mfem_lor_amg_fp32_rtx3070
```

Measured FP32 results on the RTX 3070 (three warm linear-solve samples):

| Global nodes/axis | FP64 AMG CG (s) | FP32 AMG CG (s) | Speedup | CG iterations |
|---:|---:|---:|---:|---:|
| 17 | 0.01433 | 0.01194 | 1.20× | 30 |
| 33 | 0.02685 | 0.02761 | 0.97× | 37 |
| 65 | 0.11704 | 0.08109 | 1.44× | 39 |
| 97 | 0.35289 | 0.25810 | 1.37× | 40 |
| 129 | 0.78387 | 0.60081 | 1.30× | 40 |

Iteration counts match the FP64 baseline at every resolution. Relative changes
in the element-local solution coefficients are below 2.9e-12, and all true
FP64 residuals are below 5.1e-13. The one-element polynomial guard has relative
L2 error 4.31e-13. At n=129 the turbulence L2 error remains 1.386e-5. Full warm
RHS/boundary/solve time is 10.503 s; host RHS work still dominates that timing.
Clocks were not locked, and these are short timing samples, so the small
n=33 regression should not be treated as a significant precision effect.

[Precision comparison](../evidence/mfem_lor_amg_fp32_rtx3070/precision_comparison.json)
can be regenerated with `python bench/turbulence_3d/mfem_poisson/report_precision.py`.
The [updated solver plot](../evidence/bspf_cuda3d_optimization/optimized_solver_comparison.png)
uses FP32 preconditioners for MFEM, BSPF, and NekRS; their algorithms and trial
spaces still differ. All builds and benchmark artifacts are separate from the
original FP64-preconditioner evidence.
The FP32 campaign also passed a fresh e=16 operator-quadrature check (11 to
13 points/axis): relative change in the audited L2 error is 1.46e-8. The
retained FP64 AMG option passed a polynomial compatibility run in the same
executable. See `mfem_lor_amg_fp32_rtx3070/validation.txt` for the checks.
