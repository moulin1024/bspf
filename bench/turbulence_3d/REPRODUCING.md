# Reproducing the curved 3D Poisson campaign

This is the entry point for the current **plane-wave plus eight Gaussian bubbles,
no tanh** campaign. The physical domain and all six Dirichlet boundaries are
nonperiodic. Only the scalar Poisson MMS is augmented; the underlying plane-wave
velocity remains divergence-free. Historical READMEs and evidence also describe
older plane-wave-only, tanh, CPU-RHS, and unfused runs.

## Dependencies and pinned sources

The measured platform is Linux x86-64, RTX 3070 (8 GB, compute capability 8.6),
GCC/G++/gfortran, CMake >=3.21, GNU Make, Git, and NVIDIA HPC SDK 26.5 (CUDA
13.2, cuBLAS, cuSPARSE, cuSOLVER, cuRAND, and its Open MPI/HPC-X). Install the
SDK separately from NVIDIA, with a driver supporting its CUDA runtime. The
scripts target SM86; other GPUs require adjusting CUDA architecture flags,
including the NekRS Hypre patch. This is a single-GPU, single-rank campaign.

| Component | Recorded source/version | Role |
|---|---|---|
| libCEED | `4018a20a98d451fac24765d3ddb936861647ce8d` | MFEM CUDA partial assembly |
| MFEM | release `v4.9` (local release source, not a Git checkout) | degree-8 finite elements and LOR |
| Hypre | `9dc9e18aed6a945a95f966e57daacfb1c269f6ec` | GPU AMG, separate FP64 and FP32 builds |
| NekRS | v26.0, `96b3cf9e5bacede16568826c04a21bc0fe50dc7d` | degree-8 spectral elements, bundled OCCA/Hypre |
| CuPy | `cupy-cuda12x==13.6.0`, CUDA 12.6 | MMS generation, preparation and audits |
| PyTorch | `2.6.0+cu124` | FC-Gram GPU baseline |
| TensorLy | `0.9.0` | FC extension contractions |
| NeuralOperator FC-Gram | `a6648eeeb60fe0837f2d7d021517a945a159a0c8` | vendored extension code/matrices |
| PETSc (optional) | 3.26.0 | experimental FD8, excluded from current five-solver plot |

The FC code, small matrix files, upstream MIT license, and hashes are committed
under `fc_poisson/vendor`; no NeuralOperator installation is needed. NumPy,
SciPy, matplotlib and h5py requirements are in `requirements.txt`. BSPF and
global Chebyshev share the native CUDA backend and require no external PDE library.

## Build order

Run commands from the repository root. The scripts retain `/tmp/bspf-*` build
and installation paths to match the drivers. These are disposable and must be
rebuilt after cleanup/reboot. Do not mix an old CMake cache with another toolkit.
Set the SDK and source locations; these overrides are honored by build scripts
and native benchmark subprocesses:

```bash
export BSPF_HPC_SDK=/path/to/nvidia/hpc_sdk/Linux_x86_64/26.5
export BSPF_HYPRE_SOURCE=/tmp/bspf-hypre-upstream
export BSPF_MFEM_SOURCE=/tmp/bspf-mfem-upstream
export PATH="$BSPF_HPC_SDK/cuda/13.2/bin:$BSPF_HPC_SDK/comm_libs/mpi/bin:$PATH"
export CPATH="$BSPF_HPC_SDK/math_libs/include"
export LIBRARY_PATH="$BSPF_HPC_SDK/math_libs/lib64"
export LD_LIBRARY_PATH="$BSPF_HPC_SDK/math_libs/lib64:$BSPF_HPC_SDK/cuda/13.2/lib64:$BSPF_HPC_SDK/comm_libs/13.2/hpcx/hpcx-2.50/ompi5/lib"
export OMPI_CC=gcc OMPI_CXX=g++ OMPI_FC=gfortran

git clone https://github.com/CEED/libCEED.git /tmp/bspf-libceed
git -C /tmp/bspf-libceed checkout 4018a20a98d451fac24765d3ddb936861647ce8d
make -C /tmp/bspf-libceed -j8 CUDA_ARCH=sm_86 CUDA_DIR="$BSPF_HPC_SDK/cuda/13.2"
git clone https://github.com/hypre-space/hypre.git "$BSPF_HYPRE_SOURCE"
git -C "$BSPF_HYPRE_SOURCE" checkout 9dc9e18aed6a945a95f966e57daacfb1c269f6ec
git clone --branch v4.9 --depth 1 https://github.com/mfem/mfem.git "$BSPF_MFEM_SOURCE"

bash bench/turbulence_3d/bspf_poisson/build.sh
bash bench/turbulence_3d/mfem_poisson/build_hypre_gpu.sh
bash bench/turbulence_3d/mfem_poisson/build_hypre_fp32.sh
bash bench/turbulence_3d/nekrs_poisson/build.sh
```

MFEM uses FP64 throughout the operator/outer CG. The private FP32 Hypre library
is built with PIC and hidden symbols through `hypre_fp32_bridge.cu`, avoiding
symbol collisions with MFEM's FP64 Hypre. Both use CUDA, explicit device memory,
MPI enabled and GPU-aware MPI disabled. Select `lor-amg-fp32`, not Jacobi or
FP64 LOR, for the comparison. The adapter caches mesh geometry and uses GPU RHS
assembly on every solve; check `rhs_assembly: mfem_device_cached_geometry`.

NekRS's script pins its source and applies three compatibility fixes: SM86 Hypre,
Thrust headers needed by CUDA 13, and registration of kernels for a SEMFEM coarse
grid within p-multigrid. These patches are part of the recipe; an unpatched
upstream binary is not the measured configuration. The bundled preconditioner
is FP32 while the main solve is FP64. OCCA JIT caches live under `/tmp` and first
compilation is excluded from warm timings.

Optional FD8: install OpenBLAS development headers/libraries and pkg-config,
then run `petsc_fd/build_petsc.sh` and `petsc_fd/build_solver.sh`. It depends on
the FP64 Hypre installation and the MFEM adapter's FP32 bridge built above.
This experiment is retained but is not a sixth completed campaign baseline.

## Python environments and dataset

Use a separate Python environment for CuPy and PyTorch. Do not inject CUDA 13
libraries into CuPy's CUDA 12 process or the Torch CUDA 12.4 environment.

```bash
python -m venv --system-site-packages /tmp/bspf-cupy
/tmp/bspf-cupy/bin/pip install -r bench/turbulence_3d/requirements.txt cupy-cuda12x==13.6.0 pytest
python -m venv --system-site-packages /tmp/bspf-fc-torch
/tmp/bspf-fc-torch/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
/tmp/bspf-fc-torch/bin/pip install tensorly==0.9.0 numpy scipy matplotlib
export BSPF_TORCH_PYTHON=/tmp/bspf-fc-torch/bin/python
export CUDA_PATH=/path/to/cuda/12.6
export LD_LIBRARY_PATH="$CUDA_PATH/lib64"
# If using HPC SDK 24.11 CUDA 12.6, also include its math_libs/12.6/lib64.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

The committed `evidence/nonperiodic_curved_gaussian/modes.npz` is the exact
campaign input (SHA256 `c9119da1d12d3bbca43497bcddc5f5b92baf6611cdd047c49866e025812c9949`).
Solvers evaluate its analytic field; large sampled HDF5 files are unnecessary.
To regenerate samples and figures, or reconstruct coefficients from the seeds:

```bash
/tmp/bspf-cupy/bin/python bench/turbulence_3d/generate.py --n 129 --out /tmp/bspf-base-mms
/tmp/bspf-cupy/bin/python bench/turbulence_3d/generate_mixed.py --no-tanh \
  --base /tmp/bspf-base-mms/modes.npz --out /tmp/bspf-gaussian-mms
```

Wave seed is 20261116, feature seed 20261117. Use the committed NPZ for exact
input identity; archive serialization and numerical-library versions may affect
regenerated bytes. Gaussian augmentation is applied to the scalar and its
analytic Laplacian, including exact boundary values.

## Run, verify and plot

Use a **new output directory** after changing code, solver options or libraries.
The runner's resume check validates dataset hashes, not the complete build
configuration, and must not be used to compare changed implementations.

```bash
/tmp/bspf-cupy/bin/python -m pytest tests/python/test_cuda_poisson3d.py \
  tests/python/test_nonperiodic_turbulence_mms.py tests/python/test_poisson_features.py \
  tests/python/test_chebyshev_poisson.py -q
/tmp/bspf-cupy/bin/python bench/turbulence_3d/run_mixed_suite.py \
  --max-n 129 --output /tmp/bspf-gaussian-campaign
/tmp/bspf-cupy/bin/python bench/turbulence_3d/fc_poisson/report.py \
  --suite-root /tmp/bspf-gaussian-campaign --max-n 129
```

Jobs run sequentially to avoid GPU contention. The default grid sizes are
17, 33, 65, 97, 129; `--max-n 145` enables the larger historical extension.
MFEM/NekRS use 2,4,8,12,16 elements per axis at degree 8. BSPF uses degree 8
polynomial enrichment with increasing Fourier pairs, Chebyshev degree n−1,
and q=max(96,ceil(1.5n)). Parity splitting, shared contractions and fused layouts
are enabled by default. FC uses degree 8 and 25 continuation points per side,
FP64 FGMRES and a complex FP32 inverse; its 1e-10 tolerance differs from the
other solvers. See each adapter's README for convergence/error-audit details.

Both plot axes use physical n³ counts including boundary nodes; FC continuation
points are excluded from that count but their work remains in the timing.
Repeated full solve means RHS/lifting, Krylov solve and recovery on fixed setup.
It excludes analytic MMS sampling, one-time mesh/operator/preconditioner setup,
JIT compilation, file I/O and error audits. It is not total campaign runtime.

Compact historical JSON results and comparison CSV are committed; raw bundles,
solution arrays, HDF5 samples, caches and binaries are omitted (about 20 GB).
The saved comparison predates fused-layout optimization; its separate alternating
A/B evidence is in `gaussian_mms_rtx3070/bspf_fused_layout/summary.json`.
A fresh run uses the consolidated optimized backend for both BSPF and Chebyshev.
Timings are hardware/load dependent; compare accuracy and residual checks first.

To redraw the archived comparison without GPU work:

```bash
python bench/turbulence_3d/fc_poisson/report.py \
  --suite-root bench/turbulence_3d/evidence/gaussian_mms_rtx3070 --max-n 129
```

The only committed MMS binary is the 184 KB coefficient NPZ, not a sampled
3D field. The three upstream FC reference matrices total less than 9 KB.
