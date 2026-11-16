# PyTorch FC-Gram GPU Poisson baseline

Current Gaussian MMS campaign: [dependencies, builds, validation and reproduction](../REPRODUCING.md). Historical measurements below retain their original configurations.

This is our Poisson adaptation of NeuralOperator's public Fourier-continuation
layer, not a reference PDE solver supplied by NeuralOperator. The unchanged
upstream module and degree-4/6/8 tables are vendored under `vendor/`, with the MIT
license and SHA-256 provenance. Upstream commit:
[`a6648ee`](https://github.com/neuraloperator/neuraloperator/tree/a6648eeeb60fe0837f2d7d021517a945a159a0c8).

## Problem and algorithm

Solve `-Delta_X phi = f`, with `phi = u_x` of the existing 1,856-mode turbulence
MMS and exact nonzero Dirichlet values on all six curved faces. The dataset is
`../evidence/nonperiodic_curved_shell32/modes.npz`; its SHA-256 is recorded in
each result. The curved map has amplitudes `(0.08, -0.06, 0.07)`. All directions
are nonperiodic. Periodicity is imposed only on the artificial continuation.

* Uniform reference nodes, tensor storage `(z,y,x)`; all physical boundary nodes
  are eliminated by exact Dirichlet lifting.
* Public FC-Gram matching parameter `d=8`, 25 continuation nodes per side.
  Cache the upstream extension matrix, extend each differentiation axis,
  differentiate with CUDA FFTs, and crop back to the physical nodes.
* Analytic transformed strong operator `-G_ij d_ij phi - b_i d_i phi`, including
  all mixed derivatives and metric drift terms. Geometry preparation uses NumPy.
  Analytic turbulence values and forcing are sampled with PyTorch on CUDA.
* FP64 operator and right-preconditioned FGMRES(40), with two-pass classical
  Gram-Schmidt and true residual checks. Small Hessenberg algebra runs on CPU;
  its time and synchronization are included in solve timing.
* FP32 reference-box inverse: diagonalize the same FC second derivative,
  use the spatially averaged diagonal metric, and apply the separable inverse
  with CUDA complex64 tensor contractions. Both real and imaginary components
  are FP32. FGMRES is used because the collocation operator is nonsymmetric.

Default relative residual tolerance is **1e-10**. An initial polynomial test at
1e-12 stalled after 400 iterations at 8.84e-12. The degree-8 continuation creates
large intermediate values; FFT cancellation prevents uniformly reliable 1e-12
residuals in this implementation. A random-vector GPU FFT / CPU dense-derivative
check differs by 1.72e-11 at n=17 (acceptance bound 1e-10). The original 1e-12
comparison threshold failed and was adjusted to reflect this measured limit.
This is a simple baseline, not a performance-tuned FC implementation or a claim
about the best attainable accuracy of Fourier continuation methods.

## Reproduce

The measured environment is `/tmp/bspf-fc-torch` (PyTorch 2.6.0+cu124,
TensorLy 0.9.0, RTX 3070). To recreate it:

```sh
python -m venv --system-site-packages /tmp/bspf-fc-torch
/tmp/bspf-fc-torch/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
/tmp/bspf-fc-torch/bin/pip install tensorly==0.9.0 numpy scipy matplotlib
export OPENBLAS_NUM_THREADS=1
/tmp/bspf-fc-torch/bin/python bench/turbulence_3d/fc_poisson/run.py \
  --n 129 --field turbulence --audit \
  --out bench/turbulence_3d/evidence/fc_pytorch_rtx3070/turbulence_n129
/tmp/bspf-fc-torch/bin/python bench/turbulence_3d/fc_poisson/validate.py
python bench/turbulence_3d/fc_poisson/report.py
```

Run `--field polynomial --n 17`, `--field smooth --n 17/33`, and turbulence at
`--n 17/33/65/97/129`, each with a separate output directory. Every benchmark
runs one warmup plus three zero-initial-guess solves. Timers synchronize CUDA;
reported solve time excludes geometry, MMS sampling, lifting, setup, and audits.
Setup time is recorded separately. Peak Torch allocation includes audits and
excludes memory not tracked by Torch's allocator.

The error audit evaluates the full tensor FC interpolant at independent
Gauss-Legendre nodes and integrates with the physical Jacobian. It compares
orders `q=max(96,ceil(1.5*n))` and `q+8`; agreement must be within
`1e-10 + 0.001*error`. The absolute allowance accounts for interpolation
roundoff near the polynomial error floor. Nodal error is saved separately.

## Measured results

| Nodes per axis | Interior unknowns | FGMRES iterations | Warm solve (s) | Physical relative L2 |
|---:|---:|---:|---:|---:|
| 17 | 3,375 | 16 | 0.0301 | 3.77 |
| 33 | 29,791 | 17 | 0.0518 | 0.596 |
| 65 | 250,047 | 16 | 0.1868 | 0.0181 |
| 97 | 857,375 | 16 | 0.6243 | 0.00113 |
| 129 | 2,048,383 | 16 | 2.3400 | 0.000119 |

True turbulence residuals are below 8.8e-11. Boundary error is exactly zero.
The n=17 curved polynomial guard has physical relative L2 error 5.43e-11;
smooth errors decrease from 8.05e-5 at n=17 to 4.09e-7 at n=33.
Checks cover the upstream extension API, CPU/GPU operator agreement, FP32
reference inverse, original NumPy MMS sampler, zero RHS, reference-box solve,
and interpolation at the original nodes. The measured n=129 peak Torch
allocation is about 1.86 GiB.

The comparison plot reuses saved BSPF, MFEM/FP32 LOR-AMG and NekRS results,
matching global nodal counts but using different trial spaces. All operators
and outer Krylov arithmetic are FP64 and preconditioners FP32; FC's tolerance
is looser than the other solvers. At n=129, BSPF reaches 2.63e-12 in 2.56 s,
MFEM reaches 1.39e-5 in 0.601 s, and NekRS reaches 1.44e-5 in 0.121 s.
These are warm linear solve timings, not end-to-end costs.

Artifacts: `../evidence/fc_pytorch_rtx3070/`, including per-run JSON, nodal
solutions, logs, `validation.json`, `comparison.csv`, and PNG/PDF plots.

The live comparison plot prefers the latest shared-contraction BSPF timings
from `../evidence/bspf_cuda3d_shared/`, then reflection-split timings from
`../evidence/bspf_cuda3d_parity/`, when available. The BSPF timing quoted in the
table discussion above is the earlier unsplit result. See the plot CSV for the
exact measurement sources.

The second panel of `comparison.png/pdf` now plots **repeated full-solve wall
time versus accuracy**, including RHS preparation, boundary lifting, Krylov
solve and recovery. One-time setup, analytic MMS sampling and error audits are
excluded. MFEM's full timing includes host RHS assembly. BSPF/MFEM/NekRS use
their recorded full wall-time fields. FC full timings are measured separately
under `../evidence/fc_pytorch_rtx3070/full_timing/nN/`; the report checks that
these solutions agree with the independently audited originals before reusing
the error measurements. `comparison.csv` records both timing metrics and their
source paths. Run the FC driver without `--audit` into these timing directories
to reproduce this measurement (one warmup plus three timed solves).

## 145 global nodes per axis

The comparison now extends to 145³ = 3,048,625 global nodal DOFs (143³ =
2,924,207 interior DOFs). BSPF uses q=218; MFEM and NekRS use 18³ degree-8
hexahedra. Reproduce the additional points with the respective CUDA environments
specified in each solver's README:

```sh
python bench/turbulence_3d/bspf_poisson/run.py --n 145 \
  --output bench/turbulence_3d/evidence/bspf_cuda3d_shared
/tmp/bspf-fc-torch/bin/python bench/turbulence_3d/fc_poisson/run.py \
  --n 145 --audit --out bench/turbulence_3d/evidence/fc_pytorch_rtx3070/turbulence_n145
python bench/turbulence_3d/mfem_poisson/run.py \
  --binary /tmp/bspf-mfem-poisson3d-fp32/mfem_poisson3d \
  --preconditioner lor-amg-fp32 --cases 18:8 \
  --output bench/turbulence_3d/evidence/mfem_lor_amg_fp32_rtx3070_145
python bench/turbulence_3d/nekrs_poisson/run.py --elements 18
python bench/turbulence_3d/fc_poisson/report.py
```

The new points use medians of three measured warm solves. Earlier shared BSPF
points used six samples. FC's new result records both linear and full-solve
timing directly; the older points use separate full-timing runs checked against
the original audited solutions.

The current Gaussian comparison uses physical `n^3` grid counts, including
boundary nodes, for every solver. FC's 50 continuation nodes per axis are
excluded from the displayed count, but their computational cost remains in
wall time. The current comparison truncates at n=129 and selects MFEM's fixed
GPU RHS assembly results. Use `--suite-root` as shown in the reproduction guide.
