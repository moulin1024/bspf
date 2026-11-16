# RTX 3070 optimization experiment

The bottleneck is FP64 tensor contractions. Nsight Systems attributes 94.3% of
GPU kernel time to FP64 GEMMs at n=129, q=258. Double tensor permutations take
2.8%, flux evaluation 2.0%; FP32 contractions and permutations are small. This
profile includes four solves, lifts, residual checks, and device setup. It is
not a per-CG-iteration profile; timings below are unprofiled.

Removing permutations changed the GEMM layout but did not accelerate this GPU:

| Nodes/axis | Original warm CG | Direct layout warm CG | Change |
|---:|---:|---:|---:|
| 65 | 0.3733 s | 0.4187 s | 12.2% slower |
| 129 | 4.5825 s | 4.6842 s | 2.2% slower |

Both layouts produced identical saved nodal solutions in these runs. The direct
layout saves about 6.9% of solver buffers and passes all seven regression tests.
It remains an opt-in experiment (`BSPF_POISSON_DIRECT_LAYOUT=ON` in the standalone
CMake build). The default retains permutations. The smaller timing difference
should not be interpreted as statistically significant: clocks were not locked,
and each median has only three measured solves after one warm-up.

Reducing quadrature is effective for this particular smooth MMS:

| Nodes/axis | Quadrature/axis | Warm CG before → after | Speedup | Relative L2 after |
|---:|---:|---:|---:|---:|
| 65 | 130 → 98 | 0.3733 → 0.2015 s | 1.85× | 2.737e-12 |
| 97 | 194 → 146 | 1.5570 → 0.8237 s | 1.89× | 1.937e-12 |
| 129 | 258 → 194 | 4.5825 → 2.5618 s | 1.79× | 2.635e-12 |

Every solve still needs 18 iterations, with a recomputed relative residual below
1e-12. Reduced-quadrature solutions differ from the original nodal solutions by
7.2e-14, 7.1e-14, and 1.3e-13 in relative Euclidean norm. Independent quadrature
orders q+8 and q+16 agree on the analytic MMS L2 error. Boundary error is zero.
At n=129, allocated solver buffers fall from 1.991 GB to 0.964 GB; full warm
RHS/lift/solve/recovery takes 2.713 s. Sampling, CPU basis construction, device
setup, and error auditing are excluded from the CG timings.

The operator and outer CG remain FP64 and the preconditioner remains FP32 with
TF32 disabled. Reduced quadrature changes the discrete operator, so these
results establish adequacy for this dataset and tested trial spaces, not for
arbitrary forcing, geometry, or higher modes. Following this validation, the
benchmark default is now `q=max(96,ceil(1.5*n))` (minimum 48 for the polynomial
guard). Explicit `--q` values override the default.

## Reproduction

Build the default with `bash bench/turbulence_3d/bspf_poisson/build.sh`.
The build script uses a persistent CMake cache: after experimenting, explicitly
set `cmake -S bench/turbulence_3d/bspf_poisson -B /tmp/bspf-poisson3d-cuda-build
-DBSPF_POISSON_DIRECT_LAYOUT=OFF` and rebuild. Use `ON` to reproduce the direct
layout and rebuild before running the same input bundles.

With the CUDA 12 CuPy environment described in README.md, run:

```bash
python bench/turbulence_3d/bspf_poisson/run.py --n 65 --q 98 --output bench/turbulence_3d/evidence/bspf_cuda3d_optimization/quadrature
python bench/turbulence_3d/bspf_poisson/run.py --n 97 --q 146 --output bench/turbulence_3d/evidence/bspf_cuda3d_optimization/quadrature
python bench/turbulence_3d/bspf_poisson/run.py --n 129 --q 194 --output bench/turbulence_3d/evidence/bspf_cuda3d_optimization/quadrature
python bench/turbulence_3d/bspf_poisson/report_optimization.py
```

Baseline/layout timings use the native executable with the original
`evidence/bspf_cuda3d_rtx3070/n{n}_q{2*n}/input.bin` and output prefix
`evidence/bspf_cuda3d_optimization/{baseline,direct}_n{n}` (paths relative to
`bench/turbulence_3d`). Each invocation performs one warm-up and three measured
solves. Baseline L2 values in the report come from the original audited campaign;
current baseline solutions are compared directly with the reduced results.

Profile with `nsys profile --trace=cuda,cublas --sample=none --cpuctxsw=none
--output=baseline_profile <native-executable> <input.bin> <output-prefix>`, then
`nsys stats --report cuda_gpu_kern_sum --format csv baseline_profile.nsys-rep`.
Native execution needs the CUDA 13 libraries described in README.md.

[Plot](../evidence/bspf_cuda3d_optimization/optimization.png),
[measurements](../evidence/bspf_cuda3d_optimization/comparison.json),
[kernel profile](../evidence/bspf_cuda3d_optimization/profile_kernels.csv).

Further work should target FP64 contraction cost or reduce operator applications;
permutation fusion alone has a low ceiling in this profile. Any stronger inner
solver must preserve the outer Krylov method's preconditioner requirements.

## Testing q=n and interpreting the ratio

The q=n trial fails to preserve the benchmark accuracy:

| n | q | Result |
|---:|---:|---|
| 65 | 65 | CG 0.1249 s, 18 iterations, residual 9.97e-13, but relative L2 error 0.03246 |
| 97 | 97 | Native CG reports loss of positive definiteness |
| 129 | 129 | CPU generalized eigensolve produces a nonpositive eigenvalue; setup rejects it |

At n=65 the relative nodal change from q=130 is 0.02210. Independent error
quadratures 96 and 104 agree. At n=97 and 129 no successful solve timing or
solution accuracy is claimed. These failures are retained in
`quadrature_1n.log` and `quadrature_1n_summary.json` in the optimization evidence.
The CPU axis mass condition numbers at q=n are 8.45e4, 8.23e8, 6.82e10;
at q=ceil(1.5n) they are 12.35, 17.62, 27.21. The weighted derivative matrices
also become nearly rank deficient at q=n. Exact positive quadrature yields
positive semidefinite Gram matrices; numerically negative generalized
eigenvalues here reflect ill conditioning and roundoff, not negative weights.

There is no established optimal constant 1.5. Gauss quadrature integrates
polynomials through degree 2q-1 exactly, but n independent polynomial/Fourier
basis functions are not polynomials of degree n-1. Basis products, amplified
high-degree tails of normalized near-dependent Fourier combinations, and
curved metric factors all require additional quadrature resolution.

A useful criterion is spectral equivalence of quadrature and reference energy:
all generalized eigenvalues of (K_q,K_ref) and (M_q,M_ref) should be close to 1.
`scan_axis_quadrature.py` checks the interior axis spaces against q_ref=2n.
The maximum absolute deviation from 1 for K is:

| n | q≈1.2n | q≈1.3n | q≈1.4n |
|---:|---:|---:|---:|
| 65 | 2.16e-3 | 1.51e-7 | 5.44e-12 |
| 97 | 4.17e-3 | 3.95e-8 | 1.86e-11 |
| 129 | 4.67e-4 | 1.41e-10 | 6.91e-11 |

Values near 1e-11 include reference assembly/eigensolve roundoff. This suggests
1.4n as a candidate, but does not validate the curved 3D operator or RHS at that
ratio. A minimum acceptable ratio depends on accuracy tolerance, basis, geometry,
and source; full MMS checks remain necessary. See `axis_quadrature_scan.json`.

## FP64 reflection splitting (subsequent optimization)

The default CUDA path now exploits reflection symmetry of the value/derivative
matrices. In orthonormal even/odd coordinates, the value matrix is block diagonal
and the derivative is block off-diagonal. Each dense 1D contraction becomes two
smaller GEMMs, approximately halving arithmetic. The unchanged full metric
couples gradients at quadrature nodes, so this does not assume symmetric geometry.
The same blocks implement forward and transpose contractions. FP64 operator/CG,
FP32 preconditioner, degree-8 Fourier-enriched trial space and 1.5n quadrature remain.

Measured with the same rebuilt executable, one warmup and three timed solves per
configuration; the original path is selected with `BSPF_POISSON_PARITY=0`:

| n | q | Original CG (s) | Split CG (s) | Speedup | Physical relative L2 |
|---:|---:|---:|---:|---:|---:|
| 17 | 96 | 0.04854 | 0.04780 | 1.02x | 0.4222 |
| 33 | 96 | 0.09052 | 0.06805 | 1.33x | 0.1286 |
| 65 | 98 | 0.20776 | 0.15072 | 1.38x | 2.737e-12 |
| 97 | 146 | 0.84076 | 0.53977 | 1.56x | 1.937e-12 |
| 129 | 194 | 2.56219 | 1.49373 | 1.72x | 2.635e-12 |

At n=129 the full warm RHS/lift/solve/recovery time is 1.580 s. The true residual
is 8.01e-13 and CG still takes 18 iterations. Two independent error quadratures
(q=202,210) agree. Boundary error is zero. Relative nodal solution changes versus
the unsplit path range from 1.0e-15 to 1.12e-14. GPU buffers increase from 964 MB
to 1081 MB (12.1%); extra workspaces hold packed parity coordinates. The n=17
speed difference is too small to interpret as significant; clocks were not locked.

All 16 CUDA regression tests pass, including odd/even tensor dimensions, native
versus independent CPU operators, preconditioner checks, true residual restarts,
invalid inputs, the on/off switch, and automatic fallback for a nonsymmetric
basis. The n=17 polynomial guard has physical relative L2 error 1.923e-13 and
CPU/GPU operator discrepancy 2.52e-15.

Reproduce the audited split campaign with the README environment:

```sh
python bench/turbulence_3d/bspf_poisson/run.py --n 17 33 65 97 129 \
  --output bench/turbulence_3d/evidence/bspf_cuda3d_parity
```

For each saved `nN_qQ` directory, run the native executable with CUDA 13 libraries
and `BSPF_POISSON_PARITY=0`, passing `input.bin` and output prefix `unsplit`.
Then `python bench/turbulence_3d/bspf_poisson/report_parity.py` checks solution
agreement and produces CSV/JSON and the optimization plot. The main solver
comparison scripts prefer the new audited results. Each result includes the
binary SHA-256 and actual `parity_optimized` flag.

[Measurements](../evidence/bspf_cuda3d_parity/optimization.json) ·
[Optimization plot](../evidence/bspf_cuda3d_parity/optimization.png) ·
[Updated solver comparison](../evidence/fc_pytorch_rtx3070/comparison.png)

## Sharing gradient prefixes and reversing the operation graph

The next optimization is enabled by default with reflection splitting. The y
and z gradients share their initial x interpolation. On the transpose side,
the corresponding contributions are accumulated before applying the final x
interpolation transpose. This reduces 18 one-dimensional contractions to 16.
Only the smallest stage is eliminated, so at q/n ≈ 1.5 the arithmetic saving is
about 7%, not 2/18 of the total floating-point work. The full spatial metric,
quadrature, degree-8 Fourier-enriched space and precision settings are unchanged.
The asymptotic complexity remains O(k N^(4/3)) time and O(N) storage.

One extra n²q FP64 workspace costs 25.8 MB at n=129, about 2.4% above the previous
reflection-split allocation. Set `BSPF_POISSON_SHARED=0` to restore independent
contractions. Nonsymmetric input matrices and experimental direct-layout builds
continue to use their previous path. The actual choice is recorded as
`shared_contractions` in native JSON.

All 18 CUDA tests pass, including the new sharing on/off test against an
independent NumPy curved operator for odd/even dimensions. The campaign uses two
alternating off/on pairs per resolution, one warmup plus three measured solves
per invocation, and reports medians over six samples per mode. It reuses the
exact same input bundles and independently integrates the resulting Fourier-enriched
BSPF solution at two higher quadrature orders. GPU clocks are not locked;
roughly 1% differences on small grids are not evidence of a useful speedup.

Reproduce with `benchmark_shared.py`, then `report_shared.py` in this directory.
Measurements are in `../evidence/bspf_cuda3d_shared/optimization.json` and the
updated cross-solver comparison uses the per-resolution results there.

| n | Independent CG (s) | Shared CG (s) | Speedup | Physical relative L2 |
|---:|---:|---:|---:|---:|
| 17 | 0.04850 | 0.04789 | 1.013x | 4.222e-01 |
| 33 | 0.07092 | 0.07176 | 0.988x | 1.286e-01 |
| 65 | 0.15909 | 0.14976 | 1.062x | 2.737e-12 |
| 97 | 0.57200 | 0.53909 | 1.061x | 1.937e-12 |
| 129 | 1.47177 | 1.39276 | 1.057x | 2.635e-12 |

The maximum relative nodal solution change is 5.42e-16; boundary error is zero in every run.

## Fusing parity conversion with tensor rotations

The parity basis change and axis permutation can be composed into one indexing
operation: unpack directly into the rotated layout in the forward tree, and
pack directly from the rotated layout in the reverse tree. All contractions
and pairwise parity arithmetic remain unchanged. This removes intermediate
global-memory traffic and kernel launches, with no extra device allocation.
It does not change the O(k N^(4/3)) time or O(N) storage complexity.

On the RTX 3070, using the Gaussian-plus-plane-wave MMS and q = ceil(1.5n):

| Physical grid | Previous full solve (s) | Fused full solve (s) | Time reduction |
|---:|---:|---:|---:|
| 65³ | 0.151279 | 0.138009 | 8.8% |
| 97³ | 0.545141 | 0.504062 | 7.5% |
| 129³ | 1.466218 | 1.365535 | 6.9% |

Each entry is the median of nine warm solves across three alternating on/off
trials on identical inputs. GPU clocks were not locked. The final saved
solutions are bit-for-bit equal at all three sizes, and every measured solve
passed the true-residual threshold of 1e-11. The measurements and measured binary
hash are in `../evidence/gaussian_mms_rtx3070/bspf_fused_layout/summary.json`.
The subsequent rebuild adds only the public getter and JSON flag to that build.

The fused path is the default, with `BSPF_POISSON_FUSED_LAYOUT=0` as an A/B
fallback. Regression coverage includes independent CPU operator comparisons,
odd/even dimensions, and fusion on/off with shared contractions both enabled
and disabled.

Remaining candidates, not yet benchmarked, are reusing the CG residual norm
inside preconditioning to remove a redundant reduction/synchronization, and
capturing fixed operator work in CUDA Graphs to reduce launch overhead.
Compressing the curved metric into separable terms could reduce arithmetic,
but would require separate accuracy and positive-definiteness validation.
