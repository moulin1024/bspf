# Persistent 257³ optimization campaign

Target: at least 2× faster warm end-to-end solves on the fixed curved Gaussian
MMS at n=257, q=386. Keep FP64 operator/CG, the existing FP32 preconditioner,
trial space, geometry, RHS and 1e-12 stopping tolerance. Independent physical
error checks must preserve the approximately 3.05e-12 relative L2 result.

The initial measured warm solve is 1.375 seconds on an A100-SXM-64GB. The
formal target uses a fresh same-GPU alternating baseline, rather than this
historical timing alone. Baseline binary and source snapshots are in
`.toolchains/leonardo/optimization257/baseline/`; the binary SHA256 is
`d9a64c8260ff2edbb32612fd613711a9d9bc8456dd383b065be6e2aeeb15e201`.
Inputs are the existing `mass_sweep_leonardo_64616443/bspf/n257_q386` bundle.

## First candidate: coalesced tensor rotations

The existing cyclic tensor rotation reads contiguous memory but scatters its
writes. Replace it with a 32×32 shared-memory tiled matrix transpose, with
256-thread blocks, covering FP64 operator rotations and FP32 preconditioner
rotations. The inverse rotation uses the same transpose with reversed matrix
dimensions. No floating-point arithmetic or operator is changed.

`BSPF_POISSON_TILED_ROTATE=0` retains the previous kernel for diagnostics.
The new path is the default. The independent campaign build leaves the
production executable untouched until validation.

The baseline has reflection, shared contractions and fused layouts disabled:
relative reflection defects are 1.51e-12 for B and 1.81e-12 for D, just beyond
the 1e-12 eligibility threshold. This is a separate potential optimization;
the first candidate deliberately keeps the original tables and eligibility.

Submit `sbatch bench/turbulence_3d/leonardo/optimization257.slurm` from the
repository root. It builds the candidate separately and compares three
alternating pairs of complete native executions, with one warmup and three
measured solves per execution. Results include the input/binary hashes,
GPU, all samples, solution and operator comparisons, true residuals, and
independent physical error audits at q=394 and q=402. Nsight kernel reports
are collected separately if available. Setup, profiling and error audits
are excluded from warm solve timings, consistently for both binaries.

Job 64616934 validated this first candidate: baseline median 1.380338 s,
candidate 0.836191 s, **1.6507×** faster over nine warm samples each. Operator,
preconditioner and solution outputs were bitwise identical. Independent
physical L2 error was 3.05455e-12. Nsight measured FP64 rotation time falling
from 2.662 s to 0.621 s across four profiled solves, and FP32 rotation time
from 0.301 s to 0.052 s. The remaining dominant work is FP64 GEMM.

## Second candidate: restore and tile reflection transfers

Job 64617047 builds separately in `optimization257/parity_tiled`. It extends
the tiled transpose to the fused reflection pack/unpack kernels. Eligibility
for reflection splitting uses a bounded axis-size roundoff allowance:
`min(1e-11, max(1e-12, 32*eps*max(n,q)))`. This is 2.74e-12 at n=257,q=386,
so the observed 1.81e-12 table defect qualifies. CG tolerance and arithmetic
precision are unchanged. Acceptance requires original-operator comparisons,
baseline solution comparison, and the same independent physical error audits.

Result: 0.889932 s versus 1.374683 s, **1.5447×**. The operator probe changed
by 1.19e-12 relative and the solution by 1.73e-14; physical audits passed.
Repeated reflection packing/unpacking outweighed some GEMM savings.

## Third candidate: retain reflection sectors throughout the operator

Job 64617515 builds `optimization257/octants`. Apply one orthonormal reflection
transform on entry and one on exit. Each of eight sectors then uses reduced
one-dimensional B/D blocks throughout gradient, metric multiply and divergence.
For even q, all quadrature sectors have q/2 entries per axis. Nodal center
planes for odd n belong only to even sectors and retain unit normalization.

This requires G_aa to be even in every coordinate, and G_ab (a != b) to be
odd in a and b and even in the third coordinate. Setup checks all six metric
components against all reflected points; a failed check or odd q falls back
to the general path. Derivatives toggle the corresponding sector parity;
the metric maps these gradient sectors back to the appropriate flux sectors.
Orthonormal reflection normalization means no additional factor of eight.
`BSPF_POISSON_OCTANTS=0` disables this optimization for comparisons.

Result: median 0.729529 s against 1.377986 s, **1.8889×**. Independent error
is 3.05428e-12; original-operator probe difference is 1.19433e-12 and solution
difference 1.73151e-14. Job 64617568 passed 39 GPU regression tests, including
odd/even axes and rejection of asymmetric metrics from the octant path.

## Subsequent candidates

Job 64617653 shares the x interpolation for the y/z derivatives and their
final adjoint contraction. It passes the same 39 tests and physical audits,
but is slower (approximately 0.763 s): reverse-order transposes increase
memory-transfer cost despite saving contractions. Keep the unshared path
available through `BSPF_POISSON_SHARED=0`.

Jobs 64617782 and 64617808 tested padded sector matrices.
At n=257,q=386, nodal sector strides become 136 and quadrature strides 200,
both multiples of eight. All added basis entries, nodal entries and flux
entries are zero. No additional physical degrees of freedom or quadrature
points were introduced. These candidates were slower (0.827 and 0.812 s),
so padding and its extra allocations were removed from the final code.
The experiment remains reproducible from each job's `candidate.cu` snapshot.

Job 64617899 batched the metric multiply across all eight sectors, loading
each metric entry once. It reached 0.700518 s against 1.377992 s (**1.9671×**).
Job 64618054 retained the final gradient GEMM's (z,x,y) layout through the
metric multiply and into the adjoint, removing six large tensor rotations
per sector. Shared contractions then became profitable. Its result was
0.628606 s against 1.378449 s (**2.1929×**), with physical audits unchanged.

## Final validation and installation

Job **64618164** validates the cleaned implementation with default settings:

| Measurement | Original | Optimized |
| --- | ---: | ---: |
| Median warm solve, nine samples | 1.378296126 s | 0.630042090 s |
| CG iterations | 18 | 18 |
| Speedup | 1× | **2.187625×** |

All **60 GPU tests passed**. They cover odd/even nodal and quadrature sizes,
n=65, independent CPU operator comparisons, polynomial solutions, zero loads,
invalid inputs, residual restarts, iteration limits, asymmetric-axis/metric
fallbacks, and shared/unshared and rotated/unrotated layout equivalence.
The fixed 257³ campaign also checks:

- Relative solution difference from baseline: 1.73184e-14.
- Original unsplit operator probe difference: 1.19433e-12, within the 2e-12 guard.
- Preconditioner probe: bitwise identical.
- Maximum independently recomputed full lifted residual: 4.98894e-13.
- Physical relative L2 at q=394/402: 3.0542869e-12 / 3.0542804e-12.

The production executable `.toolchains/leonardo/build/bspf-poisson/bspf_poisson3d`
is the exact tested final binary, SHA256
`c5666e795e04d3abd87f2f4943d46977ea90eca05eaaddf38d8a9738efe3c349`.
The normal build and static library were also rebuilt from the final source.
The original executable remains preserved in `optimization257/baseline`.
The optimized operator is enabled by default for eligible geometry.

Evidence is in `../evidence/optimization257_64618164/`: `regression.txt`,
`timing/report.json`, per-invocation native JSON and solution/probe bundles,
source snapshot, GPU/module details, and separately collected Nsight reports.
Setup, input transfer, and independent audits are excluded consistently from
both warm timings. Each solve starts from zero interior correction; there is
no solution reuse or cached boundary-lift solve. Compact metric arrays add
approximately 345 MB of device storage at this size; total reported storage
is 9,076,440,176 bytes.

Reflection splitting uses the bounded axis-table roundoff allowance described
above. It is not bitwise identical to the unsplit operator. The >=2× claim is
for this reflection-symmetric curved-domain benchmark; other geometries and
odd q use the validated general path and have separate performance behavior.

Status: target achieved and validated.
