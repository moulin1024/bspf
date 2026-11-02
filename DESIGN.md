# bspf: design

A small, fast C++20 implementation of the BSPF (B-spline plus Fourier) calculus
on uniform 1-3D rectangular grids, with a thin Python module. pybspf is the
functional reference (same mathematics, verified by parity tests), not an API
template. License: GPL-3.0-or-later. Started 2026-11-02.

## Scope

Clean-data (noise-free) differentiation, interpolation (with derivatives),
definite integration and first/second primitives, on 1D lines and on 2D/3D
tensor-product grids, with optional user-supplied endpoint jets.
Not in scope: noisy-data plans, complex fields, autodiff, GPU.

Dependencies: C++20, Eigen (header-only), FFTW3; nanobind at build time for Python.

## The 1D construction (as in pybspf)

On a uniform grid `x_0 < ... < x_{N-1}` a field is split as `f = B c + r`:

* `B` holds clamped B-splines of degree `d` (open knots with optional tanh
  clustering, or user knots).
* `c` solves `min ||W^{1/2}(B c - f)||^2 + lam ||c||^2` subject to `C c = g`,
  with `C` the derivatives `0..q-1` at both ends and `g` endpoint jets,
  estimated by a linear map `J` of the first/last `p` samples (finite-difference
  stencils or a regularized local Chebyshev fit) unless given.
* `r` is treated as periodic with period `N dx`, frequencies in numpy `fftfreq`
  order (even-N Nyquist negative), real part of the full sum.

## Design: compile operators, then apply them

Every clean BSPF operation is a fixed linear map of the samples (affine in
given jets). So the library compiles each query once and applies it many times.

* `Line` (immutable, shared): grid, knots, basis, endpoint blocks and the fit
  operators `c = F f + G g` from one LU of the KKT matrix
  `[[2(G + lam I), -C^T], [C, 0]]`.
* `Op`, made by a `Line`:
  * `derivative(k)` (N, N)
  * `evaluate(points, k)` (P, N), with the points fixed and checked when the op is built
  * `quadrature(a, b)` (1, N)
  * `primitive(order)` (N, N)
  * `jets()` (2q, N)

### One application path

Every op runs the same steps, which are BSPF's split, kept at apply time:

    c = K (2 B^T W f) + G g      fit (K, G: top rows of KKT^{-1}; g = J f unless given)
    r = f - B c                  remainder
    R = rfft(r)
    y = S c + irfft(m R) + L Re(E R)

* `S`: the op's spline rows.
* `m`: a multiplier on the half spectrum, (i omega)^k / N for derivatives and
  1 / (i omega)^k for primitives.
* `E`: fixed half-spectrum rows. They give the remainder at the evaluation
  points, its integral over [a, b], or the integration constants of primitives.
* `L`: lifts the rows of `E` onto the grid; it is the identity for points and
  integrals.

Accuracy: `omega^k` only ever multiplies the small remainder `r`. That is why
there is no precomputed single-matrix form. Measured against a 40-digit
evaluation at N = 128, degree 7, 4th derivative: this path 4.3e-9, pybspf 4.5e-9,
a fused matrix `A = S F + M (I - B F)` 2.1e-7. For the same reason the estimated
jets stay a separate factor `G (J f)` and are not folded into the fit.
`Op::matrix()` assembles `A` by applying the op to the identity, when an
explicit matrix is needed.

Efficiency:

* `B`, `B^(k)` and the projection `2 B^T W` are banded: each B-spline row has
  `degree + 1` nonzeros. They are applied as banded gathers.
* Lines are processed in blocks of about 8K samples, held in row-major
  (sample-major) workspaces, so every inner loop runs contiguously over lines
  and each block's FFT stays in cache.
* Once the fit has consumed a gathered sample block, that block becomes the
  residual buffer and then the inverse FFT output. The caller's samples remain
  read-only; there is no separate residual copy.
* Workspaces belong to one application and are reused across its blocks,
  spectral groups and axis slabs. No mutable scratch is stored on an `Op`, so
  concurrent applications remain independent. Eigen may still reallocate when
  block dimensions change (for example, a short final block).
* FFTW plans are built with FFTW_MEASURE, once per (batch, layout).
* Point evaluations collect spectra in groups of at most about 32K samples
  (rounded to whole line blocks) and apply `E` with a GEMM per group. This bounds
  the spectrum and its real/imaginary copies independently of the field volume.
  Integral rows are applied immediately within each line block: a single output
  row does not benefit from storing whole-field spectra.

### Tensor grids

Fields are C-ordered arrays; the leading axes are spatial (x, y, z) and any
trailing axes are batches/components. `apply_along(op, field, axis)` runs
without transposing the data:

* innermost axis: contiguous lines gathered into cache-local blocks;
* other axes: interleaved lines processed slab by slab with a shared call-local
  workspace. Each block applies the same split kernel.

`Grid` caches derivative ops per axis and order. It provides mixed partials,
tensor-mesh interpolation (with per-axis derivatives), box integrals, and a
native Laplacian. The Laplacian writes the first second partial into its output
and adds subsequent partials directly from their kernel blocks, avoiding full
partial-field temporaries and Python array sums. In Python it also provides
gradient, divergence, hessian and curl.

### 3D memory-traffic fix (2026-11-02)

`bench/bench_3d.py` measures warmed calls in separate processes against the saved
original extension, with one thread and five batches of fifteen calls. The
results are in `bench/results/3d_before.json` and `3d_after.json`. On this machine:

| 96³ operation | Before | After |
| --- | ---: | ---: |
| Laplacian | 28.84 ms | 27.68 ms |
| Interpolation, precompiled ops | 33.20 ms | 29.45 ms |
| Grid.interpolate, including op construction | 35.41 ms | 29.88 ms |
| Box integration | 8.43 ms | 6.26 ms |

The main interpolation/integration issue was whole-axis spectral scratch space,
not just the tensor intermediates. At 96³ the first interpolation axis formerly
stored 49 × 9216 complex values plus their real/imaginary copies. Groups now
contain at most 320 lines at this size. Integral rows no longer retain spectra
beyond the current block. Tensor intermediates still exist between axes.

Native accumulation eliminates full Laplacian partial-field temporaries, but
most of its runtime remains in the three derivative kernels. It does not by
itself eliminate the speed gap versus jitted pybspf. Small timing differences
should be read loosely on a shared machine.

The full comparison is saved separately as
`bench/results/vs_pybspf_3d_fix.json`, preserving the previous results. At 96³,
the updated C++ / jitted pybspf times were 27.64 / 24.78 ms for Laplacian,
38.20 / 32.10 ms for precompiled interpolation, and 6.62 / 10.68 ms for
integration. Integration now wins; Laplacian and interpolation still trail.
The focused timings above and the full JAX comparison run in different
processes and have different memory/load conditions; use each comparison
within its own run.

### Buffer and workspace reuse (2026-11-02)

Two local changes retain the same arithmetic and public API: reuse the gathered
sample block after fitting, and carry a call-local workspace through spectral
groups and middle-axis slabs. The workspace also retains the gathered blocks
and deferred spectral-product temporaries. No persistent or thread-local cache
is added.

The focused benchmark was run against three saved builds, then repeated in
reverse order, with no tests running alongside the measurements. At 96³:

| Laplacian build | First run | Reverse-order repeat |
| --- | ---: | ---: |
| Before reuse | 28.85 ms | 28.29 ms |
| Buffer reuse only | 27.91 ms | 27.16 ms |
| Buffer and workspace reuse | 27.41 ms | 26.98 ms |

Both runs show about a 5% reduction from the combined changes. The incremental
workspace benefit is smaller and should be read cautiously. This does not
establish parity with JAX: the previous JAX timing was 24.78 ms, and JAX was not
retimed in this experiment. Raw results are in
`bench/results/3d_reuse_{before,buffer,both}{,_repeat}.json`; the benchmark is
`bench/bench_3d.py`.

Regression coverage includes read-only inputs and jets, repeated slabs with a
short final block, derivatives, primitives, spline-only evaluation, point
evaluation, quadrature, endpoint jets, and simultaneous calls sharing one op.

### Execution

Ops are immutable and safe to share across threads. FFTW plans are cached per
batch size; planning happens under a process-wide mutex, while execution with
new arrays is lock-free. Single-threaded by default.

## Layout

    include/bspf/   bspline, endpoints, line, op, grid (public headers)
    src/            implementations, private LineData/FFT, nanobind module
    python/bspf_cpp Python package (Line, Op, Grid + vector calculus helpers)
    tests/cpp       C++ consistency checks (ctest)
    tests/python    parity with pybspf, 40-digit accuracy reference
    bench/          C++ per-sample cost; throughput vs jitted pybspf

## Validation

* `tests/python/test_parity.py` covers 9 configurations: degrees 3–7, lam,
  clustering, n_basis, custom q/p, and Chebyshev endpoints, each at N = 48, 63
  and 128.
  * Operations: derivatives 0–4 with and without
    correction, given jets, evaluation, integration, primitives, every axis,
    1–3D grids and vector calculus.
  * Tolerance: a multiple of pybspf's own rounding floor, measured by perturbing
    the input at the 1e-16 level.
* `tests/python/test_accuracy.py` checks that the op rounds no worse than
  pybspf against an mpmath evaluation of the same map.

* `tests/python/test_large_batches.py` covers group boundaries, native Laplacian
  accumulation, input immutability, scratch reuse and concurrent calls.

Validation of the consolidated implementation: 149 Python tests and CTest pass.
