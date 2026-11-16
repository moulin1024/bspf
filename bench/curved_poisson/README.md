# Curved-domain scalar Poisson campaign

This self-contained benchmark bundle implements -Δu=f with nonzero Dirichlet
boundary values on a nonconformal quadrilateral bounded by four cubic splines.
It contains the optimized native Galerkin BSPF solver (direct or nodal RHS), an
in-house native fourth-order finite-difference baseline, and the stock Nektar++
ContField::HelmSolve adapter. It is a scalar Poisson experiment, not a flow or
acoustic evolution solver. It does not alter the main bspf library API.

## Build and check

The measured BSPF executables use macOS Accelerate, dynamically loaded from
its system framework. FD4 additionally requires FFTW3. Preparation and audits
require Python with NumPy, SciPy and Matplotlib. No external Python checkout,
private scratch module, or downloaded benchmark data is required.

From this directory:

```sh
python3 -m pip install -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build build -j 3
PYTHONDONTWRITEBYTECODE=1 python3 check.py
```

To build the optional Nektar++ adapter, configure with
`-DCURVED_POISSON_NEKTAR=ON -DNektar++_DIR=/path/to/nektar/cmake`.
The recorded campaign used Nektar++ 5.10.0, DirectStaticCond and SumFac.
No upstream numerical kernel was modified. MFEM is not required by this bundle.

## Reproduce and inspect

```sh
python3 benchmarks/nonconformal_bspf_opt/run.py --hard --n 248 --forcing-source direct
python3 benchmarks/nonconformal_bspf_opt/run.py --hard --n 248 --forcing-source nodal
python3 benchmarks/finite_difference_poisson/run.py --kind smooth --n 17 33 65 129
python3 benchmarks/finite_difference_poisson/run.py --kind hard --n 65 129 257 513 1025
python3 benchmarks/nonconformal_bspf_opt/fine_sweep.py
python3 benchmarks/nonconformal_bspf_opt/report_fine_sweep.py
python3 benchmarks/finite_difference_poisson/report.py
```

Case result files are reused, never silently retimed. The committed records
therefore let reporting run immediately, without rerunning PDE solves. For a
fresh timing campaign, use another copy of this directory and move its
`evidence` directory aside first. The MMS coefficients regenerate from seed
20261115 (shells 4–128, RMS 1, shell energy k^(-5/3)); NumPy versions may affect
bitwise generation even with the same seed. Source field hashes in the archived
protocol identify the exact field used for the reported measurements.

BSPF uses q=2n Gauss nodes per axis. Direct RHS samples analytic forcing there;
nodal RHS samples only at BSPF nodes and interpolates inside the timer. Nektar++
samples the same analytic forcing at its element quadrature nodes. Thus the
direct policies match analytic information access, not sample counts. The
nodal policy deliberately includes reconstruction error and cost.

Each configuration has one single-thread native solve, no warmup or timing
repetition. Setup, factorization, analytic sampling and error integration are
excluded. Boundary lifting, RHS processing, iteration and residual checks are
included. BSPF uses matrix-free tensor contractions in a generalized eigenbasis
and defect correction. FD4 uses local fourth-order boundary closures and
BiCGSTAB with an FFTW sine-transform Poisson preconditioner; no dense PDE solve.

## Results

- [Fine-sweep report](evidence/nonconformal_bspf_opt/hard_shell128/fine_sweep_v1/report.md)
- [Accuracy versus wall time](evidence/nonconformal_bspf_opt/hard_shell128/fine_sweep_v1/accuracy_wall_time.png)
- [Accuracy versus resolution](evidence/nonconformal_bspf_opt/hard_shell128/fine_sweep_v1/accuracy_resolution.png)
- [Native FD4 report](evidence/finite_difference_poisson/v1/report.md)

The fine sweep has 25 BSPF resolutions per RHS policy and 35 Nektar++
configurations. Failed audits remain recorded and are excluded from target
selection. The wall-time plot is a best-measured staircase, not an interpolation
of target-crossing times. All timings are single observations; small differences
are not statistically established speedups. FD4 is an in-house baseline, not a
claim to represent the fastest established finite-difference implementation.

Only compact result records and figures are versioned. Large input bundles,
solutions, basis/audit caches, binaries, third-party builds and temporary logs
are excluded. Archived binary hashes describe the original measured builds;
rebuilding from these sources need not produce byte-identical executables.
