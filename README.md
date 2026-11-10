# bspf

C++20 BSPF (B-spline plus Fourier) calculus on uniform 1-3D rectangular grids,
with mapped and conforming multipatch grids in 2D: differentiation,
interpolation, integration and primitives on tensor grids, and physical
derivatives on mapped grids. Dependencies are
Eigen and FFTW, plus an optional thin nanobind module. GPL-3.0-or-later.

The C++ library and namespace are `bspf`; the Python bindings are imported as
`bspf_cpp`. Operators preserve the spline/Fourier split during evaluation.
The native Laplacian accumulates directly into one output, and each application
reuses its private buffers across blocks and slabs.

See [DESIGN.md](DESIGN.md) for construction, accuracy, and measured performance.

## Build

```sh
brew install eigen fftw            # or your package manager
python -m pip install nanobind     # only for the Python module
cmake -S . -B build -DPython_EXECUTABLE=$(which python)
cmake --build build -j
ctest --test-dir build             # C++ checks
```

Optional FP32 CUDA backend (cuBLAS + cuFFT; NVIDIA GPU and CUDA toolkit):

```sh
cmake -S . -B build -DBSPF_CUDA=ON -DCUDAToolkit_ROOT=/path/to/cuda   # CMAKE_CUDA_ARCHITECTURES defaults to native
```

The module is written to `python/bspf_cpp/_core*.so`, so `PYTHONPATH=python`
makes `import bspf_cpp` work. Options: `-DBSPF_PYTHON=OFF`, `-DBSPF_TESTS=OFF`,
`-DBSPF_NATIVE=OFF` (disable `-march/-mcpu=native`).

## C++

```cpp
#include "bspf/grid.hpp"

bspf::Line line(x, {.degree = 5});
bspf::Op d1 = line.derivative(1);          // compile once
bspf::Matrix df = d1(f);                   // f: (N, m) column-major
bspf::Matrix v = line.evaluate(points)(f); // (P, m)
double area = line.quadrature(a, b)(f)(0, 0);

bspf::Grid grid({bspf::Line(x), bspf::Line(y), bspf::Line(z)});
bspf::Field u = ...;                       // C-ordered (nx, ny, nz, *batch)
bspf::Field uxy = grid.partial(u.view(), {1, 1, 0});
bspf::Field total = grid.integrate(u.view());
bspf::Field lap = grid.laplacian(u.view());
```

## Python

```python
import numpy as np
import bspf_cpp as bc

line = bc.Line(x, degree=5)                 # chebyshev: endpoint_method="chebyshev"
d1 = line.derivative(1)                     # Op; d1.matrix is the dense (N, N) operator
df = d1(f, axis=1)                          # any axis of a float64 array
df = d1(f, axis=0, jets=g)                  # given endpoint jets, 2q along that axis
values = line.evaluate(t, derivative=0)(f)
integral = line.quadrature(a, b)(f)[0]

grid = bc.Grid.uniform((x, y))
grid.partial(u, (1, 0)); grid.laplacian(u); grid.gradient(u)
grid.interpolate(u, (xs, ys)); grid.integrate(u, bounds=[(0, 0.5), (0, 1)])
```

### GPU (float32)

With `-DBSPF_CUDA=ON`, `bc.cuda` runs the same operators on an NVIDIA GPU in
float32, 40-300x faster than one CPU core when fields stay on the device:

```python
gpu = bc.cuda.Grid(grid)                    # from a bc.Grid
u = bc.cuda.Array(f)                        # upload (float64 -> float32)
lap = gpu.laplacian(u)                      # device Array; also partial, interpolate, integrate
lap.get()                                   # download as float64
gpu.laplacian(f)                            # numpy in, numpy out (includes transfers)
d1 = bc.cuda.Op(line.derivative(1)); d1(u, axis=0)
```

Float32 rounding grows with derivative order and N: values, integrals and first
derivatives are accurate to ~1e-7 and ~1e-5; second derivatives reach
1e-3..1e-2 at N >= 256. See DESIGN.md for the measurements.

For time-dependent problems with Dirichlet walls, `grid.weak_laplacian(u,
wall_values)` is the energy-stable (Galerkin) Laplacian on the same BSPF trial
space: its spectrum is real for any endpoint jets, so wave schemes stay stable.
`wall_values` are the result's values on the walls (for example the boundary
data's time derivative); see DESIGN.md.

Jets are ordered left `f, f', ..., f^(q-1)`, then right. Evaluation points and
integration bounds outside the grid raise `ValueError` when the op is built.

For nonlinear Galerkin PDEs, the structured quadrature and mass kernels are
also available directly:

```python
gal = bc.GalerkinLine(line, quadrature_order=8)  # default: 16
points, weights = gal.quadrature_points, gal.quadrature_weights
uq = gal.evaluate(u)                        # u: (N, batch)
uxq = gal.evaluate(u, derivative=1)
load = gal.project(flux, derivative=1)       # Phi_x.T @ (weights * flux)
du = gal.solve_mass(load)                   # full mass, (N, batch)
du_inner = gal.solve_mass(load[1:-1], interior=True)
```

Quadrature rows are compact and not necessarily sorted. Evaluation uses shifted
FFTs plus spline corrections; projection is its weighted adjoint. The full
mass inverse uses the existing low-rank interior inverse and a 2x2 wall Schur
solve. The products and inverses use the selected quadrature order; check
quadrature convergence for nonlinear fluxes. The compressible
[`shock_tube_1d.py`](examples/pde/shock_tube_1d.py) example uses these kernels
by default and retains `--backend dense` for validation.

## Mapped and multipatch differentiation

`MappedGrid2D(bottom, right, top, left, shape, **options)` builds a transfinite
map from four `SplineCurve` boundaries. Fields have shape `(n_xi, n_eta, *batch)`.
`partial(f, (ox, oy))`, `gradient(f)` and `laplacian(f)` return physical
derivatives. First and second derivatives use analytic map metrics and direct
reference derivatives. The grid exposes `x`, `y`, `jacobian`, and first map
derivatives. Optional `dist_xi`, `dist_eta` (or per-edge `dist_*`) Bernstein
distributions control node placement; `correction=(cx, cy)` adds an interior
Bernstein correction that vanishes on every edge.

`MultiPatch2D(patches, interfaces, overlap=8, stencil=12)` joins complete faces.
Each interface is `((patch, axis, side), (patch, axis, side), reverse)`;
`periodic=` accepts the same tuples with an additional `(dx, dy)` translation.
Fields and results are lists, one array per patch. Interface-crossing lines
use neighbour samples interpolated onto ghost points. `overlap=0` selects
one-sided patch derivatives. Shared results are averaged by default;
`partial(fields, orders, average_shared=False)` preserves the separate traces
for inspection with `shared_jump`. Ghosts remain within the assembled patches;
physical boundary lines remain one-sided. See the runnable
[mapped](examples/basic/diff_mapped_2d.py) and
[multipatch](examples/basic/diff_multipatch_2d.py) examples.

For high-order endpoint fits, `constraint_order`, `chebyshev_modes` and
`boundary_points` are independent controls. Increasing the mode count without
enough endpoint samples can amplify rounding error. A tested configuration
for smooth mapped first derivatives is:

```python
options = dict(degree=9, n_basis=18, lam=1e-6, constraint_order=9,
               endpoint_method="chebyshev", chebyshev_modes=16,
               boundary_points=24, chebyshev_alpha=1e-12)
line = bc.Line(np.linspace(0, 1, 129), **options)
```

Use a refinement study for each field and map; these settings are not an
accuracy guarantee or a change to the library defaults. The local
rectangle/cylinder studies used 16, 24 and 32 endpoint samples at line sizes
65, 129 and 257, respectively. Their analytic AAA/lightning geometry and
patch-wise BSPF/Fourier operators are a separate research implementation,
not the ghost-overlap `MultiPatch2D` algorithm.

## Examples

[`examples/`](examples/README.md) has runnable scripts on pybspf's benchmark
fields: 1D differentiation, integration and interpolation, 2D partials and 3D
vector calculus, plus time-dependent heat and wave equations with
non-periodic Dirichlet walls (`examples/pde`). Start with `PYTHONPATH=python python examples/basic/diff_1d.py`.

## Tests and benchmarks

The release consists of `include/`, `src/`, `python/`, `tests/`, `examples/`
and the small `bench/` microbenchmarks. Local research under `experimental/`,
case data under `benchmarks/` and `vki_ls89/`, and `writing-block.md` are
Git-ignored and are not dependencies of the library, examples or tests.
AAA/lightning meshing, triangular residual prototypes and Nektar comparisons
remain local experiments; their results are not release performance guarantees.

The tests include parity with pybspf and 40-digit accuracy checks, plus
large batches, explicit jets, read-only inputs, concurrent calls, the
energy-stable Galerkin Laplacian (exact products, real spectrum, wave energy)
and execution of every example. Mapped and multipatch tests cover geometry,
physical derivatives, interface orientation, overlap and periodic topology.


```sh
python -m pip install numpy pytest mpmath matplotlib
# Full parity suite also needs JAX and the separate pybspf reference package.
PYTHONPATH=python:/path/to/pybspf/src python -m pytest tests/python
./build/bench_bspf                                    # ns per sample, both layouts
PYTHONPATH=python:/path/to/pybspf/src python bench/bench_vs_pybspf.py
PYTHONPATH=python python bench/bench_3d.py bench/results/3d.json
```

Without the optional JAX/pybspf reference, run the standalone suite with:

```sh
PYTHONPATH=python python -m pytest tests/python --ignore=tests/python/test_parity.py --ignore=tests/python/test_accuracy.py
```

For a viscous time-stepping stiffness diagnostic:

```sh
PYTHONPATH=python python bench/bench_viscous_stiffness.py --nu 0.01 --json viscous_stiffness.json --plot viscous_stiffness.png
```

This measures homogeneous-Dirichlet diffusion spectra on [0,1] for BSPF
Galerkin, FFT Chebyshev collocation, and Chebyshev Galerkin. The latter uses
exact polynomial mass/stiffness integration, so the comparison checks both
collocation and weak Chebyshev formulations. Dense matrices are used only for
the eigenvalue diagnostic. BSPF uses the tested Galerkin configuration with
degree 9, 18 spline basis functions, constraint order 4, 16 endpoint samples,
12 Chebyshev endpoint modes and lam=1e-6; full settings are saved in JSON.

At 257 nodes, nu=0.01, the RK4 negative-real-axis eigenvalue limits are
3.893e-4 (BSPF), 3.423e-7 (Chebyshev collocation), and 6.252e-7 (Chebyshev
Galerkin). This demonstrates reduced diffusion stiffness at equal node counts,
not equal-error efficiency, a measured time-stepping speedup, or a validated
compressible shock-tube solver. The diagnostic checks stable spectra and the
first Dirichlet eigenvalue, and reports a sine-mode differentiation error.
Full viscous-flow comparisons must additionally resolve the same physical
layers, check nonlinear stability/conservation/positivity, and include implicit
or IMEX Chebyshev time stepping when making broader efficiency claims.
