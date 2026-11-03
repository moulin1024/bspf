# bspf

C++20 BSPF (B-spline plus Fourier) calculus on uniform 1-3D rectangular grids:
differentiation, interpolation, integration and primitives. Dependencies are
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

For time-dependent problems with Dirichlet walls, `grid.weak_laplacian(u,
wall_values)` is the energy-stable (Galerkin) Laplacian on the same BSPF trial
space: its spectrum is real for any endpoint jets, so wave schemes stay stable.
`wall_values` are the result's values on the walls (for example the boundary
data's time derivative); see DESIGN.md.

Jets are ordered left `f, f', ..., f^(q-1)`, then right. Evaluation points and
integration bounds outside the grid raise `ValueError` when the op is built.

## Examples

[`examples/`](examples/README.md) has runnable scripts on pybspf's benchmark
fields: 1D differentiation, integration and interpolation, 2D partials and 3D
vector calculus, plus time-dependent heat and wave equations with
non-periodic Dirichlet walls (`examples/pde`). Start with `PYTHONPATH=python python examples/basic/diff_1d.py`.

## Tests and benchmarks

The accepted implementation passes 190 Python tests (including parity with
pybspf and 40-digit accuracy checks) and the C++ checks. The tests also cover
large batches, explicit jets, read-only inputs, concurrent calls, the
energy-stable Galerkin Laplacian (exact products, real spectrum, wave energy)
and run every example.


```sh
PYTHONPATH=python:/path/to/pybspf/src python -m pytest tests/python
./build/bench_bspf                                    # ns per sample, both layouts
PYTHONPATH=python:/path/to/pybspf/src python bench/bench_vs_pybspf.py
PYTHONPATH=python python bench/bench_3d.py bench/results/3d.json
```
