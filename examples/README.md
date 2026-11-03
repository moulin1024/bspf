# Examples

High-level scripts using the Python bindings (`bspf_cpp`) on pybspf's benchmark
fields (`pybspf/examples/operation`). The fields, domains and settings are
collected in [`basic/benchmarks.py`](basic/benchmarks.py) with their exact
derivatives. Each script prints errors against the exact solution and a
refinement study; `--plot out.png` also saves a figure (needs matplotlib).

```sh
cmake --build build          # builds python/bspf_cpp/_core*.so
PYTHONPATH=python python examples/basic/diff_1d.py --plot diff_1d.png
```

| script | operation | field |
|---|---|---|
| `basic/diff_1d.py` | `Line.derivative`, spline-only comparison, batched fields | sin(x/(1.2 + cos x)) on [0, 2π] |
| `basic/integrate_1d.py` | `Line.quadrature(a, b)`, `Line.primitive` | same, from samples of f′ |
| `basic/interpolate_1d.py` | `Line.evaluate` values and derivatives off the grid | same |
| `basic/diff_2d.py` | `Grid.partial`, `Grid.gradient` | cosine modes + circular tanh front |
| `basic/diff_3d.py` | `Grid.gradient`, `curl`, `divergence`, `laplacian` | shifted Taylor–Green vortex, Chebyshev endpoints |

| `pde/wave_2d.py` | linear wave equation as a first-order system + RK4, Dirichlet walls; checks the clamped spectrum is real | two crossing plane waves, time-dependent wall data |
| `pde/heat_2d.py` | time-dependent heat equation: `Grid.laplacian` + RK4, Dirichlet walls | decaying mode with non-periodic, time-dependent wall data |

The 2D example draws its random mode amplitudes with NumPy (seed 123), so the
field differs slightly from pybspf's JAX-generated one.
