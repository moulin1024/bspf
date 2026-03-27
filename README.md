# `bspf`

`bspf` provides B-spline plus spectral/Fourier operators for uniformly sampled
one-dimensional data.

The package API under [`src/bspf`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/src/bspf) is now the preferred interface. The root-level legacy module [`bspf1d.py`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/bspf1d.py) remains in the repository as a compatibility reference while package-native implementations continue to replace legacy internals under regression tests.

## What The Package Does

The core workflow is:

1. fit a constrained B-spline representation to sampled data,
2. compute derivatives or related quantities from the spline,
3. correct the spline residual with FFT-based spectral operations,
4. optionally split the domain when discontinuities are known.

The main user-facing objects are:

- `BSPF1D`: 1D operator for spline fitting, differentiation, integration, and interpolation
- `PiecewiseBSPF1D`: segmented wrapper for piecewise-smooth functions with known breakpoints
- `Grid1D`: uniform grid abstraction used by the operators

## Current Status

- `src/bspf` is the preferred API and owns construction, package module boundaries, backend helpers, and piecewise orchestration.
- The root-level [`bspf1d.py`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/bspf1d.py) is still kept as the legacy compatibility path.
- Regression tests compare package behavior to the legacy implementation during migration.
- A concrete refactor history and remaining migration strategy are documented in:
  - [docs/refactor_backlog.md](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/docs/refactor_backlog.md)
  - [docs/compatibility_strategy.md](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/docs/compatibility_strategy.md)
  - [docs/design.md](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/docs/design.md)

## Installation

Editable install:

```bash
pip install -e .
```

Development install:

```bash
pip install -e .[dev]
```

Optional GPU dependencies:

```bash
pip install -e .[gpu]
```

## Quick Start

```python
import numpy as np

from bspf import BSPF1D

x = np.linspace(0.0, 2.0 * np.pi, 128)
f = np.sin(x) + 0.1 * np.cos(3.0 * x)

op = BSPF1D.from_grid(degree=5, x=x)
df1, df2, f_spline = op.differentiate_1_2(f, lam=0.01)
```

Piecewise example:

```python
import numpy as np

from bspf import PiecewiseBSPF1D

x = np.linspace(0.0, 1.0, 256)
f = np.where(x < 0.4, np.sin(8.0 * x), np.sin(8.0 * x) + 1.0)

op = PiecewiseBSPF1D(degree=5, x=x, breakpoints=[0.4], min_points_per_seg=32)
df1, df2, f_spline = op.differentiate_1_2(f, lam=0.01)
```

## Public API

Preferred imports:

```python
from bspf import BSPF1D, PiecewiseBSPF1D, Grid1D
```

Compatibility alias:

```python
from bspf import bspf1d
```

Legacy path still available:

```python
from bspf1d import bspf1d, PiecewiseBSPF1D
```

See [docs/api.md](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/docs/api.md) for the current API summary.

## Supported Features

- uniform-grid 1D operators
- automatic or explicit knot generation
- first, second, and third derivatives
- batched first/second derivatives
- definite integrals and first/second antiderivatives
- spline fitting and interpolation
- piecewise differentiation for known discontinuities
- optional GPU mode when CuPy is available

## Known Limits

- the grid must be uniform
- GPU mode requires CuPy and explicit backend-consistent arrays
- true CPU/GPU parity testing is not currently exercised in this repository because the local test environment does not include CuPy
- some operation-family implementations still delegate internally to legacy numerical bodies while package-native replacements continue

## Development Notes

- Package code lives under [`src/bspf`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/src/bspf).
- The legacy reference implementation remains in [`bspf1d.py`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/bspf1d.py).
- Tests live under [`tests`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/tests).
- CI is configured to run the test suite on pushes and pull requests.
- CD is configured to publish to PyPI on version tags matching `v*` through GitHub Actions trusted publishing.
- Before release publishing works, the GitHub repository must be registered as a trusted publisher in the target PyPI project.

## Documentation

- [docs/index.md](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/docs/index.md)
- [docs/api.md](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/docs/api.md)
- [docs/design.md](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/docs/design.md)
- [docs/compatibility_strategy.md](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/docs/compatibility_strategy.md)
