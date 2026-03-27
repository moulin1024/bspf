# `bspf`

`bspf` is being reorganized from a single-file prototype into a structured Python package.

The current numerical implementation still lives in [`bspf1d.py`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/bspf1d.py). The package scaffold under `src/bspf` provides a stable import surface while the internals are migrated module by module.

## Current Status

- Public package scaffold exists under `src/bspf`.
- The package currently delegates to the legacy implementation in [`bspf1d.py`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/bspf1d.py).
- A concrete refactor backlog is tracked in [docs/refactor_backlog.md](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/docs/refactor_backlog.md).

## Install

```bash
pip install -e .
```

Optional GPU dependencies:

```bash
pip install -e .[gpu]
```

## Public API

```python
import numpy as np

from bspf import BSPF1D, Grid1D, PiecewiseBSPF1D

x = np.linspace(0.0, 2.0 * np.pi, 128)
op = BSPF1D.from_grid(degree=5, x=x)
f = np.sin(x)

df1, df2, f_spline = op.differentiate_1_2(f)
```

Compatibility alias:

```python
from bspf import bspf1d
```

## Planned Layout

```text
src/bspf/
  backend.py
  basis.py
  boundary.py
  correction.py
  grid.py
  kkt.py
  knots.py
  ops/
  operators/
```
