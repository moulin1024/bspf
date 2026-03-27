# Design Notes

## Design Goal

The package is organized around the mathematical objects the user works with,
rather than around one monolithic implementation file.

The package architecture is currently:

```text
src/bspf/
  backend.py
  grid.py
  knots.py
  basis.py
  boundary.py
  correction.py
  kkt.py
  ops/
  operators/
```

## Numerical Model

The operator follows a split formulation:

1. fit a constrained B-spline approximation to the sampled data
2. compute a residual `r = f - f_spline`
3. correct derivatives or antiderivatives using FFT-based operations on the residual

This is why the package separates:

- spline basis construction
- endpoint constraint operators
- residual correction
- constrained KKT solves
- high-level operation families

## Package Layers

### Foundational layer

- [`backend.py`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/src/bspf/backend.py)
- [`grid.py`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/src/bspf/grid.py)
- [`knots.py`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/src/bspf/knots.py)

This layer defines backend/device rules, uniform-grid metadata, and knot generation.

### Spline operator layer

- [`basis.py`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/src/bspf/basis.py)
- [`boundary.py`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/src/bspf/boundary.py)
- [`correction.py`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/src/bspf/correction.py)
- [`kkt.py`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/src/bspf/kkt.py)

This layer contains the reusable numerical kernels used by the main operator.

### Operation layer

- [`ops/differentiation.py`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/src/bspf/ops/differentiation.py)
- [`ops/integration.py`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/src/bspf/ops/integration.py)
- [`ops/interpolation.py`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/src/bspf/ops/interpolation.py)

This layer groups public operations by behavior rather than by class size.

### Public API layer

- [`operators/bspf1d.py`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/src/bspf/operators/bspf1d.py)
- [`operators/piecewise.py`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/src/bspf/operators/piecewise.py)
- [`__init__.py`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/src/bspf/__init__.py)

This layer exposes the package-facing API.

## Compatibility Design

The repository still keeps [`bspf1d.py`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/bspf1d.py) as a legacy compatibility implementation and regression reference.

That allows the package to:

- migrate functionality incrementally
- verify numerical behavior during refactors
- avoid breaking downstream scripts immediately

The compatibility policy itself is documented in [compatibility_strategy.md](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/docs/compatibility_strategy.md).

## Current Tradeoff

The package owns the public module structure and most foundational implementation now, but some operation-family functions still delegate internally to legacy numerical bodies. This keeps risk low while the public API and package boundaries stabilize under test coverage.

The next meaningful cleanup after Phase 8 would be replacing those remaining internal delegations with fully package-native numerical implementations.
