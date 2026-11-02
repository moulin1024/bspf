# SPDX-License-Identifier: GPL-3.0-or-later
"""C++ BSPF (B-spline plus Fourier) calculus on uniform 1-3D grids.

Compile once, apply many times::

    line = Line(x, degree=5)
    d1 = line.derivative(1)          # Op, (N, N)
    df = d1(f, axis=0)               # any axis of a float64 array
    e = line.evaluate(points)        # Op, (P, N)
    w = line.quadrature(a, b)        # Op, (1, N)

    grid = Grid.uniform((x, y))      # fields have shape (nx, ny, *batch)
    grid.partial(u, (1, 1)); grid.interpolate(u, (xs, ys)); grid.integrate(u)
"""
from __future__ import annotations

import numpy as np

from ._core import Line, Op, basis_matrix, endpoint_scaled_knots, open_knots
from ._core import Grid as _Grid

__all__ = ["Grid", "Line", "Op", "basis_matrix", "endpoint_scaled_knots", "open_knots"]


class Grid(_Grid):
    """Tensor-product grid of one to three Lines; leading field axes are spatial."""

    def __init__(self, lines):
        super().__init__(list(lines))

    @classmethod
    def uniform(cls, coordinates, **options):
        """One Line per coordinate vector, all with the same options."""
        return cls([Line(np.asarray(c, dtype=float), **options) for c in coordinates])

    def _unit(self, axis, order=1):
        return tuple(order if i == axis else 0 for i in range(self.dimension))

    def gradient(self, f):
        """Partials stacked on a new leading component axis."""
        return np.stack([self.partial(f, self._unit(i)) for i in range(self.dimension)])

    def divergence(self, vector):
        """vector.shape == (dimension, *spatial_shape, *batch_shape)."""
        vector = np.asarray(vector, dtype=float)
        if vector.shape[0] != self.dimension:
            raise ValueError("vector must have one leading component per spatial axis")
        return sum(self.partial(np.ascontiguousarray(vector[i]), self._unit(i)) for i in range(self.dimension))

    def hessian(self, f):
        """H[i, j] = d_i d_j f; the diagonal uses direct second derivatives."""
        d = self.dimension
        return np.stack([np.stack([self.partial(f, tuple(int(k == i) + int(k == j) for k in range(d)))
                                   for j in range(d)]) for i in range(d)])

    def curl(self, vector):
        """Scalar curl in 2D, vector curl in 3D."""
        vector = np.asarray(vector, dtype=float)
        d = self.dimension
        if d not in (2, 3) or vector.shape[0] != d:
            raise ValueError("curl needs a 2D or 3D vector field")
        p = lambda i, axis: self.partial(np.ascontiguousarray(vector[i]), self._unit(axis))
        if d == 2:
            return p(1, 0) - p(0, 1)
        return np.stack([p(2, 1) - p(1, 2), p(0, 2) - p(2, 0), p(1, 0) - p(0, 1)])
