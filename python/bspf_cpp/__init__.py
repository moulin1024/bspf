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

from ._core import Boundary, BoundaryKind, CompressibleSolver2D, multigrid_solve, newton_krylov_solve, GalerkinLine, Line, Op, SplineCurve, basis_matrix, endpoint_scaled_knots, open_knots
from ._core import MappedGrid2D as _MappedGrid2D
from ._core import MultiPatch2D as _MultiPatch2D
from ._core import Grid as _Grid

try:  # FP32 CUDA backend, present when built with -DBSPF_CUDA=ON
    from ._core import cuda
except ImportError:
    cuda = None

__all__ = ["GalerkinLine", "Grid", "Line", "MappedGrid2D", "MultiPatch2D", "SplineCurve", "Op", "Boundary", "BoundaryKind", "CompressibleSolver2D", "multigrid_solve", "newton_krylov_solve", "basis_matrix", "cuda", "endpoint_scaled_knots", "open_knots"]


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
        h = [[None] * d for _ in range(d)]
        for i in range(d):
            for j in range(i, d):  # symmetric: compute the upper triangle once
                h[i][j] = h[j][i] = self.partial(f, tuple(int(k == i) + int(k == j) for k in range(d)))
        return np.stack([np.stack(row) for row in h])

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


class MappedGrid2D(_MappedGrid2D):
    """Structured grid on a region bounded by four spline curves.

    Curves (SplineCurve) are oriented bottom: P00 -> P10, right: P10 -> P11,
    top: P01 -> P11, left: P00 -> P01. Fields have shape (n_xi, n_eta, *batch);
    derivatives are physical (x, y) derivatives by the chain rule.
    """

    def gradient(self, f):
        """(f_x, f_y) stacked on a new leading axis."""
        return np.stack([self.partial(f, (1, 0)), self.partial(f, (0, 1))])

    def laplacian(self, f):
        """f_xx + f_yy using direct reference second derivatives and map metrics."""
        return self.partial(f, (2, 0)) + self.partial(f, (0, 2))


class MultiPatch2D(_MultiPatch2D):
    """Conforming deformed patches (MappedGrid2D) joined at complete faces.

    Faces are (patch, axis, side) as in bspf-fluid's PatchMesh2D; interfaces are
    ((patch, axis, side), (patch, axis, side), reverse). Periodic pairs (keyword
    ``periodic``) add a translation, ((patch, axis, side), (patch, axis, side),
    reverse, (dx, dy)), with the right face equal to the left face shifted by
    (dx, dy), e.g. the pitch of a cascade. Fields are lists with one
    (n_xi, n_eta, *batch) array per patch; nodes shared by several patches are
    averaged after differentiation. By default, interface-crossing lines use
    up to eight neighbour ghost samples; ``overlap=0`` disables this extension.
    ``partial(..., average_shared=False)`` retains both interface traces.
    """

    def gradient(self, fields):
        """[(f_x, f_y) stacked on a new leading axis] per patch."""
        fx, fy = self.partial(fields, (1, 0)), self.partial(fields, (0, 1))
        return [np.stack(pair) for pair in zip(fx, fy)]

    def laplacian(self, fields):
        """f_xx + f_yy per patch (shared nodes averaged)."""
        return self.average_shared([a + b for a, b in zip(self.partial(fields, (2, 0), average_shared=False),
                                                         self.partial(fields, (0, 2), average_shared=False))])
