"""B-spline basis functions and operations."""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import numpy.typing as npt
from scipy.interpolate import BSpline

from .grid import Grid1D

Array = npt.NDArray[np.float64]


class BSplineBasis1D:
    """B-spline basis on a uniform grid with lazy derivative matrices."""
    
    def __init__(self, degree: int, knots: Array, grid: Grid1D):
        self.degree = int(degree)
        self.knots: Array = np.asarray(knots, dtype=np.float64)
        self.grid = grid

        self._splines = self._mk_splines()
        n_basis = len(self._splines)
        B0 = np.empty((n_basis, grid.n), dtype=np.float64)
        for i, s in enumerate(self._splines):
            B0[i, :] = s(grid.x)
        self._B0: Array = B0
        self._BT0: Array = B0.T.copy()

        self._BkT: Dict[int, Array] = {}
        self._eval_cache: Dict[Tuple[float, int], Array] = {}

    def _mk_splines(self) -> list[BSpline]:
        """Create individual B-spline functions."""
        n_basis = len(self.knots) - self.degree - 1
        coeffs = np.eye(n_basis, dtype=np.float64)
        return [BSpline(self.knots, c, self.degree) for c in coeffs]

    def _evaluate_splines_vectorized(self, x: Array, deriv_order: int = 0) -> Array:
        """Evaluate all splines at given points with caching."""
        cache_key = (float(x[0]), deriv_order)  # uniform-grid key
        if cache_key in self._eval_cache:
            return self._eval_cache[cache_key]
        n_basis = len(self._splines)
        result = np.empty((n_basis, len(x)), dtype=np.float64)
        for i, s in enumerate(self._splines):
            result[i, :] = (s.derivative(deriv_order) if deriv_order else s)(x)
        self._eval_cache[cache_key] = result
        return result

    @property
    def B0(self) -> Array: 
        """Basis matrix (n_basis × n_points)."""
        return self._B0
    
    @property
    def BT0(self) -> Array: 
        """Transposed basis matrix (n_points × n_basis)."""
        return self._BT0

    def BkT(self, k: int) -> Array:
        """Get k-th derivative basis matrix transpose (n_points × n_basis)."""
        if k == 0:
            return self._BT0
        if k not in self._BkT:
            Bk = self._evaluate_splines_vectorized(self.grid.x, deriv_order=k)
            self._BkT[k] = Bk.T.copy()
        return self._BkT[k]

    def integrate_basis(self, a: float, b: float) -> Array:
        """Integrate each basis function over [a, b]."""
        return np.array([s.integrate(a, b) for s in self._splines])
