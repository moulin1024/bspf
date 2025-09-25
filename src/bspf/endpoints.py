"""Endpoint constraint operations for B-splines."""
from __future__ import annotations
import math
import numpy as np
import numpy.typing as npt
from scipy import linalg as sla

from .basis import BSplineBasis1D

Array = npt.NDArray[np.float64]


class EndpointOps1D:
    """Endpoint constraints and sample-to-endpoint operator for endpoint derivatives."""
    
    def __init__(self, basis: BSplineBasis1D, *, order: int, num_bd: int):
        """Initialize endpoint operations.
        
        Parameters
        ----------
        basis : BSplineBasis1D
            B-spline basis object
        order : int
            Maximum derivative order for constraints
        num_bd : int
            Number of boundary points to use for finite difference
        """
        self.order = int(order)
        self.num_bd = int(num_bd)
        self.grid = basis.grid
        B0 = basis.B0
        Bk = {0: B0}
        for k in range(1, order):
            Bk[k] = basis.BkT(k).T

        n_basis, n_points = B0.shape

        # Constraint matrix: evaluate basis and derivatives at endpoints
        C = np.zeros((2 * order, n_basis), dtype=np.float64)
        for p in range(order):
            C[p, :] = Bk[p][:, 0]          # left endpoint derivatives
            C[order + p, :] = Bk[p][:, -1]  # right endpoint derivatives

        # Finite difference operators for boundary values
        i, j = np.meshgrid(np.arange(num_bd), np.arange(num_bd), indexing="ij")
        fact = np.array([math.factorial(k) for k in range(num_bd)], dtype=np.float64)
        A_left = (j**i) / fact[:, None]
        A_right = np.flip(A_left * ((-1.0) ** i), axis=(0, 1))

        # Extract relevant rows for derivative orders
        E_left = np.eye(num_bd, dtype=np.float64)[:order, :].T
        idx = np.arange(num_bd - 1, num_bd - order - 1, -1)
        E_right = np.eye(num_bd, dtype=np.float64)[idx, :].T

        # Solve for finite difference weights
        X_left = sla.solve(A_left, E_left).T
        X_right = sla.solve(A_right, E_right).T

        # Scale by grid spacing powers
        dx_pows = self.grid.dx ** np.arange(order, dtype=np.float64)
        BND = np.zeros((2 * order, n_points), dtype=np.float64)
        BND[:order, :num_bd] = X_left / dx_pows[:, None]
        BND[order:, n_points - num_bd:] = X_right / dx_pows[:, None]

        self.C: Array = C.astype(np.float64)
        self.BND: Array = BND.astype(np.float64)
        self.X_left: Array = X_left.astype(np.float64)
        self.X_right: Array = X_right.astype(np.float64)
