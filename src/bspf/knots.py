"""Knot generation utilities for B-splines."""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt

from .grid import Grid1D

Array = npt.NDArray[np.float64]


class KnotGenerator:
    """Utility class for generating B-spline knot vectors."""
    
    @staticmethod
    def generate(
        *, 
        degree: int, 
        domain: Tuple[float, float], 
        n_basis: int,
        use_clustering: bool, 
        clustering_factor: float
    ) -> Array:
        """Generate knot vector for B-spline basis.
        
        Parameters
        ----------
        degree : int
            Degree of B-spline
        domain : Tuple[float, float]
            Domain bounds (a, b)
        n_basis : int
            Number of basis functions
        use_clustering : bool
            Whether to use clustering near boundaries
        clustering_factor : float
            Clustering parameter (higher = more clustering)
            
        Returns
        -------
        Array
            Knot vector
        """
        if n_basis <= degree:
            raise ValueError("n_basis must exceed degree.")
        n_knots = n_basis + degree + 1
        n_interior = n_knots - 2 * (degree + 1)

        if n_interior > 0:
            u = np.linspace(-1.0, 1.0, n_interior + 2)
            if use_clustering:
                u = np.tanh(clustering_factor * u) / np.tanh(clustering_factor)

            uniq = degree * (u + 1.0) / 2.0  # in [0, degree]
            ks = [float(uniq[0])] * (degree + 1) + list(map(float, uniq[1:-1])) + [float(uniq[-1])] * (degree + 1)
            k = np.array(ks, dtype=np.float64)
        else:
            k = np.concatenate([np.zeros(degree + 1), np.full(degree + 1, degree)], dtype=np.float64)

        a, b = domain
        return (k / degree) * (b - a) + a

    @staticmethod
    def resolve(
        *, 
        degree: int, 
        grid: Grid1D,
        knots: Optional[Array],
        n_basis: Optional[int],
        domain: Optional[Tuple[float, float]],
        use_clustering: bool,
        clustering_factor: float,
    ) -> Array:
        """Resolve knot vector from various input options.
        
        Parameters
        ----------
        degree : int
            Degree of B-spline
        grid : Grid1D
            Grid object
        knots : Optional[Array]
            Explicit knot vector (if provided, other params ignored)
        n_basis : Optional[int]
            Number of basis functions (default: 2*(degree+1)*2)
        domain : Optional[Tuple[float, float]]
            Domain bounds (default: grid bounds)
        use_clustering : bool
            Whether to use clustering
        clustering_factor : float
            Clustering parameter
            
        Returns
        -------
        Array
            Resolved knot vector
        """
        if knots is not None:
            k = np.asarray(knots, dtype=np.float64)
            if k.ndim != 1:
                raise ValueError("knots must be a 1D array.")
            return k
        if n_basis is None:
            n_basis = 2 * (degree + 1) * 2
        if domain is None:
            domain = (grid.a, grid.b)
        return KnotGenerator.generate(
            degree=degree, domain=domain, n_basis=n_basis,
            use_clustering=use_clustering, clustering_factor=clustering_factor
        )
