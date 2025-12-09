"""
Heat equation solver utilities.

Provides reusable functions for solving the 1D and 2D heat equation:
    1D: ∂u/∂t = ν*∂²u/∂x²
    2D: ∂u/∂t = ν*(∂²u/∂x² + ∂²u/∂y²)

Supports Neumann boundary conditions (zero flux or specified flux).
Supports both CPU (NumPy) and GPU (CuPy) computing.
"""

import numpy as np
from typing import Callable, Optional, Tuple, Union
from ..bspf1d import bspf1d
from ..bspf2d import bspf2d

# Optional GPU backend
_HAS_CUPY = False
try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None


def build_laplacian_matrix_1d(
    bspf_op: bspf1d,
    neumann_bc: Tuple[Optional[float], Optional[float]] = (0.0, 0.0),
    use_gpu: bool = False
) -> np.ndarray:
    """
    Build Laplacian matrix using BSPF differentiate method.
    
    The Laplacian matrix L is such that L @ u computes the second derivative
    of u with the specified Neumann boundary conditions.
    
    Parameters
    ----------
    bspf_op : bspf1d
        B-spline operator for spatial differentiation
    neumann_bc : tuple, optional
        Neumann boundary conditions (left_flux, right_flux).
        Default: (0.0, 0.0) for zero flux (homogeneous Neumann)
    use_gpu : bool, optional
        Whether to use GPU arrays. Default: False
    
    Returns
    -------
    L : array
        Laplacian matrix of shape (n, n) where n is the grid size.
        On GPU if use_gpu=True and CuPy is available.
    """
    n = bspf_op.grid.n
    
    # Detect backend
    if use_gpu and _HAS_CUPY:
        xp = cp
    else:
        xp = np
    
    # Create identity matrix
    I = xp.eye(n, dtype=xp.float64)
    
    # Apply differentiate to each column of identity matrix
    # This builds the Laplacian matrix column by column
    L_columns = []
    for k in range(n):
        d2u, _ = bspf_op.differentiate(I[:, k], k=2, neumann_bc=neumann_bc)
        L_columns.append(d2u)
    
    # Convert list of columns to array and transpose
    L = xp.array(L_columns, dtype=xp.float64).T
    
    return L


def create_heat_rhs_1d(
    bspf_op: bspf1d,
    nu: float,
    neumann_bc: Tuple[Optional[float], Optional[float]] = (0.0, 0.0)
) -> Callable:
    """
    Create RHS function for the 1D heat equation.
    
    The equation: ∂u/∂t = ν*∂²u/∂x²
    
    Parameters
    ----------
    bspf_op : bspf1d
        BSPF operator for spatial differentiation
    nu : float
        Diffusion coefficient (ν > 0)
    neumann_bc : tuple, optional
        Neumann boundary conditions (left_flux, right_flux).
        Default: (0.0, 0.0) for zero flux (homogeneous Neumann)
    
    Returns
    -------
    rhs_func : callable
        RHS function with signature: rhs_func(u, t) -> du_dt
        where:
            u : array, current solution
            t : float, current time (not used but kept for interface consistency)
            du_dt : array, time derivative
    """
    def rhs_func(u: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        Compute RHS of heat equation: du/dt = nu * d²u/dx²
        
        Parameters
        ----------
        u : array
            Current solution
        t : float, optional
            Current time (not used, kept for interface consistency)
        
        Returns
        -------
        du_dt : array
            Time derivative
        """
        # Compute second derivative with Neumann BC
        _, d2u_dx2, _ = bspf_op.differentiate_1_2(u, neumann_bc=neumann_bc)
        return nu * d2u_dx2
    
    return rhs_func


def create_heat_jacobian_1d(
    L: np.ndarray,
    nu: float
) -> Callable:
    """
    Create Jacobian function for the 1D heat equation.
    
    The Jacobian is constant: J = ν * L, where L is the Laplacian matrix.
    This is used for implicit time-stepping methods (e.g., BDF2).
    
    Parameters
    ----------
    L : array
        Precomputed Laplacian matrix (from build_laplacian_matrix_1d)
    nu : float
        Diffusion coefficient (ν > 0)
    
    Returns
    -------
    jacobian_func : callable
        Jacobian function with signature: jacobian_func(u, t) -> J
        where:
            u : array, current solution (not used but kept for interface consistency)
            t : float, current time (not used but kept for interface consistency)
            J : array, Jacobian matrix
    """
    # Precompute Jacobian matrix: J = nu * L
    # Detect backend from L array
    if _HAS_CUPY and isinstance(L, cp.ndarray):
        J = nu * L
    else:
        J = nu * np.asarray(L)
    
    def jacobian_func(u: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        Return the Jacobian matrix for the heat equation.
        
        Parameters
        ----------
        u : array
            Current solution (not used, kept for interface consistency)
        t : float, optional
            Current time (not used, kept for interface consistency)
        
        Returns
        -------
        J : array
            Jacobian matrix (constant: J = nu * L)
        """
        return J
    
    return jacobian_func


def create_heat_rhs_2d(
    bspf_op: bspf2d,
    nu: float,
    flux_x: Tuple[Optional[float], Optional[float]] = (0.0, 0.0),
    flux_y: Tuple[Optional[float], Optional[float]] = (0.0, 0.0),
    lam: float = 0.0
) -> Callable:
    """
    Create RHS function for the 2D heat equation.
    
    The equation: ∂u/∂t = ν*(∂²u/∂x² + ∂²u/∂y²)
    
    Parameters
    ----------
    bspf_op : bspf2d
        BSPF 2D operator for spatial differentiation
    nu : float
        Diffusion coefficient (ν > 0)
    flux_x : tuple, optional
        Neumann boundary conditions for x-direction (left_flux, right_flux).
        Default: (0.0, 0.0) for zero flux (homogeneous Neumann)
    flux_y : tuple, optional
        Neumann boundary conditions for y-direction (bottom_flux, top_flux).
        Default: (0.0, 0.0) for zero flux (homogeneous Neumann)
    lam : float, optional
        Tikhonov regularization parameter for differentiation.
        Default: 0.0
    
    Returns
    -------
    rhs_func : callable
        RHS function with signature: rhs_func(u, t) -> du_dt
        where:
            u : array, shape (ny, nx), current solution
            t : float, current time (not used but kept for interface consistency)
            du_dt : array, shape (ny, nx), time derivative
    """
    def rhs_func(u: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        Compute RHS of 2D heat equation: du/dt = nu * (d²u/dx² + d²u/dy²)
        
        Parameters
        ----------
        u : array, shape (ny, nx)
            Current solution
        t : float, optional
            Current time (not used, kept for interface consistency)
        
        Returns
        -------
        du_dt : array, shape (ny, nx)
            Time derivative
        """
        # Compute second derivatives with Neumann BCs
        u_xx = bspf_op.partial_dxx_neumann(u, lam=lam, flux=flux_x)
        u_yy = bspf_op.partial_dyy_neumann(u, lam=lam, flux=flux_y)
        
        # Laplacian: ∇²u = ∂²u/∂x² + ∂²u/∂y²
        laplacian = u_xx + u_yy
        
        # RHS: du/dt = nu * ∇²u
        return nu * laplacian
    
    return rhs_func

