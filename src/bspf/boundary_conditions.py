"""
Boundary condition enforcement utilities for BSPF.

This module provides functions for explicitly enforcing boundary conditions
on solution arrays, complementing the built-in BC handling in BSPF operators.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.complex128 | np.float64]


def enforce_zero_flux_neumann_bc(psi: Array, dx: float, order: int = 2) -> Array:
    """
    Enforce zero-flux Neumann boundary conditions explicitly using finite difference.
    
    For zero flux: ∂ψ/∂x = 0 at boundaries
    Uses one-sided finite difference stencils of specified order.
    
    This function is useful when you need to explicitly enforce Neumann BCs
    on a solution array, for example after time stepping or when using
    methods that don't automatically enforce BCs.
    
    Parameters:
    -----------
    psi : array (complex or float)
        Solution array to enforce BCs on. Can be 1D array of any numeric type.
    dx : float
        Grid spacing
    order : int
        Order of accuracy for finite difference approximation (1, 2, 3, 4, or 5)
        Should match or be close to the BSPF degree for consistency.
        Default is 2.
        
    Returns:
    --------
    psi_bc : array
        Solution array with enforced Neumann BCs (same type as input)
    
    Notes:
    -----
    The function uses one-sided finite difference stencils to approximate
    ∂ψ/∂x = 0 at boundaries. Higher order stencils require more interior
    points, so the function automatically falls back to lower-order stencils
    if there aren't enough grid points.
    
    Examples:
    --------
    >>> import numpy as np
    >>> from bspf import enforce_zero_flux_neumann_bc
    >>> x = np.linspace(0, 1, 101)
    >>> dx = x[1] - x[0]
    >>> psi = np.sin(np.pi * x)
    >>> psi_bc = enforce_zero_flux_neumann_bc(psi, dx, order=2)
    """
    psi_bc = psi.copy()
    n = len(psi)
    
    if order == 1:
        # 1st order: ψ[0] = ψ[1], ψ[-1] = ψ[-2]
        psi_bc[0] = psi_bc[1]
        psi_bc[-1] = psi_bc[-2]
        
    elif order == 2:
        # 2nd order forward difference at left boundary:
        # ∂ψ/∂x ≈ (-3*ψ[0] + 4*ψ[1] - ψ[2])/(2*dx) = 0
        # => ψ[0] = (4*ψ[1] - ψ[2])/3
        if n >= 3:
            psi_bc[0] = (4.0 * psi_bc[1] - psi_bc[2]) / 3.0
        else:
            psi_bc[0] = psi_bc[1]
        
        # 2nd order backward difference at right boundary:
        # ∂ψ/∂x ≈ (3*ψ[-1] - 4*ψ[-2] + ψ[-3])/(2*dx) = 0
        # => ψ[-1] = (4*ψ[-2] - ψ[-3])/3
        if n >= 3:
            psi_bc[-1] = (4.0 * psi_bc[-2] - psi_bc[-3]) / 3.0
        else:
            psi_bc[-1] = psi_bc[-2]
            
    elif order == 3:
        # 3rd order forward difference at left boundary:
        # ∂ψ/∂x ≈ (-11*ψ[0] + 18*ψ[1] - 9*ψ[2] + 2*ψ[3])/(6*dx) = 0
        # => ψ[0] = (18*ψ[1] - 9*ψ[2] + 2*ψ[3])/11
        if n >= 4:
            psi_bc[0] = (18.0 * psi_bc[1] - 9.0 * psi_bc[2] + 2.0 * psi_bc[3]) / 11.0
        elif n >= 3:
            psi_bc[0] = (4.0 * psi_bc[1] - psi_bc[2]) / 3.0
        else:
            psi_bc[0] = psi_bc[1]
        
        # 3rd order backward difference at right boundary:
        # ∂ψ/∂x ≈ (11*ψ[-1] - 18*ψ[-2] + 9*ψ[-3] - 2*ψ[-4])/(6*dx) = 0
        # => ψ[-1] = (18*ψ[-2] - 9*ψ[-3] + 2*ψ[-4])/11
        if n >= 4:
            psi_bc[-1] = (18.0 * psi_bc[-2] - 9.0 * psi_bc[-3] + 2.0 * psi_bc[-4]) / 11.0
        elif n >= 3:
            psi_bc[-1] = (4.0 * psi_bc[-2] - psi_bc[-3]) / 3.0
        else:
            psi_bc[-1] = psi_bc[-2]
            
    elif order == 4:
        # 4th order forward difference at left boundary:
        # ∂ψ/∂x ≈ (-25*ψ[0] + 48*ψ[1] - 36*ψ[2] + 16*ψ[3] - 3*ψ[4])/(12*dx) = 0
        # => ψ[0] = (48*ψ[1] - 36*ψ[2] + 16*ψ[3] - 3*ψ[4])/25
        if n >= 5:
            psi_bc[0] = (48.0 * psi_bc[1] - 36.0 * psi_bc[2] + 16.0 * psi_bc[3] - 3.0 * psi_bc[4]) / 25.0
        elif n >= 4:
            psi_bc[0] = (18.0 * psi_bc[1] - 9.0 * psi_bc[2] + 2.0 * psi_bc[3]) / 11.0
        elif n >= 3:
            psi_bc[0] = (4.0 * psi_bc[1] - psi_bc[2]) / 3.0
        else:
            psi_bc[0] = psi_bc[1]
        
        # 4th order backward difference at right boundary:
        # ∂ψ/∂x ≈ (25*ψ[-1] - 48*ψ[-2] + 36*ψ[-3] - 16*ψ[-4] + 3*ψ[-5])/(12*dx) = 0
        # => ψ[-1] = (48*ψ[-2] - 36*ψ[-3] + 16*ψ[-4] - 3*ψ[-5])/25
        if n >= 5:
            psi_bc[-1] = (48.0 * psi_bc[-2] - 36.0 * psi_bc[-3] + 16.0 * psi_bc[-4] - 3.0 * psi_bc[-5]) / 25.0
        elif n >= 4:
            psi_bc[-1] = (18.0 * psi_bc[-2] - 9.0 * psi_bc[-3] + 2.0 * psi_bc[-4]) / 11.0
        elif n >= 3:
            psi_bc[-1] = (4.0 * psi_bc[-2] - psi_bc[-3]) / 3.0
        else:
            psi_bc[-1] = psi_bc[-2]
            
    elif order == 5:
        # 5th order forward difference at left boundary:
        # ∂ψ/∂x ≈ (-137*ψ[0] + 300*ψ[1] - 300*ψ[2] + 200*ψ[3] - 75*ψ[4] + 12*ψ[5])/(60*dx) = 0
        # => ψ[0] = (300*ψ[1] - 300*ψ[2] + 200*ψ[3] - 75*ψ[4] + 12*ψ[5])/137
        if n >= 6:
            psi_bc[0] = (300.0 * psi_bc[1] - 300.0 * psi_bc[2] + 200.0 * psi_bc[3] - 75.0 * psi_bc[4] + 12.0 * psi_bc[5]) / 137.0
        elif n >= 5:
            psi_bc[0] = (48.0 * psi_bc[1] - 36.0 * psi_bc[2] + 16.0 * psi_bc[3] - 3.0 * psi_bc[4]) / 25.0
        elif n >= 4:
            psi_bc[0] = (18.0 * psi_bc[1] - 9.0 * psi_bc[2] + 2.0 * psi_bc[3]) / 11.0
        elif n >= 3:
            psi_bc[0] = (4.0 * psi_bc[1] - psi_bc[2]) / 3.0
        else:
            psi_bc[0] = psi_bc[1]
        
        # 5th order backward difference at right boundary:
        # ∂ψ/∂x ≈ (137*ψ[-1] - 300*ψ[-2] + 300*ψ[-3] - 200*ψ[-4] + 75*ψ[-5] - 12*ψ[-6])/(60*dx) = 0
        # => ψ[-1] = (300*ψ[-2] - 300*ψ[-3] + 200*ψ[-4] - 75*ψ[-5] + 12*ψ[-6])/137
        if n >= 6:
            psi_bc[-1] = (300.0 * psi_bc[-2] - 300.0 * psi_bc[-3] + 200.0 * psi_bc[-4] - 75.0 * psi_bc[-5] + 12.0 * psi_bc[-6]) / 137.0
        elif n >= 5:
            psi_bc[-1] = (48.0 * psi_bc[-2] - 36.0 * psi_bc[-3] + 16.0 * psi_bc[-4] - 3.0 * psi_bc[-5]) / 25.0
        elif n >= 4:
            psi_bc[-1] = (18.0 * psi_bc[-2] - 9.0 * psi_bc[-3] + 2.0 * psi_bc[-4]) / 11.0
        elif n >= 3:
            psi_bc[-1] = (4.0 * psi_bc[-2] - psi_bc[-3]) / 3.0
        else:
            psi_bc[-1] = psi_bc[-2]
    else:
        # Default to 2nd order if order not in [1,2,3,4,5]
        if n >= 3:
            psi_bc[0] = (4.0 * psi_bc[1] - psi_bc[2]) / 3.0
            psi_bc[-1] = (4.0 * psi_bc[-2] - psi_bc[-3]) / 3.0
        else:
            psi_bc[0] = psi_bc[1]
            psi_bc[-1] = psi_bc[-2]
    
    return psi_bc


def enforce_zero_flux_neumann_bc_2d(psi: Array, dx: float, dy: float, order: int = 2) -> Array:
    """
    Enforce zero-flux Neumann boundary conditions on a 2D array using vectorized operations.
    
    For zero flux: ∂ψ/∂x = 0 at x-boundaries, ∂ψ/∂y = 0 at y-boundaries
    Uses one-sided finite difference stencils of specified order.
    
    This function is a vectorized version that processes all rows/columns simultaneously,
    avoiding explicit loops for better performance.
    
    Parameters:
    -----------
    psi : array (complex or float), shape (ny, nx)
        Solution array to enforce BCs on. Must be 2D array.
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
    order : int
        Order of accuracy for finite difference approximation (1, 2, 3, 4, or 5)
        Should match or be close to the BSPF degree for consistency.
        Default is 2.
        
    Returns:
    --------
    psi_bc : array, shape (ny, nx)
        Solution array with enforced Neumann BCs (same type as input)
    
    Notes:
    -----
    The function uses one-sided finite difference stencils to approximate
    zero flux at boundaries. Higher order stencils require more interior
    points, so the function automatically falls back to lower-order stencils
    if there aren't enough grid points.
    
    Examples:
    --------
    >>> import numpy as np
    >>> from bspf import enforce_zero_flux_neumann_bc_2d
    >>> x = np.linspace(0, 1, 101)
    >>> y = np.linspace(0, 1, 101)
    >>> dx = x[1] - x[0]
    >>> dy = y[1] - y[0]
    >>> psi = np.ones((101, 101), dtype=complex)
    >>> psi_bc = enforce_zero_flux_neumann_bc_2d(psi, dx, dy, order=2)
    """
    if psi.ndim != 2:
        raise ValueError(f"psi must be 2D array, got shape {psi.shape}")
    
    psi_bc = psi.copy()
    ny, nx = psi_bc.shape
    
    # Enforce BCs on x-boundaries (left and right edges) - process all rows at once
    if order == 1:
        # 1st order: ψ[:, 0] = ψ[:, 1], ψ[:, -1] = ψ[:, -2]
        psi_bc[:, 0] = psi_bc[:, 1]
        psi_bc[:, -1] = psi_bc[:, -2]
        
    elif order == 2:
        # 2nd order forward difference at left boundary
        if nx >= 3:
            psi_bc[:, 0] = (4.0 * psi_bc[:, 1] - psi_bc[:, 2]) / 3.0
        else:
            psi_bc[:, 0] = psi_bc[:, 1]
        
        # 2nd order backward difference at right boundary
        if nx >= 3:
            psi_bc[:, -1] = (4.0 * psi_bc[:, -2] - psi_bc[:, -3]) / 3.0
        else:
            psi_bc[:, -1] = psi_bc[:, -2]
            
    elif order == 3:
        # 3rd order forward difference at left boundary
        if nx >= 4:
            psi_bc[:, 0] = (18.0 * psi_bc[:, 1] - 9.0 * psi_bc[:, 2] + 2.0 * psi_bc[:, 3]) / 11.0
        elif nx >= 3:
            psi_bc[:, 0] = (4.0 * psi_bc[:, 1] - psi_bc[:, 2]) / 3.0
        else:
            psi_bc[:, 0] = psi_bc[:, 1]
        
        # 3rd order backward difference at right boundary
        if nx >= 4:
            psi_bc[:, -1] = (18.0 * psi_bc[:, -2] - 9.0 * psi_bc[:, -3] + 2.0 * psi_bc[:, -4]) / 11.0
        elif nx >= 3:
            psi_bc[:, -1] = (4.0 * psi_bc[:, -2] - psi_bc[:, -3]) / 3.0
        else:
            psi_bc[:, -1] = psi_bc[:, -2]
            
    elif order == 4:
        # 4th order forward difference at left boundary
        if nx >= 5:
            psi_bc[:, 0] = (48.0 * psi_bc[:, 1] - 36.0 * psi_bc[:, 2] + 16.0 * psi_bc[:, 3] - 3.0 * psi_bc[:, 4]) / 25.0
        elif nx >= 4:
            psi_bc[:, 0] = (18.0 * psi_bc[:, 1] - 9.0 * psi_bc[:, 2] + 2.0 * psi_bc[:, 3]) / 11.0
        elif nx >= 3:
            psi_bc[:, 0] = (4.0 * psi_bc[:, 1] - psi_bc[:, 2]) / 3.0
        else:
            psi_bc[:, 0] = psi_bc[:, 1]
        
        # 4th order backward difference at right boundary
        if nx >= 5:
            psi_bc[:, -1] = (48.0 * psi_bc[:, -2] - 36.0 * psi_bc[:, -3] + 16.0 * psi_bc[:, -4] - 3.0 * psi_bc[:, -5]) / 25.0
        elif nx >= 4:
            psi_bc[:, -1] = (18.0 * psi_bc[:, -2] - 9.0 * psi_bc[:, -3] + 2.0 * psi_bc[:, -4]) / 11.0
        elif nx >= 3:
            psi_bc[:, -1] = (4.0 * psi_bc[:, -2] - psi_bc[:, -3]) / 3.0
        else:
            psi_bc[:, -1] = psi_bc[:, -2]
            
    elif order == 5:
        # 5th order forward difference at left boundary
        if nx >= 6:
            psi_bc[:, 0] = (300.0 * psi_bc[:, 1] - 300.0 * psi_bc[:, 2] + 200.0 * psi_bc[:, 3] - 75.0 * psi_bc[:, 4] + 12.0 * psi_bc[:, 5]) / 137.0
        elif nx >= 5:
            psi_bc[:, 0] = (48.0 * psi_bc[:, 1] - 36.0 * psi_bc[:, 2] + 16.0 * psi_bc[:, 3] - 3.0 * psi_bc[:, 4]) / 25.0
        elif nx >= 4:
            psi_bc[:, 0] = (18.0 * psi_bc[:, 1] - 9.0 * psi_bc[:, 2] + 2.0 * psi_bc[:, 3]) / 11.0
        elif nx >= 3:
            psi_bc[:, 0] = (4.0 * psi_bc[:, 1] - psi_bc[:, 2]) / 3.0
        else:
            psi_bc[:, 0] = psi_bc[:, 1]
        
        # 5th order backward difference at right boundary
        if nx >= 6:
            psi_bc[:, -1] = (300.0 * psi_bc[:, -2] - 300.0 * psi_bc[:, -3] + 200.0 * psi_bc[:, -4] - 75.0 * psi_bc[:, -5] + 12.0 * psi_bc[:, -6]) / 137.0
        elif nx >= 5:
            psi_bc[:, -1] = (48.0 * psi_bc[:, -2] - 36.0 * psi_bc[:, -3] + 16.0 * psi_bc[:, -4] - 3.0 * psi_bc[:, -5]) / 25.0
        elif nx >= 4:
            psi_bc[:, -1] = (18.0 * psi_bc[:, -2] - 9.0 * psi_bc[:, -3] + 2.0 * psi_bc[:, -4]) / 11.0
        elif nx >= 3:
            psi_bc[:, -1] = (4.0 * psi_bc[:, -2] - psi_bc[:, -3]) / 3.0
        else:
            psi_bc[:, -1] = psi_bc[:, -2]
    else:
        # Default to 2nd order if order not in [1,2,3,4,5]
        if nx >= 3:
            psi_bc[:, 0] = (4.0 * psi_bc[:, 1] - psi_bc[:, 2]) / 3.0
            psi_bc[:, -1] = (4.0 * psi_bc[:, -2] - psi_bc[:, -3]) / 3.0
        else:
            psi_bc[:, 0] = psi_bc[:, 1]
            psi_bc[:, -1] = psi_bc[:, -2]
    
    # Enforce BCs on y-boundaries (bottom and top edges) - process all columns at once
    if order == 1:
        # 1st order: ψ[0, :] = ψ[1, :], ψ[-1, :] = ψ[-2, :]
        psi_bc[0, :] = psi_bc[1, :]
        psi_bc[-1, :] = psi_bc[-2, :]
        
    elif order == 2:
        # 2nd order forward difference at bottom boundary
        if ny >= 3:
            psi_bc[0, :] = (4.0 * psi_bc[1, :] - psi_bc[2, :]) / 3.0
        else:
            psi_bc[0, :] = psi_bc[1, :]
        
        # 2nd order backward difference at top boundary
        if ny >= 3:
            psi_bc[-1, :] = (4.0 * psi_bc[-2, :] - psi_bc[-3, :]) / 3.0
        else:
            psi_bc[-1, :] = psi_bc[-2, :]
            
    elif order == 3:
        # 3rd order forward difference at bottom boundary
        if ny >= 4:
            psi_bc[0, :] = (18.0 * psi_bc[1, :] - 9.0 * psi_bc[2, :] + 2.0 * psi_bc[3, :]) / 11.0
        elif ny >= 3:
            psi_bc[0, :] = (4.0 * psi_bc[1, :] - psi_bc[2, :]) / 3.0
        else:
            psi_bc[0, :] = psi_bc[1, :]
        
        # 3rd order backward difference at top boundary
        if ny >= 4:
            psi_bc[-1, :] = (18.0 * psi_bc[-2, :] - 9.0 * psi_bc[-3, :] + 2.0 * psi_bc[-4, :]) / 11.0
        elif ny >= 3:
            psi_bc[-1, :] = (4.0 * psi_bc[-2, :] - psi_bc[-3, :]) / 3.0
        else:
            psi_bc[-1, :] = psi_bc[-2, :]
            
    elif order == 4:
        # 4th order forward difference at bottom boundary
        if ny >= 5:
            psi_bc[0, :] = (48.0 * psi_bc[1, :] - 36.0 * psi_bc[2, :] + 16.0 * psi_bc[3, :] - 3.0 * psi_bc[4, :]) / 25.0
        elif ny >= 4:
            psi_bc[0, :] = (18.0 * psi_bc[1, :] - 9.0 * psi_bc[2, :] + 2.0 * psi_bc[3, :]) / 11.0
        elif ny >= 3:
            psi_bc[0, :] = (4.0 * psi_bc[1, :] - psi_bc[2, :]) / 3.0
        else:
            psi_bc[0, :] = psi_bc[1, :]
        
        # 4th order backward difference at top boundary
        if ny >= 5:
            psi_bc[-1, :] = (48.0 * psi_bc[-2, :] - 36.0 * psi_bc[-3, :] + 16.0 * psi_bc[-4, :] - 3.0 * psi_bc[-5, :]) / 25.0
        elif ny >= 4:
            psi_bc[-1, :] = (18.0 * psi_bc[-2, :] - 9.0 * psi_bc[-3, :] + 2.0 * psi_bc[-4, :]) / 11.0
        elif ny >= 3:
            psi_bc[-1, :] = (4.0 * psi_bc[-2, :] - psi_bc[-3, :]) / 3.0
        else:
            psi_bc[-1, :] = psi_bc[-2, :]
            
    elif order == 5:
        # 5th order forward difference at bottom boundary
        if ny >= 6:
            psi_bc[0, :] = (300.0 * psi_bc[1, :] - 300.0 * psi_bc[2, :] + 200.0 * psi_bc[3, :] - 75.0 * psi_bc[4, :] + 12.0 * psi_bc[5, :]) / 137.0
        elif ny >= 5:
            psi_bc[0, :] = (48.0 * psi_bc[1, :] - 36.0 * psi_bc[2, :] + 16.0 * psi_bc[3, :] - 3.0 * psi_bc[4, :]) / 25.0
        elif ny >= 4:
            psi_bc[0, :] = (18.0 * psi_bc[1, :] - 9.0 * psi_bc[2, :] + 2.0 * psi_bc[3, :]) / 11.0
        elif ny >= 3:
            psi_bc[0, :] = (4.0 * psi_bc[1, :] - psi_bc[2, :]) / 3.0
        else:
            psi_bc[0, :] = psi_bc[1, :]
        
        # 5th order backward difference at top boundary
        if ny >= 6:
            psi_bc[-1, :] = (300.0 * psi_bc[-2, :] - 300.0 * psi_bc[-3, :] + 200.0 * psi_bc[-4, :] - 75.0 * psi_bc[-5, :] + 12.0 * psi_bc[-6, :]) / 137.0
        elif ny >= 5:
            psi_bc[-1, :] = (48.0 * psi_bc[-2, :] - 36.0 * psi_bc[-3, :] + 16.0 * psi_bc[-4, :] - 3.0 * psi_bc[-5, :]) / 25.0
        elif ny >= 4:
            psi_bc[-1, :] = (18.0 * psi_bc[-2, :] - 9.0 * psi_bc[-3, :] + 2.0 * psi_bc[-4, :]) / 11.0
        elif ny >= 3:
            psi_bc[-1, :] = (4.0 * psi_bc[-2, :] - psi_bc[-3, :]) / 3.0
        else:
            psi_bc[-1, :] = psi_bc[-2, :]
    else:
        # Default to 2nd order if order not in [1,2,3,4,5]
        if ny >= 3:
            psi_bc[0, :] = (4.0 * psi_bc[1, :] - psi_bc[2, :]) / 3.0
            psi_bc[-1, :] = (4.0 * psi_bc[-2, :] - psi_bc[-3, :]) / 3.0
        else:
            psi_bc[0, :] = psi_bc[1, :]
            psi_bc[-1, :] = psi_bc[-2, :]
    
    return psi_bc











