"""Unified derivative computation interface for BSPF, Chebyshev, and Padé methods.

This module provides a uniform interface for computing derivatives using different
numerical methods, automatically handling their specific input requirements.
"""

import numpy as np
from typing import Union, Optional, Tuple, Dict, Any
from enum import Enum

from .core import bspf1d

class DerivativeMethod(Enum):
    """Available derivative computation methods."""
    BSPF = "bspf"

class DerivativeComputer:
    """Unified interface for derivative computation using different methods.
    
    This class provides a consistent API for computing derivatives using BSPF,
    Chebyshev spectral, or Padé finite difference methods, automatically handling
    their specific input requirements.
    
    Parameters
    ----------
    method : DerivativeMethod or str
        The derivative computation method to use
    **kwargs
        Method-specific parameters (see method documentation)
        
    Examples
    --------
    >>> import numpy as np
    >>> from src.bspf import DerivativeComputer, DerivativeMethod
    >>> 
    >>> # Create uniform grid
    >>> x = np.linspace(0, 2*np.pi, 100)
    >>> f = np.sin(x)
    >>> 
    >>> # BSPF method
    >>> bspf_comp = DerivativeComputer('bspf', degree=11, domain=(0, 2*np.pi))
    """
    
    def __init__(self, method: Union[DerivativeMethod, str], **kwargs):
        self.method = DerivativeMethod(method) if isinstance(method, str) else method
        self.kwargs = kwargs
        self._bspf_model = None
        self._pade_operator = None
        
    def compute_derivative(
        self, 
        f: np.ndarray, 
        x: np.ndarray, 
        k: int = 1,
        **method_kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Compute derivative using the specified method.
        
        Parameters
        ----------
        f : ndarray
            Function values at grid points
        x : ndarray
            Grid points (must be uniformly spaced for BSPF and Padé)
        k : int, optional
            Derivative order (default: 1)
        **method_kwargs
            Additional method-specific parameters
            
        Returns
        -------
        ndarray or tuple
            For BSPF and Padé: derivative values at the same grid points
            For Chebyshev: if input is not Chebyshev nodes, returns (derivative, chebyshev_nodes)
            
        Raises
        ------
        ValueError
            If method parameters are invalid or grid is not uniform
        """
        f = np.asarray(f, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        
        if len(f) != len(x):
            raise ValueError("f and x must have the same length")
        if len(f) < 2:
            raise ValueError("Need at least 2 points for derivative computation")
            
        # Merge method-specific kwargs
        params = {**self.kwargs, **method_kwargs}
        
        if self.method == DerivativeMethod.BSPF:
            return self._compute_bspf_derivative(f, x, k, params)
        elif self.method == DerivativeMethod.CHEBYSHEV:
            return self._compute_chebyshev_derivative(f, x, k, params)
        elif self.method == DerivativeMethod.PADE:
            return self._compute_pade_derivative(f, x, k, params)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _compute_bspf_derivative(self, f: np.ndarray, x: np.ndarray, k: int, params: Dict[str, Any]) -> np.ndarray:
        """Compute derivative using BSPF method."""
        # Check if grid is uniform
        dx = np.diff(x)
        if not np.allclose(dx, dx[0], rtol=1e-10):
            raise ValueError("BSPF method requires uniform grid")
            
        # Initialize BSPF model if not already done
        if self._bspf_model is None:
            domain = params.get('domain', (float(x[0]), float(x[-1])))
            degree = params.get('degree', 11)
            order = params.get('order', degree)
            n_basis = params.get('n_basis', 4 * degree)
            num_boundary_points = params.get('num_boundary_points', degree + 5)
            use_clustering = params.get('use_clustering', True)
            clustering_factor = params.get('clustering_factor', 2.0)
            correction = params.get('correction', 'spectral')
            
            self._bspf_model = bspf1d.from_grid(
                degree=degree,
                x=x,
                domain=domain,
                order=order,
                n_basis=n_basis,
                num_boundary_points=num_boundary_points,
                use_clustering=use_clustering,
                clustering_factor=clustering_factor,
                correction=correction
            )
        
        # Compute derivative
        lam = params.get('lam', 0.0)
        neumann_bc = params.get('neumann_bc', None)
        
        return self._bspf_model.differentiate(f, k=k, lam=lam, neumann_bc=neumann_bc)

def compute_derivative(
    f: np.ndarray,
    x: np.ndarray,
    method: Union[DerivativeMethod, str] = DerivativeMethod.BSPF,
    k: int = 1,
    **kwargs
) -> np.ndarray:
    """Convenience function for one-shot derivative computation.
    
    Parameters
    ----------
    f : ndarray
        Function values at grid points
    x : ndarray
        Grid points
    method : DerivativeMethod or str, optional
        Derivative computation method (default: 'bspf')
    k : int, optional
        Derivative order (default: 1)
    **kwargs
        Method-specific parameters
        
    Returns
    -------
    ndarray
        Derivative values at the same grid points
        
    Examples
    --------
    >>> import numpy as np
    >>> from src.bspf import compute_derivative
    >>> 
    >>> x = np.linspace(0, 2*np.pi, 100)
    >>> f = np.sin(x)
    >>> 
    >>> # Compute derivative using different methods
    >>> df_bspf = compute_derivative(f, x, method='bspf', degree=11)
    """
    computer = DerivativeComputer(method, **kwargs)
    return computer.compute_derivative(f, x, k=k)