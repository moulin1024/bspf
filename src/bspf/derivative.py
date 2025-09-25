"""Unified derivative computation interface for BSPF, Chebyshev, and Padé methods.

This module provides a uniform interface for computing derivatives using different
numerical methods, automatically handling their specific input requirements.
"""

import numpy as np
from typing import Union, Optional, Tuple, Dict, Any
from enum import Enum

from .core import bspf1d
from .chebyshev import chebyshev_derivative, construct_chebyshev_nodes
from .padefd import padefd


class DerivativeMethod(Enum):
    """Available derivative computation methods."""
    BSPF = "bspf"
    CHEBYSHEV = "chebyshev"
    PADE = "pade"


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
    >>> df_bspf = bspf_comp.compute_derivative(f, x)
    >>> 
    >>> # Chebyshev method
    >>> cheb_comp = DerivativeComputer('chebyshev', domain=(0, 2*np.pi))
    >>> df_cheb = cheb_comp.compute_derivative(f, x)
    >>> 
    >>> # Padé method
    >>> pade_comp = DerivativeComputer('pade', order=10)
    >>> df_pade = pade_comp.compute_derivative(f, x)
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
    
    def _compute_chebyshev_derivative(self, f: np.ndarray, x: np.ndarray, k: int, params: Dict[str, Any]) -> np.ndarray:
        """Compute derivative using Chebyshev spectral method.
        
        Note: Chebyshev spectral methods require evaluation on Chebyshev nodes
        to maintain spectral accuracy. If the input grid is not Chebyshev nodes,
        this method will return the derivative evaluated on Chebyshev nodes
        rather than interpolating back to the original grid.
        """
        if k != 1:
            raise ValueError("Chebyshev method currently only supports first derivatives")
            
        domain = params.get('domain', (float(x[0]), float(x[-1])))
        
        # For Chebyshev method, we need to work with Chebyshev nodes
        # Use the same number of points as the input
        N = len(x) - 1  # Number of intervals
        x_cheb, _ = construct_chebyshev_nodes(N, domain)
        
        # Check if input grid is already Chebyshev nodes
        if np.allclose(x, x_cheb, rtol=1e-10):
            # Input is already on Chebyshev nodes - use directly
            return chebyshev_derivative(f, domain=domain)
        else:
            # Input is not on Chebyshev nodes
            # For spectral accuracy, we should work on Chebyshev nodes
            # Interpolate function values to Chebyshev nodes
            f_cheb = np.interp(x_cheb, x, f)
            
            # Compute derivative at Chebyshev nodes
            df_cheb = chebyshev_derivative(f_cheb, domain=domain)
            
            # Return derivative on Chebyshev nodes (not interpolated back)
            # This maintains spectral accuracy
            return df_cheb, x_cheb
    
    def _compute_pade_derivative(self, f: np.ndarray, x: np.ndarray, k: int, params: Dict[str, Any]) -> np.ndarray:
        """Compute derivative using Padé finite difference method."""
        if k != 1:
            raise ValueError("Padé method currently only supports first derivatives")
            
        # Check if grid is uniform
        dx = np.diff(x)
        if not np.allclose(dx, dx[0], rtol=1e-10):
            raise ValueError("Padé method requires uniform grid")
            
        h = float(dx[0])
        N = len(x)
        order = params.get('order', 10)
        acc = params.get('acc', None)
        
        # Initialize Padé operator if not already done or parameters changed
        if (self._pade_operator is None or 
            self._pade_operator.N != N or 
            self._pade_operator.h != h or 
            self._pade_operator.order != order):
            
            self._pade_operator = padefd(N, h, order=order, acc=acc)
        
        return self._pade_operator(f)
    
    def compute_derivative_on_chebyshev_nodes(
        self, 
        f: np.ndarray, 
        x: np.ndarray, 
        k: int = 1,
        **method_kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute derivative on Chebyshev nodes for spectral accuracy.
        
        This method is specifically designed for Chebyshev spectral methods
        and ensures the derivative is computed on the appropriate Chebyshev nodes.
        
        Parameters
        ----------
        f : ndarray
            Function values at grid points
        x : ndarray
            Grid points
        k : int, optional
            Derivative order (default: 1)
        **method_kwargs
            Additional method-specific parameters
            
        Returns
        -------
        tuple
            (derivative_values, chebyshev_nodes)
        """
        if self.method != DerivativeMethod.CHEBYSHEV:
            raise ValueError("This method is only available for Chebyshev spectral methods")
            
        f = np.asarray(f, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        
        if len(f) != len(x):
            raise ValueError("f and x must have the same length")
        if len(f) < 2:
            raise ValueError("Need at least 2 points for derivative computation")
            
        params = {**self.kwargs, **method_kwargs}
        return self._compute_chebyshev_derivative(f, x, k, params)


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
    >>> df_cheb = compute_derivative(f, x, method='chebyshev')
    >>> df_pade = compute_derivative(f, x, method='pade', order=10)
    """
    computer = DerivativeComputer(method, **kwargs)
    return computer.compute_derivative(f, x, k=k)


def compare_methods(
    f: np.ndarray,
    x: np.ndarray,
    methods: Optional[list] = None,
    k: int = 1,
    **kwargs
) -> Dict[str, np.ndarray]:
    """Compare derivative computation across multiple methods.
    
    Parameters
    ----------
    f : ndarray
        Function values at grid points
    x : ndarray
        Grid points
    methods : list, optional
        List of methods to compare (default: all available methods)
    k : int, optional
        Derivative order (default: 1)
    **kwargs
        Method-specific parameters
        
    Returns
    -------
    dict
        Dictionary mapping method names to derivative arrays
        
    Examples
    --------
    >>> import numpy as np
    >>> from src.bspf import compare_methods
    >>> 
    >>> x = np.linspace(0, 2*np.pi, 100)
    >>> f = np.sin(x)
    >>> 
    >>> results = compare_methods(f, x, methods=['bspf', 'chebyshev', 'pade'])
    >>> for method, df in results.items():
    ...     print(f"{method}: max error = {np.max(np.abs(df - np.cos(x)))}")
    """
    if methods is None:
        methods = [DerivativeMethod.BSPF, DerivativeMethod.CHEBYSHEV, DerivativeMethod.PADE]
    
    results = {}
    for method in methods:
        try:
            computer = DerivativeComputer(method, **kwargs)
            method_name = method.value if hasattr(method, 'value') else str(method)
            results[method_name] = computer.compute_derivative(f, x, k=k)
        except Exception as e:
            method_name = method.value if hasattr(method, 'value') else str(method)
            print(f"Warning: Failed to compute derivative with {method_name}: {e}")
            continue
    
    return results


def compute_derivative_on_chebyshev_nodes(
    f: np.ndarray,
    x: np.ndarray,
    domain: Optional[Tuple[float, float]] = None,
    k: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function for computing derivatives on Chebyshev nodes.
    
    This function ensures spectral accuracy by computing the derivative
    on the appropriate Chebyshev nodes for the given domain.
    
    Parameters
    ----------
    f : ndarray
        Function values at grid points
    x : ndarray
        Grid points
    domain : tuple, optional
        Domain interval [a, b] (default: [x[0], x[-1]])
    k : int, optional
        Derivative order (default: 1)
        
    Returns
    -------
    tuple
        (derivative_values, chebyshev_nodes)
        
    Examples
    --------
    >>> import numpy as np
    >>> from src.bspf import compute_derivative_on_chebyshev_nodes
    >>> 
    >>> x = np.linspace(0, 2*np.pi, 100)
    >>> f = np.sin(x)
    >>> 
    >>> df_cheb, x_cheb = compute_derivative_on_chebyshev_nodes(f, x)
    >>> # df_cheb is the derivative evaluated at Chebyshev nodes x_cheb
    """
    if domain is None:
        domain = (float(x[0]), float(x[-1]))
    
    computer = DerivativeComputer(DerivativeMethod.CHEBYSHEV, domain=domain)
    return computer.compute_derivative_on_chebyshev_nodes(f, x, k=k)
