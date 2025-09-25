"""Chebyshev spectral differentiation methods.

This module provides efficient Chebyshev spectral methods for computing derivatives
using FFT-based discrete cosine transforms (DCT-I). The implementation uses
stable coefficient-space recurrence relations for high accuracy.

Main Functions
--------------
chebyshev_derivative : Compute derivative using Chebyshev spectral method
construct_chebyshev_nodes : Generate Chebyshev nodes mapped to arbitrary domain

Notes
-----
The implementation follows Trefethen's "Spectral Methods in MATLAB" (2000)
for the derivative coefficient recurrence relations.
"""

import numpy as np
from typing import Tuple, Union


# Core Chebyshev spectral methods
# ================================

def _chebyshev_coeffs_from_values(f_vals: np.ndarray) -> np.ndarray:
    """
    Compute Chebyshev coefficients from function values at Chebyshev nodes.
    
    Uses FFT-based DCT-I via real FFT for efficiency. The input should be
    function values at first-kind Chebyshev nodes.
    
    Parameters
    ----------
    f_vals : ndarray
        Function values at N+1 Chebyshev nodes
        
    Returns
    -------
    ndarray
        Chebyshev coefficients a_k for the expansion
        f(x) = sum_{k=0}^N a_k T_k(x)
        
    Notes
    -----
    Uses scaling convention: v_ext = [f_0, ..., f_N, f_{N-1}, ..., f_1]
    with a_k = (1/N) * Re{RFFT_k(v_ext)} and a_0, a_N halved.
    """
    N = f_vals.size - 1
    if N < 0:
        raise ValueError("f_vals must have length at least 1 (N+1 >= 1).")
    if N == 0:
        # With only one node, a_0 = f(x_0)
        return f_vals.copy()

    # Build even extension without concatenation
    v_ext = np.empty(2 * N, dtype=f_vals.dtype)
    v_ext[:N + 1] = f_vals
    v_ext[N + 1:] = f_vals[-2:0:-1]

    half = np.fft.rfft(v_ext)        # length N+1
    a = (half.real / N)              # real for even real input
    a[0] *= 0.5
    a[-1] *= 0.5
    return a


def _values_from_chebyshev_coeffs(a: np.ndarray) -> np.ndarray:
    """
    Evaluate Chebyshev series at Chebyshev nodes using inverse DCT-I.
    
    Parameters
    ----------
    a : ndarray
        Chebyshev coefficients (with endpoints appropriately scaled)
        
    Returns
    -------
    ndarray
        Function values at N+1 first-kind Chebyshev nodes t_k = cos(πk/N)
        
    Notes
    -----
    This is the inverse operation of _chebyshev_coeffs_from_values.
    """
    N = a.size - 1
    if N < 0:
        raise ValueError("a must have length at least 1 (N+1 >= 1).")
    if N == 0:
        return a.copy()

    # Rebuild the rFFT half-spectrum that would have been produced in forward pass
    half = a.astype(np.result_type(a.dtype, np.float64), copy=True)
    half[0] *= 2.0
    half[-1] *= 2.0
    half *= N

    v_ext = np.fft.irfft(half, n=2 * N)
    return v_ext[:N + 1]


def _chebyshev_derivative_coeffs(a: np.ndarray) -> np.ndarray:
    """
    Compute Chebyshev coefficients for the derivative of a Chebyshev series.
    
    Given coefficients a_k for f(x) = sum a_k T_k(x), computes coefficients
    b_k for f'(x) = sum b_k T_k(x) using the stable vectorized recurrence.
    
    Parameters
    ----------
    a : ndarray
        Chebyshev coefficients for the function
        
    Returns
    -------
    ndarray
        Chebyshev coefficients for the derivative
        
    Notes
    -----
    Uses the recurrence relation from Trefethen's "Spectral Methods in MATLAB":
    b_j = b_{j+2} + 2(j+1) a_{j+1}, with boundary conditions b_N = 0,
    b_{N-1} = 2N a_N, and final scaling b_0 *= 0.5.
    
    The implementation is vectorized using parity chains for efficiency.
    """
    N = a.size - 1
    b = np.zeros_like(a)
    if N == 0:
        return b

    # r_j = 2 (j+1) a_{j+1}, j = 0..N-1  (length N)
    # Build it as r = (2*k*a_k)[1:] with k = 0..N
    r = (2.0 * np.arange(N + 1)) * a
    r = r[1:]  # length N, index j corresponds to coefficient a_{j+1}

    # Descending index lists for the two parity chains
    idx0 = np.arange(N - 1, -1, -2)  # same parity as N-1
    idx1 = np.arange(N - 2, -1, -2)

    # b_j = r_j + r_{j+2} + r_{j+4} + ...  (reversed cumsum on each chain)
    b[idx0] = np.cumsum(r[idx0])
    if idx1.size:
        b[idx1] = np.cumsum(r[idx1])

    # b_N is already zero; final halving of b_0 per the standard recurrence
    b[0] *= 0.5
    return b


def construct_chebyshev_nodes(N: int, domain: Tuple[float, float] = (-1.0, 1.0)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct first-kind Chebyshev nodes and map them to a given domain.
    
    Parameters
    ----------
    N : int
        Number of intervals (produces N+1 Chebyshev nodes)
    domain : tuple of float, optional
        Target interval [a, b] (default: [-1, 1])
        
    Returns
    -------
    x : ndarray
        Chebyshev nodes mapped to the target domain
    t : ndarray
        Original Chebyshev nodes on [-1, 1]
        
    Notes
    -----
    First-kind Chebyshev nodes are given by t_k = cos(πk/N) for k = 0, 1, ..., N.
    These are then linearly mapped to the target domain [a, b].
    """
    a_dom, b_dom = domain
    k = np.arange(N + 1)
    t = np.cos(np.pi * k / N)                 # first-kind nodes on [-1,1]
    x = (b_dom - a_dom) * 0.5 * t + (b_dom + a_dom) * 0.5
    return x, t

def chebyshev_derivative(f_vals: np.ndarray, 
                        x: np.ndarray = None, 
                        domain: Tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
    """
    Compute derivative using Chebyshev spectral differentiation.
    
    This function computes the derivative of a function given its values
    at Chebyshev nodes using FFT-based spectral methods.
    
    Parameters
    ----------
    f_vals : ndarray
        Function values at Chebyshev nodes
    x : ndarray, optional
        Pre-computed Chebyshev nodes (not used in computation, kept for compatibility)
    domain : tuple of float, optional
        Domain interval [a, b] for the Chebyshev nodes (default: [-1, 1])
    
    Returns
    -------
    ndarray
        Derivative values at the same Chebyshev nodes
        
    Examples
    --------
    >>> import numpy as np
    >>> # Create Chebyshev nodes for domain [0, 2π]
    >>> x, _ = construct_chebyshev_nodes(100, (0, 2*np.pi))
    >>> f_vals = np.sin(x)  # Function values
    >>> df_dx = chebyshev_derivative(f_vals, domain=(0, 2*np.pi))
    >>> # df_dx should approximate cos(x)
    
    Notes
    -----
    The algorithm follows these steps:
    1. Transform function values to Chebyshev coefficients via DCT-I
    2. Apply derivative operator in coefficient space using stable recurrence
    3. Transform back to physical space via inverse DCT-I
    4. Apply chain rule scaling for domain mapping
    """
    if len(f_vals) < 2:
        raise ValueError("f_vals must have length at least 2.")
    a_dom, b_dom = domain
    if not np.isfinite(a_dom) or not np.isfinite(b_dom) or a_dom == b_dom:
        raise ValueError("`domain` must be finite with a != b.")

    # 1) Transform to Chebyshev coefficient space
    a_k = _chebyshev_coeffs_from_values(f_vals)

    # 2) Apply derivative operator in coefficient space
    b_k = _chebyshev_derivative_coeffs(a_k)

    # 3) Transform back to physical space
    df_dt = _values_from_chebyshev_coeffs(b_k)

    # 4) Chain rule for x-mapping t -> x
    df_dx = df_dt * (2.0 / (b_dom - a_dom))

    return df_dx


# Convenience functions and aliases
# ==================================

# Keep old function name for backward compatibility
chebyshev_derivative_from_values = chebyshev_derivative

# Keep old internal function name for backward compatibility 
_construct_chebyshev_nodes = construct_chebyshev_nodes


if __name__ == "__main__":
    # Simple demonstration and test
    import matplotlib.pyplot as plt
    import time
    
    print("Chebyshev Spectral Differentiation Demo")
    print("=======================================\n")
    
    # Test function: f(x) = sin(10x) * exp(x/2)
    domain = (0.0, 2.0)
    alpha = 10
    
    func = lambda x: np.sin(alpha * x) * np.exp(x/2)
    dfunc = lambda x: (alpha * np.cos(alpha * x) + 0.5 * np.sin(alpha * x)) * np.exp(x/2)
    
    # Test with different grid sizes
    n_test = 64
    x, _ = construct_chebyshev_nodes(n_test, domain)
    f_vals = func(x)
    df_numerical = chebyshev_derivative(f_vals, domain=domain)
    df_exact = dfunc(x)
    
    error = np.max(np.abs(df_numerical - df_exact))
    print(f"Test with N={n_test+1} nodes:")
    print(f"Maximum error: {error:.2e}\n")
    
    # Simple convergence test
    print("Convergence study:")
    print("N\t\tMax Error")
    print("-" * 25)
    
    for n in [16, 32, 64, 128]:
        x_test, _ = construct_chebyshev_nodes(n, domain)
        f_test = func(x_test)
        df_test = chebyshev_derivative(f_test, domain=domain)
        df_exact_test = dfunc(x_test)
        error_test = np.max(np.abs(df_test - df_exact_test))
        print(f"{n+1}\t\t{error_test:.2e}")
