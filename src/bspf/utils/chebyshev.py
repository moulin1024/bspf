"""
In-house Chebyshev spectral differentiation utilities.

Uses FFT-based DCT-I for coefficient computation and stable recurrence
for derivative computation.
"""

import numpy as np


# ---------- FFT-based Chebyshev helpers (DCT-I / IDCT-I via rfft/irfft) ----------

def _chebyshev_coeffs_rfft(f_vals):
    """
    Chebyshev–I coefficients a_k from values at first-kind nodes via
    an rFFT-based DCT-I. Length of `f_vals` must be N+1.
    Scaling convention matches the even-FFT version:
        v_ext = [f_0, ..., f_N, f_{N-1}, ..., f_1]  (length 2N)
        a_k   = (1/N) * Re{RFFT_k(v_ext)} with a_0 and a_N halved.
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


def _values_from_cheb_coeffs_irfft(a):
    """
    Inverse DCT-I: given Chebyshev–I coefficients a_k (with endpoints halved
    per `_chebyshev_coeffs_rfft`), return the values at first-kind nodes.

    Returns array of length N+1 with f(t_k), k=0..N, where t_k=cos(pi*k/N).
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


# ---------- Coefficient-space derivative (unchanged complexity, stable) ----------

def _chebyshev_derivative_coeffs(a):
    """
    Vectorized version (no Python loop).
    Given Chebyshev T_k coefficients a_k, return b_k for d/dx in the T_k basis.
    Uses the fact that b_j = b_{j+2} + 2(j+1) a_{j+1} with b_N = 0 and
    b_{N-1} = 2N a_N, which splits into two parity chains.
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


def _chebyshev_second_derivative_coeffs(a):
    """
    Compute second derivative coefficients directly from original coefficients.
    This avoids intermediate evaluation errors by applying the derivative
    transformation twice in coefficient space.
    
    Parameters
    ----------
    a : ndarray
        Chebyshev coefficients for the function
        
    Returns
    -------
    c : ndarray
        Chebyshev coefficients for the second derivative
    """
    # Apply derivative transformation twice
    b = _chebyshev_derivative_coeffs(a)  # First derivative coefficients
    c = _chebyshev_derivative_coeffs(b)  # Second derivative coefficients
    return c


def construct_chebyshev_nodes(N, domain=(-1.0, 1.0)):
    """
    Construct first-kind Chebyshev nodes and map them to the given domain.
    
    Parameters
    ----------
    N : int
        Number of *intervals* (→ N+1 Chebyshev nodes).
    domain : (float, float), optional
        Interval [a, b] to map nodes to (default [-1, 1]).
        
    Returns
    -------
    x : ndarray
        Mapped Chebyshev nodes
    t : ndarray
        Original nodes on [-1, 1]
    """
    a_dom, b_dom = domain
    k = np.arange(N + 1)
    t = np.cos(np.pi * k / N)                 # first-kind nodes on [-1,1]
    x = (b_dom - a_dom) * 0.5 * t + (b_dom + a_dom) * 0.5
    return x, t


def chebyshev_derivative_from_values(f_vals, x, domain=(-1.0, 1.0)):
    """
    Compute first derivative using Chebyshev spectral method.
    
    Parameters
    ----------
    f_vals : ndarray
        Function values at Chebyshev nodes
    x : ndarray
        Pre-computed Chebyshev nodes mapped to domain
    domain : (float, float), optional
        Interval [a, b] on which nodes are mapped (default [-1, 1])
    
    Returns
    -------
    df_dx : ndarray
        First derivative values at the nodes
    """
    if len(f_vals) < 2:
        raise ValueError("f_vals must have length at least 2.")
    a_dom, b_dom = domain
    if not np.isfinite(a_dom) or not np.isfinite(b_dom) or a_dom == b_dom:
        raise ValueError("`domain` must be finite with a != b.")

    # 1) Chebyshev coefficients via rFFT-based DCT-I
    a_k = _chebyshev_coeffs_rfft(f_vals)

    # 2) Derivative coefficients via stable recurrence
    b_k = _chebyshev_derivative_coeffs(a_k)

    # 3) Evaluate derivative series at the nodes via inverse DCT-I
    df_dt = _values_from_cheb_coeffs_irfft(b_k)

    # 4) Chain rule for x-mapping t -> x
    df_dx = df_dt * (2.0 / (b_dom - a_dom))

    return df_dx


def chebyshev_second_derivative_from_values(f_vals, x, domain=(-1.0, 1.0)):
    """
    Compute second derivative using Chebyshev spectral method.
    
    This function computes the second derivative directly from the original
    function values by applying the derivative transformation twice in
    coefficient space, avoiding intermediate evaluation errors.
    
    Parameters
    ----------
    f_vals : ndarray
        Function values at Chebyshev nodes
    x : ndarray
        Pre-computed Chebyshev nodes mapped to domain
    domain : (float, float), optional
        Interval [a, b] on which nodes are mapped (default [-1, 1])
    
    Returns
    -------
    d2f_dx2 : ndarray
        Second derivative values at the nodes
    """
    if len(f_vals) < 2:
        raise ValueError("f_vals must have length at least 2.")
    a_dom, b_dom = domain
    if not np.isfinite(a_dom) or not np.isfinite(b_dom) or a_dom == b_dom:
        raise ValueError("`domain` must be finite with a != b.")

    # 1) Chebyshev coefficients via rFFT-based DCT-I
    a_k = _chebyshev_coeffs_rfft(f_vals)

    # 2) Second derivative coefficients via stable recurrence (applied twice)
    c_k = _chebyshev_second_derivative_coeffs(a_k)

    # 3) Evaluate second derivative series at the nodes via inverse DCT-I
    d2f_dt2 = _values_from_cheb_coeffs_irfft(c_k)

    # 4) Chain rule for x-mapping t -> x: d²/dx² = (2/(b-a))² * d²/dt²
    d2f_dx2 = d2f_dt2 * (2.0 / (b_dom - a_dom))**2

    return d2f_dx2


