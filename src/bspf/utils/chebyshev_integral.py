"""
Chebyshev spectral methods for integration.

FFT-based Chebyshev antiderivatives using DCT-I via rFFT/irFFT.
"""

import numpy as np
import numpy.typing as npt

from .chebyshev import (
    _chebyshev_coeffs_rfft,
    _values_from_cheb_coeffs_irfft,
    construct_chebyshev_nodes
)

Array = npt.NDArray[np.float64]


# --------------------------------------
# Chebyshev integration in coefficient space
# --------------------------------------
def _chebyshev_integral_coeffs_t(a):
    """
    Integrate Chebyshev-T series in t on [-1,1] using the same normalization
    as _chebyshev_coeffs_rfft / _values_from_cheb_coeffs_irfft.
    Returns coefficients b for G'(t)=f(t). b[0] is the free constant.
    """
    N = a.size - 1
    b = np.zeros_like(a)
    if N == 0:
        return b

    if N == 1:
        # f = a0*T0 + a1*T1  => ∫f dt = (a0) T1/1 + (a1/4) T2 + const
        b[1] = a[0]                      # special: a0 contributes fully to T1
        b[1] += 0.0                      # (keep structure; no a2 here)
        # b[N] for N=1 handled below
    else:
        # k = 1 needs special handling because our a0 is "halved"
        b[1] = 0.5 * (2.0 * a[0] - a[2])  # = (a0_std - a2)/(2*1)
        # k = 2..N-1: standard recurrence
        if N >= 3:
            k = np.arange(2, N)
            b[2:N] = (a[1:N-1] - a[3:N+1]) / (2.0 * k)

    # tail (k = N): a_{N+1}=0
    b[N] = a[N-1] / (2.0 * N)
    # b[0] left as the free constant; set it later to match the anchor
    return b


def _apply_constant_to_match_anchor(b_coeffs, anchor, target_value):
    """
    Adjust constant term b0 so that the series value matches target_value at the anchor.
    Anchor: "left" -> x=a (t=-1, last node); "right" -> x=b (t=+1, first node).
    """
    vals = _values_from_cheb_coeffs_irfft(b_coeffs)
    idx = -1 if anchor == "left" else 0
    C = float(target_value) - vals[idx]
    b_coeffs = b_coeffs.copy()
    b_coeffs[0] += C
    return b_coeffs


# -------------------------------------------------
# FFT-based antiderivatives with single-side anchor
# -------------------------------------------------
def chebyshev_antiderivatives_fft(
    f, N=64, domain=(-1.0, 1.0), order=2,
    anchor="left", c1=0.0, c2=0.0
):
    """
    Compute 1st (and optionally 2nd) antiderivatives via FFT-based Chebyshev integration.

    Collocation nodes are Chebyshev–Lobatto (descending): x[0]=b, x[-1]=a.

    We integrate in 't' and scale by s=(b-a)/2:
      U1(t) = C1 + s * ∫ f(t) dt, chosen so U1(anchor)=c1
      U2(t) = C2 + s * ∫ U1(t) dt, chosen so U2(anchor)=c2
    
    Parameters
    ----------
    f : callable
        Function to integrate
    N : int, default 64
        Number of intervals (N+1 nodes)
    domain : (float, float), default (-1.0, 1.0)
        Domain [a, b] for integration
    order : {1, 2}, default 2
        Order of antiderivative (1 or 2)
    anchor : {"left", "right"}, default "left"
        Anchor point for boundary condition
    c1 : float, default 0.0
        Value of first antiderivative at anchor
    c2 : float, default 0.0
        Value of second antiderivative at anchor (if order=2)
    
    Returns
    -------
    x_nodes : ndarray
        Chebyshev nodes (ascending order)
    U1_vals : ndarray
        First antiderivative values
    U2_vals : ndarray, optional
        Second antiderivative values (if order=2)
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")
    if N < 1:
        # N=0 trivial but not useful for anchored antiderivative; require at least 2 nodes
        raise ValueError("N must be >= 1.")

    a_dom, b_dom = domain
    s = 0.5 * (b_dom - a_dom)
    x_nodes, _ = construct_chebyshev_nodes(N, domain=domain)  # descending
    f_vals = f(x_nodes)

    # --- U1: integrate w.r.t t, then scale by s, then set constant at anchor ---
    a_f = _chebyshev_coeffs_rfft(f_vals)
    b_u1_t = _chebyshev_integral_coeffs_t(a_f)
    b_u1 = s * b_u1_t
    b_u1 = _apply_constant_to_match_anchor(b_u1, anchor, c1)
    U1_vals = _values_from_cheb_coeffs_irfft(b_u1)

    if order == 1:
        return x_nodes, U1_vals

    # --- U2: integrate U1 w.r.t t, scale by s, set constant at anchor ---
    a_u1 = _chebyshev_coeffs_rfft(U1_vals)
    b_u2_t = _chebyshev_integral_coeffs_t(a_u1)
    b_u2 = s * b_u2_t
    b_u2 = _apply_constant_to_match_anchor(b_u2, anchor, c2)
    U2_vals = _values_from_cheb_coeffs_irfft(b_u2)

    x_nodes = np.flip(x_nodes)
    U1_vals = np.flip(U1_vals)
    U2_vals = np.flip(U2_vals)

    return x_nodes, U1_vals, U2_vals

