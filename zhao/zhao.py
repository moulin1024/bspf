"""
Zhao2025 boundary-interval extension + FFT spectral derivatives (Optimized).

Optimizations:
  1. Precomputes the extension operator M = A_gap @ pinv(A) to replace
     runtime least-squares solves with matrix multiplication.
  2. Supports batched inputs (y shape: [..., N]) for vectorized processing.
  3. Allows disabling grid checks for tight loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import numpy as np

try:
    # use scipy pinv/lstsq for stability if available
    from scipy.linalg import pinv
except ImportError:
    # fallback to numpy if scipy is missing
    from numpy.linalg import pinv


def _pinv_compat(A: np.ndarray, *, rcond: float) -> np.ndarray:
    """
    Compatibility wrapper for pseudoinverse across NumPy/SciPy versions.

    - NumPy: pinv(A, rcond=...)
    - SciPy: pinv(A, rtol=...) / pinv(A, atol=..., rtol=...) (no rcond)
    - Some SciPy versions used `cond` historically.
    """
    try:
        return pinv(A, rcond=rcond)  # NumPy API
    except TypeError:
        pass
    try:
        return pinv(A, rtol=rcond)  # SciPy API (rtol)
    except TypeError:
        pass
    try:
        return pinv(A, cond=rcond)  # legacy SciPy API
    except TypeError:
        return pinv(A)


@dataclass(frozen=True)
class _ZhaoExtOperator:
    # Precomputed linear operator mapping boundary values directly to the extension gap
    # Shape: (2*m_delta, num_ext_points) -> Transposed for efficient (..., 2m) @ (2m, gap)
    op_T: np.ndarray | None  
    indices_right: np.ndarray
    indices_left: np.ndarray
    num_ext_points: int


@lru_cache(maxsize=128)
def _precompute_zhao_operator(
    *,
    M: int,
    T_delta: int,
    m_delta: int,
    gamma_delta: int,
    rcond: float,
) -> _ZhaoExtOperator:
    """
    Precompute the linear operator that maps boundary samples to the extension gap.
    
    The extension g_gap is defined by:
        c = argmin ||A c - b||  => c = pinv(A) @ b
        g_gap = A_gap @ c
    
    We precompute M = A_gap @ pinv(A). 
    At runtime: g_gap = M @ b.
    """
    n_delta = int((m_delta - 1) / gamma_delta)
    L_delta = 2 * int(np.ceil(T_delta * (m_delta - 1)))
    
    # 1. Setup Grid & Matrix A
    # Using simple linspace is sufficient; no need for large arrays here
    x_ext = np.linspace(0.0, 2.0 * np.pi, L_delta, endpoint=False)
    k_vals = np.arange(-n_delta, n_delta + 1)
    
    idx_J1 = np.arange(0, m_delta)
    idx_J2 = np.arange(L_delta // 2, L_delta // 2 + m_delta)
    idx_J = np.concatenate([idx_J1, idx_J2])
    x_sample = x_ext[idx_J]

    # Design matrix A (2*m_delta, 2*n_delta+1)
    A = np.exp(1j * x_sample[:, None] * k_vals[None, :]) / np.sqrt(L_delta)

    # 2. Setup Gap Matrix A_gap
    num_ext_points = (L_delta // 2) - m_delta
    
    op_T = None
    if num_ext_points > 0:
        ext_indices = np.arange(m_delta, m_delta + num_ext_points)
        x_gap = x_ext[ext_indices]
        
        # Matrix mapping coeffs to gap (num_ext, 2*n_delta+1)
        A_gap = np.exp(1j * x_gap[:, None] * k_vals[None, :]) / np.sqrt(L_delta)
        
        # 3. Compute fused operator
        # pinv_A shape: (cols, rows)
        pinv_A = _pinv_compat(A, rcond=rcond)
        
        # Operator shape: (num_ext, 2*m_delta)
        # We store Transpose for easier matmul with input vectors: (2*m_delta, num_ext)
        op_T = (A_gap @ pinv_A).T

    # 4. Indices relative to input array
    # Python slices are cheaper than fancy indexing if contiguous, but these are split
    # so we keep index arrays.
    indices_right = (2 * M) - m_delta + 1 + np.arange(m_delta)
    indices_left = np.arange(m_delta)

    return _ZhaoExtOperator(
        op_T=op_T,
        indices_right=indices_right,
        indices_left=indices_left,
        num_ext_points=num_ext_points,
    )


@lru_cache(maxsize=128)
def _fft_multiplier(N_ext: int, *, M: int, order: int) -> np.ndarray:
    """Cache FFT derivative multipliers."""
    dt = 1.0 / M
    freqs = np.fft.fftfreq(N_ext, d=dt)
    omega = 2.0 * np.pi * freqs
    mult = (1j * omega) ** order
    if N_ext % 2 == 0:
        mult[N_ext // 2] = 0.0
    mult.setflags(write=False)
    return mult


def boundary_interval_extension_samples(
    f_vals: np.ndarray,
    *,
    M: int,
    T_delta: int = 6,
    m_delta: int = 25,
    gamma_delta: int = 1,
    rcond: float = 1e-14,
) -> np.ndarray:
    """
    Optimized periodic extension. Supports batched inputs.

    Parameters
    ----------
    f_vals : np.ndarray
        Input samples. Shape (..., 2M+1). The last dimension is the grid.
    M : int
        Grid parameter (N = 2M+1).

    Returns
    -------
    f_ext : np.ndarray
        Extended samples. Shape (..., N_ext).
    """
    # Ensure last dimension is correct
    if f_vals.shape[-1] != 2 * M + 1:
        raise ValueError(f"Last dimension of f_vals must be 2*M+1 (got {f_vals.shape[-1]})")
    
    # Retrieve cached operator
    op = _precompute_zhao_operator(
        M=M, T_delta=T_delta, m_delta=m_delta, gamma_delta=gamma_delta, rcond=rcond
    )

    if op.op_T is None:
        return f_vals

    # Construct boundary vector b
    # Shape: (..., 2*m_delta)
    # We use np.take or direct indexing. For batching, simple indexing with broadcasting works best
    # if indices are for the last axis.
    b_right = f_vals[..., op.indices_right]
    b_left = f_vals[..., op.indices_left]
    b_vals = np.concatenate([b_right, b_left], axis=-1)

    # Compute Gap: g = b @ op.T
    # b: (..., 2m), op.T: (2m, gap) -> (..., gap)
    # If f_vals is 1D (2M+1,), b is (2m,), result (gap,)
    # If f_vals is (B, 2M+1), b is (B, 2m), result (B, gap)
    g_gap = np.real(b_vals @ op.op_T)

    # Concatenate final result
    return np.concatenate([f_vals, g_gap], axis=-1)


def zhao2025_spectral_derivative(
    x: np.ndarray,
    y: np.ndarray,
    *,
    order: int = 1,
    domain: tuple[float, float] | None = None,
    T_delta: int = 6,
    m_delta: int = 25,
    gamma_delta: int = 1,
    rcond: float = 1e-14,
    check_grid: bool = True,
) -> np.ndarray:
    """
    Batched spectral derivative using Zhao2025 extension.

    Parameters
    ----------
    x : np.ndarray
        1D grid points.
    y : np.ndarray
        Function values. Shape (..., N) where N matches x.size.
    check_grid : bool
        If False, skips O(N) grid uniformity checks. 
        Use this in loops where x is constant.
    """
    # Grid validation
    N = x.size
    if N % 2 == 0:
        raise ValueError("N must be odd (N=2M+1)")
    
    if check_grid:
        dx = x[1] - x[0]
        if not np.allclose(np.diff(x), dx, rtol=1e-10, atol=1e-14):
            raise ValueError("Uniform grid required")

    M = (N - 1) // 2

    # Verify y shape
    if y.shape[-1] != N:
        raise ValueError("Last dimension of y must match x size")

    # Extension (Vectorized)
    f_ext = boundary_interval_extension_samples(
        y, M=M, T_delta=T_delta, m_delta=m_delta, gamma_delta=gamma_delta, rcond=rcond
    )
    N_ext = f_ext.shape[-1]

    # FFT Derivative
    # Calculate multiplier (cached)
    mult = _fft_multiplier(N_ext, M=M, order=order)
    
    # Apply FFT along last axis
    coeffs = np.fft.fft(f_ext, axis=-1)
    df_ext = np.real(np.fft.ifft(coeffs * mult, axis=-1))

    # Scaling factor
    if domain is None:
        span = float(x[-1]) - float(x[0])
    else:
        span = domain[1] - domain[0]
        
    scale = (2.0 / span) ** order
    
    # Truncate and scale
    return scale * df_ext[..., :N]


def zhao2025_extend(
    x: np.ndarray,
    y: np.ndarray,
    *,
    domain: tuple[float, float] | None = None,
    T_delta: int = 6,
    m_delta: int = 25,
    gamma_delta: int = 1,
    rcond: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns extended grid and values. Supports batched y.
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    dx = x[1] - x[0]
    M = (N - 1) // 2

    y_ext = boundary_interval_extension_samples(
        y, M=M, T_delta=T_delta, m_delta=m_delta, gamma_delta=gamma_delta, rcond=rcond
    )
    
    N_ext = y_ext.shape[-1]
    
    # Extend x grid
    # We only extend x to the right based on dx
    x_ext = x[0] + dx * np.arange(N_ext, dtype=float)
    
    return x_ext, y_ext