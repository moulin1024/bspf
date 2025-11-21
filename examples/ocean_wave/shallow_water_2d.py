#!/usr/bin/env python3
"""
2-D shallow-water (BSLF spatial ops) + DCT/DST exponential dealias filters

BCs:
  • η : zero-flux (Neumann)  -> filtered with DCT-II (even extension)
  • M,N : Dirichlet(0)       -> filtered with DST-II (odd  extension)

We precompute bspf2d derivative plans once, then time-step explicitly.
"""

from __future__ import annotations
import os

# Set CUDA_PATH for NVHPC SDK before importing CuPy
# This ensures CuPy can find CUDA headers during JIT compilation
if 'CUDA_PATH' not in os.environ:
    nvhpc_cuda_path = '/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6'
    if os.path.exists(nvhpc_cuda_path):
        os.environ['CUDA_PATH'] = nvhpc_cuda_path
        os.environ['CUDA_HOME'] = nvhpc_cuda_path

from bspf import bspf2d
import numpy as np
from scipy.fft import dct, idct, dst, idst
import time

# Optional GPU backend
_HAS_CUPY = False
_HAS_CUPY_FFT = False
try:
    import cupy as cp
    _HAS_CUPY = True
    # Check if cupyx.scipy.fft has DCT/DST support
    try:
        import cupyx.scipy.fft as cpfft
        # Test if dct/dst functions exist
        if hasattr(cpfft, 'dct') and hasattr(cpfft, 'dst') and hasattr(cpfft, 'idct') and hasattr(cpfft, 'idst'):
            _HAS_CUPY_FFT = True
        else:
            # Fallback: use CPU for DCT/DST if not available
            cpfft = None
    except Exception:
        cpfft = None
except Exception:
    cp = None
    cpfft = None


# ───────────────────────────── helpers: BCs ─────────────────────────────

def _enforce_dirichlet_zero(F, use_gpu: bool = False) -> None:
    """Enforce Dirichlet zero boundary conditions. Works with NumPy and CuPy arrays.
    Optimized for GPU: uses single operation when possible."""
    if use_gpu and _HAS_CUPY and isinstance(F, cp.ndarray):
        # On GPU, use a single operation to set all boundaries at once
        # This reduces kernel launch overhead
        F[0, :] = 0.0
        F[-1, :] = 0.0
        F[:, 0] = 0.0
        F[:, -1] = 0.0
        # Note: CuPy will batch these operations, but we could also use:
        # F[0, :] = F[-1, :] = F[:, 0] = F[:, -1] = 0.0
        # However, separate assignments are clearer and CuPy optimizes them
    else:
        F[0, :]  = 0.0
        F[-1, :] = 0.0
        F[:, 0]  = 0.0
        F[:, -1] = 0.0

def _enforce_neumann_zero_eta_in_field(eta, use_gpu: bool = False) -> None:
    """Copy interior to boundary so stored field respects ∂η/∂n≈0. Works with NumPy and CuPy arrays.
    Optimized for GPU: uses single operation when possible."""
    if use_gpu and _HAS_CUPY and isinstance(eta, cp.ndarray):
        # On GPU, these operations are already efficient
        eta[0, :]  = eta[1, :]
        eta[-1, :] = eta[-2, :]
        eta[:, 0]  = eta[:, 1]
        eta[:, -1] = eta[:, -2]
    else:
        eta[0, :]  = eta[1, :]
        eta[-1, :] = eta[-2, :]
        eta[:, 0]  = eta[:, 1]
        eta[:, -1] = eta[:, -2]


# ─────────────────────── BC-compatible spectral filters ──────────────────────
# We use type-II transforms with 'ortho' norm so idct/dst(type=2) invert dct/dst(type=2)

def _dct2(a, use_gpu: bool = False):
    """2D DCT-II transform, GPU-aware."""
    if use_gpu and _HAS_CUPY_FFT:
        return cpfft.dct(cpfft.dct(a, type=2, axis=0, norm='ortho'), type=2, axis=1, norm='ortho')
    else:
        # Convert to CPU if needed for DCT
        if use_gpu and _HAS_CUPY and cp is not None:
            try:
                if isinstance(a, cp.ndarray):
                    a_cpu = cp.asnumpy(a)
                    result = dct(dct(a_cpu, type=2, axis=0, norm='ortho'), type=2, axis=1, norm='ortho')
                    result = cp.asarray(result)
                    return result
            except (AttributeError, TypeError):
                pass
        result = dct(dct(a, type=2, axis=0, norm='ortho'), type=2, axis=1, norm='ortho')
        return result

def _idct2(A, use_gpu: bool = False):
    """2D inverse DCT-II transform, GPU-aware."""
    if use_gpu and _HAS_CUPY_FFT:
        return cpfft.idct(cpfft.idct(A, type=2, axis=0, norm='ortho'), type=2, axis=1, norm='ortho')
    else:
        # Convert to CPU if needed for IDCT
        if use_gpu and _HAS_CUPY and cp is not None:
            try:
                if isinstance(A, cp.ndarray):
                    A_cpu = cp.asnumpy(A)
                    result = idct(idct(A_cpu, type=2, axis=0, norm='ortho'), type=2, axis=1, norm='ortho')
                    result = cp.asarray(result)
                    return result
            except (AttributeError, TypeError):
                pass
        result = idct(idct(A, type=2, axis=0, norm='ortho'), type=2, axis=1, norm='ortho')
        return result

def _dst2(a, use_gpu: bool = False):
    """2D DST-II transform, GPU-aware."""
    if use_gpu and _HAS_CUPY_FFT:
        return cpfft.dst(cpfft.dst(a, type=2, axis=0, norm='ortho'), type=2, axis=1, norm='ortho')
    else:
        # Convert to CPU if needed for DST
        if use_gpu and _HAS_CUPY and cp is not None:
            try:
                if isinstance(a, cp.ndarray):
                    a_cpu = cp.asnumpy(a)
                    result = dst(dst(a_cpu, type=2, axis=0, norm='ortho'), type=2, axis=1, norm='ortho')
                    result = cp.asarray(result)
                    return result
            except (AttributeError, TypeError):
                pass
        result = dst(dst(a, type=2, axis=0, norm='ortho'), type=2, axis=1, norm='ortho')
        return result

def _idst2(A, use_gpu: bool = False):
    """2D inverse DST-II transform, GPU-aware."""
    if use_gpu and _HAS_CUPY_FFT:
        return cpfft.idst(cpfft.idst(A, type=2, axis=0, norm='ortho'), type=2, axis=1, norm='ortho')
    else:
        # Convert to CPU if needed for IDST
        if use_gpu and _HAS_CUPY and cp is not None:
            try:
                if isinstance(A, cp.ndarray):
                    A_cpu = cp.asnumpy(A)
                    result = idst(idst(A_cpu, type=2, axis=0, norm='ortho'), type=2, axis=1, norm='ortho')
                    result = cp.asarray(result)
                    return result
            except (AttributeError, TypeError):
                pass
        result = idst(idst(A, type=2, axis=0, norm='ortho'), type=2, axis=1, norm='ortho')
        return result


def dct_filter_neumann(F, alpha: float = 36.0, order: int = 8, use_gpu: bool = False, sigma_cache=None):
    """
    Exponential filter in DCT-II space (even extension → compatible with Neumann).
    sigma(k) = exp( -alpha * (|k|/k_max)^order )
    GPU-aware. Uses precomputed sigma if provided in sigma_cache.
    """
    xp = cp if (use_gpu and _HAS_CUPY) else np
    ny, nx = F.shape
    
    # Use precomputed sigma if available, otherwise compute it
    if sigma_cache is not None and sigma_cache.shape == (ny, nx):
        sigma = sigma_cache
    else:
        ky = xp.arange(ny)           # DCT-II mode index 0..N-1
        kx = xp.arange(nx)
        ky_max = max(float(ky[-1]), 1)
        kx_max = max(float(kx[-1]), 1)

        KY = ky[:, None] / ky_max
        KX = kx[None, :] / kx_max
        rr = xp.sqrt(KY**2 + KX**2)
        sigma = xp.exp(-alpha * (rr**order))

    Fhat = _dct2(F, use_gpu=use_gpu)
    Fhat *= sigma
    result = _idct2(Fhat, use_gpu=use_gpu)
    
    return result


def dst_filter_dirichlet(F, alpha: float = 36.0, order: int = 8, use_gpu: bool = False, sigma_cache=None):
    """
    Exponential filter in DST-II space (odd extension → compatible with Dirichlet 0).
    sigma(k) = exp( -alpha * (|k|/k_max)^order )
    GPU-aware. Uses precomputed sigma if provided in sigma_cache.
    """
    xp = cp if (use_gpu and _HAS_CUPY) else np
    ny, nx = F.shape
    
    # Use precomputed sigma if available, otherwise compute it
    if sigma_cache is not None and sigma_cache.shape == (ny, nx):
        sigma = sigma_cache
    else:
        ky = xp.arange(1, ny + 1)    # DST-II modes 1..N
        kx = xp.arange(1, nx + 1)
        ky_max = max(float(ky[-1]), 1)
        kx_max = max(float(kx[-1]), 1)

        KY = ky[:, None] / ky_max
        KX = kx[None, :] / kx_max
        rr = xp.sqrt(KY**2 + KX**2)
        sigma = xp.exp(-alpha * (rr**order))

    Fhat = _dst2(F, use_gpu=use_gpu)
    Fhat *= sigma
    result = _idst2(Fhat, use_gpu=use_gpu)
    
    return result


# ───────────────────────── BSLF update kernels ──────────────────────────

def update_eta_2D_bspf(eta, M, N, dt, plan_dx, plan_dy, use_gpu: bool = False):
    """
    Continuity: η_t = -(M_x + N_y).
    We impose Dirichlet(0) on M,N before taking derivatives.
    Optimized: enforce boundaries in-place (M and N are updated later in timestep anyway).
    """
    # Enforce boundaries in-place on M and N
    # Note: M and N are updated later in the timestep, so modifying them here is safe
    # With uniform_bc=True, the plan uses precomputed zero BCs, but we still
    # need to ensure input boundaries are zero for correctness
    # On GPU, batch all boundary assignments together to minimize kernel launches
    if use_gpu and _HAS_CUPY:
        # Batch all boundary assignments - CuPy will optimize these together
        # This reduces the number of separate kernel launches
        M[0, :] = M[-1, :] = M[:, 0] = M[:, -1] = 0.0
        N[0, :] = N[-1, :] = N[:, 0] = N[:, -1] = 0.0
    else:
        _enforce_dirichlet_zero(M, use_gpu=False)
        _enforce_dirichlet_zero(N, use_gpu=False)

    # Apply derivatives directly on M and N (boundaries already zero)
    # The bspf2d library handles array conversion and operations efficiently
    dMdx = plan_dx.apply(M)   # ∂M/∂x
    dNdy = plan_dy.apply(N)   # ∂N/∂y
    
    eta_new = eta - dt * (dMdx + dNdy)
    
    return eta_new


def update_M_2D_bspf(eta, M, N, D, g, h, alpha, dt, plan_dx, plan_dy, plan_dx_eta_neu, use_gpu: bool = False):
    """
    M_t = -(∂x(M^2/D) + ∂y(MN/D) + g D ∂x η) - friction
    Use Neumann(0) for ∂x η at boundaries; standard plans for the rest.
    GPU-aware.
    """
    xp = cp if (use_gpu and _HAS_CUPY) else np
    detadx  = plan_dx_eta_neu.apply(eta, flux=(0.0, 0.0))
    # For uniform_bc plans, we need to ensure boundaries are zero
    # Compute expressions and enforce BCs in-place
    arg1 = (M**2) / D
    arg2 = (M * N) / D
    if use_gpu and _HAS_CUPY:
        arg1[0, :] = arg1[-1, :] = arg1[:, 0] = arg1[:, -1] = 0.0
        arg2[0, :] = arg2[-1, :] = arg2[:, 0] = arg2[:, -1] = 0.0
    else:
        _enforce_dirichlet_zero(arg1, use_gpu=False)
        _enforce_dirichlet_zero(arg2, use_gpu=False)
    darg1dx = plan_dx.apply(arg1)
    darg2dy = plan_dy.apply(arg2)

    fric = g * alpha**2 * M * xp.sqrt(M**2 + N**2) / (D ** (7.0 / 3.0))
    M_new = M - dt * (darg1dx + darg2dy + g * D * detadx + fric)

    _enforce_dirichlet_zero(M_new)
    
    return M_new


def update_N_2D_bspf(eta, M, N, D, g, h, alpha, dt, plan_dx, plan_dy, plan_dy_eta_neu, use_gpu: bool = False):
    """
    N_t = -(∂x(MN/D) + ∂y(N^2/D) + g D ∂y η) - friction
    Use Neumann(0) for ∂y η at boundaries; standard plans for the rest.
    GPU-aware.
    """
    xp = cp if (use_gpu and _HAS_CUPY) else np
    detady  = plan_dy_eta_neu.apply(eta, flux=(0.0, 0.0))
    # For uniform_bc plans, we need to ensure boundaries are zero
    # Compute expressions and enforce BCs in-place
    arg1 = (M * N) / D
    arg2 = (N**2) / D
    if use_gpu and _HAS_CUPY:
        arg1[0, :] = arg1[-1, :] = arg1[:, 0] = arg1[:, -1] = 0.0
        arg2[0, :] = arg2[-1, :] = arg2[:, 0] = arg2[:, -1] = 0.0
    else:
        _enforce_dirichlet_zero(arg1, use_gpu=False)
        _enforce_dirichlet_zero(arg2, use_gpu=False)
    darg1dx = plan_dx.apply(arg1)
    darg2dy = plan_dy.apply(arg2)

    fric = g * alpha**2 * N * xp.sqrt(M**2 + N**2) / (D ** (7.0 / 3.0))
    N_new = N - dt * (darg1dx + darg2dy + g * D * detady + fric)

    _enforce_dirichlet_zero(N_new)
    
    return N_new


# ───────────────────────────── Finite Difference solver ──────────────────────────────

def Shallow_water_2D_fd(
    eta0, M0, N0, h, g, alpha, nt, dt, x, y,
    use_gpu=False,
):
    """
    Finite difference solver for 2D shallow water equations.
    Uses 2nd order central differences for spatial derivatives.
    Boundary conditions: Dirichlet(0) for M,N, Neumann(0) for eta.
    GPU-accelerated with CuPy when use_gpu=True.
    """
    # Check GPU availability
    if use_gpu and not _HAS_CUPY:
        print("Warning: use_gpu=True but CuPy is not available. Falling back to CPU.")
        print("Install CuPy with: pip install cupy-cuda12x (or cupy-cuda11x)")
        use_gpu = False
    
    xp = cp if (use_gpu and _HAS_CUPY) else np
    
    # Convert arrays to GPU if needed
    if use_gpu and _HAS_CUPY:
        eta = cp.asarray(eta0.copy())
        M = cp.asarray(M0.copy())
        N = cp.asarray(N0.copy())
        h = cp.asarray(h)
    else:
        eta = eta0.copy()
        M = M0.copy()
        N = N0.copy()
    
    D = eta + h
    
    # Grid parameters
    ny, nx = eta.shape
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # Precompute finite difference coefficients
    # 2nd order central difference: df/dx ≈ (f[i+1] - f[i-1]) / (2*dx)
    inv_2dx = 1.0 / (2.0 * dx)
    inv_2dy = 1.0 / (2.0 * dy)
    
    # Time stepping
    t_start = time.time()
    for n in range(nt + 1):
        # 1) Continuity: η_t = -(M_x + N_y)
        # Compute derivatives using central differences
        # M_x: ∂M/∂x using central difference
        dMdx = xp.zeros_like(M)
        dMdx[:, 1:-1] = (M[:, 2:] - M[:, :-2]) * inv_2dx
        # At boundaries, use one-sided differences (but M=0 at boundaries)
        dMdx[:, 0] = (M[:, 1] - M[:, 0]) / dx  # Forward difference at left
        dMdx[:, -1] = (M[:, -1] - M[:, -2]) / dx  # Backward difference at right
        
        # N_y: ∂N/∂y using central difference
        dNdy = xp.zeros_like(N)
        dNdy[1:-1, :] = (N[2:, :] - N[:-2, :]) * inv_2dy
        # At boundaries, use one-sided differences (but N=0 at boundaries)
        dNdy[0, :] = (N[1, :] - N[0, :]) / dy  # Forward difference at bottom
        dNdy[-1, :] = (N[-1, :] - N[-2, :]) / dy  # Backward difference at top
        
        eta = eta - dt * (dMdx + dNdy)
        
        # 2) Momentum x: M_t = -(∂x(M^2/D) + ∂y(MN/D) + g D ∂x η) - friction
        D = eta + h
        
        # Compute nonlinear terms
        arg1 = (M**2) / D
        arg2 = (M * N) / D
        
        # Enforce BCs: arg1 and arg2 = 0 at boundaries
        arg1[0, :] = arg1[-1, :] = arg1[:, 0] = arg1[:, -1] = 0.0
        arg2[0, :] = arg2[-1, :] = arg2[:, 0] = arg2[:, -1] = 0.0
        
        # Derivatives
        darg1dx = xp.zeros_like(arg1)
        darg1dx[:, 1:-1] = (arg1[:, 2:] - arg1[:, :-2]) * inv_2dx
        darg1dx[:, 0] = (arg1[:, 1] - arg1[:, 0]) / dx
        darg1dx[:, -1] = (arg1[:, -1] - arg1[:, -2]) / dx
        
        darg2dy = xp.zeros_like(arg2)
        darg2dy[1:-1, :] = (arg2[2:, :] - arg2[:-2, :]) * inv_2dy
        darg2dy[0, :] = (arg2[1, :] - arg2[0, :]) / dy
        darg2dy[-1, :] = (arg2[-1, :] - arg2[-2, :]) / dy
        
        # ∂η/∂x with Neumann BC (copy interior to boundary)
        eta_bc = eta.copy()
        eta_bc[0, :] = eta_bc[1, :]
        eta_bc[-1, :] = eta_bc[-2, :]
        eta_bc[:, 0] = eta_bc[:, 1]
        eta_bc[:, -1] = eta_bc[:, -2]
        
        detadx = xp.zeros_like(eta)
        detadx[:, 1:-1] = (eta_bc[:, 2:] - eta_bc[:, :-2]) * inv_2dx
        detadx[:, 0] = (eta_bc[:, 1] - eta_bc[:, 0]) / dx
        detadx[:, -1] = (eta_bc[:, -1] - eta_bc[:, -2]) / dx
        
        fric = g * alpha**2 * M * xp.sqrt(M**2 + N**2) / (D ** (7.0 / 3.0))
        M = M - dt * (darg1dx + darg2dy + g * D * detadx + fric)
        
        # Enforce BCs on M
        M[0, :] = M[-1, :] = M[:, 0] = M[:, -1] = 0.0
        
        # 3) Momentum y: N_t = -(∂x(MN/D) + ∂y(N^2/D) + g D ∂y η) - friction
        arg1 = (M * N) / D
        arg2 = (N**2) / D
        
        # Enforce BCs
        arg1[0, :] = arg1[-1, :] = arg1[:, 0] = arg1[:, -1] = 0.0
        arg2[0, :] = arg2[-1, :] = arg2[:, 0] = arg2[:, -1] = 0.0
        
        # Derivatives
        darg1dx = xp.zeros_like(arg1)
        darg1dx[:, 1:-1] = (arg1[:, 2:] - arg1[:, :-2]) * inv_2dx
        darg1dx[:, 0] = (arg1[:, 1] - arg1[:, 0]) / dx
        darg1dx[:, -1] = (arg1[:, -1] - arg1[:, -2]) / dx
        
        darg2dy = xp.zeros_like(arg2)
        darg2dy[1:-1, :] = (arg2[2:, :] - arg2[:-2, :]) * inv_2dy
        darg2dy[0, :] = (arg2[1, :] - arg2[0, :]) / dy
        darg2dy[-1, :] = (arg2[-1, :] - arg2[-2, :]) / dy
        
        detady = xp.zeros_like(eta)
        detady[1:-1, :] = (eta_bc[2:, :] - eta_bc[:-2, :]) * inv_2dy
        detady[0, :] = (eta_bc[1, :] - eta_bc[0, :]) / dy
        detady[-1, :] = (eta_bc[-1, :] - eta_bc[-2, :]) / dy
        
        fric = g * alpha**2 * N * xp.sqrt(M**2 + N**2) / (D ** (7.0 / 3.0))
        N = N - dt * (darg1dx + darg2dy + g * D * detady + fric)
        
        # Enforce BCs on N
        N[0, :] = N[-1, :] = N[:, 0] = N[:, -1] = 0.0
        
        # 4) Update column height
        D = eta + h
        
        if (n % 100) == 0:
            print(f"Time step {n} of {nt}")
    
    # Synchronize GPU if used
    if use_gpu and _HAS_CUPY:
        cp.cuda.Stream.null.synchronize()
    
    # Time measurement
    t_end = time.time()
    t_total = t_end - t_start
    t_per_step = t_total / (nt + 1)
    
    # Print timing report
    print(f"\n{'='*70}")
    print(f"Performance Report - Finite Difference ({'GPU' if (use_gpu and _HAS_CUPY) else 'CPU'})")
    print(f"{'='*70}")
    print(f"  Total time:           {t_total:10.4f} seconds")
    print(f"  Time per step:        {t_per_step:10.6f} seconds")
    print(f"  Steps per second:     {(nt + 1) / t_total:10.2f}")
    print(f"  Number of steps:      {nt + 1}")
    
    if use_gpu and _HAS_CUPY:
        print(f"  GPU acceleration:     Enabled (CuPy)")
    elif use_gpu:
        print(f"  GPU acceleration:     Requested but not available (falling back to CPU)")
    
    print(f"{'='*70}")
    
    # Convert results back to CPU if needed
    if use_gpu and _HAS_CUPY:
        eta = cp.asnumpy(eta)
        M = cp.asnumpy(M)
        N = cp.asnumpy(N)
    
    return eta, M, N


# ───────────────────────────── main driver ──────────────────────────────

def Shallow_water_2D_bspf(
    eta0, M0, N0, h, g, alpha, nt, dt, x, y, degree=10,
    use_filter=True, filter_alpha=36.0, filter_order=8, filter_stride=1,
    filter_eta=True,  # toggle filtering of η (you can turn this off)
    use_gpu=False,    # enable GPU acceleration with CuPy
):
    """
    Uses bspf2d spatial derivatives with precomputed plans and applies
    BC-compatible DCT/DST exponential filters for dealiasing.
    GPU-accelerated with CuPy when use_gpu=True.
    """
    # Check GPU availability
    if use_gpu and not _HAS_CUPY:
        print("Warning: use_gpu=True but CuPy is not available. Falling back to CPU.")
        print("Install CuPy with: pip install cupy-cuda12x (or cupy-cuda11x)")
        use_gpu = False
    
    xp = cp if (use_gpu and _HAS_CUPY) else np
    
    # Convert arrays to GPU if needed
    if use_gpu and _HAS_CUPY:
        eta = cp.asarray(eta0.copy())
        M = cp.asarray(M0.copy())
        N = cp.asarray(N0.copy())
        h = cp.asarray(h)
    else:
        eta = eta0.copy()
        M = M0.copy()
        N = N0.copy()
    
    D = eta + h

    # Build BSLF2D operator once
    op = bspf2d.from_grids(x=x, 
                            y=y, 
                            degree_x=degree, 
                            degree_y=degree, 
                            order_x= degree, 
                            order_y=degree, 
                            num_boundary_points_x=degree+1,
                            num_boundary_points_y=degree+1,
                            use_clustering_x=True,
                            use_clustering_y=True,
                            correction="spectral",
                            use_gpu=use_gpu,
                            )

    # Precompute derivative plans (fast path)
    # Use uniform_bc=True for Dirichlet(0) BCs to avoid expensive matrix-vector products
    plan_dx = op.make_plan_dx(order=1, lam=0.0, neumann=False, uniform_bc=True, bc=0.0)
    plan_dy = op.make_plan_dy(order=1, lam=0.0, neumann=False, uniform_bc=True, bc=0.0)
    plan_dx_eta_neu = op.make_plan_dx(order=1, lam=0.0, neumann=True)
    plan_dy_eta_neu = op.make_plan_dy(order=1, lam=0.0, neumann=True)
    
    # Precompute filter masks (sigma) - these are constant for given grid size, alpha, and order
    ny, nx = eta.shape
    if use_gpu and _HAS_CUPY:
        # Precompute DST filter mask
        ky_dst = cp.arange(1, ny + 1)    # DST-II modes 1..N
        kx_dst = cp.arange(1, nx + 1)
        ky_max_dst = max(float(ky_dst[-1]), 1)
        kx_max_dst = max(float(kx_dst[-1]), 1)
        KY_dst = ky_dst[:, None] / ky_max_dst
        KX_dst = kx_dst[None, :] / kx_max_dst
        rr_dst = cp.sqrt(KY_dst**2 + KX_dst**2)
        sigma_dst = cp.exp(-filter_alpha * (rr_dst**filter_order))
        
        # Precompute DCT filter mask (if needed)
        if filter_eta:
            ky_dct = cp.arange(ny)           # DCT-II mode index 0..N-1
            kx_dct = cp.arange(nx)
            ky_max_dct = max(float(ky_dct[-1]), 1)
            kx_max_dct = max(float(kx_dct[-1]), 1)
            KY_dct = ky_dct[:, None] / ky_max_dct
            KX_dct = kx_dct[None, :] / kx_max_dct
            rr_dct = cp.sqrt(KY_dct**2 + KX_dct**2)
            sigma_dct = cp.exp(-filter_alpha * (rr_dct**filter_order))
        else:
            sigma_dct = None
    else:
        # Precompute DST filter mask
        ky_dst = np.arange(1, ny + 1)    # DST-II modes 1..N
        kx_dst = np.arange(1, nx + 1)
        ky_max_dst = max(float(ky_dst[-1]), 1)
        kx_max_dst = max(float(kx_dst[-1]), 1)
        KY_dst = ky_dst[:, None] / ky_max_dst
        KX_dst = kx_dst[None, :] / kx_max_dst
        rr_dst = np.sqrt(KY_dst**2 + KX_dst**2)
        sigma_dst = np.exp(-filter_alpha * (rr_dst**filter_order))
        
        # Precompute DCT filter mask (if needed)
        if filter_eta:
            ky_dct = np.arange(ny)           # DCT-II mode index 0..N-1
            kx_dct = np.arange(nx)
            ky_max_dct = max(float(ky_dct[-1]), 1)
            kx_max_dct = max(float(kx_dct[-1]), 1)
            KY_dct = ky_dct[:, None] / ky_max_dct
            KX_dct = kx_dct[None, :] / kx_max_dct
            rr_dct = np.sqrt(KY_dct**2 + KX_dct**2)
            sigma_dct = np.exp(-filter_alpha * (rr_dct**filter_order))
        else:
            sigma_dct = None
    
    # Time measurement
    t_start = time.time()
    for n in range(nt+1):
        # 1) continuity
        eta = update_eta_2D_bspf(eta, M, N, dt, plan_dx, plan_dy, use_gpu=use_gpu)

        # 2) momentum x
        M = update_M_2D_bspf(eta, M, N, D, g, h, alpha, dt, plan_dx, plan_dy, plan_dx_eta_neu, use_gpu=use_gpu)

        # 3) momentum y
        N = update_N_2D_bspf(eta, M, N, D, g, h, alpha, dt, plan_dx, plan_dy, plan_dy_eta_neu, use_gpu=use_gpu)

        # 4) update column height
        D = eta + h

        # 5) BC-compatible dealiasing filters (every filter_stride steps)
        if use_filter and (n % filter_stride == 0):
            # Dirichlet(0) fluxes → DST filter (using precomputed sigma)
            M = dst_filter_dirichlet(M, alpha=filter_alpha, order=filter_order, use_gpu=use_gpu, sigma_cache=sigma_dst)
            N = dst_filter_dirichlet(N, alpha=filter_alpha, order=filter_order, use_gpu=use_gpu, sigma_cache=sigma_dst)
            
            # Enforce BCs - optimized for GPU
            # Note: DST-II naturally enforces zero boundaries, but we enforce them
            # explicitly to ensure numerical stability after filtering operations
            if use_gpu and _HAS_CUPY:
                # On GPU, use direct in-place assignment which CuPy optimizes
                # This is faster than function calls and reduces overhead
                # CuPy batches these operations efficiently
                M[0, :] = 0.0
                M[-1, :] = 0.0
                M[:, 0] = 0.0
                M[:, -1] = 0.0
                N[0, :] = 0.0
                N[-1, :] = 0.0
                N[:, 0] = 0.0
                N[:, -1] = 0.0
            else:
                # On CPU, use the helper function for consistency
                _enforce_dirichlet_zero(M, use_gpu=False)
                _enforce_dirichlet_zero(N, use_gpu=False)

            # Neumann(0) surface → DCT filter (optional, using precomputed sigma)
            if filter_eta:
                eta = dct_filter_neumann(eta, alpha=filter_alpha, order=filter_order, use_gpu=use_gpu, sigma_cache=sigma_dct)
                _enforce_neumann_zero_eta_in_field(eta, use_gpu=use_gpu)

        if (n % 100) == 0:
            print(f"Time step {n} of {nt}")
    
    # Synchronize GPU if used
    if use_gpu and _HAS_CUPY:
        cp.cuda.Stream.null.synchronize()
    
    # Time measurement
    t_end = time.time()
    t_total = t_end - t_start
    t_per_step = t_total / (nt + 1)
    
    # Print timing report
    print(f"\n{'='*70}")
    print(f"Performance Report ({'GPU' if (use_gpu and _HAS_CUPY) else 'CPU'})")
    print(f"{'='*70}")
    print(f"  Total time:           {t_total:10.4f} seconds")
    print(f"  Time per step:        {t_per_step:10.6f} seconds")
    print(f"  Steps per second:     {(nt + 1) / t_total:10.2f}")
    print(f"  Number of steps:      {nt + 1}")
    
    if use_gpu and _HAS_CUPY:
        print(f"  GPU acceleration:     Enabled (CuPy)")
        if _HAS_CUPY_FFT:
            print(f"  GPU FFT support:      Enabled (cupyx.scipy.fft)")
        else:
            print(f"  GPU FFT support:      Disabled (fallback to CPU)")
    
    print(f"{'='*70}")

    # Convert results back to CPU if needed
    if use_gpu and _HAS_CUPY:
        eta = cp.asnumpy(eta)
        M = cp.asnumpy(M)
        N = cp.asnumpy(N)

    return eta, M, N


if __name__ == "__main__":
    # Domain
    Lx = 100.0
    Ly = 100.0
    nx = 512
    ny = 512

    x = np.linspace(0.0, Lx, num=nx)
    y = np.linspace(0.0, Ly, num=ny)
    X, Y = np.meshgrid(x, y)

    # Depth (example: mild shelf)
    h = 10 - 7.5 * np.tanh((X - 75.) / 10.)

    # Initial surface (strong Gaussian pulse)
    eta0 = 1 * np.exp(-((X - 50)**2 / 10) - ((Y - 50)**2 / 10))

    # Initial fluxes
    M0 = 100.0 * eta0
    N0 = 0.0 * M0

    g = 9.81
    alpha = 0.025

    Tmax = 5
    # Estimate dt from CFL condition
    # c = np.sqrt(g * np.max(h))
    # dx = x[1] - x[0]
    # dy = y[1] - y[0]
    # dt = 0.2 * np.min(dx / c)
    dt = 1 / 1000
    # print(f"dt = {dt}")
    nt = int(Tmax / dt)

    # # Run both CPU and GPU versions for comparison
    # print("\n" + "="*70)
    # print("Running CPU version...")
    # print("="*70)
    # t_cpu_start = time.time()
    # eta_cpu, M_cpu, N_cpu = Shallow_water_2D_bspf(
    #     eta0, M0, N0, h, g, alpha, nt, dt, x, y, degree=9,
    #     use_filter=True, filter_alpha=36.0, filter_order=36, filter_stride=1,
    #     filter_eta=True,
    #     use_gpu=False,
    # )
    # t_cpu_total = time.time() - t_cpu_start

    print("\n" + "="*70)
    print("Running BSLF GPU version...")
    print("="*70)
    t_bslf_start = time.time()
    eta_bslf, M_bslf, N_bslf = Shallow_water_2D_bspf(
        eta0, M0, N0, h, g, alpha, nt, dt, x, y, degree=9,
        use_filter=True, filter_alpha=36.0, filter_order=36, filter_stride=1,
        filter_eta=True,
        use_gpu=True,
    )
    t_bslf_total = time.time() - t_bslf_start
    
    print("\n" + "="*70)
    print("Running Finite Difference GPU version...")
    print("="*70)
    t_fd_start = time.time()
    eta_fd, M_fd, N_fd = Shallow_water_2D_fd(
        eta0, M0, N0, h, g, alpha, nt, dt, x, y,
        use_gpu=True,
    )
    t_fd_total = time.time() - t_fd_start

    # Comparison report
    print("\n" + "="*70)
    print("BSLF vs Finite Difference Comparison")
    print("="*70)
    print(f"{'Metric':<30} {'BSLF':>15} {'FD':>15} {'Speedup':>15}")
    print("-"*70)
    print(f"{'Total time (s)':<30} {t_bslf_total:15.4f} {t_fd_total:15.4f} {t_bslf_total/t_fd_total:15.2f}x")
    print(f"{'Time per step (s)':<30} {t_bslf_total/(nt+1):15.6f} {t_fd_total/(nt+1):15.6f} {t_bslf_total/t_fd_total:15.2f}x")
    print(f"{'Steps per second':<30} {(nt+1)/t_bslf_total:15.2f} {(nt+1)/t_fd_total:15.2f} {t_fd_total/t_bslf_total:15.2f}x")
    print("="*70)
    
    # Plot the final surface
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2,3,1)
    plt.imshow(eta_bslf, cmap='bwr', origin='lower', vmax=1.0, vmin=-1.0)
    plt.colorbar()
    plt.title("BSLF - Final surface")
    
    plt.subplot(2,3,2)
    plt.imshow(eta_fd, cmap='bwr', origin='lower', vmax=1.0, vmin=-1.0)
    plt.colorbar()
    plt.title("Finite Difference - Final surface")
    
    plt.subplot(2,3,3)
    diff = np.abs(eta_bslf - eta_fd)
    plt.imshow(diff, cmap='hot', origin='lower')
    plt.colorbar()
    plt.title("Absolute difference")
    
    plt.subplot(2,3,4)
    plt.plot(x, eta_bslf[ny//2, :], label='BSLF', linewidth=2)
    plt.plot(x, eta_fd[ny//2, :], '--', label='FD', linewidth=2)
    plt.xlabel("x")
    plt.ylabel("eta")
    plt.title(f"Cross-section at y={y[ny//2]:.1f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2,3,5)
    plt.plot(x, M_bslf[ny//2, :], label='BSLF', linewidth=2)
    plt.plot(x, M_fd[ny//2, :], '--', label='FD', linewidth=2)
    plt.xlabel("x")
    plt.ylabel("M")
    plt.title(f"Momentum M at y={y[ny//2]:.1f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2,3,6)
    plt.plot(x, diff[ny//2, :], linewidth=2)
    plt.xlabel("x")
    plt.ylabel("|eta_BSLF - eta_FD|")
    plt.title("Difference along cross-section")
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

    # # Comparison report
    # print("\n" + "="*70)
    # print("CPU vs GPU Comparison")
    # print("="*70)
    # print(f"{'Metric':<30} {'CPU':>15} {'GPU':>15} {'Speedup':>15}")
    # print("-"*70)
    # print(f"{'Total time (s)':<30} {t_cpu_total:15.4f} {t_gpu_total:15.4f} {t_cpu_total/t_gpu_total:15.2f}x")
    # print(f"{'Time per step (s)':<30} {t_cpu_total/(nt+1):15.6f} {t_gpu_total/(nt+1):15.6f} {t_cpu_total/t_gpu_total:15.2f}x")
    # print(f"{'Steps per second':<30} {(nt+1)/t_cpu_total:15.2f} {(nt+1)/t_gpu_total:15.2f} {t_gpu_total/t_cpu_total:15.2f}x")
    # print("="*70)
    
    # # Verify results match
    # max_diff_eta = np.max(np.abs(eta_cpu - eta_gpu))
    # max_diff_M = np.max(np.abs(M_cpu - M_gpu))
    # max_diff_N = np.max(np.abs(N_cpu - N_gpu))
    # print(f"\nResult Verification:")
    # print(f"  Max difference in eta: {max_diff_eta:.2e}")
    # print(f"  Max difference in M:   {max_diff_M:.2e}")
    # print(f"  Max difference in N:   {max_diff_N:.2e}")
    # if max_diff_eta < 1e-10 and max_diff_M < 1e-10 and max_diff_N < 1e-10:
    #     print("  ✓ Results match perfectly!")
    # elif max_diff_eta < 1e-6 and max_diff_M < 1e-6 and max_diff_N < 1e-6:
    #     print("  ⚠ Results differ slightly (likely numerical precision)")
    # else:
    #     print("  ✗ Results differ significantly!")
    
    # print("\n" + "="*70)
