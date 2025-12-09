"""
2D Synthetic Turbulence Test Case using BSPF2D.

This script generates 2D synthetic turbulence and tests the BSPF2D implementation
for computing first and second derivatives with respect to x and y.

The turbulence field is generated using Fourier modes with a power-law energy
spectrum (e.g., k^(-5/3) for Kolmogorov turbulence).

Run from repository root:
    python examples/basic/turbulence_2d.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from bspf import bspf2d
from bspf.utils import construct_chebyshev_nodes, chebyshev_derivative_from_values


# ============================================================================
# Parameters
# ============================================================================

# Grid parameters
NX = 512  # Grid points along x-axis
NY = 512  # Grid points along y-axis
DOMAIN_X = [0, 2*np.pi]  # Domain [a, b] for x
DOMAIN_Y = [0, 2*np.pi]  # Domain [c, d] for y

# BSPF parameters
DEGREE = 7                  # B-spline polynomial degree
NUM_BOUNDARY_POINTS = DEGREE + 3 # Number of boundary points
N_BASIS = 4 * DEGREE         # Number of basis functions
REG_PARAM = 1e-3             # Tikhonov regularization strength (lambda)
CLUSTERING_FACTOR = 2.0      # Clustering factor for B-spline basis functions
USE_CLUSTERING = True        # Enable grid clustering

# Turbulence parameters
TURB_N_MODES = 64           # Number of Fourier modes
TURB_POWER_LAW = -5/3        # Kolmogorov spectrum exponent
TURB_SEED = 123              # Random seed for reproducibility

# Shock wave parameters
SHOCK_ENABLED = True         # Enable circular shock wave
SHOCK_CENTER_X = np.pi       # X-coordinate of shock center
SHOCK_CENTER_Y = np.pi       # Y-coordinate of shock center
SHOCK_RADIUS = 0.5           # Radius of the shock
SHOCK_AMPLITUDE = 2.0        # Amplitude of the shock jump
SHOCK_WIDTH = 0.02           # Width of the shock transition (for smooth shock)


# ============================================================================
# 2D Synthetic Turbulence Generator
# ============================================================================

def random_turbulence_signal_2d_with_derivatives(
    Lx=1.0, 
    Ly=1.0,
    Nx=128, 
    Ny=128,
    nmodes=50, 
    seed=None
):
    """
    Generate a 2D turbulence-like signal using cosine modes with
    a Kolmogorov-like spectrum, and return analytical first
    and second derivatives (Neumann BC automatically satisfied).
    
    Returns data in (ny, nx) format.
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)  # (ny, nx)

    # Mode indices n, m = 1 ... nmodes
    n = np.arange(1, nmodes + 1)
    m = np.arange(1, nmodes + 1)
    
    # Wavenumbers
    kx = n * np.pi / Lx
    ky = m * np.pi / Ly

    # Kolmogorov-like scaling: E(k) ~ k^(-5/3) → amplitude ~ k^(-5/6)
    # Use combined wavenumber magnitude
    k_mag = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
    sigma = k_mag**(-5/6)
    
    # Random amplitudes (flatten and sample)
    n_total = nmodes * nmodes
    indices = np.random.choice(n_total, size=min(nmodes, n_total), replace=False)
    i_flat, j_flat = np.unravel_index(indices, (nmodes, nmodes))
    
    # Allocate arrays (ny, nx)
    u = np.zeros((Ny, Nx))
    u_x = np.zeros((Ny, Nx))
    u_y = np.zeros((Ny, Nx))
    u_xx = np.zeros((Ny, Nx))
    u_yy = np.zeros((Ny, Nx))
    u_xy = np.zeros((Ny, Nx))

    # Build signal + exact derivatives
    for idx in range(len(i_flat)):
        ni = n[i_flat[idx]]
        mi = m[j_flat[idx]]
        
        kx_val = ni * np.pi / Lx
        ky_val = mi * np.pi / Ly
        ai = sigma[i_flat[idx], j_flat[idx]] * np.random.randn()
        
        cos_x = np.cos(kx_val * X)
        cos_y = np.cos(ky_val * Y)
        sin_x = np.sin(kx_val * X)
        sin_y = np.sin(ky_val * Y)
        
        # u = a * cos(kx*x) * cos(ky*y)
        u += ai * cos_x * cos_y
        
        # u_x = -a * kx * sin(kx*x) * cos(ky*y)
        u_x += ai * (-kx_val) * sin_x * cos_y
        
        # u_y = -a * ky * cos(kx*x) * sin(ky*y)
        u_y += ai * (-ky_val) * cos_x * sin_y
        
        # u_xx = -a * kx^2 * cos(kx*x) * cos(ky*y)
        u_xx += ai * (-kx_val**2) * cos_x * cos_y
        
        # u_yy = -a * ky^2 * cos(kx*x) * cos(ky*y)
        u_yy += ai * (-ky_val**2) * cos_x * cos_y
        
        # u_xy = a * kx * ky * sin(kx*x) * sin(ky*y)
        u_xy += ai * kx_val * ky_val * sin_x * sin_y

    # Optional: remove mean of u (common in turbulence signals)
    u -= np.mean(u)

    return x, y, u, u_x, u_y, u_xx, u_yy, u_xy


# ============================================================================
# Finite Difference Functions for 2D Data
# ============================================================================

def finite_difference_2nd_order_2d_x(x, y, F):
    """
    Compute first derivative with respect to x using 2nd-order finite differences.
    
    Parameters
    ----------
    x : array_like
        Grid points along x (must be uniformly spaced)
    y : array_like
        Grid points along y (must be uniformly spaced)
    F : array_like
        2D field values, shape (ny, nx)
    
    Returns
    -------
    dF_dx : ndarray
        First derivative with respect to x, shape (ny, nx)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    F = np.asarray(F)
    
    if F.ndim != 2:
        raise ValueError("F must be 2D")
    
    ny, nx = F.shape
    
    if len(x) != nx or len(y) != ny:
        raise ValueError(f"Shape mismatch: F is ({ny}, {nx}), but x has {len(x)} points, y has {len(y)} points")
    
    # Check if grid is uniform
    dx = x[1] - x[0]
    if not np.allclose(np.diff(x), dx, rtol=1e-10):
        raise ValueError("Grid must be uniformly spaced for finite differences")
    
    dF_dx = np.zeros_like(F)
    
    # Interior points: central difference (2nd order)
    dF_dx[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / (2 * dx)
    
    # Left boundary: 2nd-order forward difference
    dF_dx[:, 0] = (-3 * F[:, 0] + 4 * F[:, 1] - F[:, 2]) / (2 * dx)
    
    # Right boundary: 2nd-order backward difference
    dF_dx[:, -1] = (3 * F[:, -1] - 4 * F[:, -2] + F[:, -3]) / (2 * dx)
    
    return dF_dx


def finite_difference_2nd_order_2d_y(x, y, F):
    """
    Compute first derivative with respect to y using 2nd-order finite differences.
    
    Parameters
    ----------
    x : array_like
        Grid points along x (must be uniformly spaced)
    y : array_like
        Grid points along y (must be uniformly spaced)
    F : array_like
        2D field values, shape (ny, nx)
    
    Returns
    -------
    dF_dy : ndarray
        First derivative with respect to y, shape (ny, nx)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    F = np.asarray(F)
    
    if F.ndim != 2:
        raise ValueError("F must be 2D")
    
    ny, nx = F.shape
    
    if len(x) != nx or len(y) != ny:
        raise ValueError(f"Shape mismatch: F is ({ny}, {nx}), but x has {len(x)} points, y has {len(y)} points")
    
    # Check if grid is uniform
    dy = y[1] - y[0]
    if not np.allclose(np.diff(y), dy, rtol=1e-10):
        raise ValueError("Grid must be uniformly spaced for finite differences")
    
    dF_dy = np.zeros_like(F)
    
    # Interior points: central difference (2nd order)
    dF_dy[1:-1, :] = (F[2:, :] - F[:-2, :]) / (2 * dy)
    
    # Bottom boundary: 2nd-order forward difference
    dF_dy[0, :] = (-3 * F[0, :] + 4 * F[1, :] - F[2, :]) / (2 * dy)
    
    # Top boundary: 2nd-order backward difference
    dF_dy[-1, :] = (3 * F[-1, :] - 4 * F[-2, :] + F[-3, :]) / (2 * dy)
    
    return dF_dy


def finite_difference_2nd_order_2d_xx(x, y, F):
    """
    Compute second derivative with respect to x using 2nd-order finite differences.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    F = np.asarray(F)
    
    ny, nx = F.shape
    dx = x[1] - x[0]
    
    d2F_dx2 = np.zeros_like(F)
    
    # Interior points: central difference (2nd order)
    d2F_dx2[:, 1:-1] = (F[:, 2:] - 2 * F[:, 1:-1] + F[:, :-2]) / (dx**2)
    
    # Boundaries
    d2F_dx2[:, 0] = (F[:, 0] - 2 * F[:, 1] + F[:, 2]) / (dx**2)
    d2F_dx2[:, -1] = (F[:, -1] - 2 * F[:, -2] + F[:, -3]) / (dx**2)
    
    return d2F_dx2


def finite_difference_2nd_order_2d_yy(x, y, F):
    """
    Compute second derivative with respect to y using 2nd-order finite differences.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    F = np.asarray(F)
    
    ny, nx = F.shape
    dy = y[1] - y[0]
    
    d2F_dy2 = np.zeros_like(F)
    
    # Interior points: central difference (2nd order)
    d2F_dy2[1:-1, :] = (F[2:, :] - 2 * F[1:-1, :] + F[:-2, :]) / (dy**2)
    
    # Boundaries
    d2F_dy2[0, :] = (F[0, :] - 2 * F[1, :] + F[2, :]) / (dy**2)
    d2F_dy2[-1, :] = (F[-1, :] - 2 * F[-2, :] + F[-3, :]) / (dy**2)
    
    return d2F_dy2


# ============================================================================
# Chebyshev 2D Derivative Functions
# ============================================================================

# Cache for turbulence field parameters (modes and amplitudes)
_turbulence_params_cache = {}

def _get_turbulence_parameters(Lx, Ly, n_modes, seed):
    """
    Generate and cache turbulence field parameters (modes and amplitudes).
    
    This function generates fixed random amplitudes based on seed, ensuring
    the same field is generated regardless of grid size.
    
    Parameters
    ----------
    Lx, Ly : float
        Domain lengths
    n_modes : int
        Number of Fourier modes
    seed : int or None
        Random seed for reproducibility
    
    Returns
    -------
    modes : list of tuples
        List of (ni, mi, kx_val, ky_val, ai) tuples for each mode
    """
    cache_key = (Lx, Ly, n_modes, seed)
    if cache_key in _turbulence_params_cache:
        return _turbulence_params_cache[cache_key]
    
    if seed is not None:
        np.random.seed(seed)
    
    # Mode indices n, m = 1 ... nmodes
    n = np.arange(1, n_modes + 1)
    m = np.arange(1, n_modes + 1)
    
    # Wavenumbers
    kx = n * np.pi / Lx
    ky = m * np.pi / Ly
    
    # Kolmogorov-like scaling: E(k) ~ k^(-5/3) → amplitude ~ k^(-5/6)
    k_mag = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
    sigma = k_mag**(-5/6)
    
    # Random amplitudes (flatten and sample) - FIXED regardless of grid size
    n_total = n_modes * n_modes
    indices = np.random.choice(n_total, size=min(n_modes, n_total), replace=False)
    i_flat, j_flat = np.unravel_index(indices, (n_modes, n_modes))
    
    # Store mode parameters
    modes = []
    for idx in range(len(i_flat)):
        ni = n[i_flat[idx]]
        mi = m[j_flat[idx]]
        
        kx_val = ni * np.pi / Lx
        ky_val = mi * np.pi / Ly
        ai = sigma[i_flat[idx], j_flat[idx]] * np.random.randn()
        
        modes.append((ni, mi, kx_val, ky_val, ai))
    
    _turbulence_params_cache[cache_key] = modes
    return modes


def evaluate_turbulence_field_on_grid(x, y, Lx, Ly, n_modes, seed):
    """
    Evaluate turbulence field on arbitrary grid points using direct numerical computation.
    
    This ensures the same field is generated regardless of grid size by using
    fixed random amplitudes (same approach as notebook).
    
    Parameters
    ----------
    x, y : array_like
        Grid points along x and y axes
    Lx, Ly : float
        Domain lengths
    n_modes : int
        Number of Fourier modes
    seed : int or None
        Random seed for reproducibility
    
    Returns
    -------
    u : ndarray
        Field values, shape (ny, nx)
    u_x : ndarray
        First derivative with respect to x, shape (ny, nx)
    """
    # Get fixed turbulence parameters
    modes = _get_turbulence_parameters(Lx, Ly, n_modes, seed)
    
    X, Y = np.meshgrid(x, y)  # (ny, nx)
    
    # Allocate arrays
    u = np.zeros_like(X)
    u_x = np.zeros_like(X)
    
    # Build signal + exact derivatives (same as notebook approach)
    for ni, mi, kx_val, ky_val, ai in modes:
        cos_x = np.cos(kx_val * X)
        cos_y = np.cos(ky_val * Y)
        sin_x = np.sin(kx_val * X)
        sin_y = np.sin(ky_val * Y)
        
        # u = a * cos(kx*x) * cos(ky*y)
        u += ai * cos_x * cos_y
        
        # u_x = -a * kx * sin(kx*x) * cos(ky*y)
        u_x += ai * (-kx_val) * sin_x * cos_y
    
    # Remove mean (common in turbulence signals)
    u -= np.mean(u)
    
    return u, u_x


def chebyshev_derivative_2d_x(x, y, domain_x, domain_y, n_modes, seed, shock_enabled=False, shock_params=None):
    """
    Compute first derivative with respect to x using Chebyshev spectral method.
    
    This function:
    1. Constructs Chebyshev nodes in 2D (tensor product)
    2. Evaluates turbulence field directly on Chebyshev nodes using symbolic expression
    3. Computes derivative along x using 1D Chebyshev method
    4. Interpolates derivative back to uniform grid
    
    Parameters
    ----------
    x : array_like
        Uniform grid points along x (for output interpolation)
    y : array_like
        Uniform grid points along y (for output interpolation)
    domain_x : (float, float)
        Domain [a, b] for x
    domain_y : (float, float)
        Domain [c, d] for y
    n_modes : int
        Number of Fourier modes for turbulence generation
    seed : int or None
        Random seed for turbulence generation
    shock_enabled : bool
        Whether to add circular shock wave
    shock_params : dict or None
        Parameters for shock wave (center_x, center_y, radius, amplitude, width)
    
    Returns
    -------
    dF_dx : ndarray
        First derivative with respect to x on uniform grid, shape (ny, nx)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    ny, nx = len(y), len(x)
    
    # Number of Chebyshev nodes (use same number as uniform grid for fair comparison)
    Nx_cheb = nx - 1  # Number of intervals for Chebyshev (N+1 nodes)
    Ny_cheb = ny - 1
    
    # Construct Chebyshev nodes
    x_cheb, _ = construct_chebyshev_nodes(Nx_cheb, domain=tuple(domain_x))
    y_cheb, _ = construct_chebyshev_nodes(Ny_cheb, domain=tuple(domain_y))
    
    # Evaluate turbulence field directly on Chebyshev nodes using direct numerical computation
    Lx = domain_x[1] - domain_x[0]
    Ly = domain_y[1] - domain_y[0]
    X_cheb_grid, Y_cheb_grid = np.meshgrid(x_cheb, y_cheb)
    F_cheb, _ = evaluate_turbulence_field_on_grid(x_cheb, y_cheb, Lx, Ly, n_modes, seed)
    
    # Remove mean
    F_cheb -= np.mean(F_cheb)
    
    # Add shock wave if enabled
    if shock_enabled and shock_params is not None:
        F_cheb, _, _, _, _, _ = add_circular_shock_wave(
            X_cheb_grid, Y_cheb_grid, F_cheb, 
            np.zeros_like(F_cheb), np.zeros_like(F_cheb),
            np.zeros_like(F_cheb), np.zeros_like(F_cheb), np.zeros_like(F_cheb),
            center_x=shock_params['center_x'],
            center_y=shock_params['center_y'],
            radius=shock_params['radius'],
            amplitude=shock_params['amplitude'],
            width=shock_params['width']
        )
    
    # Compute derivative along x for each y-slice using Chebyshev
    dF_dx_cheb = np.zeros_like(F_cheb)
    for j in range(F_cheb.shape[0]):
        dF_dx_cheb[j, :] = chebyshev_derivative_from_values(
            F_cheb[j, :], x_cheb, domain=tuple(domain_x)
        )
    
    # Interpolate derivative back to uniform grid
    X_unif, Y_unif = np.meshgrid(x, y)  # (ny, nx)
    interp_deriv = RegularGridInterpolator((y_cheb, x_cheb), dF_dx_cheb, method='linear')
    points_unif = np.column_stack([Y_unif.ravel(), X_unif.ravel()])
    dF_dx_flat = interp_deriv(points_unif)
    dF_dx = dF_dx_flat.reshape(Y_unif.shape)  # (ny, nx)
    
    return dF_dx


# ============================================================================
# Circular Shock Wave Function
# ============================================================================

def add_circular_shock_wave(X, Y, u, u_x, u_y, u_xx, u_yy, u_xy,
                           center_x, center_y, radius, amplitude, width):
    """
    Add a circular shock wave to the field and update derivatives.
    
    Uses a smooth transition (tanh) to create a circular shock.
    The shock is a jump in the field value at a given radius from the center.
    
    Parameters
    ----------
    X, Y : ndarray
        Meshgrid arrays, shape (ny, nx)
    u, u_x, u_y, u_xx, u_yy, u_xy : ndarray
        Field and its derivatives, shape (ny, nx)
    center_x, center_y : float
        Center coordinates of the shock
    radius : float
        Radius of the shock circle
    amplitude : float
        Amplitude of the jump
    width : float
        Width of the transition (smaller = sharper shock)
    
    Returns
    -------
    u_shock, u_x_shock, u_y_shock, u_xx_shock, u_yy_shock, u_xy_shock : ndarray
        Field and derivatives with shock added
    """
    # Distance from center
    r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Smooth step function: tanh((r - R) / width)
    # This gives -1 inside (r < R) and +1 outside (r > R)
    step = np.tanh((r - radius) / width)
    
    # Shock contribution: amplitude * (step + 1) / 2
    # This gives 0 inside and amplitude outside
    shock = amplitude * (step + 1) / 2
    
    # Add shock to field
    u_shock = u + shock
    
    # Compute derivatives of the shock
    # d/dx[tanh((r-R)/w)] = (1/w) * sech²((r-R)/w) * (x - cx) / r
    # d/dy[tanh((r-R)/w)] = (1/w) * sech²((r-R)/w) * (y - cy) / r
    r_safe = np.where(r > 1e-10, r, 1e-10)  # Avoid division by zero
    sech2 = 1.0 / np.cosh((r - radius) / width)**2
    factor = amplitude / (2 * width) * sech2
    
    shock_x = factor * (X - center_x) / r_safe
    shock_y = factor * (Y - center_y) / r_safe
    
    # Second derivatives (more complex, but we'll compute them)
    # d²/dx² = factor * [1/r - (x-cx)²/r³] + d_factor/dx * (x-cx)/r
    # where d_factor/dx = -2*amplitude/(2*w²) * sech² * tanh * (x-cx)/r
    d_factor_dx = -2 * amplitude / (2 * width**2) * sech2 * step * (X - center_x) / r_safe
    d_factor_dy = -2 * amplitude / (2 * width**2) * sech2 * step * (Y - center_y) / r_safe
    
    shock_xx = factor * (1.0 / r_safe - (X - center_x)**2 / r_safe**3) + \
               d_factor_dx * (X - center_x) / r_safe
    shock_yy = factor * (1.0 / r_safe - (Y - center_y)**2 / r_safe**3) + \
               d_factor_dy * (Y - center_y) / r_safe
    shock_xy = factor * (-(X - center_x) * (Y - center_y) / r_safe**3) + \
               d_factor_dx * (Y - center_y) / r_safe
    
    # Add shock derivatives
    u_x_shock = u_x + shock_x
    u_y_shock = u_y + shock_y
    u_xx_shock = u_xx + shock_xx
    u_yy_shock = u_yy + shock_yy
    u_xy_shock = u_xy + shock_xy
    
    return u_shock, u_x_shock, u_y_shock, u_xx_shock, u_yy_shock, u_xy_shock


# ============================================================================
# Convergence Study Function
# ============================================================================

def run_convergence_study_2d(n_points_list, domain_x, domain_y, degree, n_basis, 
                             num_boundary_points, reg_param, clustering_factor, 
                             use_clustering, n_modes, seed, shock_enabled, 
                             shock_params):
    """
    Run convergence study for different grid resolutions in 2D.
    
    Returns errors for BSPF2D, FD-2, and Chebyshev at each resolution.
    """
    Lx = domain_x[1] - domain_x[0]
    Ly = domain_y[1] - domain_y[0]
    errors_bspf_dx_l2 = []
    errors_bspf_dx_linf = []
    errors_fd_dx_l2 = []
    errors_fd_dx_linf = []
    errors_cheb_dx_l2 = []
    errors_cheb_dx_linf = []
    n_total_list = []
    
    for N in n_points_list:
        print(f"  Running convergence study: N = {N} x {N}...")
        # Generate grid
        x = np.linspace(domain_x[0], domain_x[1], N)
        y = np.linspace(domain_y[0], domain_y[1], N)
        X_grid, Y_grid = np.meshgrid(x, y)
        
        # Evaluate turbulence field on grid using direct numerical computation
        u_turb, u_x_exact = evaluate_turbulence_field_on_grid(x, y, Lx, Ly, n_modes, seed)
        
        # Add shock if enabled
        if shock_enabled:
            u_turb, u_x_exact, _, _, _, _ = \
                add_circular_shock_wave(
                    X_grid, Y_grid, u_turb, u_x_exact,
                    np.zeros_like(u_turb), np.zeros_like(u_turb),
                    np.zeros_like(u_turb), np.zeros_like(u_turb),
                    center_x=shock_params['center_x'],
                    center_y=shock_params['center_y'],
                    radius=shock_params['radius'],
                    amplitude=shock_params['amplitude'],
                    width=shock_params['width']
                )
        
        # Initialize BSPF2D model
        from bspf import bspf1d
        x_model = bspf1d.from_grid(
            degree=degree,
            x=x,
            domain=tuple(domain_x),
            order=degree,
            n_basis=n_basis,
            num_boundary_points=num_boundary_points,
            use_clustering=use_clustering,
            clustering_factor=clustering_factor,
            correction="spectral"
        )
        y_model = bspf1d.from_grid(
            degree=degree,
            x=y,
            domain=tuple(domain_y),
            order=degree,
            n_basis=n_basis,
            num_boundary_points=num_boundary_points,
            use_clustering=use_clustering,
            clustering_factor=clustering_factor,
            correction="spectral"
        )
        model_2d = bspf2d(x=x, y=y, x_model=x_model, y_model=y_model, use_gpu=False)
        
        # Apply enforced_zero_flux
        u_turb_corrected = u_turb.copy()
        for j in range(N):
            f_left, f_right = model_2d.x_model.enforced_zero_flux(u_turb[j, :])
            u_turb_corrected[j, 0] = f_left
            u_turb_corrected[j, -1] = f_right
        for i in range(N):
            f_bottom, f_top = model_2d.y_model.enforced_zero_flux(u_turb[:, i])
            u_turb_corrected[0, i] = f_bottom
            u_turb_corrected[-1, i] = f_top
        
        # Compute BSPF derivatives
        plan_dx = model_2d.make_plan_dx(order=1, lam=reg_param, neumann=True)
        u_x_bspf = plan_dx.apply(u_turb_corrected, flux=(0.0, 0.0))
        
        # Compute FD derivatives
        u_x_fd = finite_difference_2nd_order_2d_x(x, y, u_turb)
        
        # Compute Chebyshev derivatives (generate field directly on Chebyshev nodes)
        u_x_cheb = chebyshev_derivative_2d_x(
            x, y, domain_x, domain_y, n_modes, seed, 
            shock_enabled=shock_enabled, shock_params=shock_params
        )
        
        # Compute errors
        errors_bspf_dx_l2.append(np.sqrt(np.mean((u_x_bspf - u_x_exact)**2)))
        errors_bspf_dx_linf.append(np.max(np.abs(u_x_bspf - u_x_exact)))
        errors_fd_dx_l2.append(np.sqrt(np.mean((u_x_fd - u_x_exact)**2)))
        errors_fd_dx_linf.append(np.max(np.abs(u_x_fd - u_x_exact)))
        errors_cheb_dx_l2.append(np.sqrt(np.mean((u_x_cheb - u_x_exact)**2)))
        errors_cheb_dx_linf.append(np.max(np.abs(u_x_cheb - u_x_exact)))
        n_total_list.append(N * N)
    
    return {
        'n_total': np.array(n_total_list),
        'errors_bspf_dx_l2': np.array(errors_bspf_dx_l2),
        'errors_bspf_dx_linf': np.array(errors_bspf_dx_linf),
        'errors_fd_dx_l2': np.array(errors_fd_dx_l2),
        'errors_fd_dx_linf': np.array(errors_fd_dx_linf),
        'errors_cheb_dx_l2': np.array(errors_cheb_dx_l2),
        'errors_cheb_dx_linf': np.array(errors_cheb_dx_linf)
    }


# ============================================================================
# Main Test Function
# ============================================================================

def main():
    """Main computation and visualization."""
    
    print("=" * 60)
    print("2D Synthetic Turbulence Test Case")
    print("=" * 60)
    print(f"Grid points: {NX} x {NY}")
    print(f"Domain: x ∈ [{DOMAIN_X[0]}, {DOMAIN_X[1]}], y ∈ [{DOMAIN_Y[0]}, {DOMAIN_Y[1]}]")
    print(f"Fourier modes: {TURB_N_MODES}")
    print(f"Power-law exponent: {TURB_POWER_LAW}")
    print(f"BSPF degree: {DEGREE}")
    print(f"Regularization: {REG_PARAM}")
    print("=" * 60)
    
    # Generate synthetic turbulence field using direct numerical computation
    print("\nGenerating 2D turbulence field...")
    Lx = DOMAIN_X[1] - DOMAIN_X[0]
    Ly = DOMAIN_Y[1] - DOMAIN_Y[0]
    
    # Create grid
    x = np.linspace(DOMAIN_X[0], DOMAIN_X[1], NX)
    y = np.linspace(DOMAIN_Y[0], DOMAIN_Y[1], NY)
    X_grid, Y_grid = np.meshgrid(x, y)
    
    # Evaluate turbulence field on grid using direct numerical computation
    u_turb, u_x_exact = evaluate_turbulence_field_on_grid(x, y, Lx, Ly, TURB_N_MODES, TURB_SEED)
    
    # Remove mean
    u_turb -= np.mean(u_turb)
    
    print(f"Field shape: {u_turb.shape}")
    print(f"Field range: [{u_turb.min():.6f}, {u_turb.max():.6f}]")
    
    # Add circular shock wave if enabled
    if SHOCK_ENABLED:
        print("\nAdding circular shock wave...")
        u_turb, u_x_exact, _, _, _, _ = \
            add_circular_shock_wave(
                X_grid, Y_grid, u_turb, u_x_exact,
                np.zeros_like(u_turb), np.zeros_like(u_turb),
                np.zeros_like(u_turb), np.zeros_like(u_turb),
                center_x=SHOCK_CENTER_X, center_y=SHOCK_CENTER_Y,
                radius=SHOCK_RADIUS, amplitude=SHOCK_AMPLITUDE, width=SHOCK_WIDTH
            )
        print(f"Shock center: ({SHOCK_CENTER_X}, {SHOCK_CENTER_Y})")
        print(f"Shock radius: {SHOCK_RADIUS}")
        print(f"Shock amplitude: {SHOCK_AMPLITUDE}")
        print(f"Field range after shock: [{u_turb.min():.6f}, {u_turb.max():.6f}]")
    
    # Initialize BSPF2D model
    print("\nInitializing BSPF2D model...")
    # Note: bspf2d.from_grids doesn't directly support clustering_factor,
    # so we need to create the 1D models separately if we want clustering
    from bspf import bspf1d
    x_model = bspf1d.from_grid(
        degree=DEGREE,
        x=x,
        domain=tuple(DOMAIN_X),
        order=DEGREE,
        n_basis=N_BASIS,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        use_clustering=USE_CLUSTERING,
        clustering_factor=CLUSTERING_FACTOR,
        correction="spectral"
    )
    y_model = bspf1d.from_grid(
        degree=DEGREE,
        x=y,
        domain=tuple(DOMAIN_Y),
        order=DEGREE,
        n_basis=N_BASIS,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        use_clustering=USE_CLUSTERING,
        clustering_factor=CLUSTERING_FACTOR,
        correction="spectral"
    )
    model_2d = bspf2d(x=x, y=y, x_model=x_model, y_model=y_model, use_gpu=False)
    
    # Apply enforced_zero_flux for zero gradient boundary conditions
    print("  Applying enforced zero-flux boundary conditions...")
    u_turb_corrected = u_turb.copy()
    # Apply to x-boundaries (left and right columns)
    for j in range(NY):
        f_left, f_right = model_2d.x_model.enforced_zero_flux(u_turb[j, :])
        u_turb_corrected[j, 0] = f_left
        u_turb_corrected[j, -1] = f_right
    # Apply to y-boundaries (bottom and top rows)
    for i in range(NX):
        f_bottom, f_top = model_2d.y_model.enforced_zero_flux(u_turb[:, i])
        u_turb_corrected[0, i] = f_bottom
        u_turb_corrected[-1, i] = f_top
    
    # Compute derivatives using different methods
    print("\nComputing derivatives...")
    
    # 1. BSPF2D method with zero-flux Neumann BC
    print("  Computing with BSPF2D (zero-flux Neumann BC)...")
    # Create plan for du/dx only
    plan_dx = model_2d.make_plan_dx(order=1, lam=REG_PARAM, neumann=True)
    
    u_x_bspf = plan_dx.apply(u_turb_corrected, flux=(0.0, 0.0))
    
    # 2. 2nd-order finite difference method
    print("  Computing with 2nd-order finite differences...")
    u_x_fd = finite_difference_2nd_order_2d_x(x, y, u_turb)
    
    # 3. Chebyshev spectral method
    print("  Computing with Chebyshev spectral method...")
    shock_params = {
        'center_x': SHOCK_CENTER_X,
        'center_y': SHOCK_CENTER_Y,
        'radius': SHOCK_RADIUS,
        'amplitude': SHOCK_AMPLITUDE,
        'width': SHOCK_WIDTH
    } if SHOCK_ENABLED else None
    u_x_cheb = chebyshev_derivative_2d_x(
        x, y, DOMAIN_X, DOMAIN_Y, TURB_N_MODES, TURB_SEED,
        shock_enabled=SHOCK_ENABLED, shock_params=shock_params
    )
    
    # Compute errors
    error_dx_bspf_l2 = np.sqrt(np.mean((u_x_bspf - u_x_exact)**2))
    error_dx_bspf_linf = np.max(np.abs(u_x_bspf - u_x_exact))
    
    error_dx_fd_l2 = np.sqrt(np.mean((u_x_fd - u_x_exact)**2))
    error_dx_fd_linf = np.max(np.abs(u_x_fd - u_x_exact))
    
    error_dx_cheb_l2 = np.sqrt(np.mean((u_x_cheb - u_x_exact)**2))
    error_dx_cheb_linf = np.max(np.abs(u_x_cheb - u_x_exact))
    
    print("\n" + "=" * 60)
    print("Derivative Errors (du/dx only):")
    print("=" * 60)
    print(f"du/dx:")
    print(f"  BSPF2D:")
    print(f"    L² norm:  {error_dx_bspf_l2:.6e}")
    print(f"    L∞ norm:  {error_dx_bspf_linf:.6e}")
    print(f"  Finite Difference (FD-2):")
    print(f"    L² norm:  {error_dx_fd_l2:.6e}")
    print(f"    L∞ norm:  {error_dx_fd_linf:.6e}")
    print(f"  Chebyshev:")
    print(f"    L² norm:  {error_dx_cheb_l2:.6e}")
    print(f"    L∞ norm:  {error_dx_cheb_linf:.6e}")
    print("=" * 60)
    
    # ========================================================================
    # Convergence Study
    # ========================================================================
    print("\nRunning convergence study...")
    # Ensure smallest N is larger than 2*TURB_N_MODES for proper resolution
    min_n = 2*TURB_N_MODES
    max_n = NX
    num_points = 5  # Number of points in the convergence study
    
    print(f"Minimum grid points per dimension: {min_n} (2 * {TURB_N_MODES} + 1)")
    print(f"Maximum grid points per dimension: {max_n}")
    
    # Generate geometric space between min_n and max_n
    n_points_list = np.geomspace(min_n, max_n, num_points, dtype=int).tolist()
    # Remove duplicates and sort
    n_points_list = sorted(list(set(n_points_list)))
    
    print(f"Convergence study grid resolutions: {n_points_list}")
    
    shock_params = {
        'center_x': SHOCK_CENTER_X,
        'center_y': SHOCK_CENTER_Y,
        'radius': SHOCK_RADIUS,
        'amplitude': SHOCK_AMPLITUDE,
        'width': SHOCK_WIDTH
    }
    
    conv_results = run_convergence_study_2d(
        n_points_list=n_points_list,
        domain_x=DOMAIN_X,
        domain_y=DOMAIN_Y,
        degree=DEGREE,
        n_basis=N_BASIS,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        reg_param=REG_PARAM,
        clustering_factor=CLUSTERING_FACTOR,
        use_clustering=USE_CLUSTERING,
        n_modes=TURB_N_MODES,
        seed=TURB_SEED,
        shock_enabled=SHOCK_ENABLED,
        shock_params=shock_params
    )
    
    # ========================================================================
    # Visualization
    # ========================================================================
    
    # Set up plotting parameters
    plt.rcParams.update({
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.5
    })
    
    fig = plt.figure(figsize=(12, 10))
    X_grid, Y_grid = np.meshgrid(x, y)
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # (a) Turbulence field
    ax1 = plt.subplot(2, 2, 1)
    im1 = ax1.contourf(X_grid, Y_grid, u_turb, levels=20, cmap='viridis')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.set_title('(a)', loc='left', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax1)
    
    # (b) Exact du/dx
    ax2 = plt.subplot(2, 2, 2)
    im2 = ax2.contourf(X_grid, Y_grid, u_x_exact, levels=20, cmap='RdBu_r')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')
    ax2.set_title('(b)', loc='left', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=ax2)
    
    # (c) Error in du/dx
    ax3 = plt.subplot(2, 2, 3)
    error_dx = np.abs(u_x_bspf - u_x_exact)
    im3 = ax3.contourf(X_grid, Y_grid, error_dx, levels=20, cmap='hot')
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$y$')
    ax3.set_title('(c)', loc='left', fontsize=14, fontweight='bold')
    plt.colorbar(im3, ax=ax3)
    
    # (d) Convergence study (L²)
    ax4 = plt.subplot(2, 2, 4)
    n_total = conv_results['n_total']
    
    # Plot L² errors
    ax4.loglog(n_total, conv_results['errors_bspf_dx_l2'], 'o-', 
               label='BSPF2D $\\partial u/\\partial x$ (L²)', color=default_colors[0], linewidth=1.5, markersize=6)
    ax4.loglog(n_total, conv_results['errors_fd_dx_l2'], 's--', 
               label='FD-2 $\\partial u/\\partial x$ (L²)', color=default_colors[2], linewidth=1.5, markersize=6)
    ax4.loglog(n_total, conv_results['errors_cheb_dx_l2'], '^-', 
               label='Chebyshev $\\partial u/\\partial x$ (L²)', color=default_colors[1], linewidth=1.5, markersize=6)
    
    # Add reference lines for convergence rates
    if len(n_total) > 1:
        ref_x = np.array([n_total[0], n_total[-1]])
        # Reference line for N^-1 (since h ~ 1/N, and h^2 ~ 1/N^2 for 2nd order in 2D)
        ref_y = conv_results['errors_bspf_dx_l2'][0] * (ref_x / n_total[0])**(-1)
        ax4.loglog(ref_x, ref_y, 'k:', linewidth=1, alpha=0.5, label='$N^{-1}$ reference')
    
    ax4.set_xlabel('Total number of nodes $N^2$')
    ax4.set_ylabel('L² Error')
    ax4.legend(fontsize=9)
    ax4.set_title('(d)', loc='left', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'x': x,
        'y': y,
        'u_turb': u_turb,
        'u_x_exact': u_x_exact,
        'u_x_bspf': u_x_bspf,
        'u_x_fd': u_x_fd,
        'u_x_cheb': u_x_cheb,
        'error_dx_bspf_l2': error_dx_bspf_l2,
        'error_dx_bspf_linf': error_dx_bspf_linf,
        'error_dx_fd_l2': error_dx_fd_l2,
        'error_dx_fd_linf': error_dx_fd_linf,
        'error_dx_cheb_l2': error_dx_cheb_l2,
        'error_dx_cheb_linf': error_dx_cheb_linf
    }


if __name__ == "__main__":
    main()

