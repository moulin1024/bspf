"""
Test derivative computation on 2D turbulence-like fields.

This script compares BSPF, Chebyshev spectral, and 2nd-order finite difference
methods for computing first and second derivatives of 2D turbulence-like fields
with Neumann (zero-flux) boundary conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from bspf import bspf2d
from bspf.utils import (
    construct_chebyshev_nodes,
    chebyshev_derivative_from_values,
    chebyshev_second_derivative_from_values
)


# ============================================================================
# Helper Functions
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
    F : ndarray
        Field values, shape (ny, nx)
    
    Returns
    -------
    dF_dx : ndarray
        First derivative with respect to x, shape (ny, nx)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    F = np.asarray(F)
    ny, nx = F.shape
    
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
    F : ndarray
        Field values, shape (ny, nx)
    
    Returns
    -------
    dF_dy : ndarray
        First derivative with respect to y, shape (ny, nx)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    F = np.asarray(F)
    ny, nx = F.shape
    
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
    if not np.allclose(np.diff(x), dx, rtol=1e-10):
        raise ValueError("Grid must be uniformly spaced for finite differences")
    
    d2F_dx2 = np.zeros_like(F)
    
    # Interior points: central difference (2nd order)
    d2F_dx2[:, 1:-1] = (F[:, 2:] - 2 * F[:, 1:-1] + F[:, :-2]) / (dx**2)
    
    # Boundaries: 2nd-order one-sided differences
    if nx >= 4:
        d2F_dx2[:, 0] = (2 * F[:, 0] - 5 * F[:, 1] + 4 * F[:, 2] - F[:, 3]) / (dx**2)
        d2F_dx2[:, -1] = (2 * F[:, -1] - 5 * F[:, -2] + 4 * F[:, -3] - F[:, -4]) / (dx**2)
    else:
        d2F_dx2[:, 0] = (F[:, 1] - 2 * F[:, 0] + F[:, 0]) / (dx**2)
        d2F_dx2[:, -1] = (F[:, -1] - 2 * F[:, -2] + F[:, -2]) / (dx**2)
    
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
    if not np.allclose(np.diff(y), dy, rtol=1e-10):
        raise ValueError("Grid must be uniformly spaced for finite differences")
    
    d2F_dy2 = np.zeros_like(F)
    
    # Interior points: central difference (2nd order)
    d2F_dy2[1:-1, :] = (F[2:, :] - 2 * F[1:-1, :] + F[:-2, :]) / (dy**2)
    
    # Boundaries: 2nd-order one-sided differences
    if ny >= 4:
        d2F_dy2[0, :] = (2 * F[0, :] - 5 * F[1, :] + 4 * F[2, :] - F[3, :]) / (dy**2)
        d2F_dy2[-1, :] = (2 * F[-1, :] - 5 * F[-2, :] + 4 * F[-3, :] - F[-4, :]) / (dy**2)
    else:
        d2F_dy2[0, :] = (F[1, :] - 2 * F[0, :] + F[0, :]) / (dy**2)
        d2F_dy2[-1, :] = (F[-1, :] - 2 * F[-2, :] + F[-2, :]) / (dy**2)
    
    return d2F_dy2


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
    
    Zero-flux (Neumann) boundary conditions are analytically satisfied:
    - Uses cosine modes: u(x,y) = sum(a_i * cos(kx_i * x) * cos(ky_i * y))
    - with kx_i = n_i * π / Lx, ky_i = m_i * π / Ly
    - Derivatives automatically satisfy zero-flux BC at boundaries
    
    Returns data in (ny, nx) format.
    
    Parameters
    ----------
    Lx, Ly : float
        Domain lengths
    Nx, Ny : int
        Number of grid points
    nmodes : int
        Number of Fourier modes (uses nmodes x nmodes modes)
    seed : int or None
        Random seed for reproducibility
    
    Returns
    -------
    x, y : ndarray
        Grid points
    u : ndarray
        Field values, shape (ny, nx)
    u_x : ndarray
        First derivative with respect to x, shape (ny, nx)
    u_y : ndarray
        First derivative with respect to y, shape (ny, nx)
    u_xx : ndarray
        Second derivative with respect to x, shape (ny, nx)
    u_yy : ndarray
        Second derivative with respect to y, shape (ny, nx)
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

    # Remove mean (common in turbulence signals)
    u -= np.mean(u)

    return x, y, u, u_x, u_y, u_xx, u_yy


def add_circular_shock_wave(X, Y, u, u_x, u_y, u_xx, u_yy,
                            center_x=0.0, center_y=0.0, radius=0.5, 
                            amplitude=1.0, width=0.02):
    """
    Add a circular shock wave to the field and update derivatives.
    
    Uses a smooth transition (tanh) to create a circular shock.
    The shock is a jump in the field value at a given radius from the center.
    
    Parameters
    ----------
    X, Y : ndarray
        Meshgrid coordinates, shape (ny, nx)
    u, u_x, u_y, u_xx, u_yy : ndarray
        Field and derivatives, shape (ny, nx)
    center_x, center_y : float
        Center coordinates of the shock
    radius : float
        Radius of the shock circle
    amplitude : float
        Amplitude of the shock jump
    width : float
        Width of the transition (smaller = sharper shock)
    
    Returns
    -------
    u_shock, u_x_shock, u_y_shock, u_xx_shock, u_yy_shock : ndarray
        Field and derivatives with shock added
    """
    # Distance from center
    r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Smooth step function: tanh((r - radius) / width)
    # This creates a transition from -1 (inside) to +1 (outside)
    step = np.tanh((r - radius) / width)
    
    # Shock contribution: amplitude * (step + 1) / 2
    # This gives 0 inside (r < radius) and amplitude outside (r > radius)
    shock = amplitude * (step + 1) / 2
    
    # Add shock to field
    u_shock = u + shock
    
    # Compute derivatives of the shock
    # Avoid division by zero
    r_safe = np.maximum(r, 1e-10)
    
    # Derivative factor: d/dr[tanh((r - radius) / width)] = (1/width) * sech^2((r - radius) / width)
    sech2 = 1.0 / np.cosh((r - radius) / width)**2
    factor = (amplitude / (2 * width)) * sech2
    
    # First derivatives: d/dx[shock] = factor * (x - center_x) / r
    shock_x = factor * (X - center_x) / r_safe
    shock_y = factor * (Y - center_y) / r_safe
    
    # Second derivatives (more complex due to chain rule)
    # d²/dx²[shock] = d/dx[factor * (x - center_x) / r]
    # This involves derivatives of factor and (x - center_x) / r
    d_factor_dr = -(amplitude / (2 * width**2)) * sech2 * step
    
    shock_xx = (d_factor_dr * (X - center_x)**2 / r_safe**2 + 
                factor * (1.0 / r_safe - (X - center_x)**2 / r_safe**3))
    shock_yy = (d_factor_dr * (Y - center_y)**2 / r_safe**2 + 
                factor * (1.0 / r_safe - (Y - center_y)**2 / r_safe**3))
    
    # Add shock derivatives
    u_x_shock = u_x + shock_x
    u_y_shock = u_y + shock_y
    u_xx_shock = u_xx + shock_xx
    u_yy_shock = u_yy + shock_yy
    
    return u_shock, u_x_shock, u_y_shock, u_xx_shock, u_yy_shock


# ============================================================================
# Parameters
# ============================================================================

# BSPF parameters
DEGREE = 9      # B-spline polynomial degree
NUM_BOUNDARY_POINTS = DEGREE + 3
N_BASIS = 4 * DEGREE
REG_PARAM = 1e-3      # Tikhonov regularisation strength (lam)
domain_x = [0, 2*np.pi]
domain_y = [0, 2*np.pi]
NUM_POINTS_X = 1000   # Grid resolution along x
NUM_POINTS_Y = 1000   # Grid resolution along y

# Grid parameters
clustering_flag = True 

# Turbulence parameters
TURB_N_MODES = 1
TURB_SEED = 123

# Shock wave parameters
SHOCK_ENABLED = True         # Enable circular shock wave
SHOCK_CENTER_X = np.pi       # X-coordinate of shock center (center of domain)
SHOCK_CENTER_Y = np.pi       # Y-coordinate of shock center (center of domain)
SHOCK_RADIUS = 1.0           # Radius of the shock
SHOCK_AMPLITUDE = 1.0        # Amplitude of the shock jump
SHOCK_WIDTH = 0.025           # Width of the shock transition (for smooth shock)

# Convergence study parameters
grid_sizes = np.geomspace(100, 500, 10).astype(int)


# ============================================================================
# Main Computation
# ============================================================================

def main():
    """Main computation and visualization."""
    
    # Generate grid on the requested domain
    x = np.linspace(domain_x[0], domain_x[1], NUM_POINTS_X, endpoint=True)
    y = np.linspace(domain_y[0], domain_y[1], NUM_POINTS_Y, endpoint=True)
    dx = (domain_x[1] - domain_x[0]) / (NUM_POINTS_X - 1)
    dy = (domain_y[1] - domain_y[0]) / (NUM_POINTS_Y - 1)
    
    # Initialize bspf2d model
    model = bspf2d.from_grids(
        x=x,
        y=y,
        degree_x=DEGREE,
        degree_y=DEGREE,
        domain_x=tuple(domain_x),
        domain_y=tuple(domain_y),
        n_basis_x=N_BASIS,
        n_basis_y=N_BASIS,
        num_boundary_points_x=NUM_BOUNDARY_POINTS,
        num_boundary_points_y=NUM_BOUNDARY_POINTS,
        use_clustering_x=clustering_flag,
        use_clustering_y=clustering_flag
    )
    
    # Generate 2D turbulence field
    x_signal, y_signal, u, u_x, u_y, u_xx, u_yy = random_turbulence_signal_2d_with_derivatives(
        Lx=domain_x[1] - domain_x[0],
        Ly=domain_y[1] - domain_y[0],
        Nx=NUM_POINTS_X,
        Ny=NUM_POINTS_Y,
        nmodes=TURB_N_MODES,
        seed=TURB_SEED,
    )
    
    # Override x and y with the signal grid to be safe
    x = x_signal
    y = y_signal
    
    # Create meshgrid for shock wave
    X_grid, Y_grid = np.meshgrid(x, y)  # (ny, nx)
    
    # Add circular shock wave if enabled
    if SHOCK_ENABLED:
        u, u_x, u_y, u_xx, u_yy = add_circular_shock_wave(
            X_grid, Y_grid, u, u_x, u_y, u_xx, u_yy,
            center_x=SHOCK_CENTER_X, center_y=SHOCK_CENTER_Y,
            radius=SHOCK_RADIUS, amplitude=SHOCK_AMPLITUDE, width=SHOCK_WIDTH
        )
    
    # Function and exact derivatives
    F = u.copy()
    F_original = u.copy()
    F_x_exact = u_x.copy()
    F_y_exact = u_y.copy()
    F_xx_exact = u_xx.copy()
    F_yy_exact = u_yy.copy()
    
    # Compute derivatives using different methods
    # 1. BSPF method on uniform grid with enforced zero-flux Neumann BC
    # Enforce zero-flux boundary conditions along both axes
    F_corrected = F_original.copy()
    # Enforce BCs along x (left/right boundaries) - process each row
    for j in range(NUM_POINTS_Y):
        f_left, f_right = model.x_model.enforced_zero_flux(F_corrected[j, :])
        F_corrected[j, 0] = f_left
        F_corrected[j, -1] = f_right
    # Enforce BCs along y (bottom/top boundaries) - process each column
    for i in range(NUM_POINTS_X):
        f_bottom, f_top = model.y_model.enforced_zero_flux(F_corrected[:, i])
        F_corrected[0, i] = f_bottom
        F_corrected[-1, i] = f_top
    
    # Compute derivatives with zero-flux Neumann BC
    # For x-derivative: flux is (left, right) = (0, 0)
    F_x_bspf = model.partial_dx_neumann(F_corrected, order=1, lam=REG_PARAM, flux=(0.0, 0.0))
    # For y-derivative: flux is (bottom, top) = (0, 0)
    F_y_bspf = model.partial_dy_neumann(F_corrected, order=1, lam=REG_PARAM, flux=(0.0, 0.0))
    # Second derivatives
    F_xx_bspf = model.partial_dxx_neumann(F_corrected, lam=REG_PARAM, flux=(0.0, 0.0))
    F_yy_bspf = model.partial_dyy_neumann(F_corrected, lam=REG_PARAM, flux=(0.0, 0.0))
    
    # Explicitly enforce zero-flux at boundaries for first derivatives
    F_x_bspf[:, 0] = 0.0
    F_x_bspf[:, -1] = 0.0
    F_y_bspf[0, :] = 0.0
    F_y_bspf[-1, :] = 0.0
    
    # 2. Chebyshev method on Chebyshev nodes using in-house implementation
    N_cheb_x = NUM_POINTS_X  # Number of intervals (N+1 nodes)
    N_cheb_y = NUM_POINTS_Y
    x_cheb, _ = construct_chebyshev_nodes(N_cheb_x, domain=tuple(domain_x))
    y_cheb, _ = construct_chebyshev_nodes(N_cheb_y, domain=tuple(domain_y))
    
    # Evaluate field on Chebyshev nodes (tensor product)
    X_cheb, Y_cheb = np.meshgrid(x_cheb, y_cheb)
    
    # Generate turbulence field parameters (fixed by seed) - same as uniform grid
    Lx = domain_x[1] - domain_x[0]
    Ly = domain_y[1] - domain_y[0]
    np.random.seed(TURB_SEED)
    n = np.arange(1, TURB_N_MODES + 1)
    m = np.arange(1, TURB_N_MODES + 1)
    kx = n * np.pi / Lx
    ky = m * np.pi / Ly
    k_mag = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
    sigma = k_mag**(-5/6)
    n_total = TURB_N_MODES * TURB_N_MODES
    indices = np.random.choice(n_total, size=min(TURB_N_MODES, n_total), replace=False)
    i_flat, j_flat = np.unravel_index(indices, (TURB_N_MODES, TURB_N_MODES))
    
    # Store mode parameters for evaluation
    modes = []
    for idx in range(len(i_flat)):
        ni = n[i_flat[idx]]
        mi = m[j_flat[idx]]
        kx_val = ni * np.pi / Lx
        ky_val = mi * np.pi / Ly
        ai = sigma[i_flat[idx], j_flat[idx]] * np.random.randn()
        modes.append((ni, mi, kx_val, ky_val, ai))
    
    # Evaluate turbulence directly at Chebyshev nodes
    F_cheb = np.zeros_like(X_cheb)
    F_x_cheb_exact = np.zeros_like(X_cheb)
    F_y_cheb_exact = np.zeros_like(X_cheb)
    F_xx_cheb_exact = np.zeros_like(X_cheb)
    F_yy_cheb_exact = np.zeros_like(X_cheb)
    
    for ni, mi, kx_val, ky_val, ai in modes:
        cos_x = np.cos(kx_val * X_cheb)
        cos_y = np.cos(ky_val * Y_cheb)
        sin_x = np.sin(kx_val * X_cheb)
        sin_y = np.sin(ky_val * Y_cheb)
        
        # Field
        F_cheb += ai * cos_x * cos_y
        
        # First derivatives
        F_x_cheb_exact += ai * (-kx_val) * sin_x * cos_y
        F_y_cheb_exact += ai * (-ky_val) * cos_x * sin_y
        
        # Second derivatives
        F_xx_cheb_exact += ai * (-kx_val**2) * cos_x * cos_y
        F_yy_cheb_exact += ai * (-ky_val**2) * cos_x * cos_y
    
    # Remove mean (common in turbulence signals)
    F_cheb -= np.mean(F_cheb)
    
    # Add circular shock wave if enabled (evaluate at Chebyshev nodes)
    if SHOCK_ENABLED:
        F_cheb, F_x_cheb_exact, F_y_cheb_exact, F_xx_cheb_exact, F_yy_cheb_exact = \
            add_circular_shock_wave(
                X_cheb, Y_cheb, F_cheb, F_x_cheb_exact, F_y_cheb_exact, 
                F_xx_cheb_exact, F_yy_cheb_exact,
                center_x=SHOCK_CENTER_X, center_y=SHOCK_CENTER_Y,
                radius=SHOCK_RADIUS, amplitude=SHOCK_AMPLITUDE, width=SHOCK_WIDTH
            )
    
    # Compute derivatives using Chebyshev (apply along each axis)
    # For x-derivative: differentiate along axis=1 (columns)
    F_x_cheb = np.zeros_like(F_cheb)
    for j in range(F_cheb.shape[0]):
        F_x_cheb[j, :] = chebyshev_derivative_from_values(F_cheb[j, :], x_cheb, domain=tuple(domain_x))
    # For y-derivative: differentiate along axis=0 (rows)
    F_y_cheb = np.zeros_like(F_cheb)
    for i in range(F_cheb.shape[1]):
        F_y_cheb[:, i] = chebyshev_derivative_from_values(F_cheb[:, i], y_cheb, domain=tuple(domain_y))
    # Second derivatives
    F_xx_cheb = np.zeros_like(F_cheb)
    for j in range(F_cheb.shape[0]):
        F_xx_cheb[j, :] = chebyshev_second_derivative_from_values(F_cheb[j, :], x_cheb, domain=tuple(domain_x))
    F_yy_cheb = np.zeros_like(F_cheb)
    for i in range(F_cheb.shape[1]):
        F_yy_cheb[:, i] = chebyshev_second_derivative_from_values(F_cheb[:, i], y_cheb, domain=tuple(domain_y))
    
    # Compute errors for each method (L2 norm)
    error_x_bspf = np.linalg.norm(F_x_bspf - F_x_exact, 'fro') * np.sqrt(dx * dy)
    error_y_bspf = np.linalg.norm(F_y_bspf - F_y_exact, 'fro') * np.sqrt(dx * dy)
    dx_cheb = (domain_x[1] - domain_x[0]) / N_cheb_x
    dy_cheb = (domain_y[1] - domain_y[0]) / N_cheb_y
    error_x_cheb = np.linalg.norm(F_x_cheb - F_x_cheb_exact, 'fro') * np.sqrt(dx_cheb * dy_cheb)
    error_y_cheb = np.linalg.norm(F_y_cheb - F_y_cheb_exact, 'fro') * np.sqrt(dx_cheb * dy_cheb)
    
    # Second derivative errors
    error_xx_bspf = np.linalg.norm(F_xx_bspf - F_xx_exact, 'fro') * np.sqrt(dx * dy)
    error_yy_bspf = np.linalg.norm(F_yy_bspf - F_yy_exact, 'fro') * np.sqrt(dx * dy)
    error_xx_cheb = np.linalg.norm(F_xx_cheb - F_xx_cheb_exact, 'fro') * np.sqrt(dx_cheb * dy_cheb)
    error_yy_cheb = np.linalg.norm(F_yy_cheb - F_yy_cheb_exact, 'fro') * np.sqrt(dx_cheb * dy_cheb)
    
    print("First Derivative Errors (L^2 Norm):")
    print(f"  BSPF (x):     {error_x_bspf:.6e}")
    print(f"  BSPF (y):     {error_y_bspf:.6e}")
    print(f"  Chebyshev (x): {error_x_cheb:.6e}")
    print(f"  Chebyshev (y): {error_y_cheb:.6e}")
    print("\nSecond Derivative Errors (L^2 Norm):")
    print(f"  BSPF (xx):    {error_xx_bspf:.6e}")
    print(f"  BSPF (yy):    {error_yy_bspf:.6e}")
    print(f"  Chebyshev (xx): {error_xx_cheb:.6e}")
    print(f"  Chebyshev (yy): {error_yy_cheb:.6e}")
    
    # ========================================================================
    # Convergence Study (run before visualization)
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("Running convergence study...")
    print("=" * 60)
    
    errors_x_bspf = []
    errors_x_cheb = []
    errors_xx_bspf = []
    errors_xx_cheb = []
    
    for n_points in grid_sizes:
        # Create grid (square grid: n_points x n_points)
        x_test = np.linspace(domain_x[0], domain_x[1], n_points, endpoint=True)
        y_test = np.linspace(domain_y[0], domain_y[1], n_points, endpoint=True)
        dx_test = (domain_x[1] - domain_x[0]) / (n_points - 1)
        dy_test = (domain_y[1] - domain_y[0]) / (n_points - 1)
        
        # Generate turbulence field
        x_signal_test, y_signal_test, u_test, u_x_test, u_y_test, u_xx_test, u_yy_test = \
            random_turbulence_signal_2d_with_derivatives(
                Lx=domain_x[1] - domain_x[0],
                Ly=domain_y[1] - domain_y[0],
                Nx=n_points,
                Ny=n_points,
                nmodes=TURB_N_MODES,
                seed=TURB_SEED,
            )
        
        # Add shock wave if enabled
        X_test, Y_test = np.meshgrid(x_test, y_test)
        if SHOCK_ENABLED:
            u_test, u_x_test, u_y_test, u_xx_test, u_yy_test = add_circular_shock_wave(
                X_test, Y_test, u_test, u_x_test, u_y_test, u_xx_test, u_yy_test,
                center_x=SHOCK_CENTER_X, center_y=SHOCK_CENTER_Y,
                radius=SHOCK_RADIUS, amplitude=SHOCK_AMPLITUDE, width=SHOCK_WIDTH
            )
        
        F_test = u_test.copy()
        F_x_exact_test = u_x_test.copy()
        F_xx_exact_test = u_xx_test.copy()
        
        # BSPF method
        model_test = bspf2d.from_grids(
            x=x_test,
            y=y_test,
            degree_x=DEGREE,
            degree_y=DEGREE,
            domain_x=tuple(domain_x),
            domain_y=tuple(domain_y),
            n_basis_x=N_BASIS,
            n_basis_y=N_BASIS,
            num_boundary_points_x=NUM_BOUNDARY_POINTS,
            num_boundary_points_y=NUM_BOUNDARY_POINTS,
            use_clustering_x=clustering_flag,
            use_clustering_y=clustering_flag
        )
        
        # Enforce zero-flux boundary conditions
        F_corrected_test = F_test.copy()
        for j in range(n_points):
            f_left, f_right = model_test.x_model.enforced_zero_flux(F_corrected_test[j, :])
            F_corrected_test[j, 0] = f_left
            F_corrected_test[j, -1] = f_right
        for i in range(n_points):
            f_bottom, f_top = model_test.y_model.enforced_zero_flux(F_corrected_test[:, i])
            F_corrected_test[0, i] = f_bottom
            F_corrected_test[-1, i] = f_top
        
        F_x_bspf_test = model_test.partial_dx_neumann(F_corrected_test, order=1, lam=REG_PARAM, flux=(0.0, 0.0))
        F_x_bspf_test[:, 0] = 0.0
        F_x_bspf_test[:, -1] = 0.0
        
        error_x_bspf_test = np.linalg.norm(F_x_bspf_test - F_x_exact_test, 'fro') * np.sqrt(dx_test * dy_test)
        errors_x_bspf.append(error_x_bspf_test)
        
        # Second derivative for BSPF
        F_xx_bspf_test = model_test.partial_dxx_neumann(F_corrected_test, lam=REG_PARAM, flux=(0.0, 0.0))
        error_xx_bspf_test = np.linalg.norm(F_xx_bspf_test - F_xx_exact_test, 'fro') * np.sqrt(dx_test * dy_test)
        errors_xx_bspf.append(error_xx_bspf_test)
        
        # Chebyshev method
        N_cheb_test = n_points
        x_cheb_test, _ = construct_chebyshev_nodes(N_cheb_test, domain=tuple(domain_x))
        y_cheb_test, _ = construct_chebyshev_nodes(N_cheb_test, domain=tuple(domain_y))
        X_cheb_test, Y_cheb_test = np.meshgrid(x_cheb_test, y_cheb_test)
        
        # Generate turbulence at Chebyshev nodes
        np.random.seed(TURB_SEED)
        n = np.arange(1, TURB_N_MODES + 1)
        m = np.arange(1, TURB_N_MODES + 1)
        kx = n * np.pi / (domain_x[1] - domain_x[0])
        ky = m * np.pi / (domain_y[1] - domain_y[0])
        k_mag = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
        sigma = k_mag**(-5/6)
        n_total = TURB_N_MODES * TURB_N_MODES
        indices = np.random.choice(n_total, size=min(TURB_N_MODES, n_total), replace=False)
        i_flat, j_flat = np.unravel_index(indices, (TURB_N_MODES, TURB_N_MODES))
        
        modes_test = []
        for idx in range(len(i_flat)):
            ni = n[i_flat[idx]]
            mi = m[j_flat[idx]]
            kx_val = ni * np.pi / (domain_x[1] - domain_x[0])
            ky_val = mi * np.pi / (domain_y[1] - domain_y[0])
            ai = sigma[i_flat[idx], j_flat[idx]] * np.random.randn()
            modes_test.append((ni, mi, kx_val, ky_val, ai))
        
        F_cheb_test = np.zeros_like(X_cheb_test)
        F_x_cheb_exact_test = np.zeros_like(X_cheb_test)
        F_xx_cheb_exact_test = np.zeros_like(X_cheb_test)
        
        for ni, mi, kx_val, ky_val, ai in modes_test:
            cos_x = np.cos(kx_val * X_cheb_test)
            cos_y = np.cos(ky_val * Y_cheb_test)
            sin_x = np.sin(kx_val * X_cheb_test)
            F_cheb_test += ai * cos_x * cos_y
            F_x_cheb_exact_test += ai * (-kx_val) * sin_x * cos_y
            F_xx_cheb_exact_test += ai * (-kx_val**2) * cos_x * cos_y
        
        F_cheb_test -= np.mean(F_cheb_test)
        
        # Add shock wave at Chebyshev nodes
        if SHOCK_ENABLED:
            F_cheb_test, F_x_cheb_exact_test, _, F_xx_cheb_exact_test, _ = add_circular_shock_wave(
                X_cheb_test, Y_cheb_test, F_cheb_test, F_x_cheb_exact_test,
                np.zeros_like(F_cheb_test), F_xx_cheb_exact_test, np.zeros_like(F_cheb_test),
                center_x=SHOCK_CENTER_X, center_y=SHOCK_CENTER_Y,
                radius=SHOCK_RADIUS, amplitude=SHOCK_AMPLITUDE, width=SHOCK_WIDTH
            )
        
        # Compute x-derivative along axis=1 (columns)
        F_x_cheb_test = np.zeros_like(F_cheb_test)
        for j in range(F_cheb_test.shape[0]):
            F_x_cheb_test[j, :] = chebyshev_derivative_from_values(F_cheb_test[j, :], x_cheb_test, domain=tuple(domain_x))
        dx_cheb_test = (domain_x[1] - domain_x[0]) / N_cheb_test
        dy_cheb_test = (domain_y[1] - domain_y[0]) / N_cheb_test
        error_x_cheb_test = np.linalg.norm(F_x_cheb_test - F_x_cheb_exact_test, 'fro') * np.sqrt(dx_cheb_test * dy_cheb_test)
        errors_x_cheb.append(error_x_cheb_test)
        
        # Compute second derivative for Chebyshev
        F_xx_cheb_test = np.zeros_like(F_cheb_test)
        for j in range(F_cheb_test.shape[0]):
            F_xx_cheb_test[j, :] = chebyshev_second_derivative_from_values(F_cheb_test[j, :], x_cheb_test, domain=tuple(domain_x))
        error_xx_cheb_test = np.linalg.norm(F_xx_cheb_test - F_xx_cheb_exact_test, 'fro') * np.sqrt(dx_cheb_test * dy_cheb_test)
        errors_xx_cheb.append(error_xx_cheb_test)
        
        print(f"N = {n_points:5d} | 1st: BSPF: {error_x_bspf_test:.6e} | Cheb: {error_x_cheb_test:.6e}")
        print(f"      | 2nd: BSPF: {error_xx_bspf_test:.6e} | Cheb: {error_xx_cheb_test:.6e}")
    
    # ========================================================================
    # Visualization - 2x3 layout
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("Creating visualization...")
    print("=" * 60)
    
    # Set up global plotting parameters
    plt.rcParams.update({
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    # Plotting - 2x3 layout for 2D fields
    fig = plt.figure(figsize=(18, 12))
    
    # Row 1: Field and derivatives
    # (a) Original field
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.contourf(x, y, F_original, levels=20, cmap='Blues')
    ax1.set_xlabel('$x$', fontsize=16)
    ax1.set_ylabel('$y$', fontsize=16)
    ax1.set_title('(a) Field $f(x,y)$', fontsize=16, fontweight='bold')
    plt.colorbar(im1, ax=ax1)
    
    # (b) BSPF 1st order x-derivative
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.contourf(x, y, F_x_bspf, levels=20, cmap='Blues')
    ax2.set_xlabel('$x$', fontsize=16)
    ax2.set_ylabel('$y$', fontsize=16)
    ax2.set_title('(b) BSPF $\\partial f/\\partial x$', fontsize=16, fontweight='bold')
    plt.colorbar(im2, ax=ax2)
    
    # (c) BSPF 2nd order x-derivative
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.contourf(x, y, F_xx_bspf, levels=20, cmap='Blues')
    ax3.set_xlabel('$x$', fontsize=16)
    ax3.set_ylabel('$y$', fontsize=16)
    ax3.set_title('(c) BSPF $\\partial^2 f/\\partial x^2$', fontsize=16, fontweight='bold')
    plt.colorbar(im3, ax=ax3)
    
    # Row 2: Convergence and errors
    # (d) Convergence plot
    ax4 = plt.subplot(2, 3, 4)
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Plot convergence curves for first derivatives
    ax4.loglog(grid_sizes, errors_x_bspf, '.-', linewidth=2, label='BSPF (1st)',
              color=default_colors[0], alpha=1, markersize=8)
    ax4.loglog(grid_sizes, errors_x_cheb, '.-', linewidth=2, label='Chebyshev (1st)',
              color=default_colors[1], alpha=1, markersize=8)
    
    # Plot convergence curves for second derivatives
    ax4.loglog(grid_sizes, errors_xx_bspf, 's-', linewidth=2, label='BSPF (2nd)',
              color=default_colors[0], alpha=0.7, markersize=6)
    ax4.loglog(grid_sizes, errors_xx_cheb, 's-', linewidth=2, label='Chebyshev (2nd)',
              color=default_colors[1], alpha=0.7, markersize=6)
    
    # Add reference n^-2 line
    # Scale the reference line to match the error magnitude
    ref_scale = errors_x_bspf[0] * (grid_sizes[0]**2)
    ax4.loglog(grid_sizes, ref_scale / (grid_sizes**2), 'k--', linewidth=1.5, 
               label='$O(N^{-2})$', alpha=0.6)
    
    # Mark the grid size used in main computation
    ax4.axvline(NUM_POINTS_X, linestyle='--', color='gray', linewidth=1)
    ax4.text(NUM_POINTS_X * 1.1, errors_x_bspf[0] * 0.5, f'Main: {NUM_POINTS_X}', 
            color='gray', fontsize=12)
    
    ax4.set_xlabel('$N$ (grid points per dimension)', fontsize=16)
    ax4.set_ylabel('$\\|Error\\|_2$', fontsize=16)
    ax4.set_title('(d) Convergence Study: $\\partial f/\\partial x$ and $\\partial^2 f/\\partial x^2$', fontsize=16, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=11, loc='best', ncol=2)
    
    # (e) Error in 1st order x-derivative (BSPF)
    ax5 = plt.subplot(2, 3, 5)
    error_x_bspf_field = np.abs(F_x_bspf - F_x_exact)
    im5 = ax5.contourf(x, y, error_x_bspf_field, levels=20, cmap='Blues')
    ax5.set_xlabel('$x$', fontsize=16)
    ax5.set_ylabel('$y$', fontsize=16)
    ax5.set_title('(e) Error $|\\partial f/\\partial x|$ (BSPF)', fontsize=16, fontweight='bold')
    plt.colorbar(im5, ax=ax5)
    
    # (f) Error in 2nd order x-derivative (BSPF)
    ax6 = plt.subplot(2, 3, 6)
    error_xx_bspf_field = np.abs(F_xx_bspf - F_xx_exact)
    im6 = ax6.contourf(x, y, error_xx_bspf_field, levels=20, cmap='Blues')
    ax6.set_xlabel('$x$', fontsize=16)
    ax6.set_ylabel('$y$', fontsize=16)
    ax6.set_title('(f) Error $|\\partial^2 f/\\partial x^2|$ (BSPF)', fontsize=16, fontweight='bold')
    plt.colorbar(im6, ax=ax6)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

