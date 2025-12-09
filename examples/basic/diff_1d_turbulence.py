"""
Test derivative computation on noisy turbulence-like signals.

This script compares BSPF, Chebyshev spectral, and 2nd-order finite difference
methods for computing first and second derivatives of turbulence-like signals
with Neumann (zero-flux) boundary conditions.
"""

import numpy as np
import matplotlib.pyplot as plt

from bspf import bspf1d
from specderiv import cheb_deriv


# ============================================================================
# Helper Functions
# ============================================================================

def finite_difference_2nd_order(x, y):
    """
    Compute first derivative using 2nd-order finite differences.
    
    Uses central differences for interior points and 2nd-order one-sided
    differences for boundary points.
    
    Parameters
    ----------
    x : array_like
        Grid points (must be uniformly spaced)
    y : array_like
        Function values at grid points
    
    Returns
    -------
    dy : ndarray
        Derivative values at grid points
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    dy = np.zeros_like(y)
    
    # Check if grid is uniform
    dx = x[1] - x[0]
    if not np.allclose(np.diff(x), dx, rtol=1e-10):
        raise ValueError("Grid must be uniformly spaced for finite differences")
    
    # Interior points: central difference (2nd order)
    # f'(x_i) ≈ (f(x_{i+1}) - f(x_{i-1})) / (2*dx)
    dy[1:-1] = (y[2:] - y[:-2]) / (2 * dx)
    
    # Left boundary: 2nd-order forward difference
    # f'(x_0) ≈ (-3*f(x_0) + 4*f(x_1) - f(x_2)) / (2*dx)
    dy[0] = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * dx)
    
    # Right boundary: 2nd-order backward difference
    # f'(x_{n-1}) ≈ (3*f(x_{n-1}) - 4*f(x_{n-2}) + f(x_{n-3})) / (2*dx)
    dy[-1] = (3 * y[-1] - 4 * y[-2] + y[-3]) / (2 * dx)
    
    return dy


def finite_difference_2nd_order_2nd_deriv(x, y):
    """
    Compute second derivative using 2nd-order finite differences.
    
    Uses central differences for interior points and 2nd-order one-sided
    differences for boundary points.
    
    Parameters
    ----------
    x : array_like
        Grid points (must be uniformly spaced)
    y : array_like
        Function values at grid points
    
    Returns
    -------
    d2y : ndarray
        Second derivative values at grid points
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    d2y = np.zeros_like(y)
    
    # Check if grid is uniform
    dx = x[1] - x[0]
    if not np.allclose(np.diff(x), dx, rtol=1e-10):
        raise ValueError("Grid must be uniformly spaced for finite differences")
    
    # Interior points: central difference (2nd order)
    # f''(x_i) ≈ (f(x_{i+1}) - 2*f(x_i) + f(x_{i-1})) / (dx^2)
    d2y[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / (dx**2)
    
    # Left boundary: 2nd-order forward difference
    # f''(x_0) ≈ (2*f(x_0) - 5*f(x_1) + 4*f(x_2) - f(x_3)) / (dx^2)
    if n >= 4:
        d2y[0] = (2 * y[0] - 5 * y[1] + 4 * y[2] - y[3]) / (dx**2)
    else:
        # Fallback for very small arrays
        d2y[0] = (y[1] - 2 * y[0] + y[0]) / (dx**2)
    
    # Right boundary: 2nd-order backward difference
    # f''(x_{n-1}) ≈ (2*f(x_{n-1}) - 5*f(x_{n-2}) + 4*f(x_{n-3}) - f(x_{n-4})) / (dx^2)
    if n >= 4:
        d2y[-1] = (2 * y[-1] - 5 * y[-2] + 4 * y[-3] - y[-4]) / (dx**2)
    else:
        # Fallback for very small arrays
        d2y[-1] = (y[-1] - 2 * y[-2] + y[-2]) / (dx**2)
    
    return d2y


def random_turbulence_signal_with_derivatives(
    L=1.0, 
    N=2048, 
    nmodes=200, 
    seed=None
):
    """
    Generate a turbulence-like 1D signal using cosine modes with
    a Kolmogorov-like spectrum, and return analytical first
    and second derivatives.
    
    Zero-flux (Neumann) boundary conditions are analytically satisfied:
    - Uses cosine modes: u(x) = sum(a_i * cos(k_i * x)) with k_i = n_i * π / L
    - Derivative: u_x(x) = sum(-a_i * k_i * sin(k_i * x))
    - At x=0: sin(0) = 0 → u_x(0) = 0 (zero flux)
    - At x=L: sin(k_i * L) = sin(n_i * π) = 0 → u_x(L) = 0 (zero flux)
    
    Parameters
    ----------
    L : float
        Domain length
    N : int
        Number of grid points
    nmodes : int
        Number of Fourier modes
    seed : int or None
        Random seed for reproducibility
    
    Returns
    -------
    x : ndarray
        Grid points
    u : ndarray
        Signal values
    u_x : ndarray
        First derivative (analytical, zero flux at boundaries)
    u_xx : ndarray
        Second derivative (analytical)
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(0, L, N)

    # Mode index n = 1 ... nmodes
    # Using n >= 1 ensures k = n*π/L, which gives sin(k*L) = sin(n*π) = 0
    # This analytically satisfies zero-flux BC at x = L
    n = np.arange(1, nmodes + 1)
    k = n * np.pi / L  # wavenumbers chosen to satisfy zero-flux BC

    # Kolmogorov-like scaling: E(k) ~ k^(-5/3) → amplitude ~ k^(-5/6)
    sigma = k**(-5/6)

    # Random amplitudes (fixed by seed)
    a = sigma * np.random.randn(nmodes)

    # Allocate arrays
    u = np.zeros_like(x)
    u_x = np.zeros_like(x)
    u_xx = np.zeros_like(x)

    # Build signal + exact derivatives using cosine modes
    # Cosine modes with k = n*π/L automatically satisfy zero-flux BC
    for ai, ni in zip(a, n):
        w = ni * np.pi / L  # angular wavenumber
        cos_term = np.cos(w * x)
        sin_term = np.sin(w * x)

        u     += ai * cos_term
        u_x   += ai * (-w)      * sin_term  # sin(0) = sin(n*π) = 0 → zero flux
        u_xx  += ai * (-w**2)   * cos_term

    # Remove mean of u (common in turbulence signals)
    # This does not affect the zero-flux BC since mean removal is a constant
    u -= np.mean(u)

    # Verify zero-flux boundary conditions analytically satisfied
    # (should be exactly zero up to numerical precision)
    u_x_at_0 = u_x[0]
    u_x_at_L = u_x[-1]
    
    # Check that boundaries are zero (within numerical precision)
    if abs(u_x_at_0) > 1e-12 or abs(u_x_at_L) > 1e-12:
        import warnings
        warnings.warn(
            f"Zero-flux BC not exactly satisfied: u_x(0)={u_x_at_0:.2e}, "
            f"u_x(L)={u_x_at_L:.2e}. This may indicate a numerical issue."
        )

    return x, u, u_x, u_xx


def evaluate_turbulence_at_points(x_eval, L, nmodes, seed, a=None):
    """
    Evaluate turbulence signal and derivatives at arbitrary points.
    
    Parameters
    ----------
    x_eval : ndarray
        Points at which to evaluate
    L : float
        Domain length
    nmodes : int
        Number of Fourier modes
    seed : int or None
        Random seed (must match the one used to generate original signal)
    a : ndarray or None
        Pre-computed random amplitudes (if None, will regenerate)
    
    Returns
    -------
    u : ndarray
        Signal values at x_eval
    u_x : ndarray
        First derivative at x_eval
    u_xx : ndarray
        Second derivative at x_eval
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Mode index n = 1 ... nmodes
    n = np.arange(1, nmodes + 1)
    k = n * np.pi / L  # wavenumbers
    
    # Kolmogorov-like scaling: E(k) ~ k^(-5/3) → amplitude ~ k^(-5/6)
    sigma = k**(-5/6)
    
    # Random amplitudes (fixed by seed)
    if a is None:
        a = sigma * np.random.randn(nmodes)
    
    # Allocate arrays
    u = np.zeros_like(x_eval)
    u_x = np.zeros_like(x_eval)
    u_xx = np.zeros_like(x_eval)
    
    # Build signal + exact derivatives
    for ai, ni in zip(a, n):
        w = ni * np.pi / L  # angular wavenumber
        cos_term = np.cos(w * x_eval)
        sin_term = np.sin(w * x_eval)
        
        u     += ai * cos_term
        u_x   += ai * (-w)      * sin_term
        u_xx  += ai * (-w**2)   * cos_term
    
    # Remove mean (consistent with random_turbulence_signal_with_derivatives)
    u -= np.mean(u)
    
    return u, u_x, u_xx


def patch_second_derivative_boundaries(d2f, f, dx, n_boundary=3, order=4):
    """
    Patch boundary values of second derivative using one-sided finite differences.
    
    Parameters
    ----------
    d2f : ndarray
        Second derivative array (will be modified in-place)
    f : ndarray
        Original function values
    dx : float
        Grid spacing
    n_boundary : int
        Number of boundary points to patch
    order : int
        Order of finite difference scheme
    
    Returns
    -------
    d2f : ndarray
        Patched second derivative
    """
    d2f = d2f.copy()
    n = len(f)
    
    # Left boundary: forward differences
    for i in range(min(n_boundary, n // 2)):
        if i == 0:
            # 2nd order: f''(0) ≈ (f(2) - 2*f(1) + f(0)) / dx^2
            d2f[i] = (f[i+2] - 2*f[i+1] + f[i]) / (dx**2)
        elif i == 1:
            # 3rd order
            d2f[i] = (f[i+2] - 2*f[i+1] + f[i]) / (dx**2)
        else:
            # Higher order if needed
            d2f[i] = (f[i+2] - 2*f[i+1] + f[i]) / (dx**2)
    
    # Right boundary: backward differences
    for i in range(max(0, n - n_boundary), n):
        if i == n - 1:
            # 2nd order: f''(n-1) ≈ (f(n-1) - 2*f(n-2) + f(n-3)) / dx^2
            d2f[i] = (f[i] - 2*f[i-1] + f[i-2]) / (dx**2)
        elif i == n - 2:
            d2f[i] = (f[i] - 2*f[i-1] + f[i-2]) / (dx**2)
        else:
            d2f[i] = (f[i] - 2*f[i-1] + f[i-2]) / (dx**2)
    
    return d2f


# ============================================================================
# Parameters
# ============================================================================

# BSPF parameters
DEGREE = 7      # B-spline polynomial degree
MATCH_ORDER = DEGREE
NUM_BOUNDARY_POINTS = DEGREE + 3
N_BASIS = 4 * (DEGREE)
REG_PARAM = 1e-3      # Tikhonov regularisation strength (lam)
domain = [0, 2*np.pi]
NUM_POINTS = 1000   # Grid resolution

# Grid parameters
clustering_factor = 3.0  # Stronger clustering near endpoints
clustering_flag = True 

# Turbulence parameters
TURB_N_MODES = 100
TURB_SEED = 123

# Convergence study parameters
grid_sizes = np.geomspace(100, 1000, 50).astype(int)


# ============================================================================
# Main Computation
# ============================================================================

def main():
    """Main computation and visualization."""
    
    # Generate grid on the requested domain
    x = np.linspace(domain[0], domain[1], NUM_POINTS, endpoint=True)
    dx = (domain[1] - domain[0]) / (NUM_POINTS - 1)
    
    # Initialize bspf1d model
    model = bspf1d.from_grid(
        degree=DEGREE,
        x=x,
        domain=tuple(domain),
        order=DEGREE,
        n_basis=N_BASIS,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        use_clustering=clustering_flag,
        clustering_factor=clustering_factor
    )
    
    # Use random_turbulence_signal_with_derivatives to generate u, u_x, u_xx
    # This ensures Neumann (zero-flux) BC automatically.
    x_signal, u, u_x, u_xx = random_turbulence_signal_with_derivatives(
        L=domain[1] - domain[0],
        N=NUM_POINTS,
        nmodes=TURB_N_MODES,
        seed=TURB_SEED,
    )
    
    # Override x and dx with the signal grid to be safe
    x = x_signal
    dx = (x[-1] - x[0]) / (len(x) - 1)
    
    # Identity mapping (no coordinate transform)
    xi = x
    dphi_exact = np.ones_like(x)
    
    # Function and exact derivatives ("original" = on physical grid)
    y = u.copy()
    y_original = u.copy()
    y_deriv_exact = u_x.copy()
    y_deriv_exact_original = u_x.copy()
    y_deriv2_exact_original = u_xx.copy()
    y_deriv2_exact = u_xx.copy()
    
    # Pre-compute random amplitudes for evaluating at Chebyshev nodes
    np.random.seed(TURB_SEED)
    n = np.arange(1, TURB_N_MODES + 1)
    k = n * np.pi / (domain[1] - domain[0])
    sigma = k**(-5/6)
    a_turb = sigma * np.random.randn(TURB_N_MODES)
    
    # Create functions to evaluate turbulence at arbitrary points
    def test_func_original(x_eval):
        u_eval, _, _ = evaluate_turbulence_at_points(
            x_eval, domain[1] - domain[0], TURB_N_MODES, TURB_SEED, a=a_turb
        )
        return u_eval
    
    def test_func_deriv_original(x_eval):
        _, u_x_eval, _ = evaluate_turbulence_at_points(
            x_eval, domain[1] - domain[0], TURB_N_MODES, TURB_SEED, a=a_turb
        )
        return u_x_eval
    
    def test_func_deriv2_original(x_eval):
        _, _, u_xx_eval = evaluate_turbulence_at_points(
            x_eval, domain[1] - domain[0], TURB_N_MODES, TURB_SEED, a=a_turb
        )
        return u_xx_eval
    
    # Compute derivatives using different methods
    # 1. BSPF method on uniform grid with enforced zero-flux Neumann BC
    # First, enforce zero-flux boundary conditions
    y_corrected = y_original.copy()
    f_left_corrected, f_right_corrected = model.enforced_zero_flux(y_original)
    y_corrected[0] = f_left_corrected
    y_corrected[-1] = f_right_corrected
    
    # Now differentiate with zero-flux Neumann BC
    y_deriv_bspf, y_deriv2_bspf, y_spline = model.differentiate_1_2(
        y_corrected, lam=REG_PARAM, neumann_bc=(0.0, 0.0)
    )
    # Explicitly enforce zero-flux at boundaries for first derivative
    y_deriv_bspf[0] = 0.0
    y_deriv_bspf[-1] = 0.0
    
    # 2. Chebyshev method on Chebyshev nodes using spectral-derivatives package
    N_cheb = NUM_POINTS  # Number of intervals (N+1 nodes)
    # Construct Chebyshev nodes: x_k = cos(k*π/N) for k=0,...,N, then map to domain
    x_cheb_canonical = np.cos(np.arange(N_cheb + 1) * np.pi / N_cheb)  # [-1, 1]
    # Map to domain [a, b]: x = (b-a)/2 * t + (b+a)/2
    a, b = domain
    x_cheb = x_cheb_canonical * (b - a) / 2.0 + (b + a) / 2.0
    f_vals = test_func_original(x_cheb)
    y_deriv_cheb = cheb_deriv(f_vals, x_cheb, order=1)
    y_deriv_cheb_exact = test_func_deriv_original(x_cheb)
    # Compute second derivative using spectral-derivatives package
    y_deriv2_cheb = cheb_deriv(f_vals, x_cheb, order=2)
    y_deriv2_cheb_exact = test_func_deriv2_original(x_cheb)
    
    # 3. 2nd-order finite difference
    y_deriv_fd = finite_difference_2nd_order(x, y_original)
    # Compute second derivative using 2nd-order finite difference
    y_deriv2_fd = finite_difference_2nd_order_2nd_deriv(x, y_original)
    
    # Compute errors for each method (L2 norm)
    error_bspf = np.linalg.norm(y_deriv_bspf - y_deriv_exact_original, 2) * np.sqrt(dx)
    dx_cheb = (domain[1] - domain[0]) / (len(x_cheb) - 1)  # Approximate grid spacing for Chebyshev
    error_cheb = np.linalg.norm(y_deriv_cheb - y_deriv_cheb_exact, 2) * np.sqrt(dx_cheb)
    error_fd = np.linalg.norm(y_deriv_fd - y_deriv_exact_original, 2) * np.sqrt(dx)
    
    # Second derivative errors
    error2_bspf = np.linalg.norm(y_deriv2_bspf - y_deriv2_exact_original, 2) * np.sqrt(dx)
    error2_cheb = np.linalg.norm(y_deriv2_cheb - y_deriv2_cheb_exact, 2) * np.sqrt(dx_cheb)
    error2_fd = np.linalg.norm(y_deriv2_fd - y_deriv2_exact_original, 2) * np.sqrt(dx)
    
    print("First Derivative Errors (L^2 Norm):")
    print("BSPF:", error_bspf)
    print("Chebyshev:", error_cheb)
    print("FD-2:", error_fd)
    print("\nSecond Derivative Errors (L^2 Norm):")
    print("BSPF:", error2_bspf)
    print("Chebyshev:", error2_cheb)
    print("FD-2:", error2_fd)
    
    # ========================================================================
    # Convergence Study
    # ========================================================================
    
    errors_bspf = []
    errors_cheb = []
    errors_fd = []
    errors2_bspf = []
    errors2_cheb = []
    errors2_fd = []
    
    for n_points in grid_sizes:
        # Create grid
        x_test = np.linspace(domain[0], domain[1], n_points)
        dx_test = (domain[1] - domain[0]) / (n_points - 1)
        
        # Compute exact solution
        y_test = test_func_original(x_test)
        y_deriv_exact_test = test_func_deriv_original(x_test)
        y_deriv2_exact_test = test_func_deriv2_original(x_test)
        
        # BSPF method on uniform grid with enforced zero-flux Neumann BC
        model_test = bspf1d.from_grid(
            degree=DEGREE,
            x=x_test,
            domain=tuple(domain),
            order=DEGREE,
            n_basis=N_BASIS,
            num_boundary_points=NUM_BOUNDARY_POINTS,
            use_clustering=clustering_flag,
            clustering_factor=clustering_factor
        )
        # Enforce zero-flux boundary conditions
        y_test_corrected = y_test.copy()
        f_left_corrected_test, f_right_corrected_test = model_test.enforced_zero_flux(y_test)
        y_test_corrected[0] = f_left_corrected_test
        y_test_corrected[-1] = f_right_corrected_test
        
        y_deriv_bspf_test, _ = model_test.differentiate(
            y_test_corrected, k=1, lam=REG_PARAM, neumann_bc=(0.0, 0.0)
        )
        # Explicitly enforce zero-flux at boundaries for first derivative
        y_deriv_bspf_test[0] = 0.0
        y_deriv_bspf_test[-1] = 0.0
        errors_bspf.append(
            np.linalg.norm(y_deriv_bspf_test - y_deriv_exact_test, 2) * np.sqrt(dx_test)
        )
        
        y_deriv2_bspf_test, _ = model_test.differentiate(
            y_test_corrected, k=2, lam=REG_PARAM, neumann_bc=(0.0, 0.0)
        )
        errors2_bspf.append(
            np.linalg.norm(y_deriv2_bspf_test - y_deriv2_exact_test, 2) * np.sqrt(dx_test)
        )
        
        # Chebyshev method on Chebyshev nodes using spectral-derivatives library
        N_cheb_test = n_points  # Number of intervals (N+1 nodes)
        x_cheb_canonical_test = np.cos(np.arange(N_cheb_test + 1) * np.pi / N_cheb_test)  # [-1, 1]
        # Map to domain [a, b]: x = (b-a)/2 * t + (b+a)/2
        a, b = domain
        x_cheb_test = x_cheb_canonical_test * (b - a) / 2.0 + (b + a) / 2.0
        f_vals_test = test_func_original(x_cheb_test)
        y_deriv_cheb_test = cheb_deriv(f_vals_test, x_cheb_test, order=1)
        y_deriv_cheb_exact_test = test_func_deriv_original(x_cheb_test)
        dx_cheb_test = (domain[1] - domain[0]) / (len(x_cheb_test) - 1)
        errors_cheb.append(
            np.linalg.norm(y_deriv_cheb_test - y_deriv_cheb_exact_test, 2) * np.sqrt(dx_cheb_test)
        )
        
        # Second derivative for Chebyshev using spectral-derivatives library
        y_deriv2_cheb_test = cheb_deriv(f_vals_test, x_cheb_test, order=2)
        y_deriv2_cheb_exact_test = test_func_deriv2_original(x_cheb_test)
        errors2_cheb.append(
            np.linalg.norm(y_deriv2_cheb_test - y_deriv2_cheb_exact_test, 2) * np.sqrt(dx_cheb_test)
        )
        
        # 2nd-order finite difference
        y_deriv_fd_test = finite_difference_2nd_order(x_test, y_test)
        errors_fd.append(
            np.linalg.norm(y_deriv_fd_test - y_deriv_exact_test, 2) * np.sqrt(dx_test)
        )
        
        # Second derivative for FD-2
        y_deriv2_fd_test = finite_difference_2nd_order_2nd_deriv(x_test, y_test)
        errors2_fd.append(
            np.linalg.norm(y_deriv2_fd_test - y_deriv2_exact_test, 2) * np.sqrt(dx_test)
        )
        
        print(f"N = {n_points:5d} | error1 fd2 = {errors_fd[-1]:.6e} | "
              f"error1 bspf = {errors_bspf[-1]:.6e} | error1 cheb = {errors_cheb[-1]:.6e}")
        print(f"      error2 fd2 = {errors2_fd[-1]:.6e} | "
              f"error2 bspf = {errors2_bspf[-1]:.6e} | error2 cheb = {errors2_cheb[-1]:.6e}")
    
    # ========================================================================
    # Visualization
    # ========================================================================
    
    # Set up global plotting parameters
    plt.rcParams.update({
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 16,
        'figure.titlesize': 24,
        'axes.grid': True,
        'grid.alpha': 0.5
    })
    
    # Plotting - 3x2 layout
    plt.figure(figsize=(16, 15))  # Increased height for 3 rows
    
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # (a) Original function and spline approximation
    plt.subplot(3, 2, 1)
    plt.plot(xi, y, '-', label='$f(x)$', linewidth=1)
    plt.plot(xi, y_spline, '-', label='$f_s(x)$', linewidth=1.5, alpha=1)
    plt.legend(ncol=1)
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel('$f(x)$', fontsize=20)
    plt.title('(a)', loc='left', x=-0.15, fontsize=24, fontweight='bold')
    
    # (b) First derivative comparison
    plt.subplot(3, 2, 2)
    plt.plot(x, y_deriv_exact_original, '-', color='k', label='Exact', linewidth=1)
    plt.plot(x, y_deriv_bspf, '-', label='BSPF', linewidth=1, markersize=4)
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel('$df/dx$', fontsize=20)
    plt.title('(b)', loc='left', x=-0.15, fontsize=24, fontweight='bold')
    plt.legend(ncol=1)
    
    # (c) Second derivative comparison
    plt.subplot(3, 2, 3)
    plt.plot(x, y_deriv2_exact_original, '-', color='k', label='Exact', linewidth=1)
    plt.plot(x, y_deriv2_bspf, '-', label='BSPF', linewidth=1, markersize=4)
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel('$d^2f/dx^2$', fontsize=20)
    plt.title('(c)', loc='left', x=-0.15, fontsize=24, fontweight='bold')
    plt.legend(ncol=1)
    
    # (d) First derivative errors
    plt.subplot(3, 2, 4)
    plt.semilogy(x, np.abs(y_deriv_bspf - y_deriv_exact_original), '-',
                 label='BSPF', color=default_colors[0], alpha=1)
    plt.semilogy(x_cheb, np.abs(y_deriv_cheb - y_deriv_cheb_exact), '-',
                 label='Chebyshev', color=default_colors[1], alpha=1)
    plt.semilogy(x, np.abs(y_deriv_fd - y_deriv_exact_original), '-',
                 label='FD-2', color=default_colors[2], alpha=1)
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel('$|Error|$', fontsize=20)
    plt.ylim(1e-15, 0.9*1e4)
    plt.legend(ncol=3)
    plt.title('(d)', loc='left', x=-0.15, fontsize=24, fontweight='bold')
    
    # (e) Second derivative errors
    plt.subplot(3, 2, 5)
    plt.semilogy(x, np.abs(y_deriv2_bspf - y_deriv2_exact_original), '-',
                 label='BSPF', color=default_colors[0], alpha=1)
    plt.semilogy(x_cheb, np.abs(y_deriv2_cheb - y_deriv2_cheb_exact), '-',
                 label='Chebyshev', color=default_colors[1], alpha=1)
    plt.semilogy(x, np.abs(y_deriv2_fd - y_deriv2_exact_original), '-',
                 label='FD-2', color=default_colors[2], alpha=1)
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel('$|Error|$', fontsize=20)
    plt.ylim(1e-15, 0.9*1e4)
    plt.legend(ncol=3)
    plt.title('(e)', loc='left', x=-0.15, fontsize=24, fontweight='bold')
    
    # (f) Convergence study
    plt.subplot(3, 2, 6)
    # Fit convergence rate for BSPF from entries 25 to 50
    fit_start, fit_end = 25, 50
    fit_start_fd, fit_end_fd = 25, 50
    
    x_fit = grid_sizes[fit_start:fit_end]
    x_fit_fd = grid_sizes[fit_start_fd:fit_end_fd]
    
    y_fit = np.array(errors_bspf[fit_start:fit_end])
    y_fit_fd = np.array(errors_fd[fit_start_fd:fit_end_fd])
    
    # Fit in log-log space: log(error) = a * log(N) + b
    log_x_fit = np.log(x_fit)
    log_x_fit_fd = np.log(x_fit_fd)
    
    log_y_fit = np.log(y_fit)
    log_y_fit_fd = np.log(y_fit_fd)
    
    coefficients = np.polyfit(log_x_fit, log_y_fit, 1)
    coefficients_fd = np.polyfit(log_x_fit_fd, log_y_fit_fd, 1)
    
    slope = coefficients[0]
    intercept = coefficients[1]
    
    slope_fd = coefficients_fd[0]
    intercept_fd = coefficients_fd[1]
    
    print(f"\nConvergence rates (first derivative):")
    print(f"BSPF slope: {slope:.3f}, intercept: {intercept:.3f}")
    print(f"FD-2 slope: {slope_fd:.3f}, intercept: {intercept_fd:.3f}")
    
    # Generate fitted line
    x_fit_line = np.linspace(x_fit[0], x_fit[-1], 100)
    y_fit_line = np.exp(intercept) * x_fit_line**slope
    
    x_fit_line_fd = np.linspace(x_fit_fd[0], x_fit_fd[-1], 100)
    y_fit_line_fd = np.exp(intercept_fd) * x_fit_line_fd**slope_fd
    
    # First derivative convergence
    plt.loglog(grid_sizes, errors_bspf, '.-', linewidth=1, label='BSPF (1st)',
               color=default_colors[0], alpha=1)
    plt.loglog(grid_sizes, errors_cheb, '.-', linewidth=1, label='Chebyshev (1st)',
               color=default_colors[1], alpha=1)
    plt.loglog(grid_sizes, errors_fd, '.-', linewidth=1, label='FD-2 (1st)',
               color=default_colors[2], alpha=1)
    
    # Second derivative convergence
    plt.loglog(grid_sizes, errors2_bspf, 's-', linewidth=1, label='BSPF (2nd)',
               color=default_colors[0], alpha=0.7, markersize=4)
    plt.loglog(grid_sizes, errors2_cheb, 's-', linewidth=1, label='Chebyshev (2nd)',
               color=default_colors[1], alpha=0.7, markersize=4)
    plt.loglog(grid_sizes, errors2_fd, 's-', linewidth=1, label='FD-2 (2nd)',
               color=default_colors[2], alpha=0.7, markersize=4)
    
    plt.text(2010, 2e-12, '$(a)-(e)$', color='gray', fontsize=18)
    plt.loglog(x_fit_line, 0.2*y_fit_line, '--', linewidth=2, color=default_colors[0])
    plt.loglog(x_fit_line_fd, 5*y_fit_line_fd, '--', linewidth=2, color=default_colors[2])
    plt.xlabel('$N$', fontsize=20)
    plt.ylabel('$\|Error\|_2$', fontsize=20)
    plt.title('(f)', loc='left', x=-0.15, fontsize=24, fontweight='bold')
    plt.grid(True)
    plt.legend(ncol=2, fontsize=12)
    plt.ylim(1e-12, 0.9*1e6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

