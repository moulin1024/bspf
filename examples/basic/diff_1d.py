"""
1D Derivative Accuracy and Convergence Comparison.

This script compares BSPF, Chebyshev spectral, and 2nd-order finite difference
methods for computing first derivatives of smooth functions. It generates a 2x2
figure showing:
    (a) Original function and B-spline approximation
    (b) Exact derivative vs BSPF approximation
    (c) Pointwise absolute errors for all methods
    (d) Convergence study showing error vs grid resolution

Run from repository root:
    python examples/basic/diff_1d.py
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from bspf import bspf1d
from specderiv import cheb_deriv


# ============================================================================
# Parameters
# ============================================================================

# BSPF parameters
DEGREE = 7                   # B-spline polynomial degree
NUM_BOUNDARY_POINTS = DEGREE # Number of boundary points
N_BASIS = 2 * DEGREE         # Number of basis functions
REG_PARAM = 1e-3             # Tikhonov regularization strength (lambda)

# Grid parameters
DOMAIN = [0, 2*np.pi]        # Domain [a, b]
NUM_POINTS = 4300            # Grid resolution for main computation
CLUSTERING_FACTOR = 2.0      # Grid clustering factor near endpoints
USE_CLUSTERING = True         # Enable grid clustering

# Convergence study parameters
GRID_SIZES = np.geomspace(500, 5000, 50).astype(int)  # Grid sizes for convergence study
N_FIT_POINTS = 15 # Number of last points to use for convergence rate fitting


# ============================================================================
# Finite Difference Function
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


# ============================================================================

# Test Function Definition
# ============================================================================

def define_test_function():
    """
    Define the test function and its analytical derivatives using SymPy.
    
    Returns
    -------
    func : callable
        Function f(x) as a NumPy-compatible function
    func_deriv : callable
        First derivative f'(x) as a NumPy-compatible function
    func_deriv2 : callable
        Second derivative f''(x) as a NumPy-compatible function
    """
    t = sp.Symbol('t')
    
    # Test function: sin(t / (1.01 + cos(t)))
    # This is a smooth, non-periodic function with varying frequency
    f_sym = sp.tanh(100*(t-np.pi/2)) - sp.tanh(100*(t-3*np.pi/2))# sp.sin(t / (1.01 + sp.cos(t)))
    df_sym = sp.diff(f_sym, t)
    d2f_sym = sp.diff(df_sym, t)
    
    # Convert to NumPy functions
    func = sp.lambdify(t, f_sym, modules='numpy')
    func_deriv = sp.lambdify(t, df_sym, modules='numpy')
    func_deriv2 = sp.lambdify(t, d2f_sym, modules='numpy')
    
    return func, func_deriv, func_deriv2


# ============================================================================
# Main Computation
# ============================================================================

def main():
    """Main computation and visualization."""
    
    # Define test function
    test_func, test_func_deriv, test_func_deriv2 = define_test_function()
    
    # Create grid
    x = np.linspace(DOMAIN[0], DOMAIN[1], NUM_POINTS, endpoint=True)
    
    # Initialize BSPF model
    model = bspf1d.from_grid(
        degree=DEGREE,
        x=x,
        domain=tuple(DOMAIN),
        order=DEGREE,
        n_basis=N_BASIS,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        use_clustering=USE_CLUSTERING,
        clustering_factor=CLUSTERING_FACTOR
    )
    
    # Compute function values and exact derivatives
    y = test_func(x)
    y_deriv_exact = test_func_deriv(x)
    y_deriv2_exact = test_func_deriv2(x)
    
    # Compute derivatives using different methods
    # 1. BSPF method
    y_deriv_bspf, y_deriv2_bspf, y_spline = model.differentiate_1_2(y, lam=REG_PARAM)
    
    # 2. Chebyshev method (on Chebyshev nodes) using spectral-derivatives package
    N_cheb = NUM_POINTS  # Number of intervals (N+1 nodes)
    # Construct Chebyshev nodes: x_k = cos(k*π/N) for k=0,...,N, then map to domain
    x_cheb_canonical = np.cos(np.arange(N_cheb + 1) * np.pi / N_cheb)  # [-1, 1]
    # Map to domain [a, b]: x = (b-a)/2 * t + (b+a)/2
    a, b = DOMAIN
    x_cheb = x_cheb_canonical * (b - a) / 2.0 + (b + a) / 2.0
    y_cheb = test_func(x_cheb)
    y_deriv_cheb = cheb_deriv(y_cheb, x_cheb, order=1)
    y_deriv_cheb_exact = test_func_deriv(x_cheb)
    y_deriv2_cheb = cheb_deriv(y_cheb, x_cheb, order=2)
    y_deriv2_cheb_exact = test_func_deriv2(x_cheb)
    
    # 3. 2nd-order finite difference method
    y_deriv_fd = finite_difference_2nd_order(x, y)
    y_deriv2_fd = finite_difference_2nd_order_2nd_deriv(x, y)
    
    # Compute errors (L2 norm)
    error_bspf = np.sqrt(np.mean((y_deriv_bspf - y_deriv_exact)**2))
    error_cheb = np.sqrt(np.mean((y_deriv_cheb - y_deriv_cheb_exact)**2))
    error_fd = np.sqrt(np.mean((y_deriv_fd - y_deriv_exact)**2))
    
    # Second derivative errors
    error2_bspf = np.sqrt(np.mean((y_deriv2_bspf - y_deriv2_exact)**2))
    error2_cheb = np.sqrt(np.mean((y_deriv2_cheb - y_deriv2_cheb_exact)**2))
    error2_fd = np.sqrt(np.mean((y_deriv2_fd - y_deriv2_exact)**2))
    
    print("=" * 60)
    print("First Derivative Errors (L² Norm):")
    print(f"  BSPF:      {error_bspf:.6e}")
    print(f"  Chebyshev: {error_cheb:.6e}")
    print(f"  FD-2:      {error_fd:.6e}")
    print("\nSecond Derivative Errors (L² Norm):")
    print(f"  BSPF:      {error2_bspf:.6e}")
    print(f"  Chebyshev: {error2_cheb:.6e}")
    print(f"  FD-2:      {error2_fd:.6e}")
    print("=" * 60)
    
    # ========================================================================
    # Convergence Study
    # ========================================================================
    
    print("\nRunning convergence study...")
    errors_bspf = []
    errors_cheb = []
    errors_fd = []
    errors2_bspf = []
    errors2_cheb = []
    errors2_fd = []
    
    for n_points in GRID_SIZES:
        # Create grid
        x_test = np.linspace(DOMAIN[0], DOMAIN[1], n_points, endpoint=True)
        
        # Compute exact solution
        y_test = test_func(x_test)
        y_deriv_exact_test = test_func_deriv(x_test)
        y_deriv2_exact_test = test_func_deriv2(x_test)
        
        # BSPF method
        model_test = bspf1d.from_grid(
            degree=DEGREE,
            x=x_test,
            domain=tuple(DOMAIN),
            order=DEGREE,
            n_basis=N_BASIS,
            num_boundary_points=NUM_BOUNDARY_POINTS,
            use_clustering=USE_CLUSTERING,
            clustering_factor=CLUSTERING_FACTOR
        )
        y_deriv_bspf_test, y_deriv2_bspf_test, _ = model_test.differentiate_1_2(y_test, lam=REG_PARAM)
        error_bspf_test = np.max(np.abs(y_deriv_bspf_test - y_deriv_exact_test))
        errors_bspf.append(error_bspf_test)
        error2_bspf_test = np.max(np.abs(y_deriv2_bspf_test - y_deriv2_exact_test))
        errors2_bspf.append(error2_bspf_test)
        
        # Chebyshev method using spectral-derivatives package
        N_cheb_test = n_points  # Number of intervals (N+1 nodes)
        # Construct Chebyshev nodes: x_k = cos(k*π/N) for k=0,...,N, then map to domain
        x_cheb_canonical_test = np.cos(np.arange(N_cheb_test + 1) * np.pi / N_cheb_test)  # [-1, 1]
        # Map to domain [a, b]: x = (b-a)/2 * t + (b+a)/2
        a, b = DOMAIN
        x_cheb_test = x_cheb_canonical_test * (b - a) / 2.0 + (b + a) / 2.0
        y_cheb_test = test_func(x_cheb_test)
        y_deriv_cheb_test = cheb_deriv(y_cheb_test, x_cheb_test, order=1)
        y_deriv_cheb_exact_test = test_func_deriv(x_cheb_test)
        error_cheb_test = np.max(np.abs(y_deriv_cheb_test - y_deriv_cheb_exact_test))
        errors_cheb.append(error_cheb_test)
        
        y_deriv2_cheb_test = cheb_deriv(y_cheb_test, x_cheb_test, order=2)
        y_deriv2_cheb_exact_test = test_func_deriv2(x_cheb_test)
        error2_cheb_test = np.max(np.abs(y_deriv2_cheb_test - y_deriv2_cheb_exact_test))
        errors2_cheb.append(error2_cheb_test)
        
        # 2nd-order finite difference method
        y_deriv_fd_test = finite_difference_2nd_order(x_test, y_test)
        error_fd_test = np.max(np.abs(y_deriv_fd_test - y_deriv_exact_test))
        errors_fd.append(error_fd_test)
        
        y_deriv2_fd_test = finite_difference_2nd_order_2nd_deriv(x_test, y_test)
        error2_fd_test = np.max(np.abs(y_deriv2_fd_test - y_deriv2_exact_test))
        errors2_fd.append(error2_fd_test)
        
        print(f"N = {n_points:5d} | 1st: BSPF: {error_bspf_test:.6e} | Cheb: {error_cheb_test:.6e} | FD-2: {error_fd_test:.6e}")
        print(f"      | 2nd: BSPF: {error2_bspf_test:.6e} | Cheb: {error2_cheb_test:.6e} | FD-2: {error2_fd_test:.6e}")
    
    # Compute convergence rate for BSPF (using last N_FIT_POINTS)
    n_fit = min(N_FIT_POINTS, len(GRID_SIZES))
    fit_start = len(GRID_SIZES) - n_fit
    x_fit = GRID_SIZES[fit_start:]
    y_fit_bspf = np.array(errors_bspf[fit_start:])
    log_x_fit = np.log(x_fit)
    log_y_fit_bspf = np.log(y_fit_bspf)
    coefficients_bspf = np.polyfit(log_x_fit, log_y_fit_bspf, 1)
    slope_bspf = coefficients_bspf[0]
    intercept_bspf = coefficients_bspf[1]
    
    # Compute convergence rate for finite difference
    y_fit_fd = np.array(errors_fd[fit_start:])
    log_y_fit_fd = np.log(y_fit_fd)
    coefficients_fd = np.polyfit(log_x_fit, log_y_fit_fd, 1)
    slope_fd = coefficients_fd[0]
    intercept_fd = coefficients_fd[1]
    
    # Second derivative convergence rates
    y_fit2_bspf = np.array(errors2_bspf[fit_start:])
    log_y_fit2_bspf = np.log(y_fit2_bspf)
    coefficients2_bspf = np.polyfit(log_x_fit, log_y_fit2_bspf, 1)
    slope2_bspf = coefficients2_bspf[0]
    
    y_fit2_fd = np.array(errors2_fd[fit_start:])
    log_y_fit2_fd = np.log(y_fit2_fd)
    coefficients2_fd = np.polyfit(log_x_fit, log_y_fit2_fd, 1)
    slope2_fd = coefficients2_fd[0]
    
    print(f"\nConvergence rates (from last {n_fit} points):")
    print(f"  First derivative:")
    print(f"    BSPF:      {slope_bspf:.3f}")
    print(f"    FD-2:      {slope_fd:.3f} (expected: -2.0 for O(h^2))")
    print(f"  Second derivative:")
    print(f"    BSPF:      {slope2_bspf:.3f}")
    print(f"    FD-2:      {slope2_fd:.3f} (expected: -2.0 for O(h^2))")
    
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
    
    fig = plt.figure(figsize=(18, 10))
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # (a) Original function and spline approximation
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(x, y, '-', label='$f(x)$', linewidth=1)
    ax1.plot(x, y_spline, '-', label='$f_s(x)$', linewidth=1, alpha=1)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$f(x)$')
    ax1.legend(ncol=1)
    ax1.set_title('(a)', loc='left', x=-0.15, fontsize=16, fontweight='bold')
    
    # (b) First and second derivatives comparison (dual y-axis)
    ax2 = plt.subplot(2, 2, 2)
    
    # Left y-axis: First derivative
    ax2.plot(x, y_deriv_exact, '-', color='k', label='Exact (1st)', linewidth=1)
    ax2.plot(x, y_deriv_bspf, '-', label='BSPF (1st)', linewidth=1, alpha=0.8, color=default_colors[0])
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$df/dx$', color=default_colors[0], fontsize=12)
    ax2.tick_params(axis='y', labelcolor=default_colors[0])
    
    # Right y-axis: Second derivative
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, y_deriv2_exact, '--', color='k', label='Exact (2nd)', linewidth=1)
    ax2_twin.plot(x, y_deriv2_bspf, '--', label='BSPF (2nd)', linewidth=1, alpha=0.8, color=default_colors[1])
    ax2_twin.set_ylabel('$d^2f/dx^2$', color=default_colors[1], fontsize=12)
    ax2_twin.tick_params(axis='y', labelcolor=default_colors[1])
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best', ncol=1, fontsize=9)
    ax2.set_title('(b) Derivatives Comparison', loc='left', x=-0.15, fontsize=16, fontweight='bold')
    
    # (c) First and second derivative errors (merged)
    ax3 = plt.subplot(2, 2, 3)
    ax3.semilogy(x, np.abs(y_deriv_bspf - y_deriv_exact), '-', 
                label='BSPF (1st)', color=default_colors[0], alpha=1, linewidth=1.5)
    ax3.semilogy(x_cheb, np.abs(y_deriv_cheb - y_deriv_cheb_exact), '-', 
                label='Chebyshev (1st)', color=default_colors[1], alpha=1, linewidth=1.5)
    ax3.semilogy(x, np.abs(y_deriv_fd - y_deriv_exact), '-', 
                label='FD-2 (1st)', color=default_colors[2], alpha=1, linewidth=1.5)
    ax3.semilogy(x, np.abs(y_deriv2_bspf - y_deriv2_exact), '--', 
                label='BSPF (2nd)', color=default_colors[0], alpha=0.7, linewidth=1.5)
    ax3.semilogy(x_cheb, np.abs(y_deriv2_cheb - y_deriv2_cheb_exact), '--', 
                label='Chebyshev (2nd)', color=default_colors[1], alpha=0.7, linewidth=1.5)
    ax3.semilogy(x, np.abs(y_deriv2_fd - y_deriv2_exact), '--', 
                label='FD-2 (2nd)', color=default_colors[2], alpha=0.7, linewidth=1.5)
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$|Error|$')
    ax3.set_ylim(1e-15, 0.9*1e6)
    ax3.legend(ncol=3, fontsize=9)
    ax3.set_title('(c) Derivative Errors (1st & 2nd)', loc='left', x=-0.15, fontsize=16, fontweight='bold')
    
    # (d) First and second derivative convergence (merged)
    ax4 = plt.subplot(2, 2, 4)
    ax4.loglog(GRID_SIZES, errors_bspf, '.-', linewidth=1.5, 
               label='BSPF (1st)', color=default_colors[0], alpha=1, markersize=5)
    ax4.loglog(GRID_SIZES, errors_cheb, '.-', linewidth=1.5, 
               label='Chebyshev (1st)', color=default_colors[1], alpha=1, markersize=5)
    ax4.loglog(GRID_SIZES, errors_fd, '.-', linewidth=1.5, 
               label='FD-2 (1st)', color=default_colors[2], alpha=1, markersize=5)
    ax4.loglog(GRID_SIZES, errors2_bspf, 's-', linewidth=1.5, 
               label='BSPF (2nd)', color=default_colors[0], alpha=0.7, markersize=5)
    ax4.loglog(GRID_SIZES, errors2_cheb, 's-', linewidth=1.5, 
               label='Chebyshev (2nd)', color=default_colors[1], alpha=0.7, markersize=5)
    ax4.loglog(GRID_SIZES, errors2_fd, 's-', linewidth=1.5, 
               label='FD-2 (2nd)', color=default_colors[2], alpha=0.7, markersize=5)
    
    # Add fitted lines from log-log fit
    err_fit_fd = np.exp(intercept_fd) * (x_fit ** slope_fd)
    slope_str_fd = f'{slope_fd:.1f}'
    ax4.loglog(x_fit, 0.5*err_fit_fd, '--', linewidth=1, 
               label=f'O($N^{{{slope_str_fd}}}$) (1st)', 
               color=default_colors[2], alpha=0.5)
    
    # Add fitted lines for second derivative
    err_fit2_fd = np.exp(coefficients2_fd[1]) * (x_fit ** slope2_fd)
    slope_str2_fd = f'{slope2_fd:.1f}'
    ax4.loglog(x_fit, 0.5*err_fit2_fd, ':', linewidth=1, 
               label=f'O($N^{{{slope_str2_fd}}}$) (2nd)', 
               color=default_colors[2], alpha=0.5)
    
    # Mark the grid size used
    ax4.axvline(NUM_POINTS, linestyle='--', color='gray', linewidth=1)
    ax4.text(NUM_POINTS + 10, 2e-11, '$(a)-(c)$', color='gray', fontsize=12)
    
    ax4.set_xlabel('$N$')
    ax4.set_ylabel('$\\|Error\\|_\\infty$')
    ax4.set_ylim(1e-11, 0.9*1e5)
    ax4.legend(ncol=2, fontsize=9)
    ax4.set_title('(d) Convergence Study (1st & 2nd)', loc='left', x=-0.15, fontsize=16, fontweight='bold')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
