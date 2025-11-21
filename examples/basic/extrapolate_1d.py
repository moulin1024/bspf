"""
1D Extrapolation Accuracy Test.

This script tests BSPF extrapolation by half grid size beyond domain boundaries
and compares with the analytical solution. It generates a 2x2 figure showing:
    (a) Original function and B-spline approximation with extrapolation
    (b) Extrapolation points and analytical comparison
    (c) Extrapolation errors at left and right boundaries
    (d) Convergence study of extrapolation errors vs grid resolution

Run from repository root:
    python examples/basic/extrapolate_1d.py
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from bspf import bspf1d
from bspf.utils import chebyshev_derivative_from_values, construct_chebyshev_nodes
import scipy.linalg as sla


# ============================================================================
# Parameters
# ============================================================================

# BSPF parameters
DEGREE = 5                    # B-spline polynomial degree
NUM_BOUNDARY_POINTS = DEGREE  # Number of boundary points
N_BASIS = 2 * DEGREE         # Number of basis functions
REG_PARAM = 1e-3             # Tikhonov regularization strength (lambda)

# Grid parameters
DOMAIN = [0, 2*np.pi]        # Domain [a, b]
NUM_POINTS = 4300            # Grid resolution for main computation
CLUSTERING_FACTOR = 2.0      # Grid clustering factor near endpoints
USE_CLUSTERING = True         # Enable grid clustering

# Convergence study parameters
GRID_SIZES = np.geomspace(1000, 10000, 50).astype(int)  # Grid sizes for convergence study
FIT_START, FIT_END = 25, 30  # Indices for convergence rate fitting

# Extrapolation parameters
EXTRAPOLATION_FACTOR = 0.5  # Extrapolate by half grid size


# ============================================================================
# Test Function Definition
# ============================================================================

def define_test_function():
    """
    Define the test function and its analytical derivative using SymPy.
    
    Returns
    -------
    func : callable
        Function f(x) as a NumPy-compatible function
    func_deriv : callable
        Derivative f'(x) as a NumPy-compatible function
    """
    t = sp.Symbol('t')
    
    # Test function: sin(t / (1.01 + cos(t)))
    # This is a smooth, non-periodic function with varying frequency
    f_sym = sp.sin(t / (1.01 + sp.cos(t)))
    df_sym = sp.diff(f_sym, t)
    
    # Convert to NumPy functions
    func = sp.lambdify(t, f_sym, modules='numpy')
    func_deriv = sp.lambdify(t, df_sym, modules='numpy')
    
    return func, func_deriv


# ============================================================================
# Helper Functions
# ============================================================================

def get_spline_coefficients(model, y, lam=0.0):
    """
    Get B-spline coefficients P from the model by solving the KKT system.
    
    Parameters
    ----------
    model : bspf1d
        BSPF model instance
    y : Array
        Function values on the grid
    lam : float
        Regularization parameter
        
    Returns
    -------
    P : Array
        B-spline coefficients
    """
    # Build RHS (same as in differentiate method)
    rhs_2bw = 2.0 * (model.BW @ y)
    dY = model.end.BND @ y
    rhs = np.concatenate((rhs_2bw, dY))
    
    # Solve KKT system
    lu, piv = model._kkt_lu(lam)
    sol = sla.lu_solve((lu, piv), rhs, overwrite_b=False)
    P = sol[:model.basis.B0.shape[0]]
    
    return P


def evaluate_spline_at_points(model, P, x_eval):
    """
    Evaluate B-spline at arbitrary points using coefficients.
    
    Parameters
    ----------
    model : bspf1d
        BSPF model instance
    P : Array
        B-spline coefficients
    x_eval : Array
        Points at which to evaluate the spline
        
    Returns
    -------
    y_eval : Array
        Spline values at x_eval
    """
    # Evaluate basis functions at x_eval
    B_eval = model.basis._evaluate_splines_vectorized(x_eval, deriv_order=0)
    
    # Evaluate spline: y = B^T @ P
    y_eval = B_eval.T @ P
    
    return y_eval


# ============================================================================
# Main Computation
# ============================================================================

def main():
    """Main computation and visualization."""
    
    # Define test function
    test_func, test_func_deriv = define_test_function()
    
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
    
    # Compute function values and exact derivative
    y = test_func(x)
    y_deriv_exact = test_func_deriv(x)
    
    # Compute derivatives and spline approximation
    y_deriv_bspf, _, y_spline = model.differentiate_1_2(y, lam=REG_PARAM)
    
    # Get spline coefficients for extrapolation
    P = get_spline_coefficients(model, y, lam=REG_PARAM)
    
    # ========================================================================
    # Extrapolation
    # ========================================================================
    
    # Calculate grid spacing
    dx = x[1] - x[0]
    extrapolation_distance = EXTRAPOLATION_FACTOR * dx
    
    # Create extrapolation points (half grid size beyond boundaries)
    x_left_extrap = DOMAIN[0] - extrapolation_distance
    x_right_extrap = DOMAIN[1] + extrapolation_distance
    
    x_extrap = np.array([x_left_extrap, x_right_extrap])
    
    # Evaluate spline at extrapolation points
    y_extrap = evaluate_spline_at_points(model, P, x_extrap)
    
    # Get analytical values at extrapolation points
    y_extrap_exact = test_func(x_extrap)
    
    # Compute extrapolation errors
    error_left = np.abs(y_extrap[0] - y_extrap_exact[0])
    error_right = np.abs(y_extrap[1] - y_extrap_exact[1])
    error_max = np.max([error_left, error_right])
    
    print("=" * 60)
    print("Extrapolation Results (half grid size):")
    print(f"  Left boundary:")
    print(f"    x = {x_left_extrap:.8f}, y_extrap = {y_extrap[0]:.8f}, y_exact = {y_extrap_exact[0]:.8f}")
    print(f"    Error = {error_left:.6e}")
    print(f"  Right boundary:")
    print(f"    x = {x_right_extrap:.8f}, y_extrap = {y_extrap[1]:.8f}, y_exact = {y_extrap_exact[1]:.8f}")
    print(f"    Error = {error_right:.6e}")
    print(f"  Max error = {error_max:.6e}")
    print("=" * 60)
    
    # ========================================================================
    # Convergence Study for Extrapolation
    # ========================================================================
    
    print("\nRunning extrapolation convergence study...")
    errors_extrap_left = []
    errors_extrap_right = []
    errors_extrap_max = []
    
    for n_points in GRID_SIZES:
        # Create grid
        x_test = np.linspace(DOMAIN[0], DOMAIN[1], n_points, endpoint=True)
        
        # Compute function values
        y_test = test_func(x_test)
        
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
        
        # Get spline coefficients
        P_test = get_spline_coefficients(model_test, y_test, lam=REG_PARAM)
        
        # Calculate extrapolation points
        dx_test = x_test[1] - x_test[0]
        extrap_dist_test = EXTRAPOLATION_FACTOR * dx_test
        x_left_test = DOMAIN[0] - extrap_dist_test
        x_right_test = DOMAIN[1] + extrap_dist_test
        x_extrap_test = np.array([x_left_test, x_right_test])
        
        # Evaluate spline at extrapolation points
        y_extrap_test = evaluate_spline_at_points(model_test, P_test, x_extrap_test)
        
        # Get analytical values
        y_extrap_exact_test = test_func(x_extrap_test)
        
        # Compute errors
        error_left_test = np.abs(y_extrap_test[0] - y_extrap_exact_test[0])
        error_right_test = np.abs(y_extrap_test[1] - y_extrap_exact_test[1])
        error_max_test = np.max([error_left_test, error_right_test])
        
        errors_extrap_left.append(error_left_test)
        errors_extrap_right.append(error_right_test)
        errors_extrap_max.append(error_max_test)
        
        print(f"N = {n_points:5d} | Left: {error_left_test:.6e} | Right: {error_right_test:.6e} | Max: {error_max_test:.6e}")
    
    # Compute convergence rate
    x_fit = GRID_SIZES[FIT_START:FIT_END]
    y_fit = np.array(errors_extrap_max[FIT_START:FIT_END])
    log_x_fit = np.log(x_fit)
    log_y_fit = np.log(y_fit)
    coefficients = np.polyfit(log_x_fit, log_y_fit, 1)
    slope = coefficients[0]
    intercept = coefficients[1]
    
    print(f"\nExtrapolation convergence rate (from indices {FIT_START}-{FIT_END}): {slope:.3f}")
    
    # ========================================================================
    # Visualization
    # ========================================================================
    
    # Set up plotting parameters
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
    
    fig = plt.figure(figsize=(16, 10))
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Evaluate spline on extended domain for visualization
    x_vis = np.linspace(x_left_extrap, x_right_extrap, 500)
    y_spline_vis = evaluate_spline_at_points(model, P, x_vis)
    y_exact_vis = test_func(x_vis)
    
    # (a) Original function and spline approximation with extrapolation
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(x, y, '-', label='$f(x)$ (domain)', linewidth=2, color='black')
    ax1.plot(x_vis, y_spline_vis, '--', label='$f_s(x)$ (spline)', linewidth=1.5, alpha=0.8)
    ax1.plot(x_vis, y_exact_vis, '-', label='$f(x)$ (exact)', linewidth=1, alpha=0.6, color='gray')
    ax1.axvline(DOMAIN[0], linestyle=':', color='red', linewidth=1, alpha=0.7, label='Domain boundaries')
    ax1.axvline(DOMAIN[1], linestyle=':', color='red', linewidth=1, alpha=0.7)
    ax1.plot(x_extrap, y_extrap, 'o', label='Extrapolated', markersize=8, color='red', zorder=5)
    ax1.plot(x_extrap, y_extrap_exact, 'x', label='Exact at extrap', markersize=10, color='blue', zorder=5)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$f(x)$')
    ax1.legend(ncol=2, fontsize=12)
    ax1.set_title('(a)', loc='left', x=-0.15, fontsize=24, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # (b) Zoom on extrapolation regions
    ax2 = plt.subplot(2, 2, 2)
    # Left boundary zoom
    x_left_zoom = np.linspace(x_left_extrap - 0.1*extrapolation_distance, 
                               DOMAIN[0] + 0.1*extrapolation_distance, 200)
    y_left_spline = evaluate_spline_at_points(model, P, x_left_zoom)
    y_left_exact = test_func(x_left_zoom)
    ax2.plot(x_left_zoom, y_left_exact, '-', label='Exact (left)', linewidth=2, color='black')
    ax2.plot(x_left_zoom, y_left_spline, '--', label='Spline (left)', linewidth=1.5, alpha=0.8)
    ax2.axvline(DOMAIN[0], linestyle=':', color='red', linewidth=1, alpha=0.7)
    ax2.plot(x_left_extrap, y_extrap[0], 'o', markersize=8, color='red', zorder=5)
    ax2.plot(x_left_extrap, y_extrap_exact[0], 'x', markersize=10, color='blue', zorder=5)
    
    # Right boundary zoom
    x_right_zoom = np.linspace(DOMAIN[1] - 0.1*extrapolation_distance,
                                x_right_extrap + 0.1*extrapolation_distance, 200)
    y_right_spline = evaluate_spline_at_points(model, P, x_right_zoom)
    y_right_exact = test_func(x_right_zoom)
    ax2.plot(x_right_zoom, y_right_exact, '-', label='Exact (right)', linewidth=2, color='gray')
    ax2.plot(x_right_zoom, y_right_spline, '--', label='Spline (right)', linewidth=1.5, alpha=0.8, color='orange')
    ax2.axvline(DOMAIN[1], linestyle=':', color='red', linewidth=1, alpha=0.7)
    ax2.plot(x_right_extrap, y_extrap[1], 'o', markersize=8, color='red', zorder=5)
    ax2.plot(x_right_extrap, y_extrap_exact[1], 'x', markersize=10, color='blue', zorder=5)
    
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$f(x)$')
    ax2.legend(ncol=2, fontsize=12)
    ax2.set_title('(b)', loc='left', x=-0.15, fontsize=24, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # (c) Extrapolation errors
    ax3 = plt.subplot(2, 2, 3)
    ax3.semilogy(GRID_SIZES, errors_extrap_left, '.-', linewidth=1.5, 
                label='Left boundary', color=default_colors[0], alpha=1, markersize=6)
    ax3.semilogy(GRID_SIZES, errors_extrap_right, '.-', linewidth=1.5, 
                label='Right boundary', color=default_colors[1], alpha=1, markersize=6)
    ax3.semilogy(GRID_SIZES, errors_extrap_max, '.-', linewidth=2, 
                label='Max error', color='black', alpha=0.8, markersize=6)
    ax3.axvline(NUM_POINTS, linestyle='--', color='gray', linewidth=1)
    ax3.text(NUM_POINTS + 10, errors_extrap_max[0] * 0.5, '$(a)-(b)$', color='gray', fontsize=18)
    ax3.set_xlabel('$N$')
    ax3.set_ylabel('$|Error|$')
    ax3.legend(ncol=1, fontsize=14)
    ax3.set_title('(c)', loc='left', x=-0.15, fontsize=24, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # (d) Convergence rate analysis
    ax4 = plt.subplot(2, 2, 4)
    ax4.loglog(GRID_SIZES, errors_extrap_max, '.-', linewidth=2, 
               label='Max extrapolation error', color='black', alpha=0.8, markersize=6)
    
    # Plot fitted line
    x_fit_plot = GRID_SIZES[FIT_START:FIT_END]
    y_fit_plot = np.exp(intercept) * x_fit_plot ** slope
    ax4.loglog(x_fit_plot, y_fit_plot, '--', linewidth=2, 
               label=f'Fit: $\\sim N^{{{slope:.2f}}}$', color='red', alpha=0.7)
    
    ax4.axvline(NUM_POINTS, linestyle='--', color='gray', linewidth=1)
    ax4.text(NUM_POINTS + 10, errors_extrap_max[0] * 0.5, '$(a)-(b)$', color='gray', fontsize=18)
    
    ax4.set_xlabel('$N$')
    ax4.set_ylabel('$\\|Error\\|_\\infty$')
    ax4.legend(ncol=1, fontsize=14)
    ax4.set_title('(d)', loc='left', x=-0.15, fontsize=24, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
