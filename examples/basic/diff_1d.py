"""
1D Derivative Accuracy and Convergence Comparison.

This script compares BSPF and Chebyshev spectral methods for computing first
derivatives of smooth functions. It generates a 2x2 figure showing:
    (a) Original function and B-spline approximation
    (b) Exact derivative vs BSPF approximation
    (c) Pointwise absolute errors for both methods
    (d) Convergence study showing error vs grid resolution

Run from repository root:
    python examples/basic/diff_1d.py
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from bspf import bspf1d
from bspf.utils import chebyshev_derivative_from_values, construct_chebyshev_nodes


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
    
    # Compute derivatives using different methods
    # 1. BSPF method
    y_deriv_bspf, _, y_spline = model.differentiate_1_2(y, lam=REG_PARAM)
    
    # 2. Chebyshev method (on Chebyshev nodes)
    x_cheb, _ = construct_chebyshev_nodes(NUM_POINTS, DOMAIN)
    y_cheb = test_func(x_cheb)
    y_deriv_cheb = chebyshev_derivative_from_values(y_cheb, x_cheb, DOMAIN)
    y_deriv_cheb_exact = test_func_deriv(x_cheb)
    
    # Compute errors (L2 norm)
    error_bspf = np.sqrt(np.mean((y_deriv_bspf - y_deriv_exact)**2))
    error_cheb = np.sqrt(np.mean((y_deriv_cheb - y_deriv_cheb_exact)**2))
    
    print("=" * 60)
    print("Errors (L² Norm):")
    print(f"  BSPF:      {error_bspf:.6e}")
    print(f"  Chebyshev: {error_cheb:.6e}")
    print("=" * 60)
    
    # ========================================================================
    # Convergence Study
    # ========================================================================
    
    print("\nRunning convergence study...")
    errors_bspf = []
    errors_cheb = []
    
    for n_points in GRID_SIZES:
        # Create grid
        x_test = np.linspace(DOMAIN[0], DOMAIN[1], n_points)
        
        # Compute exact solution
        y_test = test_func(x_test)
        y_deriv_exact_test = test_func_deriv(x_test)
        
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
        y_deriv_bspf_test, _ = model_test.differentiate(y_test, k=1, lam=REG_PARAM)
        error_bspf_test = np.max(np.abs(y_deriv_bspf_test - y_deriv_exact_test))
        errors_bspf.append(error_bspf_test)
        
        # Chebyshev method
        x_cheb_test, _ = construct_chebyshev_nodes(n_points, DOMAIN)
        y_cheb_test = test_func(x_cheb_test)
        y_deriv_cheb_test = chebyshev_derivative_from_values(y_cheb_test, x_cheb_test, DOMAIN)
        y_deriv_cheb_exact_test = test_func_deriv(x_cheb_test)
        error_cheb_test = np.max(np.abs(y_deriv_cheb_test - y_deriv_cheb_exact_test))
        errors_cheb.append(error_cheb_test)
        
        print(f"N = {n_points:5d} | BSPF: {error_bspf_test:.6e} | Chebyshev: {error_cheb_test:.6e}")
    
    # Compute convergence rate for BSPF
    x_fit = GRID_SIZES[FIT_START:FIT_END]
    y_fit = np.array(errors_bspf[FIT_START:FIT_END])
    log_x_fit = np.log(x_fit)
    log_y_fit = np.log(y_fit)
    coefficients = np.polyfit(log_x_fit, log_y_fit, 1)
    slope = coefficients[0]
    intercept = coefficients[1]
    
    print(f"\nBSPF convergence rate (from indices {FIT_START}-{FIT_END}): {slope:.3f}")
    
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
    
    # (a) Original function and spline approximation
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(x, y, '-', label='$f(x)$', linewidth=1)
    ax1.plot(x, y_spline, '-', label='$f_s(x)$', linewidth=1.5, alpha=1)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$f(x)$')
    ax1.legend(ncol=1)
    ax1.set_title('(a)', loc='left', x=-0.15, fontsize=24, fontweight='bold')
    
    # (b) Derivatives comparison
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(x, y_deriv_exact, '-', color='k', label='Exact', linewidth=1)
    ax2.plot(x, y_deriv_bspf, '-', label='BSPF', linewidth=1.5)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$df/dx$')
    ax2.legend(ncol=1)
    ax2.set_title('(b)', loc='left', x=-0.15, fontsize=24, fontweight='bold')
    
    # (c) Pointwise errors
    ax3 = plt.subplot(2, 2, 3)
    ax3.semilogy(x, np.abs(y_deriv_bspf - y_deriv_exact), '-', 
                label='BSPF', color=default_colors[0], alpha=1)
    ax3.semilogy(x_cheb, np.abs(y_deriv_cheb - y_deriv_cheb_exact), '-', 
                label='Chebyshev', color=default_colors[1], alpha=1)
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$|Error|$')
    ax3.set_ylim(1e-15, 0.9*1e6)
    ax3.legend(ncol=1)
    ax3.set_title('(c)', loc='left', x=-0.15, fontsize=24, fontweight='bold')
    
    # (d) Convergence study
    ax4 = plt.subplot(2, 2, 4)
    ax4.loglog(GRID_SIZES, errors_bspf, '.-', linewidth=1, 
               label='BSPF', color=default_colors[0], alpha=1)
    ax4.loglog(GRID_SIZES, errors_cheb, '.-', linewidth=1, 
               label='Chebyshev', color=default_colors[1], alpha=1)
    
    # Mark the grid size used in (a)-(c)
    ax4.axvline(NUM_POINTS, linestyle='--', color='gray', linewidth=1)
    ax4.text(NUM_POINTS + 10, 2e-11, '$(a)-(c)$', color='gray', fontsize=18)
    
    ax4.set_xlabel('$N$')
    ax4.set_ylabel('$\\|Error\\|_\\infty$')
    ax4.set_ylim(1e-11, 0.9*1e5)
    ax4.legend(ncol=1)
    ax4.set_title('(d)', loc='left', x=-0.15, fontsize=24, fontweight='bold')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
