"""
1D Antiderivative Accuracy and Convergence Comparison.

This script compares BSPF and Chebyshev spectral methods for computing
antiderivatives (indefinite integrals) of smooth functions. It generates
a 2x2 figure showing:
    (a) Integrand and B-spline approximation
    (b) Exact antiderivative vs numerical approximations
    (c) Pointwise absolute errors for both methods
    (d) Convergence study showing error vs grid resolution

Run from repository root:
    python examples/basic/integral_1d.py
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from bspf import bspf1d
from bspf.utils import chebyshev_antiderivatives_fft


# ============================================================================
# Parameters
# ============================================================================

# BSPF parameters
DEGREE = 5                    # B-spline polynomial degree
BOUNDARY_ORDER = DEGREE        # Number of constraints per side
ALPHA = 2                      # Factor for extra degrees of freedom
NUM_BOUNDARY_POINTS = DEGREE   # Number of boundary points
N_BASIS = 2 * DEGREE * ALPHA  # Number of basis functions
REG_PARAM = 1e-3              # Tikhonov regularization strength (lambda)

# Grid parameters
DOMAIN = [0, 2*np.pi]         # Domain [a, b]
NUM_POINTS = 4300             # Grid resolution for main computation
CLUSTERING_FACTOR = 2.0       # Grid clustering factor near endpoints
USE_CLUSTERING = True          # Enable grid clustering

# Convergence study parameters
GRID_SIZES = np.geomspace(1000, 10000, 50).astype(int)  # Grid sizes for convergence study


# ============================================================================
# Test Function Definition
# ============================================================================

def define_test_function():
    """
    Define the test function and its derivative using SymPy.
    
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
        order=BOUNDARY_ORDER,
        n_basis=N_BASIS,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        use_clustering=USE_CLUSTERING,
        clustering_factor=CLUSTERING_FACTOR,
        correction="spectral"
    )
    
    # Compute function values and exact derivative
    y = test_func(x)
    y_deriv = test_func_deriv(x)
    
    # Compute antiderivatives using different methods
    # 1. BSPF method
    u_bspf, deriv_spline = model.antiderivative(
        y_deriv, 
        order=1, 
        left_value=y[0], 
        match_right=None, 
        lam=0.0
    )
    
    # 2. Chebyshev method (on Chebyshev nodes)
    x_cheb, u_cheb = chebyshev_antiderivatives_fft(
        test_func_deriv, 
        N=2000, 
        domain=DOMAIN, 
        order=1, 
        anchor="left", 
        c1=0.0, 
        c2=0.0
    )
    u_cheb = u_cheb + y[0]  # Adjust for left boundary value
    
    # Compute errors
    error_bspf = np.max(np.abs(u_bspf - y))
    error_cheb = np.max(np.abs(u_cheb - test_func(x_cheb)))
    
    print("=" * 60)
    print("Errors (L∞ Norm):")
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
        x_test = np.linspace(DOMAIN[0], DOMAIN[1], n_points, endpoint=True)
        
        # Compute exact solution
        y_test = test_func(x_test)
        y_deriv_test = test_func_deriv(x_test)
        
        # BSPF method
        model_test = bspf1d.from_grid(
            degree=DEGREE,
            x=x_test,
            domain=tuple(DOMAIN),
            order=BOUNDARY_ORDER,
            n_basis=N_BASIS,
            num_boundary_points=NUM_BOUNDARY_POINTS,
            use_clustering=USE_CLUSTERING,
            clustering_factor=CLUSTERING_FACTOR,
            correction="spectral"
        )
        u_bspf_test, _ = model_test.antiderivative(
            y_deriv_test, 
            order=1, 
            left_value=y_test[0], 
            match_right=None, 
            lam=0.0
        )
        error_bspf_test = np.max(np.abs(u_bspf_test - y_test))
        errors_bspf.append(error_bspf_test)
        
        # Chebyshev method
        x_cheb_test, u_cheb_test = chebyshev_antiderivatives_fft(
            test_func_deriv, 
            N=n_points, 
            domain=DOMAIN, 
            order=1, 
            anchor="left", 
            c1=0.0, 
            c2=0.0
        )
        u_cheb_test = u_cheb_test + y_test[0]
        error_cheb_test = np.max(np.abs(u_cheb_test - test_func(x_cheb_test)))
        errors_cheb.append(error_cheb_test)
        
        print(f"N = {n_points:5d} | BSPF: {error_bspf_test:.6e} | "
              f"Chebyshev: {error_cheb_test:.6e}")
    
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
    
    # (a) Integrand and spline approximation
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(x, y_deriv, '-', label='$f\'(x)$', linewidth=1)
    ax1.plot(x, deriv_spline, '-', label='$f\'_s(x)$', linewidth=1.5)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$f\'(x)$')
    ax1.legend(loc='upper left')
    ax1.set_title('(a)', loc='left', x=-0.15, fontsize=24, fontweight='bold')
    
    # (b) Antiderivatives comparison
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(x, y, 'k-', label='Exact', linewidth=1)
    ax2.plot(x, u_bspf, '-', label='BSPF', linewidth=1)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$f(x)$')
    ax2.legend()
    ax2.set_title('(b)', loc='left', x=-0.15, fontsize=24, fontweight='bold')
    
    # (c) Pointwise errors
    ax3 = plt.subplot(2, 2, 3)
    ax3.semilogy(x, np.abs(u_bspf - y), '-', label='BSPF', linewidth=1)
    ax3.semilogy(x_cheb, np.abs(u_cheb - test_func(x_cheb)), '-', 
                label='Chebyshev', linewidth=1)
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$|Error|$')
    ax3.set_ylim(1e-17, 0.91e4)
    ax3.legend(loc='upper left')
    ax3.set_title('(c)', loc='left', x=-0.15, fontsize=24, fontweight='bold')
    ax3.grid(True)
    
    # (d) Convergence study
    ax4 = plt.subplot(2, 2, 4)
    ax4.loglog(GRID_SIZES, errors_bspf, '.-', label='BSPF', linewidth=1)
    ax4.loglog(GRID_SIZES, errors_cheb, '.-', label='Chebyshev', linewidth=1)
    
    # Mark the grid size used in (a)-(c)
    ax4.axvline(NUM_POINTS, linestyle='--', color='gray', linewidth=1)
    ax4.text(NUM_POINTS + 10, 2e-12, '$(a)-(c)$', color='gray', fontsize=18)
    
    ax4.set_xlabel('$N$')
    ax4.set_ylabel('$\\|Error\\|_\\infty$')
    ax4.set_ylim(1e-12, 0.9*1e4)
    ax4.legend()
    ax4.set_title('(d)', loc='left', x=-0.15, fontsize=24, fontweight='bold')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
