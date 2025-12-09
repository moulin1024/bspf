"""
1D Derivative with Local Refinement using PiecewiseBSPF.

This script demonstrates how PiecewiseBSPF1D can be used to achieve local
refinement at specific points by segmenting the domain with independent mesh
resolutions. It compares:
    - Global BSPF (single segment, uniform resolution)
    - PiecewiseBSPF with two breakpoints (same resolution per segment)
    - PiecewiseBSPF with independent mesh resolutions per subdomain

The test function from diff_1d.py is used as a benchmark:
    f(x) = sin(x / (1.01 + cos(x)))

This function has varying frequency, making it a good candidate for local
refinement strategies. The independent mesh resolution approach allows finer
grids in regions of high variation while keeping coarser grids elsewhere.

Run from repository root:
    python examples/basic/diff_with_local_refinement.py
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from bspf import bspf1d, PiecewiseBSPF1D

if PiecewiseBSPF1D is None:
    raise ImportError(
        "PiecewiseBSPF1D is not available. "
        "Please ensure the piecewise_bspf1d module is properly installed."
    )


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
NUM_POINTS = 5000            # Grid resolution for main computation
CLUSTERING_FACTOR = 2.0      # Grid clustering factor near endpoints
USE_CLUSTERING = True         # Enable grid clustering

# Refinement point - choose a location where the function has high variation
# The function sin(x / (1.01 + cos(x))) has rapid changes around x ≈ π
REFINEMENT_POINT = np.pi     # Point for local refinement


# ============================================================================
# Test Function Definition (from diff_1d.py)
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
    f_sym = sp.sin(t / (1.1 + sp.cos(t)))
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
    
    # Compute function values and exact derivative
    y = test_func(x)
    y_deriv_exact = test_func_deriv(x)
    
    # ========================================================================
    # Method 1: Global BSPF (single segment)
    # ========================================================================
    print("Computing with global BSPF...")
    model_global = bspf1d.from_grid(
        degree=DEGREE,
        x=x,
        domain=tuple(DOMAIN),
        order=DEGREE,
        n_basis=N_BASIS,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        use_clustering=USE_CLUSTERING,
        clustering_factor=CLUSTERING_FACTOR
    )
    y_deriv_global, _, y_spline_global = model_global.differentiate_1_2(y, lam=REG_PARAM)
    
    # ========================================================================
    # Method 2: PiecewiseBSPF with two breakpoints for local refinement
    # ========================================================================
    # Add breakpoints on both sides of the refinement point for better isolation
    breakpoints = [
        REFINEMENT_POINT - np.pi/2,
        REFINEMENT_POINT + np.pi/2
    ]
    print(f"Computing with PiecewiseBSPF (breakpoints: {breakpoints})...")
    model_piecewise = PiecewiseBSPF1D(
        degree=DEGREE,
        x=x,
        breakpoints=breakpoints,
        order=DEGREE,
        n_basis=N_BASIS,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        use_clustering=USE_CLUSTERING,
        clustering_factor=CLUSTERING_FACTOR
    )
    y_deriv_piecewise, _, y_spline_piecewise = model_piecewise.differentiate_1_2(y, lam=REG_PARAM)
    
    # ========================================================================
    # Method 3: PiecewiseBSPF with independent mesh resolutions per subdomain
    # ========================================================================
    # Define domain boundaries: [0, π/2, 3π/2, 2π] creates 3 subdomains
    domain_boundaries = [
        DOMAIN[0],
        REFINEMENT_POINT - np.pi/2,
        REFINEMENT_POINT + np.pi/2,
        DOMAIN[1]
    ]
    
    # Specify different resolutions for each subdomain
    # Use finer resolution in the middle subdomain (around refinement point)
    n_points_per_segment = [
        1000,   # Left subdomain: coarser
        3000,   # Middle subdomain: finer (around refinement point)
        1000    # Right subdomain: coarser
    ]
    
    print(f"Computing with PiecewiseBSPF (independent resolutions: {n_points_per_segment})...")
    print(f"  Domain boundaries: {domain_boundaries}")
    model_piecewise_refined = PiecewiseBSPF1D.from_domains(
        degree=DEGREE,
        domain_boundaries=domain_boundaries,
        n_points_per_segment=n_points_per_segment,
        order=DEGREE,
        n_basis=N_BASIS,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        use_clustering=USE_CLUSTERING,
        clustering_factor=CLUSTERING_FACTOR
    )
    
    # Interpolate function values to the new grid
    y_refined = np.interp(model_piecewise_refined.x, x, y)
    y_deriv_piecewise_refined, _, y_spline_piecewise_refined = model_piecewise_refined.differentiate_1_2(
        y_refined, lam=REG_PARAM
    )
    
    # Interpolate results back to original grid for comparison
    y_deriv_piecewise_refined_interp = np.interp(x, model_piecewise_refined.x, y_deriv_piecewise_refined)
    
    # ========================================================================
    # Error Analysis
    # ========================================================================
    error_global = np.abs(y_deriv_global - y_deriv_exact)
    error_piecewise = np.abs(y_deriv_piecewise - y_deriv_exact)
    error_piecewise_refined = np.abs(y_deriv_piecewise_refined_interp - y_deriv_exact)
    
    # Compute L∞ and L² errors
    linf_global = np.max(error_global)
    linf_piecewise = np.max(error_piecewise)
    linf_piecewise_refined = np.max(error_piecewise_refined)
    
    l2_global = np.sqrt(np.mean(error_global**2))
    l2_piecewise = np.sqrt(np.mean(error_piecewise**2))
    l2_piecewise_refined = np.sqrt(np.mean(error_piecewise_refined**2))
    
    print("\n" + "=" * 60)
    print("Error Comparison (L∞ and L² norms):")
    print("=" * 60)
    print(f"Global BSPF:              L∞ = {linf_global:.6e}, L² = {l2_global:.6e}")
    print(f"PiecewiseBSPF (2 bp):     L∞ = {linf_piecewise:.6e}, L² = {l2_piecewise:.6e}")
    print(f"PiecewiseBSPF (refined):  L∞ = {linf_piecewise_refined:.6e}, L² = {l2_piecewise_refined:.6e}")
    print("=" * 60)
    
    # Analyze errors near the refinement point
    mask_near_refinement = np.abs(x - REFINEMENT_POINT) < 0.5
    error_near_global = np.max(error_global[mask_near_refinement])
    error_near_piecewise = np.max(error_piecewise[mask_near_refinement])
    error_near_piecewise_refined = np.max(error_piecewise_refined[mask_near_refinement])
    
    print(f"\nError near refinement point (|x - {REFINEMENT_POINT:.4f}| < 0.5):")
    print(f"Global BSPF:              {error_near_global:.6e}")
    print(f"PiecewiseBSPF (2 bp):     {error_near_piecewise:.6e}")
    print(f"PiecewiseBSPF (refined):  {error_near_piecewise_refined:.6e}")
    
    # ========================================================================
    # Visualization
    # ========================================================================
    
    # Set up plotting parameters
    plt.rcParams.update({
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'figure.titlesize': 20,
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.5, 1, 1], hspace=0.3)
    axes = [[fig.add_subplot(gs[0, :])], 
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
            [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]]
    
    # (a) Grid structure visualization
    ax = axes[0][0]
    # Show grid points for each method
    ax.scatter(x[::50], np.zeros_like(x[::50]), s=10, alpha=0.5, label='Global BSPF grid', marker='o')
    # Show refined grid points
    x_refined = model_piecewise_refined.x
    ax.scatter(x_refined[::10], np.ones_like(x_refined[::10]) * 0.5, s=10, alpha=0.7, 
               label='PiecewiseBSPF refined grid', marker='s', color='orange')
    ax.axvline(REFINEMENT_POINT, color='r', linestyle=':', linewidth=2, alpha=0.7)
    for bp in breakpoints:
        ax.axvline(bp, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('Grid')
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['Global', 'Refined', ''])
    ax.legend(loc='best', fontsize=10)
    ax.set_title('(a) Grid structure comparison')
    ax.grid(True, alpha=0.3, axis='x')
    
    # (b) Function and its derivative
    ax = axes[1][0]
    ax.plot(x, y, 'k-', label='$f(x)$', linewidth=1.5)
    ax.plot(x, y_deriv_exact, 'k--', label="$f'(x)$ (exact)", linewidth=1.5, alpha=0.7)
    ax.axvline(REFINEMENT_POINT, color='r', linestyle=':', linewidth=2, 
               label=f'Refinement point ({REFINEMENT_POINT:.2f})', alpha=0.7)
    for bp in breakpoints:
        ax.axvline(bp, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, 
                   label=f'Breakpoint ({bp:.2f})')
    ax.set_ylabel('$f(x)$, $f\'(x)$')
    ax.legend(loc='best')
    ax.set_title('(a) Test function and exact derivative')
    ax.grid(True, alpha=0.3)
    
    # (c) Derivative comparison
    ax = axes[1][1]
    ax.plot(x, y_deriv_exact, 'k-', label='Exact', linewidth=2, alpha=0.7)
    ax.plot(x, y_deriv_global, '-', label='Global BSPF', linewidth=1.5, alpha=0.8)
    ax.plot(x, y_deriv_piecewise, '--', label='PiecewiseBSPF (2 bp)', linewidth=1.5, alpha=0.8)
    ax.plot(x, y_deriv_piecewise_refined_interp, '-.', label='PiecewiseBSPF (refined)', 
            linewidth=1.5, alpha=0.8)
    ax.axvline(REFINEMENT_POINT, color='r', linestyle=':', linewidth=2, alpha=0.7)
    for bp in breakpoints:
        ax.axvline(bp, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_ylabel("$f'(x)$")
    ax.legend(loc='best')
    ax.set_title('(b) Numerical derivatives')
    ax.grid(True, alpha=0.3)
    
    # (d) Pointwise errors
    ax = axes[2][0]
    ax.semilogy(x, error_global, '-', label='Global BSPF', linewidth=1.5, alpha=0.8)
    ax.semilogy(x, error_piecewise, '--', label='PiecewiseBSPF (2 bp)', linewidth=1.5, alpha=0.8)
    ax.semilogy(x, error_piecewise_refined, '-.', label='PiecewiseBSPF (refined)', 
                linewidth=1.5, alpha=0.8)
    ax.axvline(REFINEMENT_POINT, color='r', linestyle=':', linewidth=2, alpha=0.7)
    for bp in breakpoints:
        ax.axvline(bp, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|Error|$')
    ax.legend(loc='best')
    ax.set_title('(c) Pointwise absolute errors')
    ax.grid(True, alpha=0.3)
    
    # (e) Zoomed view near refinement point
    ax = axes[2][1]
    mask_zoom = (x >= REFINEMENT_POINT - 1.0) & (x <= REFINEMENT_POINT + 1.0)
    x_zoom = x[mask_zoom]
    ax.semilogy(x_zoom, error_global[mask_zoom], '-', label='Global BSPF', 
                linewidth=1.5, alpha=0.8, marker='o', markersize=3)
    ax.semilogy(x_zoom, error_piecewise[mask_zoom], '--', label='PiecewiseBSPF (2 bp)', 
                linewidth=1.5, alpha=0.8, marker='s', markersize=3)
    ax.semilogy(x_zoom, error_piecewise_refined[mask_zoom], '-.', label='PiecewiseBSPF (refined)', 
                linewidth=1.5, alpha=0.8, marker='^', markersize=3)
    ax.axvline(REFINEMENT_POINT, color='r', linestyle=':', linewidth=2, alpha=0.7)
    for bp in breakpoints:
        ax.axvline(bp, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|Error|$')
    ax.legend(loc='best')
    ax.set_title(f'(d) Zoomed view near x = {REFINEMENT_POINT:.2f}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

