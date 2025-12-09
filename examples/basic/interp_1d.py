"""
1D Interpolation Test with Mesh Splitting.

This script tests the `interpolate_split_mesh` method which interpolates
a non-periodic function on a refined grid using B-splines + FFT split-mesh
interpolation of the periodic residual.

Uses the test function from diff_1d.py:
    f(x) = sin(x / (1.01 + cos(x)))

The method:
1. Fits a B-spline to the input data
2. Computes the residual (difference between data and spline)
3. Uses FFT split-mesh interpolation on the periodic residual
4. Evaluates the spline on the refined grid
5. Combines spline + residual to get the final interpolated values

This approach provides high-order accuracy for non-periodic functions by
treating the smooth component with B-splines and the residual with spectral
methods.

Run from repository root:
    python examples/basic/interpolation_1d.py
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from bspf import bspf1d


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
NUM_POINTS = 4300             # Initial grid resolution (will be doubled by interpolate)

# Convergence study parameters
CONVERGENCE_GRID_SIZES = np.geomspace(1000, 10000, 50).astype(int)  # Grid sizes for convergence study


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
    # f_sym = sp.tanh(100*(t-np.pi))  # Periodic - for testing pure FFT (achieves machine precision)
    f_sym = sp.sin(t / (1.01 + sp.cos(t)))  # Non-periodic - original test function
    # Note: For non-periodic functions, FFT residual correction still works but
    # may introduce small boundary errors due to periodicity enforcement
    df_sym = sp.diff(f_sym, t)
    
    # Convert to NumPy functions
    func = sp.lambdify(t, f_sym, modules='numpy')
    func_deriv = sp.lambdify(t, df_sym, modules='numpy')
    
    return func, func_deriv


# ============================================================================
# Interpolation Test Function
# ============================================================================

def interpolation_1d():
    """
    Test bspf1d interpolation functionality that splits the mesh.
    """
    # Define test function
    test_func, test_func_deriv = define_test_function()
    
    # ========================================================================
    # Create initial grid and BSPF model
    # ========================================================================
    # For FFT interpolation to work optimally, we should use endpoint=False
    # to ensure periodicity. However, BSPF typically uses endpoint=True.
    # We'll use endpoint=True for BSPF but handle periodicity in the residual.
    x_original = np.linspace(DOMAIN[0], DOMAIN[1], NUM_POINTS, endpoint=True)
    
    print("=" * 60)
    print("Interpolation Test with Mesh Splitting")
    print("=" * 60)
    print(f"Original grid size: {NUM_POINTS}")
    print(f"Domain: [{DOMAIN[0]}, {DOMAIN[1]}]")
    print("=" * 60)
    
    # Initialize BSPF model
    model = bspf1d.from_grid(
        degree=DEGREE,
        x=x_original,
        domain=tuple(DOMAIN),
        order=DEGREE,
        n_basis=N_BASIS,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        use_clustering=True,
        clustering_factor=2.0
    )
    
    # Compute exact function values on original grid
    y_original = test_func(x_original)
    
    # ========================================================================
    # Interpolation using interpolate_split_mesh
    # ========================================================================
    print("\nTesting interpolation with split-mesh method...")
    
    # Use interpolate_split_mesh with refine_factor=2 (doubles resolution)
    # This method:
    # 1. Fits a B-spline to the input data
    # 2. Computes the residual (difference between data and spline)
    # 3. Uses FFT split-mesh interpolation on the periodic residual
    # 4. Evaluates the spline on the refined grid
    # 5. Combines spline + residual to get the final interpolated values
    refine_factor = 2
    x_interp, y_interp, y_spline_interp, r_interp = model.interpolate_split_mesh(
        y_original, 
        refine_factor=refine_factor,
        lam=REG_PARAM
    )
    
    print(f"Interpolated grid size: {len(x_interp)}")
    print(f"Expected size (M*(N-1)+1): {refine_factor * (NUM_POINTS - 1) + 1}")
    print(f"Size matches: {len(x_interp) == refine_factor * (NUM_POINTS - 1) + 1}")
    
    # Compute exact function values on interpolated grid
    y_exact_interp = test_func(x_interp)
    
    # Compute interpolation errors
    error_interp = np.abs(y_interp - y_exact_interp)
    error_spline_only = np.abs(y_spline_interp - y_exact_interp)
    
    # linear interpolation using interp1d
    interp_scipy_linear = interp1d(x_original, y_original, kind='linear')
    y_scipy_linear_interp = interp_scipy_linear(x_interp)
    error_scipy_linear = np.abs(y_scipy_linear_interp - y_exact_interp)
    
    # Identify interpolated points (exclude original coarse points)
    coarse_indices = np.arange(0, len(x_interp), refine_factor)
    interpolated_mask = np.ones(len(x_interp), dtype=bool)
    interpolated_mask[coarse_indices] = False
    
    # Error on interpolated points only (what matters for interpolation quality)
    err_interpolated = error_interp[interpolated_mask]
    linf_interp = np.max(err_interpolated)
    l2_interp = np.sqrt(np.mean(err_interpolated**2))
    
    # Error on spline only (for comparison)
    err_spline_only_interp = error_spline_only[interpolated_mask]
    linf_spline_only = np.max(err_spline_only_interp)
    l2_spline_only = np.sqrt(np.mean(err_spline_only_interp**2))
    
    # Error on linear interpolation (for comparison)
    err_scipy_linear_interp = error_scipy_linear[interpolated_mask]
    linf_scipy_linear = np.max(err_scipy_linear_interp)
    l2_scipy_linear = np.sqrt(np.mean(err_scipy_linear_interp**2))
    
    print("\n" + "=" * 60)
    print("Interpolation Error Analysis")
    print("=" * 60)
    print(f"Interpolated points: {np.sum(interpolated_mask)} (new points only)")
    print(f"\nInterpolation error (with FFT residual correction, on interpolated points only):")
    print(f"  L∞: {linf_interp:.6e}")
    print(f"  L²:  {l2_interp:.6e}")
    print(f"\nInterpolation error (spline only, no correction, on interpolated points only):")
    print(f"  L∞: {linf_spline_only:.6e}")
    print(f"  L²:  {l2_spline_only:.6e}")
    print(f"\nInterpolation error (linear, on interpolated points only):")
    print(f"  L∞: {linf_scipy_linear:.6e}")
    print(f"  L²:  {l2_scipy_linear:.6e}")
    if linf_spline_only > 0:
        improvement = linf_spline_only / linf_interp
        print(f"\nFFT residual correction improvement (vs spline only): {improvement:.2f}x")
    if linf_scipy_linear > 0:
        improvement_vs_scipy = linf_scipy_linear / linf_interp
        print(f"FFT residual correction improvement (vs linear): {improvement_vs_scipy:.2f}x")
    
    # Sanity check: coarse points should match exactly (machine precision)
    coarse_from_fine = y_interp[coarse_indices]
    max_diff_coarse = np.max(np.abs(coarse_from_fine - y_original))
    if max_diff_coarse > 1e-12:
        print(f"\n⚠ Warning: Coarse point mismatch: {max_diff_coarse:.6e}")
    else:
        print(f"\n✓ Coarse points preserved exactly (max diff: {max_diff_coarse:.6e})")
    
    # ========================================================================
    # Convergence Study: BSPF vs Scipy Interpolation
    # ========================================================================
    print("\n" + "=" * 60)
    print("Convergence Study: BSPF vs Scipy Interpolation")
    print("=" * 60)
    
    errors_bspf_fft = []
    errors_scipy_linear = []
    grid_sizes_conv = []
    
    for n_points in CONVERGENCE_GRID_SIZES:
        print(f"Processing N = {n_points}...")
        
        # Create coarse grid
        x_coarse = np.linspace(DOMAIN[0], DOMAIN[1], n_points, endpoint=True)
        y_coarse = test_func(x_coarse)
        
        # Create fine grid (2x resolution)
        n_fine = 2 * n_points - 1
        x_fine = np.linspace(DOMAIN[0], DOMAIN[1], n_fine, endpoint=True)
        y_exact_fine = test_func(x_fine)
        
        # 1. BSPF interpolation using interpolate_split_mesh
        model_conv = bspf1d.from_grid(
            degree=DEGREE,
            x=x_coarse,
            domain=tuple(DOMAIN),
            order=DEGREE,
            n_basis=N_BASIS,
            num_boundary_points=NUM_BOUNDARY_POINTS,
            use_clustering=True,
            clustering_factor=2.0
        )
        
        # Use interpolate_split_mesh with refine_factor=2
        x_interp_conv, y_interp_bspf_fft, _, _ = model_conv.interpolate_split_mesh(
            y_coarse, 
            refine_factor=2,
            lam=REG_PARAM
        )
        
        # Evaluate on fine grid (interpolate to match fine grid exactly)
        y_bspf_fft_fine = np.interp(x_fine, x_interp_conv, y_interp_bspf_fft)
        error_bspf_fft = np.max(np.abs(y_bspf_fft_fine - y_exact_fine))  # L∞ norm
        errors_bspf_fft.append(error_bspf_fft)
        
        # 2. linear interpolation
        interp_linear = interp1d(x_coarse, y_coarse, kind='linear')
        y_scipy_linear_conv = interp_linear(x_fine)
        error_scipy_linear_conv = np.max(np.abs(y_scipy_linear_conv - y_exact_fine))  # L∞ norm
        errors_scipy_linear.append(error_scipy_linear_conv)
        
        grid_sizes_conv.append(n_points)
        
        print(f"  BSPF:     L∞ = {error_bspf_fft:.6e}")
        print(f"  linear:   L∞ = {error_scipy_linear_conv:.6e}")
    
    errors_bspf_fft = np.array(errors_bspf_fft)
    errors_scipy_linear = np.array(errors_scipy_linear)
    grid_sizes_conv = np.array(grid_sizes_conv)
    
    # Compute log-log fit to last 20 points for convergence rate
    n_fit_points = min(20, len(grid_sizes_conv))
    fit_start = len(grid_sizes_conv) - n_fit_points
    
    # Fit for linear
    log_N_linear = np.log(grid_sizes_conv[fit_start:])
    log_err_linear = np.log(errors_scipy_linear[fit_start:])
    coeffs_linear = np.polyfit(log_N_linear, log_err_linear, 1)
    slope_linear = coeffs_linear[0]
    intercept_linear = coeffs_linear[1]
    
    print("\n" + "=" * 60)
    print("Convergence Summary")
    print("=" * 60)
    print(f"Grid sizes tested: {len(grid_sizes_conv)}")
    print(f"Range: {grid_sizes_conv.min()} - {grid_sizes_conv.max()} points")
    print(f"\nFinal errors (N = {grid_sizes_conv[-1]}):")
    print(f"  BSPF:     L∞ = {errors_bspf_fft[-1]:.6e}")
    print(f"  linear:   L∞ = {errors_scipy_linear[-1]:.6e}")
    print(f"\nConvergence rates (log-log fit to last {n_fit_points} points):")
    print(f"  linear:    {slope_linear:.3f} (O(N^{slope_linear:.3f}))")
    print(f"  Expected (linear): -2.0 (O(h^2))")
    
    # ========================================================================
    # Visualization
    # ========================================================================
    
    # Set up plotting parameters (matching diff_1d.py style)
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
    
    fig = plt.figure(figsize=(15, 4))
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # (a) Original function and interpolation
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(x_interp, y_exact_interp, '-', label='$f(x)$', linewidth=1)
    ax1.plot(x_interp, y_spline_interp, '-', label='$f_s(x)$', linewidth=1)
    # ax1.plot(x_interp, y_interp, '-', label='BSPF', linewidth=1)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$f(x)$')
    ax1.legend()
    ax1.set_title('(a)', loc='left', x=-0.15, fontsize=16, fontweight='bold')
    
    # (b) Interpolation errors
    ax3 = plt.subplot(1, 3, 2)
    ax3.semilogy(x_interp[interpolated_mask], error_interp[interpolated_mask], '-', 
                label='BSPF', color=default_colors[0], alpha=1, linewidth=1)
    ax3.semilogy(x_interp[interpolated_mask], error_scipy_linear[interpolated_mask], '-', 
                label='Linear', color=default_colors[2], alpha=1, linewidth=1)
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$|Error|$')
    ax3.set_ylim(1e-15, 0.9*1e1)
    ax3.legend()
    ax3.set_title('(b)', loc='left', x=-0.15, fontsize=16, fontweight='bold')
    
    # (c) Convergence study
    ax4 = plt.subplot(1, 3, 3)
    ax4.loglog(grid_sizes_conv, errors_bspf_fft, '.-', linewidth=1, 
               label='BSPF', color=default_colors[0], alpha=1)
    ax4.loglog(grid_sizes_conv, errors_scipy_linear, '.-', linewidth=1, 
               label='Linear', color=default_colors[2], alpha=1)
    # Add fitted lines from log-log fit to last 20 points
    # linear fitted line
    N_fit = grid_sizes_conv[fit_start:]
    err_fit_linear = np.exp(intercept_linear) * (N_fit ** slope_linear)
    # Format label: need triple braces for LaTeX superscript with f-string variable
    slope_str = f'{slope_linear:.1f}'
    ax4.loglog(N_fit, 0.5*err_fit_linear, '--', linewidth=1.5, 
               label=f'O($N^{{{slope_str}}}$)', 
               color=default_colors[2], alpha=0.8)
    # Mark the grid size used in (a)-(b)
    ax4.axvline(NUM_POINTS, linestyle='--', color='gray', linewidth=1)
    ax4.text(NUM_POINTS + 10, 2e-11, '$(a)-(b)$', color='gray', fontsize=12)
    
    ax4.set_xlabel('$N$')
    ax4.set_ylabel('$\\|Error\\|_\\infty$')
    ax4.set_ylim(1e-12, 0.9*1e1)
    ax4.legend(ncol=1)
    ax4.set_title('(c)', loc='left', x=-0.15, fontsize=16, fontweight='bold')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return model, x_original, x_interp, y_original, y_interp, y_exact_interp


if __name__ == "__main__":
    interpolation_1d()
