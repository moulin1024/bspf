"""
1D Split-Mesh Interpolation Example.

This script demonstrates the `interpolate_split_mesh` method which interpolates
a non-periodic function on a refined grid using B-splines + FFT split-mesh
interpolation of the periodic residual.

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
    python examples/basic/check.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from bspf import bspf1d


# ============================================================================
# Test Function
# ============================================================================

def define_test_function():
    """
    Define a non-periodic test function with varying amplitude.
    
    Returns
    -------
    func : callable
        Function f(x) as a NumPy-compatible function
    """
    def f_fun(xx):
        # Non-periodic function with gently varying amplitude
        return (0.5 + 0.3 * xx) * np.sin(2.0 * xx)
    
    return f_fun


# ============================================================================
# Main Example
# ============================================================================

def main():
    """Main example demonstrating split-mesh interpolation."""
    
    # Define test function
    f_fun = define_test_function()
    
    # Domain and grid setup
    L = 2.0 * math.pi
    N = 128
    x = np.linspace(0.0, L, N, endpoint=True)  # non-periodic: include both endpoints
    
    # Compute function values
    f = f_fun(x)
    
    # Construct bspf1d model
    # Using degree=5 with default settings (enforces function + derivatives at endpoints)
    model = bspf1d.from_grid(
        degree=7,
        x=x
    )
    
    # Interpolate to fine grid using split-mesh method
    refine_factor = 2
    x_fine, f_fine, f_spline_fine, r_fine = model.interpolate_split_mesh(
        f, 
        refine_factor=refine_factor,
        lam=0.0  # No regularization
    )
    
    # Compute exact values on fine grid for error analysis
    f_exact = f_fun(x_fine)
    err = f_fine - f_exact
    
    # Identify interpolated points (exclude original coarse points)
    # Original coarse points are at indices: 0, M, 2*M, 3*M, ...
    coarse_indices = np.arange(0, len(x_fine), refine_factor)
    interpolated_mask = np.ones(len(x_fine), dtype=bool)
    interpolated_mask[coarse_indices] = False
    
    # Error on interpolated points only (what matters for interpolation quality)
    err_interpolated = err[interpolated_mask]
    max_err_interpolated = np.max(np.abs(err_interpolated))
    l2_err_interpolated = np.sqrt(np.mean(err_interpolated**2))
    
    print("=" * 60)
    print("Split-Mesh Interpolation Results")
    print("=" * 60)
    print(f"Original grid size: {N}")
    print(f"Refined grid size: {len(f_fine)}")
    print(f"Refine factor: {refine_factor}x")
    print(f"Interpolated points: {np.sum(interpolated_mask)} (new points only)")
    print(f"\nInterpolation errors (on interpolated points only):")
    print(f"  L∞ error: {max_err_interpolated:.6e}")
    print(f"  L² error:  {l2_err_interpolated:.6e}")
    
    # Sanity check: coarse points should match exactly (machine precision)
    coarse_from_fine = f_fine[coarse_indices]
    max_diff_coarse = np.max(np.abs(coarse_from_fine - f))
    if max_diff_coarse > 1e-12:
        print(f"\n⚠ Warning: Coarse point mismatch: {max_diff_coarse:.6e}")
    else:
        print(f"\n✓ Coarse points preserved exactly (max diff: {max_diff_coarse:.6e})")
    
    # ========================================================================
    # Visualization
    # ========================================================================
    
    plt.rcParams.update({
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (a) Function and interpolation
    ax = axes[0, 0]
    ax.plot(x, f, "o", markersize=6, label="Coarse samples", alpha=0.7, zorder=3)
    ax.plot(x_fine, f_exact, "-", linewidth=2, label="Exact", alpha=0.8, color='k')
    ax.plot(x_fine, f_fine, "--", linewidth=1.5, label="BSPF+FFT interp", alpha=0.8, color='C1')
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.legend(loc='best')
    ax.set_title("(a) Function and interpolation")
    ax.grid(True, alpha=0.3)
    
    # (b) Components: spline and residual
    ax = axes[0, 1]
    ax.plot(x_fine, f_spline_fine, "-", linewidth=1.5, label="Spline component", alpha=0.8, color='C2')
    ax.plot(x_fine, r_fine, "-", linewidth=1.5, label="Residual component", alpha=0.8, color='C3')
    ax.plot(x_fine, f_fine, "--", linewidth=1.5, label="Total (spline + residual)", alpha=0.8, color='C1')
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.legend(loc='best')
    ax.set_title("(b) Spline and residual components")
    ax.grid(True, alpha=0.3)
    
    # (c) Interpolation error
    ax = axes[1, 0]
    ax.semilogy(x_fine, np.abs(err), "-", linewidth=1.5, alpha=0.8, color='C1')
    # Mark coarse grid points
    for x_coarse in x[::max(1, len(x)//20)]:
        ax.axvline(x_coarse, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$|Error|$")
    ax.set_title("(c) Interpolation error")
    ax.grid(True, alpha=0.3)
    
    # (d) Zoomed view
    ax = axes[1, 1]
    zoom_center = L / 2
    zoom_width = L / 4
    mask = (x_fine >= zoom_center - zoom_width) & (x_fine <= zoom_center + zoom_width)
    ax.plot(x_fine[mask], f_exact[mask], "-", linewidth=2, label="Exact", alpha=0.8, color='k')
    ax.plot(x_fine[mask], f_fine[mask], "--", linewidth=1.5, label="BSPF+FFT interp", alpha=0.8, color='C1')
    # Mark coarse grid points in zoomed region
    mask_coarse = (x >= zoom_center - zoom_width) & (x <= zoom_center + zoom_width)
    if np.any(mask_coarse):
        ax.plot(x[mask_coarse], f[mask_coarse], "o", markersize=6, 
                label="Coarse samples", alpha=0.7, zorder=3)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.legend(loc='best')
    ax.set_title("(d) Zoomed view (middle region)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
