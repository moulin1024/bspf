"""
Compare Two Interpolation Implementations.

This script compares:
1. `interpolate_split_mesh` method (from check.py approach)
2. `interpolate` + manual FFT residual correction (from interpolation_1d.py approach)

Both should give consistent results for the same input.

Run from repository root:
    python examples/basic/compare_interpolation_methods.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from bspf import bspf1d


# ============================================================================
# Test Function (same as check.py)
# ============================================================================

def define_test_function():
    """Define a non-periodic test function with varying amplitude."""
    def f_fun(xx):
        return (0.5 + 0.3 * xx) * np.sin(2.0 * xx)
    return f_fun


# ============================================================================
# FFT Interpolate Residual (from interpolation_1d.py)
# ============================================================================

def fft_interpolate_residual(f_coarse, x_coarse, x_fine):
    """
    FFT-based interpolation of residual using zero-padding.
    
    This is the approach from interpolation_1d.py.
    """
    f_coarse = np.asarray(f_coarse, dtype=float)
    N = len(f_coarse)
    N_fine = len(x_fine)
    
    # Use N-1 points for true periodicity
    f_periodic = f_coarse[:-1].copy()
    N_periodic = len(f_periodic)
    
    # Determine refine factor
    refine_factor = N_fine / N_periodic
    refine_factor_int = int(round(refine_factor))
    N_fft_fine = refine_factor_int * N_periodic
    
    # Forward real FFT on periodic grid
    F = np.fft.rfft(f_periodic)
    
    # Zero-pad in frequency space to interpolate
    f_fft_fine = np.fft.irfft(F, n=N_fft_fine) * (N_fft_fine / N_periodic)
    
    # Replace values at original coarse grid points with known values
    coarse_indices = np.arange(0, N_fft_fine, refine_factor_int)
    if len(coarse_indices) == N_periodic:
        f_fft_fine[coarse_indices] = f_periodic
    
    # Create FFT grid coordinates (periodic, endpoint=False)
    domain = (x_coarse[0], x_coarse[-1])
    L = domain[1] - domain[0]
    x_fft_fine = np.linspace(domain[0], domain[0] + L, N_fft_fine, endpoint=False)
    
    # Interpolate from FFT grid to target grid (x_fine)
    x_fine_wrapped = ((x_fine - domain[0]) % L) + domain[0]
    
    # Use np.interp for efficient linear interpolation with periodic boundaries
    f_fine = np.interp(x_fine_wrapped, x_fft_fine, f_fft_fine,
                      left=f_fft_fine[-1], right=f_fft_fine[0])
    
    # For points at the exact boundary, use f(domain[0])
    boundary_mask = np.abs(x_fine_wrapped - (domain[0] + L)) < 1e-12
    if np.any(boundary_mask):
        f_fine[boundary_mask] = f_fft_fine[0]
    
    return f_fine


# ============================================================================
# Main Comparison
# ============================================================================

def main():
    """Compare the two interpolation methods."""
    
    # Define test function
    f_fun = define_test_function()
    
    # Domain and grid setup (same as check.py)
    L = 2.0 * math.pi
    N = 128
    x = np.linspace(0.0, L, N, endpoint=True)
    f = f_fun(x)
    
    # Construct bspf1d model (same settings as check.py)
    model = bspf1d.from_grid(
        degree=7,
        x=x
    )
    
    refine_factor = 2
    lam = 0.0  # No regularization
    
    print("=" * 60)
    print("Comparing Two Interpolation Methods")
    print("=" * 60)
    print(f"Original grid size: {N}")
    print(f"Refine factor: {refine_factor}x")
    print(f"Regularization: {lam}")
    print()
    
    # ========================================================================
    # Method 1: interpolate_split_mesh (from check.py)
    # ========================================================================
    print("Method 1: interpolate_split_mesh")
    print("-" * 60)
    x_fine1, f_fine1, f_spline_fine1, r_fine1 = model.interpolate_split_mesh(
        f, 
        refine_factor=refine_factor,
        lam=lam
    )
    print(f"  Refined grid size: {len(f_fine1)}")
    print(f"  Expected: {refine_factor * (N - 1) + 1}")
    
    # ========================================================================
    # Method 2: interpolate + manual FFT residual (from interpolation_1d.py)
    # ========================================================================
    print("\nMethod 2: interpolate + manual FFT residual")
    print("-" * 60)
    
    # Step 1: Get spline on original grid
    _, _, f_spline_original = model.differentiate_1_2(f, lam=lam)
    
    # Step 2: Compute residual
    residual_original = f - f_spline_original
    
    # Step 3: Make residual periodic
    residual_periodic = residual_original.copy()
    periodicity_error = abs(residual_periodic[0] - residual_periodic[-1])
    if periodicity_error > 1e-12:
        avg_endpoint = 0.5 * (residual_periodic[0] + residual_periodic[-1])
        residual_periodic[0] = avg_endpoint
        residual_periodic[-1] = avg_endpoint
    
    # Step 4: Get interpolated grid using model.interpolate
    x_fine2, f_spline_fine2 = model.interpolate(f, lam=lam, use_fft=False)
    
    # Step 5: Interpolate residual using FFT
    r_fine2 = fft_interpolate_residual(residual_periodic, x, x_fine2)
    
    # Step 6: Combine
    f_fine2 = f_spline_fine2 + r_fine2
    
    print(f"  Refined grid size: {len(f_fine2)}")
    print(f"  Expected: {2 * N - 1}")
    
    # ========================================================================
    # Compare Results
    # ========================================================================
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    
    # Check if grids match
    if len(x_fine1) == len(x_fine2):
        grid_diff = np.max(np.abs(x_fine1 - x_fine2))
        print(f"Grid comparison:")
        print(f"  Grid sizes match: {len(x_fine1)} == {len(x_fine2)}")
        print(f"  Max grid difference: {grid_diff:.6e}")
        if grid_diff > 1e-12:
            print(f"  ⚠ Warning: Grids differ!")
        else:
            print(f"  ✓ Grids match exactly")
    else:
        print(f"⚠ Warning: Grid sizes differ: {len(x_fine1)} vs {len(x_fine2)}")
        # Interpolate to common grid for comparison
        x_fine_common = np.sort(np.unique(np.concatenate([x_fine1, x_fine2])))
        f_fine1_common = np.interp(x_fine_common, x_fine1, f_fine1)
        f_fine2_common = np.interp(x_fine_common, x_fine2, f_fine2)
        f_fine1 = f_fine1_common
        f_fine2 = f_fine2_common
        x_fine1 = x_fine_common
        x_fine2 = x_fine_common
    
    # Compare interpolated values
    if len(f_fine1) == len(f_fine2):
        diff = f_fine1 - f_fine2
        max_diff = np.max(np.abs(diff))
        l2_diff = np.sqrt(np.mean(diff**2))
        
        print(f"\nInterpolated values comparison:")
        print(f"  L∞ difference: {max_diff:.6e}")
        print(f"  L² difference:  {l2_diff:.6e}")
        
        if max_diff < 1e-10:
            print(f"  ✓ Methods give consistent results (within numerical precision)")
        elif max_diff < 1e-6:
            print(f"  ⚠ Methods differ slightly (may be due to implementation details)")
        else:
            print(f"  ✗ Methods differ significantly!")
        
        # Compare components
        if len(f_spline_fine1) == len(f_spline_fine2):
            diff_spline = f_spline_fine1 - f_spline_fine2
            max_diff_spline = np.max(np.abs(diff_spline))
            print(f"\nSpline component comparison:")
            print(f"  L∞ difference: {max_diff_spline:.6e}")
        
        if len(r_fine1) == len(r_fine2):
            diff_residual = r_fine1 - r_fine2
            max_diff_residual = np.max(np.abs(diff_residual))
            print(f"\nResidual component comparison:")
            print(f"  L∞ difference: {max_diff_residual:.6e}")
    
    # Compare errors against exact solution
    f_exact = f_fun(x_fine1)
    err1 = f_fine1 - f_exact
    err2 = f_fine2 - f_exact
    
    # Identify interpolated points (exclude original coarse points)
    coarse_indices = np.arange(0, len(x_fine1), refine_factor)
    interpolated_mask = np.ones(len(x_fine1), dtype=bool)
    interpolated_mask[coarse_indices] = False
    
    err1_interp = err1[interpolated_mask]
    err2_interp = err2[interpolated_mask]
    
    max_err1 = np.max(np.abs(err1_interp))
    max_err2 = np.max(np.abs(err2_interp))
    l2_err1 = np.sqrt(np.mean(err1_interp**2))
    l2_err2 = np.sqrt(np.mean(err2_interp**2))
    
    print(f"\nError comparison (on interpolated points only):")
    print(f"  Method 1 (interpolate_split_mesh):")
    print(f"    L∞ error: {max_err1:.6e}")
    print(f"    L² error:  {l2_err1:.6e}")
    print(f"  Method 2 (interpolate + manual FFT):")
    print(f"    L∞ error: {max_err2:.6e}")
    print(f"    L² error:  {l2_err2:.6e}")
    
    # ========================================================================
    # Visualization
    # ========================================================================
    
    plt.rcParams.update({
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (a) Function comparison
    ax = axes[0, 0]
    ax.plot(x, f, "o", markersize=4, label="Coarse samples", alpha=0.7, zorder=3)
    ax.plot(x_fine1, f_exact, "-", linewidth=2, label="Exact", alpha=0.8, color='k')
    ax.plot(x_fine1, f_fine1, "--", linewidth=1.5, label="Method 1", alpha=0.8, color='C1')
    ax.plot(x_fine2, f_fine2, ":", linewidth=1.5, label="Method 2", alpha=0.8, color='C2')
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.legend(loc='best')
    ax.set_title("(a) Function comparison")
    ax.grid(True, alpha=0.3)
    
    # (b) Difference between methods
    ax = axes[0, 1]
    if len(f_fine1) == len(f_fine2):
        diff_plot = f_fine1 - f_fine2
        ax.plot(x_fine1, diff_plot, "-", linewidth=1.5, alpha=0.8, color='C3')
        ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        # Mark coarse grid points
        for x_coarse in x[::max(1, len(x)//20)]:
            ax.axvline(x_coarse, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f_1 - f_2$")
    ax.set_title("(b) Difference between methods")
    ax.grid(True, alpha=0.3)
    
    # (c) Error comparison
    ax = axes[1, 0]
    ax.semilogy(x_fine1, np.abs(err1), "-", linewidth=1.5, label="Method 1", alpha=0.8, color='C1')
    ax.semilogy(x_fine2, np.abs(err2), "--", linewidth=1.5, label="Method 2", alpha=0.8, color='C2')
    # Mark coarse grid points
    for x_coarse in x[::max(1, len(x)//20)]:
        ax.axvline(x_coarse, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$|Error|$")
    ax.legend(loc='best')
    ax.set_title("(c) Error comparison")
    ax.grid(True, alpha=0.3)
    
    # (d) Zoomed difference
    ax = axes[1, 1]
    zoom_center = L / 2
    zoom_width = L / 4
    mask = (x_fine1 >= zoom_center - zoom_width) & (x_fine1 <= zoom_center + zoom_width)
    if len(f_fine1) == len(f_fine2):
        diff_plot = f_fine1 - f_fine2
        ax.plot(x_fine1[mask], diff_plot[mask], "-", linewidth=1.5, alpha=0.8, color='C3')
        ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f_1 - f_2$")
    ax.set_title("(d) Zoomed difference")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()







