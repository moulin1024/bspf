"""
Debug script to check if field values match between uniform and Chebyshev grids.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from bspf.utils import construct_chebyshev_nodes
from diff_2d import (
    random_turbulence_signal_2d_with_derivatives,
    evaluate_turbulence_field_on_grid,
    DOMAIN_X, DOMAIN_Y, TURB_N_MODES, TURB_SEED,
    SHOCK_ENABLED, SHOCK_CENTER_X, SHOCK_CENTER_Y,
    SHOCK_RADIUS, SHOCK_AMPLITUDE, SHOCK_WIDTH,
    add_circular_shock_wave
)

# Grid parameters
NX = 512
NY = 512

print("=" * 60)
print("Debugging Chebyshev Field Generation")
print("=" * 60)

# Generate on uniform grid
Lx = DOMAIN_X[1] - DOMAIN_X[0]
Ly = DOMAIN_Y[1] - DOMAIN_Y[0]
x_unif = np.linspace(DOMAIN_X[0], DOMAIN_X[1], NX)
y_unif = np.linspace(DOMAIN_Y[0], DOMAIN_Y[1], NY)

print(f"\nUniform grid: {NX} x {NY}")

# Method 1: Using original function
x1, y1, u_unif_orig, _, _, _, _, _ = random_turbulence_signal_2d_with_derivatives(
    Lx=Lx, Ly=Ly, Nx=NX, Ny=NY, nmodes=TURB_N_MODES, seed=TURB_SEED
)

# Method 2: Using new evaluation function
u_unif_new, _ = evaluate_turbulence_field_on_grid(
    x_unif, y_unif, Lx, Ly, TURB_N_MODES, TURB_SEED
)

print(f"\nComparison on uniform grid:")
print(f"  Original function field range: [{u_unif_orig.min():.6f}, {u_unif_orig.max():.6f}]")
print(f"  New function field range: [{u_unif_new.min():.6f}, {u_unif_new.max():.6f}]")
diff_unif = np.abs(u_unif_orig - u_unif_new)
print(f"  Difference L²: {np.sqrt(np.mean(diff_unif**2)):.6e}")
print(f"  Difference L∞: {np.max(diff_unif):.6e}")

# Generate on Chebyshev nodes
Nx_cheb = NX - 1
Ny_cheb = NY - 1
x_cheb, _ = construct_chebyshev_nodes(Nx_cheb, domain=tuple(DOMAIN_X))
y_cheb, _ = construct_chebyshev_nodes(Ny_cheb, domain=tuple(DOMAIN_Y))

print(f"\nChebyshev grid: {len(x_cheb)} x {len(y_cheb)}")

# Method 1: Using original function (different grid size = different field!)
x2, y2, u_cheb_orig, _, _, _, _, _ = random_turbulence_signal_2d_with_derivatives(
    Lx=Lx, Ly=Ly, Nx=len(x_cheb), Ny=len(y_cheb), nmodes=TURB_N_MODES, seed=TURB_SEED
)

# Method 2: Using new evaluation function (same field, different grid)
u_cheb_new, _ = evaluate_turbulence_field_on_grid(
    x_cheb, y_cheb, Lx, Ly, TURB_N_MODES, TURB_SEED
)

print(f"\nComparison on Chebyshev grid:")
print(f"  Original function field range: [{u_cheb_orig.min():.6f}, {u_cheb_orig.max():.6f}]")
print(f"  New function field range: [{u_cheb_new.min():.6f}, {u_cheb_new.max():.6f}]")
diff_cheb = np.abs(u_cheb_orig - u_cheb_new)
print(f"  Difference L²: {np.sqrt(np.mean(diff_cheb**2)):.6e}")
print(f"  Difference L∞: {np.max(diff_cheb):.6e}")

# Interpolate uniform field to Chebyshev nodes and compare
print(f"\n" + "-" * 60)
print("Interpolating uniform field to Chebyshev nodes:")
print("-" * 60)

interp_unif = RegularGridInterpolator((y_unif, x_unif), u_unif_new, method='linear')
X_cheb, Y_cheb = np.meshgrid(x_cheb, y_cheb)
points_cheb = np.column_stack([Y_cheb.ravel(), X_cheb.ravel()])
u_unif_interp_to_cheb = interp_unif(points_cheb).reshape(Y_cheb.shape)

print(f"  Uniform field (interpolated to Chebyshev): [{u_unif_interp_to_cheb.min():.6f}, {u_unif_interp_to_cheb.max():.6f}]")
print(f"  Chebyshev field (evaluated directly): [{u_cheb_new.min():.6f}, {u_cheb_new.max():.6f}]")
diff_interp = np.abs(u_unif_interp_to_cheb - u_cheb_new)
print(f"  Difference L²: {np.sqrt(np.mean(diff_interp**2)):.6e}")
print(f"  Difference L∞: {np.max(diff_interp):.6e}")

# Check at a few specific points
print(f"\n" + "-" * 60)
print("Pointwise comparison (first 5 points in x, middle row):")
print("-" * 60)
j = len(y_cheb) // 2
print(f"Row {j} (y = {y_cheb[j]:.6f}):")
for i in range(min(5, len(x_cheb))):
    print(f"  x[{i}] = {x_cheb[i]:.6f}:")
    print(f"    Uniform (interpolated): {u_unif_interp_to_cheb[j, i]:.10f}")
    print(f"    Chebyshev (direct):      {u_cheb_new[j, i]:.10f}")
    print(f"    Difference:             {abs(u_unif_interp_to_cheb[j, i] - u_cheb_new[j, i]):.10e}")

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print("The new evaluation function should produce the same field")
print("regardless of grid type (uniform or Chebyshev).")
print("Small differences are expected due to interpolation errors.")
print("=" * 60)



