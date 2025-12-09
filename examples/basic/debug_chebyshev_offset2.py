"""
Debug script to check if the offset is due to mean removal or interpolation.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from bspf.utils import construct_chebyshev_nodes, chebyshev_derivative_from_values
from diff_2d import (
    create_symbolic_turbulence_field,
    DOMAIN_X, DOMAIN_Y, TURB_N_MODES, TURB_SEED
)

# Grid parameters
NX = 512
test_row_idx = NX // 2

print("=" * 60)
print("Debugging Chebyshev Offset - Mean Removal Impact")
print("=" * 60)

# Create uniform grid
Lx = DOMAIN_X[1] - DOMAIN_X[0]
Ly = DOMAIN_Y[1] - DOMAIN_Y[0]
x_unif = np.linspace(DOMAIN_X[0], DOMAIN_X[1], NX)
y_unif = np.linspace(DOMAIN_Y[0], DOMAIN_Y[1], NX)
y_val = y_unif[test_row_idx]

# Get symbolic functions
u_func, u_x_func, _ = create_symbolic_turbulence_field(Lx, Ly, TURB_N_MODES, TURB_SEED)

# Evaluate on uniform grid
u_1d = u_func(x_unif, np.full_like(x_unif, y_val))
u_x_exact_1d = u_x_func(x_unif, np.full_like(x_unif, y_val))
u_1d -= np.mean(u_1d)  # Remove mean

# Chebyshev nodes
x_cheb, _ = construct_chebyshev_nodes(NX, domain=tuple(DOMAIN_X))

# Evaluate on Chebyshev nodes
u_cheb_1d = u_func(x_cheb, np.full_like(x_cheb, y_val))
u_x_cheb_exact_1d = u_x_func(x_cheb, np.full_like(x_cheb, y_val))

print(f"\nBefore mean removal:")
print(f"  Uniform grid - mean of u: {np.mean(u_func(x_unif, np.full_like(x_unif, y_val))):.10e}")
print(f"  Chebyshev nodes - mean of u: {np.mean(u_cheb_1d):.10e}")

# Test: Remove mean BEFORE computing derivative
print("\n" + "-" * 60)
print("Test: Remove mean BEFORE computing Chebyshev derivative")
print("-" * 60)
u_cheb_1d_no_mean = u_cheb_1d - np.mean(u_cheb_1d)
u_x_cheb = chebyshev_derivative_from_values(
    u_cheb_1d_no_mean, x_cheb, domain=tuple(DOMAIN_X)
)

# The exact derivative should be the same (derivative of constant is zero)
# But let's check if there's a difference
error_on_nodes = np.abs(u_x_cheb - u_x_cheb_exact_1d)
print(f"  Error on Chebyshev nodes:")
print(f"    L²: {np.sqrt(np.mean(error_on_nodes**2)):.10e}")
print(f"    L∞: {np.max(error_on_nodes):.10e}")
print(f"    Mean error: {np.mean(error_on_nodes):.10e}")

# Check mean of derivatives
print(f"\n  Mean of Chebyshev derivative: {np.mean(u_x_cheb):.10e}")
print(f"  Mean of exact derivative: {np.mean(u_x_cheb_exact_1d):.10e}")
print(f"  Difference: {np.mean(u_x_cheb) - np.mean(u_x_cheb_exact_1d):.10e}")

# Interpolate to uniform grid
print("\n" + "-" * 60)
print("Interpolating to uniform grid")
print("-" * 60)
interp_deriv = RegularGridInterpolator((x_cheb,), u_x_cheb, method='linear')
u_x_cheb_interp = interp_deriv(x_unif)

error_interp = np.abs(u_x_cheb_interp - u_x_exact_1d)
print(f"  Error after interpolation:")
print(f"    L²: {np.sqrt(np.mean(error_interp**2)):.10e}")
print(f"    L∞: {np.max(error_interp):.10e}")
print(f"    Mean error: {np.mean(error_interp):.10e}")

# Check mean of interpolated derivative
print(f"\n  Mean of interpolated Chebyshev derivative: {np.mean(u_x_cheb_interp):.10e}")
print(f"  Mean of exact derivative (uniform): {np.mean(u_x_exact_1d):.10e}")
print(f"  Difference: {np.mean(u_x_cheb_interp) - np.mean(u_x_exact_1d):.10e}")

# Check if the offset is constant
offset = u_x_cheb_interp - u_x_exact_1d
print(f"\n  Offset statistics:")
print(f"    Mean offset: {np.mean(offset):.10e}")
print(f"    Std offset: {np.std(offset):.10e}")
print(f"    Min offset: {np.min(offset):.10e}")
print(f"    Max offset: {np.max(offset):.10e}")

# Check if offset is approximately constant
if np.std(offset) < 1e-10:
    print(f"  -> Offset appears to be constant (std < 1e-10)")
    print(f"  -> This suggests a systematic bias, not random error")
else:
    print(f"  -> Offset varies (std = {np.std(offset):.10e})")
    print(f"  -> This suggests interpolation or evaluation errors")

print("\n" + "=" * 60)



