"""
Debug script to investigate the offset in Chebyshev derivative results.
"""

import numpy as np
from bspf.utils import construct_chebyshev_nodes, chebyshev_derivative_from_values
from diff_2d import (
    create_symbolic_turbulence_field,
    DOMAIN_X, DOMAIN_Y, TURB_N_MODES, TURB_SEED
)

# Grid parameters
NX = 512
test_row_idx = NX // 2

print("=" * 60)
print("Debugging Chebyshev Offset")
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

# Remove mean
u_1d_mean = np.mean(u_1d)
u_1d_no_mean = u_1d - u_1d_mean

print(f"\nUniform grid:")
print(f"  Mean of u: {u_1d_mean:.10e}")
print(f"  Mean of u_x (exact): {np.mean(u_x_exact_1d):.10e}")

# Chebyshev nodes
x_cheb, _ = construct_chebyshev_nodes(NX, domain=tuple(DOMAIN_X))

# Evaluate on Chebyshev nodes
u_cheb_1d = u_func(x_cheb, np.full_like(x_cheb, y_val))
u_x_cheb_exact_1d = u_x_func(x_cheb, np.full_like(x_cheb, y_val))

# Remove mean
u_cheb_mean = np.mean(u_cheb_1d)
u_cheb_1d_no_mean = u_cheb_1d - u_cheb_mean

print(f"\nChebyshev nodes:")
print(f"  Mean of u: {u_cheb_mean:.10e}")
print(f"  Mean of u_x (exact): {np.mean(u_x_cheb_exact_1d):.10e}")
print(f"  Difference in means (uniform vs Chebyshev): {u_1d_mean - u_cheb_mean:.10e}")

# Test 1: Chebyshev derivative WITHOUT mean removal
print("\n" + "-" * 60)
print("Test 1: Chebyshev derivative WITHOUT mean removal")
print("-" * 60)
u_x_cheb_1 = chebyshev_derivative_from_values(
    u_cheb_1d, x_cheb, domain=tuple(DOMAIN_X)
)
error_1 = np.abs(u_x_cheb_1 - u_x_cheb_exact_1d)
print(f"  Mean error: {np.mean(error_1):.10e}")
print(f"  Max error: {np.max(error_1):.10e}")
print(f"  Mean of Chebyshev derivative: {np.mean(u_x_cheb_1):.10e}")
print(f"  Mean of exact derivative: {np.mean(u_x_cheb_exact_1d):.10e}")
print(f"  Offset (mean difference): {np.mean(u_x_cheb_1) - np.mean(u_x_cheb_exact_1d):.10e}")

# Test 2: Chebyshev derivative WITH mean removal
print("\n" + "-" * 60)
print("Test 2: Chebyshev derivative WITH mean removal")
print("-" * 60)
u_x_cheb_2 = chebyshev_derivative_from_values(
    u_cheb_1d_no_mean, x_cheb, domain=tuple(DOMAIN_X)
)
error_2 = np.abs(u_x_cheb_2 - u_x_cheb_exact_1d)
print(f"  Mean error: {np.mean(error_2):.10e}")
print(f"  Max error: {np.max(error_2):.10e}")
print(f"  Mean of Chebyshev derivative: {np.mean(u_x_cheb_2):.10e}")
print(f"  Mean of exact derivative: {np.mean(u_x_cheb_exact_1d):.10e}")
print(f"  Offset (mean difference): {np.mean(u_x_cheb_2) - np.mean(u_x_cheb_exact_1d):.10e}")

# Test 3: Check if constant term affects derivative
print("\n" + "-" * 60)
print("Test 3: Constant offset test")
print("-" * 60)
# Add a constant to the function
u_cheb_const = u_cheb_1d + 1.0
u_x_cheb_3 = chebyshev_derivative_from_values(
    u_cheb_const, x_cheb, domain=tuple(DOMAIN_X)
)
error_3 = np.abs(u_x_cheb_3 - u_x_cheb_exact_1d)
print(f"  Function with constant offset (+1.0):")
print(f"  Mean error: {np.mean(error_3):.10e}")
print(f"  Max error: {np.max(error_3):.10e}")
print(f"  Mean of Chebyshev derivative: {np.mean(u_x_cheb_3):.10e}")
print(f"  Mean of exact derivative: {np.mean(u_x_cheb_exact_1d):.10e}")
print(f"  Offset (mean difference): {np.mean(u_x_cheb_3) - np.mean(u_x_cheb_exact_1d):.10e}")

# Test 4: Check Chebyshev coefficients
print("\n" + "-" * 60)
print("Test 4: Chebyshev coefficients analysis")
print("-" * 60)
from bspf.utils.chebyshev import _chebyshev_coeffs_rfft, _chebyshev_derivative_coeffs

a_k = _chebyshev_coeffs_rfft(u_cheb_1d)
b_k = _chebyshev_derivative_coeffs(a_k)

print(f"  a_0 (constant term): {a_k[0]:.10e}")
print(f"  b_0 (derivative constant term): {b_k[0]:.10e}")
print(f"  Sum of a_k: {np.sum(a_k):.10e}")
print(f"  Sum of b_k: {np.sum(b_k):.10e}")

# Check if a_0 contributes to derivative
print(f"\n  Note: The derivative of a constant should be zero.")
print(f"  If a_0 is non-zero, it represents a constant offset in the function.")
print(f"  The derivative b_0 should be zero (or very small) if the constant term")
print(f"  is handled correctly.")

print("\n" + "=" * 60)



