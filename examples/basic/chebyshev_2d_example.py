"""
Standalone 2D Chebyshev Differentiation Example.

This script demonstrates how to use the spectral-derivatives library
for computing derivatives of 2D fields using Chebyshev spectral methods.

Uses synthetic turbulence data with zero-flux boundary conditions.

Based on examples from the spectral-derivatives library.
"""

import numpy as np
import matplotlib.pyplot as plt
from specderiv import cheb_deriv


# ============================================================================
# Parameters
# ============================================================================

# Grid parameters
N = 1024  # Number of intervals (N+1 nodes)
domain_x = [0, 2*np.pi]  # Domain [a, b] for x
domain_y = [0, 2*np.pi]  # Domain [c, d] for y

# Turbulence parameters
TURB_N_MODES = 128  # Number of Fourier modes
TURB_SEED = 123    # Random seed for reproducibility


# ============================================================================
# Create Chebyshev Nodes
# ============================================================================

# Construct Chebyshev nodes: cosine-spaced points on [-1, 1]
# x_n = cos(π * n / N) for n = 0, ..., N
x_n_canonical = np.cos(np.arange(N + 1) * np.pi / N)  # [-1, 1]
y_n_canonical = np.cos(np.arange(N + 1) * np.pi / N)  # [-1, 1]

# Map to domain [a, b]: x = (b-a)/2 * t + (b+a)/2
a_x, b_x = domain_x
a_y, b_y = domain_y
x_n = x_n_canonical * (b_x - a_x) / 2.0 + (b_x + a_x) / 2.0
y_n = y_n_canonical * (b_y - a_y) / 2.0 + (b_y + a_y) / 2.0

# Create 2D grid using meshgrid
# Note: meshgrid returns (X, Y) where X varies along columns (axis=1) and Y varies along rows (axis=0)
X_n, Y_n = np.meshgrid(x_n, y_n)  # Shape: (N+1, N+1)


# ============================================================================
# Turbulence Field Generation
# ============================================================================

def random_turbulence_signal_2d_with_derivatives(
    Lx=1.0, 
    Ly=1.0,
    Nx=128, 
    Ny=128,
    nmodes=50, 
    seed=None
):
    """
    Generate a 2D turbulence-like signal using cosine modes with
    a Kolmogorov-like spectrum, and return analytical first
    and second derivatives (Neumann BC automatically satisfied).
    
    Zero-flux (Neumann) boundary conditions are analytically satisfied:
    - Uses cosine modes: u(x,y) = sum(a_i * cos(kx_i * x) * cos(ky_i * y))
    - with kx_i = n_i * π / Lx, ky_i = m_i * π / Ly
    - Derivatives automatically satisfy zero-flux BC at boundaries
    
    Returns data in (ny, nx) format.
    
    Parameters
    ----------
    Lx, Ly : float
        Domain lengths
    Nx, Ny : int
        Number of grid points
    nmodes : int
        Number of Fourier modes (uses nmodes x nmodes modes)
    seed : int or None
        Random seed for reproducibility
    
    Returns
    -------
    x, y : ndarray
        Grid points
    u : ndarray
        Field values, shape (ny, nx)
    u_x : ndarray
        First derivative with respect to x, shape (ny, nx)
    u_y : ndarray
        First derivative with respect to y, shape (ny, nx)
    u_xx : ndarray
        Second derivative with respect to x, shape (ny, nx)
    u_yy : ndarray
        Second derivative with respect to y, shape (ny, nx)
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)  # (ny, nx)

    # Mode indices n, m = 1 ... nmodes
    n = np.arange(1, nmodes + 1)
    m = np.arange(1, nmodes + 1)
    
    # Wavenumbers
    kx = n * np.pi / Lx
    ky = m * np.pi / Ly

    # Kolmogorov-like scaling: E(k) ~ k^(-5/3) → amplitude ~ k^(-5/6)
    # Use combined wavenumber magnitude
    k_mag = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
    sigma = k_mag**(-5/6)
    
    # Random amplitudes (flatten and sample)
    n_total = nmodes * nmodes
    indices = np.random.choice(n_total, size=min(nmodes, n_total), replace=False)
    i_flat, j_flat = np.unravel_index(indices, (nmodes, nmodes))
    
    # Allocate arrays (ny, nx)
    u = np.zeros((Ny, Nx))
    u_x = np.zeros((Ny, Nx))
    u_y = np.zeros((Ny, Nx))
    u_xx = np.zeros((Ny, Nx))
    u_yy = np.zeros((Ny, Nx))

    # Build signal + exact derivatives
    for idx in range(len(i_flat)):
        ni = n[i_flat[idx]]
        mi = m[j_flat[idx]]
        
        kx_val = ni * np.pi / Lx
        ky_val = mi * np.pi / Ly
        ai = sigma[i_flat[idx], j_flat[idx]] * np.random.randn()
        
        cos_x = np.cos(kx_val * X)
        cos_y = np.cos(ky_val * Y)
        sin_x = np.sin(kx_val * X)
        sin_y = np.sin(ky_val * Y)
        
        # u = a * cos(kx*x) * cos(ky*y)
        u += ai * cos_x * cos_y
        
        # u_x = -a * kx * sin(kx*x) * cos(ky*y)
        u_x += ai * (-kx_val) * sin_x * cos_y
        
        # u_y = -a * ky * cos(kx*x) * sin(ky*y)
        u_y += ai * (-ky_val) * cos_x * sin_y
        
        # u_xx = -a * kx^2 * cos(kx*x) * cos(ky*y)
        u_xx += ai * (-kx_val**2) * cos_x * cos_y
        
        # u_yy = -a * ky^2 * cos(kx*x) * cos(ky*y)
        u_yy += ai * (-ky_val**2) * cos_x * cos_y

    # Remove mean (common in turbulence signals)
    u -= np.mean(u)

    return x, y, u, u_x, u_y, u_xx, u_yy


# ============================================================================
# Generate Turbulence Field on Chebyshev Grid
# ============================================================================

# Generate turbulence field parameters (fixed by seed)
Lx = domain_x[1] - domain_x[0]
Ly = domain_y[1] - domain_y[0]

np.random.seed(TURB_SEED)
n = np.arange(1, TURB_N_MODES + 1)
m = np.arange(1, TURB_N_MODES + 1)
kx = n * np.pi / Lx
ky = m * np.pi / Ly
k_mag = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
sigma = k_mag**(-5/6)
n_total = TURB_N_MODES * TURB_N_MODES
indices = np.random.choice(n_total, size=min(TURB_N_MODES, n_total), replace=False)
i_flat, j_flat = np.unravel_index(indices, (TURB_N_MODES, TURB_N_MODES))

# Store mode parameters for evaluation
modes = []
for idx in range(len(i_flat)):
    ni = n[i_flat[idx]]
    mi = m[j_flat[idx]]
    kx_val = ni * np.pi / Lx
    ky_val = mi * np.pi / Ly
    ai = sigma[i_flat[idx], j_flat[idx]] * np.random.randn()
    modes.append((ni, mi, kx_val, ky_val, ai))

# Evaluate turbulence at Chebyshev nodes
f_n = np.zeros_like(X_n)
df_dx_exact_n = np.zeros_like(X_n)
df_dy_exact_n = np.zeros_like(X_n)
d2f_dx2_exact_n = np.zeros_like(X_n)
d2f_dy2_exact_n = np.zeros_like(X_n)
d2f_dxdy_exact_n = np.zeros_like(X_n)

for ni, mi, kx_val, ky_val, ai in modes:
    cos_x = np.cos(kx_val * X_n)
    cos_y = np.cos(ky_val * Y_n)
    sin_x = np.sin(kx_val * X_n)
    sin_y = np.sin(ky_val * Y_n)
    
    # Field
    f_n += ai * cos_x * cos_y
    
    # First derivatives
    df_dx_exact_n += ai * (-kx_val) * sin_x * cos_y
    df_dy_exact_n += ai * (-ky_val) * cos_x * sin_y
    
    # Second derivatives
    d2f_dx2_exact_n += ai * (-kx_val**2) * cos_x * cos_y
    d2f_dy2_exact_n += ai * (-ky_val**2) * cos_x * cos_y
    
    # Mixed derivative
    d2f_dxdy_exact_n += ai * kx_val * ky_val * sin_x * sin_y

# Remove mean (common in turbulence signals)
f_n -= np.mean(f_n)

# Laplacian
laplacian_exact_n = d2f_dx2_exact_n + d2f_dy2_exact_n


# ============================================================================
# Compute Derivatives Using Chebyshev Spectral Method
# ============================================================================

print("=" * 60)
print("2D Chebyshev Differentiation Example with Synthetic Turbulence")
print("=" * 60)
print(f"Grid size: {N+1} x {N+1}")
print(f"Domain x: [{domain_x[0]}, {domain_x[1]}]")
print(f"Domain y: [{domain_y[0]}, {domain_y[1]}]")
print(f"Turbulence modes: {TURB_N_MODES}")
print(f"Random seed: {TURB_SEED}")
print()

# First derivatives
# For x-derivative: differentiate along axis=1 (columns, x-direction)
# For y-derivative: differentiate along axis=0 (rows, y-direction)
print("Computing first derivatives...")
df_dx_cheb = cheb_deriv(f_n, x_n, order=1, axis=1)  # x-derivative
df_dy_cheb = cheb_deriv(f_n, y_n, order=1, axis=0)  # y-derivative

# Second derivatives
print("Computing second derivatives...")
d2f_dx2_cheb = cheb_deriv(f_n, x_n, order=2, axis=1)  # ∂²f/∂x²
d2f_dy2_cheb = cheb_deriv(f_n, y_n, order=2, axis=0)  # ∂²f/∂y²

# Mixed derivative: differentiate first w.r.t. x, then w.r.t. y
# Or: differentiate first w.r.t. y, then w.r.t. x (should give same result)
print("Computing mixed derivative...")
d2f_dxdy_cheb = cheb_deriv(cheb_deriv(f_n, x_n, order=1, axis=1), y_n, order=1, axis=0)

# Laplacian: ∇²f = ∂²f/∂x² + ∂²f/∂y²
print("Computing Laplacian...")
laplacian_cheb = d2f_dx2_cheb + d2f_dy2_cheb

# Exact derivatives are already computed above


# ============================================================================
# Compute Errors
# ============================================================================

print("\n" + "=" * 60)
print("Error Analysis (L² and L∞ norms)")
print("=" * 60)

# Approximate grid spacing for error normalization
dx = (domain_x[1] - domain_x[0]) / N
dy = (domain_y[1] - domain_y[0]) / N

errors = {
    'df_dx': {
        'cheb': df_dx_cheb,
        'exact': df_dx_exact_n,
        'name': '∂f/∂x'
    },
    'df_dy': {
        'cheb': df_dy_cheb,
        'exact': df_dy_exact_n,
        'name': '∂f/∂y'
    },
    'd2f_dx2': {
        'cheb': d2f_dx2_cheb,
        'exact': d2f_dx2_exact_n,
        'name': '∂²f/∂x²'
    },
    'd2f_dy2': {
        'cheb': d2f_dy2_cheb,
        'exact': d2f_dy2_exact_n,
        'name': '∂²f/∂y²'
    },
    'd2f_dxdy': {
        'cheb': d2f_dxdy_cheb,
        'exact': d2f_dxdy_exact_n,
        'name': '∂²f/∂x∂y'
    },
    'laplacian': {
        'cheb': laplacian_cheb,
        'exact': laplacian_exact_n,
        'name': '∇²f'
    }
}

for key, data in errors.items():
    error = data['cheb'] - data['exact']
    l2_error = np.linalg.norm(error, 'fro') * np.sqrt(dx * dy)
    linf_error = np.max(np.abs(error))
    print(f"{data['name']:15s} | L²: {l2_error:.6e} | L∞: {linf_error:.6e}")


# ============================================================================
# Visualization
# ============================================================================

print("\n" + "=" * 60)
print("Creating visualization...")
print("=" * 60)

# Set up plotting parameters
plt.rcParams.update({
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14
})

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# (a) Original function
ax1 = plt.subplot(3, 3, 1)
im1 = ax1.contourf(x_n, y_n, f_n, levels=20, cmap='viridis')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
ax1.set_title('(a) Turbulence Field $f(x,y)$')
plt.colorbar(im1, ax=ax1)

# (b) Exact ∂f/∂x
ax2 = plt.subplot(3, 3, 2)
im2 = ax2.contourf(x_n, y_n, df_dx_exact_n, levels=20, cmap='RdBu')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$y$')
ax2.set_title('(b) Exact $\\partial f/\\partial x$')
plt.colorbar(im2, ax=ax2)

# (c) Chebyshev ∂f/∂x
ax3 = plt.subplot(3, 3, 3)
im3 = ax3.contourf(x_n, y_n, df_dx_cheb, levels=20, cmap='RdBu')
ax3.set_xlabel('$x$')
ax3.set_ylabel('$y$')
ax3.set_title('(c) Chebyshev $\\partial f/\\partial x$')
plt.colorbar(im3, ax=ax3)

# (d) Exact ∂f/∂y
ax4 = plt.subplot(3, 3, 4)
im4 = ax4.contourf(x_n, y_n, df_dy_exact_n, levels=20, cmap='RdBu')
ax4.set_xlabel('$x$')
ax4.set_ylabel('$y$')
ax4.set_title('(d) Exact $\\partial f/\\partial y$')
plt.colorbar(im4, ax=ax4)

# (e) Chebyshev ∂f/∂y
ax5 = plt.subplot(3, 3, 5)
im5 = ax5.contourf(x_n, y_n, df_dy_cheb, levels=20, cmap='RdBu')
ax5.set_xlabel('$x$')
ax5.set_ylabel('$y$')
ax5.set_title('(e) Chebyshev $\\partial f/\\partial y$')
plt.colorbar(im5, ax=ax5)

# (f) Exact Laplacian
ax6 = plt.subplot(3, 3, 6)
im6 = ax6.contourf(x_n, y_n, laplacian_exact_n, levels=20, cmap='RdBu')
ax6.set_xlabel('$x$')
ax6.set_ylabel('$y$')
ax6.set_title('(f) Exact $\\nabla^2 f$')
plt.colorbar(im6, ax=ax6)

# (g) Chebyshev Laplacian
ax7 = plt.subplot(3, 3, 7)
im7 = ax7.contourf(x_n, y_n, laplacian_cheb, levels=20, cmap='RdBu')
ax7.set_xlabel('$x$')
ax7.set_ylabel('$y$')
ax7.set_title('(g) Chebyshev $\\nabla^2 f$')
plt.colorbar(im7, ax=ax7)

# (h) Error in ∂f/∂x
ax8 = plt.subplot(3, 3, 8)
error_dx = np.abs(df_dx_cheb - df_dx_exact_n)
im8 = ax8.contourf(x_n, y_n, error_dx, levels=20, cmap='hot', norm=plt.colors.LogNorm())
ax8.set_xlabel('$x$')
ax8.set_ylabel('$y$')
ax8.set_title('(h) Error $|\\partial f/\\partial x|$')
plt.colorbar(im8, ax=ax8)

# (i) Error in Laplacian
ax9 = plt.subplot(3, 3, 9)
error_laplacian = np.abs(laplacian_cheb - laplacian_exact_n)
im9 = ax9.contourf(x_n, y_n, error_laplacian, levels=20, cmap='hot', norm=plt.colors.LogNorm())
ax9.set_xlabel('$x$')
ax9.set_ylabel('$y$')
ax9.set_title('(i) Error $|\\nabla^2 f|$')
plt.colorbar(im9, ax=ax9)

plt.tight_layout()
plt.savefig('chebyshev_2d_example.png', dpi=150, bbox_inches='tight')
print("Saved plot to chebyshev_2d_example.png")
plt.show()

print("\n" + "=" * 60)
print("Example completed successfully!")
print("=" * 60)

