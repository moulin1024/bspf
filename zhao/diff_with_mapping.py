

# Example usage and testing
from grid_mapping import (
    build_expr_via_connections_with_values,
    create_simple_mapping,
    build_multi_sigmoid_expr,
    transform_to_unit_interval,
    transform_from_unit_interval,
    validate_domain,
    logistic
)
from chebyshev import chebyshev_derivative_from_values, construct_chebyshev_nodes
# from bfpsm1d import bfpsm1d
from bspf1d import bspf1d
from zhao import zhao2025_spectral_derivative, zhao2025_extend
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from findiff import FinDiff


def zhao2025_derivative_with_extension(x_uniform, y_uniform, *, domain):
    """
    Apply Zhao2025 boundary-interval extension, then compute derivative via FFT.
    Returns dy/dx evaluated on the original (non-extended) uniform grid.
    """
    x_uniform = np.asarray(x_uniform, dtype=float).reshape(-1)
    y_uniform = np.asarray(y_uniform, dtype=float).reshape(-1)
    if x_uniform.size != y_uniform.size:
        raise ValueError("x_uniform and y_uniform must have the same length")
    if x_uniform.size % 2 == 0:
        raise ValueError("Zhao2025 requires odd N (N = 2M+1)")

    # Zhao extension on the uniform grid
    _, y_ext = zhao2025_extend(x_uniform, y_uniform, domain=domain)
    N = x_uniform.size
    M = (N - 1) // 2
    N_ext = y_ext.size

    coeffs = np.fft.fft(y_ext)
    dt = 1.0 / M
    freqs = np.fft.fftfreq(N_ext, d=dt)
    omega = 2.0 * np.pi * freqs
    multiplier = 1j * omega
    if N_ext % 2 == 0:
        multiplier[N_ext // 2] = 0  # Nyquist handling for real signals

    df_ext = np.real(np.fft.ifft(coeffs * multiplier))
    scale = 2.0 / (domain[1] - domain[0])
    return scale * df_ext[:N]

# ------------------------------------------------------------------
# Parameter block - now supports arbitrary domains!
# ------------------------------------------------------------------
DEGREE = 9       # B-spline polynomial degree
ALPHA = 2            # Factor for extra degrees of freedom (basis count)
REG_PARAM = 1e-3      # Tikhonov regularisation strength (lam)

# Test with different domain intervals
# domain = [0, 1]          # Original unit interval
# domain = [-1, 2]         # Different interval  
domain = [0, 2*np.pi]   # Symmetric about zero
# domain = [10, 50]        # Large positive interval

NUM_POINTS = 1001   # Grid resolution
NUM_BOUNDARY_POINTS = DEGREE + 3

# Choose number of B-spline basis functions
N_BASIS = 3 * (DEGREE)

# Grid parameters
clustering_factor = 2.0  # Stronger clustering near endpoints
clustering_flag = True
# grid_sizes = np.geomspace(1000,10000,50).astype(int)#np.arange(600,10001,500)#[100,200,400,800,1600,3200,6400]#np.arange(1000,3001,100)

# Generate grid on the requested domain
x = np.linspace(domain[0], domain[1], NUM_POINTS)
dx = (domain[1] - domain[0]) / (NUM_POINTS - 1)

# Define symbolic variables and function
t = sp.Symbol('t')
phi = sp.Symbol('phi')

# Demonstrate different mapping approaches for arbitrary intervals
print(f"Working with domain: [{domain[0]}, {domain[1]}]")
print("Using Manual sigmoid-based endpoint clustering")

# Manual sigmoid-based endpoint clustering using a SINGLE sigmoid function
# To use only one sigmoid (no connection points):
# - Set p_vals = [] (empty list, no connection points = 1 segment)
# - Set k_vals = [k] (single sharpness value)
# - Set h_vals = [h] (single height value)
# - This creates a single sigmoid function across the entire domain

# # Single sigmoid parameters (no connection points = single segment)
# p_vals = [0.5]  # Empty list = no connection points, single segment
# k_vals = [1.0,1.0]  # Single sharpness value
# h_vals = [1.0,1.0]  # Single height value (controls clustering strength)
# m_val = 0.01  # Baseline slope

# # Build single sigmoid mapping
# phi, dphi, centers = build_expr_via_connections_with_values(
#     p_vals, k_vals, h_vals, m_val, normalize=True, domain=domain
# )


# Method 1: Using existing function with domain parameter
p_vals = [0.5]              # connection points
k_vals = [3.0, 3.0]      # sharpness (>0)
h_vals = [0.5, 0.5]      # heights (>=0)
m_val  = 0.01                    # baseline (>0)

# Build expressions with arbitrary domain support
phi, dphi, centers = build_expr_via_connections_with_values(
    p_vals, k_vals, h_vals, m_val, normalize=True, domain=domain
)

# Method 2: Using the simple mapping interface
# phi_simple, dphi_simple, centers_simple = create_simple_mapping(
#     domain_source=domain, 
#     p_vals=p_vals, k_vals=k_vals, h_vals=h_vals, m_val=m_val
# )

print(f"Mapping centers in domain coordinates: {centers}")
print(f"Domain length: {domain[1] - domain[0]:.3f}")

# Example: Test multiple domains
test_domains = [
    [0, 2*np.pi]          # Large positive interval
]

print("\n" + "="*60)
print("TESTING MAPPING WITH DIFFERENT DOMAINS")
print("="*60)

for test_domain in test_domains:
    print(f"\nDomain: [{test_domain[0]:.3f}, {test_domain[1]:.3f}]")
    
    # Create mapping for this domain
    test_phi, test_dphi, test_centers = create_simple_mapping(
        domain_source=test_domain,
        p_vals=p_vals,  # Two connection points
        k_vals=k_vals,  # Three segments
        h_vals=h_vals,
        m_val=m_val
    )
    
    # Create numerical functions
    test_phi_func = sp.lambdify(t, test_phi, modules='numpy')
    test_dphi_func = sp.lambdify(t, test_dphi, modules='numpy')
    
    # Test points
    test_x = np.linspace(test_domain[0], test_domain[1], 10)
    test_y = test_phi_func(test_x)
    test_dy = test_dphi_func(test_x)
    
    print(f"  Input range:  [{test_x[0]:.3f}, {test_x[-1]:.3f}]")
    print(f"  Output range: [{test_y.min():.3f}, {test_y.max():.3f}]")
    print(f"  Derivative range: [{test_dy.min():.3f}, {test_dy.max():.3f}]")
    print(f"  Centers: {[f'{c:.3f}' for c in test_centers]}")

print("\n" + "="*60)
print("PROCEEDING WITH MAIN ANALYSIS")
print("="*60)

print(f"Single sigmoid mapping parameters:")
print(f"  Connection points (p_vals): {p_vals} (empty = single segment)")
print(f"  Sharpness (k_vals): {k_vals}")
print(f"  Height (h_vals): {h_vals}")
print(f"  Baseline slope (m_val): {m_val}")
print(f"  Mapping centers: {centers}")
print(f"  Domain length: {domain[1] - domain[0]:.3f}")

# Optional: Test mapping on different domains (commented out)
# test_domains = [
#     [0, 2*np.pi]          # Large positive interval
# ]
# 
# print("\n" + "="*60)
# print("TESTING MAPPING WITH DIFFERENT DOMAINS")
# print("="*60)
# 
# for test_domain in test_domains:
#     print(f"\nDomain: [{test_domain[0]:.3f}, {test_domain[1]:.3f}]")
#     
#     # Create mapping for this domain using same parameters
#     test_phi, test_dphi, test_centers = build_expr_via_connections_with_values(
#         p_vals, k_vals, h_vals, m_val, normalize=True, domain=test_domain
#     )
#     
#     # Create numerical functions
#     test_phi_func = sp.lambdify(t, test_phi, modules='numpy')
#     test_dphi_func = sp.lambdify(t, test_dphi, modules='numpy')
#     
#     # Test points
#     test_x = np.linspace(test_domain[0], test_domain[1], 10)
#     test_y = test_phi_func(test_x)
#     test_dy = test_dphi_func(test_x)
#     
#     print(f"  Input range:  [{test_x[0]:.3f}, {test_x[-1]:.3f}]")
#     print(f"  Output range: [{test_y.min():.3f}, {test_y.max():.3f}]")
#     print(f"  Derivative range: [{test_dy.min():.3f}, {test_dy.max():.3f}]")
#     print(f"  Centers: {[f'{c:.3f}' for c in test_centers]}")

print("\n" + "="*60)
print("PROCEEDING WITH MAIN ANALYSIS")
print("="*60)

# Test function and its analytical derivative using symbolic computation
alpha = 100
beta = 1.02

# Create synthetic signal
# f_sym = sp.cos(alpha*phi)
# f_sym_original = sp.cos(alpha*t)

# Create synthetic signal
f_sym = sp.sin(phi/(beta+sp.cos(phi))) + sp.cos(100.5*phi)
f_sym_original = sp.sin(t/(beta+sp.cos(t))) + sp.cos(100.5*t)



# # f_sym =  sp.tanh(200*(phi-np.pi))
# for i in range(n_components):
#     f_sym += magnitudes[i] * sp.cos(frequencies[i]*phi + phases[i])

# # # f_sym_original =  sp.tanh(200*(t-np.pi))
# for i in range(n_components):
#     f_sym_original += magnitudes[i] * sp.cos(frequencies[i]*t + phases[i])



grid_sizes = np.geomspace(100,3000,50).astype(int)#np.arange(600,10001,500)#[100,200,400,800,1600,3200,6400]#np.arange(1000,3001,100)

# Take derivative symbolically
df_sym = sp.diff(f_sym, t)
df_sym_original = sp.diff(f_sym_original, t)

# Convert to numpy functions
test_func = sp.lambdify(t, f_sym, modules='numpy')
test_func_deriv = sp.lambdify(t, df_sym, modules='numpy')
test_phi = sp.lambdify(t, phi, modules='numpy')
test_dphi = sp.lambdify(t, dphi, modules='numpy')

test_func_original = sp.lambdify(t, f_sym_original, modules='numpy')
test_func_deriv_original = sp.lambdify(t, df_sym_original, modules='numpy')



# Compute function values
y = test_func(x)
y_deriv_exact = test_func_deriv(x)

xi = test_phi(x)
yi = test_func(xi)
dphi_exact = test_dphi(x)

y_original = test_func_original(x)
y_deriv_exact_original = test_func_deriv_original(x)

# Initialize bfpsm1d model
model = bspf1d.from_grid(degree=DEGREE,
        x=x,
        n_basis=N_BASIS,
        domain=tuple(domain),
        use_clustering=clustering_flag,
        clustering_factor=2.5,
        order=DEGREE,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        correction="spectral")


# Compute derivatives using different methods
# 1. BSPF method with mapped grid
y_deriv_bfpsm, y_spline = model.differentiate(y, k=1, lam=REG_PARAM)

# 2. BSPF method with original grid
model_orig = bspf1d.from_grid(
    degree=DEGREE,
    x=x,
    n_basis=N_BASIS,
    domain=tuple(domain),
    use_clustering=clustering_flag,
    clustering_factor=3.0,
    order=DEGREE,
    num_boundary_points=NUM_BOUNDARY_POINTS,
    correction="spectral"
)
y_deriv_bfpsm_orig, y_spline_orig = model_orig.differentiate(y_original, k=1, lam=REG_PARAM)

# 3. 4th order finite difference
d_dx = FinDiff(0, dx, 1, acc=4)  # 4th order accurate first derivative
y_deriv_fd = d_dx(y)

# 4. Chebyshev spectral method with original grid
N_cheb = NUM_POINTS - 1  # Chebyshev polynomial degree
x_cheb_orig, _ = construct_chebyshev_nodes(N_cheb, domain=tuple(domain))
y_cheb_orig = test_func_original(x_cheb_orig)
y_deriv_cheb_orig = chebyshev_derivative_from_values(y_cheb_orig, x_cheb_orig, domain=tuple(domain))

# 5. Zhao2025 spectral method (requires odd number of points and uniform grid)
# Ensure odd number of points for Zhao
if NUM_POINTS % 2 == 0:
    x_zhao = np.linspace(domain[0], domain[1], NUM_POINTS + 1)  # Make odd
else:
    x_zhao = x.copy()
y_zhao = test_func_original(x_zhao)
y_deriv_zhao = zhao2025_spectral_derivative(x_zhao, y_zhao, order=1, domain=tuple(domain))
# Interpolate back to original grid if needed
if len(x_zhao) != len(x):
    from scipy.interpolate import interp1d
    zhao_interp = interp1d(x_zhao, y_deriv_zhao, kind='cubic', bounds_error=False, fill_value='extrapolate')
    y_deriv_zhao = zhao_interp(x)

# 6. Zhao2025 spectral method with coordinate mapping
# Here x is the uniform computational grid, and xi = phi(x) is the physical grid.
# Apply Zhao extension in uniform x, then convert to physical derivative dy/dxi.
if NUM_POINTS % 2 == 0:
    x_zhao_mapped = np.linspace(domain[0], domain[1], NUM_POINTS + 1)  # Make odd
else:
    x_zhao_mapped = x.copy()

xi_zhao_mapped = test_phi(x_zhao_mapped)
y_zhao_mapped = test_func_original(xi_zhao_mapped)

# Zhao extension + FFT derivative on uniform x
y_deriv_zhao_x = zhao2025_derivative_with_extension(
    x_zhao_mapped, y_zhao_mapped, domain=tuple(domain)
)

# Transform to physical derivative: dy/dxi = (dy/dx) / (dphi/dx)
dphi_zhao_mapped = test_dphi(x_zhao_mapped)
dphi_zhao_mapped = np.where(np.abs(dphi_zhao_mapped) < 1e-12, 1e-12, dphi_zhao_mapped)
y_deriv_zhao_mapped = y_deriv_zhao_x / dphi_zhao_mapped

# Interpolate back to original grid if needed
if len(x_zhao_mapped) != len(x):
    from scipy.interpolate import interp1d
    zhao_mapped_interp = interp1d(x_zhao_mapped, y_deriv_zhao_mapped, kind='cubic', bounds_error=False, fill_value='extrapolate')
    y_deriv_zhao_mapped = zhao_mapped_interp(x)

# Compute errors for each (L∞ norm for single point comparison)
error_bfpsm = np.max(np.abs((y_deriv_bfpsm/dphi_exact - y_deriv_exact/dphi_exact)**1))
error_bfpsm_orig = np.max(np.abs((y_deriv_bfpsm_orig - y_deriv_exact_original)**1))

# Chebyshev errors
y_deriv_exact_cheb_orig = test_func_deriv_original(x_cheb_orig)
error_cheb_orig = np.max(np.abs(y_deriv_cheb_orig - y_deriv_exact_cheb_orig))

# Zhao errors
y_deriv_exact_zhao = test_func_deriv_original(x)
error_zhao = np.max(np.abs(y_deriv_zhao - y_deriv_exact_zhao))

# Zhao (mapped) errors (physical coordinate xi = phi(x))
y_deriv_exact_zhao_mapped = test_func_deriv_original(xi)
error_zhao_mapped = np.max(np.abs(y_deriv_zhao_mapped - y_deriv_exact_zhao_mapped))

print("Errors (L^inf Norm):")
print("BSPF (mapped):", error_bfpsm)
print("BSPF (original):", error_bfpsm_orig)
print("Chebyshev (original):", error_cheb_orig)
print("Zhao2025:", error_zhao)
print("Zhao2025 (mapped):", error_zhao_mapped)

errors_bfpsm = []
errors_bfpsm_orig = []
errors_cheb_orig = []
errors_zhao = []
errors_zhao_mapped = []
for n_points in grid_sizes:
    print(f"Processing grid size: {n_points}")
    # Create grid
    x_test = np.linspace(domain[0], domain[1], n_points)
    xi_test = test_phi(x_test)
    dx_test = (domain[1] - domain[0]) / (n_points - 1)
    
    # Compute exact solution
    y_test = test_func(x_test)
    y_deriv_exact_test = test_func_deriv(x_test)
    dphi_test = test_dphi(x_test)

    y_test_original = test_func_original(x_test)
    y_deriv_exact_test_original = test_func_deriv_original(x_test)
    
    # BSPF method with mapped grid
    model_test = bspf1d.from_grid(
        degree=DEGREE,
        x=x_test,
        n_basis=N_BASIS,
        domain=tuple(domain),
        use_clustering=clustering_flag,
        clustering_factor=2.5,
        order=DEGREE,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        correction="spectral"
    )
    y_deriv_bfpsm_test, _ = model_test.differentiate(y_test, k=1, lam=REG_PARAM)
    # L2 norm: ||e||_2 = sqrt(integral |e|^2 dx) ≈ sqrt(sum |e_i|^2 * dx)
    error_bfpsm_test = (y_deriv_bfpsm_test/dphi_test - y_deriv_exact_test/dphi_test)
    errors_bfpsm.append(np.linalg.norm(error_bfpsm_test, ord=2) * np.sqrt(dx_test))
    
    # BSPF method with original grid
    model_orig_test = bspf1d.from_grid(
        degree=DEGREE,
        x=x_test,
        n_basis=N_BASIS,
        domain=tuple(domain),
        use_clustering=clustering_flag,
        clustering_factor=3.0,
        order=DEGREE,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        correction="spectral"
    )
    y_deriv_bfpsm_orig_test, _ = model_orig_test.differentiate(y_test_original, k=1, lam=REG_PARAM)
    error_bfpsm_orig_test = (y_deriv_bfpsm_orig_test - y_deriv_exact_test_original)
    errors_bfpsm_orig.append(np.linalg.norm(error_bfpsm_orig_test, ord=2) * np.sqrt(dx_test))
    
    # Chebyshev method with original grid
    N_cheb_test = n_points - 1
    x_cheb_test, _ = construct_chebyshev_nodes(N_cheb_test, domain=tuple(domain))
    y_cheb_test = test_func_original(x_cheb_test)
    y_deriv_cheb_test = chebyshev_derivative_from_values(y_cheb_test, x_cheb_test, domain=tuple(domain))
    y_deriv_exact_cheb_test = test_func_deriv_original(x_cheb_test)
    error_cheb_test = (y_deriv_cheb_test - y_deriv_exact_cheb_test)
    # For Chebyshev nodes, use approximate spacing
    dx_cheb_approx = (domain[1] - domain[0]) / (n_points - 1)
    errors_cheb_orig.append(np.linalg.norm(error_cheb_test, ord=2) * np.sqrt(dx_cheb_approx))
    
    # Zhao2025 method (requires odd number of points)
    n_points_zhao = n_points if n_points % 2 == 1 else n_points + 1
    x_zhao_test = np.linspace(domain[0], domain[1], n_points_zhao)
    y_zhao_test = test_func_original(x_zhao_test)
    y_deriv_zhao_test = zhao2025_spectral_derivative(x_zhao_test, y_zhao_test, order=1, domain=tuple(domain))
    y_deriv_exact_zhao_test = test_func_deriv_original(x_zhao_test)
    error_zhao_test = (y_deriv_zhao_test - y_deriv_exact_zhao_test)
    dx_zhao_test = (domain[1] - domain[0]) / (n_points_zhao - 1)
    errors_zhao.append(np.linalg.norm(error_zhao_test, ord=2) * np.sqrt(dx_zhao_test))
    
    # Zhao2025 method with coordinate mapping
    # Uniform x grid, physical xi = phi(x)
    x_zhao_mapped_test = np.linspace(domain[0], domain[1], n_points_zhao)
    xi_zhao_mapped_test = test_phi(x_zhao_mapped_test)
    
    # Evaluate function on physical coordinates
    y_zhao_mapped_test = test_func_original(xi_zhao_mapped_test)
    # Apply Zhao in uniform x space with extension + truncation
    y_deriv_zhao_x_test = zhao2025_derivative_with_extension(
        x_zhao_mapped_test, y_zhao_mapped_test, domain=tuple(domain)
    )
    # Transform to physical derivative: dy/dxi = (dy/dx) / (dphi/dx)
    dphi_zhao_mapped_test = test_dphi(x_zhao_mapped_test)
    dphi_zhao_mapped_test = np.where(np.abs(dphi_zhao_mapped_test) < 1e-12, 1e-12, dphi_zhao_mapped_test)
    y_deriv_zhao_mapped_test = y_deriv_zhao_x_test / dphi_zhao_mapped_test
    # Compute error in physical coordinate xi
    y_deriv_exact_zhao_mapped_test = test_func_deriv_original(xi_zhao_mapped_test)
    error_zhao_mapped_test = (y_deriv_zhao_mapped_test - y_deriv_exact_zhao_mapped_test)
    dx_zhao_mapped_test = (domain[1] - domain[0]) / (n_points_zhao - 1)
    errors_zhao_mapped.append(np.linalg.norm(error_zhao_mapped_test, ord=2) * np.sqrt(dx_zhao_mapped_test))


# Set up global plotting parameters
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
# Create figure with custom grid layout
fig = plt.figure(figsize=(15, 10))  
gs = fig.add_gridspec(2, 2)  # 2x2 grid

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
n_grid_points = 3000
x_fine = np.linspace(domain[0], domain[1], n_grid_points)

# ==== Panel (a): Original function ====
ax1 = fig.add_subplot(gs[0, 0])
y_fine = test_func_original(x_fine)
ax1.plot(x_fine, y_fine, '-', color=default_colors[0], linewidth=1.5, label='$y(x)$')
ax1.set_xlabel('$x$ (physical coordinate)')
ax1.set_ylabel('$y(x)$')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_title('(a)', loc='left', x=-0.06, fontsize=12, fontweight='bold')

# ==== Panel (b): Mapping function ====
ax2 = fig.add_subplot(gs[0, 1])
xi_fine = test_phi(x_fine)
dphi_fine = test_dphi(x_fine)

# Primary plot - mapping function
ax2.plot(xi_fine, x_fine, '-', color=default_colors[0], linewidth=1, label='$\zeta(x)$')
ax2.plot(x_fine, x_fine, '--', color='gray', alpha=0.7, linewidth=1, label='$\zeta(x) = x$')

# Show some grid points
n_sample = 100
x_sample = np.linspace(domain[0], domain[1], n_sample)
xi_sample = test_phi(x_sample)
ax2.plot(xi_sample, np.zeros_like(xi_sample), 'o', color=default_colors[4], markersize=2, alpha=0.8, label='Grid points')

# Chebyshev-equivalent mapping function xi(x) on uniform x
s_fine = (x_fine - domain[0]) / (domain[1] - domain[0])
xi_cheb_map = 0.5 * (domain[0] + domain[1]) + 0.5 * (domain[1] - domain[0]) * np.cos(np.pi * (1.0 - s_fine))
ax2.plot(xi_cheb_map, x_fine, '--', color=default_colors[2], linewidth=1, label='Chebyshev mapping')

for i in range(len(x_sample)):
    ax2.annotate('', xy=(xi_sample[i], 0), xytext=(xi_sample[i], x_sample[i]),
                 arrowprops=dict(arrowstyle='-', color=default_colors[4], alpha=0.8, lw=1))

ax2.set_ylabel('$x$')
ax2.set_xlabel('$\zeta(x)$')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_title('(b)', loc='left', x=-0.06, fontsize=12, fontweight='bold')

# ==== Panel (c): Error distribution ====
ax3 = fig.add_subplot(gs[1, 0])
ax3.semilogy(xi, np.abs(y_deriv_bfpsm - y_deriv_exact)/dphi_exact,
             '-', label='BSPF (mapped)', color=default_colors[4], linewidth=1)
ax3.semilogy(x, np.abs(y_deriv_bfpsm_orig - y_deriv_exact_original),
             '-', label='BSPF (original)', color=default_colors[0], linewidth=1)
ax3.semilogy(x_cheb_orig, np.abs(y_deriv_cheb_orig - y_deriv_exact_cheb_orig),
             '--', label='Chebyshev', color=default_colors[2], linewidth=1, marker='s', markersize=3)
ax3.semilogy(x, np.abs(y_deriv_zhao - y_deriv_exact_zhao),
             '-.', label='Zhao2025', color=default_colors[1], linewidth=1, marker='o', markersize=3)
# Directly plot on physical coordinate xi (no interpolation)
xi_zhao_mapped_plot = test_phi(x_zhao_mapped)
error_zhao_mapped_plot = np.abs(
    y_deriv_zhao_mapped - test_func_deriv_original(xi_zhao_mapped_plot)
)
ax3.semilogy(xi_zhao_mapped_plot, error_zhao_mapped_plot,
             ':', label='Zhao2025 (mapped)', color=default_colors[3], linewidth=1, marker='^', markersize=3)
ax3.set_xlabel('$x$')
ax3.set_ylabel('$|Error|$')
ax3.legend(fontsize=10)
ax3.set_title('(c)', loc='left', x=-0.15, fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# ==== Panel (d): Convergence study ====
ax4 = fig.add_subplot(gs[1, 1])
ax4.loglog(grid_sizes, errors_bfpsm_orig, '.-', label='BSPF (original)', color=default_colors[0], linewidth=1)
ax4.loglog(grid_sizes, errors_bfpsm, '.-', label='BSPF (mapped)', color=default_colors[4], linewidth=1)
ax4.loglog(grid_sizes, errors_cheb_orig, 's-', label='Chebyshev', color=default_colors[2], linewidth=1, markersize=4)
ax4.loglog(grid_sizes, errors_zhao, 'o-', label='Zhao2025', color=default_colors[1], linewidth=1, markersize=4)
ax4.loglog(grid_sizes, errors_zhao_mapped, '^-', label='Zhao2025 (mapped)', color=default_colors[3], linewidth=1, markersize=4)
ax4.plot([NUM_POINTS, NUM_POINTS], [1e-12, 1e6], '--', color='gray', linewidth=1.5)
ax4.text(NUM_POINTS + 10, 2*1e-10, f'$(d): N = {NUM_POINTS}$', color='gray', fontsize=12)
ax4.set_xlabel('$N$', fontsize=12)
ax4.set_ylabel('$\Vert Error \Vert_{2}$')
ax4.set_title('(d)', loc='left', x=-0.15, fontsize=12, fontweight='bold')
ax4.set_ylim(1e-12, 0.9*1e5)
ax4.grid(True)
ax4.legend(fontsize=10)

plt.tight_layout()
plt.show()
# plt.savefig('figs/fig2.pdf', dpi=300, bbox_inches='tight')
