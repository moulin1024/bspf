

# Example usage and testing
from bspf.utils import (
    chebyshev_derivative_from_values, 
    _construct_chebyshev_nodes,
    build_expr_via_connections_with_values,
    create_simple_mapping,
    build_multi_sigmoid_expr,
    transform_to_unit_interval,
    transform_from_unit_interval,
    validate_domain,
    logistic
)
# from bfpsm1d import bfpsm1d
from bspf import bspf1d
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from findiff import FinDiff

# ------------------------------------------------------------------
# Parameter block - now supports arbitrary domains!
# ------------------------------------------------------------------
DEGREE = 5       # B-spline polynomial degree
ALPHA = 2            # Factor for extra degrees of freedom (basis count)
REG_PARAM = 1e-3      # Tikhonov regularisation strength (lam)

# Test with different domain intervals
# domain = [0, 1]          # Original unit interval
# domain = [-1, 2]         # Different interval  
domain = [0, 2*np.pi]   # Symmetric about zero
# domain = [10, 50]        # Large positive interval

NUM_POINTS = 1900   # Grid resolution
NUM_BOUNDARY_POINTS = DEGREE + 5

# Choose number of B-spline basis functions
N_BASIS = 2 * (DEGREE)

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

# Method 1: Using existing function with domain parameter
p_vals = [0.5]              # connection points
k_vals = [4.0, 4.0]      # sharpness (>0)
h_vals = [0.5, 0.5]      # heights (>=0)
m_val  = 0.1                    # baseline (>0)

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
        p_vals=[0.5],  # Two connection points
        k_vals=[5.0, 5.0],  # Three segments
        h_vals=[0.25, 0.25],
        m_val=0.01
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

# Test function and its analytical derivative using symbolic computation
alpha = 100
beta = 1.01

# Create synthetic signal
f_sym = sp.sin(phi/(beta+sp.cos(phi)))
f_sym_original = sp.sin(t/(beta+sp.cos(t)))

# # f_sym =  sp.tanh(200*(phi-np.pi))
# for i in range(n_components):
#     f_sym += magnitudes[i] * sp.cos(frequencies[i]*phi + phases[i])

# # # f_sym_original =  sp.tanh(200*(t-np.pi))
# for i in range(n_components):
#     f_sym_original += magnitudes[i] * sp.cos(frequencies[i]*t + phases[i])



grid_sizes = np.geomspace(1000,10000,50).astype(int)#np.arange(600,10001,500)#[100,200,400,800,1600,3200,6400]#np.arange(1000,3001,100)

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
x_cheb_orig, _ = _construct_chebyshev_nodes(N_cheb, domain=tuple(domain))
y_cheb_orig = test_func_original(x_cheb_orig)
y_deriv_cheb_orig = chebyshev_derivative_from_values(y_cheb_orig, x_cheb_orig, domain=tuple(domain))

# 5. Chebyshev spectral method with mapped grid (interpolate to Chebyshev nodes)
x_cheb_mapped, _ = _construct_chebyshev_nodes(N_cheb, domain=tuple(domain))
xi_cheb_mapped = test_phi(x_cheb_mapped)
y_cheb_mapped = test_func(xi_cheb_mapped)
y_deriv_cheb_mapped_raw = chebyshev_derivative_from_values(y_cheb_mapped, x_cheb_mapped, domain=tuple(domain))
# Apply chain rule for mapping
dphi_cheb_mapped = test_dphi(x_cheb_mapped)
y_deriv_cheb_mapped = y_deriv_cheb_mapped_raw / dphi_cheb_mapped

# Compute errors for each 
error_bfpsm = np.max(np.abs((y_deriv_bfpsm/dphi_exact - y_deriv_exact/dphi_exact)**1))  # L2 norm
error_bfpsm_orig = np.max(np.abs((y_deriv_bfpsm_orig - y_deriv_exact_original)**1))


print("Errors (L^inf Norm):")
print("BSPF (mapped):", error_bfpsm)
print("BSPF (original):", error_bfpsm_orig)

errors_bfpsm = []
errors_bfpsm_orig = []
errors_fd = []
errors_cheb_orig = []
errors_cheb_mapped = []
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
    errors_bfpsm.append(np.max(np.abs((y_deriv_bfpsm_test/dphi_test - y_deriv_exact_test/dphi_test)**1)))
    
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
    errors_bfpsm_orig.append(np.max(np.abs((y_deriv_bfpsm_orig_test - y_deriv_exact_test_original)**1)))


# Set up global plotting parameters
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
# Create figure with custom grid layout
fig = plt.figure(figsize=(16, 10))  
gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1])  # two rows, first taller

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# ==== First subplot spanning full row ====
ax1 = fig.add_subplot(gs[0, :])  
n_grid_points = 3000
x_fine = np.linspace(domain[0], domain[1], n_grid_points)
xi_fine = test_phi(x_fine)
dphi_fine = test_dphi(x_fine)

# Primary plot - mapping function
ax1.plot(xi_fine, x_fine, '-', color=default_colors[0], linewidth=1, label='$\zeta(x)$')
ax1.plot(x_fine, x_fine, '--', color='gray', alpha=0.7, linewidth=1, label='$\zeta(x) = x$')

# Show some grid points
n_sample = 300
x_sample = np.linspace(domain[0], domain[1], n_sample)
xi_sample = test_phi(x_sample)
ax1.plot(xi_sample, np.zeros_like(xi_sample), 'o', color=default_colors[4], markersize=2, alpha=0.8, label='Grid points')

for i in range(len(x_sample)):
    ax1.annotate('', xy=(xi_sample[i], 0), xytext=(xi_sample[i], x_sample[i]),
                 arrowprops=dict(arrowstyle='-', color=default_colors[4], alpha=0.8, lw=1))

ax1.set_ylabel('$x$')
ax1.set_xlabel('$\zeta(x)$')
ax1.legend(loc='upper left', fontsize=16)
ax1.grid(True, alpha=0.3)
ax1.set_title('(a)', loc='left', x=-0.06, fontsize=24, fontweight='bold')
ax1.text(1.5, 5, 
         '$\zeta(x)=0.39 x-0.0036+\\frac{1.93}{1+1.54 \\times 10^{8} e^{-4 x}}+\\frac{1.93}{1+535.49 e^{-4 x}}$', 
         fontsize=18)
#  0.3863*t - 0.0036 + 1.9316/(1 + 153552935.3954*exp(-4.0000*t)) + 1.9316/(1 + 535.4917*exp(-4.0000*t))
# 0.39*t - 0.00 + 1.93/(1 + 153552935.40*exp(-4.00*t)) + 1.93/(1 + 535.49*exp(-4.00*t))
# ==== Second subplot (bottom-left) ====
ax2 = fig.add_subplot(gs[1, 0])
ax2.semilogy(xi, np.abs(y_deriv_bfpsm - y_deriv_exact)/dphi_exact,
             '-', label='BSPF (mapped)', color=default_colors[4], linewidth=1)
ax2.semilogy(x, np.abs(y_deriv_bfpsm_orig - y_deriv_exact_original),
             '-', label='BSPF (original)', linewidth=1)
ax2.set_xlabel('$x$')
ax2.set_ylabel('$|Error|$')
ax2.legend(fontsize=16)
ax2.set_title('(b)', loc='left', x=-0.15, fontsize=24, fontweight='bold')

# ==== Third subplot (bottom-right) ====
ax3 = fig.add_subplot(gs[1, 1])
ax3.loglog(grid_sizes, errors_bfpsm_orig, '.-', label='BSPF (original)', linewidth=1)
ax3.loglog(grid_sizes, errors_bfpsm, '.-', label='BSPF (mapped)', color=default_colors[4], linewidth=1)
ax3.plot([1900,1900], [1e-12,1e6], '--',color='gray', linewidth=1.5)
ax3.text(1910, 2*1e-10, '$(b)$',color='gray',fontsize=18)
ax3.set_xlabel('$N$', fontsize=18)
ax3.set_ylabel('$\Vert Error \Vert_{\infty}$')
ax3.set_title('(c)', loc='left', x=-0.15, fontsize=24, fontweight='bold')
ax3.set_ylim(1e-10, 0.9*1e5)
ax3.grid(True)
ax3.legend(loc='upper right', fontsize=16)

plt.tight_layout()
plt.show()
# plt.savefig('figs/fig2.pdf', dpi=300, bbox_inches='tight')
