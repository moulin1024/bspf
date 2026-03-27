# Data generation for fig1
from grid_mapping import (
    build_expr_via_connections_with_values,
    create_simple_mapping,
    build_multi_sigmoid_expr,
    transform_to_unit_interval,
    transform_from_unit_interval,
    validate_domain,
    logistic
)
from chebyshev import chebyshev_derivative_from_values, _construct_chebyshev_nodes as construct_chebyshev_nodes
from bspf1d import bspf1d
import numpy as np
import sympy as sp

def fourth_order_finite_difference(y, dx):
    """
    Standard 4th order finite difference scheme for first derivative.
    
    Uses centered differences for interior points:
    f'(x_i) ≈ (-f(x_{i+2}) + 8f(x_{i+1}) - 8f(x_{i-1}) + f(x_{i-2})) / (12h)
    
    Uses forward/backward differences for boundary points.
    """
    n = len(y)
    dy = np.zeros_like(y)
    
    # Forward difference for first two points
    if n >= 5:
        # 4th order forward: f'(x_0) ≈ (-25f_0 + 48f_1 - 36f_2 + 16f_3 - 3f_4) / (12h)
        dy[0] = (-25*y[0] + 48*y[1] - 36*y[2] + 16*y[3] - 3*y[4]) / (12*dx)
        # 4th order forward: f'(x_1) ≈ (-3f_0 - 10f_1 + 18f_2 - 6f_3 + f_4) / (12h)
        dy[1] = (-3*y[0] - 10*y[1] + 18*y[2] - 6*y[3] + y[4]) / (12*dx)
    elif n >= 3:
        # Fallback to lower order if not enough points
        dy[0] = (-3*y[0] + 4*y[1] - y[2]) / (2*dx)
        if n >= 4:
            dy[1] = (-y[0] + y[2]) / (2*dx)
    
    # Centered differences for interior points
    if n >= 5:
        for i in range(2, n-2):
            dy[i] = (-y[i+2] + 8*y[i+1] - 8*y[i-1] + y[i-2]) / (12*dx)
    
    # Backward difference for last two points
    if n >= 5:
        # 4th order backward: f'(x_{n-2}) ≈ (-f_{n-4} + 6f_{n-3} - 18f_{n-2} + 10f_{n-1} + 3f_n) / (12h)
        dy[n-2] = (-y[n-5] + 6*y[n-4] - 18*y[n-3] + 10*y[n-2] + 3*y[n-1]) / (12*dx)
        # 4th order backward: f'(x_{n-1}) ≈ (3f_{n-5} - 16f_{n-4} + 36f_{n-3} - 48f_{n-2} + 25f_{n-1}) / (12h)
        dy[n-1] = (3*y[n-5] - 16*y[n-4] + 36*y[n-3] - 48*y[n-2] + 25*y[n-1]) / (12*dx)
    elif n >= 3:
        # Fallback to lower order if not enough points
        if n >= 4:
            dy[n-2] = (-y[n-3] + y[n-1]) / (2*dx)
        dy[n-1] = (y[n-3] - 4*y[n-2] + 3*y[n-1]) / (2*dx)
    
    return dy


def chebyshev_piecewise_derivative(
    f,
    *,
    domain: tuple[float, float],
    n_points_total: int,
    split: float | None = None,
):
    """
    Piecewise Chebyshev derivative: split domain at `split` (default midpoint),
    apply Chebyshev on each sub-interval, then stitch results (dropping duplicate split point).
    Returns (x_nodes, df_dx) with total length == n_points_total.
    """
    a, b = map(float, domain)
    m = 0.5 * (a + b) if split is None else float(split)
    if not (a < m < b):
        raise ValueError("split must satisfy a < split < b")
    if n_points_total < 3:
        raise ValueError("n_points_total must be >= 3")

    # Choose subdomain sizes so n_left + n_right - 1 == n_points_total
    n_left = n_points_total // 2 + 1
    n_right = n_points_total - n_left + 1

    x_left, _ = construct_chebyshev_nodes(n_left - 1, domain=(a, m))
    x_right, _ = construct_chebyshev_nodes(n_right - 1, domain=(m, b))

    y_left = f(x_left)
    y_right = f(x_right)
    dy_left = chebyshev_derivative_from_values(y_left, x_left, domain=(a, m))
    dy_right = chebyshev_derivative_from_values(y_right, x_right, domain=(m, b))

    x_all = np.concatenate([x_left, x_right[1:]])
    dy_all = np.concatenate([dy_left, dy_right[1:]])
    return x_all, dy_all

# ------------------------------------------------------------------
# Parameter block
# ------------------------------------------------------------------
DEGREE = 8       # B-spline polynomial degree
REG_PARAM = 1e-3      # Tikhonov regularisation strength (lam)

# Test with different domain intervals
domain = [0, 2*np.pi]   # Symmetric about zero

NUM_POINTS = 1001   # Grid resolution
NUM_BOUNDARY_POINTS = DEGREE + 5

# Choose number of B-spline basis functions
N_BASIS = 4 * (DEGREE)

# Grid parameters
clustering_factor = 2.0  # Stronger clustering near endpoints
clustering_flag = True

# Generate grid on the requested domain
x = np.linspace(domain[0], domain[1], NUM_POINTS)
dx = (domain[1] - domain[0]) / (NUM_POINTS - 1)

# Define symbolic variables and function
t = sp.Symbol('t')
phi = sp.Symbol('phi')

# Demonstrate different mapping approaches for arbitrary intervals
print(f"Working with domain: [{domain[0]}, {domain[1]}]")
print("Using Manual sigmoid-based endpoint clustering")

# Method 1: Using existing function with domain parameter
p_vals = [0.5]              # connection points
k_vals = [2.75, 2.75]      # sharpness (>0)
h_vals = [0.5, 0.5]      # heights (>=0)
m_val  = 0.01                    # baseline (>0)

# Build expressions with arbitrary domain support
phi, dphi, centers = build_expr_via_connections_with_values(
    p_vals, k_vals, h_vals, m_val, normalize=True, domain=domain
)

print(f"Mapping centers in domain coordinates: {centers}")
print(f"Domain length: {domain[1] - domain[0]:.3f}")

# Test function and its analytical derivative using symbolic computation
beta = 1.02

# Create synthetic signal
f_sym = sp.sin(phi/(beta+sp.cos(phi))) + sp.cos(100.5*phi)
f_sym_original = sp.sin(t/(beta+sp.cos(t))) + sp.cos(100.5*t)

grid_sizes = np.geomspace(400,4000,50).astype(int)

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
        clustering_factor=clustering_factor,
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
    clustering_factor=clustering_factor,
    order=DEGREE,
    num_boundary_points=NUM_BOUNDARY_POINTS,
    correction="spectral"
)
y_deriv_bfpsm_orig, y_spline_orig = model_orig.differentiate(y_original, k=1, lam=REG_PARAM)

# 3. Standard 4th order finite difference
y_deriv_fd = fourth_order_finite_difference(y, dx)

# 4. Chebyshev spectral method with original grid
N_cheb = NUM_POINTS - 1  # Chebyshev polynomial degree
x_cheb_orig, _ = construct_chebyshev_nodes(N_cheb, domain=tuple(domain))
y_cheb_orig = test_func_original(x_cheb_orig)
y_deriv_cheb_orig = chebyshev_derivative_from_values(y_cheb_orig, x_cheb_orig, domain=tuple(domain))

# 4b. Piecewise Chebyshev (split in the middle)
split_point = 0.5 * (domain[0] + domain[1])
x_cheb_pw, y_deriv_cheb_pw = chebyshev_piecewise_derivative(
    test_func_original, domain=tuple(domain), n_points_total=NUM_POINTS, split=split_point
)

# Compute errors for each (L∞ norm)
error_bfpsm = np.max(np.abs(y_deriv_bfpsm/dphi_exact - y_deriv_exact/dphi_exact))
error_bfpsm_orig = np.max(np.abs(y_deriv_bfpsm_orig - y_deriv_exact_original))

# Chebyshev errors
y_deriv_exact_cheb_orig = test_func_deriv_original(x_cheb_orig)
error_cheb_orig = np.max(np.abs(y_deriv_cheb_orig - y_deriv_exact_cheb_orig))

# Piecewise Chebyshev errors
y_deriv_exact_cheb_pw = test_func_deriv_original(x_cheb_pw)
error_cheb_pw = np.max(np.abs(y_deriv_cheb_pw - y_deriv_exact_cheb_pw))

# 4th order finite difference errors
error_fd = np.max(np.abs(y_deriv_fd - y_deriv_exact_original))

# Compute normalization factors for relative errors
norm_exact_bfpsm_init = np.max(np.abs(y_deriv_exact/dphi_exact))
norm_exact_orig_init = np.max(np.abs(y_deriv_exact_original))
norm_exact_cheb_init = np.max(np.abs(y_deriv_exact_cheb_orig))

print("Relative L∞ errors:")
print("BSPF (mapped):", error_bfpsm / norm_exact_bfpsm_init)
print("BSPF (original):", error_bfpsm_orig / norm_exact_orig_init)
print("Chebyshev (original):", error_cheb_orig / norm_exact_cheb_init)
print("Chebyshev (piecewise):", error_cheb_pw / norm_exact_orig_init)
print("Finite difference (4th order):", error_fd / norm_exact_orig_init)

errors_bfpsm = []
errors_bfpsm_orig = []
errors_cheb_orig = []
errors_cheb_pw = []
errors_fd = []
# Store error distributions for panel (c)
error_distributions_bfpsm = []  # List of (x, error) tuples
error_distributions_bfpsm_orig = []  # List of (x, error) tuples for uniform BSPF
error_distributions_cheb = []   # List of (x, error) tuples
error_distributions_fd = []   # List of (x, error) tuples
delta_x_values = []             # List of delta_x values

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
        clustering_factor=clustering_factor,
        order=DEGREE,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        correction="spectral"
    )
    y_deriv_bfpsm_test, _ = model_test.differentiate(y_test, k=1, lam=REG_PARAM)
    # L∞ norm: ||e||_∞ = max|e_i|
    error_bfpsm_test = (y_deriv_bfpsm_test/dphi_test - y_deriv_exact_test/dphi_test)
    norm_error_bfpsm = np.max(np.abs(error_bfpsm_test))
    # Relative L∞ error: normalize by max of exact solution
    norm_exact_bfpsm = np.max(np.abs(y_deriv_exact_test/dphi_test))
    errors_bfpsm.append(norm_error_bfpsm / norm_exact_bfpsm)
    # Store error distribution for panel (c)
    error_distributions_bfpsm.append((x_test.copy(), error_bfpsm_test.copy()))
    delta_x_values.append(dx_test)
    
    # BSPF method with original grid
    model_orig_test = bspf1d.from_grid(
        degree=DEGREE,
        x=x_test,
        n_basis=N_BASIS,
        domain=tuple(domain),
        use_clustering=clustering_flag,
        clustering_factor=clustering_factor,
        order=DEGREE,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        correction="spectral"
    )
    y_deriv_bfpsm_orig_test, _ = model_orig_test.differentiate(y_test_original, k=1, lam=REG_PARAM)
    error_bfpsm_orig_test = (y_deriv_bfpsm_orig_test - y_deriv_exact_test_original)
    norm_error_bfpsm_orig = np.max(np.abs(error_bfpsm_orig_test))
    # Relative L∞ error: normalize by max of exact solution
    norm_exact_bfpsm_orig = np.max(np.abs(y_deriv_exact_test_original))
    errors_bfpsm_orig.append(norm_error_bfpsm_orig / norm_exact_bfpsm_orig)
    # Store error distribution for panel (c)
    error_distributions_bfpsm_orig.append((x_test.copy(), error_bfpsm_orig_test.copy()))
    
    # Chebyshev method with original grid
    N_cheb_test = n_points - 1
    x_cheb_test, _ = construct_chebyshev_nodes(N_cheb_test, domain=tuple(domain))
    y_cheb_test = test_func_original(x_cheb_test)
    y_deriv_cheb_test = chebyshev_derivative_from_values(y_cheb_test, x_cheb_test, domain=tuple(domain))
    y_deriv_exact_cheb_test = test_func_deriv_original(x_cheb_test)
    error_cheb_test = (y_deriv_cheb_test - y_deriv_exact_cheb_test)
    # L∞ norm
    norm_error_cheb = np.max(np.abs(error_cheb_test))
    # Relative L∞ error: normalize by max of exact solution
    norm_exact_cheb = np.max(np.abs(y_deriv_exact_cheb_test))
    errors_cheb_orig.append(norm_error_cheb / norm_exact_cheb)
    # Store error distribution for panel (c)
    error_distributions_cheb.append((x_cheb_test.copy(), error_cheb_test.copy()))

    # Piecewise Chebyshev (split in the middle)
    x_cheb_pw_test, y_deriv_cheb_pw_test = chebyshev_piecewise_derivative(
        test_func_original, domain=tuple(domain), n_points_total=n_points, split=split_point
    )
    y_deriv_exact_cheb_pw_test = test_func_deriv_original(x_cheb_pw_test)
    error_cheb_pw_test = (y_deriv_cheb_pw_test - y_deriv_exact_cheb_pw_test)
    # L∞ norm
    errors_cheb_pw.append(np.max(np.abs(error_cheb_pw_test)))
    
    # 4th order finite difference method
    y_deriv_fd_test = fourth_order_finite_difference(y_test_original, dx_test)
    error_fd_test = (y_deriv_fd_test - y_deriv_exact_test_original)
    # L∞ norm
    norm_error_fd = np.max(np.abs(error_fd_test))
    # Relative L∞ error: normalize by max of exact solution
    norm_exact_fd = np.max(np.abs(y_deriv_exact_test_original))
    errors_fd.append(norm_error_fd / norm_exact_fd)
    # Store error distribution for panel (c)
    error_distributions_fd.append((x_test.copy(), error_fd_test.copy()))

# Prepare data for plotting
# Fine grid for panels (a) and (b)
n_grid_points = 701
x_fine = np.linspace(domain[0], domain[1], n_grid_points)
y_fine = test_func_original(x_fine)
xi_fine = test_phi(x_fine)

# Sample points for panel (b)
n_sample = 100
x_sample = np.linspace(domain[0], domain[1], n_sample)
xi_sample = test_phi(x_sample)

# Find index with grid number closest to target for panel (c)
target_N = 1000
grid_sizes_array = np.array(grid_sizes)
idx_closest = np.argmin(np.abs(grid_sizes_array - target_N))
selected_N = grid_sizes_array[idx_closest]
selected_delta_x = delta_x_values[idx_closest]
print(f"Using N = {selected_N} (closest to {target_N}, index {idx_closest}, delta_x = {selected_delta_x:.6f})")

# Extract error distributions from convergence study
x_bfpsm_error, error_bfpsm_error = error_distributions_bfpsm[idx_closest]
x_bfpsm_orig_error, error_bfpsm_orig_error = error_distributions_bfpsm_orig[idx_closest]
x_cheb_error, error_cheb_error = error_distributions_cheb[idx_closest]
x_fd_error, error_fd_error = error_distributions_fd[idx_closest]

# Compute normalization factors (max of exact solutions) for relative error
# For BSPF mapped: normalize by max of y_deriv_exact/dphi
x_test_selected = np.linspace(domain[0], domain[1], selected_N)
y_deriv_exact_selected = test_func_deriv(x_test_selected)
dphi_selected = test_dphi(x_test_selected)
norm_exact_bfpsm_selected = np.max(np.abs(y_deriv_exact_selected/dphi_selected))
# For others: normalize by max of y_deriv_exact_original
y_deriv_exact_orig_selected = test_func_deriv_original(x_test_selected)
norm_exact_orig_selected = np.max(np.abs(y_deriv_exact_orig_selected))
# For Chebyshev: need to compute on Chebyshev nodes
N_cheb_selected = selected_N - 1
x_cheb_selected, _ = construct_chebyshev_nodes(N_cheb_selected, domain=tuple(domain))
y_deriv_exact_cheb_selected = test_func_deriv_original(x_cheb_selected)
norm_exact_cheb_selected = np.max(np.abs(y_deriv_exact_cheb_selected))

# Save data to .npz file
data_file = 'data/fig1_data.npz'
np.savez(
    data_file,
    # Domain and parameters
    domain=domain,
    # Original grid data for panel (a)
    x=x,
    y_spline=y_spline,
    # Fine grid data for panels (a) and (b)
    x_fine=x_fine,
    y_fine=y_fine,
    xi_fine=xi_fine,
    # Sample points for panel (b)
    x_sample=x_sample,
    xi_sample=xi_sample,
    # Convergence data
    grid_sizes=grid_sizes,
    errors_bfpsm=np.array(errors_bfpsm),
    errors_bfpsm_orig=np.array(errors_bfpsm_orig),
    errors_cheb_orig=np.array(errors_cheb_orig),
    errors_fd=np.array(errors_fd),
    # Error distributions for panel (c)
    x_bfpsm_error=x_bfpsm_error,
    error_bfpsm_error=error_bfpsm_error,
    x_cheb_error=x_cheb_error,
    error_cheb_error=error_cheb_error,
    x_fd_error=x_fd_error,
    error_fd_error=error_fd_error,
    # Normalization factors
    norm_exact_bfpsm_selected=norm_exact_bfpsm_selected,
    norm_exact_orig_selected=norm_exact_orig_selected,
    norm_exact_cheb_selected=norm_exact_cheb_selected,
    # Selected N
    selected_N=selected_N,
    target_N=target_N,
)

print(f"Data saved to {data_file}")
