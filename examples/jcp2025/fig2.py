

# Example usage and testing
from chebyshev import chebyshev_derivative_from_values, _construct_chebyshev_nodes
from bfpsm1d import bfpsm1d
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from findiff import FinDiff

# ---- Add below your existing code ------------------------------------------

def logistic(x):
    """σ(x) = 1 / (1 + exp(-x)) (symbolic)."""
    return 1 / (1 + sp.exp(-x))

def build_multi_sigmoid_expr(t, centers, sharpness, heights=None, baseline=None, normalize=True, domain=None):
    """
    φ_raw(t) = baseline*t + Σ h_i * σ(k_i*(t - c_i))
    If normalize: φ(t) = (φ_raw(t) - φ_raw(a)) / (φ_raw(b) - φ_raw(a))
    
    Parameters:
    -----------
    t : symbolic variable
    centers : list of sigmoid centers
    sharpness : list of sigmoid sharpness values
    heights : list of sigmoid heights (optional)
    baseline : baseline slope (optional)
    normalize : if True, maps domain to domain (default True)
    domain : tuple (a, b) defining the domain interval. If None, uses [0, 1]
    
    Returns:
    --------
    Symbolic expression for the mapping φ(t)
    """
    if heights is None:
        heights = [sp.Integer(1)] * len(centers)
    if baseline is None:
        baseline = sp.symbols('m', positive=True)
    if domain is None:
        domain = (0, 1)
    
    a, b = domain
    a_sym = sp.Float(a) if isinstance(a, (int, float)) else a
    b_sym = sp.Float(b) if isinstance(b, (int, float)) else b

    phi_raw = baseline * t
    for c, k, h in zip(centers, sharpness, heights):
        phi_raw += h * logistic(k * (t - c))

    if not normalize:
        return phi_raw

    phi_a = phi_raw.subs(t, a_sym)
    phi_b = phi_raw.subs(t, b_sym)
    denom = phi_b - phi_a
    return (phi_raw - phi_a) / denom * (b_sym - a_sym) + a_sym

def transform_to_unit_interval(x, domain):
    """Transform from [a,b] to [0,1]"""
    a, b = domain
    return (x - a) / (b - a)

def transform_from_unit_interval(s, domain):
    """Transform from [0,1] to [a,b]"""
    a, b = domain
    return s * (b - a) + a

def validate_domain(domain):
    """Validate domain parameter"""
    if domain is None:
        return (0, 1)
    
    if not isinstance(domain, (list, tuple)) or len(domain) != 2:
        raise ValueError("domain must be a tuple/list of length 2: (a, b)")
    
    a, b = domain
    if not isinstance(a, (int, float, sp.Basic)) or not isinstance(b, (int, float, sp.Basic)):
        raise ValueError("domain endpoints must be numeric or symbolic")
    
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a >= b:
        raise ValueError("domain must satisfy a < b")
    
    return tuple(domain)

def build_expr_via_connections_with_values(p_vals, k_vals, h_vals, m_val, normalize=True, domain=None):
    """
    Numeric-parameter builder for arbitrary domain intervals.
    
    Inputs
    ------
    p_vals : list length K-1 of internal connection points in (0,1), strictly increasing
    k_vals : list length K (>0) sharpness values
    h_vals : list length K (>=0) heights
    m_val  : scalar (>0) baseline slope
    normalize : if True, maps exactly domain → domain
    domain : tuple (a, b) defining the domain interval. If None, uses [0, 1]

    Returns
    -------
    expr_v  : SymPy expression φ(t) depending only on t
    dexpr_v : SymPy expression dφ/dt depending only on t
    centers : list of SymPy midpoints [c1, ..., cK] computed from p_vals in domain coordinates
    """
    domain = validate_domain(domain)
    a, b = domain
    
    K = len(k_vals)
    if len(h_vals) != K:
        raise ValueError("len(h_vals) must equal len(k_vals) (= K).")
    if len(p_vals) != max(0, K - 1):
        raise ValueError("len(p_vals) must be K-1.")
    if K >= 2:
        if not all(0.0 < p < 1.0 for p in p_vals):
            raise ValueError("All p_vals must lie strictly inside (0,1).")
        if not all(p_vals[i] < p_vals[i+1] for i in range(K - 2)):
            raise ValueError("p_vals must be strictly increasing.")

    # Transform connection points to domain coordinates
    # Boundaries in unit interval, then transform to domain
    b_unit = [0.0] + list(p_vals) + [1.0]
    b_domain = [transform_from_unit_interval(p, domain) for p in b_unit]
    
    # Midpoint centers in domain coordinates
    centers = [(b_domain[i] + b_domain[i+1]) / 2 for i in range(K)]

    # Build φ using the existing builder (keep numerics as Floats; no simplify)
    expr_v = build_multi_sigmoid_expr(
        t,
        centers=centers,
        sharpness=[sp.Float(k) for k in k_vals],
        heights=[sp.Float(h) for h in h_vals],
        baseline=sp.Float(m_val),
        normalize=normalize,
        domain=domain,
    )
    dexpr_v = sp.diff(expr_v, t)
    return expr_v, dexpr_v, centers

def create_adaptive_mapping(domain_source, domain_target, n_segments=2, sharpness_range=(10, 20), 
                          height_range=(0.1, 0.5), baseline_slope=0.1, normalize=True):
    """
    Create an adaptive mapping from source domain to target domain.
    
    Parameters:
    -----------
    domain_source : tuple (a, b) - source domain interval
    domain_target : tuple (c, d) - target domain interval  
    n_segments : int - number of sigmoid segments (default 2)
    sharpness_range : tuple - range for sigmoid sharpness values
    height_range : tuple - range for sigmoid heights
    baseline_slope : float - baseline slope parameter
    normalize : bool - whether to normalize the mapping
    
    Returns:
    --------
    expr_v : SymPy expression for the mapping φ(t)
    dexpr_v : SymPy expression for dφ/dt
    centers : list of sigmoid centers
    """
    # Validate domains
    domain_source = validate_domain(domain_source)
    domain_target = validate_domain(domain_target)
    
    # Generate connection points for n_segments
    if n_segments <= 1:
        p_vals = []
        k_vals = [np.random.uniform(*sharpness_range)]
        h_vals = [np.random.uniform(*height_range)]
    else:
        # Evenly spaced connection points in unit interval
        p_vals = [i / n_segments for i in range(1, n_segments)]
        k_vals = [np.random.uniform(*sharpness_range) for _ in range(n_segments)]
        h_vals = [np.random.uniform(*height_range) for _ in range(n_segments)]
    
    # Build mapping from source domain to unit interval, then to target domain
    t = sp.Symbol('t')
    
    # First, create mapping from source to unit interval
    t_unit = (t - domain_source[0]) / (domain_source[1] - domain_source[0])
    
    # Create mapping from unit interval using sigmoid functions
    expr_unit, dexpr_unit, centers = build_expr_via_connections_with_values(
        p_vals, k_vals, h_vals, baseline_slope, normalize=True, domain=(0, 1)
    )
    
    # Transform to target domain
    expr_final = expr_unit * (domain_target[1] - domain_target[0]) + domain_target[0]
    
    # Apply chain rule for derivative
    dexpr_final = sp.diff(expr_final, t)
    
    # Substitute the unit interval transformation
    expr_final = expr_final.subs(t, t_unit)
    dexpr_final = dexpr_final.subs(t, t_unit)
    
    return expr_final, dexpr_final, centers

def create_simple_mapping(domain_source, domain_target=None, p_vals=None, k_vals=None, h_vals=None, m_val=0.1):
    """
    Simple interface to create mappings between arbitrary intervals.
    
    Parameters:
    -----------
    domain_source : tuple (a, b) - source domain interval
    domain_target : tuple (c, d) - target domain interval. If None, uses source domain
    p_vals : list - connection points in (0,1). If None, uses [0.5]
    k_vals : list - sharpness values. If None, uses [15.0, 15.0]
    h_vals : list - height values. If None, uses [0.25, 0.25]
    m_val : float - baseline slope
    
    Returns:
    --------
    expr_v : SymPy expression for the mapping φ(t)
    dexpr_v : SymPy expression for dφ/dt
    centers : list of sigmoid centers
    """
    # Set defaults
    if domain_target is None:
        domain_target = domain_source
    if p_vals is None:
        p_vals = [0.5]
    if k_vals is None:
        k_vals = [15.0, 15.0]
    if h_vals is None:
        h_vals = [0.25, 0.25]
    
    # Validate domains
    domain_source = validate_domain(domain_source)
    domain_target = validate_domain(domain_target)
    
    t = sp.Symbol('t')
    
    # If source and target domains are the same, use direct mapping
    if domain_source == domain_target:
        return build_expr_via_connections_with_values(
            p_vals, k_vals, h_vals, m_val, normalize=True, domain=domain_source
        )
    
    # Otherwise, map through unit interval
    # First normalize to [0,1]
    t_norm = (t - domain_source[0]) / (domain_source[1] - domain_source[0])
    
    # Create mapping on unit interval
    expr_unit, _, centers_unit = build_expr_via_connections_with_values(
        p_vals, k_vals, h_vals, m_val, normalize=True, domain=(0, 1)
    )
    
    # Scale to target domain
    expr_target = expr_unit * (domain_target[1] - domain_target[0]) + domain_target[0]
    
    # Substitute normalized variable
    expr_final = expr_target.subs(t, t_norm)
    dexpr_final = sp.diff(expr_final, t)
    
    # Transform centers to target domain
    centers_target = [c * (domain_target[1] - domain_target[0]) + domain_target[0] 
                     for c in centers_unit]
    
    return expr_final, dexpr_final, centers_target

def compute_fourier_derivative(f, dx, domain=None):
    """
    Compute derivative using Fourier transform method.
    
    Args:
        f: array of function values
        dx: grid spacing
        domain: tuple of (start, end) points. If None, assumes periodic [0, 2π]
    
    Returns:
        Array of derivative values
    """
    N = len(f)
    if domain is not None:
        L = domain[1] - domain[0]
    else:
        L = 2 * np.pi
    
    # Compute frequency components
    k = 2 * np.pi * np.fft.fftfreq(N, dx)
    
    # Compute FFT, multiply by ik, and inverse transform
    f_hat = np.fft.fft(f)
    df_hat = 1j * k * f_hat
    df = np.real(np.fft.ifft(df_hat))
    
    return df

# ------------------------------------------------------------------
# Parameter block - now supports arbitrary domains!
# ------------------------------------------------------------------
DEGREE = 11       # B-spline polynomial degree
ALPHA = 2            # Factor for extra degrees of freedom (basis count)
REG_PARAM = 1e-3      # Tikhonov regularisation strength (lam)

# Test with different domain intervals
# domain = [0, 1]          # Original unit interval
# domain = [-1, 2]         # Different interval  
domain = [0, 2*np.pi]   # Symmetric about zero
# domain = [10, 50]        # Large positive interval

NUM_POINTS = 800   # Grid resolution
NUM_BOUNDARY_POINTS = DEGREE + 5

# Choose number of B-spline basis functions
N_BASIS = 4 * (DEGREE)

# Grid parameters
clustering_factor = 3.0  # Stronger clustering near endpoints
clustering_flag = True
grid_sizes = np.geomspace(300,3000,50).astype(int)#np.arange(600,10001,500)#[100,200,400,800,1600,3200,6400]#np.arange(1000,3001,100)

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
beta = 1.02


# Generate turbulent signal following Kolmogorov spectrum
k_min = 1   # Start of inertial range
k_max = 100 # End of inertial range
n_components = 1000

np.random.seed(42)
# Logarithmically spaced wavenumbers to resolve all scales
k = np.random.uniform(k_min, k_max, n_components)
frequencies = k  # Convert to angular frequencies

magnitudes = np.random.uniform(0, 0.01, n_components)
# Normalize to have maximum amplitude of 0.5
# magnitudes = 0.01*magnitudes/np.max(magnitudes)


# Random phases for each mode
phases = 2 * np.pi * np.random.rand(n_components)

# Create synthetic signal
f_sym = sp.sin(phi/(beta+sp.cos(phi)))
f_sym_original = sp.sin(t/(beta+sp.cos(t)))

# f_sym =  sp.tanh(200*(phi-np.pi))
for i in range(n_components):
    f_sym += magnitudes[i] * sp.cos(frequencies[i]*phi + phases[i])

# # f_sym_original =  sp.tanh(200*(t-np.pi))
for i in range(n_components):
    f_sym_original += magnitudes[i] * sp.cos(frequencies[i]*t + phases[i])



grid_sizes = np.geomspace(300,3000,50).astype(int)#np.arange(600,10001,500)#[100,200,400,800,1600,3200,6400]#np.arange(1000,3001,100)

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
model = bfpsm1d.from_grid(degree=DEGREE,
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
model_orig = bfpsm1d.from_grid(
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
    model_test = bfpsm1d.from_grid(
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
    model_orig_test = bfpsm1d.from_grid(
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


import matplotlib.pyplot as plt
import numpy as np
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
ax3.plot([800,800], [1e-12,1e6], '--',color='gray', linewidth=1.5)
ax3.text(810, 2*1e-12, '$(b)$',color='gray',fontsize=18)
ax3.set_xlabel('$N$', fontsize=18)
ax3.set_ylabel('$\Vert Error \Vert_{\infty}$')
ax3.set_title('(c)', loc='left', x=-0.15, fontsize=24, fontweight='bold')
ax3.set_ylim(1e-12, 0.9*1e6)
ax3.grid(True)
ax3.legend(loc='upper right', fontsize=16)

plt.tight_layout()
# plt.show()
plt.savefig('figs/fig2.pdf', dpi=300, bbox_inches='tight')
