

# Example usage and testing
from chebyshev import chebyshev_derivative_from_values, _construct_chebyshev_nodes
from bfpsm1d import bfpsm1d
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from findiff import Diff
from padefd import padefd

# --------------------------------------------------    ----------------
# Parameter block
# ------------------------------------------------------------------
# N_MATCH = 10  # Enforce derivatives 0 … N_MATCH at both ends
DEGREE = 11      # B-spline polynomial degree
MATCH_ORDER = DEGREE
NUM_BOUNDARY_POINTS = DEGREE + 5
N_BASIS = 4 * (DEGREE)
REG_PARAM = 1e-3      # Tikhonov regularisation strength (lam)
domain = [0, 2*np.pi]
NUM_POINTS = 2000   # Grid resolution


# Grid parameters
clustering_factor = 3.0  # Stronger clustering near endpoints
clustering_flag = True 

# Generate grid on the requested domain
x = np.linspace(domain[0], domain[1], NUM_POINTS, endpoint=True)
dx = (domain[1] - domain[0]) / (NUM_POINTS - 1)

# Initialize bfpsm1d model
model = bfpsm1d.from_grid(degree=DEGREE,
        x=x,
        domain=tuple(domain),
        order=DEGREE,
        n_basis=N_BASIS,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        use_clustering=clustering_flag,
        clustering_factor=clustering_factor)

# Test function and its analytical derivative using symbolic computation
alpha = 10

# Define symbolic variables and function
t = sp.Symbol('t')
phi = sp.Symbol('phi')

# Generate turbulent signal following Kolmogorov spectrum
k_min = 1   # Start of inertial range
k_max = 100 # End of inertial range
n_components = 1000
grid_sizes = np.geomspace(1000,3000,50).astype(int)#np.arange(600,10001,500)#[100,200,400,800,1600,3200,6400]#np.arange(1000,3001,100)

np.random.seed(42)
# Logarithmically spaced wavenumbers to resolve all scales
k = np.random.uniform(k_min, k_max, n_components)#np.exp(np.random.uniform(np.log(k_min), np.log(k_max), n_components))
frequencies = k  # Convert to angular frequencies

# Energy spectrum follows k^(-5/3)
# E_k = k**(-5/3)  # Energy spectrum
# Velocity amplitudes follow sqrt(E(k)) ~ k^(-5/6)
magnitudes = np.random.uniform(0, 0.01, n_components)
# Normalize to have maximum amplitude of 0.5
# magnitudes = magnitudes / np.max(magnitudes) * 0.01

# Random phases for each mode
phases = 2 * np.pi * np.random.rand(n_components)
# phi = 
phi = t #-2*t**3 + 3*t**1 + 0.1*(t**3 - 2*t**1 + t) + 0.1*(t**3 - t**1)
dphi = sp.diff(phi, t)

# Create synthetic signal
f_sym = sp.sin(t/(1.02+sp.cos(t))) #sp.tanh(alpha*(phi-np.pi))#0 # sp.sin(alpha*phi)
f_sym_original = sp.sin(t/(1.02+sp.cos(t)))#sp.tanh(alpha*(t-np.pi))#0 # sp.sin(alpha*t)

# f_sym = 0
# f_sym_original = 0
# # Create synthetic signal
# f_sym =  sp.tanh(200*(phi-np.pi))
for i in range(n_components):
    f_sym += magnitudes[i] * sp.cos(frequencies[i]*phi + phases[i])

# # f_sym_original =  sp.tanh(200*(t-np.pi))
for i in range(n_components):
    f_sym_original += magnitudes[i] * sp.cos(frequencies[i]*t + phases[i])

# # Create synthetic signal
# f_sym = 0.1*sp.sin(alpha*phi)
# f_sym_original = 0.1*sp.sin(alpha*t)

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
dphi_exact = test_dphi(x)

y_original = test_func_original(x)
y_deriv_exact_original = test_func_deriv_original(x)

# Compute derivatives using different methods
# 1. BSPF method on uniform grid
y_deriv_bfpsm, y_spline, df_spline = model.differentiate_complete(y_original, k=1, lam=REG_PARAM)

# 2. Chebyshev method on Chebyshev nodes
x_cheb, _ = _construct_chebyshev_nodes(NUM_POINTS, domain)  # N intervals = NUM_POINTS-1 nodes
f_vals = test_func_original(x_cheb)
y_deriv_cheb = chebyshev_derivative_from_values(f_vals, x_cheb, domain)

# 3. 4th order finite difference
# d_dx = Diff(0, dx, acc=4)  # 4th order accurate first derivative
op = padefd(NUM_POINTS, dx, order=10)
y_deriv_fd = op(y_original)  # 4th order accurate first derivative


# Compute errors for each method
error_bfpsm = np.max(np.abs((y_deriv_bfpsm - y_deriv_exact_original)**1))  # L2 norm
y_deriv_cheb_exact = test_func_deriv_original(x_cheb)
error_cheb = np.max(np.abs((y_deriv_cheb - y_deriv_cheb_exact)**1))
error_fd = np.max(np.abs((y_deriv_fd - y_deriv_exact_original)**1))

print("Errors (L^2 Norm):")
print("BSPF:", error_bfpsm)
print("Chebyshev:", error_cheb)
print("Padé-10:", error_fd)

errors_bfpsm = []
errors_cheb = []
errors_fd = []
for n_points in grid_sizes:
    # Create grid
    x_test = np.linspace(domain[0], domain[1], n_points)
    dx_test = (domain[1] - domain[0]) / (n_points - 1)
    
    # Compute exact solution
    y_test = test_func_original(x_test)
    y_deriv_exact_test = test_func_deriv_original(x_test)
    
    # BSPF method on uniform grid
    model_test = bfpsm1d.from_grid(
        degree=DEGREE,
        x=x_test,
        domain=tuple(domain),
        order=DEGREE,
        n_basis=N_BASIS,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        use_clustering=clustering_flag,
        clustering_factor=clustering_factor
    )
    y_deriv_bfpsm_test, _ = model_test.differentiate(y_test, k=1, lam=REG_PARAM)
    errors_bfpsm.append(np.max(np.abs((y_deriv_bfpsm_test - y_deriv_exact_test)**1)))
    
    # Chebyshev method on Chebyshev nodes
    x_cheb_test, _ = _construct_chebyshev_nodes(n_points, domain)
    f_vals_test = test_func_original(x_cheb_test)
    y_deriv_cheb_test = chebyshev_derivative_from_values(f_vals_test, x_cheb_test, domain)
    y_deriv_cheb_exact_test = test_func_deriv_original(x_cheb_test)
    errors_cheb.append(np.max(np.abs((y_deriv_cheb_test - y_deriv_cheb_exact_test)**1)))

    # 4th order finite difference
    # d_dx_test = Diff(0, dx_test, acc=4)
    # y_deriv_fd_test = d_dx_test(y_test)
    op_test = padefd(n_points, dx_test, order=10)
    y_deriv_fd_test = op_test(y_test)  # 4th order accurate first derivative
    errors_fd.append(np.max(np.abs((y_deriv_fd_test - y_deriv_exact_test)**1)))
    print("N = ", n_points, "error pade10 = ", errors_fd[-1], "error bfpsm = ", errors_bfpsm[-1], "error cheb = ", errors_cheb[-1])




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

# Plotting
plt.figure(figsize=(16, 10))  # Increased height to accommodate legends above plots

# Original function and spline approximation
plt.subplot(2, 2, 1)
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# Plot functions

# Plot knot locations
# plt.plot(model.knots, np.zeros(len(model.knots)), '.', color=default_colors[3],
#          markersize=8,label='Bspline knots',alpha=0.5)
plt.plot(xi, y, '-', label='$f(x)$', linewidth=1)
plt.plot(xi, y_spline, '-', label='$f_s(x)$', linewidth=1.5,alpha=1)
# plt.plot(xi, y-y_spline, '-', label='$r(x)$', linewidth=1)


# Original axis settings
plt.legend(ncol=1)
plt.xlabel('$x$',fontsize=20)
plt.ylabel('$f(x)$',fontsize=20)
# plt.ylim(-1.35, 1.35)
plt.title(f'(a)', loc='left', x=-0.15,fontsize=24, fontweight='bold')

plt.subplot(2, 2, 2)
plt.plot(x, y_deriv_exact_original, '-',color='k', label='Exact', linewidth=1)
plt.plot(x, y_deriv_bfpsm, '-', label='BSPF',linewidth=1.5)
# plt.plot(x,df_spline,'-',label='$f\'_s(x)$',linewidth=1.5)
# plt.plot(x_cheb, y_deriv_cheb, '-', label='Chebyshev',linewidth=1)
# plt.plot(x, y_deriv_fd, '-', label='Padé-10', color=default_colors[2],linewidth=1)
plt.xlabel('$x$',fontsize=20)
plt.ylabel('$df/dx$',fontsize=20)
plt.title(f'(b)', loc='left', x=-0.15,fontsize=24, fontweight='bold')
plt.legend(ncol=1)
# plt.ylim(-2,12)

plt.subplot(2, 2, 3)
plt.semilogy(x, np.abs(y_deriv_bfpsm - y_deriv_exact_original), '-',label='BSPF',color=default_colors[0],alpha=1)
plt.semilogy(x_cheb, np.abs(y_deriv_cheb - y_deriv_cheb_exact), '-',label='Chebyshev',color=default_colors[1],alpha=1)
plt.semilogy(x, np.abs(y_deriv_fd - y_deriv_exact_original), '-',label='Padé-10',color=default_colors[2],alpha=1)
plt.xlabel('$x$',fontsize=20)
plt.ylabel('$|Error|$',fontsize=20)
plt.ylim(1e-15,0.9*1e4)
plt.legend(ncol=3)
plt.title('(c)', loc='left', x=-0.15,fontsize=24, fontweight='bold')

# Convergence study
plt.subplot(2, 2, 4)
# Fit convergence rate for BSPF from entries 4 to 10
fit_start, fit_end = 25, 30
fit_start_fd, fit_end_fd = 0, 50

x_fit = grid_sizes[fit_start:fit_end]
x_fit_fd = grid_sizes[fit_start_fd:fit_end_fd]

y_fit = np.array(errors_bfpsm[fit_start:fit_end])
y_fit_fd = np.array(errors_fd[fit_start_fd:fit_end_fd])

# Fit in log-log space: log(error) = a * log(N) + b
log_x_fit = np.log(x_fit)
log_x_fit_fd = np.log(x_fit_fd)

log_y_fit = np.log(y_fit)
log_y_fit_fd = np.log(y_fit_fd)

coefficients = np.polyfit(log_x_fit, log_y_fit, 1)
coefficients_fd = np.polyfit(log_x_fit_fd, log_y_fit_fd, 1)

slope = coefficients[0]
intercept = coefficients[1]

slope_fd = coefficients_fd[0]
intercept_fd = coefficients_fd[1]

print(slope, intercept)
print(slope_fd, intercept_fd)

# Generate fitted line
x_fit_line = np.linspace(x_fit[0], x_fit[-1], 100)
y_fit_line = np.exp(intercept) * x_fit_line**slope

x_fit_line_fd = np.linspace(x_fit_fd[0], x_fit_fd[-1], 100)
y_fit_line_fd = np.exp(intercept_fd) * x_fit_line_fd**slope_fd

plt.loglog(grid_sizes, errors_bfpsm, '.-', linewidth=1, label='BSPF',color=default_colors[0],alpha=1)
plt.loglog(grid_sizes, errors_cheb, '.-', linewidth=1, label='Chebyshev',color=default_colors[1],alpha=1)
plt.loglog(grid_sizes, errors_fd, '.-', linewidth=1, label='Padé-10',color=default_colors[2],alpha=1)
plt.plot([2000,2000], [1e-16,1e6], '--',color='gray')
plt.text(2010, 2e-11, '$(a)-(c)$',color='gray',fontsize=18)
# Add reference lines for different convergence rates
ref_x = np.array([grid_sizes[0], grid_sizes[-1]])
# plt.loglog(ref_x, 10*errors_fd[0]*(ref_x/ref_x[0])**(-4), '--', linewidth=1, label='$O(h^{-4})$',color=default_colors[2])
# plt.loglog(ref_x,0.005*errors_fd[0]*(ref_x/ref_x[0])**(-DEGREE), '--', linewidth=1, label='$O(N^{-' + str(DEGREE) + '})$',color=default_colors[0])
plt.loglog(x_fit_line, 0.2*y_fit_line, '--', linewidth=2, color=default_colors[0])
plt.loglog(x_fit_line_fd, 5*y_fit_line_fd, '--', linewidth=2, color=default_colors[2])
plt.text(2500, 1e-0, '$O(h^{' + str(f'{slope_fd:.1f}') + '})$',color=default_colors[2],fontsize=18)
plt.text(1500, 2*1e-10, '$O(h^{' + str(f'{slope:.1f}') + '})$',color=default_colors[0],fontsize=18)

# # # Plot fitted line
# plt.loglog(x_fit_line, 0.5*y_fit_line, '--', linewidth=2, color='r', 
#     label=f'Fit: $O(N^{{{slope:.1f}}})$')
plt.xlabel('$N$',fontsize=20)
plt.ylabel('$\|Error\|_\infty$',fontsize=20)
plt.title('(d)', loc='left', x=-0.15,fontsize=24, fontweight='bold')
plt.grid(True)
plt.legend(ncol=1)
plt.ylim(1e-11,0.9*1e6)
plt.tight_layout()
plt.savefig('figs/fig1.pdf', dpi=300, bbox_inches='tight')
# plt.show()
