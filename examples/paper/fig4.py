

# Example usage and testing
from chebyshev_integral import chebyshev_antiderivatives_fft
from scipy.integrate import cumulative_simpson
from bfpsm1d import bfpsm1d
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


# ------------------------------------------------------------------
# Parameter block
# ------------------------------------------------------------------
# N_MATCH = 10  # Enforce derivatives 0 … N_MATCH at both ends
DEGREE = 11      # B-spline polynomial degree
BOUNDARY_ORDER = DEGREE  # == number of constraints per side
ALPHA = 2            # Factor for extra degrees of freedom (basis count)
REG_PARAM = 1e-3      # Tikhonov regularisation strength (lam)
domain = [0, 2*np.pi]
NUM_POINTS = 2000   # Grid resolution
NUM_BOUNDARY_POINTS = BOUNDARY_ORDER + 5

# Choose number of B-spline basis functions
N_BASIS = 2 * (DEGREE) * ALPHA

# Grid parameters
clustering_factor = 3.0  # Stronger clustering near endpoints
clustering_flag = True
grid_sizes = np.geomspace(1000,3000,50).astype(int) #np.arange(500,10001,500)#[100,200,400,800,1600,3200,6400]#np.arange(1000,3001,100)

# Generate grid on the requested domain
x = np.linspace(domain[0], domain[1], NUM_POINTS, endpoint=True)
dx = (domain[1] - domain[0]) / (NUM_POINTS - 1)

# Initialize bfpsm1d model
model = bfpsm1d.from_grid(degree=DEGREE,
        x=x,
        n_basis=N_BASIS,
        domain=tuple(domain),
        use_clustering=clustering_flag,
        clustering_factor=clustering_factor,
        order=BOUNDARY_ORDER,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        correction="spectral")

# Test function and its analytical derivative using symbolic computation
alpha = 100

# Define symbolic variables and function
t = sp.Symbol('t')
phi = sp.Symbol('phi')

# Generate turbulent signal following Kolmogorov spectrum
k_min = 1   # Start of inertial range
k_max = 100 # End of inertial range
n_components = 1000

np.random.seed(42)
# Logarithmically spaced wavenumbers to resolve all scales
k = np.random.uniform(k_min, k_max, n_components)#np.exp(np.random.uniform(np.log(k_min), np.log(k_max), n_components))
frequencies = k  # Convert to angular frequencies

# Energy spectrum follows k^(-5/3)
# E_k = k**(-5/3)  # Energy spectrum
# Velocity amplitudes follow sqrt(E(k)) ~ k^(-5/6)
magnitudes = np.random.uniform(0, 0.01, n_components)
# Normalize to have maximum amplitude of 0.5
# magnitudes = 0.01*magnitudes / np.max(magnitudes)

# Random phases for each mode
phases = 2 * np.pi * np.random.rand(n_components)
# phi = 
phi = t #-2*t**3 + 3*t**1 + 0.1*(t**3 - 2*t**1 + t) + 0.1*(t**3 - t**1)
dphi = sp.diff(phi, t)

# Create synthetic signal
f_sym_original = sp.sin(t/(1.02+sp.cos(t)))#0 # sp.sin(alpha*t)

# f_sym_original =  sp.tanh(200*(t-np.pi))
for i in range(n_components):
    f_sym_original += magnitudes[i] * sp.cos(frequencies[i]*t + phases[i])

# # Create synthetic signal
# f_sym = 0.1*sp.sin(alpha*phi)
# f_sym_original = 0.1*sp.sin(alpha*t)

# Take derivative symbolically
df_sym_original = sp.diff(f_sym_original, t)

test_func_original = sp.lambdify(t, f_sym_original, modules='numpy')
test_func_deriv_original = sp.lambdify(t, df_sym_original, modules='numpy')

y_original = test_func_original(x)
y_deriv_exact_original = test_func_deriv_original(x)

# Compute derivatives using different methods
# 1. BSPF method on uniform grid
# y_deriv_bfpsm, y_spline = model.antiderivative(y_original, k=1, lam=REG_PARAM)


u_part, f_spline = model.antiderivative(y_deriv_exact_original, order=1, left_value=y_original[0], match_right=None, lam=0.0)
x_nodes, u_part_cheb = chebyshev_antiderivatives_fft(test_func_deriv_original, N=2000, domain=domain, order=1, anchor="left", c1=0.0, c2=0.0)
u_part_cheb = u_part_cheb + y_original[0]

u_part_simpson = cumulative_simpson(y_deriv_exact_original, x=x, initial=0)
u_part_simpson = u_part_simpson + y_original[0]
    
# error_simpson_check = np.max(np.abs(u_part_simpson - y_original))
# print(error_simpson_check)
print(np.max(np.abs(u_part_cheb - test_func_original(x_nodes))))
print(np.max(np.abs(u_part - y_original)))



error_cheb =[]
error_simpson =[]
error_bfpsm =[]


for N_grid in grid_sizes:
    x_test = np.linspace(domain[0], domain[1], N_grid, endpoint=True)
    
    dx = (domain[1] - domain[0]) / (N_grid - 1)
    model = bfpsm1d.from_grid(degree=DEGREE,
        x=x_test,
        n_basis=N_BASIS,
        domain=tuple(domain),
        use_clustering=clustering_flag,
        clustering_factor=clustering_factor,
        order=BOUNDARY_ORDER,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        correction="spectral")
    
    y_deriv_exact_original_test = test_func_deriv_original(x_test)
    y_original_test = test_func_original(x_test)
    
    u_part_test,f_spline_test = model.antiderivative(y_deriv_exact_original_test, order=1, left_value=y_original_test[0], match_right=None, lam=0.0)
    x_nodes_test, u_part_cheb_test = chebyshev_antiderivatives_fft(test_func_deriv_original, N=N_grid, domain=domain, order=1, anchor="left", c1=0.0, c2=0.0)
    u_part_cheb_test = u_part_cheb_test + y_original_test[0]
    
    u_part_simpson_test = cumulative_simpson(y_deriv_exact_original_test, x=x_test, initial=0)
    u_part_simpson_test = u_part_simpson_test + y_original_test[0]
    
    error_cheb.append(np.max(np.abs((u_part_cheb_test-test_func_original(x_nodes_test))**1)))
    error_simpson.append(np.max(np.abs((u_part_simpson_test-y_original_test)**1)))
    error_bfpsm.append(np.max(np.abs((u_part_test-y_original_test)**1)))

    print(f"N_grid: {N_grid}, Error cheb: {error_cheb[-1]}, Error bfpsm: {error_bfpsm[-1]}")


import matplotlib.pyplot as plt

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
plt.figure(figsize=(16,10))

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.subplot(2,2,1)

plt.plot(x,y_deriv_exact_original,linewidth=1,label='$f\'(x)$')
plt.plot(x,f_spline,'-',linewidth=1.5,label='$f\'_s(x)$')
# plt.plot(x,y_deriv_exact_original-f_spline,'-',linewidth=1,label='$r(x)$')
plt.xlabel('$x$')
plt.ylabel('$f\'(x)$')
plt.title('(a)', loc='left', x=-0.15,fontsize=24, fontweight='bold')
plt.legend(loc='upper left')

plt.subplot(2,2,2)
plt.plot(x,y_original,'k-',linewidth=1,label = "Exact")
plt.plot(x,u_part,'-',markersize=8,linewidth=1,label = "BSPF")
# plt.plot(x,F_spline,'-',markersize=8,linewidth=1.5,label = "$f_s(x)$")
# plt.plot(x_test,F_corr_test,'-',markersize=8,linewidth=1.5,label = "$f_c(x)$")

plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.title('(b)', loc='left', x=-0.15,fontsize=24, fontweight='bold')
plt.legend()
# print(np.mean(np.sqrt((u_part_cheb-test_func_original(x_nodes))**1)))
plt.subplot(2,2,3)
plt.semilogy(x,np.abs(u_part-y_original),linewidth=1,label='BSPF')
plt.semilogy(x_nodes,np.abs(u_part_cheb-test_func_original(x_nodes)),linewidth=1,label='Chebyshev')
plt.semilogy(x,np.abs(u_part_simpson- y_original),linewidth=1,label='Simpson-4')
plt.xlabel('$x$')
plt.ylabel('$|Error|$')
plt.ylim(1e-16,0.91e4)
plt.legend(loc='upper left')
plt.title('(c)', loc='left', x=-0.15,fontsize=24, fontweight='bold')
print(np.max(np.abs(u_part_simpson - y_original)))
print(np.max(np.abs(u_part_cheb - test_func_original(x_nodes))))
print(np.max(np.abs(u_part - y_original)))

plt.subplot(2,2,4)
plt.loglog(grid_sizes,error_bfpsm,'.-',label="BSPF")
plt.loglog(grid_sizes,error_cheb,'.-',label="Chebyshev")
plt.loglog(grid_sizes,error_simpson,'.-',label="Simpson-4")

plt.plot([2000,2000], [1e-16,0.9*1e4], '--',color='gray')
plt.text(2010, 2e-14, '$(a)-(c)$',color='gray',fontsize=18)

plt.xlabel('$N$')
plt.ylabel('$\|Error\|_\infty$')
plt.title('(d)', loc='left', x=-0.15,fontsize=24, fontweight='bold')

plt.legend()
plt.ylim(1e-14,0.9*1e3)
plt.tight_layout()
plt.savefig('figs/fig4.pdf',dpi=300,bbox_inches='tight')