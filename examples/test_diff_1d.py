"""
1D derivative accuracy and convergence demo.

Compares BSPF, Chebyshev, and Padé-10 derivatives on a smooth test function.
Generates a 2x2 figure with: (a) f and spline, (b) df vs exact, (c) pointwise
errors, and (d) convergence study in Linf.

Run from repo root after installing the package:
    python -m pip install -e .
    python examples/test_diff_1d.py
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from chebyshev import chebyshev_derivative_from_values,construct_chebyshev_nodes
from bspf import bspf1d
from padefd import padefd


# ---------------- Parameters ----------------
DEGREE = 11
NUM_BOUNDARY_POINTS = DEGREE + 5
N_BASIS = 4 * DEGREE
REG_PARAM = 1e-3
domain = (0.0, 2.0 * np.pi)
NUM_POINTS = 1000

# Grid parameters
clustering_factor = 2.0
clustering_flag = True


# ---------------- Grid + Model ----------------
x = np.linspace(domain[0], domain[1], NUM_POINTS, endpoint=True)
dx = (domain[1] - domain[0]) / (NUM_POINTS - 1)

model = bspf1d.from_grid(
    degree=DEGREE,
    x=x,
    domain=domain,
    order=DEGREE,
    n_basis=N_BASIS,
    num_boundary_points=NUM_BOUNDARY_POINTS,
    use_clustering=clustering_flag,
    clustering_factor=clustering_factor,
)


# ---------------- Test function (symbolic -> numeric) ----------------
t = sp.Symbol('t')
f_sym = sp.sin(t / (1.05 + sp.cos(t)))
df_sym = sp.diff(f_sym, t)
f = sp.lambdify(t, f_sym, 'numpy')
df = sp.lambdify(t, df_sym, 'numpy')

y = f(x)
y_deriv_exact = df(x)

# ---------------- Methods ----------------
# BSPF (with spline approximation)
y_deriv_bspf, y_spline, df_spline = model.differentiate_with_spline(y, k=1, lam=REG_PARAM)

# Chebyshev derivative on Chebyshev nodes (unpack x,t)
x_cheb, _t = construct_chebyshev_nodes(NUM_POINTS, domain)
f_vals = f(x_cheb)
y_deriv_cheb = chebyshev_derivative_from_values(f_vals, x_cheb, domain)
y_deriv_cheb_exact = df(x_cheb)

# Padé-10 compact finite difference on uniform grid
op = padefd(NUM_POINTS, dx, order=10)
y_deriv_fd = op(y)


# ---------------- Errors (Linf) ----------------
error_bspf = float(np.max(np.abs(y_deriv_bspf - y_deriv_exact)))
error_cheb = float(np.max(np.abs(y_deriv_cheb - y_deriv_cheb_exact)))
error_fd = float(np.max(np.abs(y_deriv_fd - y_deriv_exact)))

print("Errors (Linf):")
print("BSPF:", error_bspf)
print("Chebyshev:", error_cheb)
print("Padé-10:", error_fd)

# ---------------- Convergence study ----------------
grid_sizes = np.unique(np.geomspace(100, 1000, 10).astype(int))
errors_bspf, errors_cheb, errors_fd = [], [], []
for N in grid_sizes:
    xN = np.linspace(domain[0], domain[1], N, endpoint=True)
    dxN = (domain[1] - domain[0]) / (N - 1)
    fN = f(xN)
    dN = df(xN)

    mN = bspf1d.from_grid(
        degree=DEGREE,
        x=xN,
        domain=domain,
        order=DEGREE,
        n_basis=N_BASIS,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        use_clustering=clustering_flag,
        clustering_factor=clustering_factor,
    )
    d_bspf_N = mN.differentiate(fN, k=1, lam=REG_PARAM)
    errors_bspf.append(float(np.max(np.abs(d_bspf_N - dN))))

    xC, _tC = construct_chebyshev_nodes(N, domain)
    fC = f(xC)
    d_cheb_N = chebyshev_derivative_from_values(fC, xC, domain)
    d_cheb_exact_N = df(xC)
    errors_cheb.append(float(np.max(np.abs(d_cheb_N - d_cheb_exact_N))))

    opN = padefd(N, dxN, order=10)
    d_fd_N = opN(fN)
    errors_fd.append(float(np.max(np.abs(d_fd_N - dN))))

    print(f"N={N:4d} | Padé-10: {errors_fd[-1]:.3e}  BSPF: {errors_bspf[-1]:.3e}  Cheb: {errors_cheb[-1]:.3e}")


# ---------------- Plotting ----------------
plt.rcParams.update({
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 16,
    'figure.titlesize': 24,
    'axes.grid': True,
    'grid.alpha': 0.5,
})

plt.figure(figsize=(16, 10))
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# (a) f and spline approximation
plt.subplot(2, 2, 1)
plt.plot(x, y, '-', label='$f(x)$', linewidth=1)
plt.plot(x, y_spline, '-', label='$f_s(x)$', linewidth=1.5, alpha=1)
plt.legend(ncol=1)
plt.xlabel('$x$', fontsize=20)
plt.ylabel('$f(x)$', fontsize=20)
plt.title('(a)', loc='left', x=-0.15, fontsize=24, fontweight='bold')

# (b) Derivatives
plt.subplot(2, 2, 2)
plt.plot(x, y_deriv_exact, '-', color='k', label='Exact', linewidth=1)
plt.plot(x, y_deriv_bspf, '-', label='BSPF', linewidth=1)
plt.plot(x_cheb, y_deriv_cheb, '-', label='Chebyshev', linewidth=1)
plt.plot(x, y_deriv_fd, '-', label='Padé-10', color=default_colors[2], linewidth=1)
plt.xlabel('$x$', fontsize=20)
plt.ylabel('$df/dx$', fontsize=20)
plt.title('(b)', loc='left', x=-0.15, fontsize=24, fontweight='bold')
plt.legend(ncol=1)

# (c) Pointwise errors
plt.subplot(2, 2, 3)
plt.semilogy(x, np.abs(y_deriv_bspf - y_deriv_exact), '-', label='BSPF', color=default_colors[0], alpha=1)
plt.semilogy(x_cheb, np.abs(y_deriv_cheb - y_deriv_cheb_exact), '-', label='Chebyshev', color=default_colors[1], alpha=1)
plt.semilogy(x, np.abs(y_deriv_fd - y_deriv_exact), '-', label='Padé-10', color=default_colors[2], alpha=1)
plt.xlabel('$x$', fontsize=20)
plt.ylabel('$|Error|$', fontsize=20)
plt.legend(ncol=1)
plt.title('(c)', loc='left', x=-0.15, fontsize=24, fontweight='bold')

# (d) Convergence with fitted lines
plt.subplot(2, 2, 4)

def fit_loglog(xs, ys, take=5):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    xs_sel = xs[-take:]
    ys_sel = ys[-take:]
    a, b = np.polyfit(np.log(xs_sel), np.log(ys_sel), 1)
    return a, b

slope_bspf, intercept_bspf = fit_loglog(grid_sizes, errors_bspf)
slope_fd, intercept_fd = fit_loglog(grid_sizes, errors_fd)

x_fit = np.linspace(grid_sizes[0], grid_sizes[-1], 100)
y_fit_bspf = np.exp(intercept_bspf) * x_fit**slope_bspf
y_fit_fd = np.exp(intercept_fd) * x_fit**slope_fd

plt.loglog(grid_sizes, errors_bspf, 'o-', linewidth=1, label='BSPF', color=default_colors[0], alpha=1)
plt.loglog(grid_sizes, errors_cheb, 's-', linewidth=1, label='Chebyshev', color=default_colors[1], alpha=1)
plt.loglog(grid_sizes, errors_fd, '^-', linewidth=1, label='Padé-10', color=default_colors[2], alpha=1)
plt.xlabel('$N$', fontsize=20)
plt.ylabel('$\\|\\mathrm{Error}\\|_\\infty$', fontsize=20)
plt.title('(d)', loc='left', x=-0.15, fontsize=24, fontweight='bold')
plt.grid(True)
plt.legend(ncol=1)
plt.tight_layout()
plt.savefig('test_diff_1d.png', dpi=300, bbox_inches='tight')
print('Saved figure to test_diff_1d.png')
