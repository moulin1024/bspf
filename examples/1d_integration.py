"""
Integration demo comparing BSPF antiderivatives against SciPy Simpson
for the turbulent-like signal used in fig4.py.

Run from repo root after installing the package:
    python -m pip install -e .
    python examples/integration_1d.py
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from bspf import bspf1d

# Scenario configuration
domain_start = 0.0
domain_end = 2.0 * math.pi
num_points = 2001
degree = 11
reg = 1e-3
num_boundary_points = degree
n_basis = 4 * degree
use_clustering = True
clustering_factor = 3.0

# Random signal configuration
k_min = 1.0
k_max = 100.0
n_components = 1000
amplitude = 0.01
offset = 1.02
rng = np.random.default_rng(42)

frequencies = rng.uniform(k_min, k_max, n_components).astype(np.float64)
magnitudes = rng.uniform(0.0, amplitude, n_components).astype(np.float64)
phases = rng.uniform(0.0, 2.0 * np.pi, n_components).astype(np.float64)
freq_mag = frequencies * magnitudes

integrand_label = "$f'(x)$"
target_label = "$f(x)$"

# Evaluation grid for the main experiment
x = np.linspace(domain_start, domain_end, num_points, endpoint=True, dtype=np.float64)
denom = offset + np.cos(x)
theta = x[:, None] * frequencies[None, :] + phases[None, :]

base_value = np.sin(x / denom)
cos_terms = np.cos(theta) @ magnitudes
target = base_value + cos_terms

g_prime = (denom + x * np.sin(x)) / (denom ** 2)
sin_terms = np.sin(theta)
integrand = np.cos(x / denom) * g_prime - (sin_terms * freq_mag[None, :]).sum(axis=1)

left_value = float(target[0])
model = bspf1d.from_grid(
    degree=degree,
    x=x,
    domain=(domain_start, domain_end),
    order=degree,
    n_basis=n_basis,
    num_boundary_points=num_boundary_points,
    use_clustering=use_clustering,
    clustering_factor=clustering_factor,
)
F_bspf, integrand_spline = model.antiderivative(
    integrand,
    order=1,
    left_value=left_value,
    lam=reg,
)
integral_bspf = model.definite_integral(integrand, lam=reg)

F_simpson = integrate.cumulative_simpson(integrand, x=x, initial=left_value)
integral_simpson = integrate.simpson(integrand, x=x)

integral_exact = float(target[-1] - target[0])
linf_bspf = float(np.max(np.abs(F_bspf - target)))
linf_simpson = float(np.max(np.abs(F_simpson - target)))

# Convergence grids
grid_sizes = np.geomspace(1001, 2001, 20).astype(int)
grid_sizes = grid_sizes + ((grid_sizes + 1) % 2)  # ensure odd counts for Simpson
grid_sizes = np.unique(np.append(grid_sizes, num_points))

errors_bspf = []
errors_simpson = []

for n in grid_sizes:
    print(f"Convergence study: n = {n}")
    x_n = np.linspace(domain_start, domain_end, int(n), endpoint=True, dtype=np.float64)
    denom_n = offset + np.cos(x_n)
    theta_n = x_n[:, None] * frequencies[None, :] + phases[None, :]

    base_value_n = np.sin(x_n / denom_n)
    cos_terms_n = np.cos(theta_n) @ magnitudes
    target_n = base_value_n + cos_terms_n

    g_prime_n = (denom_n + x_n * np.sin(x_n)) / (denom_n ** 2)
    sin_terms_n = np.sin(theta_n)
    integrand_n = np.cos(x_n / denom_n) * g_prime_n - (sin_terms_n * freq_mag[None, :]).sum(axis=1)

    model_n = bspf1d.from_grid(
        degree=degree,
        x=x_n,
        domain=(domain_start, domain_end),
        order=degree,
        n_basis=n_basis,
        num_boundary_points=num_boundary_points,
        use_clustering=use_clustering,
        clustering_factor=clustering_factor,
    )
    F_bspf_n, _ = model_n.antiderivative(
        integrand_n,
        order=1,
        left_value=float(target_n[0]),
        lam=reg,
    )
    F_simpson_n = integrate.cumulative_simpson(
        integrand_n,
        x=x_n,
        initial=float(target_n[0]),
    )

    errors_bspf.append(float(np.max(np.abs(F_bspf_n - target_n))))
    errors_simpson.append(float(np.max(np.abs(F_simpson_n - target_n))))

errors_bspf = np.asarray(errors_bspf)
errors_simpson = np.asarray(errors_simpson)


print(
    "Definite integral results over"
    f" [{domain_start:.3f}, {domain_end:.3f}]"
)
print(f"  Exact    : {integral_exact:.10f}")
print(
    "  BSPF     : "
    f"{integral_bspf:.10f}  (error = {abs(integral_bspf - integral_exact):.2e})"
)
print(
    "  Simpson  : "
    f"{integral_simpson:.10f}  (error = {abs(integral_simpson - integral_exact):.2e})"
)
print("Antiderivative Linf error on evaluation grid:")
print(f"  BSPF     : {linf_bspf:.2e}")
print(f"  Simpson  : {linf_simpson:.2e}")

plt.rcParams.update(
    {
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 16,
        "figure.titlesize": 24,
        "axes.grid": True,
        "grid.alpha": 0.5,
    }
)

plt.figure(figsize=(16, 10))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
simpson_color = colors[2 % len(colors)] if len(colors) >= 3 else "tab:green"

# (a) Integrand and spline approximation
plt.subplot(2, 2, 1)
plt.plot(x, integrand, label=integrand_label, linewidth=1.0)
plt.plot(x, integrand_spline, label=f"{integrand_label} (spline)", linewidth=1.2)
plt.xlabel("$x$")
plt.ylabel(integrand_label)
plt.title("(a)", loc="left", x=-0.15, fontsize=24, fontweight="bold")
plt.legend(ncol=1)

# (b) Antiderivatives
plt.subplot(2, 2, 2)
plt.plot(x, target, color="k", label="Exact", linewidth=1.0)
plt.plot(x, F_bspf, label="BSPF", linewidth=1.2)
plt.plot(x, F_simpson, label="Simpson", linewidth=1.2, color=simpson_color)
plt.xlabel("$x$")
plt.ylabel(target_label)
plt.title("(b)", loc="left", x=-0.15, fontsize=24, fontweight="bold")
plt.legend(ncol=1)

# (c) Pointwise errors
plt.subplot(2, 2, 3)
plt.semilogy(x, np.abs(F_bspf - target), label="BSPF", linewidth=1.0)
plt.semilogy(x, np.abs(F_simpson - target), label="Simpson", linewidth=1.0, color=simpson_color)
plt.xlabel("$x$")
plt.ylabel("$|Error|$")
plt.title("(c)", loc="left", x=-0.15, fontsize=24, fontweight="bold")
plt.legend(ncol=1)

# (d) Convergence study
plt.subplot(2, 2, 4)
plt.loglog(grid_sizes, errors_bspf, "o-", label="BSPF", linewidth=1.0)
plt.loglog(grid_sizes, errors_simpson, "s-", label="Simpson", linewidth=1.0, color=simpson_color)
plt.xlabel("$N$")
plt.ylabel("$\\|\\mathrm{Error}\\|_\\infty$")
plt.title("(d)", loc="left", x=-0.15, fontsize=24, fontweight="bold")
plt.legend(ncol=1)

plt.tight_layout()
plt.show()

