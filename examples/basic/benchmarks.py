# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmark fields from pybspf's operation examples, with exact derivatives.

The fields, domains and BSPF settings follow pybspf/examples/operation:
* 1D: f(x) = sin(x / (beta + cos x)) on [0, 2 pi] (differentiate_1d, integrate_1d)
* 2D: cosine-mode background plus a circular tanh transition (differentiate_2d)
* 3D: shifted Taylor-Green vortex on [0, 3 pi / 2]^3 (differentiate_3d)
pybspf differentiates them with JAX; here the derivatives are written out.
"""
import numpy as np

# --- 1D: nonlinear phase -------------------------------------------------------
BETA = 1.05  # 1.05 recovers the sharper original benchmark (needs a finer grid)
DOMAIN_1D = (0.0, 2 * np.pi)
OPTIONS_1D = dict(degree=9, n_basis=20, constraint_order=9, boundary_points=13, clustering=2.0, lam=1e-3)


def phase(x, beta=BETA):
    """f(x) = sin(u), u = x / (beta + cos x)."""
    return np.sin(x / (beta + np.cos(x)))


def phase_derivative(x, beta=BETA):
    """f'(x) = cos(u) u', u' = (beta + cos x + x sin x) / (beta + cos x)^2."""
    d = beta + np.cos(x)
    return np.cos(x / d) * (d + x * np.sin(x)) / d**2


# --- 2D: smooth modes plus a circular front ------------------------------------
DOMAIN_2D = (0.0, 2 * np.pi)
OPTIONS_2D = dict(degree=9, n_basis=20, boundary_points=13, lam=1e-6)
KX = np.array([1, 2, 3, 1, 2, 4], dtype=float)
KY = np.array([2, 1, 1, 3, 3, 2], dtype=float)
# pybspf draws these with jax.random.key(123); NumPy's generator gives other values.
AMPLITUDES = np.random.default_rng(123).standard_normal(6) / (KX**2 + KY**2)


def front(x, y):
    """Background sum a cos(kx x) cos(ky y) plus 0.5 (1 + tanh((r^2 - 1) / 0.6)) about (pi, pi)."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    background = sum(a * np.cos(kx * X) * np.cos(ky * Y) for a, kx, ky in zip(AMPLITUDES, KX, KY))
    r2 = (X - np.pi) ** 2 + (Y - np.pi) ** 2
    return background + 0.5 * (1 + np.tanh((r2 - 1) / 0.6))


def front_dx(x, y):
    X, Y = np.meshgrid(x, y, indexing="ij")
    background = sum(-a * kx * np.sin(kx * X) * np.cos(ky * Y) for a, kx, ky in zip(AMPLITUDES, KX, KY))
    r2 = (X - np.pi) ** 2 + (Y - np.pi) ** 2
    return background + 0.5 / np.cosh((r2 - 1) / 0.6) ** 2 * 2 * (X - np.pi) / 0.6


def front_dy(x, y):
    X, Y = np.meshgrid(x, y, indexing="ij")
    background = sum(-a * ky * np.cos(kx * X) * np.sin(ky * Y) for a, kx, ky in zip(AMPLITUDES, KX, KY))
    r2 = (X - np.pi) ** 2 + (Y - np.pi) ** 2
    return background + 0.5 / np.cosh((r2 - 1) / 0.6) ** 2 * 2 * (Y - np.pi) / 0.6


# --- 3D: shifted Taylor-Green vortex -------------------------------------------
LENGTH_3D = 1.5 * np.pi
SHIFT_3D = (0.3, 0.5, 0.7)
OPTIONS_3D = dict(degree=9, n_basis=20, lam=1e-6, endpoint_method="chebyshev",
                  boundary_points=16, chebyshev_modes=12, chebyshev_alpha=1e-12)


def vortex(x, y, z):
    """Velocity (3, nx, ny, nz), Jacobian (derivative axis, nx, ny, nz, component), vorticity (3, ...).

    u = (sin X cos Y cos Z, -cos X sin Y cos Z, 0) with X = x - 0.3, Y = y - 0.5, Z = z - 0.7.
    It is divergence-free and its Laplacian is -3 u. It is not periodic on the box.
    """
    X, Y, Z = np.meshgrid(x - SHIFT_3D[0], y - SHIFT_3D[1], z - SHIFT_3D[2], indexing="ij")
    sx, sy, sz = np.sin(X), np.sin(Y), np.sin(Z)
    cx, cy, cz = np.cos(X), np.cos(Y), np.cos(Z)
    zero = np.zeros_like(X)
    velocity = np.stack((sx * cy * cz, -cx * sy * cz, zero))
    jacobian = np.stack((np.stack((cx * cy * cz, sx * sy * cz, zero), axis=-1),
                         np.stack((-sx * sy * cz, -cx * cy * cz, zero), axis=-1),
                         np.stack((-sx * cy * sz, cx * sy * sz, zero), axis=-1)))
    vorticity = np.stack((-cx * sy * sz, -sx * cy * sz, 2 * sx * sy * cz))
    return velocity, jacobian, vorticity


def relative_max(actual, exact):
    return float(np.max(np.abs(actual - exact)) / np.max(np.abs(exact)))


def relative_l2(actual, exact):
    return float(np.linalg.norm(actual - exact) / np.linalg.norm(exact))


def odd_sizes(lo, hi, count):
    """Geometrically spaced odd grid sizes (odd keeps the center point)."""
    return sorted({2 * round((float(n) - 1) / 2) + 1 for n in np.geomspace(lo, hi, count)})
