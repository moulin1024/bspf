# SPDX-License-Identifier: GPL-3.0-or-later
"""The energy-stable (Galerkin) Laplacian: exact products, real spectrum, grid form."""
import numpy as np
import pytest

import bspf_cpp as bc

CHEB4 = dict(degree=9, n_basis=18, lam=1e-6, endpoint_method="chebyshev", boundary_points=16,
             chebyshev_modes=12, chebyshev_alpha=1e-12, constraint_order=4)
CONFIGS = [dict(), dict(degree=6), CHEB4]


def dense_reference(line, order=16):
    """Mass and stiffness of the cardinal trial functions by Gauss rules between samples and knots."""
    edges = np.unique(np.concatenate((line.x, line.knots)))
    nodes, weights = np.polynomial.legendre.leggauss(order)
    widths = np.diff(edges)
    points = np.clip((edges[:-1, None] + widths[:, None] * (nodes + 1) / 2).ravel(), line.x[0], line.x[-1])
    w = (widths[:, None] * weights / 2).ravel()
    phi, dphi = line.evaluate(points).matrix, line.evaluate(points, derivative=1).matrix
    return phi.T @ (w[:, None] * phi), dphi.T @ (w[:, None] * dphi)


@pytest.mark.parametrize("options", CONFIGS)
@pytest.mark.parametrize("n", [24, 33, 64])
def test_products_match_dense_quadrature(n, options):
    line = bc.Line(np.linspace(-0.2, 1.1, n), **options)
    g = bc.GalerkinLine(line)
    M, K = dense_reference(line)
    eye = np.eye(n)
    for fast, dense in ((g.mass(eye), M), (g.stiffness(eye), K)):
        assert np.abs(fast - dense).max() < 1e-9 * np.abs(dense).max()
        # Symmetric by construction; rounding grows with the fit's endpoint-jet factors
        # (unit samples next to a wall give jets of size dx^-(q-1)).
        assert np.abs(fast - fast.T).max() < 1e-10 * np.abs(dense).max()
    assert np.linalg.eigvalsh(g.mass(eye)).min() > 0


@pytest.mark.parametrize("options", CONFIGS)
def test_dirichlet_spectrum_is_real_and_accurate(options):
    n = 65
    g = bc.GalerkinLine(bc.Line(np.linspace(0, 1, n), **options))
    eigenvalues = np.linalg.eigvals(g.second_derivative(np.eye(n))[:, 1:-1])
    assert np.abs(eigenvalues.imag).max() == 0 or np.abs(eigenvalues.imag).max() < 1e-9 * np.abs(eigenvalues).max()
    assert eigenvalues.real.max() < 0
    lowest = np.sort(eigenvalues.real)[::-1][:6]
    exact = -(np.pi * np.arange(1, 7)) ** 2
    assert np.abs(lowest - exact).max() < 1e-5 * abs(exact[-1])


def test_strong_chebyshev_operator_is_not_wave_stable():
    """The motivation: the collocation Laplacian with Chebyshev jets has complex modes."""
    line = bc.Line(np.linspace(0, 1, 65), **CHEB4)
    strong = np.linalg.eigvals(line.derivative(2).matrix[1:-1, 1:-1])
    assert np.abs(strong.imag).max() > 1.0


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_grid_weak_laplacian_matches_kronecker_form(dimension):
    axes = [np.linspace(0, 1, 13), np.linspace(-1, 0.5, 11), np.linspace(0, 2, 12)][:dimension]
    grid = bc.Grid.uniform(axes, degree=5)
    rng = np.random.default_rng(1)
    u = rng.standard_normal(tuple(a.size for a in axes) + (2,))
    walls = rng.standard_normal(u.shape)
    out = grid.weak_laplacian(u, walls)
    W = [grid.galerkin(i).second_derivative(np.eye(a.size)) for i, a in enumerate(axes)]
    E = [grid.galerkin(i).extension(np.eye(a.size)) for i, a in enumerate(axes)]

    def apply(ops, f):
        for axis, op in enumerate(ops):
            f = np.moveaxis(np.tensordot(op, np.moveaxis(f, axis, 0), axes=(1, 0)), 0, axis)
        return f

    mask = np.zeros(u.shape[:dimension], dtype=bool)
    mask[tuple(slice(1, -1) for _ in range(dimension))] = True
    w0 = np.where(mask[(...,) + (None,)], 0.0, walls)
    expected = sum(apply([W[a] if b == a else E[b] for b in range(dimension)], u) for a in range(dimension))
    expected = expected - apply(E, w0)
    inner = tuple(slice(1, -1) for _ in range(dimension))
    np.testing.assert_allclose(out[inner], expected, rtol=1e-12, atol=1e-10 * np.abs(expected).max())
    np.testing.assert_array_equal(out[~mask], walls[~mask])


def test_wave_energy_is_conserved_with_homogeneous_walls():
    """u_tt = Lap u, u = 0 on the walls: M-weighted energy is conserved up to RK4 dissipation."""
    x = np.linspace(0, 1, 41)
    grid = bc.Grid.uniform((x, x), **CHEB4)
    g = grid.galerkin(0)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = np.exp(-80 * ((X - 0.4) ** 2 + (Y - 0.55) ** 2))
    u[[0, -1], :] = u[:, [0, -1]] = 0.0
    v = np.zeros_like(u)
    eye = np.eye(x.size)
    M = g.mass(eye)[1:-1, 1:-1]
    K = g.stiffness(eye)

    def energy(u, v):  # 1/2 v^T (M x M) v + 1/2 u^T (K x M + M x K) u on the interior
        ui, vi = u[1:-1, 1:-1], v[1:-1, 1:-1]
        Ki = K[1:-1, 1:-1]
        return 0.5 * np.sum(vi * (M @ vi @ M)) + 0.5 * np.sum(ui * (Ki @ ui @ M + M @ ui @ Ki))

    rho = 2 * np.abs(np.linalg.eigvals(g.second_derivative(eye)[:, 1:-1])).max()
    dt = 0.5 * 2.83 / np.sqrt(rho)
    e0 = energy(u, v)
    for _ in range(400):
        f = lambda u, v: (v, grid.weak_laplacian(u))
        k1 = f(u, v); k2 = f(u + dt / 2 * k1[0], v + dt / 2 * k1[1])
        k3 = f(u + dt / 2 * k2[0], v + dt / 2 * k2[1]); k4 = f(u + dt * k3[0], v + dt * k3[1])
        u = u + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v = v + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    e1 = energy(u, v)
    assert e1 <= e0 * (1 + 1e-10)  # never grows
    assert e1 > 0.99 * e0          # and RK4 only damps it slightly


@pytest.mark.parametrize("options", CONFIGS)
@pytest.mark.parametrize("n", [33, 64, 128, 256])
def test_compressed_operator_matches_exact(n, options):
    """The applied form D_per + P Q^T reproduces W; its rank stays bounded.

    Smooth inputs only: on rough samples near a wall the exact (split) form
    itself rounds like eps |F|^2 (F: the fit, whose endpoint jets grow like
    (N / n_basis)^(q-1)), so it is not a reference there."""
    x = np.linspace(0, 1, n)
    g = bc.GalerkinLine(bc.Line(x, **options))
    u = np.column_stack([np.sin(3.1 * x + 0.4), np.exp(x) * np.cos(7 * x), np.cosh(2 * x - 0.3)])
    exact, applied = g.exact_second_derivative(u), g.second_derivative(u)
    scale = (np.pi / (x[1] - x[0])) ** 2
    assert np.abs(applied - exact).max() < 1e-11 * scale * np.abs(u).max()
    assert g.operator_rank <= 64
