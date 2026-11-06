# SPDX-License-Identifier: GPL-3.0-or-later
"""Structured grids on deformed 2D regions: spline curves, transfinite map, physical derivatives."""
from math import comb

import numpy as np
import pytest

import bspf_cpp as bc

CHEB6 = dict(degree=9, n_basis=18, lam=1e-6, endpoint_method="chebyshev", boundary_points=16,
             chebyshev_modes=12, chebyshev_alpha=1e-12, constraint_order=6)


def line_curve(a, b):
    return bc.SplineCurve(np.array([a, b], dtype=float), 1)


def bezier_side(p0, p1, bump, normal, degree=5):
    t = np.linspace(0, 1, degree + 1)[:, None]
    pts = (1 - t) * np.asarray(p0, float) + t * np.asarray(p1, float)
    pts[1:-1] += bump * np.sin(np.pi * t[1:-1]) * np.asarray(normal, float)
    return bc.SplineCurve(pts, degree)


def curved_sides():
    P00, P10, P01, P11 = (0.0, 0.0), (1.0, 0.1), (-0.1, 1.0), (1.1, 1.2)
    return (bezier_side(P00, P10, 0.12, (0, 1)), bezier_side(P10, P11, 0.10, (1, 0)),
            bezier_side(P01, P11, 0.08, (0, 1)), bezier_side(P00, P01, -0.10, (1, 0)))


def f(x, y):   return np.sin(2 * x + 0.5) * np.cos(3 * y - 0.2) + 0.5 * np.exp(x * y)
def fx(x, y):  return 2 * np.cos(2 * x + 0.5) * np.cos(3 * y - 0.2) + 0.5 * y * np.exp(x * y)
def fy(x, y):  return -3 * np.sin(2 * x + 0.5) * np.sin(3 * y - 0.2) + 0.5 * x * np.exp(x * y)
def fxx(x, y): return -4 * np.sin(2 * x + 0.5) * np.cos(3 * y - 0.2) + 0.5 * y * y * np.exp(x * y)
def fyy(x, y): return -9 * np.sin(2 * x + 0.5) * np.cos(3 * y - 0.2) + 0.5 * x * x * np.exp(x * y)
def fxy(x, y): return -6 * np.cos(2 * x + 0.5) * np.sin(3 * y - 0.2) + 0.5 * (1 + x * y) * np.exp(x * y)


def rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


def test_bezier_curve_matches_bernstein_form():
    pts = np.array([[0, 0], [0.3, 0.5], [0.7, -0.2], [1.0, 0.4]])
    curve = bc.SplineCurve(pts, 3)
    s = np.linspace(0, 1, 11)
    bern = np.stack([comb(3, k) * s**k * (1 - s) ** (3 - k) for k in range(4)], axis=1)
    np.testing.assert_allclose(curve.evaluate(s), bern @ pts, atol=1e-14)
    dbern = np.stack([comb(3, k) * (k * s ** max(k - 1, 0) * (1 - s) ** (3 - k) - (3 - k) * s**k * (1 - s) ** max(2 - k, 0))
                      for k in range(4)], axis=1)
    np.testing.assert_allclose(curve.evaluate(s, 1), dbern @ pts, atol=1e-13)


def test_unit_square_reproduces_the_cartesian_grid():
    sides = (line_curve((0, 0), (1, 0)), line_curve((1, 0), (1, 1)), line_curve((0, 1), (1, 1)), line_curve((0, 0), (0, 1)))
    g = bc.MappedGrid2D(*sides, (21, 17))
    np.testing.assert_allclose(g.jacobian, 1.0, atol=1e-15)
    u = f(g.x, g.y)[..., None] * np.array([1.0, 2.0])  # with a batch axis
    for orders in ((1, 0), (0, 1), (2, 0), (0, 2), (1, 1)):
        np.testing.assert_allclose(g.partial(u, orders), g.reference.partial(u, orders), rtol=1e-12, atol=1e-11)


def test_rectangle_scales_the_derivatives():
    sides = (line_curve((0, 0), (2, 0)), line_curve((2, 0), (2, 3)), line_curve((0, 3), (2, 3)), line_curve((0, 0), (0, 3)))
    g = bc.MappedGrid2D(*sides, (33, 41), **CHEB6)
    u = f(g.x, g.y)
    assert rel(g.partial(u, (1, 0)), fx(g.x, g.y)) < 1e-6
    assert rel(g.partial(u, (0, 2)), fyy(g.x, g.y)) < 1e-4


@pytest.mark.parametrize("orders,exact", [((1, 0), fx), ((0, 1), fy), ((2, 0), fxx), ((0, 2), fyy), ((1, 1), fxy)])
def test_curved_domain_converges(orders, exact):
    errors = []
    for n in (17, 33, 65):
        g = bc.MappedGrid2D(*curved_sides(), (n, n), **CHEB6)
        errors.append(rel(g.partial(f(g.x, g.y), orders), exact(g.x, g.y)))
    assert errors[2] < errors[0] / 1000
    assert errors[2] < (1e-9 if sum(orders) == 1 else 5e-8)


def test_helpers_and_metrics():
    g = bc.MappedGrid2D(*curved_sides(), (33, 29), **CHEB6)
    u = f(g.x, g.y)
    grad = g.gradient(u)
    assert grad.shape == (2, 33, 29)
    np.testing.assert_allclose(g.laplacian(u), g.partial(u, (2, 0)) + g.partial(u, (0, 2)))
    # The exact metrics agree with BSPF derivatives of the node coordinates
    # (to the 33-point differentiation error, about 5e-8).
    assert rel(g.reference.partial(g.x, (1, 0)), g.x_xi) < 1e-6
    assert rel(g.reference.partial(g.y, (0, 1)), g.y_eta) < 1e-6


def test_invalid_geometry_is_rejected():
    bottom, right, top, left = curved_sides()
    with pytest.raises(ValueError, match="corners"):
        bc.MappedGrid2D(bottom, right, top, line_curve((0, 0), (0, 1.5)), (9, 9))
    # Swapping top and bottom flips the orientation: the map folds.
    with pytest.raises(ValueError, match="folds"):
        bc.MappedGrid2D(top, line_curve((1.1, 1.2), (1.0, 0.1)), bottom, line_curve((-0.1, 1.0), (0, 0)), (9, 9))


def test_distributed_corrected_map_and_physical_chain_rule():
    """Independent analytic map checks distribution inversion and interior metrics."""
    sides = (line_curve((0, 0), (1, 0)), line_curve((1, 0), (1, 1)),
             line_curve((0, 1), (1, 1)), line_curve((0, 0), (0, 1)))
    n = 129
    g = bc.MappedGrid2D(*sides, (n, n), dist_xi=np.array([0., .35, 1.]),
                        correction=(np.array([[.2]]), np.array([[-.1]])), **CHEB6)
    xi, eta = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    # psi(s) = .7*s + .3*s**2; rationalized inverse avoids cancellation at zero.
    s = 2 * xi / (.7 + np.sqrt(.49 + 1.2 * xi))
    bubble = 4 * xi * (1 - xi) * eta * (1 - eta)
    bx = 4 * (1 - 2 * xi) * eta * (1 - eta)
    by = 4 * xi * (1 - xi) * (1 - 2 * eta)
    np.testing.assert_allclose(g.x, s + .2 * bubble, atol=2e-13)
    np.testing.assert_allclose(g.y, eta - .1 * bubble, atol=2e-13)
    np.testing.assert_allclose(g.x_xi, 1 / (.7 + .6 * s) + .2 * bx, atol=2e-12)
    np.testing.assert_allclose(g.x_eta, .2 * by, atol=2e-12)
    np.testing.assert_allclose(g.y_xi, -.1 * bx, atol=2e-12)
    np.testing.assert_allclose(g.y_eta, 1 - .1 * by, atol=2e-12)
    u = g.x**2 + g.x * g.y + 2 * g.y**2
    for orders, exact in [((1, 0), 2*g.x + g.y), ((0, 1), g.x + 4*g.y),
                          ((2, 0), 2.), ((1, 1), 1.), ((0, 2), 4.)]:
        np.testing.assert_allclose(g.partial(u, orders), exact, atol=2e-7, rtol=0)
