# SPDX-License-Identifier: GPL-3.0-or-later
"""FFT quadrature, weighted adjoints, and interior/full structured mass solves."""
import numpy as np
import pytest
import bspf_cpp as bc

OPTIONS = dict(degree=9, n_basis=18, lam=0., constraint_order=4,
               endpoint_method="chebyshev", boundary_points=16, chebyshev_modes=12)


@pytest.mark.parametrize("n", [32, 65, 129, 257])
@pytest.mark.parametrize("order", [8, 16])
def test_quadrature_and_adjoint(n, order):
    line = bc.Line(np.linspace(-.2, 1.3, n), **OPTIONS)
    g = bc.GalerkinLine(line, quadrature_order=order)
    points, weights = g.quadrature_points, g.quadrature_weights
    assert np.all(weights > 0)
    assert points.min() > -.2 and points.max() < 1.3
    np.testing.assert_allclose(weights.sum(), 1.5, atol=2e-14)
    rng = np.random.default_rng(17)
    u, v = rng.normal(size=(n, 3)), rng.normal(size=(len(points), 3))
    for derivative in (0, 1):
        phi = line.evaluate(points, derivative=derivative).matrix
        actual, projected = g.evaluate(u, derivative), g.project(v, derivative)
        np.testing.assert_allclose(actual, phi@u, atol=1e-10*max(1, np.abs(phi@u).max()), rtol=1e-10)
        np.testing.assert_allclose(projected, phi.T@(weights[:, None]*v), atol=1e-9, rtol=1e-9)
        # Adjoint identity is independent of explicit cardinal matrix assembly.
        np.testing.assert_allclose(np.sum(weights[:, None]*actual*v), np.sum(u*projected), atol=1e-9, rtol=1e-9)
        product = g.mass(u) if derivative == 0 else g.stiffness(u)
        np.testing.assert_allclose(g.project(actual, derivative), product, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("n", [32, 65, 129, 257])
def test_full_and_interior_mass_inverse(n):
    line = bc.Line(np.linspace(0, 1, n), **OPTIONS)
    g = bc.GalerkinLine(line, quadrature_order=8)
    phi = line.evaluate(g.quadrature_points).matrix
    mass = phi.T@(g.quadrature_weights[:, None]*phi)
    rhs = np.random.default_rng(5).normal(size=(n, 4))
    for interior in (False, True):
        load, block = (rhs[1:-1], mass[1:-1, 1:-1]) if interior else (rhs, mass)
        solution = g.solve_mass(load, interior)
        np.testing.assert_allclose(block@solution, load, atol=2e-8, rtol=2e-8)
        np.testing.assert_allclose(solution, np.linalg.solve(block, load), atol=2e-8, rtol=2e-8)


def test_transform_arguments():
    line = bc.Line(np.linspace(0, 1, 33), **OPTIONS)
    with pytest.raises(ValueError):
        bc.GalerkinLine(line, quadrature_order=3)
    g = bc.GalerkinLine(line)
    for call in (lambda: g.evaluate(np.zeros((32, 1))),
                 lambda: g.evaluate(np.zeros((33, 1)), 2),
                 lambda: g.project(np.zeros((2, 1))),
                 lambda: g.project(np.zeros((len(g.quadrature_points), 1)), -1),
                 lambda: g.solve_mass(np.zeros((33, 1)), True),
                 lambda: g.solve_mass(np.zeros((31, 1)))):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize('n,nb', [(129,41),(257,73)])
def test_many_endpoint_splines_keep_adjoint_and_mass(n,nb):
    line=bc.Line(np.linspace(0,1,n),**(OPTIONS|dict(n_basis=nb)))
    g=bc.GalerkinLine(line,quadrature_order=12)
    rng=np.random.default_rng(52)
    u=rng.normal(size=(n,3))
    v=rng.normal(size=(len(g.quadrature_points),3))
    for derivative in (0,1):
        phi=line.evaluate(g.quadrature_points,derivative=derivative).matrix
        np.testing.assert_allclose(g.evaluate(u,derivative),phi@u,atol=1e-9,rtol=1e-9)
        np.testing.assert_allclose(g.project(v,derivative),phi.T@(g.quadrature_weights[:,None]*v),atol=1e-9,rtol=1e-9)
    np.testing.assert_allclose(g.mass(g.solve_mass(u)),u,atol=2e-8,rtol=2e-8)
