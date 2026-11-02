# SPDX-License-Identifier: GPL-3.0-or-later
"""Parity of bspf_cpp with pybspf (JAX, float64) on the same linear maps.

Two float64 implementations agree only up to their rounding, which high-order
derivatives amplify by up to |D_k| ~ omega^k. Each comparison therefore allows a
multiple of the reference's own rounding floor: how far pybspf's output moves
when the input is perturbed at the 1e-16 level. tests/python/test_accuracy.py
checks both against a 40-digit evaluation of the same map."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pybspf
import pytest

import bspf_cpp as bc

RTOL = 1e-12

CONFIGS = [
    dict(degree=5),
    dict(degree=3),
    dict(degree=7),
    dict(degree=5, lam=1e-6),
    dict(degree=5, clustering=1.5),
    dict(degree=5, n_basis=30),
    dict(degree=4, constraint_order=2, boundary_points=6),
    dict(degree=5, endpoint_method="chebyshev"),
    dict(degree=6, endpoint_method="chebyshev", chebyshev_modes=8, boundary_points=14),
]
SIZES = [48, 63, 128]


def field(x, batch=3):
    t = (x - x[0]) / (x[-1] - x[0])
    cols = [np.sin(3 * t + c) + np.exp(0.7 * t) * c + t ** 7 for c in range(batch)]
    return np.stack(cols, axis=1)


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)


def close(a, b, rtol=RTOL):
    err = rel(a, b)
    assert err < rtol, err


def floor(reference, f):
    """Relative change of reference(f) under a ~1e-16 relative input perturbation."""
    g = f * (1 + 1e-16 * np.random.default_rng(7).standard_normal(f.shape))
    return rel(reference(g), reference(f))


def agrees(ours, reference, f, factor=30.0):
    """ours(f) matches reference(f) within `factor` times the reference's rounding floor."""
    close(ours(f), reference(f), max(RTOL, factor * floor(reference, f)))


def check(op, reference, f, axis=0):
    agrees(lambda g: op(g, axis=axis), reference, f)


def pair(n, options, a=-0.4, b=1.3):
    x = np.linspace(a, b, n)
    return x, bc.Line(x, **options), pybspf.plan_1d(jnp.asarray(x), **options)


@pytest.mark.parametrize("options", CONFIGS)
@pytest.mark.parametrize("n", SIZES)
def test_plan_matrices(n, options):
    x, line, plan = pair(n, options)
    close(line.knots, plan.knots, 1e-15)
    close(line.basis, plan.basis[0], 1e-14)
    q, p = line.constraint_order, line.boundary_points
    close(line.jet_map[:q, :p], plan.boundary_blocks[0], 1e-11)
    close(line.jet_map[q:, -p:], plan.boundary_blocks[1], 1e-11)


@pytest.mark.parametrize("options", CONFIGS)
@pytest.mark.parametrize("n", SIZES)
def test_derivatives(n, options):
    x, line, plan = pair(n, options)
    f = field(x)
    for order in range(5):
        check(line.derivative(order),
              lambda g: pybspf.differentiate(plan, g, order=order), f)
        check(line.derivative(order, correction=False),
              lambda g: pybspf.differentiate(plan, g, order=order, correction=False), f)


@pytest.mark.parametrize("options", CONFIGS[:4])
def test_given_jets(options):
    x, line, plan = pair(64, options)
    f = field(x)
    q = line.constraint_order
    rng = np.random.default_rng(0)
    jets = np.asarray(pybspf.endpoint_jets(plan, f)) + 1e-2 * rng.standard_normal((2, q, f.shape[1]))
    ours = jets.reshape(2 * q, -1)  # left 0..q-1, then right 0..q-1
    for order in (0, 1, 2):
        op = line.derivative(order)
        reference = lambda g: pybspf.differentiate(plan, g, order=order, boundary=jets)
        close(op(f, jets=ours), reference(f), max(RTOL, 100 * floor(reference, f)))
    # A single jet vector is broadcast over every line.
    single = jets[:, :, 0].reshape(-1)
    reference = lambda g: pybspf.differentiate(plan, g, order=1, boundary=jets[:, :, 0])
    op = line.derivative(1)
    close(op(f, jets=single), reference(f), max(RTOL, 100 * floor(reference, f)))


@pytest.mark.parametrize("options", CONFIGS)
@pytest.mark.parametrize("n", SIZES)
def test_interpolation(n, options):
    x, line, plan = pair(n, options)
    f = field(x)
    t = np.concatenate([[x[0], x[-1]], np.random.default_rng(1).uniform(x[0], x[-1], 41)])
    for k in range(4):
        check(line.evaluate(t, derivative=k),
              lambda g: pybspf.interpolate(plan, g, t, derivative=k), f)


@pytest.mark.parametrize("options", CONFIGS)
@pytest.mark.parametrize("n", SIZES)
def test_integration(n, options):
    x, line, plan = pair(n, options)
    f = field(x)
    for a, b in [(None, None), (x[0], 0.2), (0.9, -0.1), (0.31, 0.31), (0.0, x[-1])]:
        ref = pybspf.integrate(plan, f, a=a, b=b)
        close(line.quadrature(a, b)(f)[0], ref)
    for order in (1, 2):
        close(line.primitive(order)(f), pybspf.antiderivative(plan, f, order=order))


def test_out_of_domain_is_rejected():
    x, line, _ = pair(40, {})
    with pytest.raises(ValueError):
        line.evaluate(np.array([x[-1] + 1e-3]))
    with pytest.raises(ValueError):
        line.quadrature(x[0] - 1e-3, 0.0)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_axis_application(axis):
    shapes = [(31, 20, 2), (20, 31, 2), (2, 20, 31)]
    x = np.linspace(0, 1, 31)
    line = bc.Line(x)
    plan = pybspf.plan_1d(jnp.asarray(x))
    f = np.random.default_rng(2).standard_normal(shapes[axis])
    reference = lambda g: np.moveaxis(pybspf.differentiate(plan, np.moveaxis(g, axis, 0), order=1), 0, axis)
    check(line.derivative(1), reference, f, axis)
    # Non-contiguous and non-float64 input is converted.
    g = np.asfortranarray(f).astype(np.float32)
    close(line.derivative(1)(g, axis=axis), reference(g.astype(float)), max(RTOL, 30 * floor(reference, g.astype(float))))


def grids(dimension, options):
    axes = [np.linspace(0, 1, 26), np.linspace(-1, 0.5, 33), np.linspace(0, 2, 21)][:dimension]
    return axes, bc.Grid.uniform(axes, **options), pybspf.tensor_plan(*[pybspf.plan_1d(jnp.asarray(a), **options) for a in axes])


def tensor_field(axes, batch=(2,)):
    mesh = np.meshgrid(*axes, indexing="ij")
    base = np.sin(2 * mesh[0]) * np.exp(sum(m for m in mesh[1:]) if len(mesh) > 1 else 0 * mesh[0]) + mesh[0] ** 3
    return np.stack([base * (c + 1) + c for c in range(batch[0])], axis=-1)


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("options", [dict(degree=5), dict(degree=4, endpoint_method="chebyshev")])
def test_tensor_operations(dimension, options):
    axes, grid, plan = grids(dimension, options)
    f = tensor_field(axes)
    orders_list = [(1,), (2,)] if dimension == 1 else [(1, 0), (0, 2), (1, 1)] if dimension == 2 else [(1, 0, 0), (0, 1, 2), (1, 1, 1)]
    for orders in orders_list:
        agrees(lambda g: grid.partial(g, orders), lambda g: pybspf.mixed_partial(plan, g, orders), f)
    agrees(grid.gradient, lambda g: pybspf.gradient(plan, g), f)
    agrees(grid.laplacian, lambda g: pybspf.laplacian(plan, g), f)
    coords = [np.random.default_rng(3 + i).uniform(a[0], a[-1], 7 + i) for i, a in enumerate(axes)]
    agrees(lambda g: grid.interpolate(g, coords), lambda g: pybspf.interpolate_grid(plan, g, coords), f)
    deriv = [1] + [0] * (dimension - 1)

    def reference(g):  # axis by axis with 1D pybspf plans
        for i, (p, c) in enumerate(zip(plan.axes, coords)):
            g = np.moveaxis(np.asarray(pybspf.interpolate(p, np.moveaxis(g, i, 0), c, derivative=deriv[i])), 0, i)
        return g

    agrees(lambda g: grid.interpolate(g, coords, deriv), reference, f)
    close(grid.integrate(f), pybspf.integrate_box(plan, f))
    bounds = [(a[0] + 0.1, a[-1] - 0.2) for a in axes]
    close(grid.integrate(f, bounds), pybspf.integrate_box(plan, f, bounds))


def test_vector_calculus():
    axes, grid, plan = grids(3, {})
    v = np.stack([tensor_field(axes, (1,))[..., 0] * (i + 1) for i in range(3)])
    agrees(grid.divergence, lambda g: pybspf.divergence(plan, g), v)
    agrees(grid.curl, lambda g: pybspf.curl(plan, g), v)
    agrees(grid.hessian, lambda g: pybspf.hessian(plan, g), v[0])
