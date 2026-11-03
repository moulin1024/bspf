# SPDX-License-Identifier: GPL-3.0-or-later
"""Regressions for the review of 1c2bf7d: grid validation, op caching, hessian."""
import numpy as np
import pytest

import bspf_cpp as bc


def test_large_coordinate_offsets_are_uniform():
    x = np.linspace(1e6, 1e6 + 1, 1001)
    line = bc.Line(x)
    assert line.spacing == pytest.approx(1e-3, rel=1e-12)
    t = (x - x[0])
    np.testing.assert_allclose(line.derivative(1)(np.sin(3 * t)), 3 * np.cos(3 * t), atol=1e-6)


def test_nonuniform_grid_still_rejected():
    x = np.linspace(0, 1, 50)
    x[10] += 1e-6
    with pytest.raises(ValueError, match="uniformly spaced"):
        bc.Line(x)


def test_small_grid_error_names_the_cause():
    with pytest.raises(ValueError, match="too small for degree 5"):
        bc.Line(np.linspace(0, 1, 5), degree=5)
    with pytest.raises(ValueError, match="too small for 8 Chebyshev modes"):
        bc.Line(np.linspace(0, 1, 6), degree=3, endpoint_method="chebyshev", chebyshev_modes=8)


def test_cached_interpolation_and_integration_are_consistent():
    axes = [np.linspace(0, 1, 30), np.linspace(-1, 1, 25)]
    grid = bc.Grid.uniform(axes)
    mesh = np.meshgrid(*axes, indexing="ij")
    f = np.sin(2 * mesh[0]) * np.cos(mesh[1])
    rng = np.random.default_rng(0)
    coords = [rng.uniform(0, 1, 7), rng.uniform(-1, 1, 5)]
    first = grid.interpolate(f, coords)
    np.testing.assert_array_equal(grid.interpolate(f, coords), first)
    # Different points and derivative orders must not hit a stale entry.
    other = [rng.uniform(0, 1, 7), rng.uniform(-1, 1, 5)]
    expected = [grid.line(i).evaluate(c) for i, c in enumerate(other)]
    np.testing.assert_allclose(grid.interpolate(f, other), expected[1](expected[0](f, axis=0), axis=1), rtol=1e-14)
    d = grid.interpolate(f, coords, [1, 0])
    assert not np.allclose(d, first)
    np.testing.assert_array_equal(grid.integrate(f), grid.integrate(f))
    sub = grid.integrate(f, [(0.1, 0.9), (-0.5, 0.5)])
    assert sub != grid.integrate(f)
    # Many distinct probe sets exceed the bounded cache without errors.
    for _ in range(100):
        grid.interpolate(f, [rng.uniform(0, 1, 3), rng.uniform(-1, 1, 3)])


def test_hessian_is_symmetric_and_matches_partials():
    axes = [np.linspace(0, 1, 20), np.linspace(0, 2, 22), np.linspace(-1, 1, 18)]
    grid = bc.Grid.uniform(axes)
    mesh = np.meshgrid(*axes, indexing="ij")
    f = np.sin(mesh[0]) * np.exp(mesh[1] / 2) * np.cos(mesh[2])
    h = grid.hessian(f)
    assert h.shape == (3, 3) + f.shape
    np.testing.assert_array_equal(h[0, 1], h[1, 0])
    np.testing.assert_array_equal(h[1, 2], grid.partial(f, (0, 1, 1)))
    np.testing.assert_array_equal(h[2, 2], grid.partial(f, (0, 0, 2)))
