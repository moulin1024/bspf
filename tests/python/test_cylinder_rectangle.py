# SPDX-License-Identifier: GPL-3.0-or-later
"""MMS regression for the rectangle with a cylindrical hole."""
from pathlib import Path
import importlib

import numpy as np
import pytest


def test_cylinder_rectangle_geometry_and_mms(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples" / "basic"))
    example = importlib.import_module("diff_cylinder_rectangle_2d")
    results = []
    for n in (17, 129):
        mp, circle_error = example.cylinder_rectangle(n)
        assert circle_error < 1e-12
        assert len(mp.boundary_faces) == 8
        assert mp.node_count == 4*n*(n-1)
        for k in range(4):
            g = mp.patch(k)
            assert np.all(g.jacobian > 0)
            np.testing.assert_allclose(np.hypot(g.x[0], g.y[0]), 0.5, atol=1e-12, rtol=0)
            np.testing.assert_allclose(example.manufactured(g.x[0], g.y[0])["u"], 0, atol=5e-14)
            wall = g.x[-1] if k % 2 == 0 else g.y[-1]
            np.testing.assert_allclose(wall, (2, 1.5, -2, -1.5)[k], atol=1e-12, rtol=0)
        errors, _ = example.measure(mp)
        results.append(errors)
    for name in results[0]:
        coarse, fine = (r[name]["relative_max"] for r in results)
        assert fine < coarse/1000, name
        assert fine < (3e-8 if name in ("f_x", "f_y") else 1e-6), name


def test_boundary_layer_exact_derivatives(monkeypatch):
    """Check product/chain rules against independent high-precision differentiation."""
    mpmath = pytest.importorskip("mpmath")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples" / "basic"))
    example = importlib.import_module("diff_cylinder_rectangle_2d")
    with mpmath.workdps(40):
        for thickness in (0.05, 0.1, 0.2):
            def field(x, y):
                r = mpmath.sqrt(x*x+y*y)
                return (-mpmath.expm1(-(r-0.5)/thickness)
                        * (mpmath.sin(2*x+0.5)*mpmath.cos(3*y-0.2) + mpmath.exp(x*y)/2))
            for x, y in ((0.3, 0.4), (-0.55, 0.2), (1.2, -0.8)):
                exact = example.manufactured(np.array(x), np.array(y), thickness)
                for orders, name in example.NAMES.items():
                    reference = float(mpmath.diff(field, (mpmath.mpf(x), mpmath.mpf(y)), orders))
                    np.testing.assert_allclose(exact[name], reference, rtol=2e-13, atol=2e-12)
                np.testing.assert_allclose(exact["lap"], exact["f_xx"]+exact["f_yy"], rtol=2e-14, atol=2e-12)
    for thickness in (0, -0.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="thickness"):
            example.manufactured(np.array(0.5), np.array(0.), thickness)


def test_shock_ring_exact_derivatives(monkeypatch):
    mpmath = pytest.importorskip("mpmath")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples" / "basic"))
    example = importlib.import_module("diff_cylinder_rectangle_2d")
    with mpmath.workdps(50):
        for width in (.02, .05):
            def field(x, y):
                r = mpmath.sqrt(x*x+y*y)
                return (-mpmath.expm1(-(r-.5)/.1)
                        * (mpmath.sin(2*x+.5)*mpmath.cos(3*y-.2) + mpmath.exp(x*y)/2
                           + 5*(1-mpmath.tanh((r-1)/width))))
            for radius in (.5, .98, 1., 1.02, 1.4):
                x, y = radius*.6, radius*.8
                exact = example.manufactured(np.array(x), np.array(y), shock_amplitude=10., shock_width=width)
                for orders, name in example.NAMES.items():
                    reference = float(mpmath.diff(field, (mpmath.mpf(x), mpmath.mpf(y)), orders))
                    np.testing.assert_allclose(exact[name], reference, rtol=5e-12, atol=3e-9)
                reference_lap = float(mpmath.diff(field, (mpmath.mpf(x), mpmath.mpf(y)), (2, 0))
                                      + mpmath.diff(field, (mpmath.mpf(x), mpmath.mpf(y)), (0, 2)))
                np.testing.assert_allclose(exact["lap"], reference_lap, rtol=5e-12, atol=3e-9)
                if radius == .5:
                    assert abs(exact["u"]) < 1e-12
    for keyword, value in (("shock_amplitude", -1), ("shock_radius", .4),
                           ("shock_radius", 1.6), ("shock_width", 0), ("shock_width", np.nan)):
        with pytest.raises(ValueError, match="shock"):
            example.manufactured(np.array(1.), np.array(0.), **{keyword: value})
