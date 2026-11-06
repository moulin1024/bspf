# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent checks of FFT differentiation and the shared geometry."""
import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def comparison(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples" / "basic"))
    return importlib.import_module("compare_cylinder_chebyshev")


def test_fft_polynomials_including_highest_mode(comparison):
    n = 33
    s = (1-np.cos(np.pi*np.arange(n)/(n-1)))/2
    for mode in (0, 1, 2, 13, n-1):
        poly = np.polynomial.Chebyshev.basis(mode, domain=[0, 1])
        u = poly(s)[:, None]*np.array([[1., 2., -3.]])
        expected = poly.deriv()(s)[:, None]*np.array([[1., 2., -3.]])
        np.testing.assert_allclose(comparison.cheb_diff(u, 0), expected, atol=2e-9, rtol=1e-11)
        np.testing.assert_allclose(comparison.cheb_diff(u.T, 1), expected.T, atol=2e-9, rtol=1e-11)


def test_same_geometry_and_cartesian_chain_rule(comparison):
    n = 33
    mp, _ = comparison.cylinder_rectangle(n)
    grids = comparison.geometry_at(np.linspace(0, 1, n))
    for k, g in enumerate(grids):
        np.testing.assert_allclose(g["x"], mp.patch(k).x, atol=2e-14)
        np.testing.assert_allclose(g["y"], mp.patch(k).y, atol=2e-14)
    grids = comparison.geometry(n)
    fields = [g["x"]**2 + g["x"]*g["y"] + 2*g["y"]**2 for g in grids]
    result = comparison.cheb_operators(grids, fields)
    for k, g in enumerate(grids):
        for name, reference in dict(f_x=2*g["x"]+g["y"], f_y=g["x"]+4*g["y"],
                                    f_xx=2., f_yy=4., f_xy=1., lap=6.).items():
            np.testing.assert_allclose(result[name][k], reference, atol=2e-7, rtol=0)


def test_chebyshev_layer_convergence(comparison):
    errors = []
    for n in (17, 33, 65):
        grids = comparison.geometry(n)
        exact = [comparison.manufactured(g["x"], g["y"]) for g in grids]
        fields = [e["u"] for e in exact]
        result = comparison.cheb_operators(grids, fields)
        lap = comparison.cheb_operators(grids, fields, lap_only=True)["lap"]
        for a, b in zip(lap, result["lap"]):
            np.testing.assert_array_equal(a, b)
        errors.append(comparison.errors(result, exact))
    for name in errors[0]:
        assert errors[-1][name] < errors[0][name]/1000
        assert errors[-1][name] < 2e-8


def test_shock_ring_refinement(comparison):
    for method in ("cheb", "bspf"):
        results = []
        for n in (65, 257):
            if method == "cheb":
                grids = comparison.geometry(n)
                coordinates = [(g["x"], g["y"]) for g in grids]
            else:
                mp, _ = comparison.cylinder_rectangle(n)
                coordinates = [(mp.patch(k).x, mp.patch(k).y) for k in range(4)]
            exact = [comparison.manufactured(x, y, shock_amplitude=10.) for x, y in coordinates]
            fields = [e["u"] for e in exact]
            values = (comparison.cheb_operators(grids, fields, lap_only=True) if method == "cheb"
                      else {"lap": mp.laplacian(fields)})
            results.append(comparison.errors(values, exact)["lap"])
        assert results[-1] < .21
        assert results[-1] < results[0]/10
