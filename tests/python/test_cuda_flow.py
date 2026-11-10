# SPDX-License-Identifier: GPL-3.0-or-later
"""FP32 device Galerkin operators and compressible solver against the CPU (double) versions."""
import sys
from pathlib import Path

import numpy as np
import pytest

import bspf_cpp as bspf

if bspf.cuda is None or not bspf.cuda.available():
    pytest.skip("CUDA backend not available", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "pde"))
nozzle = pytest.importorskip("laval_nozzle_2d")

CHEBYSHEV = dict(degree=9, lam=0.0, endpoint_method="chebyshev", boundary_points=16, chebyshev_modes=12,
                 chebyshev_alpha=1e-12, constraint_order=4)


@pytest.mark.parametrize("n", [49, 97, 193])
def test_galerkin_axis_matches_cpu(n):
    line = bspf.Line(np.linspace(0, 1, n), n_basis=max(24, n // 3), **CHEBYSHEV)
    g = bspf.GalerkinLine(line, quadrature_order=5)
    device = bspf.cuda.GalerkinAxis(g)
    assert device.points == len(g.quadrature_points)
    x = line.x
    U = np.column_stack((np.sin(3 * x + 1), np.exp(x), np.cos(7 * x)))
    V = 1 + 0.1 * np.random.default_rng(0).standard_normal((device.points, 3))
    for k in (0, 1):
        ref = g.evaluate(U, k)
        assert np.abs(device.evaluate(U, k) - ref).max() < 1e-4 * np.abs(ref).max()
        ref = g.project(V, k)
        assert np.abs(device.project(V, k) - ref).max() < 1e-4 * np.abs(ref).max()


def test_device_steady_state_matches_cpu():
    nz = nozzle.Nozzle(4.0, shape="sextic")
    cpu = nozzle.make_solver(nz, 49, 17)
    U0 = nozzle.initial_state(nz, cpu, False)
    gpu = bspf.cuda.CompressibleSolver2D(nozzle.make_solver(nz, 49, 17, line_implicit="none"))
    Ug, info = gpu.solve(U0, tolerance=1e-5, anderson=0, max_steps=8000)
    Uc, _ = cpu.solve(U0, tolerance=1e-11, anderson=8)
    assert info["residual"] < 1e-3
    assert np.abs(Ug - Uc).max() < 1e-3 * np.abs(Uc).max()


def test_device_line_implicit_converges_faster_to_the_cpu_state():
    nz = nozzle.Nozzle(4.0, shape="sextic")
    cpu = nozzle.make_solver(nz, 49, 17)
    U0 = nozzle.initial_state(nz, cpu, False)
    Uc, _ = cpu.solve(U0, tolerance=1e-11, anderson=8)
    steps = {}
    for mode in ("none", "full"):
        gpu = bspf.cuda.CompressibleSolver2D(nozzle.make_solver(nz, 49, 17, line_implicit=mode), refresh=10)
        Ug, info = gpu.solve(U0, tolerance=1e-4, anderson=0, max_steps=8000, check=25)
        assert info["converged"]
        assert np.abs(Ug - Uc).max() < 1e-3 * np.abs(Uc).max()
        steps[mode] = info["steps"]
    assert steps["full"] < steps["none"]
