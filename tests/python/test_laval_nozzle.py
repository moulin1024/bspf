# SPDX-License-Identifier: GPL-3.0-or-later
"""bspf.CompressibleSolver2D (C++) on the Laval-nozzle block."""
import sys
from pathlib import Path

import numpy as np
import pytest

import bspf_cpp as bspf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "pde"))
nozzle = pytest.importorskip("laval_nozzle_2d")


@pytest.mark.parametrize("n", [33, 65])
def test_galerkin_first_derivative_is_summation_by_parts(n):
    line = bspf.Line(np.linspace(0, 1, n), degree=6)
    g = bspf.GalerkinLine(line)
    eye = np.eye(n)
    M = g.mass(eye)
    C = g.project(g.evaluate(eye, 1), 0)
    B = np.zeros((n, n))
    B[0, 0], B[-1, -1] = -1, 1
    assert np.abs(C + C.T - B).max() < 1e-12
    x = line.x
    assert np.abs(np.linalg.solve(M, C) @ np.sin(3 * x + 1) - 3 * np.cos(3 * x + 1)).max() < 1e-5


def uniform_solver(n, state, **options):
    grid = nozzle.Nozzle(4.0, shape="sextic").grid(*n)
    probe = bspf.CompressibleSolver2D(grid, [bspf.Boundary(bspf.BoundaryKind.slip_wall)] * 4, **options)
    count = [n[1], n[1], n[0], n[0]]
    boundaries = [bspf.Boundary(bspf.BoundaryKind.exterior, nodes=np.tile(state, (count[k], 1)),
                                points=np.tile(state, (len(probe.face_points(k)), 1))) for k in range(4)]
    return bspf.CompressibleSolver2D(grid, boundaries, **options)


@pytest.mark.parametrize("reynolds", [0.0, 100.0])
def test_free_stream_is_an_exact_steady_state(reynolds):
    state = nozzle.conservative(1.0, 0.3, 0.1, 0.6)
    solver = uniform_solver((33, 13), state, reynolds=reynolds)
    U = np.ascontiguousarray(np.broadcast_to(state, (33, 13, 4)))
    assert np.abs(solver.rhs(U)).max() < 1e-10  # rounding times the fit amplification


def test_anderson_reaches_the_plain_steady_state():
    nz = nozzle.Nozzle(4.0, shape="sextic")
    solver = nozzle.make_solver(nz, 49, 17)  # 13 points across is too coarse for the Chebyshev jets
    U0 = nozzle.initial_state(nz, solver, False)
    plain, info_plain = solver.solve(U0, tolerance=1e-11, anderson=0)
    fast, info_fast = solver.solve(U0, tolerance=1e-11, anderson=8)
    assert info_plain["converged"] and info_fast["converged"]
    assert np.abs(fast - plain).max() < 1e-8


def test_prolong_is_spectrally_accurate():
    nz = nozzle.Nozzle(4.0, shape="sextic")
    coarse, fine = nozzle.make_solver(nz, 33, 13), nozzle.make_solver(nz, 49, 17)
    field = lambda g: nozzle.conservative(1 + 0.1 * np.sin(g.x / 3) * np.cos(g.y), 0.5 + 0.1 * g.x / 4,
                                          0.05 * g.y, 0.7 + 0.01 * g.x)
    error = np.abs(fine.prolong(coarse, field(coarse.grid)) - field(fine.grid)).max()
    assert error < 1e-6  # 13 points across the channel


def test_nonphysical_state_raises():
    nz = nozzle.Nozzle(4.0, shape="sextic")
    solver = nozzle.make_solver(nz, 33, 13)
    U = nozzle.initial_state(nz, solver, False)
    U[10, 5, 3] = 0.0  # negative pressure
    with pytest.raises(ValueError):
        solver.rhs(U)


def test_manufactured_navier_stokes_converges():
    mms = pytest.importorskip("laval_nozzle_mms")
    coarse = mms.check(33, 13, 100.0, steps=20000, tolerance=1e-10)
    fine = mms.check(49, 17, 100.0, steps=20000, tolerance=1e-10)
    assert coarse["solution"] < 2e-6
    assert fine["solution"] < coarse["solution"] / 4
