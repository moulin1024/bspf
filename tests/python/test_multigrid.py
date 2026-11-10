# SPDX-License-Identifier: GPL-3.0-or-later
"""FAS multigrid (bspf.multigrid_solve) on the Laval-nozzle solvers."""
import sys
from pathlib import Path

import numpy as np
import pytest

import bspf_cpp as bspf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "pde"))
nozzle = pytest.importorskip("laval_nozzle_2d")


def test_levels_must_be_nested():
    nz = nozzle.Nozzle(4.0, shape="sextic")
    fine, coarse = nozzle.make_solver(nz, 97, 33), nozzle.make_solver(nz, 65, 17)
    with pytest.raises(ValueError):
        bspf.multigrid_solve([fine, coarse], nozzle.initial_state(nz, fine, False), max_cycles=1)


def test_cpu_multigrid_reaches_the_single_grid_steady_state():
    nz = nozzle.Nozzle(4.0, shape="sextic")
    fine, coarse = nozzle.make_solver(nz, 97, 33), nozzle.make_solver(nz, 49, 17)
    U0 = nozzle.initial_state(nz, fine, False)
    U, info = bspf.multigrid_solve([fine, coarse], U0, tolerance=1e-6, max_cycles=60)
    assert info["converged"] and info["damping_cuts"] == 0
    ref, _ = fine.solve(U0, tolerance=1e-10)
    assert np.abs(U - ref).max() < 1e-4 * np.abs(ref).max()
    # the forcing is cleared afterwards: the fine solver's residual of its own steady state is small again
    assert np.sqrt(np.mean(fine.rhs(ref) ** 2)) < 1e-9


@pytest.mark.skipif(bspf.cuda is None or not bspf.cuda.available(), reason="CUDA backend not available")
def test_device_multigrid_converges():
    nz = nozzle.Nozzle(4.0, shape="sextic")
    cpus = [nozzle.make_solver(nz, 97, 33), nozzle.make_solver(nz, 49, 17)]
    gpus = [bspf.cuda.CompressibleSolver2D(c, refresh=10) for c in cpus]
    U, info = bspf.cuda.multigrid_solve(gpus, nozzle.initial_state(nz, cpus[0], False), tolerance=1e-4, max_cycles=40)
    assert info["converged"] and info["cycles"] < 25
    assert abs(cpus[0].mass_flow(U).mean() / nozzle.critical_mass_flow() - 0.99924113) < 2e-4


def test_hierarchy_mixes_h_and_p_coarsening():
    plan = nozzle.hierarchy((193, 65), implicit=False)
    assert plan[0] == ((193, 65), "high")
    assert any(a[0] == b[0] and a[1] != b[1] for a, b in zip(plan, plan[1:]))  # a p-level
    assert all(nozzle.order_valid(s, o, False) for s, o in plan)


@pytest.mark.skipif(bspf.cuda is None or not bspf.cuda.available(), reason="CUDA backend not available")
def test_device_deep_multigrid_with_explicit_smoothing():
    nz = nozzle.Nozzle(4.0, shape="sextic")
    solver, U, log = nozzle.run_multigrid(nz, (97, 33), 0.0, 1.0, 0, device="cuda", tolerance=1e-4, verbose=False,
                                          cycle=dict(max_cycles=60), line_implicit="none")
    assert log[0]["converged"] and log[0]["cycles"] < 30 and len(log[0]["levels"]) >= 5
    assert abs(solver.mass_flow(U).mean() / nozzle.critical_mass_flow() - 0.99924113) < 3e-4


def test_newton_krylov_converges_to_rounding():
    nz = nozzle.Nozzle(4.0, shape="sextic")
    solver, U, log = nozzle.run_multigrid(nz, (49, 17), 0.0, 1.0, 0, tolerance=1e-10, verbose=False,
                                          cycle=dict(newton=True), line_implicit="none")
    info = log[0]
    assert info["converged"] and info["newton_steps"] <= 12
    assert abs(solver.mass_flow(U).mean() / nozzle.critical_mass_flow() - 0.99924113) < 1e-6


@pytest.mark.skipif(bspf.cuda is None or not bspf.cuda.available(), reason="CUDA backend not available")
def test_device_fp64_matches_cpu_and_mixed_precision_newton_converges():
    nz = nozzle.Nozzle(4.0, shape="sextic")
    cpu = nozzle.make_solver(nz, 49, 17)
    U = nozzle.initial_state(nz, cpu, False)
    fp64 = bspf.cuda.CompressibleSolver2D(cpu)  # precision double by default
    R = cpu.rhs(U)
    assert np.abs(fp64.rhs(U) - R).max() < 1e-9 * np.abs(R).max()
    solver, U, log = nozzle.run_multigrid(nz, (49, 17), 0.0, 1.0, 0, device="cuda", tolerance=1e-10, verbose=False,
                                          cycle=dict(newton=True), line_implicit="none")
    assert log[0]["converged"] and log[0]["newton_steps"] <= 12
    assert abs(solver.mass_flow(U).mean() / nozzle.critical_mass_flow() - 0.99924113) < 1e-6
