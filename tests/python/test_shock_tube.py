# SPDX-License-Identifier: GPL-3.0-or-later
"""Physical fluxes, weak-form consistency and closed-tube time integration."""
import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def shock(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/"examples"/"pde"))
    return importlib.import_module("shock_tube_1d")


def test_physical_viscous_flux(shock):
    rho, rho_x, u, ux, temperature, tx = 2., .4, .3, -.2, 1.1, .6
    mu, kappa, cv = .01, .048, 2.5
    energy = rho*(cv*temperature+.5*u*u)
    ex = rho_x*(cv*temperature+.5*u*u)+rho*(cv*tx+u*ux)
    state = np.array([[rho, rho*u, energy]])
    derivative = np.array([[rho_x, rho_x*u+rho*ux, ex]])
    convective, viscous = shock.fluxes(state, derivative, mu, kappa)
    tau = 4*mu*ux/3
    np.testing.assert_allclose(viscous[0], [0., tau, u*tau+kappa*tx], atol=1e-15)
    pressure = rho*temperature
    np.testing.assert_allclose(convective[0], [rho*u, rho*u*u+pressure, u*(energy+pressure)], atol=1e-15)


def test_uniform_equilibrium_and_conservative_rates(shock):
    solver = shock.BSPFNavierStokes(33)
    state = np.tile([1., 0., 2.5], (33, 1))
    np.testing.assert_allclose(solver.rhs(state), 0, atol=2e-8)
    state = shock.initial_state(solver.x, width=.05)
    state[:, 1] = .05*np.sin(np.pi*solver.x)
    state[[0, -1], 1] = 0.
    rate = solver.rhs(state)
    totals = solver.integral@rate
    np.testing.assert_allclose(totals[[0, 2]], 0, atol=1e-10)
    np.testing.assert_array_equal(rate[[0, -1], 1], 0.)


def test_weak_rhs_against_smooth_exact_navier_stokes(shock):
    errors = []
    for n in (33, 65):
        solver = shock.BSPFNavierStokes(n)
        def smooth(x):
            rho = 1+.1*np.cos(2*np.pi*x)
            velocity = .1*np.sin(np.pi*x)
            temperature = 1+.05*np.cos(2*np.pi*x)
            return np.column_stack((rho, rho*velocity, rho*(2.5*temperature+.5*velocity**2)))
        x = solver.q
        rho = 1+.1*np.cos(2*np.pi*x)
        rx = -.2*np.pi*np.sin(2*np.pi*x)
        u = .1*np.sin(np.pi*x)
        ux = .1*np.pi*np.cos(np.pi*x)
        uxx = -.1*np.pi**2*np.sin(np.pi*x)
        temperature = 1+.05*np.cos(2*np.pi*x)
        tx = -.1*np.pi*np.sin(2*np.pi*x)
        txx = -.2*np.pi**2*np.cos(2*np.pi*x)
        pressure, px = rho*temperature, rx*temperature+rho*tx
        energy = rho*(2.5*temperature+.5*u*u)
        ex = rx*(2.5*temperature+.5*u*u)+rho*(2.5*tx+u*ux)
        exact = np.column_stack((-(rx*u+rho*ux),
            -(rx*u*u+2*rho*u*ux+px)+4*solver.mu*uxx/3,
            -(ux*(energy+pressure)+u*(ex+px))+4*solver.mu*(ux*ux+u*uxx)/3+solver.kappa*txx))
        load = solver.project(exact)
        projected = np.zeros((n, 3))
        projected[:, (0, 2)] = solver.solve_mass(load[:, (0, 2)])
        projected[1:-1, 1] = solver.solve_mass(load[1:-1, 1:2], interior=True)[:, 0]
        state = smooth(solver.x)
        state[[0, -1], 1] = 0.
        error = solver.evaluate(solver.rhs(state)-projected)
        errors.append(float(np.sqrt(np.sum(solver.weights[:, None]*error**2))))
    assert errors[-1] < 1e-5
    assert errors[-1] < errors[0]/4


@pytest.mark.parametrize("backend", ["fft", "dense"])
def test_short_viscous_shock_tube(shock, backend):
    solver = shock.BSPFNavierStokes(65, backend=backend)
    result = solver.solve(.02)
    summary = result["summary"]
    assert summary["rejected_stages"] == 0
    assert summary["max_relative_mass_drift"] < 1e-10
    assert summary["max_relative_energy_drift"] < 1e-10
    assert summary["entropy_change"] > 0
    assert summary["min_recorded_entropy_increment"] >= -1e-10
    rho, _, pressure, _ = shock.primitives(result["output"])
    assert rho.min() > .1 and pressure.min() > .07
    np.testing.assert_array_equal(result["state"][[0, -1], 1], 0.)


def test_physical_entropy_production(shock):
    """Viscous work and conduction give the thermodynamic entropy production."""
    solver = shock.BSPFNavierStokes(65)
    x = solver.x
    rho = 1+.1*np.cos(2*np.pi*x)
    u = .1*np.sin(np.pi*x)
    temperature = 1+.05*np.cos(2*np.pi*x)
    state = np.column_stack((rho, rho*u, rho*(2.5*temperature+.5*u*u)))
    state[[0, -1], 1] = 0.
    sampled, gradient, rate = solver.evaluate(state), solver.evaluate(state, 1), solver.evaluate(solver.rhs(state))
    rho, u, pressure, temperature = shock.primitives(sampled)
    entropy_variables = np.column_stack((np.log(pressure/rho**1.4)/.4-3.5+.5*u*u/temperature,
                                        -u/temperature, 1/temperature))
    ux = (gradient[:, 1]-u*gradient[:, 0])/rho
    tx = .4*((gradient[:, 2]-sampled[:, 2]*gradient[:, 0]/rho)/rho-u*ux)
    rate_of_entropy = solver.weights@np.sum(entropy_variables*rate, axis=1)
    production = solver.weights@(4*solver.mu/3*ux**2/temperature+solver.kappa*tx**2/temperature**2)
    assert production > 0
    np.testing.assert_allclose(rate_of_entropy, production, rtol=1e-8, atol=1e-11)


def test_invalid_states_are_rejected_not_clipped(shock):
    for state in (np.array([[-1., 0., 1.]]), np.array([[1., 4., 1.]]), np.array([[1., 0., np.nan]])):
        with pytest.raises(shock.NonphysicalState):
            shock.primitives(state)


def test_fft_dense_rhs_and_evolution(shock):
    fast = shock.BSPFNavierStokes(65, mu=.001)
    dense = shock.BSPFNavierStokes(65, mu=.001, backend="dense")
    # Production setup must not retain dense cardinal or mass matrices.
    for name in ("V", "G", "Gw", "M", "mass_factor", "momentum_factor"):
        assert not hasattr(fast, name)
    np.testing.assert_allclose(fast.diffusion_radius, dense.diffusion_radius, rtol=1e-9)
    state = shock.initial_state(fast.x)
    state[:, 1] = .05*np.sin(np.pi*fast.x)
    state[[0, -1], 1] = 0
    np.testing.assert_allclose(fast.rhs(state), dense.rhs(state), atol=1e-8, rtol=1e-8)
    a, b = fast.solve(.02), dense.solve(.02)
    np.testing.assert_allclose(a["output"], b["output"], atol=2e-10, rtol=2e-10)
    assert a["summary"]["steps"] == b["summary"]["steps"]
