# SPDX-License-Identifier: GPL-3.0-or-later
"""Constrained filtering: conservation, wall values and independent checks."""
import importlib
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'examples'/'pde'))
    return importlib.import_module('shock_tube_monotone')


def test_monotone_projection_preserves_integrals(module):
    base=module.BSPFNavierStokes(65,mu=.0001,quadrature=12)
    limiter=module.MonotoneLimiter(base)
    state=module.initial_state(base.x,.01)
    state[:,0]+=.003*np.sin(16*np.pi*base.x)
    before=base.integral@state
    projected=limiter(state)
    np.testing.assert_allclose(base.integral@projected,before,atol=1e-10)
    np.testing.assert_array_equal(projected[[0,-1],1],0)
    audit=module.diagnostics(base,limiter,projected,np.linspace(0,1,16385))
    assert audit['density_positive_variation']<1e-7
    assert audit['pressure_positive_variation']<1e-7
    assert audit['min_density']>.125-1e-7
    assert audit['min_pressure']>.1-1e-7


def test_filter_preserves_feasible_linear_equilibrium(module):
    base=module.BSPFNavierStokes(33,mu=.0001,quadrature=12)
    limiter=module.MonotoneLimiter(base)
    state=np.column_stack((.9-.6*base.x,np.zeros(base.n),(1-.8*base.x)/.4))
    np.testing.assert_allclose(limiter(state),state,atol=1e-8)


def test_short_limited_evolution(module):
    result=module.solve(n=33,final_time=.003,width=.02)
    s=result['summary']
    assert s['max_relative_mass_drift']<1e-10
    assert s['max_relative_energy_drift']<1e-10
    assert s['density_positive_variation']<1e-7
    assert s['pressure_positive_variation']<1e-7
    assert s['min_density']>.125-1e-7
    assert s['min_pressure']>.1-1e-7


def test_artificial_diffusion_is_conservative_and_dissipative(module):
    regularized=importlib.import_module('shock_tube_regularized')
    solver=regularized.RegularizedBSPF(n=33,epsilon=.002)
    state=module.initial_state(solver.x,.05)
    state[:,1]=.03*np.sin(np.pi*solver.x)
    state[[0,-1],1]=0
    delta=solver.rhs(state)-module.BSPFNavierStokes.rhs(solver,state)
    np.testing.assert_allclose((solver.integral@delta)[[0,2]],0,atol=1e-10)
    np.testing.assert_array_equal(delta[[0,-1],1],0)
    work=np.sum(state*solver.galerkin.mass(delta))
    assert work<0


def test_short_combined_regularization(module):
    result=module.solve(n=33,final_time=.002,width=.02,temperature_shape=True,artificial_diffusion=.002)
    assert result['summary']['max_relative_mass_drift']<1e-10
    assert result['summary']['max_relative_energy_drift']<1e-10
    assert result['summary']['density_positive_variation']<1e-7
