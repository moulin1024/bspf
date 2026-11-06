# SPDX-License-Identifier: GPL-3.0-or-later
import importlib
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'examples'/'pde'))
    return importlib.import_module('shock_tube_spectral_viscosity')


def test_high_mode_damping_conserves_and_dissipates(module):
    s=module.SpectralViscosityBSPF(33,cutoff=.4,rate=100.)
    rng=np.random.default_rng(11)
    U=rng.normal(size=(33,3));U[[0,-1],1]=0
    rate=s.damping(U)
    np.testing.assert_allclose((s.integral@rate)[[0,2]],0,atol=1e-10)
    np.testing.assert_array_equal(rate[[0,-1],1],0.)
    assert np.sum(U*s.galerkin.mass(rate))<0
    uniform=np.tile([1.,0.,2.5],(33,1))
    np.testing.assert_allclose(s.damping(uniform),0,atol=1e-9)


def test_compression_viscosity_conserves_and_dissipates(module):
    s=module.CompressionViscosityBSPF(33,strength=.1,rate=0.)
    U=np.column_stack((1+.1*np.cos(2*np.pi*s.x),.05*np.sin(2*np.pi*s.x),np.full(s.n,2.5)))
    U[[0,-1],1]=0
    delta=s.rhs(U)-module.SpectralViscosityBSPF.rhs(s,U)
    np.testing.assert_allclose((s.integral@delta)[[0,2]],0,atol=1e-10)
    np.testing.assert_array_equal(delta[[0,-1],1],0.)
    assert np.sum(U*s.galerkin.mass(delta))<0


def test_short_evolution(module):
    s=module.CompressionViscosityBSPF(33,strength=.1,rate=50.)
    result=s.solve(.01,.02,.35)
    assert result['summary']['max_relative_mass_drift']<1e-10
    assert result['summary']['max_relative_energy_drift']<1e-10
    assert result['summary']['rejected_stages']==0
