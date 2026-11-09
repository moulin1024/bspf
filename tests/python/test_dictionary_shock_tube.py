"""Physical-rate, wall, conservation and dictionary shock-tube checks."""
import importlib
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'examples/pde'))
    return importlib.import_module('dictionary_shock_tube')


def test_strong_rate_is_flux_derivative(module):
    x=np.linspace(.1,.9,101)
    def fields(x):
        U=np.column_stack((1+.2*x+.1*x*x,.3+.1*x*x,3+.4*x+.2*x*x))
        G=np.column_stack((.2+.2*x,.2*x,.4+.4*x))
        H=np.tile([.2,.2,.4],(len(x),1))
        return U,G,H
    mu=.003;kappa=.014;gamma=1.4;h=1e-5
    def flux(x):
        U,G,_=fields(x)
        return module.physical_flux(U,G,mu,kappa,gamma)
    derivative=(flux(x-2*h)-8*flux(x-h)+8*flux(x+h)-flux(x+2*h))/(12*h)
    rate=module.strong_rate(*fields(x),mu,kappa,gamma)
    np.testing.assert_allclose(rate,-derivative,rtol=2e-9,atol=2e-10)


def test_block_derivatives_and_wall_values(module):
    block=module.SplineBlock(.5,1/64)
    rng=np.random.default_rng(1);coef=rng.normal(size=(block.k,3))
    x=np.linspace(.25,.75,601);h=1e-7
    for derivative in (0,1,2):
        np.testing.assert_allclose(block.value(x,coef,derivative),block.raw(x,derivative)@coef,atol=2e-8,rtol=1e-11)
    for derivative in (1,2):
        fd=(block.value(x+h,coef,derivative-1)-block.value(x-h,coef,derivative-1))/(2*h)
        np.testing.assert_allclose(block.value(x,coef,derivative),fd,atol=.02 if derivative==2 else 2e-6,rtol=2e-6)
    np.testing.assert_allclose(block.raw(np.array([0.,1.])),0.,atol=1e-15)


def test_wall_constraints_equilibrium_and_integral_rate(module):
    base=module.BSPFNavierStokes(25,quadrature=8,line_options=dict(n_basis=10))
    space=module.TubeSpace(base,[(.4,1/64),(.6,1/128)])
    constant=np.tile([1.,0.,2.5],(len(base.x),1))
    state=np.vstack((constant,np.zeros((space.k,3))))
    rate,ok=module.tube_rhs(state,space.both,space.rate,space.wall_lift,space.n,.001,.004,1.4)
    assert ok
    np.testing.assert_allclose(rate,0.,atol=2e-8)
    x=space.base.q
    values=np.column_stack((1+.1*x,.05*np.sin(np.pi*x),3+.2*x))
    state=space.fit_fields(values)
    rate,ok=module.tube_rhs(state,space.both,space.rate,space.wall_lift,space.n,.001,.004,1.4)
    assert ok
    np.testing.assert_allclose(rate[[0,space.n-1],1],0.,atol=1e-13)
    assert np.max(abs(rate[[0,space.n-1],0]))>.01  # wall density must evolve
    np.testing.assert_allclose((space.integral@rate)[[0,2]],0.,atol=2e-10)
    np.testing.assert_allclose(space.at(space.base.q,state),space.evaluate(state),atol=2e-11)


def test_short_tube_without_detector_or_reference(module,monkeypatch):
    detector=importlib.import_module('shock_tube_enrichment')
    def forbidden(*args,**kwargs):raise AssertionError('Front detector called')
    monkeypatch.setattr(detector,'detect',forbidden)
    result=module.solve(n=33,final_time=.002,interval=.001,tolerance=1e-5,
                        probe_cells=128,spacing=1/128,output_points=1025)
    s=result['summary']
    assert s['steps']>0 and s['rejected_stages']==0
    assert s['min_density']>.12 and s['min_pressure']>.09
    assert abs(s['relative_mass_drift'])<2e-12
    assert abs(s['relative_energy_drift'])<2e-12
    np.testing.assert_allclose(result['output'][[0,-1],1],0.,atol=2e-12)
