"""Conditioning, nested injection and residual selection without front data."""
import importlib
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'examples/pde'))
    return importlib.import_module('residual_spline_shock')


def test_fixed_dictionary_derivatives_and_walls(module):
    d=module.FixedDictionary(range(4,7),5)
    x=np.linspace(.01,.99,101);h=1e-7
    v,g=d.matrices(x)
    plus,_=d.matrices(x+h);minus,_=d.matrices(x-h)
    np.testing.assert_allclose((plus-minus).toarray()/(2*h),g.toarray(),atol=3e-7,rtol=2e-7)
    wall,_=d.matrices(np.array([0.,1.]))
    np.testing.assert_array_equal(wall.toarray(),0.)


def test_conditioned_basis_and_nested_injection(module):
    s=module.ResidualSpace(25,range(4,7),5,6,8,45,step_level=0)
    rng=np.random.default_rng(5);state=rng.normal(size=(25,3))
    before=s.fine_values(state)
    for label in ((6,23),(6,25),(5,15),(6,29)):
        if s.add(s.dictionary.lookup[(*label,'detail')]):state=np.vstack((state,np.zeros((1,3))))
    np.testing.assert_allclose(s.fine_values(state),before,atol=2e-13)
    raw,_=s.raw_matrices(s.x)
    np.testing.assert_allclose(raw@s.T,s.Q,atol=3e-11)
    np.testing.assert_allclose(s.Q.T@(s.w[:,None]*s.Q),np.eye(s.Q.shape[1]),atol=2e-12)
    s.rebuild()
    assert s.mass_defect<1e-8


def test_weak_scores_equal_direct_outside_space_test(module):
    s=module.ResidualSpace(25,range(4,7),5,6,8,45,step_level=0)
    flux=np.column_stack((np.sin(7*s.x),s.x**3,np.cos(11*s.x)))
    weighted=s.w[:,None]*flux
    raw=s.Sx.T@weighted;active=s.Qx.T@weighted
    score=s.scores(raw,active)
    index=int(np.argmax(score));c=s.cross[:,index]
    h=s.S[:,index].toarray().ravel()-s.Q@c
    hx=s.Sx[:,index].toarray().ravel()-s.Qx@c
    direct=np.max(abs(hx@weighted))/np.sqrt(s.w@(h*h))
    np.testing.assert_allclose(score[index],direct,rtol=1e-10)


def test_constant_state_has_no_enrichment_residual(module):
    s=module.ResidualSpace(25,range(4,7),5,6,8,45,step_level=0)
    constant=np.array([1.,.5,3.]);initial=lambda x:np.tile(constant,(len(x),1))
    state,_=s.initialize(initial,np.tile(constant,(2,1)),1e-9)
    state,info=s.adapt(state,.001,.004,1.4,1e-7)
    assert info['added']==0
    np.testing.assert_allclose(s.fine_values(state),initial(s.x),atol=2e-12)
