# SPDX-License-Identifier: GPL-3.0-or-later
import importlib
from pathlib import Path
import numpy as np
import pytest

@pytest.fixture
def muscl(monkeypatch):
    pytest.importorskip('numba')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'examples/pde'))
    return importlib.import_module('muscl_traveling_shock')

def test_flux_consistency_and_contact(muscl):
    for u in (-3.,0.,2.):
        w=np.array([1.3,u,.8]);U=muscl.cons(w)
        exact=np.array([U[1],U[1]*u+.8,u*(U[2]+.8)])
        np.testing.assert_allclose(muscl.hllc(w,w),exact,atol=1e-14)
    np.testing.assert_allclose(muscl.hllc(np.array([1.,0.,1.]),np.array([2.,0.,1.])),[0,1,0],atol=1e-14)

def test_uniform_state_and_flux_balance(muscl):
    state=muscl.cons(np.array([1.2,1.4,.9]));U=np.tile(state,(32,1))
    np.testing.assert_allclose(muscl.rhs(U,1/32,.001,.0046667,state,state),0,atol=1e-13)
    shock=muscl.BeckerShock();U=muscl.averages(shock,512,0)
    left=shock.state(np.array([0.]))[0];right=shock.state(np.array([1.]))[0]
    rate=muscl.rhs(U,1/512,shock.mu,shock.kappa,left,right)
    def flux(u):
        w=muscl.primitive_array(u[None,:])[0]
        return muscl.hllc(w,w)
    np.testing.assert_allclose(rate.sum(axis=0)/512,flux(left)-flux(right),atol=2e-13)

def test_averages_and_refinement(muscl):
    shock=muscl.BeckerShock()
    np.testing.assert_allclose(muscl.averages(shock,256,0,16),muscl.averages(shock,256,0,32),atol=2e-12)
    coarse,_=muscl.solve(512,final_time=.01)
    fine,_=muscl.solve(1024,final_time=.01)
    assert fine['metrics']['rho']['l2'] < coarse['metrics']['rho']['l2']/2
    assert fine['cell_average_linf'][0] < coarse['cell_average_linf'][0]/2
