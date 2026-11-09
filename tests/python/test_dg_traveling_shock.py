# SPDX-License-Identifier: GPL-3.0-or-later
import importlib
from pathlib import Path
import numpy as np
import pytest

@pytest.fixture
def dg(monkeypatch):
    pytest.importorskip('numba')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'examples/pde'))
    return importlib.import_module('dg_traveling_shock')

def project(values,w,V,p):
    return np.einsum('qj,eqk->ejk',w[:,None]*V,values)*(2*np.arange(p+1)+1)[None,:,None]/2

@pytest.mark.parametrize('p',[3,5,7])
def test_constant_and_mass(dg,p):
    z,w,V,DW,E,ra,rd=dg.operators(p)
    np.testing.assert_allclose(V.T@(w[:,None]*V),np.diag(2/(2*np.arange(p+1)+1)),atol=1e-14)
    U=np.zeros((8,p+1,3));U[:,0]=[1.,.5,2.625];bc=np.array([1.,.5,1.])
    rate,_,_=dg.rhs(U,1/8,.001,.0046667,V,DW,E,bc,bc)
    np.testing.assert_allclose(rate,0,atol=3e-11)
    assert ra>0 and rd>0

def test_polynomial_volume_and_viscosity(dg):
    p=3;ne=8;h=1/ne;z,w,V,DW,E,_,_=dg.operators(p)
    x=(np.arange(ne)[:,None]+(z+1)/2)/ne;u=.5+.1*x
    values=np.stack((np.ones_like(x),u,2.5+.5*u*u),axis=-1)
    U=project(values,w,V,p);mu=.01
    rate,_,_=dg.rhs(U,h,mu,.02,V,DW,E,np.array([1.,.5,1.]),np.array([1.,.6,1.]))
    exact=np.stack((np.full_like(x,-.1),-.2*u,-.1*(3.5+1.5*u*u)+4*mu/3*.01),axis=-1)
    np.testing.assert_allclose(rate[1:-1],project(exact,w,V,p)[1:-1],atol=2e-11)

def test_flux_balance_and_refinement(dg):
    shock=dg.BeckerShock();p=3;ne=32;h=1/ne;z,w,V,DW,E,_,_=dg.operators(p)
    x=(np.arange(ne)[:,None]+(z+1)/2)/ne
    U=project(shock.state(x.ravel()).reshape(ne,len(z),3),w,V,p)
    left=np.array([1.,shock.v0+.5,1.]);right=np.array([2.,shock.v1+.5,2.75])
    rate,_,_=dg.rhs(U,h,shock.mu,shock.kappa,V,DW,E,left,right)
    np.testing.assert_allclose(h*rate[:,0].sum(axis=0),dg.hllc(left,left)-dg.hllc(right,right),atol=1e-10)
    a,_=dg.solve(64,3,final_time=.001);b,_=dg.solve(128,3,final_time=.001)
    assert b['metrics']['rho']['l2']<a['metrics']['rho']['l2']/2
