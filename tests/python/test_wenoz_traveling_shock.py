# SPDX-License-Identifier: GPL-3.0-or-later
import importlib
from pathlib import Path
import numpy as np
import pytest

@pytest.fixture
def weno(monkeypatch):
    pytest.importorskip('numba')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'examples/pde'))
    return importlib.import_module('wenoz_traveling_shock')

def test_constant_state(weno):
    U=np.tile([1.,.5,2.625],(33,1))
    np.testing.assert_allclose(weno.rhs(U,1/32,.001,.0046667),0,atol=1e-12)

def test_fifth_order_contact_derivative(weno):
    errors=[]
    for n in (65,129,257):
        x=np.linspace(0,1,n);rho=1+.1*np.sin(2*np.pi*x);u=.5
        U=np.column_stack((rho,u*rho,2.5+.5*u*u*rho))
        target=-.1*2*np.pi*np.cos(2*np.pi*x)[:,None]*np.array([u,u*u,.5*u**3])[None,:]
        error=weno.rhs(U,1/(n-1),0.,0.)-target
        errors.append(abs(error[8:-8]).max())
    assert errors[0]/errors[1]>25
    assert errors[1]/errors[2]>25

def test_sixth_order_physical_diffusion(weno):
    errors=[]
    for n in (33,65,129):
        x=np.linspace(0,1,n);u=.5+.1*np.sin(2*np.pi*x)
        ux=.2*np.pi*np.cos(2*np.pi*x);uxx=-.4*np.pi**2*np.sin(2*np.pi*x)
        U=np.column_stack((np.ones(n),u,2.5+.5*u*u));mu=.01
        visc=weno.rhs(U,1/(n-1),mu,.02)-weno.rhs(U,1/(n-1),0.,0.)
        target=np.column_stack((np.zeros(n),4*mu/3*uxx,4*mu/3*(ux*ux+u*uxx)))
        errors.append(abs((visc-target)[8:-8]).max())
    assert errors[0]/errors[1]>40
    assert errors[1]/errors[2]>40

def test_open_domain_flux_balance(weno):
    shock=weno.BeckerShock();n=513;x=np.linspace(0,1,n);U=shock.state(x)
    rate=weno.rhs(U,1/(n-1),shock.mu,shock.kappa)
    r,u,p,_=weno.primitives(U[[0,-1]])
    flux=np.column_stack((r*u,r*u*u+p,u*(U[[0,-1],2]+p)))
    np.testing.assert_allclose(rate.sum(axis=0)/(n-1),flux[0]-flux[1],atol=2e-11)
