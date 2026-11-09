import importlib
from pathlib import Path
import numpy as np
import pytest

@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'examples/pde'))
    return importlib.import_module('prior_art_traveling_shock')

def test_legendre_polynomial_mass_and_derivative(module):
    b=module.LegendreBase(25,quadrature=8)
    samples=np.column_stack((np.ones(len(b.q)),b.q,b.q**4))
    c=b.solve_mass(b.project(samples))
    np.testing.assert_allclose(b.evaluate(c),samples,atol=1e-12)
    # Endpoint differentiation amplifies the ~3e-13 mass-fit roundoff by O(n^2).
    np.testing.assert_allclose(b.evaluate(c,1),np.column_stack((b.q*0,b.q*0+1,4*b.q**3)),atol=3e-10)
    constant=np.zeros((25,1));constant[[0,-1]]=1
    np.testing.assert_allclose(b.evaluate(constant,1),0,atol=1e-14)
    np.testing.assert_allclose(b.P[:,1:-1].T@(b.weights[:,None]*b.P[:,1:-1]),np.eye(23),atol=2e-12)

def test_legendre_enriched_evolution(module):
    from viscous_traveling_shock import solve
    r=solve(33,True,mu=.01,final_time=.002,quadrature=16,base_factory=module.LegendreBase)
    assert r['summary']['metrics']['rho']['linf']<.01
    np.testing.assert_allclose(r['output'][[0,-1]],r['exact'][[0,-1]],atol=1e-9)

def test_fc_derivative_convergence(module):
    path=Path(__file__).resolve().parents[2]/'benchmarks/traveling_shock/prior_art/fc_data/FCGram_data_d5_C25.mat'
    if not path.exists():pytest.skip('Run bench/fetch_fc_data.py to obtain external matrices')
    errors=[]
    for n in (33,65,129):
        fc=module.FCGram(n,path,0);x=np.linspace(0,1,n)
        errors.append(np.max(abs(fc.derivative(np.exp(x)[:,None])[:,0]-np.exp(x))))
        np.testing.assert_allclose(fc.extend(np.exp(x)[:,None])[:n,0],np.exp(x),atol=0)
    assert errors[-1]<1e-8
    assert errors[-1]<errors[0]/10

def test_fc_traveling_refinement(module):
    path=Path(__file__).resolve().parents[2]/'benchmarks/traveling_shock/prior_art/fc_data/FCGram_data_d5_C25.mat'
    if not path.exists():pytest.skip('External FC matrices unavailable')
    errors=[]
    for n in (257,513):
        r=module.solve_fc(n,path,final_time=.002,output_points=8001)
        errors.append(max(v['linf'] for v in r['summary']['metrics'].values()))
    assert errors[1]<errors[0]/2

def test_fc_constant_boundary_fast_path_matches_general(module):
    path=Path(__file__).resolve().parents[2]/'benchmarks/traveling_shock/prior_art/fc_data/FCGram_data_d5_C25.mat'
    if not path.exists():pytest.skip('External FC matrices unavailable')
    options=dict(final_time=.002,output_points=2001,filter_order=0,fused_flux=True,cfl=1.)
    general=module.solve_fc(513,path,**options)
    fast=module.solve_fc(513,path,constant_boundaries=True,**options)
    np.testing.assert_allclose(fast['output'],general['output'],atol=3e-12,rtol=0)
    assert fast['summary']['steps']==general['summary']['steps']
