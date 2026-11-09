# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact Becker profile, Galilean flux balance, and prescribed-boundary solver."""
import importlib
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def traveling(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'examples/pde'))
    return importlib.import_module('viscous_traveling_shock')


def test_becker_limits_and_flux_invariant(traveling):
    s=traveling.BeckerShock()
    x=np.linspace(s.center-30*s.w,s.center+30*s.w,1001)
    state,gradient=s.state_gradient(x)
    convective,viscous=traveling.fluxes(state,gradient,s.mu,s.kappa,s.gamma)
    conserved_flux=convective-viscous-s.speed*state
    np.testing.assert_allclose(conserved_flux,np.broadcast_to(conserved_flux[0],conserved_flux.shape),atol=2e-13)
    rho,u,p,T=traveling.primitives(s.state(np.array([-10.,10.])))
    np.testing.assert_allclose(rho,[1.,2.],atol=1e-14)
    np.testing.assert_allclose(T,[1.,1.375],atol=1e-14)
    np.testing.assert_allclose(p,[1.,2.75],atol=1e-14)
    np.testing.assert_allclose(s.state(x,.03),s.state(x-s.speed*.03),atol=1e-12)


def test_analytic_derivative_and_weak_pde(traveling):
    s=traveling.BeckerShock()
    x=np.linspace(s.center-5*s.w,s.center+5*s.w,100)
    eps=1e-7
    _,gradient=s.state_gradient(x)
    fd=(s.state(x-2*eps)-8*s.state(x-eps)+8*s.state(x+eps)-s.state(x+2*eps))/(12*eps)
    np.testing.assert_allclose(fd,gradient,rtol=1e-7,atol=2e-7)
    base=traveling.BSPFNavierStokes(65,mu=s.mu,prandtl=.75,quadrature=64)
    values,derivative=s.state_gradient(base.q)
    convective,viscous=traveling.fluxes(values,derivative,s.mu,s.kappa,s.gamma)
    # Interior tests vanish at the boundary, so no boundary flux term remains.
    lhs=base.project(-s.speed*derivative)
    rhs=base.project(convective-viscous,1)
    np.testing.assert_allclose(lhs[1:-1],rhs[1:-1],atol=2e-8,rtol=2e-8)


@pytest.mark.parametrize('enriched',[False,True])
def test_traveling_boundary_constraints(traveling,enriched):
    r=traveling.solve(n=33,enriched=enriched,mu=.01,final_time=.003,quadrature=16)
    np.testing.assert_allclose(r['output'][[0,-1]],r['exact'][[0,-1]],atol=1e-9)
    assert r['summary']['rejected_stages']==0
    assert r['summary']['metrics']['rho']['linf']<.03


@pytest.mark.parametrize('degree', [9,13])
def test_higher_degree_enrichment_and_output_sampling(traveling, degree):
    kind=traveling.spline_space_type(degree)
    assert kind.half_support==(degree+1)//2
    x=np.linspace(-1,2,300)
    centers=np.array([.5]);spacing=.02
    values,gradient=kind.sparse_samples(x,centers,spacing)
    z=(x[:,None]-centers)/spacing
    np.testing.assert_allclose(values.toarray(),np.nan_to_num(kind.shape(z)),atol=1e-14)
    np.testing.assert_allclose(gradient.toarray(),np.nan_to_num(kind.shape_x(z))/spacing,atol=1e-12)
    r=traveling.solve(33,True,mu=.01,final_time=.001,quadrature=16,
        spline_degree=degree,refinement=4,line_options=dict(n_basis=25),output_points=10001)
    assert len(r['x'])==10001
    assert r['summary']['line_options']['n_basis']==25
    assert r['summary']['spline_degree']==degree
    assert r['summary']['metrics']['rho']['linf']<.01
    np.testing.assert_allclose(r['output'][[0,-1]],r['exact'][[0,-1]],atol=1e-9)


def test_reconditioned_updates_keep_the_represented_solution(traveling):
    r=traveling.solve(65,True,mu=.01,final_time=.006,quadrature=32,
        spline_degree=9,refinement=4,line_options=dict(n_basis=25),recondition=True,interval=.002)
    assert len(r['summary']['updates'])>=3
    assert max(row.get('remap_error',0) for row in r['summary']['updates'])<1e-7
    assert r['summary']['metrics']['rho']['linf']<.001
    np.testing.assert_allclose(r['output'][[0,-1]],r['exact'][[0,-1]],atol=1e-9)


def test_fixed_selection_does_not_grow_or_remap(traveling):
    r=traveling.solve(33,True,mu=.01,final_time=.006,interval=.002,
                     quadrature=16,selection='fixed',fixed_centers=[.4,.45])
    s=r['summary']
    assert len(s['updates'])==1
    assert s['initial_modes']==s['final_modes']
    np.testing.assert_allclose(r['output'][[0,-1]],r['exact'][[0,-1]],atol=1e-9)


def test_oracle_selection_uses_exact_snapshot(traveling,monkeypatch):
    calls=[]
    original=traveling.detect
    def recorded(x,samples,h,count):
        calls.append((x.copy(),samples.copy()))
        return original(x,samples,h,count)
    monkeypatch.setattr(traveling,'detect',recorded)
    traveling.solve(33,True,mu=.01,final_time=.005,interval=.002,
                    quadrature=16,selection='oracle',recondition=True)
    assert len(calls)==3
    shock=traveling.BeckerShock(mu=.01)
    for (x,samples),t in zip(calls,[0.,.002,.004]):
        np.testing.assert_allclose(samples,shock.state(x,t),atol=1e-12)
