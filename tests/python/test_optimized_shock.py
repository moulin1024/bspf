"""Independent consistency checks for the optimized Galerkin shock backend."""
import importlib
from pathlib import Path
import numpy as np
import pytest
from scipy.special import eval_jacobi


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'examples/pde'))
    return (importlib.import_module('fft_quadrature_shock'),
            importlib.import_module('compiled_galerkin_step'),
            importlib.import_module('viscous_traveling_shock'))


def test_jacobi_recurrence_and_gradient(modules):
    m,_,_=modules
    e=m.MappedFrontSpace.__new__(m.MappedFrontSpace)
    e.front_centers=[.4];e.front_widths=[.007];e.degree=52
    x=np.linspace(.34,.46,301);z=np.tanh((x-.4)/.007)
    v,g=e.modes(x,True)
    for j in (0,1,2,19,52):
        p=eval_jacobi(j,1,1,z)
        dp=np.zeros_like(z) if j==0 else .5*(j+3)*eval_jacobi(j-1,2,2,z)
        np.testing.assert_allclose(v[:,j+1],(1-z*z)*p,atol=1e-13)
        np.testing.assert_allclose(g[:,j+1],(-2*z*p+(1-z*z)*dp)*(1-z*z)/.007,atol=3e-11)


@pytest.mark.parametrize('backend',['uniform','window'])
def test_quadrature_adjoint_mass_and_constant(modules,backend):
    m,_,_=modules
    factory=m.uniform_base(16,12) if backend=='uniform' else m.window_base((.3,.5),128)
    b=factory(33,quadrature=8,line_options=dict(n_basis=11))
    rng=np.random.default_rng(39);u=rng.normal(size=(33,3));v=rng.normal(size=(len(b.q),3))
    for k in (0,1):
        direct=b.line.evaluate(b.q,derivative=k).matrix@u
        np.testing.assert_allclose(b.evaluate(u,k),direct,atol=2e-9,rtol=2e-10)
        np.testing.assert_allclose(np.sum(b.evaluate(u,k)*b.weights[:,None]*v),np.sum(u*b.project(v,k)),atol=1e-9)
    np.testing.assert_allclose(b.solve_mass(b.project(b.evaluate(u))),u,atol=2e-8,rtol=2e-8)
    np.testing.assert_allclose(b.evaluate(np.ones((33,1))),1,atol=2e-12)
    for x in (np.linspace(0,1,101),rng.uniform(0,1,101)):
        np.testing.assert_allclose(b.at(x,u),b.line.evaluate(x).matrix@u,atol=2e-10)


def test_compiled_rhs_and_segment_match_structured(modules):
    m,k,shock_module=modules
    shock=shock_module.BeckerShock()
    b=m.window_base((.3,.5),384)(33,quadrature=8,line_options=dict(n_basis=11))
    e=m.mapped_factory(20,1.,True)(b,[.401],[.007])
    s=shock_module.ShockSpace(b,e);state=s.fit(shock)
    values,gradient=e.evaluate_with_gradient(state)
    f,fv=shock_module.fluxes(values,gradient,shock.mu,shock.kappa)
    reference=e.mass(e.project(f-fv,1));reference-=e.wall_lift@reference[[0,32]]
    reference[[0,32]]=0
    fast,valid=k.rhs(state,e.compiled_both,e.compiled_rate,shock.mu,shock.kappa,1.4)
    assert valid
    np.testing.assert_allclose(fast,reference,atol=2e-9,rtol=2e-10)
    end,steps,rejections=k.compiled_segment(s,state,0.,1e-6,shock,.7)
    h=1e-6
    a=s.rhs(state,0.,shock);bb=s.rhs(state+h*a/2,h/2,shock)
    c=s.rhs(state+h*bb/2,h/2,shock);d=s.rhs(state+h*c,h,shock)
    np.testing.assert_allclose(end,state+h*(a+2*bb+2*c+d)/6,atol=2e-13)
    assert steps==1 and rejections==0


def test_compiled_bounds_reject_zero_density(modules):
    _,k,_=modules
    assert k.bounds(np.array([[0.,1.,2.]]),1.4)==(-1.,-1.)


def test_quadrature_window_guard(modules):
    m,_,_=modules
    b=m.window_base((.3,.5),128)(33,quadrature=8,line_options=dict(n_basis=11))
    with pytest.raises(ValueError,match='left the integration window'):
        m.mapped_factory(10)(b,[.8],[.01])


def test_graded_spline_remap_and_endpoint_lifts(modules):
    _,_,shock_module=modules
    graded=importlib.import_module('graded_spline_shock')
    base=shock_module.BSPFNavierStokes(25,quadrature=8,line_options=dict(n_basis=10))
    factory=graded.graded_factory(intervals=38,degree=13,core_scale=.25,support_width=8.,core_padding=.5,quadrature=8)
    old=factory(base,[.4],[.007]);new=factory(base,[.401],[.0071])
    constant=np.tile([1.,.5,3.],(len(old.base.q),1))
    a=shock_module.ShockSpace(old.base,old);b=shock_module.ShockSpace(new.base,new)
    state=a.fit_samples(constant,constant[:2])
    samples=a.at(new.base.q,state);remapped=b.fit_samples(samples,constant[:2])
    np.testing.assert_allclose(b.evaluate(remapped),np.tile([1.,.5,3.],(len(new.base.q),1)),atol=3e-11)
    np.testing.assert_allclose(new.raw(np.array([0.,1.]))[0],0,atol=1e-14)
    x=np.linspace(.35,.45,1001);v,g=new.raw(x)
    assert np.min(g[:,0]+1)>-1e-12  # Monotone spline step before boundary lift.
    h=1e-8
    np.testing.assert_allclose((new.raw(x+h)[0]-new.raw(x-h)[0])/(2*h),g,atol=2e-5,rtol=2e-6)
