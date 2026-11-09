import numpy as np
import pytest
from .laplacian import MappedLaplacian
from .spectral import nodes,transform2


def curved_map():
    x,y=np.meshgrid(nodes(7),nodes(7),indexing='ij')
    z=x+.13*(1-x*x)*y + 1j*(y+.11*(1-y*y)*x)
    return transform2(z[None])


@pytest.mark.parametrize('factor',[1,2,3])
def test_physical_polynomial_patch_and_sign(factor):
    op=MappedLaplacian(curved_map(),20,factor)
    z,*_=op.geometry(op.n)
    for u,exact in [(np.ones_like(z.real),0.),(z.real,0.),
                    (z.imag,0.),(abs(z)**2,4.),(z.real*z.imag,0.)]:
        assert np.max(abs(op.conservative(u)-exact))<2e-8
        assert np.max(abs(op.chain_rule(u)-exact))<2e-8


def test_trigonometric_convergence_and_independent_forms():
    errors=[]
    for p in [10,18,26]:
        op=MappedLaplacian(curved_map(),p,2)
        z,*_=op.geometry(op.n)
        u=np.sin(1.7*z.real)*np.cos(2.3*z.imag)
        exact=-(1.7**2+2.3**2)*u
        errors.append(np.max(abs(op.conservative(u)-exact)))
        if p==26:
            assert np.max(abs(op.conservative(u)-op.chain_rule(u)))<2e-8
    assert errors[-1]<2e-8 and errors[1]<errors[0]/100


def test_interface_flux_uses_physical_normal_not_chart_scale():
    x,y=np.meshgrid(nodes(5),nodes(5),indexing='ij')
    c=transform2(np.stack([.5*(x-1)+1j*y,2*(x+1)+1j*y]))
    op=MappedLaplacian(c,12)
    z,*_=op.geometry(op.n)
    gx,gy=op.gradient(z.real**2+z.real+z.imag**2)
    assert np.max(abs(gx[0,-1,:]-gx[1,0,:]))<1e-10
    assert np.max(abs(gy[0,-1,:]-gy[1,0,:]))<1e-10
