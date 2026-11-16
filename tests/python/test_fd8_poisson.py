"""CPU checks for FD8 closure order and the curved-coordinate Laplacian."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'bench/turbulence_3d/petsc_fd'),str(ROOT/'bench/turbulence_3d')]
from discretization import stencils,geometry,cpu_apply
from mms import CurvedBox

def grid(n):
    z,y,x=np.meshgrid(*[np.linspace(0,1,n)]*3,indexing='ij')
    return np.stack([x,y,z],axis=-1)

@pytest.mark.parametrize('derivative',[1,2])
def test_boundary_and_interior_moments(derivative):
    n=17;starts,w=stencils(n)
    for i in range(n):
        length=9 if derivative==1 or 4<=i<n-4 else 10
        # Normalize offsets to keep moment magnitudes bounded.
        x=(np.arange(length)+starts[derivative-1,i]-i)/9
        coef=w[derivative-1,i,:length]*(9/(n-1))**derivative
        for k in range(derivative+8):
            expected=(1 if derivative==1 else 2) if k==derivative else 0
            assert abs(coef@x**k-expected)<2e-11

def test_curved_physical_polynomials():
    n=17;p,G,b,det=geometry(grid(n),CurvedBox());start,w=stencils(n)
    assert det.min()>0
    for axis in range(3):
        assert np.max(abs(cpu_apply(p[...,axis],start,w,G,b)))<1e-9
    u=1+np.sum(p*p,axis=-1)
    assert np.max(abs(cpu_apply(u,start,w,G,b)+6))<1e-9

def test_smooth_truncation_order():
    errors=[]
    for n in [33,65]:
        p,G,b,_=geometry(grid(n),CurvedBox());start,w=stencils(n)
        u=np.prod(np.sin(2*np.pi*p),axis=-1)
        target=12*np.pi**2*u[1:-1,1:-1,1:-1]
        errors.append(np.linalg.norm(cpu_apply(u,start,w,G,b)-target)/np.linalg.norm(target))
    assert np.log2(errors[0]/errors[1])>7.5
