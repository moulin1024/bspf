"""Independent polynomial identities for the global Chebyshev trial basis."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from numpy.polynomial.legendre import Legendre,leggauss
import scipy.linalg as la
path=Path(__file__).resolve().parents[2]/'bench/turbulence_3d/chebyshev_poisson/axis.py'
spec=importlib.util.spec_from_file_location('chebyshev_test_axis',path)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

@pytest.mark.parametrize('n',[16,17,65,145])
def test_polynomial_space_and_reflection(n):
    a=mod.ChebyshevAxis(n)
    z,w=leggauss(n+3);s=(z+1)/2
    B,D,D2=a.evaluate(s,2)
    assert np.max(abs(a.evaluate(a.x,0)[0]-np.eye(n)))<1e-11
    assert np.max(abs(B-B[::-1,::-1]))<1e-11
    assert np.max(abs(D+D[::-1,::-1]))<1e-9
    # Legendre recurrence/derivatives are independent of the Chebyshev construction.
    for degree in [0,1,3,n-1]:
        p=Legendre.basis(degree);v=p(2*a.x-1)
        assert np.max(abs(B@v-p(z)))<1e-11
        assert np.max(abs(D@v-2*p.deriv()(z)))/max(1,np.max(abs(2*p.deriv()(z))))<2e-10
        # Endpoint second derivatives grow like n^4: cancellation on constants
        # needs a norm-scaled roundoff allowance (D2 is not used by the weak solver).
        reference=4*p.deriv(2)(z)
        bound=2e-10*np.max(abs(reference))+64*np.finfo(float).eps*la.norm(D2,np.inf)*np.max(abs(v))
        assert np.max(abs(D2@v-reference))<bound
    # Ordinary (unweighted) mass integration must give the Legendre norm.
    p=Legendre.basis(n-1);v=p(2*a.x-1)
    M=B.T@((w/2)[:,None]*B)
    assert abs(v@M@v-1/(2*n-1))<1e-12


def test_dirichlet_generalized_spectrum():
    a=mod.ChebyshevAxis(33);z,w=leggauss(50);B,D=a.evaluate((z+1)/2,1)
    M=B.T@((w/2)[:,None]*B);K=D.T@((w/2)[:,None]*D)
    lam,V=la.eigh(K[1:-1,1:-1],M[1:-1,1:-1])
    assert lam.min()>0
    assert np.allclose(lam[:5],(np.arange(1,6)*np.pi)**2,rtol=1e-10)
    assert np.max(abs(V.T@M[1:-1,1:-1]@V-np.eye(31)))<1e-12
