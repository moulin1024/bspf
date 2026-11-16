"""Global P_(n-1) trial space in Chebyshev-Lobatto nodal coordinates.

Chebyshev polynomials define the trial basis; the physical weak form uses the
ordinary volume measure, integrated by Gauss-Legendre quadrature.
"""
import numpy as np
from scipy.fft import dct
from numpy.polynomial.chebyshev import chebval, chebder


class ChebyshevAxis:
    def __init__(self,n):
        if isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or n<4:
            raise ValueError('At least four Chebyshev-Lobatto nodes required')
        self.n=int(n);p=n-1
        j=np.arange(n);k=j[:,None]
        self.x=(1-np.cos(np.pi*j/p))/2
        self.x=(self.x+1-self.x[::-1])/2
        self.x[0]=0;self.x[-1]=1
        weights=np.ones(n);weights[[0,-1]]=.5
        # Inverse Chebyshev Vandermonde from discrete cosine orthogonality.
        coefficients=(weights[:,None]/p)*((-1.)**k)*dct(np.eye(n),type=1,axis=0)
        # Enforce exact polynomial reflection identities up to floating-point arithmetic.
        coefficients=(coefficients+((-1.)**k)*coefficients[:,::-1])/2
        self.coefficients=coefficients
        self.derivatives=[chebder(coefficients,m=d,axis=0)*2**d for d in range(3)]
        cardinal=float(np.max(abs(self.evaluate(self.x,0)[0]-np.eye(n))))
        if cardinal>1e-10:raise ValueError('Chebyshev cardinal basis validation failed')
        self.metadata=dict(discretization='global_chebyshev_galerkin',degree=p,n=n,
            nodes='chebyshev_lobatto',cardinal_defect=cardinal,
            node_symmetry_defect=float(np.max(abs(self.x+self.x[::-1]-1))),
            integration_measure='ordinary physical volume (not Chebyshev-weighted)')

    def evaluate(self,x,order=2):
        if order not in (0,1,2):raise ValueError('derivative order must be 0, 1, or 2')
        return [chebval(2*np.asarray(x)-1,c).T for c in self.derivatives[:order+1]]
