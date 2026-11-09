"""Block-local physical Laplacians using FFT Chebyshev differentiation.

These actions do not impose boundary conditions or couple a global PDE solve.
Geometry coefficients are independent of the solution interpolation degree.
"""
import numpy as np
from .spectral import derivative, evaluate2, transform2


class MappedLaplacian:
    def __init__(self, geometry_coefficients, degree, flux_oversampling=1):
        self.c = np.asarray(geometry_coefficients)
        self.n = degree+1
        if degree < self.c.shape[-1]-1:
            raise ValueError('Solution grid must resolve the polynomial map degree')
        if flux_oversampling not in (1,2,3):
            raise ValueError('flux_oversampling must be 1, 2, or 3')
        self.factor = flux_oversampling
        self.m = flux_oversampling*degree+1
        self.z,self.a,self.b,self.j = self.geometry(self.m)
        self.A = abs(self.b)**2/self.j
        self.B = -np.real(np.conj(self.a)*self.b)/self.j
        self.C = abs(self.a)**2/self.j

    def geometry(self,m):
        z = evaluate2(self.c,m)
        a = evaluate2(derivative(self.c,-2),m)
        b = evaluate2(derivative(self.c,-1),m)
        j = np.imag(np.conj(a)*b)
        if np.min(j)<=0:
            raise ValueError('Nonpositive sampled map Jacobian')
        return z,a,b,j

    def gradient(self,u,count=None):
        m = count or self.n
        _,a,b,j = self.geometry(m)
        c = transform2(u)
        ux = evaluate2(derivative(c,-2),m)
        uy = evaluate2(derivative(c,-1),m)
        return ((b.imag*ux-a.imag*uy)/j,
                (-b.real*ux+a.real*uy)/j)

    def conservative(self,u):
        """J^-1 div_reference(J g^-1 grad_reference(u)).

        Oversampling interpolates u first, multiplies fluxes on the enlarged
        grid, then differentiates. Nested Lobatto sampling returns to n x n.
        It reduces rational-metric aliasing, without adding solution DOFs.
        """
        c = transform2(u)
        uxi = evaluate2(derivative(c,-2),self.m)
        ueta = evaluate2(derivative(c,-1),self.m)
        f = transform2(self.A*uxi+self.B*ueta)
        g = transform2(self.B*uxi+self.C*ueta)
        result = (evaluate2(derivative(f,-2),self.m)
                  + evaluate2(derivative(g,-1),self.m))/self.j
        return result[...,::self.factor,::self.factor]

    def chain_rule(self,u,count=None):
        """Independent non-divergence form with analytic polynomial map jets.

        Evaluates the physical Laplacian of the solution interpolant. No
        interpolation of rational metric fluxes is needed in this form.
        """
        m = count or self.n
        _,a,b,j = self.geometry(m)
        e,f,g = abs(b)**2/j**2,-np.real(np.conj(a)*b)/j**2,abs(a)**2/j**2
        dc = derivative(self.c,-2);ec = derivative(self.c,-1)
        curvature = (e*evaluate2(derivative(dc,-2),m)
                     + 2*f*evaluate2(derivative(dc,-1),m)
                     + g*evaluate2(derivative(ec,-1),m))
        drift_xi = -(b.imag*curvature.real-b.real*curvature.imag)/j
        drift_eta = -(-a.imag*curvature.real+a.real*curvature.imag)/j
        c = transform2(u);cx = derivative(c,-2);cy = derivative(c,-1)
        return (e*evaluate2(derivative(cx,-2),m)
                + 2*f*evaluate2(derivative(cx,-1),m)
                + g*evaluate2(derivative(cy,-1),m)
                + drift_xi*evaluate2(cx,m)+drift_eta*evaluate2(cy,m))
