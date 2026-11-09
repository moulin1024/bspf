# SPDX-License-Identifier: GPL-3.0-or-later
"""FFT quadrature and detector-driven mapped enrichment for BSPF Galerkin.

Trial functions remain the original BSPF functions. Evaluation and its weighted
transpose use padded FFTs or a hybrid Gauss rule. Each rule has its own consistent
low-rank mass inverse. Small enriched operators can optionally be precompiled;
this is a 1D optimization, not a demonstrated scalable 3D implementation.
"""
from fractions import Fraction
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
from scipy.fft import rfft, irfft
from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix
from scipy.linalg import eigh
from scipy.signal import ZoomFFT
from shock_tube_1d import BSPFNavierStokes
from shock_tube_enrichment import EnrichedSpace
from shock_tube_spline_enrichment import SparseSplineSpace


@lru_cache(maxsize=16)
def zoom_operator(modes,count,start,spacing,period):
    return ZoomFFT(modes,[-start/period,-(start+count*spacing)/period],m=count,fs=1.)


@lru_cache(None)
def endpoint_weights(order):
    """Exact-rational Euler--Maclaurin correction through degree order-1."""
    # B_m from the defining Bernoulli recurrence (B_1=-1/2).
    from math import comb
    bern=[Fraction(1)]
    for m in range(1,order+1):
        bern.append(-sum(Fraction(comb(m+1,k))*bern[k] for k in range(m))/Fraction(m+1))
    a=[[Fraction(j)**m for j in range(order)]+[
        bern[m+1]/(m+1) if m%2 else Fraction(0)] for m in range(order)]
    for i in range(order):
        pivot=next(j for j in range(i,order) if a[j][i])
        a[i],a[pivot]=a[pivot],a[i]
        scale=a[i][i];a[i]=[v/scale for v in a[i]]
        for j in range(order):
            if j!=i:
                scale=a[j][i];a[j]=[v-scale*w for v,w in zip(a[j],a[i])]
    return np.array([float(row[-1]) for row in a])


class UniformFFTBase(BSPFNavierStokes):
    oversampling=8
    endpoint_order=10

    def __init__(self,n,**kwargs):
        if n%2!=1:raise ValueError('Experimental padded FFT backend requires odd base size')
        if not isinstance(self.oversampling,int) or self.oversampling<2:
            raise ValueError('oversampling must be an integer >=2')
        if not isinstance(self.endpoint_order,int) or not 2<=self.endpoint_order<=16:
            raise ValueError('endpoint_order must be between 2 and 16')
        super().__init__(n,**kwargs)
        self.reference_q=self.q;self.reference_weights=self.weights
        self.F=self.line.fit_samples+self.line.fit_jets@self.line.jet_map
        self.B=self.line.basis
        self.M=n*self.oversampling
        self.count=(n-1)*self.oversampling+1
        self.period=n/(n-1)
        self.q=np.arange(self.count)/(self.M/self.period)
        self.weights=np.full(self.count,self.period/self.M)
        edge=endpoint_weights(self.endpoint_order).copy();edge[0]-=.5
        self.weights[:len(edge)]+=edge*self.period/self.M
        self.weights[-len(edge):]+=edge[::-1]*self.period/self.M
        self.k=2*np.pi*np.arange(n//2+1)/self.period
        self.splines=BSpline(self.line.knots,np.eye(self.line.n_basis),self.line.degree)
        self.V=csr_matrix(self.splines(self.q));self.G=csr_matrix(self.splines(self.q,nu=1))
        # Build the mass correction for THIS quadrature, then keep its low rank.
        # This prototype uses dense setup; time stepping retains structured solves.
        self.build_mass()
        self.integral=self.project(np.ones((len(self.q),1)))[:,0]

    def build_mass(self):
        n=self.n
        self.reference_galerkin=self.galerkin
        mass=self.project(self.evaluate(np.eye(n)));mass=(mass+mass.T)/2
        val,vec=eigh(mass[1:-1,1:-1]*(n-1)-np.eye(n-2))
        if np.min(val)<=-1:raise ValueError('Quadrature mass is not positive definite')
        keep=abs(val)>2e-13
        self.mass_U=vec[:,keep];self.mass_shift=val[keep]
        self.mass_shrink=self.mass_shift/(1+self.mass_shift)
        self.mass_IB=mass[1:-1][:,[0,-1]]
        self.mass_extension=self.solve_mass(self.mass_IB,True)
        self.wall_inverse=np.linalg.inv(mass[np.ix_([0,n-1],[0,n-1])]-self.mass_IB.T@self.mass_extension)
        self.galerkin=SimpleNamespace(mass=self.mass,stiffness=lambda u:self.project(self.evaluate(u,1),1),
            solve_mass=self.solve_mass,fit_amplification=self.reference_galerkin.fit_amplification,
            mass_rank=len(self.mass_shift))
        loads=np.zeros((n,2));loads[[0,-1]]=np.eye(2)
        response=self.solve_mass(loads)
        self.wall_lift=np.linalg.solve(response[[0,-1]].T,response.T).T
        # Enrichment radius estimation uses the same mass and quadrature stiffness.

    def evaluate(self,state,derivative=0):
        if not hasattr(self,'F'):return super().evaluate(state,derivative)
        c=self.F@state;r=state-self.B@c
        spectrum=rfft(r,axis=0)*(1j*self.k[:,None])**derivative
        return irfft(spectrum,n=self.M,axis=0)[:self.count]*(self.M/self.n)+(self.G if derivative else self.V)@c

    def solve_mass(self,rhs,interior=False):
        if not hasattr(self,'mass_U'):return super().solve_mass(rhs,interior)
        def inner(b):return (b-self.mass_U@(self.mass_shrink[:,None]*(self.mass_U.T@b)))*(self.n-1)
        if interior:return inner(rhs)
        z=inner(rhs[1:-1]);wall=self.wall_inverse@(rhs[[0,-1]]-self.mass_IB.T@z)
        return np.vstack((wall[:1],z-self.mass_extension@wall,wall[1:]))

    def mass(self,u):
        middle=(u[1:-1]+self.mass_U@(self.mass_shift[:,None]*(self.mass_U.T@u[1:-1])))/(self.n-1)
        middle+=self.mass_IB@u[[0,-1]]
        bb=np.linalg.inv(self.wall_inverse)+self.mass_IB.T@self.mass_extension
        wall=self.mass_IB.T@u[1:-1]+bb@u[[0,-1]]
        return np.vstack((wall[:1],middle,wall[1:]))

    def project(self,values,derivative=0):
        if not hasattr(self,'F'):return super().project(values,derivative)
        weighted=self.weights[:,None]*values
        spectrum=rfft(weighted,n=self.M,axis=0)[:len(self.k)]*(-1j*self.k[:,None])**derivative
        t=irfft(spectrum,n=self.n,axis=0)
        b=(self.G if derivative else self.V).T@weighted
        return t+self.F.T@(b-self.B.T@t)

    def evaluate_with_gradient(self,state):
        c=self.F@state;spectrum=rfft(state-self.B@c,axis=0)
        values=irfft(spectrum,n=self.M,axis=0)[:self.count]*(self.M/self.n)+self.V@c
        gradient=irfft(spectrum*(1j*self.k[:,None]),n=self.M,axis=0)[:self.count]*(self.M/self.n)+self.G@c
        return values,gradient

    def at(self,x,state):
        c=self.F@state;r=state-self.B@c
        spectrum=rfft(r,axis=0)/self.n;spectrum[1:]*=2
        if len(x)>2 and np.allclose(np.diff(x),x[1]-x[0],rtol=1e-12,atol=1e-15):
            values=zoom_operator(len(spectrum),len(x),float(x[0]),float(x[1]-x[0]),self.period)(spectrum,axis=0).real
        else:
            values=(np.exp(2j*np.pi*x[:,None]*np.arange(len(spectrum))/self.period)@spectrum).real
        return values+BSpline(self.line.knots,c,self.line.degree)(x)


def uniform_base(oversampling=8,endpoint_order=10):
    class ConfiguredBase(UniformFFTBase):pass
    ConfiguredBase.oversampling=oversampling
    ConfiguredBase.endpoint_order=endpoint_order
    return ConfiguredBase


class MappedFrontSpace(EnrichedSpace):
    """Tanh-coordinate Jacobi polynomials; center/width come from the detector."""
    def __init__(self,base,centers,widths,degree,compiled=False):
        if not isinstance(degree,int) or degree<0:raise ValueError('Nonnegative integer mapped degree required')
        if np.any(np.asarray(widths)<=0):raise ValueError('Positive mapped widths required')
        if hasattr(base,'integration_window'):
            lo,hi=base.integration_window
            if np.any(np.asarray(centers)-3*np.asarray(widths)<lo) or np.any(np.asarray(centers)+3*np.asarray(widths)>hi):
                raise ValueError('Front left the integration window; rebuild quadrature')
        self.front_centers=np.asarray(centers);self.front_widths=np.asarray(widths)
        self.degree=degree
        self.orders=np.tile(np.arange(-1,degree+1),len(centers))
        super().__init__(base,np.repeat(centers,degree+2),np.repeat(widths,degree+2),orthogonalize=True)
        loads=np.zeros((self.n+self.k,2));loads[[0,self.n-1]]=np.eye(2)
        response=self.mass(loads)
        self.wall_lift=np.linalg.solve(response[[0,self.n-1]].T,response.T).T
        raw,raw_x=self.raw(base.q)
        self.S,self.Sx=csr_matrix(raw),csr_matrix(raw_x)
        self.ST,self.SxT=self.S.T,self.Sx.T
        if compiled:
            value=np.column_stack((base.evaluate(np.eye(self.n)),self.H))
            gradient=np.column_stack((base.evaluate(np.eye(self.n),1),self.Hx))
            self.compiled_values=np.ascontiguousarray(value)
            self.compiled_both=np.ascontiguousarray(np.vstack((value,gradient)))
            rate=self.mass(gradient.T*base.weights)
            rate-=self.wall_lift@rate[[0,self.n-1]]
            rate[[0,self.n-1]]=0.
            self.compiled_rate=np.ascontiguousarray(rate)
            self.constrained_flux_rate=rate.__matmul__

    def raw(self,x):
        x=np.asarray(x)
        values,gradient=self.modes(x,True)
        ends=self.modes(np.array([0.,1.]),False);slope=ends[1]-ends[0]
        values-=ends[0]+x[:,None]*slope;gradient-=slope
        if x is self.base.q:
            self.S,self.Sx=csr_matrix(values),csr_matrix(gradient)
            self.ST,self.SxT=self.S.T,self.Sx.T
        return values,gradient

    def modes(self,x,gradient=False):
        """Build all Jacobi degrees by one recurrence, rather than O(p^2)."""
        values=[];derivatives=[]
        for center,width in zip(self.front_centers,self.front_widths):
            z=np.tanh((x-center)/width);s=1-z*z;d=s/width
            v=np.empty((len(x),self.degree+2));v[:,0]=z
            g=np.empty_like(v) if gradient else None
            if gradient:g[:,0]=d
            previous=np.zeros_like(z);poly=np.ones_like(z)
            dprevious=np.zeros_like(z);dp=np.zeros_like(z)
            for j in range(self.degree+1):
                v[:,j+1]=s*poly
                if gradient:g[:,j+1]=(-2*z*poly+s*dp)*d
                n=j+1
                a=(2*n+1)*(n+1)/(n*(n+2))
                b=(n+1)/(n+2)
                previous,poly=poly,a*z*poly-b*previous
                if gradient:dprevious,dp=dp,a*(previous+z*dp)-b*dprevious
            values.append(v)
            if gradient:derivatives.append(g)
        return (np.column_stack(values),np.column_stack(derivatives)) if gradient else np.column_stack(values)

    def evaluate_with_gradient(self,state):
        if hasattr(self,'compiled_both'):
            both=self.compiled_both@state
            return both[:len(self.base.q)],both[len(self.base.q):]
        return SparseSplineSpace.evaluate_with_gradient(self,state)

    def evaluate(self,state,derivative=0):
        if hasattr(self,'compiled_both'):
            return (self.compiled_both[len(self.base.q):] if derivative else self.compiled_values)@state
        return SparseSplineSpace.evaluate(self,state,derivative)
    project=SparseSplineSpace.project

    def at(self,x,state):
        z=self.transform@state[self.n:]
        values=self.modes(x,False);ends=self.modes(np.array([0.,1.]),False)
        values-=ends[0]+x[:,None]*(ends[1]-ends[0])
        return self.base.at(x,state[:self.n]-self.correction@z)+values@z


def mapped_factory(degree=20,width_factor=1.,compiled=False):
    def factory(base,centers,widths):
        return MappedFrontSpace(base,centers,np.asarray(widths)*width_factor,degree,compiled)
    return factory


class WindowGaussBase(UniformFFTBase):
    """FFT background Gauss rule plus a dense local integration panel.

    The solution grid/trial space is unchanged. Only quadrature is concentrated
    near a window found from initial data. The panel includes a movement margin.
    """
    window=(.25,.55)
    panel_points=512

    def __init__(self,n,**kwargs):
        if n%2!=1:raise ValueError('Hybrid FFT backend requires odd base size')
        if not isinstance(self.panel_points,int) or self.panel_points<16:
            raise ValueError('panel_points must be an integer >=16')
        if not 0<=self.window[0]<self.window[1]<=1:
            raise ValueError('Integration window must lie inside [0,1]')
        BSPFNavierStokes.__init__(self,n,**kwargs)
        self.coarse=self.galerkin;self.coarse_q=self.q;self.coarse_weights=self.weights
        h=1/(n-1)
        lo=max(0.,np.floor(self.window[0]/h)*h)
        hi=min(1.,np.ceil(self.window[1]/h)*h)
        self.integration_window=(lo,hi)
        self.outside=(self.q<lo)|(self.q>hi)
        z,w=np.polynomial.legendre.leggauss(self.panel_points)
        extra=lo+(hi-lo)*(z+1)/2;self.extra_w=w*(hi-lo)/2
        self.E=self.line.evaluate(extra).matrix
        self.Ex=self.line.evaluate(extra,derivative=1).matrix
        self.q=np.r_[self.coarse_q[self.outside],extra]
        self.weights=np.r_[self.coarse_weights[self.outside],self.extra_w]
        self.outside_count=int(self.outside.sum())
        self.F=self.line.fit_samples+self.line.fit_jets@self.line.jet_map
        self.B=self.line.basis;self.period=n/(n-1)
        self.splines=BSpline(self.line.knots,np.eye(self.line.n_basis),self.line.degree)
        self.build_mass()
        self.integral=self.project(np.ones((len(self.q),1)))[:,0]

    def evaluate(self,u,derivative=0):
        if not hasattr(self,'outside_count'):return BSPFNavierStokes.evaluate(self,u,derivative)
        return np.vstack((self.coarse.evaluate(u,derivative)[self.outside],(self.Ex if derivative else self.E)@u))

    def evaluate_with_gradient(self,u):return self.evaluate(u),self.evaluate(u,1)

    def project(self,v,derivative=0):
        if not hasattr(self,'outside_count'):return BSPFNavierStokes.project(self,v,derivative)
        coarse=np.zeros((len(self.coarse_q),v.shape[1]));coarse[self.outside]=v[:self.outside_count]
        return self.coarse.project(coarse,derivative)+(self.Ex if derivative else self.E).T@(self.extra_w[:,None]*v[self.outside_count:])


def window_base(window,panel_points=512):
    class ConfiguredBase(WindowGaussBase):pass
    ConfiguredBase.window=tuple(window);ConfiguredBase.panel_points=panel_points
    return ConfiguredBase
