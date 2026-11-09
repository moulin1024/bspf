# SPDX-License-Identifier: GPL-3.0-or-later
"""Compact graded physical B-splines plus BSPF Fourier background.

Only basis knots and integration points adapt. The uniform BSPF solution grid
is unchanged. A spline antiderivative carries the jump; compact splines correct
its shape. The detector supplies the center and width from the computed state.
"""
import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix
from fft_quadrature_shock import WindowGaussBase,MappedFrontSpace
from shock_tube_enrichment import EnrichedSpace
from shock_tube_spline_enrichment import SparseSplineSpace


class SplineQuadratureBase(WindowGaussBase):
    def __init__(self,template,edges,order):
        self.__dict__.update(template.__dict__)
        self.coarse=template.galerkin;self.coarse_q=template.q;self.coarse_weights=template.weights
        lo,hi=edges[[0,-1]]
        # Replace whole background cells so the composite measure has no gaps.
        h=1/(self.n-1);lo=max(0.,np.floor(lo/h)*h);hi=min(1.,np.ceil(hi/h)*h)
        edges=np.unique(np.r_[lo,edges,hi,self.line.knots[(self.line.knots>lo)&(self.line.knots<hi)]])
        self.outside=(self.coarse_q<lo)|(self.coarse_q>hi)
        z,w=np.polynomial.legendre.leggauss(order)
        widths=np.diff(edges)
        extra=(edges[:-1,None]+widths[:,None]*(z+1)/2).ravel()
        self.extra_w=(widths[:,None]*w/2).ravel()
        self.E=self.line.evaluate(extra).matrix;self.Ex=self.line.evaluate(extra,derivative=1).matrix
        self.q=np.r_[self.coarse_q[self.outside],extra]
        self.weights=np.r_[self.coarse_weights[self.outside],self.extra_w]
        self.outside_count=int(self.outside.sum())
        self.F=self.line.fit_samples+self.line.fit_jets@self.line.jet_map
        self.B=self.line.basis;self.period=self.n/(self.n-1)
        self.splines=BSpline(self.line.knots,np.eye(self.line.n_basis),self.line.degree)
        self.build_mass()
        self.integral=self.project(np.ones((len(self.q),1)))[:,0]


class GradedSplineSpace(EnrichedSpace):
    def __init__(self,template,center,width,intervals=48,degree=13,core_scale=.25,
                 support_width=16.,core_padding=0.,quadrature=None,compiled=True):
        self.degree=degree
        scale=width*core_scale
        a=np.arcsinh(support_width/core_scale)
        edges=center+scale*np.sinh(np.linspace(-a,a,intervals+1))
        if core_padding:
            spacing=scale*np.sinh(2*a/intervals)
            half=core_padding*width
            cells=int(np.ceil(2*half/spacing))
            edges=np.r_[edges[edges<center]-half,
                        np.linspace(center-half,center+half,cells+1),
                        edges[edges>center]+half]
        if edges[0]<=0 or edges[-1]>=1:raise ValueError('Graded enrichment support must be interior')
        self.knots=np.r_[np.repeat(edges[0],degree),edges,np.repeat(edges[-1],degree)]
        count=len(self.knots)-degree-1
        self.indices=np.arange(degree,count-degree)
        self.basis=BSpline(self.knots,np.eye(count),degree,extrapolate=False)
        # A smooth spline CDF supplies different constant states on either side.
        self.step_integral=BSpline(self.knots,(np.arange(count)>=count//2).astype(float),degree,extrapolate=False)
        self.step_spline=self.step_integral.derivative()
        self.step_left=0.;self.step_norm=1.
        base=SplineQuadratureBase(template,edges,degree+2 if quadrature is None else quadrature)
        self.support=(edges[0],edges[-1])
        k=len(self.indices)+1
        super().__init__(base,np.full(k,center),np.full(k,width),orthogonalize=True)
        loads=np.zeros((self.n+self.k,2));loads[[0,self.n-1]]=np.eye(2)
        response=self.mass(loads)
        self.wall_lift=np.linalg.solve(response[[0,self.n-1]].T,response.T).T
        if compiled:
            value=np.column_stack((base.evaluate(np.eye(self.n)),self.H))
            gradient=np.column_stack((base.evaluate(np.eye(self.n),1),self.Hx))
            self.compiled_values=np.ascontiguousarray(value)
            self.compiled_both=np.ascontiguousarray(np.vstack((value,gradient)))
            rate=self.mass(gradient.T*base.weights)
            rate-=self.wall_lift@rate[[0,self.n-1]];rate[[0,self.n-1]]=0.
            self.compiled_rate=np.ascontiguousarray(rate)
            self.constrained_flux_rate=rate.__matmul__

    def raw(self,x):
        x=np.asarray(x);lo,hi=self.support;inside=(x>=lo)&(x<=hi)
        v=np.zeros((len(x),len(self.indices)+1));g=np.zeros_like(v)
        v[:,0]=(x>hi).astype(float)-x;g[:,0]=-1.
        v[inside,0]=(self.step_integral(x[inside])-self.step_left)/self.step_norm-x[inside]
        g[inside,0]+=self.step_spline(x[inside])/self.step_norm
        v[inside,1:]=self.basis(x[inside])[:,self.indices]
        g[inside,1:]=self.basis(x[inside],nu=1)[:,self.indices]
        if x is self.base.q:
            self.S,self.Sx=csr_matrix(v),csr_matrix(g);self.ST,self.SxT=self.S.T,self.Sx.T
        return v,g

    evaluate=MappedFrontSpace.evaluate
    evaluate_with_gradient=MappedFrontSpace.evaluate_with_gradient
    project=SparseSplineSpace.project

    def at(self,x,state):
        z=self.transform@state[self.n:]
        lo,hi=self.support;inside=(x>=lo)&(x<=hi)
        step=(x>hi).astype(float)-x
        step[inside]=self.step_integral(x[inside])-x[inside]
        value=self.base.at(x,state[:self.n]-self.correction@z)+step[:,None]*z[0]
        coef=np.zeros((len(self.knots)-self.degree-1,state.shape[1]));coef[self.indices]=z[1:]
        value[inside]+=BSpline(self.knots,coef,self.degree,extrapolate=False)(x[inside])
        return value


def graded_factory(**options):
    def factory(base,centers,widths):
        if len(centers)!=1:raise ValueError('Current graded factory supports one detected front')
        return GradedSplineSpace(base,centers[0],widths[0],**options)
    return factory
