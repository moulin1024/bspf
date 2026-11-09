# SPDX-License-Identifier: GPL-3.0-or-later
"""Front-free fixed-dictionary BSPF enrichment: small 1D research prototype.

Initialization uses the supplied initial data. Later selection uses only weak
NS residuals in a fixed multilevel spline dictionary. No peak detector, fitted
front width, trajectory, exact interior data, damping, or moving knots.
Dense conditioning/compiled operators are deliberate prototype limitations.
"""
from time import perf_counter
from math import comb
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import cholesky, solve_triangular, cho_factor, cho_solve, eigh
from scipy.sparse import csr_matrix
from shock_tube_1d import BSPFNavierStokes, primitives
from compiled_galerkin_step import physical_flux, advance


def gauss_rule(edges, order):
    z,w=np.polynomial.legendre.leggauss(order)
    dx=np.diff(edges)
    return (edges[:-1,None]+dx[:,None]*(z+1)/2).ravel(),(dx[:,None]*w/2).ravel()


class FixedDictionary:
    """Compact cardinal B-splines at all allowed positions and dyadic scales."""
    def __init__(self, levels, degree=9,detail_order=4,step_level=0):
        if degree<3 or degree%2!=1:raise ValueError('Odd degree >=3 required')
        if detail_order<0 or detail_order%2:raise ValueError('Nonnegative even detail order required')
        self.degree=degree;self.detail_order=detail_order
        self.spline_half=(degree+1)//2;self.half=self.spline_half+detail_order//2
        self.shape=BSpline.basis_element(np.arange(-self.spline_half,self.spline_half+1,dtype=float),extrapolate=False)
        self.shape_x=self.shape.derivative()
        self.cdf=self.shape.antiderivative()
        self.labels=[(level,j,'detail') for level in levels for j in range(self.half,2**level-self.half+1)
                     if not detail_order or j%2==1]
        if step_level:
            self.labels += [(step_level,j,'step') for j in range(self.spline_half,2**step_level-self.spline_half+1)]
        self.lookup={label:i for i,label in enumerate(self.labels)}

    def matrices(self,x,indices=None):
        x=np.asarray(x)
        ids=range(len(self.labels)) if indices is None else indices
        rows=[];cols=[];v=[];g=[]
        for column,index in enumerate(ids):
            level,j,kind=self.labels[index];scale=2.**level
            at=np.arange(len(x)) if kind=='step' else np.flatnonzero(abs(x*scale-j)<self.half)
            z=x[at]*scale-j
            rows.extend(at);cols.extend([column]*len(at))
            if kind=='step':
                value=self.cdf(np.clip(z,-self.spline_half,self.spline_half))-x[at]
                gradient=scale*np.nan_to_num(self.shape(z))-1.
                v.extend(value);g.extend(gradient)
                continue
            value=np.zeros(len(z));gradient=np.zeros(len(z))
            for shift in range(self.detail_order+1):
                factor=(-1)**shift*comb(self.detail_order,shift)
                arg=z-shift+self.detail_order/2
                value+=factor*np.nan_to_num(self.shape(arg))
                gradient+=factor*np.nan_to_num(self.shape_x(arg))*scale
            v.extend(value);g.extend(gradient)
        shape=(len(x),len(self.labels) if indices is None else len(indices))
        return csr_matrix((v,(rows,cols)),shape=shape),csr_matrix((g,(rows,cols)),shape=shape)


class ResidualSpace:
    def __init__(self,n=33,levels=range(5,11),degree=9,probe_order=8,quadrature=10,max_modes=256,detail_order=4,step_level=8):
        self.n=n;self.quadrature=quadrature;self.max_modes=max_modes
        self.base=BSPFNavierStokes(n,quadrature=8,line_options=dict(n_basis=10))
        self.dictionary=FixedDictionary(tuple(levels),degree,detail_order,step_level)
        edges=np.unique(np.r_[np.linspace(0,1,2**max(levels)+1),self.base.line.knots])
        self.x,self.w=gauss_rule(edges,probe_order)
        self.P=self.base.line.evaluate(self.x).matrix
        self.Px=self.base.line.evaluate(self.x,derivative=1).matrix
        p=self.P[:,1:-1];g=p.T@(self.w[:,None]*p)
        t=solve_triangular(cholesky(g,lower=True).T,np.eye(n-2),lower=False)
        self.T=np.zeros((n,n-2));self.T[1:-1]=t
        self.Q=self.P@self.T;self.Qx=self.Px@self.T
        self.S,self.Sx=self.dictionary.matrices(self.x)
        self.norm2=np.asarray(self.S.power(2).T@self.w).ravel()
        self.cross=(self.S.T@(self.w[:,None]*self.Q)).T
        self.selected=[];self.blocked=np.zeros(len(self.dictionary.labels),dtype=bool)
        self.minimum_relative_pivot=1.;self.rebuild()

    def raw_matrices(self,x):
        p=self.base.line.evaluate(x).matrix
        px=self.base.line.evaluate(x,derivative=1).matrix
        s,sx=self.dictionary.matrices(x,self.selected)
        return np.column_stack((p,s.toarray())),np.column_stack((px,sx.toarray()))

    def at(self,x,state):
        result=[]
        for start in range(0,len(x),4096):
            raw,_=self.raw_matrices(np.asarray(x)[start:start+4096])
            result.append(raw[:,:self.n][:,[0,-1]]@state[:2]+(raw@self.T)@state[2:])
        return np.vstack(result)

    def fine_values(self,state):
        return self.P[:,[0,-1]]@state[:2]+self.Q@state[2:]

    def fine_gradient(self,state):
        return self.Px[:,[0,-1]]@state[:2]+self.Qx@state[2:]

    def add(self,index):
        if self.blocked[index] or self.n+len(self.selected)>=self.max_modes:return False
        self.blocked[index]=True
        v=self.S[:,index].toarray().ravel();vx=self.Sx[:,index].toarray().ravel()
        c=np.zeros(self.Q.shape[1])
        for _ in range(2):
            a=self.Q.T@(self.w*v);c+=a;v-=self.Q@a;vx-=self.Qx@a
        norm=np.sqrt(self.w@(v*v));relative=norm/np.sqrt(self.norm2[index])
        if relative<.03:return False
        column=np.r_[-self.T@c,1.]/norm
        # A quadrature-space QR can look orthogonal while its physical-basis
        # coordinates suffer catastrophic cancellation. Reject that direction.
        if np.max(abs(column))>1e4:return False
        self.T=np.column_stack((np.vstack((self.T,np.zeros((1,self.T.shape[1])))),column))
        self.Q=np.column_stack((self.Q,v/norm));self.Qx=np.column_stack((self.Qx,vx/norm))
        self.cross=np.vstack((self.cross,np.asarray(self.S.T@(self.w*v/norm)).ravel()))
        self.selected.append(index);self.minimum_relative_pivot=min(self.minimum_relative_pivot,relative)
        return True

    def scores(self,raw_load,active_load):
        remainder=np.maximum(self.norm2-np.sum(self.cross**2,axis=0),0.)
        valid=(~self.blocked)&(remainder>.03**2*self.norm2)
        score=np.zeros(len(remainder))
        score[valid]=np.max(abs(raw_load[valid]-self.cross[:,valid].T@active_load),axis=1)/np.sqrt(remainder[valid])
        return score

    def select(self,raw_load,active_load_fn,tolerance,neighbors=1):
        added=0;peak=0.
        while self.n+len(self.selected)<self.max_modes:
            scores=self.scores(raw_load,active_load_fn());index=int(np.argmax(scores));peak=float(scores[index])
            if peak<=tolerance:break
            level,j,kind=self.dictionary.labels[index]
            stride=2 if self.dictionary.detail_order and kind=='detail' else 1
            candidates=[index]+[self.dictionary.lookup[(level,j+offset*stride,kind)]
                for offset in range(-neighbors,neighbors+1) if offset and (level,j+offset*stride,kind) in self.dictionary.lookup]
            for candidate in candidates:added+=int(self.add(candidate))
        # Report the actual remaining indicator, including when the cap is hit.
        peak=float(np.max(self.scores(raw_load,active_load_fn())))
        return added,peak

    def initialize(self,initial,boundary,tolerance,neighbors=1):
        data=initial(self.x)-self.P[:,[0,-1]]@boundary
        raw=np.asarray(self.S.T@(self.w[:,None]*data))
        added,peak=self.select(raw,lambda:self.Q.T@(self.w[:,None]*data),tolerance,neighbors)
        self.rebuild()
        state=np.vstack((boundary,self.Q.T@(self.w[:,None]*data)))
        error=np.sqrt(np.sum(self.w[:,None]*(self.fine_values(state)-initial(self.x))**2,axis=0))
        return state,dict(added=added,indicator=peak,initial_l2=error.tolist())

    def adapt(self,state,mu,kappa,gamma,tolerance,neighbors=1):
        # Weak residual against functions outside the active Galerkin space.
        # Neither the reference solution nor a front locator enters this method.
        flux=physical_flux(self.fine_values(state),self.fine_gradient(state),mu,kappa,gamma)
        weighted=self.w[:,None]*flux
        raw=np.asarray(self.Sx.T@weighted)
        added,peak=self.select(raw,lambda:self.Qx.T@weighted,tolerance,neighbors)
        if added:
            state=np.vstack((state,np.zeros((added,3))))  # Exact nested injection.
            self.rebuild()
        return state,dict(added=added,indicator=peak)

    def rebuild(self):
        edges=[np.linspace(0,1,self.n),self.base.line.knots]
        for index in self.selected:
            level,j,kind=self.dictionary.labels[index]
            half=self.dictionary.spline_half if kind=='step' else self.dictionary.half
            edges.append(np.arange(j-half,j+half+1)/2**level)
        x,w=gauss_rule(np.unique(np.concatenate(edges)),self.quadrature)
        raw,rawx=self.raw_matrices(x)
        q=raw@self.T;qx=rawx@self.T
        value=np.column_stack((raw[:,:self.n][:,[0,-1]],q))
        gradient=np.column_stack((rawx[:,:self.n][:,[0,-1]],qx))
        mass=q.T@(w[:,None]*q);mass=(mass+mass.T)/2
        stiffness=qx.T@(w[:,None]*qx);stiffness=(stiffness+stiffness.T)/2
        self.radius=float(eigh(stiffness,mass,subset_by_index=[len(mass)-1,len(mass)-1],eigvals_only=True)[0])
        self.value=np.ascontiguousarray(value)
        self.both=np.ascontiguousarray(np.vstack((value,gradient)))
        self.rate=np.ascontiguousarray(np.vstack((np.zeros((2,len(x))),cho_solve(cho_factor(mass),qx.T*w))))
        self.integration_points=len(x)
        self.mass_defect=float(np.max(abs(mass-np.eye(len(mass)))))


def solve(initial,boundary,*,n=33,levels=range(5,11),degree=9,tolerance=1e-7,interval=.0025,
          final_time=.05,mu=.001,prandtl=.75,gamma=1.4,cfl=1.,quadrature=10,probe_order=8,
          max_modes=256,neighbors=1,adaptive=True,freeze=False,detail_order=4,step_level=8,output_points=16001,progress=False):
    """Fixed endpoint states; no exact interior callback is accepted after t=0."""
    if tolerance<=0 or interval<=0 or final_time<=0 or not 0<cfl<=1:raise ValueError('Invalid tolerances/time')
    start=perf_counter();kappa=mu*gamma/((gamma-1)*prandtl)
    space=ResidualSpace(n,levels,degree,probe_order,quadrature,max_modes if adaptive else n,detail_order,step_level)
    state,info=space.initialize(initial,boundary,tolerance*.05,neighbors)
    setup=perf_counter()-start;selection_seconds=0.;integration_seconds=0.
    t=0.;steps=0;rejected=0;updates=[];initial_state=state.copy()
    probes=np.linspace(0,1,output_points);initial_output=space.at(probes,state)
    while t<final_time-1e-14:
        tick=perf_counter()
        if adaptive and not freeze:state,info=space.adapt(state,mu,kappa,gamma,tolerance/interval,neighbors)
        selection_seconds+=perf_counter()-tick
        updates.append(dict(time=t,modes=n+len(space.selected),quadrature_points=space.integration_points,
            radius=space.radius,mass_defect=space.mass_defect,selected=[space.dictionary.labels[i] for i in space.selected],**info))
        stop=min(t+interval,final_time);tick=perf_counter()
        state,used,bad=advance(state,t,stop,space.value,space.both,space.rate,space.radius,mu,kappa,gamma,cfl)
        integration_seconds+=perf_counter()-tick;t=stop;steps+=used;rejected+=bad
        if progress:print(f'residual BSPF t={t:.5f}, modes={len(state)}, steps={steps}, indicator={info["indicator"]:.3e}',flush=True)
    output=space.at(probes,state);rho,u,p,temp=primitives(output)
    summary=dict(method='BSPF + residual-selected fixed splines' if adaptive else 'ordinary BSPF',
        n=n,levels=list(levels),degree=degree,detail_order=detail_order,step_level=step_level,tolerance=tolerance,interval=interval,mu=mu,prandtl=prandtl,freeze=freeze,
        gamma=gamma,time=t,cfl=cfl,quadrature=quadrature,probe_order=probe_order,neighbors=neighbors,
        candidate_count=len(space.dictionary.labels),residual_probe_points=len(space.x),
        initial_modes=len(initial_state),final_modes=len(state),max_modes=max_modes,
        min_density=float(rho.min()),min_pressure=float(p.min()),steps=steps,rejected_stages=rejected,
        minimum_relative_pivot=space.minimum_relative_pivot,setup_seconds=setup,
        selection_seconds=selection_seconds,integration_seconds=integration_seconds,
        seconds=perf_counter()-start,updates=updates,
        note='Dense 1D Galerkin prototype, not an enriched FC collocation implementation or a demonstrated scalable backend.')
    return dict(summary=summary,x=probes,output=output,initial=initial_output)
