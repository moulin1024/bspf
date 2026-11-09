# SPDX-License-Identifier: GPL-3.0-or-later
"""Multiple fixed spline blocks for a rigid insulated viscous shock tube.

Selection uses state and scaled physical-NS-rate approximation residuals.
There is no front detector, trajectory input, numerical reference feedback,
filter, artificial viscosity or finite-volume blend. Dense small operators
are a 1D prototype optimization, not a scalable 3D implementation.
"""
from time import perf_counter
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import cho_factor,cho_solve
from scipy.sparse import csr_matrix
from numba import njit
from shock_tube_1d import BSPFNavierStokes,initial_state,primitives
from shock_tube_enrichment import EnrichedSpace
from shock_tube_spline_enrichment import SparseSplineSpace
from graded_spline_shock import SplineQuadratureBase
from dictionary_spline_shock import BlockDictionary
from compiled_galerkin_step import physical_flux,bounds


class SplineBlock:
    def __init__(self,center,width,intervals=38,degree=13,core_scale=.25,support_width=8.,core_padding=1.):
        self.center,self.width,self.degree=center,width,degree
        a=np.arcsinh(support_width/core_scale);scale=width*core_scale
        edges=center+scale*np.sinh(np.linspace(-a,a,intervals+1))
        if core_padding:
            h=scale*np.sinh(2*a/intervals);half=core_padding*width
            edges=np.r_[edges[edges<center]-half,np.linspace(center-half,center+half,int(np.ceil(2*half/h))+1),edges[edges>center]+half]
        self.edges=edges;self.support=(edges[0],edges[-1])
        if edges[0]<=0 or edges[-1]>=1:raise ValueError('Only interior blocks supported')
        self.knots=np.r_[np.repeat(edges[0],degree),edges,np.repeat(edges[-1],degree)]
        self.count=len(self.knots)-degree-1;self.indices=np.arange(degree,self.count-degree)
        self.step=BSpline(self.knots,(np.arange(self.count)>=self.count//2).astype(float),degree,extrapolate=False)
        self.basis=BSpline(self.knots,np.eye(self.count),degree,extrapolate=False)
        self.k=len(self.indices)+1

    def raw(self,x,derivative=0):
        lo,hi=self.support;inside=(x>=lo)&(x<=hi);v=np.zeros((len(x),self.k))
        if derivative==0:v[:,0]=(x>hi)-x
        elif derivative==1:v[:,0]=-1.
        v[inside,0]+=self.step(x[inside],nu=derivative)
        v[inside,1:]=self.basis(x[inside],nu=derivative)[:,self.indices]
        return v

    def value(self,x,coef,derivative=0):
        lo,hi=self.support;inside=(x>=lo)&(x<=hi)
        step=np.zeros(len(x))
        if derivative==0:step=(x>hi)-x
        elif derivative==1:step.fill(-1.)
        step[inside]+=self.step(x[inside],nu=derivative)
        v=step[:,None]*coef[0]
        c=np.zeros((self.count,coef.shape[1]));c[self.indices]=coef[1:]
        v[inside]+=BSpline(self.knots,c,self.degree,extrapolate=False)(x[inside],nu=derivative)
        return v


class TubeSpace(EnrichedSpace):
    def __init__(self,template,labels,quadrature=8,**options):
        self.blocks=[SplineBlock(*label,**options) for label in labels]
        edges=np.unique(np.concatenate([b.edges for b in self.blocks]))
        edges=np.unique(np.r_[edges,template.x[(template.x>edges[0])&(template.x<edges[-1])]])
        base=SplineQuadratureBase(template,edges,quadrature)
        centers=np.concatenate([np.full(b.k,b.center) for b in self.blocks])
        widths=np.concatenate([np.full(b.k,b.width) for b in self.blocks])
        super().__init__(base,centers,widths,orthogonalize=True)
        loads=np.zeros((self.n+self.k,2));loads[[0,self.n-1]]=np.eye(2)
        response=self.mass(loads)
        self.wall_lift=np.ascontiguousarray(np.linalg.solve(response[[0,self.n-1]].T,response.T).T)
        value=np.column_stack((base.evaluate(np.eye(self.n)),self.H))
        gradient=np.column_stack((base.evaluate(np.eye(self.n),1),self.Hx))
        self.value_matrix=np.ascontiguousarray(value)
        self.both=np.ascontiguousarray(np.vstack((value,gradient)))
        self.rate=np.ascontiguousarray(self.mass(gradient.T*base.weights))
        self.labels=tuple(labels)

    def raw(self,x):
        v=np.column_stack([b.raw(x) for b in self.blocks]);g=np.column_stack([b.raw(x,1) for b in self.blocks])
        self.S,self.Sx=csr_matrix(v),csr_matrix(g);self.ST,self.SxT=self.S.T,self.Sx.T
        return v,g

    _radius_operators=SparseSplineSpace._radius_operators

    def at(self,x,state,derivative=0):
        z=self.transform@state[self.n:];b=state[:self.n]-self.correction@z
        if derivative==0:value=self.base.at(x,b)
        else:
            cached=getattr(self.base,'probe_operators',None)
            p=cached[derivative] if cached is not None and x is cached[0] else self.base.line.evaluate(x,derivative=derivative).matrix
            value=p@b
        offset=0
        for block in self.blocks:
            value+=block.value(x,z[offset:offset+block.k],derivative);offset+=block.k
        return value

    def fit_fields(self,values):
        load=self.project(values);state=self.mass(load)
        for field in range(1,values.shape[1],3):
            state[:,field]-=self.wall_lift@state[[0,self.n-1],field]
            state[[0,self.n-1],field]=0.
        return state


@njit(cache=True)
def strong_rate(U,G,H,mu,kappa,gamma):
    result=np.empty_like(U)
    for i in range(len(U)):
        r,m,e=U[i];rx,mx,ex=G[i];rxx,mxx,exx=H[i]
        u=m/r;ux=(mx-u*rx)/r;uxx=(mxx-u*rxx-2*rx*ux)/r
        p=(gamma-1)*(e-.5*m*u);px=(gamma-1)*(ex-.5*mx*u-.5*m*ux)
        pxx=(gamma-1)*(exx-.5*mxx*u-mx*ux-.5*m*uxx)
        T=p/r;Tx=(px-T*rx)/r;Txx=(pxx-T*rxx-2*rx*Tx)/r
        tau=4*mu*ux/3;taux=4*mu*uxx/3
        result[i,0]=-mx
        result[i,1]=-(mx*u+m*ux+px-taux)
        result[i,2]=-(ux*(e+p)+u*(ex+px)-ux*tau-u*taux-kappa*Txx)
    return result


@njit(cache=True)
def tube_rhs(state,both,rate,lift,n,mu,kappa,gamma):
    values=both@state;q=len(values)//2
    _,minimum=bounds(values[:q],gamma)
    if minimum<=0:return np.zeros_like(state),False
    result=rate@physical_flux(values[:q],values[q:],mu,kappa,gamma)
    left=result[0,1];right=result[n-1,1]
    for j in range(len(result)):result[j,1]-=lift[j,0]*left+lift[j,1]*right
    result[0,1]=0.;result[n-1,1]=0.
    return result,True


@njit(cache=True)
def advance_tube(state,start,stop,value,both,rate,lift,n,radius,mu,kappa,gamma,cfl):
    state=state.copy();t=start;steps=0;rejected=0
    while t<stop-1e-14:
        speed,density=bounds(value@state,gamma)
        if density<=0:raise ValueError('Nonphysical tube state')
        diffusion=max(4*mu/3,kappa*(gamma-1))/density
        dt=min(cfl/(speed*np.sqrt(radius)/2.83+diffusion*radius/2.7852935634),stop-t)
        accepted=False
        for attempt in range(13):
            a,oa=tube_rhs(state,both,rate,lift,n,mu,kappa,gamma)
            b,ob=tube_rhs(state+dt*a/2,both,rate,lift,n,mu,kappa,gamma)
            c,oc=tube_rhs(state+dt*b/2,both,rate,lift,n,mu,kappa,gamma)
            d,od=tube_rhs(state+dt*c,both,rate,lift,n,mu,kappa,gamma)
            trial=state+dt*(a+2*b+2*c+d)/6
            _,minimum=bounds(value@trial,gamma)
            if oa and ob and oc and od and minimum>0:accepted=True;break
            dt*=.5;rejected+=1
        if not accepted or dt<1e-14:raise ValueError('Tube RK4 failed')
        state=trial;t+=dt;steps+=1
        if steps>100000:raise ValueError('Excessive tube steps')
    return state,steps,rejected


class Selector:
    def __init__(self,base,*,spacing=1/1024,widths=(1/32,1/64,1/128,1/256,1/512),probe_cells=512,
                 intervals=38,degree=13,core_padding=1.,quadrature=8,max_blocks=3,tolerance=1e-6):
        self.base=base;self.max_blocks=max_blocks;self.tolerance=tolerance;self.quadrature=quadrature
        self.dictionary=BlockDictionary(base,spacing,widths,probe_cells,intervals,degree,core_padding=core_padding)
        self.options=self.dictionary.options;self.x=self.dictionary.x;self.w=self.dictionary.weights
        self.P=base.line.evaluate(self.x).matrix
        base.probe_operators=(self.x,base.line.evaluate(self.x,derivative=1).matrix,base.line.evaluate(self.x,derivative=2).matrix)
        mass=self.P.T@(self.w[:,None]*self.P);self.factor=cho_factor(mass)
        self.inner_factor=cho_factor(mass[1:-1,1:-1])

    def select(self,target,previous=None):
        data=target(self.x);best_error=np.inf;trials=0
        if previous is not None:
            space=previous
            fitted=space.fit_fields(target(space.base.q))
            residual=data-space.at(self.x,fitted);best_error=float(np.max(abs(residual)))
            if best_error<=self.tolerance:return space,fitted,dict(error=best_error,trials=0,cap_hit=False)
        load=self.P.T@(self.w[:,None]*data);coef=cho_solve(self.factor,load)
        for field in range(1,data.shape[1],3):
            coef[:,field]=0.;coef[1:-1,field]=cho_solve(self.inner_factor,load[1:-1,field])
        residual=data-self.P@coef;labels=[];best=None
        for number in range(self.max_blocks):
            scores=np.sum((self.dictionary.analysis@residual)**2,axis=1)
            candidates=[]
            for width in sorted(set(self.dictionary.widths),reverse=True):
                valid=self.dictionary.widths==width
                for c,w in labels:
                    if w==width:valid &= abs(self.dictionary.centers-c)>width
                ids=np.flatnonzero(valid)
                if len(ids):candidates.append(int(ids[np.argmax(scores[ids])]))
            best=None
            for index in candidates:
                label=self.dictionary.labels[index]
                space=TubeSpace(self.base,labels+[label],quadrature=self.quadrature,**self.options);trials+=1
                fitted=space.fit_fields(target(space.base.q))
                res=data-space.at(self.x,fitted);error=float(np.max(abs(res)))
                if best is None or error<best[0]:best=(error,space,fitted,res,label)
                if error<=self.tolerance:break
            if best is None:raise RuntimeError('No eligible dictionary block')
            best_error,space,fitted,residual,label=best;labels.append(label)
            if best_error<=self.tolerance:break
        return space,fitted,dict(error=best_error,trials=trials,cap_hit=best_error>self.tolerance)


def warmup():
    U=np.array([[1.,0.,3.],[1.,0.,3.]])
    strong_rate(U,np.zeros_like(U),np.zeros_like(U),.001,.004,1.4)
    advance_tube(U,0.,1e-4,np.eye(2),np.vstack((np.eye(2),np.zeros((2,2)))),np.zeros((2,2)),np.eye(2),2,1.,.001,.004,1.4,1.)


def solve(n=65,mu=1e-4,prandtl=.72,width=.01,final_time=.1,cfl=1.,interval=.001,
          tolerance=1e-6,max_blocks=3,quadrature=8,probe_cells=512,intervals=38,core_padding=1.,
          spacing=1/1024,output_points=64001,progress=False,rate_weight=1.,
          widths=(1/32,1/64,1/128,1/256,1/512)):
    if final_time<=0 or interval<=0 or tolerance<=0 or max_blocks<1 or not 0<cfl<=1:
        raise ValueError('Positive time, interval, tolerance, block count and 0 < CFL <= 1 required')
    start=perf_counter();gamma=1.4;kappa=mu*gamma/((gamma-1)*prandtl)
    base=BSPFNavierStokes(n,mu=mu,prandtl=prandtl,quadrature=8,line_options=dict(n_basis=18))
    selector=Selector(base,spacing=spacing,probe_cells=probe_cells,intervals=intervals,core_padding=core_padding,
        quadrature=quadrature,max_blocks=max_blocks,tolerance=tolerance,widths=widths)
    space,state,info=selector.select(lambda x:initial_state(x,width))
    t=0.;steps=0;rejected=0;updates=[];setup=perf_counter()-start;selection_time=0.;integration_time=0.
    initial_total=space.integral@state
    while t<final_time-1e-14:
        remap=0.;correction=np.zeros(3)
        if t>0:
            tick=perf_counter();old_space=space;old_state=state
            def target(x):
                U=old_space.at(x,old_state)
                if not rate_weight:return U
                R=strong_rate(U,old_space.at(x,old_state,1),old_space.at(x,old_state,2),mu,kappa,gamma)
                return np.column_stack((U,rate_weight*interval*R))
            candidate,fitted,info=selector.select(target,previous=space)
            if candidate is not space:
                totals=space.integral@state;new_state=fitted[:,:3].copy()
                new_totals=candidate.integral@new_state
                correction[[0,2]]=totals[[0,2]]-new_totals[[0,2]]
                new_state[:n]+=correction
                remap=float(np.max(abs(candidate.at(selector.x,new_state)-space.at(selector.x,state))))
                space,state=candidate,new_state
            selection_time+=perf_counter()-tick
        updates.append(dict(time=t,modes=len(state),blocks=space.labels,quadrature_points=len(space.base.q),
            remap_error=remap,integral_correction=correction.tolist(),**info))
        stop=min(t+interval,final_time);tick=perf_counter()
        state,used,bad=advance_tube(state,t,stop,space.value_matrix,space.both,space.rate,space.wall_lift,
            n,space.radius,mu,kappa,gamma,cfl)
        elapsed=perf_counter()-tick
        updates[-1].update(radius=space.radius,steps=used,integration_seconds=elapsed)
        integration_time+=elapsed;t=stop;steps+=used;rejected+=bad
        if progress:print(f'dictionary tube t={t:.5f}, modes={len(state)}, blocks={len(space.blocks)}, steps={steps}, residual={info["error"]:.3e}',flush=True)
    x=np.linspace(0,1,output_points);output=space.at(x,state);rho,u,p,temp=primitives(output)
    total=space.integral@state
    summary=dict(method='BSPF + multiple fixed dictionary blocks',n=n,mu=mu,prandtl=prandtl,gamma=gamma,
        width=width,time=t,cfl=cfl,interval=interval,tolerance=tolerance,max_blocks=max_blocks,
        quadrature=quadrature,probe_cells=probe_cells,selection_points=len(selector.x),intervals=intervals,
        core_padding=core_padding,spacing=spacing,rate_weight=rate_weight,widths=list(widths),
        final_modes=len(state),peak_modes=max(u['modes'] for u in updates),steps=steps,rejected_stages=rejected,
        setup_seconds=setup,selection_seconds=selection_time,integration_seconds=integration_time,
        seconds=perf_counter()-start,min_density=float(rho.min()),min_pressure=float(p.min()),
        relative_mass_drift=float(total[0]/initial_total[0]-1),relative_energy_drift=float(total[2]/initial_total[2]-1),
        max_selection_error=max(u['error'] for u in updates),cap_hits=sum(u['cap_hit'] for u in updates),updates=updates)
    return dict(summary=summary,x=x,output=output)
