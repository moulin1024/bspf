# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixed spline-block dictionary for the analytical NS experiment.

No peak detector, continuous center/width fit, trajectory, or exact interior
feedback. The current prototype selects ONE block by a normalized projection
residual correlation. This tests the 1D position/scale component of directional
dictionary selection, not a 3D shearlet solver or a multiple-front algorithm.
"""
from time import perf_counter
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import cho_factor,cho_solve
from shock_tube_1d import BSPFNavierStokes,primitives
from viscous_traveling_shock import ShockSpace
from graded_spline_shock import GradedSplineSpace
from compiled_galerkin_step import advance


class BlockDictionary:
    def __init__(self,base,spacing=1/512,widths=(1/64,1/128,1/256),probe_cells=512,
                 intervals=38,degree=13,core_scale=.25,support_width=8.,core_padding=.5):
        self.options=dict(intervals=intervals,degree=degree,core_scale=core_scale,
                          support_width=support_width,core_padding=core_padding)
        a=np.arcsinh(support_width/core_scale)
        edges=core_scale*np.sinh(np.linspace(-a,a,intervals+1))
        if core_padding:
            cell=core_scale*np.sinh(2*a/intervals);half=core_padding
            cells=int(np.ceil(2*half/cell))
            edges=np.r_[edges[edges<0]-half,np.linspace(-half,half,cells+1),edges[edges>0]+half]
        knots=np.r_[np.repeat(edges[0],degree),edges,np.repeat(edges[-1],degree)]
        count=len(knots)-degree-1
        template=BSpline(knots,(np.arange(count)>=count//2).astype(float),degree,extrapolate=False)
        self.labels=[(float(c),float(w)) for w in widths
            for c in np.arange(spacing,1,spacing) if c+w*edges[0]>0 and c+w*edges[-1]<1]
        labels=np.asarray(self.labels);self.centers=labels[:,0];self.widths=labels[:,1]
        self.lookup={label:index for index,label in enumerate(self.labels)};self.spacing=spacing
        z,w=np.polynomial.legendre.leggauss(4)
        self.x=((np.arange(probe_cells)[:,None]+(z+1)/2)/probe_cells).ravel()
        self.weights=np.tile(w/(2*probe_cells),probe_cells)
        p=base.line.evaluate(self.x).matrix;interior=p[:,1:-1]
        arg=(self.x[:,None]-self.centers)/self.widths
        raw=template(np.clip(arg,edges[0],edges[-1]))-self.x[:,None]
        cross=interior.T@(self.weights[:,None]*raw)
        correction=cho_solve(cho_factor(interior.T@(self.weights[:,None]*interior)),cross)
        h=raw-interior@correction
        norms=np.sqrt(np.sum(self.weights[:,None]*h*h,axis=0))
        self.analysis=(h*self.weights[:,None]/norms).T
        self.boundary_values=p[:,[0,-1]]

    def select(self,values,boundary):
        # Correlation with the part of each step signature not represented by
        # the homogeneous BSPF background. No gradients or peak locations.
        residual=values-self.boundary_values@boundary
        loads=self.analysis@residual
        scores=np.sum(loads**2,axis=1)
        index=int(np.argmax(scores))
        return index,float(np.sqrt(scores[index]))

    def ranked_scales(self,values,boundary):
        loads=self.analysis@(values-self.boundary_values@boundary)
        scores=np.sum(loads**2,axis=1)
        self.last_scores=scores
        return [(int(ids[np.argmax(scores[ids])]),float(np.sqrt(np.max(scores[ids]))))
            for width in sorted(set(self.widths),reverse=True)
            for ids in [np.flatnonzero(self.widths==width)]]

    def neighbors(self,index):
        center,width=self.labels[index]
        return [index]+[self.lookup[(center+offset*self.spacing,width)]
            for offset in (-1,1,-2,2) if (center+offset*self.spacing,width) in self.lookup]


def solve(initial,boundary,*,n=25,mu=.001,prandtl=.75,gamma=1.4,final_time=.05,cfl=1.,
          interval=.005,spacing=1/512,widths=(1/64,1/128,1/256),probe_cells=512,
          intervals=38,degree=13,quadrature=8,output_points=64001,freeze=False,progress=False,
          fit_tolerance=1e-7,outer_quadrature=8,trace_points=0):
    start=perf_counter()
    base=BSPFNavierStokes(n,mu=mu,prandtl=prandtl,gamma=gamma,quadrature=outer_quadrature,line_options=dict(n_basis=10))
    dictionary=BlockDictionary(base,spacing,widths,probe_cells,intervals,degree)
    def build(index):
        c,w=dictionary.labels[index]
        e=GradedSplineSpace(base,c,w,quadrature=quadrature,compiled=True,**dictionary.options)
        return ShockSpace(e.base,e)
    def fitted_candidate(index,samples):
        candidate=build(index);values=samples(candidate.base.q)
        fitted=candidate.fit_samples(values,boundary)
        error=float(np.max(abs(candidate.evaluate(fitted)-values)))
        return candidate,fitted,error
    initial_trials=[]
    for index,score in dictionary.ranked_scales(initial(dictionary.x),boundary):
        space,state,fit_error=fitted_candidate(index,initial)
        initial_trials.append(dict(index=index,block=dictionary.labels[index],fit_error=fit_error))
        if fit_error<=fit_tolerance:break
    else:raise RuntimeError(f'Dictionary cannot resolve initial data to {fit_tolerance}: {initial_trials}')
    setup=perf_counter()-start
    x=np.linspace(0,1,output_points);initial_output=space.at(x,state)
    initial_modes=len(state);steps=0;rejected=0;t=0.;updates=[]
    trace_x=np.linspace(0,1,trace_points) if trace_points else np.empty(0)
    trace=[space.at(trace_x,state)] if trace_points else [];trace_times=[0.] if trace_points else []
    selection_time=0.;integration_time=0.;rebuild_time=0.
    while t<final_time-1e-14:
        remap_error=0.
        if t>0 and not freeze:
            tick=perf_counter()
            ranked=dictionary.ranked_scales(space.at(dictionary.x,state),boundary)
            selection_time+=perf_counter()-tick
            # Keep the selected scale until it fails the representation check;
            # finer dictionary scales are available without estimating a width.
            ranked=[item for item in ranked if dictionary.labels[item[0]][1]<=dictionary.labels[index][1]]
            tick=perf_counter()
            trials=[j for chosen,_ in ranked for j in dictionary.neighbors(chosen)]
            for new_index in trials:
                if new_index==index:
                    if new_index==ranked[0][0]:
                        remap_error=0.
                        break
                    continue
                candidate,new_state,remap_error=fitted_candidate(new_index,lambda x:space.at(x,state))
                if remap_error<=fit_tolerance:
                    state,space,index=new_state,candidate,new_index
                    break
            else:raise RuntimeError(f'Dictionary transfer error {remap_error} exceeds {fit_tolerance}')
            score=float(np.sqrt(dictionary.last_scores[index]))
            rebuild_time+=perf_counter()-tick
        updates.append(dict(time=t,index=index,block=dictionary.labels[index],score=score,
            modes=len(state),quadrature_points=len(space.base.q),remap_error=remap_error))
        stop=min(t+interval,final_time);tick=perf_counter();e=space.enrichment
        state,used,bad=advance(state,t,stop,e.compiled_values,e.compiled_both,e.compiled_rate,
            space.radius,mu,mu*gamma/((gamma-1)*prandtl),gamma,cfl)
        integration_time+=perf_counter()-tick;t=stop;steps+=used;rejected+=bad
        if trace_points:trace.append(space.at(trace_x,state));trace_times.append(t)
        if progress:print(f'dictionary t={t:.5f}, modes={len(state)}, block={index}, steps={steps}',flush=True)
    output=space.at(x,state);rho,u,p,temp=primitives(output)
    summary=dict(method='BSPF + fixed spline-block dictionary',n=n,mu=mu,prandtl=prandtl,
        gamma=gamma,time=t,cfl=cfl,interval=interval,spacing=spacing,widths=list(widths),
        probe_cells=probe_cells,selection_points=len(dictionary.x),dictionary_blocks=len(dictionary.labels),
        intervals=intervals,degree=degree,quadrature=quadrature,outer_quadrature=outer_quadrature,freeze=freeze,
        initial_modes=initial_modes,final_modes=len(state),steps=steps,rejected_stages=rejected,
        fit_tolerance=fit_tolerance,initial_trials=initial_trials,
        min_density=float(rho.min()),min_pressure=float(p.min()),setup_seconds=setup,
        selection_seconds=selection_time,rebuild_seconds=rebuild_time,integration_seconds=integration_time,
        seconds=perf_counter()-start,updates=updates,
        note='One-block 1D dictionary prototype. Selection is a state-approximation residual correlation, not a PDE-residual certificate. No demonstrated 3D implementation.')
    return dict(summary=summary,x=x,output=output,initial=initial_output,
                trace_x=trace_x,trace_times=np.asarray(trace_times),trace=np.asarray(trace))
