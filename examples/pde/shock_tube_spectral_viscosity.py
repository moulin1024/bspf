# SPDX-License-Identifier: GPL-3.0-or-later
"""Selective high-mode damping of the BSPF Galerkin shock-tube solver.

Prototype: modes are generalized eigenvectors K z = lambda M z, not coefficient
indices. Low modes (including constants) are untouched. Full and wall-constrained
spaces have separate mass-orthonormal bases. Dense 1D eigensolves are for the
experiment; this is not an FFT-scalable 3D stabilization implementation.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.linalg import eigh
from shock_tube_1d import BSPFNavierStokes, primitives


class SpectralViscosityBSPF(BSPFNavierStokes):
    def __init__(self,n=65,mu=.0001,cutoff=.4,rate=300.,power=2.,quadrature=12):
        if not 0<=cutoff<1 or not np.isfinite(rate) or rate<0 or not np.isfinite(power) or power<1:
            raise ValueError('0 <= cutoff < 1, finite rate >= 0 and finite power >= 1 required')
        super().__init__(n,mu=mu,quadrature=quadrature)
        self.cutoff,self.rate,self.power=cutoff,rate,power
        M=self.galerkin.mass(np.eye(n));K=self.galerkin.stiffness(np.eye(n))
        M=(M+M.T)/2;K=(K+K.T)/2
        self.filters={}
        for interior in (False,True):
            m=M[1:-1,1:-1] if interior else M
            k=K[1:-1,1:-1] if interior else K
            _,z=eigh(k,m)
            coordinate=np.linspace(0,1,len(m))
            q=np.maximum((coordinate-cutoff)/(1-cutoff),0)**power
            active=q>0
            self.filters[interior]=(z[:,active],z[:,active].T@m,rate*q[active])

    def damping(self,state):
        result=np.zeros_like(state)
        for cols,interior in [([0,2],False),([1],True)]:
            indices=np.arange(1,self.n-1) if interior else np.arange(self.n)
            z,project,rates=self.filters[interior]
            result[np.ix_(indices,cols)]=-z@(rates[:,None]*(project@state[np.ix_(indices,cols)]))
        return result

    def rhs(self,state):
        return super().rhs(state)+self.damping(state)

    def timestep(self,state,cfl):
        return 1/(1/super().timestep(state,cfl)+self.rate/(cfl*2.785293563405282))


class CompressionViscosityBSPF(SpectralViscosityBSPF):
    """Conservative componentwise viscosity activated by compression only."""
    def __init__(self,*args,strength=.1,**kwargs):
        if not np.isfinite(strength) or strength<0:
            raise ValueError('nonnegative finite compression strength required')
        super().__init__(*args,**kwargs)
        self.strength=strength

    def rhs(self,state):
        rate=super().rhs(state)
        U=self.evaluate(state);gradient=self.evaluate(state,1)
        rho,u,p,_=primitives(U)
        ux=(gradient[:,1]-u*gradient[:,0])/rho
        h=1/(self.n-1)
        speed=abs(u)+np.sqrt(self.gamma*p/rho)
        nu=self.strength*h*np.minimum(h*np.maximum(-ux,0),speed)
        load=-self.project(nu[:,None]*gradient,1)
        rate[:,(0,2)]+=self.solve_mass(load[:,(0,2)])
        rate[1:-1,1]+=self.solve_mass(load[1:-1,1:2],interior=True)[:,0]
        return rate

    def timestep(self,state,cfl):
        rho,u,p,_=primitives(self.evaluate(state))
        nu_bound=self.strength/(self.n-1)*np.max(abs(u)+np.sqrt(self.gamma*p/rho))
        return 1/(1/super().timestep(state,cfl)+nu_bound*self.diffusion_radius/(cfl*2.785293563405282))


def metrics(x,output,reference):
    values=primitives(output); target=primitives(reference)
    result={name:dict(l1=float(np.trapezoid(abs(v-r),x)),linf=float(abs(v-r).max()),
        variation=float(abs(np.diff(v)).sum())) for name,v,r in zip(('rho','u','p','T'),values,target)}
    result['tail_velocity_error']=float(abs(values[1]-target[1])[(x>.70)&(x<.9)].max())
    result['density_undershoot']=float(max(0.,.125-values[0].min()))
    result['pressure_undershoot']=float(max(0.,.1-values[2].min()))
    p=values[2];pr=target[2]
    left=float(np.median(pr[(x>.62)&(x<.65)]));right=float(np.median(pr[(x>.72)&(x<.75)]))
    midpoint_level=(left+right)/2
    reference_crossings=np.flatnonzero((pr[:-1]>=midpoint_level)&(pr[1:]<midpoint_level)&(x[:-1]>.62)&(x[:-1]<.75))
    j=reference_crossings[np.argmax(abs(np.diff(pr)[reference_crossings]))]
    anchor=x[j]+(x[j+1]-x[j])*(pr[j]-midpoint_level)/(pr[j]-pr[j+1])
    locations=[]
    for fraction in (.9,.5,.1):
        level=right+fraction*(left-right)
        ids=np.flatnonzero((p[:-1]>=level)&(p[1:]<level)&(x[:-1]>.62)&(x[:-1]<.75))
        if not len(ids):
            locations.append(None);continue
        candidates=x[ids]+(x[ids+1]-x[ids])*(p[ids]-level)/(p[ids]-p[ids+1])
        locations.append(float(candidates[np.argmin(abs(candidates-anchor))]))
    result['shock_width']=None if None in locations else locations[2]-locations[0]
    result['shock_midpoint']=locations[1]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',type=int,default=65)
    parser.add_argument('--cutoffs',type=float,nargs='+',default=[.2,.4,.6,.8])
    parser.add_argument('--rates',type=float,nargs='+',default=[100.,300.,1000.])
    parser.add_argument('--local-strengths',type=float,nargs='+')
    parser.add_argument('--prefix',default='benchmarks/shock_tube/spectral_viscosity')
    args=parser.parse_args()
    data=np.load('benchmarks/shock_tube/enrichment_n65_mu0001_fine.npz')
    baseline=data['base'] if args.n==65 else BSPFNavierStokes(args.n,mu=.0001,quadrature=12).solve(.1,.01,.35)['output']
    report=dict(parameters=vars(args),baseline=metrics(data['x'],baseline,data['reference']),runs={})
    arrays=dict(x=data['x'],reference=data['reference'],baseline=baseline)
    cases=[(c,r,l) for c in args.cutoffs for r in args.rates for l in (args.local_strengths or [None])]
    for cutoff,rate,strength in cases:
        name=f'cut{cutoff:g}_rate{rate:g}'+('' if strength is None else f'_local{strength:g}')
        stamp=perf_counter()
        cls=SpectralViscosityBSPF if strength is None else CompressionViscosityBSPF
        extra={} if strength is None else dict(strength=strength)
        solver=cls(args.n,cutoff=cutoff,rate=rate,**extra)
        result=solver.solve(.1,.01,.35)
        row=dict(summary=result['summary'],total_seconds=perf_counter()-stamp,
            metrics=metrics(data['x'],result['output'],data['reference']))
        report['runs'][name]=row;arrays[name]=result['output'];arrays[name+'_state']=result['state']
        print(name,json.dumps(row['metrics']),flush=True)
    prefix=Path(args.prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2))
    np.savez(prefix.with_suffix('.npz'),**arrays)


if __name__=='__main__':
    main()
