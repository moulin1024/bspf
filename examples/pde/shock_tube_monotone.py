# SPDX-License-Identifier: GPL-3.0-or-later
"""Conservative constrained BSPF filtering for the pre-reflection Sod problem.

A mass-norm projection imposes decreasing density/pressure and a single velocity
maximum at sampled constraints. This is deliberately problem-specific and adds
numerical dissipation. It is not a general monotone or entropy-stable NS scheme.
The FFT BSPF RHS and physical viscous terms are unchanged. No FVM is used.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.linalg import lstsq
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import LinearConstraint, minimize
from scipy.signal import find_peaks
from shock_tube_1d import BSPFNavierStokes, initial_state, primitives


class MonotoneLimiter:
    def __init__(self, base, samples_per_cell=8, tolerance=1e-7, temperature_shape=False):
        if samples_per_cell<1 or samples_per_cell*(base.n-1)>8192:
            raise ValueError('initial constraints must fit within the 8193-point audit grid')
        self.base=base
        self.temperature_shape=temperature_shape
        self.x=np.linspace(0,1,8193)
        self.initial_constraints=np.linspace(0,len(self.x)-1,samples_per_cell*(base.n-1)+1,dtype=int)
        self.V=base.line.evaluate(self.x).matrix
        self.G=base.line.evaluate(self.x,derivative=1).matrix
        self.M=base.galerkin.mass(np.eye(base.n))
        self.M=(self.M+self.M.T)/2
        self.tolerance=tolerance
        self.calls=0
        self.iterations=0

    def qp(self, target, A, lower, integral, wall=False):
        free=np.arange(1,self.base.n-1) if wall else np.arange(self.base.n)
        M=self.M[np.ix_(free,free)]*self.base.n
        a=A[:,free]
        scale=np.maximum(np.linalg.norm(a,axis=1),1e-12)
        a=a/scale[:,None]; b=lower/scale
        weights=self.base.integral[free]
        goal=target[free]
        objective=lambda c: .5*(c-goal)@M@(c-goal)
        jac=lambda c: M@(c-goal)
        blocks=len(b)//len(self.x)
        active=np.unique(np.concatenate([self.initial_constraints+j*len(self.x) for j in range(blocks)]+
            [np.arange(blocks*len(self.x),len(b))]))
        guess=goal.copy()
        for exchange in range(15):
            constraints=[LinearConstraint(a[active],b[active],np.inf),LinearConstraint(weights[None,:],integral,integral)]
            result=minimize(objective,guess,jac=jac,constraints=constraints,method='SLSQP',
                options=dict(ftol=1e-12,maxiter=200))
            self.iterations+=result.nit
            c=np.zeros(self.base.n);c[free]=result.x
            violations=lower-A@c
            violation=float(violations.max())
            if violation<2e-7 and abs(self.base.integral@c-integral)<1e-9:
                return c
            peaks=find_peaks(violations,height=1e-7)[0]
            added=np.unique(np.r_[peaks,np.argmax(violations)])
            active=np.union1d(active,added)
            guess=result.x
        raise RuntimeError(f'constraint solve failed: {result.message}; violation={violation:g}')

    def __call__(self, state):
        self.calls+=1
        out=state.copy()
        totals=self.base.integral@state
        size=len(self.x); tol=self.tolerance
        out[:,0]=self.qp(state[:,0],np.vstack((-self.G,self.V[[0,-1]],-self.V[[0,-1]])),
            np.r_[np.full(size,-tol),np.full(2,.125),np.full(2,-1.)],totals[0])
        rho=self.V@out[:,0];rx=self.G@out[:,0]
        tentative=self.V@state
        velocity=tentative[:,1]/np.maximum(tentative[:,0],.05)
        smoothed=gaussian_filter1d(velocity,.25/(self.base.n-1)/(self.x[1]-self.x[0]))
        peak=int(np.argmax(smoothed)) if smoothed.max()>1e-12 else size//2
        peak=np.clip(peak,1,size-2)
        sign=np.where(np.arange(size)<peak,1.,-1.)
        H=self.G-(rx/rho)[:,None]*self.V
        out[:,1]=self.qp(state[:,1],sign[:,None]*H,
            np.full(size,-tol),totals[1],wall=True)
        m=self.V@out[:,1];mx=self.G@out[:,1]
        kinetic=.5*m*m/rho
        kinetic_x=m*mx/rho-.5*m*m*rx/rho**2
        energy_A=[-self.G]
        energy_b=[-kinetic_x-tol/(self.base.gamma-1)]
        if self.temperature_shape:
            temperature=(self.base.gamma-1)*(self.V@state[:,2]-kinetic)/rho
            smooth=gaussian_filter1d(temperature,.5/(self.base.n-1)/(self.x[1]-self.x[0]))
            valleys=find_peaks(-smooth,prominence=.003)[0]
            peaks=find_peaks(smooth,prominence=.003)[0]
            pairs=[(smooth[j]-smooth[i],i,j) for i in valleys for j in peaks if i<j]
            direction=-np.ones(size)
            if pairs:
                _,valley,peak=max(pairs)
                direction[valley:peak]=1.
            energy_A.append(direction[:,None]*H)
            energy_b.append(direction*(kinetic_x-rx/rho*kinetic)-tol*rho/(self.base.gamma-1))
        energy_A += [self.V[[0,-1]],-self.V[[0,-1]]]
        energy_b += [kinetic[[0,-1]]+.1/(self.base.gamma-1),-kinetic[[0,-1]]-1/(self.base.gamma-1)]
        out[:,2]=self.qp(state[:,2],np.vstack(energy_A),np.concatenate(energy_b),totals[2])
        return out


def diagnostics(base, limiter, state, x):
    V=base.line.evaluate(x).matrix
    G=base.line.evaluate(x,derivative=1).matrix
    U=V@state;Ux=G@state
    rho,u,p,T=primitives(U)
    ux=(Ux[:,1]-u*Ux[:,0])/rho
    px=(base.gamma-1)*(Ux[:,2]-u*Ux[:,1]+.5*u*u*Ux[:,0])
    peak=np.argmax(u)
    return dict(min_density=float(rho.min()),min_pressure=float(p.min()),
        max_density_slope=float(Ux[:,0].max()),max_pressure_slope=float(px.max()),
        velocity_shape_violation=float(max(0.,-ux[:peak].min() if peak else 0.,ux[peak+1:].max() if peak<len(x)-1 else 0.)),
        density_positive_variation=float(np.maximum(np.diff(rho),0).sum()),
        pressure_positive_variation=float(np.maximum(np.diff(p),0).sum()),
        velocity_min=float(u.min()),filter_calls=limiter.calls,qp_iterations=limiter.iterations)


def solve(n=65,final_time=.1,mu=.0001,width=.01,cfl=.35,samples_per_cell=8,temperature_shape=False,artificial_diffusion=0.):
    if not np.isfinite(final_time) or final_time<=0 or not 0<cfl<=1:
        raise ValueError('positive finite time and 0 < CFL <= 1 required')
    if not np.isfinite(artificial_diffusion) or artificial_diffusion<0:
        raise ValueError('nonnegative finite artificial diffusivity required')
    start=perf_counter()
    base=BSPFNavierStokes(n,mu=mu,quadrature=12)
    limiter=MonotoneLimiter(base,samples_per_cell,temperature_shape=temperature_shape)
    state=limiter(initial_state(base.x,width))
    initial=base.integral@state
    t=0.;steps=0;filter_seconds=0.;last=perf_counter();drift=np.zeros(2)
    def filtered(value):
        nonlocal filter_seconds
        stamp=perf_counter();value=limiter(value);filter_seconds+=perf_counter()-stamp
        return value
    def rhs(value):
        rate=base.rhs(value)
        if artificial_diffusion:
            load=-artificial_diffusion*base.galerkin.stiffness(value)
            rate[:,(0,2)]+=base.solve_mass(load[:,(0,2)])
            rate[1:-1,1]+=base.solve_mass(load[1:-1,1:2],interior=True)[:,0]
        return rate
    # SSPRK3, with constrained filtering after every stage.
    while t<final_time-1e-14:
        dt=min(1/(1/base.timestep(state,cfl)+artificial_diffusion*base.diffusion_radius/(cfl*2.785293563405282)),final_time-t)
        a=filtered(state+dt*rhs(state))
        b=filtered(.75*state+.25*(a+dt*rhs(a)))
        state=filtered(state/3+2*(b+dt*rhs(b))/3)
        t+=dt;steps+=1
        now=base.integral@state
        drift=np.maximum(drift,abs(now[[0,2]]/initial[[0,2]]-1))
        if perf_counter()-last>15:
            print(f't={t:.5f}, steps={steps}, filter={filter_seconds:.1f}s',flush=True);last=perf_counter()
    x=np.linspace(0,1,8193)
    output=base.line.evaluate(x).matrix@state
    summary=dict(n=n,mu=mu,final_time=t,artificial_diffusion=artificial_diffusion,temperature_shape=temperature_shape,steps=steps,seconds=perf_counter()-start,filter_seconds=filter_seconds,
        max_relative_mass_drift=float(drift[0]),max_relative_energy_drift=float(drift[1]),
        **diagnostics(base,limiter,state,x))
    return dict(x=x,output=output,state=state,summary=summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',type=int,default=65)
    parser.add_argument('--time',type=float,default=.1)
    parser.add_argument('--samples-per-cell',type=int,default=8)
    parser.add_argument('--fit-only',action='store_true')
    parser.add_argument('--temperature-shape',action='store_true',help='experimental: also constrain the temperature to a trough followed by a peak')
    parser.add_argument('--artificial-diffusion',type=float,default=0.)
    parser.add_argument('--prefix',default='benchmarks/shock_tube/monotone')
    args=parser.parse_args()
    ref=np.load('benchmarks/shock_tube/enrichment_n65_mu0001_fine.npz')
    if args.fit_only:
        start=perf_counter();base=BSPFNavierStokes(args.n,mu=.0001,quadrature=12)
        limiter=MonotoneLimiter(base,args.samples_per_cell,temperature_shape=args.temperature_shape)
        V=base.line.evaluate(ref['x']).matrix
        state=lstsq(V,ref['reference'])[0]
        state[[0,-1],1]=0
        state[1:-1,1]=lstsq(V[:,1:-1],ref['reference'][:,1])[0]
        state=limiter(state)
        x=np.linspace(0,1,8193)
        result=dict(x=x,output=base.line.evaluate(x).matrix@state,state=state,
            summary=dict(n=args.n,fit_only=True,seconds=perf_counter()-start,**diagnostics(base,limiter,state,x)))
    else:
        result=solve(args.n,args.time,samples_per_cell=args.samples_per_cell,temperature_shape=args.temperature_shape,artificial_diffusion=args.artificial_diffusion)
    prefix=Path(args.prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(result['summary'],indent=2))
    np.savez(prefix.with_suffix('.npz'),**{k:result[k] for k in ('x','output','state')})
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    curves=[('Constrained BSPF',result['x'],result['output'])]
    if args.fit_only or np.isclose(args.time,.1):
        curves.append(('Reference at t=0.1',ref['x'],ref['reference']))
    for label,x,U in curves:
        for ax,name,v in zip(axes.flat,('Density','Velocity','Pressure','Temperature'),primitives(U)):
            ax.plot(x,v,label=label,lw=1.2);ax.set(title=name,xlabel='x');ax.legend()
    fig.suptitle('Constrained reference fit' if args.fit_only else f'BSPF with stagewise monotonicity filtering, n={args.n}, t={args.time}')
    fig.savefig(prefix.with_suffix('.png'),dpi=160)
    print(json.dumps(result['summary'],indent=2),flush=True)


if __name__=='__main__':
    main()
