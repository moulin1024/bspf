# SPDX-License-Identifier: GPL-3.0-or-later
"""Fused physical NS flux and RK4 for small precompiled Galerkin operators.

No numerical flux, artificial viscosity, filtering or finite-volume update.
The matrix R is the constrained Galerkin mass inverse times weak projection.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def physical_flux(values,gradient,mu,kappa,gamma):
    out=np.empty_like(values)
    for i in range(len(values)):
        rho,m,e=values[i];u=m/rho
        p=(gamma-1)*(e-.5*m*u)
        ux=(gradient[i,1]-u*gradient[i,0])/rho
        tx=(gamma-1)*((gradient[i,2]-e*gradient[i,0]/rho)/rho-u*ux)
        tau=4*mu*ux/3
        out[i,0]=m
        out[i,1]=m*u+p-tau
        out[i,2]=u*(e+p)-u*tau-kappa*tx
    return out


@njit(cache=True)
def bounds(values,gamma):
    speed=0.;density=np.inf
    for i in range(len(values)):
        rho,m,e=values[i]
        if rho<=0 or not np.isfinite(rho):return -1.,-1.
        u=m/rho;p=(gamma-1)*(e-.5*m*u)
        if p<=0 or not np.isfinite(rho+p+u):return -1.,-1.
        speed=max(speed,abs(u)+np.sqrt(gamma*p/rho));density=min(density,rho)
    return speed,density


@njit(cache=True)
def rhs(state,both,rate,mu,kappa,gamma):
    samples=both@state;q=len(samples)//2
    speed,density=bounds(samples[:q],gamma)
    if density<=0:return np.zeros_like(state),False
    return rate@physical_flux(samples[:q],samples[q:],mu,kappa,gamma),True


@njit(cache=True)
def advance(state,start,stop,value,both,rate,radius,mu,kappa,gamma,cfl):
    state=state.copy();t=start;steps=0;rejected=0
    samples=value@state
    while t<stop-1e-14:
        if steps>=100000:raise ValueError('Excessive Galerkin steps; check quadrature and stability radius')
        speed,density=bounds(samples,gamma)
        if density<=0:raise ValueError('Nonphysical Galerkin state')
        diffusion=max(4*mu/3,kappa*(gamma-1))/density
        dt=min(cfl/(speed*np.sqrt(radius)/2.83+diffusion*radius/2.785293563405282),stop-t)
        accepted=False
        for attempt in range(13):
            k1,ok1=rhs(state,both,rate,mu,kappa,gamma)
            k2,ok2=rhs(state+dt*k1/2,both,rate,mu,kappa,gamma)
            k3,ok3=rhs(state+dt*k2/2,both,rate,mu,kappa,gamma)
            k4,ok4=rhs(state+dt*k3,both,rate,mu,kappa,gamma)
            trial=state+dt*(k1+2*k2+2*k3+k4)/6
            trial_samples=value@trial
            _,rho=bounds(trial_samples,gamma)
            if ok1 and ok2 and ok3 and ok4 and rho>0:
                accepted=True;break
            dt*=.5;rejected+=1
        if not accepted or dt<1e-14:raise ValueError('Galerkin RK4 step failed')
        state=trial;samples=trial_samples;t+=dt;steps+=1
    return state,steps,rejected


def compiled_segment(space,state,start,stop,shock,cfl):
    """Use the fused kernel only for stationary prescribed endpoint states."""
    endpoints=np.array([0.,1.])
    for time in (start,(start+stop)/2,stop):
        if (np.max(abs(shock.state(endpoints,time)-state[space.walls]))>1e-14 or
                np.max(abs(shock.time_derivative(endpoints,time)))>1e-14):
            raise ValueError('Compiled segment requires stationary endpoint data')
    e=space.enrichment
    return advance(state,start,stop,e.compiled_values,e.compiled_both,e.compiled_rate,
                   space.radius,shock.mu,shock.kappa,shock.gamma,cfl)


def warmup():
    """Explicit one-time JIT preparation, excluded from warm benchmark times."""
    state=np.array([[1.,0.,3.],[1.,0.,3.]])
    value=np.eye(2);both=np.vstack((value,np.zeros_like(value)));rate=np.zeros((2,2))
    advance(state,0.,.001,value,both,rate,1.,.001,.004,1.4,.7)
