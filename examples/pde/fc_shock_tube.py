# SPDX-License-Identifier: GPL-3.0-or-later
"""FC-Gram/RK4 physical Navier--Stokes in a rigid, insulated shock tube.

Momentum is zero at each wall; total mass and energy fluxes are zero there.
Boundary density and energy evolve. This is a local FC spatial adaptation,
not a reproduction of the original authors' complete NS solver.
"""
from time import perf_counter
import numpy as np
from prior_art_traveling_shock import FCGram
from shock_tube_1d import initial_state,primitives,NonphysicalState
from compiled_galerkin_step import physical_flux,bounds


class FCShockTube:
    def __init__(self,n,data,mu=1e-4,prandtl=.72,gamma=1.4,filter_order=0):
        if n<17 or mu<=0 or prandtl<=0 or gamma<=1:
            raise ValueError('n>=17 and positive physical transport coefficients required')
        self.fc=FCGram(n,data,filter_order)
        self.x=np.linspace(0,1,n)
        self.mu,self.prandtl,self.gamma=mu,prandtl,gamma
        self.kappa=mu*gamma/((gamma-1)*prandtl)

    def derivative(self,values):
        # Analytic differentiation of the removed constant is exactly zero.
        # This avoids amplifying the continuation matrices' constant roundoff.
        return self.fc.derivative(values-values[:1])

    def rhs(self,state):
        state=state.copy();state[[0,-1],1]=0.
        _,density=bounds(state,self.gamma)
        if density<=0:raise NonphysicalState('Nonphysical FC stage')
        gradient=self.derivative(state)
        total=physical_flux(state,gradient,self.mu,self.kappa,self.gamma)
        total[[0,-1],0]=0.;total[[0,-1],2]=0.
        rate=-self.derivative(total)
        rate[[0,-1],1]=0.
        return rate


def solve_fc_tube(n,data,mu=1e-4,prandtl=.72,width=.01,final_time=.1,cfl=1.,
                  filter_order=0,output_points=64001,progress=False):
    if not np.isfinite(final_time+cfl+width) or final_time<=0 or width<=0 or not 0<cfl<=1:
        raise ValueError('Positive finite time/width and 0<cfl<=1 required')
    start=perf_counter();solver=FCShockTube(n,data,mu,prandtl,filter_order=filter_order)
    U=initial_state(solver.x,width);t=0.;steps=0;rejected=0
    initial_integral=np.trapezoid(U,solver.x,axis=0)
    history=[]
    def record():
        rho,_,p,_=primitives(U)
        total=np.trapezoid(U,solver.x,axis=0)
        history.append(dict(time=t,mass=float(total[0]),energy=float(total[2]),
                            min_density=float(rho.min()),min_pressure=float(p.min())))
    record();setup=perf_counter()-start;begin=perf_counter();last_report=begin;next_record=.0025
    while t<final_time-1e-14:
        speed,density=bounds(U,solver.gamma)
        if density<=0:raise NonphysicalState('Nonphysical FC state')
        diffusivity=max(4*mu/3,solver.kappa*(solver.gamma-1))/density
        kmax=np.pi/solver.fc.h
        dt=min(cfl/(speed*kmax/2.83+diffusivity*kmax*kmax/2.7852935634),final_time-t)
        for attempt in range(13):
            try:
                a=solver.rhs(U);b=solver.rhs(U+dt*a/2)
                c=solver.rhs(U+dt*b/2);d=solver.rhs(U+dt*c)
                trial=U+dt*(a+2*b+2*c+d)/6;trial[[0,-1],1]=0.
                _,minimum=bounds(trial,solver.gamma)
                if minimum<=0:raise NonphysicalState('Nonphysical FC update')
                break
            except NonphysicalState:
                rejected+=1;dt*=.5
        else:raise RuntimeError(f'FC positivity rejection failed at t={t}')
        if dt<1e-14:raise RuntimeError('FC step underflow')
        U=trial;t+=dt;steps+=1
        if t>=next_record:
            record();next_record+=.0025
        if progress and perf_counter()-last_report>15:
            print(f'FC tube n={n}: t={t:.6f}, steps={steps}',flush=True);last_report=perf_counter()
    record();integration=perf_counter()-begin
    probes=np.linspace(0,1,output_points);output=solver.fc.at(probes,U)
    rho,u,p,T=primitives(output)
    dense_integral=np.trapezoid(output,probes,axis=0)
    summary=dict(method='FC-Gram shock tube',n=n,fft_points=solver.fc.m,continuation_points=solver.fc.C,
        matching_points=solver.fc.d,mu=mu,prandtl=prandtl,gamma=solver.gamma,width=width,time=t,cfl=cfl,
        filter_order=filter_order,steps=steps,rejected_stages=rejected,setup_seconds=setup,
        integration_seconds=integration,seconds=perf_counter()-start,
        min_density=float(rho.min()),min_pressure=float(p.min()),wall_momentum=U[[0,-1],1].tolist(),
        final_integral=dense_integral.tolist(),
        relative_mass_drift=float((dense_integral[0]-initial_integral[0])/initial_integral[0]),
        relative_energy_drift=float((dense_integral[2]-initial_integral[2])/initial_integral[2]),
        max_recorded_relative_mass_drift=max(abs(h['mass']/initial_integral[0]-1) for h in history),
        max_recorded_relative_energy_drift=max(abs(h['energy']/initial_integral[2]-1) for h in history))
    return dict(summary=summary,x=probes,output=output,grid=solver.x,state=U,history=history)
