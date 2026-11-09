# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact Becker traveling shock and ordinary/adaptive BSPF convergence.

Constant dynamic viscosity, ideal gas R=1, gamma=1.4, Pr=3/4, compression 2.
Source: B. M. Johnson, Closed-form shock solutions, arXiv:1403.6754, Eq. 3.3.
All three endpoint states are prescribed from the exact traveling profile;
these are open-flow verification boundaries, not closed shock-tube walls.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
from scipy.interpolate import BSpline
import numpy as np
from shock_tube_1d import BSPFNavierStokes, NonphysicalState, fluxes, primitives
from shock_tube_enrichment import detect
from shock_tube_spline_enrichment import LocalRefinement, SparseSplineSpace


class BeckerShock:
    def __init__(self, mu=.001, speed=.5, center=.4):
        if not all(np.isfinite(v) for v in (mu,speed,center)) or mu <= 0:
            raise ValueError('finite parameters and positive viscosity required')
        self.mu, self.speed, self.center = mu, speed, center
        self.gamma, self.prandtl = 1.4, .75
        self.cp = self.gamma/(self.gamma-1)
        self.v0 = np.sqrt(4*self.gamma/(3-self.gamma))  # rho0=T0=1
        self.v1 = self.v0/2
        self.mass_flux = self.v0
        self.enthalpy = self.cp+self.v0**2/2
        self.kappa = mu*self.cp/self.prandtl
        self.w = 8*mu*self.gamma/(3*self.mass_flux*(self.gamma+1))

    def state_gradient(self, x, time=0.):
        xi = np.asarray(x)-self.center-self.speed*time
        # eta=(1+q)/2. Becker's implicit relation, centered at q=1/2:
        # exp(xi/w)=2*(1-q)^2/q. Stable analytic quadratic root.
        log_a = xi/self.w-np.log(2.)
        q = np.exp(-2*np.arcsinh(.5*np.exp(np.minimum(log_a/2, 350))))
        q = np.where(log_a > 700, np.exp(-np.maximum(log_a, 0)), q)
        v = self.v1+(self.v0-self.v1)*q
        vx = -(self.v0-self.v1)*q*(1-q)/(self.w*(1+q))
        rho = self.mass_flux/v
        rhox = -rho*vx/v
        temperature = (self.enthalpy-v*v/2)/self.cp
        tx = -v*vx/self.cp
        velocity = v+self.speed
        specific_energy = temperature/(self.gamma-1)+velocity**2/2
        state = np.column_stack((rho, rho*velocity, rho*specific_energy))
        gradient = np.column_stack((rhox, rhox*velocity+rho*vx,
            rhox*specific_energy+rho*(tx/(self.gamma-1)+velocity*vx)))
        return state, gradient

    def state(self, x, time=0.):
        return self.state_gradient(x,time)[0]

    def time_derivative(self, x, time=0.):
        return -self.speed*self.state_gradient(x,time)[1]


class ShockSpace:
    """Identical Galerkin evolution and prescribed-boundary treatment for both spaces."""
    def __init__(self, base, enrichment=None):
        self.base, self.enrichment = base, enrichment
        self.n = base.n
        self.k = 0 if enrichment is None else enrichment.k
        self.evaluate = base.evaluate if enrichment is None else enrichment.evaluate
        self.project = base.project if enrichment is None else enrichment.project
        self.mass = base.solve_mass if enrichment is None else enrichment.mass
        self.lift = base.wall_lift if enrichment is None else enrichment.wall_lift
        self.radius = base.diffusion_radius if enrichment is None else enrichment.radius
        self.walls = [0, base.n-1]
        self._probes = None

    def at(self, x, state):
        if self.enrichment is not None:
            return self.enrichment.at(x,state)
        if hasattr(self.base,'at'):
            return self.base.at(x,state)
        if len(x) > 8192:
            return np.vstack([self.base.line.evaluate(x[start:start+4096]).matrix@state
                              for start in range(0,len(x),4096)])
        if self._probes is None or not np.array_equal(x,self._probes[0]):
            self._probes = (x.copy(),self.base.line.evaluate(x).matrix)
        return self._probes[1]@state

    def fit_samples(self, samples, boundary):
        state = self.mass(self.project(samples))
        state += self.lift@(boundary-state[self.walls])
        state[self.walls] = boundary
        return state

    def fit(self, shock):
        return self.fit_samples(shock.state(self.base.q),shock.state(np.array([0.,1.])))

    def rhs(self, state, time, shock):
        state = state.copy()
        state[self.walls] = shock.state(np.array([0.,1.]),time)
        if self.enrichment is None:
            values, gradient = self.evaluate(state), self.evaluate(state,1)
        else:
            values, gradient = self.enrichment.evaluate_with_gradient(state)
        convective, viscous = fluxes(values,gradient,shock.mu,shock.kappa,shock.gamma)
        # Interior test functions vanish at both boundaries. Constraining the
        # full inverse removes endpoint load components, including flux terms.
        if self.enrichment is not None and hasattr(self.enrichment,'constrained_flux_rate'):
            rate=self.enrichment.constrained_flux_rate(convective-viscous)
        else:
            rate = self.mass(self.project(convective-viscous,1))
        boundary_rate = shock.time_derivative(np.array([0.,1.]),time)
        rate += self.lift@(boundary_rate-rate[self.walls])
        rate[self.walls] = boundary_rate
        return rate


def spline_space_type(degree):
    if degree < 3 or degree % 2 != 1:
        raise ValueError('odd spline degree >=3 required')
    if degree == 5:
        return SparseSplineSpace
    support=(degree+1)//2
    class HigherOrderSpace(SparseSplineSpace):
        half_support=support
        shape=BSpline.basis_element(np.arange(-support,support+1,dtype=float),extrapolate=False)
        shape_x=shape.derivative()
    return HigherOrderSpace


def solve(n=65, enriched=False, mu=.001, final_time=.05, cfl=.35,
          quadrature=32, refinement=8, interval=.005, speed=.5, halfwidth=2., progress=False, output_points=4001, line_options=None, spline_degree=5, recondition=False,
          selection='detected', fixed_centers=None, base_factory=None, enrichment_factory=None, segment_stepper=None,
          detector_spacing=None):
    if selection not in ('detected', 'fixed', 'oracle'):
        raise ValueError('selection must be detected, fixed, or oracle')
    if fixed_centers is not None and (selection != 'fixed' or not len(fixed_centers)
            or not np.all(np.isfinite(fixed_centers)) or np.any(np.asarray(fixed_centers) <= 0)
            or np.any(np.asarray(fixed_centers) >= 1)):
        raise ValueError('fixed_centers requires fixed selection and interior finite centers')
    if not all(np.isfinite(v) for v in (final_time,cfl,interval)) or final_time <= 0 or interval <= 0 or not 0 < cfl <= 1:
        raise ValueError('positive finite time/interval and 0<cfl<=1 required')
    if not isinstance(output_points, (int, np.integer)) or output_points < 2:
        raise ValueError('output_points must be an integer >=2')
    if detector_spacing is not None and (not np.isfinite(detector_spacing) or detector_spacing<=0):
        raise ValueError('detector_spacing must be positive and finite')
    start = perf_counter()
    shock = BeckerShock(mu=mu,speed=speed)
    base = (BSPFNavierStokes if base_factory is None else base_factory)(n,mu=mu,prandtl=shock.prandtl,quadrature=quadrature,line_options=line_options)
    probes = np.linspace(0,1,4001)
    factory = enrichment_factory or LocalRefinement(refinement,halfwidth,space_type=spline_space_type(spline_degree),retain_directions=not recondition)
    def adapt(samples, time=0.):
        if selection == 'oracle':
            # Diagnostic control only: identical detector applied to exact data.
            samples = shock.state(probes, time)
        centers,widths = detect(probes,samples,1/(n-1) if detector_spacing is None else detector_spacing,count=1)
        if fixed_centers is not None:
            centers, widths = list(fixed_centers), [1/(n-1)]*len(fixed_centers)
        if not centers:
            raise RuntimeError('front detector found no peak')
        enrichment=factory(base,centers,widths)
        return ShockSpace(enrichment.base,enrichment), centers
    if enriched:
        space, centers = adapt(shock.state(probes))
    else:
        space, centers = ShockSpace(base), []
    state = space.fit(shock)
    output_probes = np.linspace(0,1,output_points)
    initial_output = space.at(output_probes,state)
    initial_modes = n+space.k
    updates = [dict(time=0.,modes=initial_modes,centers=centers)]
    t, steps, rejections = 0.,0,0
    next_update = interval
    last_report = perf_counter()
    while t < final_time-1e-14:
        if segment_stepper is not None:
            target=min(final_time,next_update)
            state,used,rejected=segment_stepper(space,state,t,target,shock,cfl)
            t=target;steps+=used;rejections+=rejected
        else:
            rho,u,p,_ = primitives(space.evaluate(state))
            speed_bound = np.max(abs(u)+np.sqrt(shock.gamma*p/rho))
            diffusion = max(4*mu/3,shock.kappa*(shock.gamma-1))/rho.min()
            dt = min(cfl/(speed_bound*np.sqrt(space.radius)/2.83+diffusion*space.radius/2.785293563405282),
                     final_time-t,next_update-t)
            for attempt in range(13):
                try:
                    k1=space.rhs(state,t,shock)
                    k2=space.rhs(state+dt*k1/2,t+dt/2,shock)
                    k3=space.rhs(state+dt*k2/2,t+dt/2,shock)
                    k4=space.rhs(state+dt*k3,t+dt,shock)
                    trial=state+dt*(k1+2*k2+2*k3+k4)/6
                    trial[space.walls]=shock.state(np.array([0.,1.]),t+dt)
                    primitives(space.evaluate(trial))
                    break
                except NonphysicalState:
                    rejections += 1
                    dt *= .5
            else:
                raise RuntimeError('nonphysical state after repeated step rejection')
            if dt < 1e-14:
                raise RuntimeError('time step underflow')
            state,t,steps=trial,t+dt,steps+1
        if progress and perf_counter()-last_report > 15:
            print(f'n={n}, enriched={enriched}: t={t:.6f}, steps={steps}, modes={n+space.k}',flush=True)
            last_report=perf_counter()
        if t >= next_update-1e-13:
            if enriched and selection != 'fixed' and t < final_time-1e-13:
                candidate,centers=adapt(space.at(probes,state), t)
                if recondition and candidate.enrichment is not space.enrichment:
                    samples=space.evaluate(state) if candidate.base is space.base else space.at(candidate.base.q,state)
                    remapped=candidate.fit_samples(samples,shock.state(np.array([0.,1.]),t))
                    remap_error=float(abs(candidate.evaluate(remapped)-samples).max())
                    state=remapped
                else:
                    state=candidate.enrichment.transfer_from(space.enrichment,state)
                    remap_error=0.
                space=candidate
                updates.append(dict(time=t,modes=n+space.k,centers=centers,remap_error=remap_error,
                    max_transform=float(abs(space.enrichment.transform).max())))
            next_update += interval
    probes=output_probes
    output=space.at(probes,state)
    exact=shock.state(probes,t)
    def metrics(values, target):
        fields={}
        for key,v,r in zip(('rho','u','p','T'),primitives(values),primitives(target)):
            error=v-r
            fields[key]=dict(l1=float(np.trapezoid(abs(error),probes)),
                l2=float(np.sqrt(np.trapezoid(error**2,probes))),linf=float(abs(error).max()),
                overshoot=float(max(0,v.max()-r.max(),r.min()-v.min())))
        return fields
    # Smooth exact solution supplies the open-domain balance, not constant totals.
    initial_total=space.base.weights@shock.state(space.base.q,0)
    final_total=space.base.weights@space.evaluate(state)
    exact_total=space.base.weights@shock.state(space.base.q,t)
    summary=dict(n=n,enriched=enriched,mu=mu,prandtl=shock.prandtl,speed=speed,center=shock.center,time=t,cfl=cfl,
        quadrature=quadrature,refinement=refinement,halfwidth=halfwidth,spline_degree=spline_degree,recondition=recondition,
        selection=selection,fixed_centers=None if fixed_centers is None else list(fixed_centers),
        detector_spacing=detector_spacing,
        line_options=base.line_options,
        quadrature_points=len(space.base.q),
        fit_amplification=base.galerkin.fit_amplification,output_points=output_points,initial_modes=initial_modes,
        final_modes=n+space.k,steps=steps,rejected_stages=rejections,
        seconds=perf_counter()-start,metrics=metrics(output,exact),
        initial_metrics=metrics(initial_output,shock.state(probes)),
        integral_error=(final_total-exact_total).tolist(),
        exact_integral_change=(exact_total-initial_total).tolist(),updates=updates)
    return dict(summary=summary,x=probes,output=output,exact=exact,initial=initial_output,
                coefficients=state,centers=np.array([]) if space.enrichment is None else space.enrichment.centers,
                correction=np.empty((0,0)) if space.enrichment is None else space.enrichment.correction,
                enrichment_widths=np.array([]) if space.enrichment is None else space.enrichment.widths,
                enrichment_knots=np.array([]) if space.enrichment is None else getattr(space.enrichment,'knots',np.array([])),
                transform=np.empty((0,0)) if space.enrichment is None else space.enrichment.transform)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',type=int,default=65)
    parser.add_argument('--enriched',action='store_true')
    parser.add_argument('--mu',type=float,default=.001)
    parser.add_argument('--time',type=float,default=.05)
    parser.add_argument('--cfl',type=float,default=.35)
    parser.add_argument('--quadrature',type=int,default=32)
    parser.add_argument('--prefix',default='benchmarks/traveling_shock/example')
    args=parser.parse_args()
    r=solve(args.n,args.enriched,args.mu,args.time,args.cfl,args.quadrature)
    prefix=Path(args.prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(r['summary'],indent=2))
    np.savez(prefix.with_suffix('.npz'),**{k:v for k,v in r.items() if k!='summary'})
    print(json.dumps(r['summary'],indent=2))


if __name__=='__main__':main()
