# SPDX-License-Identifier: GPL-3.0-or-later
"""BSPF shock tube with explicit conservative artificial diffusion.

Adds epsilon * U_xx to all three conservative equations. This is numerical
regularization, including artificial mass diffusion, not physical viscosity.
Natural zero artificial flux for density/energy; fixed zero wall momentum.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from shock_tube_1d import BSPFNavierStokes, primitives, initial_state
from shock_tube_monotone import MonotoneLimiter, diagnostics


class RegularizedBSPF(BSPFNavierStokes):
    def __init__(self,n=65,mu=.0001,epsilon=.005):
        if not np.isfinite(epsilon) or epsilon<0:
            raise ValueError('nonnegative artificial diffusivity required')
        super().__init__(n,mu=mu,quadrature=12)
        self.epsilon=epsilon
        self.initial_limiter=MonotoneLimiter(self)

    def initial_coefficients(self,width):
        return self.initial_limiter(initial_state(self.x,width,self.gamma))

    def rhs(self,state):
        rate=super().rhs(state)
        load=-self.epsilon*self.galerkin.stiffness(state)
        rate[:,(0,2)]+=self.solve_mass(load[:,(0,2)])
        rate[1:-1,1]+=self.solve_mass(load[1:-1,1:2],interior=True)[:,0]
        return rate

    def timestep(self,state,cfl):
        rho,u,p,_=primitives(self.evaluate(state),self.gamma)
        speed=np.max(abs(u)+np.sqrt(self.gamma*p/rho))
        diffusion=max(4*self.mu/3,self.kappa*(self.gamma-1))/rho.min()+self.epsilon
        return cfl/(speed*np.sqrt(self.diffusion_radius)/2.83+diffusion*self.diffusion_radius/2.785293563405282)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epsilons',type=float,nargs='+',default=[.002,.005,.01])
    parser.add_argument('--prefix',default='benchmarks/shock_tube/regularization')
    args=parser.parse_args()
    ref=np.load('benchmarks/shock_tube/enrichment_n65_mu0001_fine.npz')
    results={}
    for epsilon in args.epsilons:
        start=perf_counter();solver=RegularizedBSPF(epsilon=epsilon)
        result=solver.solve(.1,.01,.35,progress=True)
        result['summary']['artificial_diffusivity']=epsilon
        result['summary']['total_seconds']=perf_counter()-start
        result['audit']=diagnostics(solver,solver.initial_limiter,result['state'],np.linspace(0,1,32769))
        results[str(epsilon)]=result
        print(epsilon,result['summary']['total_seconds'],result['audit'],flush=True)
    prefix=Path(args.prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps({k:{s:v[s] for s in ('summary','audit')} for k,v in results.items()},indent=2))
    np.savez(prefix.with_suffix('.npz'),x=ref['x'],**{k:v['output'] for k,v in results.items()},
        **{k+'_state':v['state'] for k,v in results.items()})
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for name,output in [('Original BSPF',ref['base'])]+[(f'Artificial diffusion {k}',v['output']) for k,v in results.items()]+[('Reference',ref['reference'])]:
        for ax,title,v in zip(axes.flat,('Density','Velocity','Pressure','Temperature'),primitives(output)):
            ax.plot(ref['x'],v,label=name,lw=1.1);ax.set(title=title,xlabel='x',xlim=(.3,.85));ax.legend(fontsize=8)
    fig.suptitle('65-mode BSPF: conservative artificial diffusion, mu=1e-4, t=0.1')
    fig.savefig(prefix.with_suffix('.png'),dpi=160)


if __name__=='__main__':
    main()
