"""Analytical Becker test of front-free fixed-dictionary spline selection."""
import argparse,json,sys
from pathlib import Path
from time import perf_counter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/pde'))
from residual_spline_shock import solve
from viscous_traveling_shock import BeckerShock,primitives
from compiled_galerkin_step import warmup


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=33)
    p.add_argument('--max-level',type=int,default=10)
    p.add_argument('--min-level',type=int,default=5)
    p.add_argument('--degree',type=int,default=9)
    p.add_argument('--detail-order',type=int,default=4)
    p.add_argument('--step-level',type=int,default=8)
    p.add_argument('--tolerance',type=float,default=1e-7)
    p.add_argument('--interval',type=float,default=.0025)
    p.add_argument('--time',type=float,default=.05)
    p.add_argument('--center',type=float,default=.4)
    p.add_argument('--speed',type=float,default=.5)
    p.add_argument('--mu',type=float,default=.001)
    p.add_argument('--cfl',type=float,default=1.)
    p.add_argument('--quadrature',type=int,default=10)
    p.add_argument('--probe-order',type=int,default=8)
    p.add_argument('--max-modes',type=int,default=256)
    p.add_argument('--neighbors',type=int,default=1)
    p.add_argument('--ordinary',action='store_true')
    p.add_argument('--freeze',action='store_true')
    p.add_argument('--tag',default='trial')
    a=p.parse_args();warmup();shock=BeckerShock(a.mu,a.speed,a.center)
    boundary=shock.state(np.array([0.,1.]))
    np.testing.assert_allclose(boundary,shock.state(np.array([0.,1.]),a.time),atol=1e-14,rtol=0)
    start=perf_counter()
    r=solve(shock.state,boundary,n=a.n,levels=range(a.min_level,a.max_level+1),degree=a.degree,
        tolerance=a.tolerance,interval=a.interval,final_time=a.time,mu=a.mu,cfl=a.cfl,
        quadrature=a.quadrature,probe_order=a.probe_order,max_modes=a.max_modes,
        neighbors=a.neighbors,adaptive=not a.ordinary,progress=True,freeze=a.freeze,detail_order=a.detail_order,step_level=a.step_level)
    r['summary']['end_to_end_seconds']=perf_counter()-start
    exact=shock.state(r['x'],a.time);initial=shock.state(r['x'])
    def error(v,target):
        return {key:dict(linf=float(np.max(abs(x-y))),l2=float(np.sqrt(np.trapezoid((x-y)**2,r['x']))))
            for key,x,y in zip(('rho','u','p','T'),primitives(v),primitives(target))}
    r['summary'].update(center=a.center,speed=a.speed,metrics=error(r['output'],exact),
        initial_metrics=error(r['initial'],initial),arguments=vars(a))
    out=ROOT/'benchmarks/traveling_shock/residual_splines';out.mkdir(parents=True,exist_ok=True)
    (out/f'{a.tag}.json').write_text(json.dumps(r['summary'],indent=2))
    np.savez(out/f'{a.tag}.npz',**{k:v for k,v in r.items() if k!='summary'},exact=exact)
    print(json.dumps({k:v for k,v in r['summary'].items() if k!='updates'}),flush=True)


if __name__=='__main__':main()
