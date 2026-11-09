"""Fixed-dictionary BSPF accuracy and timing against the Becker reference."""
import argparse,json,sys
from pathlib import Path
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'examples/pde'))
from dictionary_spline_shock import solve
from viscous_traveling_shock import BeckerShock,primitives
from compiled_galerkin_step import warmup


def run(**options):
    center=options.pop('center',.4);speed=options.pop('speed',.5)
    shock=BeckerShock(mu=options.get('mu',.001),center=center,speed=speed)
    boundary=shock.state(np.array([0.,1.]))
    np.testing.assert_allclose(shock.state(np.array([0.,1.]),options.get('final_time',.05)),boundary,rtol=0,atol=1e-14)
    start=perf_counter();r=solve(shock.state,boundary,**options)
    exact=shock.state(r['x'],r['summary']['time'])
    def error(v,e):return {name:dict(linf=float(np.max(abs(a-b)))) for name,a,b in zip(('rho','u','p','T'),primitives(v),primitives(e))}
    r['summary'].update(center=center,speed=speed,metrics=error(r['output'],exact),
        initial_metrics=error(r['initial'],shock.state(r['x'])))
    if len(r['trace_times']):
        target=np.stack([shock.state(r['trace_x'],t) for t in r['trace_times']])
        r['summary']['trace_metrics']=error(r['trace'],target)
    r['summary']['end_to_end_seconds']=perf_counter()-start;r['exact']=exact
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--intervals',type=int,default=38);p.add_argument('--spacing',type=float,default=1/512)
    p.add_argument('--widths',nargs='+',type=float,default=[1/64,1/128,1/256])
    p.add_argument('--center',type=float,default=.4);p.add_argument('--speed',type=float,default=.5)
    p.add_argument('--interval',type=float,default=.005);p.add_argument('--time',type=float,default=.05)
    p.add_argument('--probe-cells',type=int,default=512);p.add_argument('--quadrature',type=int,default=8)
    p.add_argument('--outer-quadrature',type=int,default=8)
    p.add_argument('--cfl',type=float,default=1.);p.add_argument('--freeze',action='store_true')
    p.add_argument('--fit-tolerance',type=float,default=1e-7)
    p.add_argument('--trace-points',type=int,default=0)
    p.add_argument('--tag',default='trial');a=p.parse_args();warmup()
    options=vars(a).copy();tag=options.pop('tag');options['final_time']=options.pop('time')
    r=run(**options,progress=True)
    folder=ROOT/'benchmarks/traveling_shock/dictionary_splines';folder.mkdir(parents=True,exist_ok=True)
    (folder/f'{tag}.json').write_text(json.dumps(r['summary'],indent=2))
    np.savez(folder/f'{tag}.npz',**{k:v for k,v in r.items() if k!='summary'})
    print(json.dumps({k:v for k,v in r['summary'].items() if k!='updates'}),flush=True)


if __name__=='__main__':main()
