"""Compare evolving dictionary BSPF with the stored fine FC shock-tube run."""
import argparse,json,sys
from pathlib import Path
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'examples/pde'))
from dictionary_shock_tube import solve,warmup
from shock_tube_1d import primitives


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=65);p.add_argument('--time',type=float,default=.1)
    p.add_argument('--interval',type=float,default=.001);p.add_argument('--tolerance',type=float,default=1e-6)
    p.add_argument('--max-blocks',type=int,default=3);p.add_argument('--quadrature',type=int,default=8)
    p.add_argument('--probe-cells',type=int,default=512);p.add_argument('--intervals',type=int,default=38)
    p.add_argument('--core-padding',type=float,default=1.);p.add_argument('--cfl',type=float,default=1.)
    p.add_argument('--rate-weight',type=float,default=1.);p.add_argument('--tag',default='trial')
    p.add_argument('--widths',type=float,nargs='+',default=[1/32,1/64,1/128,1/256,1/512])
    args=p.parse_args();kw=vars(args).copy();tag=kw.pop('tag');kw['final_time']=kw.pop('time')
    warmup();tick=perf_counter();r=solve(**kw,progress=True)
    r['summary']['end_to_end_seconds']=perf_counter()-tick
    if abs(kw['final_time']-.1)<1e-14:
        ref=np.load(ROOT/'benchmarks/shock_tube/fc_gram/fc_6119.npz')
        assert np.array_equal(ref['x'],r['x'])
        r['summary']['reference_discrepancy']={name:float(np.max(abs(v-e))) for name,v,e in
            zip(('rho','u','p','T'),primitives(r['output']),primitives(ref['output']))}
    folder=ROOT/'benchmarks/shock_tube/dictionary';folder.mkdir(parents=True,exist_ok=True)
    (folder/f'{tag}.json').write_text(json.dumps(r['summary'],indent=2))
    np.savez(folder/f'{tag}.npz',**{k:v for k,v in r.items() if k!='summary'})
    print(json.dumps({k:v for k,v in r['summary'].items() if k!='updates'}),flush=True)


if __name__=='__main__':main()
