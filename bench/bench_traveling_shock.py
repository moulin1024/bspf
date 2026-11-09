# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatial convergence against an exact viscous traveling shock."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/pde'))
from viscous_traveling_shock import solve


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ordinary',type=int,nargs='*',default=[33,49,65,97,129,193,257])
    p.add_argument('--enriched',type=int,nargs='*',default=[33,49,65,97,129])
    p.add_argument('--time',type=float,default=.05)
    p.add_argument('--mu',type=float,default=.001)
    p.add_argument('--cfl',type=float,default=.35)
    p.add_argument('--quadrature',type=int,default=32)
    p.add_argument('--prefix',default='benchmarks/traveling_shock/convergence')
    a=p.parse_args();prefix=ROOT/a.prefix;prefix.parent.mkdir(parents=True,exist_ok=True)
    rows={};outputs={}
    for enriched,ns in ((False,a.ordinary),(True,a.enriched)):
        for n in ns:
            key=f'{"enriched" if enriched else "ordinary"}_{n}'
            try:
                r=solve(n,enriched,a.mu,a.time,a.cfl,a.quadrature)
            except (RuntimeError,ValueError) as exc:
                rows[key]=dict(failed=str(exc));print(key,rows[key],flush=True)
            else:
                rows[key]=r['summary'];outputs[key]=r['output'];outputs[key+'_initial']=r['initial']
                outputs['x']=r['x'];outputs['exact']=r['exact']
                print(key,'modes',rows[key]['final_modes'],'seconds',round(rows[key]['seconds'],3),
                      'Linf',[float(f'{v["linf"]:.4g}') for v in rows[key]['metrics'].values()],flush=True)
            prefix.with_suffix('.json').write_text(json.dumps(dict(parameters=vars(a),runs=rows),indent=2))
            np.savez(prefix.with_suffix('.npz'),**outputs)


if __name__=='__main__':main()
