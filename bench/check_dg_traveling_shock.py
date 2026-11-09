# SPDX-License-Identifier: GPL-3.0-or-later
"""Full evolution sensitivity checks for the selected degree-seven DG run."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'examples/pde'))
from dg_traveling_shock import solve,primitives,evaluate,operators,BeckerShock
p=argparse.ArgumentParser();p.add_argument('check',choices=['time','quadrature']);p.add_argument('--elements',type=int,default=576);a=p.parse_args()
out=ROOT/'benchmarks/traveling_shock/dg'
r,d=solve(a.elements,7,cfl=.35 if a.check=='time' else .7,quadrature=34 if a.check=='quadrature' else 17)
base=np.load(out/f'dg_p7_ne{a.elements}.npz')
r['profile_difference']={k:float(np.max(abs(v-z))) for k,v,z in zip(('rho','u','p','T'),primitives(d['output']),primitives(base['output']))}
E=operators(7)[4]
r['trace_difference']={k:float(np.max(abs(v-z))) for k,v,z in zip(('rho','u','p','T'),
    primitives(evaluate(d['coefficients'],E).reshape(-1,3)),primitives(evaluate(base['coefficients'],E).reshape(-1,3)))}
key=f'check_p7_ne{a.elements}_{a.check}'
(out/(key+'.json')).write_text(json.dumps(r,indent=2));np.savez(out/(key+'.npz'),**d)
print(r,flush=True)
