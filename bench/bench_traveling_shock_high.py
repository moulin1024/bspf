# SPDX-License-Identifier: GPL-3.0-or-later
"""High-accuracy continuation of the exact traveling-shock benchmark."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/pde'))
from viscous_traveling_shock import solve
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--n',type=int,required=True)
p.add_argument('--enriched',action='store_true')
p.add_argument('--basis',type=int,required=True)
p.add_argument('--degree',type=int,default=13)
p.add_argument('--refinement',type=int,default=4)
p.add_argument('--halfwidth',type=float,default=12.)
p.add_argument('--quadrature',type=int,default=64)
p.add_argument('--recondition',action='store_true')
p.add_argument('--cfl',type=float,default=.7)
p.add_argument('--prefix',required=True)
a=p.parse_args()
r=solve(a.n,a.enriched,quadrature=a.quadrature,cfl=a.cfl,progress=True,output_points=16001,
        line_options=dict(n_basis=a.basis),spline_degree=a.degree,refinement=a.refinement,halfwidth=a.halfwidth,recondition=a.recondition)
prefix=ROOT/a.prefix;prefix.parent.mkdir(parents=True,exist_ok=True)
prefix.with_suffix('.json').write_text(json.dumps(r['summary'],indent=2))
np.savez(prefix.with_suffix('.npz'),**{k:v for k,v in r.items() if k!='summary'})
print(json.dumps(r['summary'],indent=2),flush=True)
