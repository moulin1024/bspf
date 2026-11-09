# SPDX-License-Identifier: GPL-3.0-or-later
"""Resolve enriched quadrature and test patch-coverage sensitivity."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/pde'))
from viscous_traveling_shock import solve
out=ROOT/'benchmarks/traveling_shock';rows={};outputs={}
configs=[(33,64,2.),(49,64,2.),(129,128,2.),(193,64,2.),(193,128,2.),(129,64,4.)]
for n,q,halfwidth in configs:
    key=f'enriched_{n}_q{q}_hw{halfwidth:g}'
    r=solve(n,True,quadrature=q,halfwidth=halfwidth)
    rows[key]=r['summary'];outputs[key]=r['output'];outputs['x']=r['x'];outputs['exact']=r['exact']
    (out/'validation_fine.json').write_text(json.dumps(rows,indent=2));np.savez(out/'validation_fine.npz',**outputs)
    print(key,'modes',r['summary']['final_modes'],'seconds',r['summary']['seconds'],'rho',r['summary']['metrics']['rho'],flush=True)
