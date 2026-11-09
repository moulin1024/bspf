# SPDX-License-Identifier: GPL-3.0-or-later
"""Serial warm-run comparison at a common 5e-4 primitive peak-error threshold."""
import json,sys
from pathlib import Path
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'examples/pde'))
from wenoz_traveling_shock import solve as weno
from viscous_traveling_shock import solve as bspf
out=ROOT/'benchmarks/traveling_shock/wenoz';out.mkdir(parents=True,exist_ok=True)
weno(33,final_time=0.)
bspf(65,final_time=1e-6,output_points=101)
rows=[]
for kind,n in [('WENO-Z5',2305),('ordinary BSPF',897),('enriched BSPF',193)]:
    start=perf_counter()
    if kind=='WENO-Z5':r,_=weno(n);modes=n
    else:
        enriched=kind=='enriched BSPF'
        r=bspf(n,enriched,quadrature=64 if enriched else 12,cfl=.7,output_points=64001,
               line_options={'n_basis':(n-1)//16+9},spline_degree=13,refinement=4,halfwidth=12.,recondition=enriched,
               progress=True)['summary'];modes=r['final_modes']
    wall=perf_counter()-start
    rows.append(dict(method=kind,n=n,modes=modes,wall_seconds=wall,report=r))
    (out/'serial_timing.json').write_text(json.dumps(rows,indent=2))
    print(kind,n,modes,wall,max(v['linf'] for v in r['metrics'].values()),flush=True)
