# SPDX-License-Identifier: GPL-3.0-or-later
"""Serial warm solves at a 1e-6 peak primitive-field error threshold.

Select the first passing candidate per method; these are tested configurations,
not a proof of globally minimal resolution or runtime.
"""
import json,sys
from pathlib import Path
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'examples/pde'))
from wenoz_traveling_shock import solve as weno
from viscous_traveling_shock import solve as bspf
out=ROOT/'benchmarks/traveling_shock/wenoz';out.mkdir(parents=True,exist_ok=True)
weno(33,final_time=0.)
bspf(65,final_time=1e-6,output_points=101)
rows=[];attempts=[]
for kind,candidates in [('enriched BSPF',(417,449,513)),('ordinary BSPF',(1601,1665,1793)),('WENO-Z5',(8193,9217))]:
    for n in candidates:
        start=perf_counter()
        if kind=='WENO-Z5':r,data=weno(n);modes=n
        else:
            enriched=kind=='enriched BSPF'
            data=bspf(n,enriched,quadrature=64 if enriched else 12,cfl=.7,output_points=64001,
                line_options={'n_basis':(n-1)//16+9},spline_degree=13,refinement=4,halfwidth=12.,recondition=enriched,
                progress=True)
            r=data.pop('summary');modes=r['final_modes']
        wall=perf_counter()-start
        peak=max(v['linf'] for v in r['metrics'].values())
        record=dict(method=kind,n=n,modes=modes,wall_seconds=wall,worst_error=peak,passed=peak<=1e-6,report=r)
        attempts.append(record)
        key=kind.lower().replace(' ','_').replace('-','_')+f'_{n}_threshold_1e6'
        np.savez(out/(key+'.npz'),**data)
        if record['passed']:rows.append(record)
        (out/'serial_timing_1e6.json').write_text(json.dumps(dict(threshold=1e-6,selected=rows,attempts=attempts,
            timing='Serial single warm runs; setup, integration and 64001-point error evaluation included; JIT and saving excluded.'),indent=2))
        print(kind,n,modes,wall,peak,record['passed'],flush=True)
        if record['passed']:break
    else:raise RuntimeError(f'No passing candidate for {kind}')
