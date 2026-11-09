# SPDX-License-Identifier: GPL-3.0-or-later
"""Separate quadrature and time-step errors in the traveling-shock benchmark."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/pde'))
from viscous_traveling_shock import solve,primitives
out=ROOT/'benchmarks/traveling_shock'
baseline=np.load(out/'convergence.npz')
rows={};outputs={}
for n,q,cfl in ((65,64,.35),(97,64,.35),(129,64,.35),(129,32,.175)):
    key=f'enriched_{n}_q{q}_cfl{cfl}'
    r=solve(n,True,quadrature=q,cfl=cfl)
    fields=primitives(r['output']);old=primitives(baseline[f'enriched_{n}'])
    rows[key]=dict(summary=r['summary'],difference={f:float(abs(a-b).max()) for f,a,b in zip(('rho','u','p','T'),fields,old)})
    outputs[key]=r['output'];outputs['x']=r['x'];outputs['exact']=r['exact']
    (out/'validation.json').write_text(json.dumps(rows,indent=2));np.savez(out/'validation.npz',**outputs)
    print(key,rows[key]['difference'],r['summary']['metrics']['rho'],flush=True)
