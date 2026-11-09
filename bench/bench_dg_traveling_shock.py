# SPDX-License-Identifier: GPL-3.0-or-later
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'examples/pde'))
from dg_traveling_shock import solve
out=ROOT/'benchmarks/traveling_shock/dg';out.mkdir(parents=True,exist_ok=True)
for p in (3,5,7):
    for ne in ((64,128,256,512,576,640) if p==7 else (64,128,256,512,1024)):
        key=f'dg_p{p}_ne{ne}';path=out/(key+'.json')
        if path.exists():r=json.loads(path.read_text())
        else:
            try:
                r,d=solve(ne,p)
            except ValueError as exc:
                print(key,str(exc),flush=True);continue
            path.write_text(json.dumps(r,indent=2));np.savez(out/(key+'.npz'),**d)
        print(key,r,flush=True)
        if r['worst_error']<=1e-6:break
