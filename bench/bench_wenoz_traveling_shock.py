# SPDX-License-Identifier: GPL-3.0-or-later
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'examples/pde'))
from wenoz_traveling_shock import solve
out=ROOT/'benchmarks/traveling_shock/wenoz';out.mkdir(parents=True,exist_ok=True)
for n,cfl in [(n,.7) for n in (257,513,1025,2049,4097,8193)]+[(2049,.35),(4097,.35)]:
    key=f'wenoz_{n}_cfl{round(cfl*100)}';path=out/(key+'.json')
    if path.exists():continue
    r,data=solve(n,cfl)
    path.write_text(json.dumps(r,indent=2));np.savez(out/(key+'.npz'),**data)
    print(key,r,flush=True)
