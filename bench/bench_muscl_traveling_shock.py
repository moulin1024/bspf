# SPDX-License-Identifier: GPL-3.0-or-later
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'examples/pde'))
from muscl_traveling_shock import solve
out=ROOT/'benchmarks/traveling_shock/muscl';out.mkdir(parents=True,exist_ok=True)
for n,cfl in [(n,.4) for n in (256,512,1024,2048,4096,8192)]+[(4096,.2)]:
    key=f'muscl_{n}_cfl{int(cfl*100)}';path=out/(key+'.json')
    if path.exists():continue
    report,data=solve(n,cfl)
    path.write_text(json.dumps(report,indent=2));np.savez(out/(key+'.npz'),**data)
    print(key,report,flush=True)
