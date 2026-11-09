# SPDX-License-Identifier: GPL-3.0-or-later
"""Matched fresh-process timings before/after ordinary BSPF optimization."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
PROGRAM = '''
import json,sys,numpy as np
from time import perf_counter
from shock_tube_1d import BSPFNavierStokes
n=int(sys.argv[1]);start=perf_counter()
r=BSPFNavierStokes(n,mu=.0001,quadrature=12).solve(.1,.01,.7)
seconds=perf_counter()-start
np.savez(sys.argv[2],output=r['output'])
print(json.dumps(dict(seconds=seconds,summary=r['summary'])))
'''
def main():
    out=ROOT/'benchmarks/shock_tube'
    baseline=out/'ordinary_before_opt'
    if not (baseline/'shock_tube_1d.py').exists():
        raise RuntimeError('Pre-change shock_tube_1d.py snapshot required in '+str(baseline))
    runs={};outputs={}
    configs=[(label,n) for n in (237,450,480) for label in ('previous','updated')]
    for repetition in range(3):
        for label,n in configs if repetition%2==0 else configs[::-1]:
            key=f'{label}_{n}';file=out/f'ordinary_optimization_{key}.npz'
            env=os.environ.copy();env['OPENBLAS_NUM_THREADS']=env['OMP_NUM_THREADS']='1'
            paths=([baseline] if label=='previous' else [])+[ROOT/'examples/pde',ROOT/'python']
            env['PYTHONPATH']=os.pathsep.join(map(str,paths))
            p=subprocess.run([sys.executable,'-c',PROGRAM,str(n),str(file)],env=env,cwd=ROOT,text=True,capture_output=True,check=True)
            row=json.loads(p.stdout);runs.setdefault(key,[]).append(row);outputs[key]=np.load(file)['output']
            print(repetition+1,key,round(row['seconds'],4),flush=True)
    summary={}
    for n in (237,450,480):
        old,new=(f'{label}_{n}' for label in ('previous','updated'))
        times={key:float(np.median([r['seconds'] for r in runs[key]])) for key in (old,new)}
        summary[n]=dict(median_seconds=times,speedup=times[old]/times[new],
            max_conservative_difference=float(abs(outputs[old]-outputs[new]).max()))
    (out/'ordinary_optimization.json').write_text(json.dumps(dict(summary=summary,runs=runs),indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
