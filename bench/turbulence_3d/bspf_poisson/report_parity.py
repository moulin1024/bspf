"""Audit reflection splitting against the same executable's unsplit path."""
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/bspf-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'mfem_poisson'))
# Binary bundle reader has no GPU side effects on import.
from run import read_bundle
OUT=ROOT/'evidence/bspf_cuda3d_parity'
rows=[]
for n in [17,33,65,97,129]:
    q=max(96,(3*n+1)//2);folder=OUT/f'n{n}_q{q}'
    on=json.loads((folder/'result.json').read_text());off=json.loads((folder/'unsplit.json').read_text())
    a=read_bundle(folder/'solution.bin')['solution'];b=read_bundle(folder/'unsplit.bin')['solution']
    change=float(np.linalg.norm(a-b)/np.linalg.norm(b))
    before=float(np.median([r['cg_s'] for r in off['runs'][1:]]))
    assert on['parity_optimized'] and not off['parity_optimized']
    assert change<1e-11 and on['boundary_error']==0
    assert max(r['relative_residual'] for r in on['runs'])<1e-12
    rows.append(dict(n=n,q=q,before_s=before,after_s=on['warm_cg_s'],speedup=before/on['warm_cg_s'],
        relative_l2=on['relative_l2'],relative_solution_change=change,
        before_device_bytes=off['device_bytes'],after_device_bytes=on['device_bytes'],
        iterations=on['runs'][-1]['iterations']))
(OUT/'optimization.json').write_text(json.dumps(rows,indent=2)+'\n')
with (OUT/'optimization.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
fig,axes=plt.subplots(1,2,figsize=(10,4.5))
for key,label in [('before_s','Original contractions'),('after_s','Even/odd contractions')]:
    axes[0].loglog([r['n'] for r in rows],[r[key] for r in rows],'o-',label=label)
axes[0].set(xlabel='Nodes per axis',ylabel='Warm CG time (s)');axes[0].legend()
axes[1].plot([r['n'] for r in rows],[r['speedup'] for r in rows],'o-');axes[1].set(xlabel='Nodes per axis',ylabel='Speedup');axes[1].axhline(1,color='gray',linestyle='--')
for ax in axes:ax.grid(alpha=.25)
fig.suptitle('BSPF degree 8: FP64 reflection splitting on RTX 3070')
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(OUT/f'optimization.{ext}',dpi=180)
print(json.dumps(rows,indent=2))
