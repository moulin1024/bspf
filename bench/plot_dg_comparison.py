# SPDX-License-Identifier: GPL-3.0-or-later
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'benchmarks/traveling_shock/dg'
rows=[json.loads(p.read_text()) for p in sorted(out.glob('dg_p*_ne*.json'))]
base=json.loads((out.parent/'high_resolution/full_convergence.json').read_text())
timed=json.loads((out.parent/'wenoz/serial_timing_1e6.json').read_text())['selected']
for group,method in [('ordinary','ordinary BSPF'),('enriched','enriched BSPF')]:
    r=next(r for r in timed if r['method']==method)
    if all(b['n']!=r['n'] for b in base[group]):
        base[group].append(dict(n=r['n'],final_modes=r['modes'],dense_metrics=r['report']['metrics']))
        base[group].sort(key=lambda b:b['n'])
weno=json.loads((out.parent/'wenoz/comparison.json').read_text())['wenoz']
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for ax,k,title in zip(axes.flat,('rho','u','p','T'),('Density','Velocity','Pressure','Temperature')):
    for p in sorted(set(r['degree'] for r in rows)):
        data=sorted([r for r in rows if r['degree']==p],key=lambda r:r['elements'])
        ax.loglog([r['modes'] for r in data],[max(r['metrics'][k]['linf'],r['trace_linf'][k]) for r in data],'o-',label=f'DG p={p}')
    ax.loglog([r['n'] for r in weno],[r['metrics'][k]['linf'] for r in weno],'x--',color='gray',label='WENO-Z5')
    for group,label in [('ordinary','Ordinary BSPF'),('enriched','Enriched BSPF')]:
        data=base[group];ax.loglog([r['final_modes'] for r in data],[r['dense_metrics'][k]['linf'] for r in data],'^--',label=label)
    ax.axhline(1e-6,color='black',ls=':',label='1e-6 target')
    ax.set(title=title,xlabel='Coefficients per conserved field',ylabel='Peak absolute error')
    ax.grid(which='both',alpha=.2);ax.legend(fontsize=7)
fig.suptitle('Exact viscous traveling shock: modal DG / BR1 versus BSPF\nμ=0.001, Pr=0.75, t=0.05; DG checks both interface traces')
fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf')
(out/'sweep_summary.json').write_text(json.dumps(rows,indent=2))
print([(r['degree'],r['elements'],r['worst_error']) for r in rows])
