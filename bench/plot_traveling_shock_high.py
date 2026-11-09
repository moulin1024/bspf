# SPDX-License-Identifier: GPL-3.0-or-later
"""Plot the high-accuracy continuation, distinguishing h and p refinement."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'benchmarks/traveling_shock';high=out/'high_resolution'
old=json.loads((out/'summary.json').read_text())
def read(name):return json.loads((high/(name+'.json')).read_text())
ordinary=[read(f'ordinary{n}_conditioned') for n in (193,257,385,513,641,769,1025,1537)]+[read('ordinary2305')]
conditioned=read('ordinary2305_check')
ordinary[-1]['metrics']=conditioned['metrics']
enriched=[read(f'enriched{n}_p13_reconditioned') for n in (193,257,385,513,641,769)]
validation=read('enriched769_p13_reconditioned_check')
enriched[-1]['metrics']=validation['metrics']
fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
for ax,field,title in zip(axes.flat,('rho','u','p','T'),('Density','Velocity','Pressure','Temperature')):
    for rows,label,color,marker in ((ordinary,'Ordinary BSPF','#D55E00','o'),
        (old['enriched'],'Quintic enrichment (earlier)','#999999','x'),
        (enriched,'Degree-13, reconditioned enrichment','#0072B2','s')):
        ax.loglog([r['final_modes'] for r in rows],[r['metrics'][field]['linf'] for r in rows],
            color=color,marker=marker,label=label,lw=1.5,markersize=5)
    ax.axhline(1e-8,color='black',ls='--',lw=1,label='10⁻⁸ target')
    ax.set(title=title,xlabel='Final coefficients per field',ylabel='Peak absolute error')
    ax.grid(alpha=.2,which='both');ax.legend(fontsize=8)
fig.suptitle('Exact traveling shock: convergence toward 10⁻⁸\n'
    'μ = 10⁻³, Pr = 3/4, t = 0.05; finest runs checked on 64,001 points',fontsize=14)
fig.savefig(high/'convergence_1e8.png',dpi=180);fig.savefig(high/'convergence_1e8.pdf')
summary=dict(target=1e-8,ordinary=ordinary,enriched_high_order=enriched,
    finest_validation=dict(ordinary=conditioned,enriched=validation),
    achieved={name:all(r['metrics'][field]['linf']<=1e-8 for field in ('rho','u','p','T'))
              for name,r in (('ordinary',ordinary[-1]),('enriched',enriched[-1]))},
    notes=['High-order enrichment is a new p-refinement branch, not a continuation of the fixed quintic space.',
           'Endpoint basis counts increase to control fit amplification.',
           'Reconditioning keeps selected centers but rebuilds basis directions; remapping errors are reported.',
           'Time/quadrature checks are local at the final state, not full refined re-integrations.',
           'Runs overlapped in execution; their seconds are not a matched performance benchmark.'])
(high/'summary_1e8.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary['achieved']))
