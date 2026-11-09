# SPDX-License-Identifier: GPL-3.0-or-later
"""Summarize exact traveling-shock convergence and quadrature sensitivity."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/pde'))
from viscous_traveling_shock import primitives
out=ROOT/'benchmarks/traveling_shock'
a=json.loads((out/'convergence.json').read_text())['runs']
b=json.loads((out/'convergence_fine.json').read_text())['runs']
v=json.loads((out/'validation.json').read_text())
f=json.loads((out/'validation_fine.json').read_text())
ordinary=[row for key,row in (a|b).items() if key.startswith('ordinary_') and 'failed' not in row]
ordinary.sort(key=lambda row:row['n'])
low=[row for key,row in (a|b).items() if key.startswith('enriched_') and 'failed' not in row]
low.sort(key=lambda row:row['n'])
high=[f['enriched_33_q64_hw2'],f['enriched_49_q64_hw2'],v['enriched_65_q64_cfl0.35']['summary'],
      v['enriched_97_q64_cfl0.35']['summary'],f['enriched_129_q128_hw2'],f['enriched_193_q128_hw2']]
def orders(rows,metric):
    return [dict(n_from=a['n'],n_to=b['n'],order=float(np.log(a['metrics']['rho'][metric]/b['metrics']['rho'][metric])/
        np.log((b['n']-1)/(a['n']-1)))) for a,b in zip(rows,rows[1:])]
summary=dict(ordinary=ordinary,enriched=high,enriched_q32=low,
    observed_density_orders={label:{metric:orders(rows,metric) for metric in ('l2','linf')}
        for label,rows in (('ordinary',ordinary),('enriched',high))},
    note='Observed rates use base spacing h=1/(n-1). Adaptive spaces and quadrature vary; these are not formal fixed-space orders.')
(out/'summary.json').write_text(json.dumps(summary,indent=2))
d=np.load(out/'convergence.npz');fine=np.load(out/'convergence_fine.npz');vf=np.load(out/'validation_fine.npz');vv=np.load(out/'validation.npz')
x=d['x'];exact=primitives(d['exact'])
fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
ax=axes.flat[0]
for values,label,color,ls in ((d['exact'],'Exact Becker','black','-'),(d['ordinary_65'],'Ordinary 65','#D55E00','--'),
                           (vv['enriched_65_q64_cfl0.35'],'Enriched 65 + 45','#0072B2','-')):
    ax.plot(x,primitives(values)[0],label=label,color=color,ls=ls,lw=1.4)
ax.set(xlim=(.38,.47),xlabel='x',ylabel='Density',title='Same 65-mode base: shock profile');ax.legend(fontsize=8)
ax=axes.flat[1]
for values,label,color in ((fine['ordinary_513'],'Ordinary 513','#D55E00'),
                         (vv['enriched_65_q64_cfl0.35'],'Enriched 110 total','#0072B2'),
                         (vf['enriched_193_q128_hw2'],'Enriched 262 total','#009E73')):
    ax.semilogy(x,np.maximum(abs(primitives(values)[0]-exact[0]),1e-14),label=label,color=color,lw=1.1)
ax.set(xlim=(.37,.48),ylim=(1e-9,.03),xlabel='x',ylabel='Absolute density error',title='Error against exact traveling profile');ax.legend(fontsize=8)
for ax,key,title in zip(list(axes.flat)[2:],('rho','u','p','T'),('Density','Velocity','Pressure','Temperature')):
    for rows,label,color,ls,marker in ((ordinary,'Ordinary, q32','#D55E00','-','o'),
            (low,'Enriched, q32','#999999',':','x'),(high,'Enriched, q64/128','#0072B2','-','s')):
        ax.loglog([r['final_modes'] for r in rows],[r['metrics'][key]['linf'] for r in rows],
                  label=label,color=color,ls=ls,marker=marker,markersize=4)
    ax.set(xlabel='Final coefficients per field',ylabel='Peak absolute error',title=title+' convergence')
    ax.legend(fontsize=8)
for ax in axes.flat:ax.grid(alpha=.2,which='both')
fig.suptitle('Exact viscous traveling shock: μ = 10⁻³, Pr = 3/4, t = 0.05\n'
             'Detected spline enrichment; no forcing, artificial viscosity, or prescribed front location',fontsize=14)
fig.savefig(out/'convergence.png',dpi=180);fig.savefig(out/'convergence.pdf')
print(json.dumps(summary['observed_density_orders'],indent=2))
