# SPDX-License-Identifier: GPL-3.0-or-later
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'benchmarks/traveling_shock/muscl'
bspf=json.loads((out.parent/'high_resolution/full_convergence.json').read_text())
rows=[json.loads((out/f'muscl_{n}_cfl40.json').read_text()) for n in (256,512,1024,2048,4096,8192)]
for a,b in zip(rows,rows[1:]):
    b['observed_rates']={k:np.log(a['metrics'][k]['linf']/b['metrics'][k]['linf'])/np.log(b['n']/a['n']) for k in ('rho','u','p','T')}
check=json.loads((out/'muscl_4096_cfl20.json').read_text())
a=np.load(out/'muscl_4096_cfl40.npz')['output'];b=np.load(out/'muscl_4096_cfl20.npz')['output']
check['profile_difference']=dict(zip(('rho','u','p','T'),np.max(abs(a-b),axis=0).tolist()))
timing=json.loads((out/'serial_timing.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for ax,k,title in zip(axes.flat,('rho','u','p','T'),('Density','Velocity','Pressure','Temperature')):
    ax.loglog([r['n'] for r in rows],[r['metrics'][k]['linf'] for r in rows],'o-',label='MUSCL-MC/HLLC')
    for group,label,marker in [('ordinary','Ordinary BSPF','s'),('enriched','Enriched BSPF','^')]:
        data=bspf[group]
        ax.loglog([r['final_modes'] for r in data],[r['dense_metrics'][k]['linf'] for r in data],marker+'-',label=label)
    ax.axhline(1e-8,color='gray',ls='--')
    ax.set(title=title,xlabel='Unknowns per conserved field',ylabel='Sampled peak absolute error')
    ax.grid(which='both',alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Exact viscous traveling shock: MUSCL versus BSPF\n'
             'μ=0.001, Pr=0.75, t=0.05; 64,001-point profile evaluation')
fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf')
summary=dict(muscl=rows,step_check=check,serial_timing=timing,
             notes=['MUSCL: primitive MC reconstruction, HLLC, centered physical viscosity, SSPRK2.',
                    'Finite-volume pointwise errors use piecewise-linear primitive reconstruction.',
                    'Conservative cell-average errors are separately recorded against quadrature averages.',
                    'Serial timings exclude JIT warmup and include setup, integration and common dense error evaluation.',
                    'Timing configurations meet a common threshold, not exactly equal errors; single measurements.',
                    'No claim that MUSCL has been run to 1e-8; no 3-D performance inference.'])
(out/'comparison.json').write_text(json.dumps(summary,indent=2))
lines=['MUSCL comparison: exact viscous traveling shock','',
       'MUSCL-MC/HLLC, centered viscous flux, SSPRK2, CFL=.4.',
       'Peak errors use the reconstructed solution on 64,001 points.','',
       'Cells | Worst primitive peak error | Density cell-average peak error | Pressure observed rate']
for r in rows:
    lines.append(f"{r['n']} | {max(v['linf'] for v in r['metrics'].values()):.6e} | "
                 f"{r['cell_average_linf'][0]:.6e} | {r.get('observed_rates',{}).get('p',float('nan')):.3f}")
lines+=['','Serial warm-run wall times (seconds), including setup/integration/error evaluation:',
        'Method | Unknowns per field | Worst error | Seconds']
for r in timing:
    lines.append(f"{r['method']} | {r['modes']} | {max(v['linf'] for v in r['report']['metrics'].values()):.6e} | {r['wall_seconds']:.3f}")
lines+=['',f"4096-cell full CFL .4 -> .2 check: maximum profile change {max(check['profile_difference'].values()):.6e}."]+summary['notes']
(out/'comparison.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
