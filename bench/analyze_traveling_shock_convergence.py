# SPDX-License-Identifier: GPL-3.0-or-later
"""Dense error norms, observed rates and full-evolution sensitivity checks."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from check_traveling_shock_high import restore, sample, BeckerShock, primitives
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/traveling_shock/high_resolution'
FIELDS=('rho','u','p','T')
x=np.linspace(0,1,64001)
def read(name):
    prefix=OUT/name
    report=json.loads(prefix.with_suffix('.json').read_text())
    cache=Path(str(prefix)+'_dense.npz')
    if cache.exists():
        with np.load(cache) as d:
            output=d['output'];exact=d['exact']
            assert np.array_equal(d['x'],x)
    else:
        with np.load(prefix.with_suffix('.npz')) as d:
            space=restore(report,d,report['quadrature'])
            output=sample(space,d['coefficients'],x)
        exact=BeckerShock(mu=report['mu'],speed=report.get('speed',.5)).state(x,report['time'])
        np.savez(cache,x=x,output=output,exact=exact)
    values=primitives(output);targets=primitives(exact)
    report['dense_metrics']={k:{'linf':float(np.max(abs(v-z))),
        'l2':float(np.sqrt(np.trapezoid((v-z)**2,x)))} for k,v,z in zip(FIELDS,values,targets)}
    return report,values
rows=[];profiles={}
for n in (193,257,385,513,641,769):
    r,v=read(f'enriched{n}_p13_reconditioned')
    profiles[n]=v
    if rows:
        prev=rows[-1]
        r['observed_rates']={k:{norm:float(np.log(prev['dense_metrics'][k][norm]/r['dense_metrics'][k][norm])/
            np.log((n-1)/(prev['n']-1))) for norm in ('linf','l2')} for k in FIELDS}
    rows.append(r)
    print(n,r['final_modes'],r['dense_metrics'],flush=True)
checks=[]
for n in (513,769):
    for q,cfl in ((64,.35),(128,.7)):
        r,v=read(f'enriched{n}_p13_reconditioned_q{q}_cfl{int(round(100*cfl))}')
        r['profile_difference']={k:float(abs(vv-base).max()) for k,vv,base in zip(FIELDS,v,profiles[n])}
        checks.append(r)
        print('sensitivity',n,q,cfl,r['profile_difference'],r['dense_metrics'],flush=True)
ordinary=[read(f'ordinary{n}_conditioned')[0] for n in (193,257,385,513,641,769,1025,1537)]
ordinary.append(read('ordinary2305')[0])
summary={'ordinary':ordinary,'sampling_points':len(x),'enriched':rows,'full_evolution_checks':checks,
         'notes':['Same degree 13, spacing h/4, halfwidth 12h, sensor and adaptation interval.',
                  'Endpoint spline count follows (N-1)/16 + 9; this controls conditioning as N increases.',
                  'Rates are observed along adaptive spaces, not formal fixed-space orders.',
                  'Sensitivity runs independently repeat adaptation and initialization.',
                  'Physical viscosity is 1e-3; no artificial viscosity or forcing.']}
(OUT/'full_convergence.json').write_text(json.dumps(summary,indent=2))
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for ax,k,title in zip(axes.flat,FIELDS,('Density','Velocity','Pressure','Temperature')):
    modes=[r['final_modes'] for r in rows]
    ax.loglog([r['n'] for r in ordinary],[r['dense_metrics'][k]['linf'] for r in ordinary],
              'o-',color='gray',alpha=.7,label='Ordinary Linf')
    for norm,marker in (('linf','o'),('l2','s')):
        ax.loglog(modes,[r['dense_metrics'][k][norm] for r in rows],marker+'-',label='Enriched '+norm)
    for q,cfl,label,marker in ((64,.35,'Half CFL','x'),(128,.7,'Double quadrature','+')):
        selected=[r for r in checks if r['quadrature']==q and r['cfl']==cfl]
        # Position sensitivity points at the matching baseline cost; their actual mode counts are in JSON.
        ax.loglog([next(b['final_modes'] for b in rows if b['n']==r['n']) for r in selected],
                  [r['dense_metrics'][k]['linf'] for r in selected],marker,ms=9,label=label)
    ax.axhline(1e-8,ls='--',color='gray',label='1e-8')
    ax.set(title=title,xlabel='Baseline final coefficients per field',ylabel='Absolute error')
    ax.grid(which='both',alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Degree-13 reconditioned BSPF: six-resolution convergence\n'
             'Exact viscous traveling shock; t=0.05; 64,001-point evaluation')
fig.savefig(OUT/'full_convergence.png',dpi=180)
fig.savefig(OUT/'full_convergence.pdf')
lines=['# Full degree-13 convergence check','',
       'Errors are evaluated on 64,001 points after full integration to t=0.05.',
       'mu=0.001, Pr=0.75; degree 13, spacing h/4, halfwidth 12h, reconditioning enabled.',
       'Baseline q=64, CFL=0.7; endpoint basis count (N-1)/16+9.','',
       '| Base N | Total modes | rho Linf | u Linf | p Linf | T Linf | rho L2 | rho L2 rate |',
       '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for r in rows:
    rate=r.get('observed_rates',{}).get('rho',{}).get('l2')
    lines.append('| '+' | '.join([str(r['n']),str(r['final_modes'])]+
        [f"{r['dense_metrics'][k]['linf']:.4e}" for k in FIELDS]+
        [f"{r['dense_metrics']['rho']['l2']:.4e}",'-' if rate is None else f'{rate:.2f}'])+' |')
lines += ['', '## Ordinary BSPF', '',
          '| N | rho Linf | u Linf | p Linf | T Linf |',
          '| ---: | ---: | ---: | ---: | ---: |']
for r in ordinary:
    lines.append('| '+' | '.join([str(r['n'])]+[f"{r['dense_metrics'][k]['linf']:.4e}" for k in FIELDS])+' |')
lines += ['', 'Ordinary uses q12, CFL 0.7 and the same endpoint count rule through N=1537.',
          'The existing N=2305 run uses 137 endpoint functions (128 aligned knot intervals).',
          '', '## Full-evolution sensitivity', '',
          '| N | q | CFL | Max primitive error | Max change from baseline |',
          '| ---: | ---: | ---: | ---: | ---: |']
for r in checks:
    lines.append(f"| {r['n']} | {r['quadrature']} | {r['cfl']} | "
                 f"{max(v['linf'] for v in r['dense_metrics'].values()):.4e} | "
                 f"{max(r['profile_difference'].values()):.4e} |")
lines += ['','These are complete reruns, including independent adaptation and initialization.',
          'Differences therefore include sensitivity of the adaptive basis and remapping.',
          'Observed rates describe these adaptive trial spaces, not a formal order proof.',
          'Wall times are not comparable because jobs overlapped.']
(OUT/'full_convergence.txt').write_text('\n'.join(lines)+'\n')
