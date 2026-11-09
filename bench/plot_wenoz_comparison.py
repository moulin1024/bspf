# SPDX-License-Identifier: GPL-3.0-or-later
import json,sys
from pathlib import Path
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'examples/pde'))
from viscous_traveling_shock import BeckerShock,primitives
out=ROOT/'benchmarks/traveling_shock/wenoz';fields=('rho','u','p','T')
bspf=json.loads((out.parent/'high_resolution/full_convergence.json').read_text())
muscl=json.loads((out.parent/'muscl/comparison.json').read_text())['muscl']
rows=[]
for n in (257,513,1025,2049,4097,8193):
    r=json.loads((out/f'wenoz_{n}_cfl70.json').read_text())
    d=np.load(out/f'wenoz_{n}_cfl70.npz');shock=BeckerShock()
    interpolated=make_interp_spline(d['nodes'],shock.state(d['nodes'],r['time']),k=5)(d['x'])
    r['exact_profile_interpolation_linf']={k:float(np.max(abs(a-b))) for k,a,b in zip(fields,primitives(interpolated),primitives(d['exact']))}
    if rows:
        a=rows[-1]
        r['rates']={k:float(np.log(a['metrics'][k]['linf']/r['metrics'][k]['linf'])/np.log((n-1)/(a['n']-1))) for k in fields}
    rows.append(r)
checks=[]
for n in (2049,4097):
    r=json.loads((out/f'wenoz_{n}_cfl35.json').read_text())
    a=np.load(out/f'wenoz_{n}_cfl70.npz')['output'];b=np.load(out/f'wenoz_{n}_cfl35.npz')['output']
    r['profile_difference']={k:float(np.max(abs(v-z))) for k,v,z in zip(fields,primitives(a),primitives(b))};checks.append(r)
timing=json.loads((out/'serial_timing.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for ax,k,title in zip(axes.flat,fields,('Density','Velocity','Pressure','Temperature')):
    ax.loglog([r['n'] for r in muscl],[r['metrics'][k]['linf'] for r in muscl],'o--',color='gray',label='MUSCL-MC/HLLC')
    ax.loglog([r['n'] for r in rows],[r['metrics'][k]['linf'] for r in rows],'o-',label='Characteristic WENO-Z5')
    for group,label,marker in [('ordinary','Ordinary BSPF','s'),('enriched','Enriched BSPF','^')]:
        data=bspf[group];ax.loglog([r['final_modes'] for r in data],[r['dense_metrics'][k]['linf'] for r in data],marker+'-',label=label)
    ax.axhline(1e-8,color='gray',ls='--');ax.grid(which='both',alpha=.2)
    ax.set(title=title,xlabel='Unknowns per conserved field',ylabel='Sampled peak absolute error');ax.legend(fontsize=8)
fig.suptitle('Exact viscous traveling shock: WENO-Z5 versus BSPF\nμ=0.001, Pr=0.75, t=0.05; 64,001-point evaluation')
fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf')
notes=['WENO-Z5: Roe characteristic projection, global LF splitting, weight power 2, epsilon=1e-40.',
       'Sixth-order centered differentiation of physical viscous fluxes; classical RK4.',
       'Dense errors use sixth-order diagnostic spline interpolation. Nodal errors and exact-profile interpolation errors are also reported.',
       'Physical problem and final time match previous BSPF/MUSCL comparisons.',
       'Timing cases meet a common 5e-4 threshold, not exactly equal errors. Serial single measurements exclude JIT and include setup and error evaluation.',
       'No WENO 1e-8 attainment or 3-D performance claim.']
(out/'comparison.json').write_text(json.dumps(dict(wenoz=rows,step_checks=checks,serial_timing=timing,notes=notes),indent=2))
lines=['WENO-Z5 comparison','', 'N | Worst dense peak error | Worst nodal error | Pressure rate | Integration seconds']
for r in rows:
    rate='-' if 'rates' not in r else f"{r['rates']['p']:.3f}"
    lines.append(f"{r['n']} | {max(v['linf'] for v in r['metrics'].values()):.6e} | {max(v['linf'] for v in r['nodal_metrics'].values()):.6e} | {rate} | {r['seconds']:.3f}")
lines+=['','Full CFL-halving checks:']
for r in checks:lines.append(f"N={r['n']}: maximum primitive profile change {max(r['profile_difference'].values()):.6e}")
lines+=['','Serial end-to-end wall time:','Method | Unknowns | Worst error | Seconds']
for r in timing:lines.append(f"{r['method']} | {r['modes']} | {max(v['linf'] for v in r['report']['metrics'].values()):.6e} | {r['wall_seconds']:.3f}")
lines+=['']+notes
(out/'comparison.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
