"""Measured-point and budget-step plots; no inferred crossing-time estimates."""
import os
os.environ['MPLCONFIGDIR']='/private/tmp/bspf-fine-sweep-mpl'
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence/nonconformal_bspf_opt/hard_shell128/fine_sweep_v1'
data=json.loads((OUT/'results.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
def seconds(r):return r.get('solve_seconds',r.get('timing',{}).get('solve_seconds'))
def eligible(r,target=None):
 if r['status']!='validated' or 'l2_relative' not in r:return False
 errs=r.get('quadrature_errors',r.get('error_quadrature_values',[]))
 return len(errs)==2 and (target is None or r['l2_relative']<=target and abs(errs[1]-errs[0])<=.05*target)
def size(r):return r['n'] if 'n' in r else r['elements_per_axis']*r['degree']+1
def desc(r):return f"n={r['n']}, q={r['q']}" if 'n' in r else f"E={r['elements_per_axis']}, p={r['degree']}"
snapshot_label = '' if protocol['status']=='complete' else ' — partial sweep snapshot'
methods=[('direct','BSPF direct RHS','#1565c0'),('nodal','BSPF nodal RHS','#b73264'),('nektar','Nektar++ direct RHS','#d76b08')]
fig,ax=plt.subplots(figsize=(10.5,6.5),layout='constrained')
max_time=max(seconds(r) for rr in data.values() for r in rr if seconds(r) is not None)
for key,label,col in methods:
 valid=sorted([r for r in data[key] if eligible(r)],key=seconds);front=[];best=float('inf')
 for r in valid:
  if r['l2_relative']<best:front.append(r);best=r['l2_relative']
 ax.loglog([seconds(r) for r in valid],[r['l2_relative'] for r in valid],'o',color=col,alpha=.35,ms=4)
 if front:
  xx=[seconds(r) for r in front];yy=[r['l2_relative'] for r in front];ax.step(xx+[max_time*1.08],yy+[yy[-1]],where='post',color=col,label=label,lw=1.5);ax.loglog(xx,yy,'o',color=col,ms=4)
for target in (1e-5,1e-6,1e-8):ax.axhline(target,color='gray',lw=.7,ls=':',alpha=.6)
failed=[r for rows in data.values() for r in rows if r['status']!='validated' and 'l2_relative' in r and seconds(r) is not None]
if failed:ax.loglog([seconds(r) for r in failed],[r['l2_relative'] for r in failed],'x',ms=5,color='gray',label='Failed audit; excluded from staircase')
ax.set(xlabel='Native solve time budget [s] — setup excluded',ylabel='Best measured relative physical L2 error within budget',title='Four-spline Poisson · shells 4–128'+snapshot_label);ax.grid(which='both',alpha=.18);ax.legend(fontsize=9,loc='lower left')
fig.supxlabel('Markers are actual single-solve measurements. Staircases hold the best measured error until a better measured point.\nNo interpolation of target-crossing times. Nodal BSPF includes forcing reconstruction; direct methods sample analytic forcing.',fontsize=9)
for ext in ('png','pdf'):fig.savefig(OUT/f'accuracy_wall_time.{ext}',dpi=180)
fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
for key,label,col in methods:
 groups=[data[key]] if key!='nektar' else [[r for r in data[key] if r['elements_per_axis']==e] for e in (8,16)]
 for group in groups:
  rr=sorted([r for r in group if 'l2_relative' in r],key=size)
  if not rr:continue
  suffix='' if key!='nektar' else f" (E={rr[0]['elements_per_axis']})";style='--' if key=='nektar' and rr[0]['elements_per_axis']==16 else '-'
  ax.semilogy([size(r) for r in rr],[r['l2_relative'] for r in rr],style+'o',color=col,ms=3,label=label+suffix)
ax.set(xlabel='Nominal resolution per direction: BSPF n; Nektar++ E p + 1',ylabel='Relative physical L2 solution error',title='Fixed turbulence MMS · finer resolution ladder'+snapshot_label);ax.grid(which='both',alpha=.2);ax.legend(fontsize=9);fig.supxlabel('The x-axis labels resolution, not equal quadrature counts or equal work. Includes underresolved cases.',fontsize=9)
for ext in ('png','pdf'):fig.savefig(OUT/f'accuracy_resolution.{ext}',dpi=180)
records=[];lines=['# Fine resolution sweep: curved-domain scalar Poisson','', 'Campaign status at plot generation: '+protocol['status']+'. Plots use the available recorded results; pending cases are not inferred.','','Same fixed nonconformal four-cubic-edge Coons domain and synthetic turbulence MMS, shells 4–128, seed 20261115. Exact nonzero Dirichlet boundaries; no time stepping. Solver kernels, tolerances, setup exclusion and audit criteria are unchanged.','','## Sweep and timing','','- BSPF starts at n=96 and increases by 8 to n=288, with q=2n. Both direct and nodal RHS policies are included.','- Nektar++ uses E=8 elements/axis with p=12,14,...,40; E=16 with p=6,8,...,32 plus odd degrees 21,23,...,31. This starts near BSPF’s lowest nominal resolution and resolves the tighter-error transition.','- Existing matching runs are reused. Every new case has one native solve, one CPU thread, no warmup, no repeated timing. All timed jobs run sequentially.','- Analytic sampling and setup are excluded; Nektar++ static-condensation factorization is prepared before timing.','- BSPF direct and Nektar direct sample the same analytic forcing at their own quadrature points. BSPF nodal samples forcing at BSPF nodes and includes interpolation to quadrature inside timing. The nodal policy has different input information.','','## Fastest measured configurations reaching each target','','Only validated points qualify. Independent quadrature estimates must agree within 5% of the requested target. Values are selected from measured configurations; no log-linear interpolation.','','| Error target | BSPF direct ms (configuration) | BSPF nodal ms (configuration) | Nektar++ direct ms (configuration) |','|---:|---:|---:|---:|']
for target in (1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-10,1e-12):
 rec={'target':target};vals=[]
 for key,_,_ in methods:
  candidates=[r for r in data[key] if eligible(r,target)];best=min(candidates,key=seconds) if candidates else None
  if best:rec[key]={'id':best['id'],'seconds':seconds(best),'error':best['l2_relative']};vals.append(f'{1000*seconds(best):.3f} ({desc(best)})')
  else:vals.append('not reached')
 records.append(rec);lines.append(f'| {target:.0e} | '+' | '.join(vals)+' |')
lines+=['','## Qualification and interpretation','','The wall-time plot uses a best-measured staircase: with a given time budget, it reports only solutions actually measured within that budget. Vertical jumps correspond to newly affordable measured configurations; they do not imply intermediate accuracies were measured. Faint markers retain dominated measurements. Failed audits are excluded from target selection and shown as crosses.','','The resolution plot connects sampled resolutions only to make convergence trends readable; it is not used to infer solve times. Nektar E p + 1 and BSPF n label nominal resolution, not equal forcing-sample counts or equal computational effort.','','Finer sampling can change the ranking implied by a sparse ladder. Even this finer sweep only establishes the fastest tested configurations, not a global optimum or universal method ranking. Reused and new timings are single observations from different moments; small timing differences should not be interpreted as statistically established gains.','','## Recorded cases','']
for key,label,_ in methods:
 lines += [f'### {label}','','| Configuration | Solve ms | Relative L2 | Status | Reused |','|---|---:|---:|---|---|']
 for r in sorted(data[key],key=size):
  ss=seconds(r);err=r.get('l2_relative');lines.append(f"| {desc(r)} | "+(f'{1000*ss:.3f}' if ss is not None else '—')+' | '+(f'{err:.6e}' if err is not None else '—')+f" | {r['status']} | {r['reused']} |")
 lines.append('')
(OUT/'report.md').write_text('\n'.join(lines));(OUT/'targets.json').write_text(json.dumps(records,indent=2)+'\n')
files=[Path(__file__),ROOT/'benchmarks/nonconformal_bspf_opt/fine_sweep.py',ROOT/'benchmarks/nonconformal_bspf_opt/run.py',ROOT/'benchmarks/standard_poisson/campaign.py',ROOT/'benchmarks/nonconformal_bspf_opt/poisson_single',ROOT/'benchmarks/nonconformal_bspf_opt/poisson_nodal',ROOT/'benchmarks/standard_poisson/build/nektar_reference',OUT/'protocol.json',OUT/'results.json',OUT.parent/'field.npz']+[ROOT/r['source_result'] for rows in data.values() for r in rows]
(OUT/'manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},indent=2)+'\n')
print(json.dumps(dict(status=protocol['status'],counts={key:len(data[key]) for key,_,_ in methods},targets=records),indent=2))
