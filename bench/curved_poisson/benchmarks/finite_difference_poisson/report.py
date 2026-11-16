"""Reuse recorded single-solve results; never rerun a timed case."""
import os
os.environ['MPLCONFIGDIR']='/private/tmp/fd4-mpl'
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence/finite_difference_poisson/v1';SRC=ROOT/'evidence/nonconformal_bspf_opt/hard_shell128'
fd=sorted([json.loads(p.read_text()) for p in OUT.glob('hard_*/result.json')],key=lambda r:r['n']);smooth=sorted([json.loads(p.read_text()) for p in OUT.glob('smooth_*/result.json')],key=lambda r:r['n']);prior=json.loads((SRC/'explicit_input_comparison/results.json').read_text());methods=[('BSPF, direct Gauss forcing',prior['direct'],'#1565c0'),('Nektar++, direct quadrature forcing',prior['nektar'],'#d76b08'),('Native FD4, direct nodal forcing (in-house)',fd,'#218c57')]
def sec(r):return r.get('solve_seconds',r.get('timing',{}).get('solve_seconds'))
fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
for label,rows,col in methods:
 valid=sorted([r for r in rows if r['status']=='validated'],key=sec);front=[];best=float('inf')
 for r in valid:
  if r['l2_relative']<best:front.append(r);best=r['l2_relative']
 ax.loglog([sec(r) for r in valid],[r['l2_relative'] for r in valid],'o',color=col,alpha=.3)
 ax.loglog([sec(r) for r in front],[r['l2_relative'] for r in front],'-o',label=label,color=col)
 for r in rows:
  if r['status']!='validated' and 'l2_relative' in r:ax.loglog(sec(r),r['l2_relative'],'x',color='gray')
for r in fd:
 if 'l2_relative' in r:ax.annotate(str(r['n'])+'²',(sec(r),r['l2_relative']),xytext=(-5,6) if r is fd[-1] else (5,6),ha='right' if r is fd[-1] else 'left',textcoords='offset points',fontsize=8,color='#166e40')
ax.set(xlabel='Prepared native Poisson solve wall time [s] — setup excluded',ylabel='Relative physical L2 solution error',title='Four-spline curved domain · turbulence MMS, shells 4–128');ax.grid(which='both',alpha=.2);ax.legend(fontsize=9,loc='lower left');fig.supxlabel('One CPU thread · one solve per configuration · no timing repetitions\nSame analytic PDE and boundary values; each method samples forcing on its own native grid.',fontsize=9)
for ext in ('png','pdf'):fig.savefig(OUT/f'accuracy_wall_time.{ext}',dpi=180)
fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained');ns=np.array([r['n']-1 for r in smooth]);ee=np.array([r['l2_relative'] for r in smooth]);ax.loglog(ns,ee,'o-',label='Measured FD4');ax.loglog(ns,ee[-1]*(ns[-1]/ns)**4,'--',label='Fourth order');ax.set(xlabel='Intervals per reference direction',ylabel='Relative physical L2 error',title='FD4 qualification on the same curved domain');ax.grid(which='both',alpha=.2);ax.legend();fig.savefig(OUT/'smooth_convergence.png',dpi=180)
orders=np.log2(ee[:-1]/ee[1:]);assert np.all(orders>3.8) and all(r['status']=='validated' for r in smooth)
lines=['# Native fourth-order finite-difference Poisson baseline','','Problem: -Δu=f on the same nonconformal four-spline Coons quadrilateral, exact nonzero Dirichlet data on all four edges. The hard MMS is the unchanged saved seed 20261115 field with shell energy k^(-5/3), shells 4–128.','','## Implementation and timing','','C++ float64 local fourth-order strong-form mapped Laplacian, with both metric drift terms and the mixed derivative. Five-point centered first/second derivatives; fourth-order one-sided first derivatives and six-point second-derivative closures at the first/last interior rows. Mixed derivatives are successive local first derivatives. No dense global matrix or dense solve.','','Right-preconditioned BiCGSTAB uses an FFTW DST-I inverse of a constant-coefficient second-order reference-grid Poisson operator, with mean mapped diagonal coefficients. Cross and drift terms remain in the full operator. Stencil application is O(N), preconditioning O(N log N), storage O(N), for N interior unknowns. Single-thread FFTW; no threaded plan.','','Geometry preparation, analytic sampling, FFTW plan construction and preconditioner preparation are outside timing. Boundary lifting, iterative solve, true residual verification and boundary solution assembly are inside timing. All cases start at zero interior correction; exact interior solution is used only for validation. Forcing is sampled analytically at native FD nodes and only interior forcing enters the solve. BSPF and Nektar use direct analytic forcing at their own quadrature nodes. This is the same analytic-input policy, not identical sample counts.','','One solve per configuration, no warmups/repeats. BSPF and Nektar timings are reused from the previous campaign; these are individual timing observations, not confidence intervals. Nektar factorization is prepared outside its timer. Gray crosses denote failed audits, including the archived BSPF n=96 point.','','## Independent accuracy checks','','Off-grid solution reconstruction uses local degree-seven Lagrange interpolation for error integration only; it never enters the FD solve. Relative physical L2 error is independently integrated on Gauss grids of 640 and 768 points per axis for the hard MMS. Agreement must be within 1% plus 1e-11. Smooth qualification uses 100/140-point rules. Native true relative residual must be below 2e-12. The residual check validates the discrete equation; the MMS and refinement test validate consistency.','','Smooth qualification: u=sin(3x+0.3) cos(4y-0.2)+0.2x+0.1y; f=25 sin(3x+0.3) cos(4y-0.2).','','| Nodes/axis | Relative L2 | Observed order |','|---:|---:|---:|']
for i,r in enumerate(smooth):lines.append(f"| {r['n']} | {r['l2_relative']:.6e} | "+(f'{orders[i-1]:.3f}' if i else '—')+' |')
lines+=['','## Hard turbulence MMS','','| Nodes/axis | Unknowns | Solve ms | Iterations | Relative L2 | Status |','|---:|---:|---:|---:|---:|---|']
for r in fd:lines.append(f"| {r['n']} | {(r['n']-2)**2} | {1000*sec(r):.3f} | {r['iterations']} | {r['l2_relative']:.6e} | {r['status']} |")
lines+=['','## Fastest tested configuration reaching each target','','| Relative L2 target | BSPF ms | Nektar++ ms | Native FD4 ms |','|---:|---:|---:|---:|'];targets=[]
for target in (1e-2,1e-3,1e-4,1e-5,1e-6,1e-8):
 vals=[];rec={'target':target}
 for label,rows,_ in methods:
  eligible=[r for r in rows if r['status']=='validated' and r['l2_relative']<=target and abs(np.diff(r.get('quadrature_errors',r.get('error_quadrature_values')))[0])<=.05*target]
  best=min(eligible,key=sec) if eligible else None;vals.append(f'{1000*sec(best):.3f}' if best else 'not reached');rec[label]=best['id'] if best else None
 lines.append(f'| {target:.0e} | '+' | '.join(vals)+' |');targets.append(rec)
lines+=['','“Not reached” means not reached by the tested ladder. The native FD4 solver is an in-house reference, not an established upstream solver or a claim about the best possible finite-difference implementation. Coarse hard-field grids can alias the forcing severely; their large errors remain visible. These runs do not establish universal method superiority.','']
(OUT/'report.md').write_text('\n'.join(lines));(OUT/'results.json').write_text(json.dumps(dict(fd4=fd,smooth=smooth,targets=targets),indent=2)+'\n');files=list((ROOT/'benchmarks/finite_difference_poisson').glob('*'))+list(OUT.glob('*/result.json'))+[SRC/'field.npz',SRC/'explicit_input_comparison/results.json'];(OUT/'manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},indent=2)+'\n');print('\n'.join(lines))
