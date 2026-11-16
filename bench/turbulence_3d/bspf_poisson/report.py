"""Historical 2n-quadrature BSPF campaign and same-DOF SEM comparisons.

The benchmark now defaults to 1.5n; report_optimization.py compares those
validated results with this original campaign.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/bspf-matplotlib')
from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]/'evidence';OUT=ROOT/'bspf_cuda3d_rtx3070'
rows=[];results=[]
for n in [17,33,65,97,129]:
    q=max(96,2*n);p=OUT/f'n{n}_q{q}/result.json'
    if not p.exists():continue
    b=json.loads(p.read_text());e=(n-1)//8
    m=json.loads((ROOT/f'mfem_lor_amg_rtx3070/e{e}_p8/result.json').read_text())
    k=json.loads((ROOT/f'nekrs_rtx3070/e{e}_p8/result.json').read_text())
    assert b['dofs']==m['dofs']==k['unique_dofs']==n**3
    results.append(b)
    rows.append(dict(n=n,dofs=n**3,bspf_error=b['relative_l2'],bspf_cg_s=b['warm_cg_s'],bspf_wall_s=b['warm_wall_s'],bspf_iterations=b['runs'][-1]['iterations'],
                     nekrs_error=k['relative_l2'],nekrs_cg_s=k['warm_solve_s'],nekrs_wall_s=k['warm_wall_s'],
                     mfem_error=m['relative_l2'],mfem_cg_s=float(np.median(m['linear_samples_seconds'])),mfem_wall_s=m['warm_median_seconds']))
if not rows:raise SystemExit('No results')
with (OUT/'comparison.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
fig,axes=plt.subplots(1,3,figsize=(15,5))
for solver,label,marker in [('bspf','BSPF CUDA · FP32 tensor inverse','o'),('nekrs','NekRS · p-MG + GPU AMG','s'),('mfem','MFEM/libCEED · LOR + GPU AMG','^')]:
    for i,ax in enumerate(axes):
        key=['dofs',solver+'_cg_s',solver+'_wall_s'][i]
        ax.loglog([r[key] for r in rows],[r[solver+'_error'] for r in rows],marker+'-',label=label)
        if i==0 and solver!='bspf':continue
        for r in rows:ax.annotate(str(r['n']),(r[key],r[solver+'_error']),xytext=(4,4),textcoords='offset points',fontsize=8)
for ax,title,xlabel in zip(axes,['Matched DOF counts','Warm linear solve','Warm RHS + boundary + solve'],['Global DOFs','Seconds, median of 3','Seconds, median of 3']):
    ax.set_title(title);ax.set_xlabel(xlabel);ax.set_ylabel('Relative physical L² error');ax.grid(True,which='both',alpha=.2);ax.margins(x=.13,y=.1)
axes[0].legend(fontsize=8,loc='lower left')
fig.suptitle('RTX 3070 · same nonperiodic curved-domain turbulence MMS',fontsize=14)
fig.text(.5,.025,'Labels: global nodes per axis. Trial spaces and quadratures differ. BSPF/NekRS preconditioners: FP32; MFEM: FP64.\nWarm timings exclude setup, sampling, transfers and error audits. BSPF uses dense 1D tensor contractions.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.1,1,.94]);fig.savefig(OUT/'accuracy_walltime.png',dpi=180);fig.savefig(OUT/'accuracy_walltime.pdf')
quad=None
p=OUT/'n65_q146/result.json'
if p.exists():
    audit=json.loads(p.read_text());b=next(d for d in results if d['n']==65)
    # The exported solution is the named matrix bundle, not a polynomial surrogate.
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'curved_poisson/benchmarks/internal_soliton/cpp'))
    from bundle import read_bundle
    u=read_bundle(OUT/'n65_q130/solution.bin')['solution'];v=read_bundle(OUT/'n65_q146/solution.bin')['solution']
    quad=dict(n=65,orders=[130,146],relative_l2_errors=[b['relative_l2'],audit['relative_l2']],nodal_relative_solution_change=float(np.linalg.norm(u-v)/np.linalg.norm(v)))
    if quad['nodal_relative_solution_change']>1e-10:raise RuntimeError(quad)
report=dict(status='complete' if len(results)==5 and quad is not None else 'partial',gpu='RTX 3070',
            basis='Existing independent B-spline/Fourier axis; degree 8, period 1.5, symmetric QR nodes',
            operator='FP64 full-metric tensor Galerkin',cg='FP64 with true-residual replacement/restart',
            preconditioner='FP32 generalized-eigenvector fast diagonalization; pedantic cuBLAS, TF32 disabled',
            polynomial_guard=json.loads((OUT/'n17_q48_polynomial/result.json').read_text()),operator_quadrature_check=quad,
            results=results,comparison=rows)
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['status']);print(json.dumps(quad,indent=2));print(json.dumps(rows,indent=2))
