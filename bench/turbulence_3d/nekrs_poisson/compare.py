"""Plot measured NekRS and MFEM results; no interpolation or predicted timings."""
from pathlib import Path
import json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]/'evidence'
OUT=ROOT/'nekrs_rtx3070'
rows=[]
for path in sorted(OUT.glob('e*_p8/result.json'),key=lambda p:int(p.parent.name.split('_')[0][1:])):
    n=json.loads(path.read_text());e=n['elements_per_axis']
    m=json.loads((ROOT/f'mfem_lor_amg_rtx3070/e{e}_p8/result.json').read_text())
    rows.append(dict(elements_per_axis=e,dofs=n['unique_dofs'],free_dofs=n['free_dofs'],
                     nekrs_error=n['relative_l2'],mfem_error=m['relative_l2'],
                     nekrs_solve_s=n['warm_solve_s'],mfem_solve_s=float(np.median(m['linear_samples_seconds'])),
                     nekrs_wall_s=n['warm_wall_s'],mfem_wall_s=m['warm_median_seconds'],
                     nekrs_iterations=n['runs'][-1]['iterations'],mfem_iterations=m['iterations'],
                     nekrs_preconditioner_setup_s=n['preconditioner_setup_s'],mfem_preconditioner_setup_s=m['preconditioner_setup_seconds']))
if not rows:raise SystemExit('No completed results')
with (OUT/'comparison.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
fig,axes=plt.subplots(1,3,figsize=(15,4.6))
for solver,label,marker in [('nekrs','NekRS CUDA · p-MG + GPU AMG','o'),('mfem','MFEM/libCEED · LOR + GPU AMG','s')]:
    error=[r[solver+'_error'] for r in rows]
    for ax,key in zip(axes,['dofs',solver+'_solve_s',solver+'_wall_s']):
        ax.loglog([r[key] for r in rows],error,marker+'-',label=label)
        if ax is axes[0] and solver=='mfem':continue
        for r in rows:ax.annotate(str(r['elements_per_axis'])+'³',(r[key],r[solver+'_error']),xytext=(4,3),textcoords='offset points',fontsize=8)
for ax,title,xlabel in zip(axes,['Matching DOFs','Warm linear solve','Warm RHS + boundary + solve'],['Global unique DOFs','Seconds (median of 3)','Seconds (median of 3)']):
    ax.set_title(title);ax.set_xlabel(xlabel);ax.set_ylabel('Relative physical L² error');ax.grid(True,which='both',alpha=.25);ax.margins(x=.12,y=.12)
axes[0].legend(fontsize=8)
fig.suptitle('RTX 3070 · nonperiodic curved-domain turbulence MMS · p=8',fontsize=13)
fig.text(.5,.025,'NekRS: 9-point GLL, FP32 preconditioner. MFEM: 11-point Gauss, FP64 preconditioner.\nWarm timings exclude setup, source sampling, I/O and error audits.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.10,1,.93]);fig.savefig(OUT/'accuracy_walltime.png',dpi=180);fig.savefig(OUT/'accuracy_walltime.pdf')
(OUT/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
