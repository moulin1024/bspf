"""Compare independently audited Jacobi and GPU LOR-AMG campaigns."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('jacobi',type=Path)
    parser.add_argument('amg',type=Path)
    args=parser.parse_args()
    baseline=json.loads((args.jacobi/'report.json').read_text())
    amg=json.loads((args.amg/'report.json').read_text())
    if baseline['modes_sha256']!=amg['modes_sha256']:
        raise ValueError('Different analytic datasets')
    before={(r['elements_per_axis'],r['degree'],r['quadrature']):r for r in baseline['results']}
    pairs=[]
    for r in amg['results']:
        key=(r['elements_per_axis'],r['degree'],r['quadrature'])
        if key not in before: continue
        b=before[key]
        if r['dofs']!=b['dofs'] or not r['hypre_gpu']:
            raise ValueError('DOFs or GPU execution mismatch')
        delta=abs(r['relative_l2']-b['relative_l2'])
        if delta>max(1e-10,b['relative_l2']*1e-3):
            raise ValueError('Preconditioners produce inconsistent errors')
        pairs.append(dict(elements_per_axis=key[0],degree=key[1],dofs=r['dofs'],
                          jacobi_iterations=b['iterations'],amg_iterations=r['iterations'],
                          jacobi_cg_seconds=float(np.median(b['linear_samples_seconds'])),
                          amg_cg_seconds=float(np.median(r['linear_samples_seconds'])),
                          cg_speedup=float(np.median(b['linear_samples_seconds'])/np.median(r['linear_samples_seconds'])),
                          warm_total_speedup=b['warm_median_seconds']/r['warm_median_seconds'],
                          amg_preconditioner_setup_seconds=r['preconditioner_setup_seconds'],
                          error_difference=delta))
    if not pairs: raise ValueError('No matched cases')
    (args.amg/'comparison.json').write_text(json.dumps(pairs,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for label,report in [('Jacobi',baseline),('LOR + GPU BoomerAMG',amg)]:
        keys={(p['elements_per_axis'],p['degree']) for p in pairs}
        rows=sorted([r for r in report['results'] if (r['elements_per_axis'],r['degree']) in keys],key=lambda r:r['elements_per_axis'])
        for ax,metric in zip(axes,('cg','warm','cold')):
            times=[float(np.median(r['linear_samples_seconds'])) if metric=='cg' else r['warm_median_seconds'] if metric=='warm' else r['setup_seconds']+r['first_solve_seconds'] for r in rows]
            ax.loglog(times,[r['relative_l2'] for r in rows],'o-',label=label)
            for t,r in zip(times,rows):
                ax.annotate(f"{r['elements_per_axis']}³",(t,r['relative_l2']),xytext=(4,5 if label=='Jacobi' else -13),textcoords='offset points',fontsize=8)
            ax.grid(True,which='both',alpha=.25);ax.margins(.15);ax.set_xlabel('Wall time [s]');ax.legend(fontsize=9)
    for ax,title in zip(axes,('Warm GPU CG','Warm assembly + CG + recovery','Setup + first solve')): ax.set_title(title)
    axes[0].set_ylabel('Relative physical-domain L² error in $u_x$')
    fig.suptitle('RTX 3070 · curved nonperiodic turbulence MMS · p = 8 · FP64\nSame high-order MFEM/libCEED operator; different preconditioners',fontsize=11)
    fig.savefig(args.amg/'preconditioner_comparison.png',dpi=200)
    fig.savefig(args.amg/'preconditioner_comparison.pdf')
    print(json.dumps(pairs,indent=2))

if __name__=='__main__': main()
