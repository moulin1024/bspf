# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare saved pre-v2 implementation against current nested-transfer solver.

Each timing is a fresh process. Imports/file IO are excluded by run_enriched's
internal timer. The baseline directory contains snapshots of the two old example
modules, saved before implementation changes; it is required for legacy timings.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

PROGRAM=r'''
import sys,json,numpy as np
from shock_tube_enrichment import run_enriched
from shock_tube_spline_enrichment import LocalRefinement,SparseSplineSpace
config=json.loads(sys.argv[1])
r=run_enriched(n=65,mu=.0001,final_time=.1,width=.01,interval=.005,
    quadrature=config['quadrature'],cfl=config['cfl'],space_factory=LocalRefinement(8,2.,space_type=SparseSplineSpace))
np.savez(config['output'],x=r['probes'],output=r['output'])
print(json.dumps(r['summary']))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--baseline-cfl',type=float,default=.35)
    parser.add_argument('--baseline-dir',default='benchmarks/shock_tube/enrichment_before_v2')
    parser.add_argument('--prefix',default='benchmarks/shock_tube/enrichment_v2_benchmark')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    baseline=root/args.baseline_dir
    if not (baseline/'shock_tube_enrichment.py').exists():parser.error('baseline snapshots missing')
    if args.repeats<1:parser.error('positive repeats required')
    prefix=root/args.prefix;prefix.parent.mkdir(parents=True,exist_ok=True)
    previous_key=f'previous_q32_cfl{round(100*args.baseline_cfl):03d}'
    variants={previous_key:(True,32,args.baseline_cfl),'updated_q32_cfl035':(False,32,.35),'updated_q32_cfl070':(False,32,.7)}
    results={k:[] for k in variants};outputs={}
    for repetition in range(args.repeats):
        names=list(variants)
        if repetition%2:names.reverse()
        for name in names:
            legacy,q,cfl=variants[name]
            file=Path(str(prefix)+'_'+name+'.npz')
            env=os.environ.copy()
            paths=([str(baseline)] if legacy else [])+[str(root/'examples/pde'),str(root/'python')]
            env['PYTHONPATH']=os.pathsep.join(paths)
            env['OPENBLAS_NUM_THREADS']=env['OMP_NUM_THREADS']='1'
            process=subprocess.run([sys.executable,'-c',PROGRAM,json.dumps(dict(quadrature=q,cfl=cfl,output=str(file)))],
                cwd=root,env=env,text=True,capture_output=True,check=True)
            row=json.loads(process.stdout)
            results[name].append(row)
            outputs[name]=np.load(file)['output']
            print(repetition+1,name,round(row['seconds'],5),row['steps'],row['enrichment_count'],flush=True)
    ref=np.load(root/'benchmarks/shock_tube/enrichment_n65_mu0001_fine.npz');x=ref['x']
    sys.path.insert(0,str(root/'examples/pde'))
    from shock_tube_1d import primitives
    target=primitives(ref['reference']);old=outputs[previous_key];summary={}
    for name in variants:
        rows=results[name]
        summary[name]=dict(median_seconds=float(np.median([r['seconds'] for r in rows])),
            steps=rows[-1]['steps'],enrichment_count=rows[-1]['enrichment_count'],
            max_conservative_difference_from_previous=float(abs(outputs[name]-old).max()),
            metrics={key:dict(l1=float(np.trapezoid(abs(v-r),x)),linf=float(abs(v-r).max()))
                for key,v,r in zip(('rho','u','p','T'),primitives(outputs[name]),target)})
    for name in variants:
        summary[name]['speedup']=summary[previous_key]['median_seconds']/summary[name]['median_seconds']
    report=dict(parameters=dict(mu=.0001,time=.1,n=65,width=.01,interval=.005,refinement=8,halfwidth=2.,repeats=args.repeats),
        baseline_snapshot=str(baseline),summary=summary,runs=results,
        note='Baseline CFL and mode counts are recorded per run; snapshots determine the previous implementation.')
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2))
    np.savez(prefix.with_suffix('.npz'),x=x,reference=ref['reference'],**outputs)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
