# SPDX-License-Identifier: GPL-3.0-or-later
"""Matched dense versus factored sparse enrichment, with quadrature control."""
import argparse
import gc
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'/'pde'))
from shock_tube_enrichment import run_enriched,initial_state,primitives
from shock_tube_spline_enrichment import LocalRefinement,SplineSpace,SparseSplineSpace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--prefix',default='benchmarks/shock_tube/sparse_enrichment_cost')
    args=parser.parse_args()
    if args.repeats<1:parser.error('positive repeats required')
    configs={'dense_q64':(SplineSpace,64),'sparse_q64':(SparseSplineSpace,64),'sparse_q32':(SparseSplineSpace,32)}
    records={k:[] for k in configs};outputs={};spaces={}
    for repetition in range(args.repeats):
        names=list(configs)
        if repetition%2:names.reverse()
        for name in names:
            cls,q=configs[name];factory=LocalRefinement(8,2.,space_type=cls)
            gc.collect();start=perf_counter()
            result=run_enriched(65,.0001,.1,.01,.005,q,space_factory=factory)
            seconds=perf_counter()-start
            records[name].append(dict(seconds=seconds,summary=result['summary']))
            outputs[name]=result['output'];spaces[name]=factory.previous
            print(repetition+1,name,round(seconds,5),result['summary']['enrichment_count'],flush=True)
    rhs_times={k:[] for k in configs}
    states={k:s.fit(initial_state(s.base.q,.01)) for k,s in spaces.items()}
    for k,s in spaces.items():s.rhs(states[k])
    for i in range(21):
        names=list(configs)
        if i%2:names.reverse()
        for k in names:
            start=perf_counter()
            for _ in range(5):spaces[k].rhs(states[k])
            rhs_times[k].append((perf_counter()-start)/5)
    ref=np.load('benchmarks/shock_tube/enrichment_n65_mu0001_fine.npz')
    x=ref['x'];target=primitives(ref['reference']);summary={}
    for name in configs:
        s=spaces[name]
        if hasattr(s,'H'):
            sample_bytes=s.H.nbytes+s.Hx.nbytes
        else:
            sample_bytes=sum(v.data.nbytes+v.indices.nbytes+v.indptr.nbytes for v in (s.S,s.Sx))
        summary[name]=dict(median_seconds=float(np.median([r['seconds'] for r in records[name]])),
            median_rhs_seconds=float(np.median(rhs_times[name])),
            enrichment_sampling_array_bytes=sample_bytes,
            max_conservative_difference_from_dense=float(abs(outputs[name]-outputs['dense_q64']).max()),
            metrics={field:dict(l1=float(np.trapezoid(abs(v-r),x)),linf=float(abs(v-r).max()))
                for field,v,r in zip(('rho','u','p','T'),primitives(outputs[name]),target)})
    for name in configs:
        summary[name]['speedup']=summary['dense_q64']['median_seconds']/summary[name]['median_seconds']
        summary[name]['rhs_speedup']=summary['dense_q64']['median_rhs_seconds']/summary[name]['median_rhs_seconds']
    report=dict(parameters=dict(n=65,mu=.0001,time=.1,width=.01,interval=.005,cfl=.35,refinement=8,halfwidth=2.,repeats=args.repeats),
        notes=['Timing includes setup, evolution and final probes; excludes plotting and file output.',
               'RHS timing uses the final space; adaptive evolution grows from 33 to 172 enrichment modes.',
               'Sampling-array memory counts H/Hx or CSR S/Sx only, not total solver memory.'],
        summary=summary,runs=records,rhs_times=rhs_times)
    prefix=Path(args.prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2))
    np.savez(prefix.with_suffix('.npz'),x=x,reference=ref['reference'],**outputs)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
