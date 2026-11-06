# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare 237 global BSPF modes with 65 + 172 adaptive spline modes."""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples'/'pde'))
from shock_tube_1d import BSPFNavierStokes, primitives, initial_state
from shock_tube_enrichment import run_enriched
from shock_tube_spline_enrichment import LocalRefinement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--reference',default='benchmarks/shock_tube/enrichment_n65_mu0001_fine.npz')
    parser.add_argument('--prefix',default='benchmarks/shock_tube/equal_modes237')
    args=parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    ref=np.load(args.reference)
    x=ref['x']
    target=primitives(ref['reference'])
    configs=('global237_q12','global237_q64','enriched237_q64')
    rows={k:[] for k in configs}
    outputs={}
    warm=BSPFNavierStokes(65,mu=.0001)
    warm.rhs(initial_state(warm.x,.01))
    for repetition in range(args.repeats):
        for name in (configs if repetition%2==0 else configs[::-1]):
            gc.collect()
            start=perf_counter()
            if name.startswith('global'):
                solver=BSPFNavierStokes(237,mu=.0001,quadrature=12 if name.endswith('q12') else 64)
                result=solver.solve(.1,.01,.35,progress=True)
            else:
                result=run_enriched(n=65,mu=.0001,final_time=.1,width=.01,interval=.005,
                    quadrature=64,cfl=.35,space_factory=LocalRefinement(8,2.))
            seconds=perf_counter()-start
            np.testing.assert_allclose(result['probes'],x,rtol=0,atol=0)
            outputs[name]=result['output']
            rows[name].append(dict(seconds=seconds,summary=result['summary']))
            print(f'{repetition+1}/{args.repeats} {name}: {seconds:.6f}s, {result["summary"]["steps"]} steps',flush=True)
    summary={}
    for name in configs:
        values=primitives(outputs[name])
        metrics={field:dict(l1=float(np.trapezoid(abs(v-r),x)),linf=float(abs(v-r).max()),
            total_variation=float(np.sum(abs(np.diff(v)))))
            for field,v,r in zip(('rho','u','p','T'),values,target)}
        timings=[r['seconds'] for r in rows[name]]
        last=rows[name][-1]['summary']
        summary[name]=dict(median_seconds=float(np.median(timings)),min_seconds=min(timings),max_seconds=max(timings),
            steps=last['steps'],final_modes=237,metrics=metrics,
            max_relative_mass_drift=last['max_relative_mass_drift'],max_relative_energy_drift=last['max_relative_energy_drift'],
            rejected_stages=last['rejected_stages'])
    delta=abs(outputs['global237_q12']-outputs['global237_q64'])
    report=dict(parameters=dict(mu=.0001,final_time=.1,width=.01,cfl=.35,repeats=args.repeats,reference=args.reference),
        environment=dict(platform=platform.platform(),threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')}),
        timing_scope='Includes setup, evolution, diagnostics and final probes. Excludes imports, plotting and file output.',
        notes=['Equal final coefficient counts; the enriched space grows during evolution.',
               'Same analytic initial data, interpolated by global BSPF and L2-projected by the enriched solver.'],
        global_quadrature_max_conservative_difference=float(delta.max()),summary=summary,runs=rows)
    prefix=Path(args.prefix)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2))
    np.savez(prefix.with_suffix('.npz'),x=x,reference=ref['reference'],**outputs)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    styles=[('global237_q12','Global BSPF: 237, q12'),('enriched237_q64','BSPF + splines: 65 + 172, q64'),('reference','Reference: 1025')]
    for key,label in styles:
        data=ref['reference'] if key=='reference' else outputs[key]
        values=primitives(data)
        for ax,title,v in zip(axes.flat,('Density','Velocity','Pressure','Temperature'),values):
            ax.plot(x,v,label=label,lw=1.1)
            ax.set(xlabel='x',title=title,xlim=(.35,.85))
        if key!='reference':
            for ax,title,index in ((axes.flat[4],'Velocity error',1),(axes.flat[5],'Pressure error',2)):
                ax.plot(x,values[index]-target[index],label=label,lw=1.1)
                ax.set(xlabel='x',title=title,xlim=(.64,.72))
                ax.axhline(0,color='gray',lw=.5)
    for ax in axes.flat:
        ax.legend(fontsize=8)
    fig.suptitle('Equal final mode count: mu=1e-4, t=0.1, initial width=0.01')
    fig.savefig(prefix.with_suffix('.png'),dpi=160)
    print(json.dumps(report['summary'],indent=2),flush=True)
    print('Quadrature max difference:',report['global_quadrature_max_conservative_difference'])


if __name__=='__main__':
    main()
