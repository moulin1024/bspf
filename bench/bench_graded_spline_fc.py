# SPDX-License-Identifier: GPL-3.0-or-later
"""Alternate warm serial BSPF/FC runs at the 1e-8 Becker-shock threshold."""
import argparse,json,sys
from pathlib import Path
from time import perf_counter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/pde'))
from viscous_traveling_shock import solve,primitives
from graded_spline_shock import graded_factory
from compiled_galerkin_step import warmup,compiled_segment
from prior_art_traveling_shock import solve_fc


def bspf(intervals=38,local_quadrature=8,outer_quadrature=8,cfl=1.,kernel='jit',final_time=.05,output_points=64001):
    result=solve(25,True,final_time=final_time,cfl=cfl,quadrature=outer_quadrature,
        recondition=True,output_points=output_points,line_options=dict(n_basis=10),
        enrichment_factory=graded_factory(degree=13,intervals=intervals,core_scale=.25,
            support_width=8.,core_padding=.5,quadrature=local_quadrature,compiled=kernel!='structured'),
        segment_stepper=compiled_segment if kernel=='jit' else None,detector_spacing=1/48)
    result['summary'].update(method='BSPF + graded physical B-splines',kernel=kernel,
        spline_degree=13,graded_intervals=intervals,core_scale=.25,support_width=8.,
        core_padding=.5,local_quadrature=local_quadrature,refinement=None,halfwidth=None)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--fc-n',type=int,default=2162)
    parser.add_argument('--output',default='benchmarks/traveling_shock/graded_spline_fc_1e8')
    args=parser.parse_args()
    if args.repeats<1:parser.error('repeats must be positive')
    data=ROOT/'benchmarks/traveling_shock/prior_art/fc_data/FCGram_data_d5_C25.mat'
    if not data.exists():raise SystemExit('Run python bench/fetch_fc_data.py first')
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    def fc(**kwargs):
        options=dict(cfl=1.,filter_order=0,fused_flux=True)
        options.update(kwargs)
        return solve_fc(args.fc_n,data,**options)
    start=perf_counter();warmup()
    bspf(final_time=1e-5,output_points=257)
    fc(final_time=1e-5,output_points=257)
    preparation=perf_counter()-start
    records=[];profiles={}
    def run(name,fn):
        start=perf_counter();result=fn();wall=perf_counter()-start
        result['summary']['end_to_end_seconds']=wall
        error=max(v['linf'] for v in result['summary']['metrics'].values())
        record=dict(case=name,seconds=wall,error=error,steps=result['summary']['steps'],
                    coefficients=result['summary'].get('final_modes',result['summary']['n']))
        records.append(record);profiles[name]=result['output']
        (out/f'{name}.json').write_text(json.dumps(result['summary'],indent=2))
        np.savez(out/f'{name}.npz',**{k:v for k,v in result.items() if k!='summary'})
        print(json.dumps(record),flush=True)
        return result
    for repetition in range(args.repeats):
        order=[('bspf',bspf),('fc',fc)]
        if repetition%2:order.reverse()
        for name,fn in order:run(f'{name}_{repetition}',fn)
    run('bspf_half_step',lambda:bspf(cfl=.5))
    run('bspf_refined_quadrature',lambda:bspf(local_quadrature=16,outer_quadrature=12))
    run('fc_half_step',lambda:fc(cfl=.5))
    for intervals in (34,36,40):run(f'bspf_intervals_{intervals}',lambda m=intervals:bspf(intervals=m))
    for kernel in ('structured','python'):run(f'bspf_kernel_{kernel}',lambda k=kernel:bspf(kernel=k))
    controls={}
    for name,reference in [('bspf_half_step','bspf_0'),('bspf_refined_quadrature','bspf_0'),
                           ('fc_half_step','fc_0'),('bspf_kernel_structured','bspf_0'),('bspf_kernel_python','bspf_0')]:
        controls[name]=float(max(np.max(abs(a-b)) for a,b in zip(primitives(profiles[name]),primitives(profiles[reference]))))
    timing={}
    for name in ('bspf','fc'):
        values=[r['seconds'] for r in records if r['case'] in [f'{name}_{i}' for i in range(args.repeats)]]
        timing[name]=dict(median=float(np.median(values)),minimum=min(values),maximum=max(values),runs=values)
    summary=dict(threshold=1e-8,fc_n=args.fc_n,repeats=args.repeats,warmup_seconds=preparation,
        timing=timing,speedup_fc_over_bspf=timing['fc']['median']/timing['bspf']['median'],
        controls=controls,records=records,
        timing_scope='Fresh setup, adaptive evolution and common dense error evaluation; imports, one-time warmup and file output excluded.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    for record in records:
        if record['case'] in ('bspf_0','fc_0','bspf_half_step','bspf_refined_quadrature','fc_half_step'):
            assert record['error']<1e-8,record
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
