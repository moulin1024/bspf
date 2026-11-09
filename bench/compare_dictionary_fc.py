"""Alternating fresh-setup warm timings for dictionary BSPF versus tuned FC."""
import argparse,json,sys
from pathlib import Path
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/pde'))
from bench_dictionary_spline_shock import run
from prior_art_traveling_shock import solve_fc
from compiled_galerkin_step import warmup
from shock_tube_1d import primitives


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--intervals',type=int,default=38)
    p.add_argument('--probe-cells',type=int,default=128)
    p.add_argument('--interval',type=float,default=.005)
    p.add_argument('--fc-n',type=int,default=2162)
    p.add_argument('--quadrature',type=int,default=8)
    p.add_argument('--outer-quadrature',type=int,default=8)
    p.add_argument('--output',default='benchmarks/traveling_shock/dictionary_fc_1e8')
    a=p.parse_args();folder=ROOT/a.output;folder.mkdir(parents=True,exist_ok=True)
    data=ROOT/'benchmarks/traveling_shock/prior_art/fc_data/FCGram_data_d5_C25.mat'
    warmup()
    def bspf(**extra):
        options=dict(intervals=a.intervals,probe_cells=a.probe_cells,interval=a.interval,
                     quadrature=a.quadrature,outer_quadrature=a.outer_quadrature)
        options.update(extra);return run(**options)
    def fc(**extra):
        options=dict(cfl=1.,filter_order=0,fused_flux=True,constant_boundaries=True)
        options.update(extra);return solve_fc(a.fc_n,data,**options)
    bspf(final_time=1e-5,output_points=257);fc(final_time=1e-5,output_points=257)
    records=[];profiles={}
    def save(name,function):
        tick=perf_counter();r=function();wall=perf_counter()-tick
        r['summary']['end_to_end_seconds']=wall
        row=dict(case=name,seconds=wall,error=max(v['linf'] for v in r['summary']['metrics'].values()),
                 coefficients=r['summary'].get('final_modes',r['summary']['n']),steps=r['summary']['steps'])
        records.append(row);profiles[name]=r['output']
        (folder/f'{name}.json').write_text(json.dumps(r['summary'],indent=2))
        np.savez(folder/f'{name}.npz',**{k:v for k,v in r.items() if k!='summary'})
        print(json.dumps(row),flush=True)
    for repeat in range(a.repeats):
        order=[('bspf',bspf),('fc',fc)]
        if repeat%2:order.reverse()
        for name,fn in order:save(f'{name}_{repeat}',fn)
    controls=[('half_step',dict(cfl=.5)),('quadrature',dict(quadrature=16,outer_quadrature=12)),
        ('probes',dict(probe_cells=512)),('half_interval',dict(interval=a.interval/2)),
        ('shifted',dict(center=.417)),('shifted_left',dict(center=.317)),
        ('reverse',dict(speed=-.5)),('frozen',dict(freeze=True)),('trace',dict(trace_points=8193))]
    for name,options in controls:
        try:save(f'bspf_{name}',lambda kw=options:bspf(**kw))
        except Exception as exc:
            failure=dict(case=f'bspf_{name}',failure=str(exc));records.append(failure)
            print(json.dumps(failure),flush=True)
    save('fc_half_step',lambda:fc(cfl=.5))
    difference={name:float(max(np.max(abs(x-y)) for x,y in zip(primitives(value),primitives(profiles['bspf_0']))))
        for name,value in profiles.items() if name in ('bspf_half_step','bspf_quadrature','bspf_probes','bspf_half_interval')}
    timing={name:dict(median=float(np.median(times)),minimum=min(times),maximum=max(times),runs=times)
        for name in ('bspf','fc') for times in [[r['seconds'] for r in records if r['case'] in [f'{name}_{i}' for i in range(a.repeats)]] ]}
    result=dict(threshold=1e-8,timing=timing,speedup_fc_over_bspf=timing['fc']['median']/timing['bspf']['median'],
        differences=difference,records=records,
        timing_scope='Serial alternating warm runs, fresh setup/selection/integration/dense error assessment. Imports, JIT warmup and file output excluded.')
    (folder/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
