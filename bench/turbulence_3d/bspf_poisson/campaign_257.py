"""Same-GPU alternating benchmark and independent audit for the 257^3 goal."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from run import read_bundle, audit, Sampler, HybridAxis


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--modes',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trials',type=int,default=3)
    args=p.parse_args()
    if args.trials<3:p.error('At least three alternating trials required')
    args.output.mkdir(parents=True,exist_ok=False)
    previous=json.loads((args.case/'result.json').read_text())
    if previous['n']!=257 or previous['q']!=386 or previous.get('rhs_assembly')!='nodal-mass':
        raise ValueError('Expected the fixed 257^3, q=386 precomputed-mass benchmark')
    if previous['modes_sha256']!=sha(args.modes):raise ValueError('MMS changed')
    env=os.environ.copy();env['LD_LIBRARY_PATH']=os.environ['BSPF_NATIVE_LIBRARY_PATH']
    report=dict(n=257,q=386,input_sha256=sha(args.case/'input.bin'),modes_sha256=sha(args.modes),
                optimization_environment={k:v for k,v in env.items() if k.startswith('BSPF_POISSON_')},
                binary_sha256={k:sha(v) for k,v in [('baseline',args.baseline),('candidate',args.candidate)]},
                runs={'baseline':[],'candidate':[]},comparison=[],
                timing_scope='Native warm RHS/lifting/CG/recovery, one warmup then three samples per invocation; setup and audits excluded')
    for trial in range(args.trials):
        order=['baseline','candidate'] if trial%2==0 else ['candidate','baseline']
        outputs={}
        for name in order:
            directory=args.output/f'{trial}_{name}';directory.mkdir()
            prefix=directory/'solution'
            command=[str(getattr(args,name)),str(args.case/'input.bin'),str(prefix)]
            if trial==0:command.append('check')
            start=time.perf_counter()
            with (directory/'native.log').open('w') as log:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            data=json.loads(prefix.with_suffix('.json').read_text())
            if not data['preassembled_load'] or data['operator_precision']!='float64':raise RuntimeError('Changed solve configuration')
            if any(r['relative_residual']>1e-11 for r in data['runs']):raise RuntimeError('Failed true residual')
            data['process_s']=time.perf_counter()-start
            report['runs'][name].append(data)
            outputs[name]=read_bundle(prefix.with_suffix('.bin'))['solution']
            print(f'trial={trial} {name} warm={np.median([r["wall_s"] for r in data["runs"][1:]]):.6f}s '
                  f'iterations={data["runs"][-1]["iterations"]} parity={data["parity_optimized"]}',flush=True)
        u,v=outputs['baseline'],outputs['candidate']
        difference=float(np.linalg.norm(u-v)/np.linalg.norm(u))
        if difference>1e-11:raise RuntimeError(f'Solution changed by {difference}')
        comparison=dict(trial=trial,solution_relative_difference=difference,solution_bitwise_equal=bool(np.array_equal(u,v)))
        if trial==0:
            a=read_bundle(args.output/'0_baseline/solution_check.bin')
            b=read_bundle(args.output/'0_candidate/solution_check.bin')
            comparison['operators']={k:dict(relative_difference=float(np.linalg.norm(a[k]-b[k])/np.linalg.norm(a[k])),
                                           bitwise_equal=bool(np.array_equal(a[k],b[k]))) for k in a}
            if any(v['relative_difference']>2e-12 for v in comparison['operators'].values()):
                raise RuntimeError('Operator or preconditioner changed')
        report['comparison'].append(comparison)
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    samples={name:[r['wall_s'] for d in invocations for r in d['runs'][1:]] for name,invocations in report['runs'].items()}
    medians={name:float(np.median(v)) for name,v in samples.items()}
    report.update(warm_samples_s=samples,median_s=medians,speedup=medians['baseline']/medians['candidate'])
    # Independent physical error integrals, never used by the solver/stopping.
    axis=HybridAxis(124,degree=8,nodes='symmetric_qr')
    sampler=Sampler(args.modes)
    report['audits']=[audit(axis,outputs['candidate'].reshape(257,257,257),sampler,q,False) for q in [394,402]]
    old_error=previous['relative_l2']
    if any(d['relative_l2']>max(1.1*old_error,old_error+1e-13) for d in report['audits']):
        raise RuntimeError('Independent physical error regressed')
    report['validated']=True
    report['target_achieved']=report['speedup']>=2
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['median_s','speedup','audits','validated','target_achieved']},indent=2),flush=True)


if __name__=='__main__':main()
