"""Same-input comparison and physical-error audit of shared tensor contractions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from run import HERE, EXE, DATA, HybridAxis, Sampler, read_bundle, audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,nargs='+',default=[17,33,65,97,129]);args=p.parse_args()
    root=HERE.parent/'evidence';out=root/'bspf_cuda3d_shared';out.mkdir(exist_ok=True)
    sampler=Sampler(DATA);rows=[]
    env=os.environ.copy();sdk=os.environ.get('BSPF_HPC_SDK','/home/moulin/Workspace/nvidia/hpc_sdk/Linux_x86_64/26.5')
    env['LD_LIBRARY_PATH']=f'{sdk}/math_libs/lib64:{sdk}/cuda/13.2/lib64'
    binary_hash=hashlib.sha256(EXE.read_bytes()).hexdigest()
    for n in args.n:
        q=max(96,(3*n+1)//2);source=root/'bspf_cuda3d_parity'/f'n{n}_q{q}'
        folder=out/f'n{n}_q{q}';folder.mkdir(exist_ok=True)
        # Two alternating pairs, each invocation has a warmup and three solves.
        samples={0:[],1:[]}
        for repeat in range(2):
            for mode in (0,1):
                prefix=folder/f'mode{mode}_repeat{repeat}'
                subprocess.run([str(EXE),str(source/'input.bin'),str(prefix)],
                               env=dict(env,BSPF_POISSON_SHARED=str(mode)),check=True)
                d=json.loads(prefix.with_suffix('.json').read_text())
                assert d['parity_optimized'] and d['shared_contractions']==bool(mode)
                samples[mode].extend(d['runs'][1:])
        result=d
        on=read_bundle(folder/'mode1_repeat1.bin')['solution'].reshape((n,)*3)
        off=read_bundle(folder/'mode0_repeat1.bin')['solution'].reshape((n,)*3)
        change=float(np.linalg.norm(on-off)/np.linalg.norm(off))
        assert change<1e-11
        wall=np.ones(on.shape,bool);wall[1:-1,1:-1,1:-1]=False
        boundary=float(np.max(abs(on[wall]-off[wall])));assert boundary==0
        axis=HybridAxis((n-9)//2,degree=8,nodes='symmetric_qr')
        audits=[audit(axis,on,sampler,k,False) for k in (max(96,q+8),max(96,q+8)+8)]
        assert abs(audits[0]['relative_l2']/audits[1]['relative_l2']-1)<1e-4
        before=float(np.median([r['cg_s'] for r in samples[0]]))
        after=float(np.median([r['cg_s'] for r in samples[1]]))
        assert max(r['relative_residual'] for r in samples[1])<1e-12
        result.update(runs=samples[1],baseline_runs=samples[0],warm_cg_s=after,
            warm_wall_s=float(np.median([r['wall_s'] for r in samples[1]])),
            baseline_cg_s=before,speedup=before/after,relative_l2=audits[-1]['relative_l2'],audits=audits,
            relative_solution_change=change,boundary_error=boundary,binary_sha256=binary_hash,
            input_path=str(source/'input.bin'),modes_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
            timing_protocol='Two alternating off/on pairs; each invocation has one warmup and three timed solves')
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        rows.append({k:result[k] for k in ['n','q','baseline_cg_s','warm_cg_s','speedup','relative_l2','relative_solution_change','device_bytes']})
        print(json.dumps(rows[-1]),flush=True)
    (out/'optimization.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
