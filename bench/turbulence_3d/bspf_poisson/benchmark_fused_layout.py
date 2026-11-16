"""Alternating on/off trials on identical saved Gaussian MMS inputs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'bench/curved_poisson/benchmarks/internal_soliton/cpp'))
from bundle import read_bundle


def main():
    evidence=ROOT/'bench/turbulence_3d/evidence/gaussian_mms_rtx3070'
    out=evidence/'bspf_fused_layout';out.mkdir(exist_ok=True)
    exe=Path('/tmp/bspf-poisson3d-cuda-build/bspf_poisson3d')
    sdk=os.environ.get('BSPF_HPC_SDK','/home/moulin/Workspace/nvidia/hpc_sdk/Linux_x86_64/26.5')
    env=os.environ.copy();env['LD_LIBRARY_PATH']=f'{sdk}/math_libs/lib64:{sdk}/cuda/13.2/lib64'
    env.update(BSPF_POISSON_PARITY='1',BSPF_POISSON_SHARED='1')
    report=dict(binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),cases=[])
    for n in [65,97,129]:
        q=max(96,(3*n+1)//2);source=evidence/f'bspf/n{n}_q{q}'
        samples={'0':[],'1':[]};solutions={}
        for trial in range(3):
            for mode in (('0','1') if trial%2==0 else ('1','0')):
                folder=out/f'n{n}'/f'mode{mode}_trial{trial}';folder.mkdir(parents=True,exist_ok=True)
                env['BSPF_POISSON_FUSED_LAYOUT']=mode
                subprocess.run([str(exe),str(source/'input.bin'),str(folder/'solution')],env=env,check=True)
                d=json.loads((folder/'solution.json').read_text())
                assert max(r['relative_residual'] for r in d['runs'])<1e-11
                samples[mode].extend(r['wall_s'] for r in d['runs'][1:])
                solutions[mode]=read_bundle(folder/'solution.bin')['solution']
                print(f'n={n} trial={trial} fused={mode} wall={np.median([r["wall_s"] for r in d["runs"][1:]]):.6f}',flush=True)
        delta=float(np.linalg.norm(solutions['0']-solutions['1'])/np.linalg.norm(solutions['0']))
        assert delta<1e-12
        off,on=(float(np.median(samples[k])) for k in ('0','1'))
        report['cases'].append(dict(n=n,q=q,off_s=off,on_s=on,speedup=off/on,
            relative_solution_change=delta,bitwise_equal=bool(np.array_equal(solutions['0'],solutions['1'])),samples=samples))
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
