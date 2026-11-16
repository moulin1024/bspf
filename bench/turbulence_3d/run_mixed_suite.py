"""Run the five GPU benchmarks sequentially, retaining per-task logs and status."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OUT = HERE/'evidence/gaussian_mms_rtx3070'
MODES = HERE/'evidence/nonperiodic_curved_gaussian/modes.npz'


def main():
    global OUT,MODES
    parser=argparse.ArgumentParser()
    parser.add_argument('--max-n',type=int,default=129,choices=[17,33,65,97,129,145])
    parser.add_argument('--modes',type=Path,default=MODES)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.output.resolve();MODES=args.modes.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               CUDA_PATH=os.environ.get('CUDA_PATH','/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6'),
               LD_LIBRARY_PATH=os.environ.get('LD_LIBRARY_PATH','/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/math_libs/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/lib64'))
    tasks = []
    for solver, driver in [('bspf', 'bspf_poisson'), ('chebyshev', 'chebyshev_poisson')]:
        for n in [v for v in [17,33,65,97,129,145] if v<=args.max_n]:
            target = OUT/solver/f'n{n}_q{max(96,(3*n+1)//2)}'/'result.json'
            tasks.append((f'{solver}_{n}', [sys.executable, str(HERE/driver/'run.py'), '--n', str(n),
                          '--modes', str(MODES), '--output', str(OUT/solver)], target, env))
    for e in [v for v in [2,4,8,12,16,18] if 8*v+1<=args.max_n]:
        tasks.append((f'nekrs_{e}', [sys.executable, str(HERE/'nekrs_poisson/run.py'), '--elements', str(e),
                      '--modes', str(MODES), '--output', str(OUT/'nekrs')], OUT/f'nekrs/e{e}_p8/result.json', env))
    mfem = [sys.executable, str(HERE/'mfem_poisson/run.py'), '--binary', '/tmp/bspf-mfem-poisson3d-fp32/mfem_poisson3d',
            '--modes', str(MODES), '--output', str(OUT/'mfem_gpu_rhs'), '--preconditioner', 'lor-amg-fp32',
            '--cases', ','.join(f'{e}:8' for e in [2,4,8,12,16,18] if 8*e+1<=args.max_n)]
    if (OUT/'mfem_gpu_rhs/report.json').exists():
        mfem.append('--resume')
    tasks.append(('mfem', mfem, None, env))
    tenv = env.copy()
    tenv.pop('LD_LIBRARY_PATH', None)
    for n in [v for v in [17,33,65,97,129,145] if v<=args.max_n]:
        target = OUT/f'fc/turbulence_n{n}'
        tasks.append((f'fc_{n}', [os.environ.get('BSPF_TORCH_PYTHON','/tmp/bspf-fc-torch/bin/python'), str(HERE/'fc_poisson/run.py'),
                      '--n', str(n), '--modes', str(MODES), '--audit', '--out', str(target)], target/'result.json', tenv))
    status = dict(modes=str(MODES),modes_sha256=hashlib.sha256(MODES.read_bytes()).hexdigest(), tasks=[])
    for name, command, target, taskenv in tasks:
        if target is not None and target.exists():
            data = json.loads(target.read_text())
            assert data.get('modes_sha256', data.get('data_sha256')) == status['modes_sha256']
            record = dict(task=name, status='cached', result=str(target))
        else:
            print(f'START {name}', flush=True)
            start = time.time()
            with (OUT/f'{name}.log').open('w') as log:
                proc = subprocess.run(command, env=taskenv, stdout=log, stderr=subprocess.STDOUT)
            record = dict(task=name, status='passed' if proc.returncode == 0 else 'failed',
                          returncode=proc.returncode, elapsed_s=time.time()-start, command=command)
        status['tasks'].append(record)
        (OUT/'suite_status.json').write_text(json.dumps(status, indent=2)+'\n')
        print(json.dumps(record), flush=True)
    if any(record['status']=='failed' for record in status['tasks']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
