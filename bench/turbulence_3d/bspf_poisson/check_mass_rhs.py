"""Compare GPU mass loads/solutions with the explicit quadrature RHS path."""
import argparse
import json
import os
import subprocess
from pathlib import Path

import cupy as cp
import numpy as np
from run import read_bundle, write_bundle, tensor, EXE


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    rows=[]
    for folder in sorted((args.root/'nodal-mass').glob('n*')):
        baseline=args.root/'quadrature'/folder.name
        a=read_bundle(folder/'input.bin');b=read_bundle(baseline/'input.bin')
        basis=cp.asarray(b['B']);q,n=basis.shape
        reference=cp.asnumpy(tensor(cp.asarray((b['measure']*b['forcing']).reshape(q,q,q)),basis.T,basis.T,basis.T,xp=cp))
        load_error=float(np.linalg.norm(a['load'].ravel()-reference.ravel())/np.linalg.norm(reference.ravel()))
        u=read_bundle(folder/'solution.bin')['solution'];v=read_bundle(baseline/'solution.bin')['solution']
        solution_error=float(np.linalg.norm(u-v)/np.linalg.norm(v))
        result=json.loads((folder/'result.json').read_text());old=json.loads((baseline/'result.json').read_text())
        if load_error>2e-12 or solution_error>1e-10 or not result['preassembled_load']:
            raise RuntimeError((load_error,solution_error))
        rows.append(dict(n=n,q=q,load_relative_difference=load_error,solution_relative_difference=solution_error,
                         mass_setup_s=result['rhs_mass_setup_s'],mass_apply_s=result['rhs_mass_apply_s'],
                         mass_bytes=result['rhs_mass_bytes'],cached_load_solve_s=result['warm_wall_s'],
                         quadrature_rhs_solve_s=old['warm_wall_s'],relative_l2=result['relative_l2']))
    # Exercise zero and invalid preassembled loads with the same native executable.
    sample=next((args.root/'nodal-mass').glob('n17_*'))
    env=os.environ.copy();env['LD_LIBRARY_PATH']=os.environ['BSPF_NATIVE_LIBRARY_PATH']
    for invalid in [False,True]:
        a=read_bundle(sample/'input.bin');a['load'].fill(0);a['boundary'].fill(0)
        if invalid:a['load'][0,0]=np.nan
        check=args.root/('invalid_load' if invalid else 'zero_load');check.mkdir()
        write_bundle(check/'input.bin',a)
        r=subprocess.run([str(EXE),str(check/'input.bin'),str(check/'solution')],env=env,capture_output=True,text=True)
        if invalid:
            if r.returncode==0:raise RuntimeError('Nonfinite load accepted')
        else:
            if r.returncode:raise RuntimeError(r.stderr)
            stats=json.loads((check/'solution.json').read_text())
            if any(s['iterations'] for s in stats['runs']) or np.any(read_bundle(check/'solution.bin')['solution']):
                raise RuntimeError('Zero load regression')
    report=dict(rows=rows,zero_and_nonfinite_load_checks='passed',
                timing_note='Mass setup/application are preprocessing; cached-load solve excludes both. Quadrature solve includes RHS projection.')
    (args.root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
