"""Run the unmodified JOREK MAST-U regression input, then compare its final field."""
import argparse
import os
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from campaign import ROOT, Restart, run

CASE = Path(os.environ.get('JOREK_SOURCE', ROOT / 'third_party/JOREK')).resolve()/'reg_tests/testcases/grid_with_patches_mastu_303'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model303',type=Path,help='one-harmonic model-303 JOREK executable')
    parser.add_argument('eqdsk',type=Path,help='original MAST-U regression eqdsk.dat')
    parser.add_argument('out',type=Path,help='new directory for the run')
    parser.add_argument('--mpiexec',default='mpiexec')
    args = parser.parse_args()
    binary = args.model303.resolve(strict=True)
    eqdsk = args.eqdsk.resolve(strict=True)
    out = args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(CASE/'input',out/'input')
    shutil.copyfile(eqdsk,out/'eqdsk.dat')
    with (out/'input').open() as namelist,(out/'jorek.log').open('w') as log:
        start = time.perf_counter()
        subprocess.run([args.mpiexec,'-n','1',str(binary)],stdin=namelist,
                       stdout=log,stderr=subprocess.STDOUT,cwd=out,check=True)
        elapsed = time.perf_counter()-start
    log = (out/'jorek.log').read_text(errors='replace')
    if 'Fixed boundary equilibrium converged:' not in log or 'not fully converged' in log:
        raise RuntimeError('No verified equilibrium convergence; inspect jorek.log')
    restart = out/'jorek_restart.h5'
    report = dict(protocol='fixed-current model-303 replay',
                  native_end_to_end_s=elapsed, native_processes=1,
                  input_sha256=digest(out/'input'),eqdsk_sha256=digest(eqdsk),
                  native_binary_sha256=digest(binary),reference_sha256=digest(restart),
                  comparison_quadrature_order=18)
    ref = Restart(restart)
    report['runs'] = [run(ref,d) for d in (2,4,6)]
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
