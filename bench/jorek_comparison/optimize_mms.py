"""Benchmark global physical-coordinate BSPF against the compiled JOREK GS element."""
import argparse
import json
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

from global_bspf import PhysicalBSPF
from mms import ASSEMBLY_ORDER, SCORE_ORDER, Restart, compile_kernel, manufactured, native


def measure(ref, wave, order):
    exact = manufactured(wave)
    start = perf_counter()
    solver = PhysicalBSPF(ref, degree=(5,13), order=order)
    solver.solve(lambda r,z: -wave**2*(1+r*r)*np.cos(wave*z),
                 lambda r,z: (1+r*r)*np.cos(wave*z))
    elapsed = perf_counter()-start
    flux, bp = solver.score(ref,exact,SCORE_ORDER)
    return dict(free_dofs=solver.coef.size,relative_linear_residual=solver.residual,
                flux_relative_L2=flux,Bp_relative_L2=bp,setup_s=elapsed-solver.factor_and_solve_s,
                factor_and_solve_s=solver.factor_and_solve_s)


def benchmark(path,repeats):
    ref = Restart(path)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        kernels = {q:compile_kernel(Path(tmp),q) for q in (4,ASSEMBLY_ORDER)}
        for wave in [3.,4.]:
            rows = {}
            methods = ['JOREK-16','BSPF-16','JOREK-4','BSPF-4','BSPF-2']
            for rep in range(repeats):
                for name in methods[rep%len(methods):]+methods[:rep%len(methods)]:
                    order = int(name.split('-')[1])
                    if name.startswith('JOREK'):
                        row = native(path,ref,wave,kernels[order],timings=True)
                    else:
                        row = measure(ref,wave,order)
                    rows.setdefault(name,[]).append(row)
            results[str(wave)] = rows
    return dict(score_order=SCORE_ORDER, degrees=[5,13], results=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('restart', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--repeats',type=int,default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    args.output.write_text(json.dumps(benchmark(args.restart,args.repeats),indent=2,allow_nan=False)+'\n')
