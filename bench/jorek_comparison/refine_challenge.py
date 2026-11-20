"""Compare residual-selected global BSPF refinement with the native tilted MMS."""
import argparse
import json
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

from challenge import LocalField
from mms import Restart, SCORE_ORDER, compile_kernel, native
from refined_global import RefinedGlobalBSPF
from tilted_mms import manufactured_tilted


def benchmark(path):
    ref = Restart(path)
    exact = manufactured_tilted(3.)
    def current(r,z):
        return exact(r,z)[3]
    def wall(r,z):
        return exact(r,z)[0]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for order,reps in [(3,3),(4,3),(16,1)]:
            kernel = compile_kernel(Path(tmp),order)
            for rep in range(reps):
                for method in (['BSPF','JOREK'] if rep%2 else ['JOREK','BSPF']):
                    local = LocalField()
                    if method=='JOREK':
                        row = native(path,ref,3.,kernel,timings=True,tilted=True,observe=local.collect)
                        elapsed = row['setup_s']+row['factor_and_solve_s']
                    else:
                        start = perf_counter()
                        solver = RefinedGlobalBSPF(ref,order=order)
                        solver.pilot(current,wall)
                        h = min(solver.width)/24
                        solver.refine(h,support_padding=2*solver.bin_width+5*min(solver.width)/32)
                        elapsed = perf_counter()-start
                        flux,bp = solver.score(ref,exact,SCORE_ORDER,observe=local.collect)
                        row = dict(free_dofs=solver.active_dofs,marked_bins=len(solver.marked),
                                   knot_spacing_m=h,flux_relative_L2=flux,Bp_relative_L2=bp,
                                   relative_linear_residual=solver.residual)
                    rows.append(dict(method=method,order=order,repeat=rep,total_s=elapsed,
                                     local_Bp_relative_L2=np.sqrt(local.error/local.energy),**row))
    return dict(score_order=SCORE_ORDER,marking_fraction=0.90,rows=rows)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('restart',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(benchmark(args.restart),indent=2,allow_nan=False)+'\n')
