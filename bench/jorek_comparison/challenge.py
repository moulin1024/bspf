"""Retest fixed BSPF and JOREK configurations on a tilted wall-adjacent layer."""
import argparse
import json
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

from campaign import run
from global_bspf import PhysicalBSPF
from mms import Restart, SCORE_ORDER, compile_kernel, native
from tilted_mms import coordinates, manufactured_tilted


class LocalField:
    def __init__(self):
        self.energy = self.error = self.area = 0.

    def collect(self,r,z,w,dr,dz,gr,gz):
        u,v = coordinates(r,z)
        inside = u*u+v*v <= 4
        weight = w[inside]/r[inside]**2
        self.area += w[inside].sum()
        self.energy += weight@(gr[inside]**2+gz[inside]**2)
        self.error += weight@((dr[inside]-gr[inside])**2+(dz[inside]-gz[inside])**2)


def benchmark(path):
    ref = Restart(path)
    wave = 3.
    exact = manufactured_tilted(wave)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for order,reps in [(4,3),(16,1)]:
            kernel = compile_kernel(Path(tmp),order)
            for repeat in range(reps):
                methods = ['JOREK','BSPF'] if repeat%2==0 else ['BSPF','JOREK']
                for method in methods:
                    local = LocalField()
                    if method=='JOREK':
                        row = native(path,ref,wave,kernel,timings=True,tilted=True,observe=local.collect)
                        elapsed = row['setup_s']+row['factor_and_solve_s']
                    else:
                        start = perf_counter()
                        solver = PhysicalBSPF(ref,degree=(5,13),order=order)
                        solver.solve(lambda r,z:exact(r,z)[3],lambda r,z:exact(r,z)[0])
                        elapsed = perf_counter()-start
                        flux,bp = solver.score(ref,exact,SCORE_ORDER,observe=local.collect)
                        row = dict(free_dofs=solver.coef.size,flux_relative_L2=flux,
                                   Bp_relative_L2=bp,relative_linear_residual=solver.residual)
                    rows.append(dict(method=method,assembly_order=order,repeat=repeat,
                                     total_s=elapsed,local_Bp_relative_L2=np.sqrt(local.error/local.energy),
                                     local_area=local.area,**row))
    local = LocalField()
    patch = run(ref,4,check_order=SCORE_ORDER,assembly_order=16,exact=exact,observe=local.collect)
    rows.append(dict(method='patch BSPF',assembly_order=16,repeat=0,
                     total_s=patch['setup_s']+patch['factor_and_solve_s'],
                     local_Bp_relative_L2=np.sqrt(local.error/local.energy),local_area=local.area,**patch))
    return dict(score_order=SCORE_ORDER,wave=wave,center_m=[1.30,0.72],widths_m=[0.08,0.24],
                amplitude=0.6,short_axis=[0.8,0.6],rows=rows)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('restart',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(benchmark(args.restart),indent=2,allow_nan=False)+'\n')
