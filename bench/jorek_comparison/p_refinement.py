"""Increase cardinal spline degree at fixed knots on the tilted MAST-U MMS."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from challenge import LocalField
from mms import Restart
from refined_global import RefinedGlobalBSPF
from tilted_mms import manufactured_tilted


def run(path,out,divisor,degrees,points,score):
    exact=manufactured_tilted(3.)
    ref=Restart(path)
    start=perf_counter()
    solver=RefinedGlobalBSPF(ref,order=8)
    solver.pilot(lambda r,z:exact(r,z)[3],lambda r,z:exact(r,z)[0])
    pilot_time=perf_counter()-start
    spacing=min(solver.width)/divisor
    report=dict(spacing_m=spacing,divisor=divisor,pilot_coefficients=solver.coef.size,
                pilot_s=pilot_time,points_per_span=points,score_order=score,rows=[])
    for degree in degrees:
        start=perf_counter()
        solver.refine_domain(spacing,degree=degree,points_per_span=points)
        elapsed=perf_counter()-start
        local=LocalField()
        flux,field=solver.score(ref,exact,score,observe=local.collect)
        row=dict(degree=degree,dofs=solver.active_dofs,correction_unknowns=solver.correction_unknowns,
                 assembly_points=solver.assembly_points,refine_s=elapsed,
                 flux_pct=100*flux,field_pct=100*field,
                 local_field_pct=100*np.sqrt(local.error/local.energy))
        if not np.all(np.isfinite(list(row.values()))):
            raise ValueError(row)
        report['rows'].append(row)
        out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(row,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('restart',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--divisor',type=int,default=32)
    p.add_argument('--degrees',type=int,nargs='+',default=list(range(3,10)))
    p.add_argument('--points-per-span',type=int,default=8)
    p.add_argument('--score',type=int,default=36)
    args=p.parse_args()
    run(args.restart,args.output,args.divisor,args.degrees,args.points_per_span,args.score)
