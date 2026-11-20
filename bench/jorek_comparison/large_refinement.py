"""Physical-coordinate spline refinement sweep on the hard MAST-U MMS."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from challenge import LocalField
from mms import Restart
from refined_global import RefinedGlobalBSPF
from tilted_mms import manufactured_tilted


def run(path,out,order,divisors,score,points_per_span,remark=False,domain=False):
    ref=Restart(path)
    exact=manufactured_tilted(3.)
    start=perf_counter()
    s=RefinedGlobalBSPF(ref,order=order)
    s.pilot(lambda r,z:exact(r,z)[3],lambda r,z:exact(r,z)[0])
    pilot_s=perf_counter()-start
    width=min(s.width)
    padding=2*s.bin_width+5*width/32
    pilot_bins=s.marked.copy()
    rows=[]
    for d in divisors:
        s.marked=pilot_bins.copy()
        start=perf_counter()
        if domain:
            s.refine_domain(width/d,points_per_span=points_per_span)
        else:
            s.refine(width/d,support_padding=padding,points_per_span=points_per_span)
        if remark:
            s.remark()
            s.refine(width/d,support_padding=padding,points_per_span=points_per_span)
        elapsed=perf_counter()-start
        local=LocalField()
        flux,bp=s.score(ref,exact,score,observe=local.collect)
        row=dict(divisor=d,spacing_m=s.h,dofs=s.active_dofs,refine_s=elapsed,assembly_points=s.assembly_points,
                 flux_pct=100*flux,field_pct=100*bp,
                 local_field_pct=100*np.sqrt(local.error/local.energy),
                 normal_equation_residual=s.residual)
        if domain:
            row.update(pilot_coefficients=s.coef.size,correction_unknowns=s.correction_unknowns)
        else:
            row.update(marked_bins=len(s.marked))
        if not np.all(np.isfinite(list(row.values()))):
            raise ValueError(row)
        rows.append(row)
        print(row,flush=True)
        out.write_text(json.dumps(dict(assembly_order=order,points_per_span=points_per_span,score_order=score,pilot_s=pilot_s,
                                       domain=domain,padding_m=None if domain else padding,remark=remark,marked=pilot_bins.tolist(),rows=rows),
                                  indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('restart',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--order',type=int,default=8)
    parser.add_argument('--score',type=int,default=24)
    parser.add_argument('--divisors',type=int,nargs='+',default=[16,32,64,128])
    parser.add_argument('--points-per-span',type=int,default=6,help='0 reproduces legacy fixed quadrature')
    parser.add_argument('--remark',action='store_true',help='Mark again from the refined PDE residual and refit')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--domain',action='store_true',help='Use domain-wide spline corrections (default)')
    modes.add_argument('--fixed-footprint',action='store_true',help='Preserve the fixed pilot-region benchmark')
    args=parser.parse_args()
    if args.domain and (args.remark or args.points_per_span<2):
        parser.error('--domain requires composite quadrature and replaces --remark')
    domain=args.domain or not (args.fixed_footprint or args.remark or args.points_per_span==0)
    run(args.restart,args.output,args.order,args.divisors,args.score,args.points_per_span or None,args.remark,domain)
