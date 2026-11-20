"""Isolate global approximation error by changing only the vertical basis degree."""
import argparse
import json
from pathlib import Path

import numpy as np

from mms import Restart
from refined_global import RefinedGlobalBSPF
from tilted_mms import manufactured_tilted, coordinates


def run(path,output,cases,score,extra_bins=0,remark=False):
    ref=Restart(path)
    exact=manufactured_tilted(3.)
    def source(r,z):
        return exact(r,z)[3]
    def wall(r,z):
        return exact(r,z)[0]
    control=RefinedGlobalBSPF(ref,order=8)
    control.pilot(source,wall)
    rows=[]
    for degree,divisor in cases:
        solver=RefinedGlobalBSPF(ref,degree=(5,degree),order=8,wall_weight=control.wall_weight)
        solver.pilot(source,wall)
        solver.marked=control.marked.copy()
        width=min(solver.width)
        padding=(2+extra_bins)*solver.bin_width+5*width/32
        solver.refine(width/divisor,support_padding=padding)
        if remark:
            solver.remark()
            solver.refine(width/divisor,support_padding=padding)
        energy=np.zeros(3)
        def collect(r,z,w,dr,dz,gr,gz):
            u,v=coordinates(r,z)
            e=w/r**2*((dr-gr)**2+(dz-gz)**2)
            energy[:]+= [e.sum(),e[u*u+v*v>4].sum(),e[~solver.near(r,z)].sum()]
        flux,field=solver.score(ref,exact,score,observe=collect)
        row=dict(degree_R=5,degree_Z=degree,divisor=divisor,dofs=solver.active_dofs,marked_bins=len(solver.marked),
                 flux_pct=100*flux,field_pct=100*field,
                 squared_error_outside_ellipse_fraction=energy[1]/energy[0],
                 squared_error_outside_enrichment_bounding_box_fraction=energy[2]/energy[0])
        print(row,flush=True)
        rows.append(row)
        output.write_text(json.dumps(dict(score_order=score,extra_padding_bins=extra_bins,wall_weight=control.wall_weight,
                                         marked=control.marked.tolist(),rows=rows),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('restart',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--case',action='append',nargs=2,type=int,required=True,metavar=('Z_DEGREE','DIVISOR'))
    parser.add_argument('--score',type=int,default=36)
    parser.add_argument('--extra-bins',type=int,default=0)
    parser.add_argument('--remark',action='store_true')
    args=parser.parse_args()
    run(args.restart,args.output,args.case,args.score,args.extra_bins,args.remark)
