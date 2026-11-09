# SPDX-License-Identifier: GPL-3.0-or-later
"""Serial same-physics comparison of BSPF, Legendre ESG and FC-Gram."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'examples/pde'))
from viscous_traveling_shock import solve
from prior_art_traveling_shock import LegendreBase,solve_fc

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['bspf','esg','fc'],required=True)
    p.add_argument('--n',type=int,required=True)
    p.add_argument('--time',type=float,default=.05)
    p.add_argument('--cfl',type=float,default=.7)
    p.add_argument('--quadrature',type=int,default=32)
    p.add_argument('--filter-order',type=int,default=50)
    p.add_argument('--tag',default='')
    a=p.parse_args();folder=ROOT/'benchmarks/traveling_shock/prior_art';folder.mkdir(parents=True,exist_ok=True)
    if a.method=='fc':
        r=solve_fc(a.n,folder/'fc_data/FCGram_data_d5_C25.mat',a.time,a.cfl,a.filter_order)
    else:
        r=solve(a.n,True,final_time=a.time,cfl=a.cfl,quadrature=a.quadrature,refinement=4,
                halfwidth=12,spline_degree=13,recondition=True,output_points=64001,progress=True,
                line_options=dict(n_basis=(a.n-1)//16+9),base_factory=LegendreBase if a.method=='esg' else None)
    r['summary']['method']=a.method
    prefix=folder/f'{a.method}_{a.n}{a.tag}'
    prefix.with_suffix('.json').write_text(json.dumps(r['summary'],indent=2))
    np.savez(prefix.with_suffix('.npz'),x=r['x'],output=r['output'],exact=r['exact'])
    print(a.method,a.n,max(v['linf'] for v in r['summary']['metrics'].values()),r['summary']['seconds'],flush=True)

if __name__=='__main__':main()
