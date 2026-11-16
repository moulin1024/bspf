"""Global Chebyshev Galerkin comparison using the shared native CUDA backend."""
import argparse
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('chebyshev_cuda_harness',HERE.parent/'bspf_poisson/run.py')
harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--modes',type=Path,default=harness.DATA)
    p.add_argument('--n',type=int,nargs='+',default=[17,33,65,97,129,145])
    p.add_argument('--q',type=int,help='Gauss points per axis; same default as BSPF')
    p.add_argument('--polynomial',action='store_true');p.add_argument('--check',action='store_true')
    p.add_argument('--output',type=Path,default=HERE.parent/'evidence/chebyshev_cuda3d_rtx3070')
    args=p.parse_args();args.basis='chebyshev'
    if min(args.n)<4 or (args.q and args.q<max(args.n)):p.error('Require 4 <= n <= q')
    for n in args.n:harness.case(args,n)
if __name__=='__main__':main()
