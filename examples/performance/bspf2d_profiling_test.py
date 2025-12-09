"""
Standalone test/driver for bspf2d_profiling.

Run:
    python examples/performance/bspf2d_profiling_test.py
"""

import argparse
import os
import sys

# Ensure repo root and src are on sys.path for direct execution
_here = os.path.abspath(os.path.dirname(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
_src = os.path.join(_root, "src")
for p in (_root, _src):
    if p not in sys.path:
        sys.path.insert(0, p)

# Import after adjusting sys.path
from examples.performance.bspf2d_profiling import run_profile_2d


def parse_args():
    p = argparse.ArgumentParser(description="Profile bspf2d.differentiate_1_2")
    p.add_argument("--nx", type=int, default=128, help="Grid points in x")
    p.add_argument("--ny", type=int, default=128, help="Grid points in y")
    p.add_argument("--degree", type=int, default=7, help="B-spline degree")
    p.add_argument("--runs", type=int, default=50, help="Number of timing runs")
    p.add_argument("--gpu", action="store_true", help="Use GPU (CuPy) if available")
    return p.parse_args()


def main():
    args = parse_args()
    run_profile_2d(
        nx=args.nx,
        ny=args.ny,
        degree=args.degree,
        n_runs=args.runs,
        use_gpu=args.gpu,
    )


if __name__ == "__main__":
    main()

