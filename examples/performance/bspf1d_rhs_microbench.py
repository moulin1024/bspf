"""
Micro-benchmark for the heavy part of bspf1d.differentiate_1_2_batched.

It mimics the CPU path:
  - Fortran-ordered BW (n_basis x n)
  - Fortran-ordered BND (m x n), here m=2*order with order=degree-1
  - Fortran-ordered input f (n x batch)
  - rhs_top = 2 * BW @ f
  - dY = BND @ f
  - stack into a preallocated rhs buffer

Run:
  python examples/performance/bspf1d_rhs_microbench.py
  python examples/performance/bspf1d_rhs_microbench.py --n 512 --n_basis 32 --batch 512 --runs 50
"""

import argparse
import os
import time
import numpy as np


def bench(n: int, n_basis: int, batch: int, runs: int):
    # Simulate order=degree-1, so m = 2*order; pick degree~5 => order=4 => m=8
    m = 8

    # Fortran order to match BLAS-friendly layout used in bspf1d
    BW = np.asfortranarray(np.random.rand(n_basis, n), dtype=np.float64)
    BND = np.asfortranarray(np.random.rand(m, n), dtype=np.float64)
    f = np.asfortranarray(np.random.rand(n, batch), dtype=np.float64)

    # Preallocate RHS buffer
    rhs_buf = np.empty((n_basis + m, batch), dtype=np.float64, order="F")

    # Warmup
    rhs_top = BW @ f
    rhs_top *= 2.0
    dY = BND @ f
    rhs_buf[:n_basis, :] = rhs_top
    rhs_buf[n_basis:, :] = dY

    # Raw matmul benchmark (no stacking)
    raw_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        BW @ f
        t1 = time.perf_counter()
        raw_times.append(t1 - t0)

    times_top = []
    times_dy = []
    times_stack = []
    times_total = []

    for _ in range(runs):
        t0 = time.perf_counter()
        rhs_top = BW @ f
        rhs_top *= 2.0
        t1 = time.perf_counter()

        dY = BND @ f
        t2 = time.perf_counter()

        rhs_buf[:n_basis, :] = rhs_top
        rhs_buf[n_basis:, :] = dY
        t3 = time.perf_counter()

        times_top.append(t1 - t0)
        times_dy.append(t2 - t1)
        times_stack.append(t3 - t2)
        times_total.append(t3 - t0)

    def stats(arr):
        arr = np.array(arr)
        return arr.mean()*1e3, arr.min()*1e3, arr.max()*1e3

    mraw, miraw, maraw = stats(raw_times)
    mt, mit, mat = stats(times_top)
    md, mid, mad = stats(times_dy)
    ms, mis, mas = stats(times_stack)
    mtot, mitot, matot = stats(times_total)

    print(f"Shapes: BW {BW.shape} (F), BND {BND.shape} (F), f {f.shape} (F)")
    print(f"Runs: {runs}")
    print(f"raw BW@f        : mean {mraw:.3f} ms (min {miraw:.3f}, max {maraw:.3f})")
    print(f"rhs_top = 2*BW@f : mean {mt:.3f} ms (min {mit:.3f}, max {mat:.3f})")
    print(f"dY = BND@f       : mean {md:.3f} ms (min {mid:.3f}, max {mad:.3f})")
    print(f"stack rhs        : mean {ms:.3f} ms (min {mis:.3f}, max {mas:.3f})")
    print(f"total (top+dy+stack): mean {mtot:.3f} ms (min {mitot:.3f}, max {matot:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Micro-benchmark for bspf1d RHS matmuls")
    parser.add_argument("--n", type=int, default=512, help="n_points")
    parser.add_argument("--n_basis", type=int, default=32, help="number of basis functions")
    parser.add_argument("--batch", type=int, default=512, help="batch size (columns of f)")
    parser.add_argument("--runs", type=int, default=20, help="number of timing runs")
    args = parser.parse_args()

    # Print BLAS threading environment for context
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        if var in os.environ:
            print(f"{var}={os.environ[var]}")

    bench(args.n, args.n_basis, args.batch, args.runs)


if __name__ == "__main__":
    main()

