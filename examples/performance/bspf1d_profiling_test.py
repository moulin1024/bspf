"""
Standalone test for bspf1d differentiate_1_2 with timing breakdown.

Run:
    python examples/performance/bspf1d_profiling_test.py
"""

import numpy as np
import sympy as sp

from bspf1d_profiling import bspf1d


def main():
    # Domain and grid
    a, b = 0.0, 2.0 * np.pi
    n = 10000
    x = np.linspace(a, b, n, endpoint=True)

    # Test function: f(x) = sin(x / (1.05 + cos x)) with Sympy exact derivatives
    t = sp.symbols("t")
    f_sym = sp.sin(t / (1.01 + sp.cos(t)))
    f1_sym = sp.diff(f_sym, t)
    f2_sym = sp.diff(f1_sym, t)
    f_func = sp.lambdify(t, f_sym, modules=["numpy"])
    f1_func = sp.lambdify(t, f1_sym, modules=["numpy"])
    f2_func = sp.lambdify(t, f2_sym, modules=["numpy"])

    f = f_func(x)
    f1_exact = f1_func(x)
    f2_exact = f2_func(x)

    # BSPF operator (CPU)
    degree = 7
    op = bspf1d.from_grid(degree=degree, x=x, n_basis=4*degree, use_clustering=True, clustering_factor=2.0, use_gpu=False)

    # Differentiate
    df1, df2, f_spline = op.differentiate_1_2(f)

    # Timing results
    timings = getattr(op, "last_timing_d12", None)

    # Error metrics vs exact derivatives
    err1 = df1 - f1_exact
    err2 = df2 - f2_exact
    max_err1 = np.max(np.abs(err1))
    max_err2 = np.max(np.abs(err2))
    l2_err1 = np.sqrt(np.mean(err1**2))
    l2_err2 = np.sqrt(np.mean(err2**2))

    print("=== bspf1d differentiate_1_2 standalone test ===")
    print(f"grid: N={n}, degree={degree}, domain=({a},{b})")
    print(f"df1 max abs: {max_err1:.3e}, df2 max abs: {max_err2:.3e}")
    print(f"df1 L2 err: {l2_err1:.3e}, df2 L2 err: {l2_err2:.3e}")
    if timings:
        print("timings (seconds):")
        for k, v in timings.items():
            print(f"  {k:12s}: {v:.6f}")
    else:
        print("No timing data found.")


if __name__ == "__main__":
    main()

