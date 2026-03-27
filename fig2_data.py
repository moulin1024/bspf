#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wall-time benchmark: BSPF vs Chebyshev (1D first derivative).

Design goals
------------
- Compare *apply time* of each method (derivative evaluation).
- Explicitly exclude BSPF preprocessing (basis/knots/grid setup).
- Avoid counting one-time caches (warmup before timing).

Notes
-----
- Chebyshev uses N nodes total (internally N-1 intervals -> N nodes).
- BSPF preprocessing includes `bspf1d.from_grid(...)` and the first call that
  triggers LU/cache builds. We warm up so timings reflect steady-state applies.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import numpy as np

from bspf1d import bspf1d
from chebyshev import _construct_chebyshev_nodes, chebyshev_derivative_from_values


def _test_function(x: np.ndarray) -> np.ndarray:
    # Smooth but mildly challenging (non-periodic on the interval).
    return np.sin(300.5 * x)

def _test_function_prime(x: np.ndarray) -> np.ndarray:
    """
    Exact derivative of _test_function.
    y = sin(300.5*x)
    dy/dx = 300.5 * cos(300.5*x)
    """
    return 300.5 * np.cos(300.5 * x)

@dataclass(frozen=True)
class TimingResult:
    n_ref: int
    n_bspf: int
    bspf_ms: float
    bspf_ms_std: float
    cheb_ms: float
    cheb_ms_std: float
    trim: int
    bspf_err_linf_all: float
    cheb_err_linf_all: float
    bspf_err_linf_trim: float
    cheb_err_linf_trim: float


def _time_call(fn, *, repeats: int) -> tuple[float, float]:
    """
    Time a callable and return (median_seconds, std_seconds) over repeats.

    The median is used as the central value (robust to outliers),
    and the standard deviation is used for error bars.
    """
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    if len(times) == 1:
        return times[0], 0.0
    return statistics.median(times), statistics.stdev(times)


def benchmark_one_n(
    n_points: int,
    *,
    domain: tuple[float, float],
    degree: int,
    n_basis: int,
    num_boundary_points: int,
    lam: float,
    repeats: int,
    warmup: int,
) -> TimingResult:
    # We treat n_points as the reference N for Chebyshev.
    # BSPF in this project is typically used with an even grid (FFT-friendly for
    # residual/spectral correction), so we benchmark BSPF at the nearest even N.

    n_ref = int(n_points)
    n_bspf = n_ref - 1  # nearest even (keeps resolution comparable)
    if n_bspf < 4:
        raise ValueError(f"Need n_points >= 5 so that N(BSPF)=N(ref)-1 is valid; got {n_ref}")

    # ---- Chebyshev (N nodes total; internally N-1 intervals -> N nodes) ----
    x_cheb, _ = _construct_chebyshev_nodes(n_ref - 1, domain=domain)
    y_cheb = _test_function(x_cheb)

    def cheb_apply():
        _ = chebyshev_derivative_from_values(y_cheb, x_cheb, domain=domain)

    for _ in range(warmup):
        cheb_apply()

    cheb_s, cheb_s_std = _time_call(cheb_apply, repeats=repeats)

    # ---- BSPF (exclude preprocessing) ----
    x_bspf = np.linspace(domain[0], domain[1], n_bspf)
    y_bspf = _test_function(x_bspf)
    # Preprocessing (EXCLUDED): operator construction
    op = bspf1d.from_grid(
        degree=degree,
        x=x_bspf,
        domain=domain,
        n_basis=n_basis,
        num_boundary_points=num_boundary_points,
        correction="spectral",
        use_clustering=True,
        clustering_factor=3.0,
    )

    def bspf_apply():
        _ = op.differentiate(y_bspf, k=1, lam=lam)

    # Warmup to exclude LU/cache builds
    for _ in range(warmup):
        bspf_apply()

    bspf_s, bspf_s_std = _time_call(bspf_apply, repeats=repeats)

    # ---- Sanity-check errors (NOT timed) ----
    trim = int(num_boundary_points)

    dy_cheb = chebyshev_derivative_from_values(y_cheb, x_cheb, domain=domain)
    dy_cheb_ex = _test_function_prime(x_cheb)
    cheb_err = np.abs(dy_cheb - dy_cheb_ex)
    cheb_err_linf_all = float(np.max(cheb_err))
    cheb_err_linf_trim = float(np.max(cheb_err[trim:-trim])) if (2 * trim) < cheb_err.size else cheb_err_linf_all

    dy_bspf, _ = op.differentiate(y_bspf, k=1, lam=lam)
    dy_bspf_ex = _test_function_prime(x_bspf)
    bspf_err = np.abs(dy_bspf - dy_bspf_ex)
    bspf_err_linf_all = float(np.max(bspf_err))
    bspf_err_linf_trim = float(np.max(bspf_err[trim:-trim])) if (2 * trim) < bspf_err.size else bspf_err_linf_all

    return TimingResult(
        n_ref=n_ref,
        n_bspf=n_bspf,
        bspf_ms=1e3 * bspf_s,
        bspf_ms_std=1e3 * bspf_s_std,
        cheb_ms=1e3 * cheb_s,
        cheb_ms_std=1e3 * cheb_s_std,
        trim=trim,
        bspf_err_linf_all=bspf_err_linf_all,
        cheb_err_linf_all=cheb_err_linf_all,
        bspf_err_linf_trim=bspf_err_linf_trim,
        cheb_err_linf_trim=cheb_err_linf_trim,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=float, default=-np.pi + 0.03)
    p.add_argument("--b", type=float, default=np.pi - 0.07)
    p.add_argument("--degree", type=int, default=8)
    p.add_argument("--n-basis", type=int, default=32)
    p.add_argument("--num-boundary-points", type=int, default=13)
    p.add_argument("--lam", type=float, default=0.01)
    p.add_argument("--repeats", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument(
        "--ns",
        type=int,
        nargs="*",
        default=[65, 129, 257, 513, 1025, 2049, 4097, 8193, 16385],
        help="Reference point counts for Chebyshev. BSPF uses N-1 (even).",
    )
    args = p.parse_args()

    domain = (float(args.a), float(args.b))
    results: list[TimingResult] = []
    for n in args.ns:
        results.append(
            benchmark_one_n(
                n,
                domain=domain,
                degree=args.degree,
                n_basis=args.n_basis,
                num_boundary_points=args.num_boundary_points,
                lam=args.lam,
                repeats=args.repeats,
                warmup=args.warmup,
            )
        )

    # Print as a simple table
    header = (
        f"{'N(ref)':>8} | {'N(BSPF)':>8} | "
        f"{'BSPF ms':>9} | {'Cheb ms':>8} | "
        f"{'BSPF err∞(trim)':>16} | {'Cheb err∞(trim)':>16}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.n_ref:8d} | {r.n_bspf:8d} | "
            f"{r.bspf_ms:9.3f} | {r.cheb_ms:8.3f} | "
            f"{r.bspf_err_linf_trim:16.2e} | {r.cheb_err_linf_trim:16.2e}"
        )

    print("\nNotes:")
    print("- BSPF timings exclude operator construction and include warmup to avoid counting LU/cache builds.")
    print("- Chebyshev timings apply rFFT-based DCT-I + coefficient recurrence + inverse DCT-I each call.")
    print("- BSPF is benchmarked at the nearest even grid size (see N(BSPF) column).")
    if results:
        print(f"- Error is L∞ on each method's own grid after trimming {results[0].trim} points from each end.")


    # Plot the results in log-log scale
    import matplotlib.pyplot as plt
    
    n_refs = np.array([r.n_ref for r in results], dtype=float)
    bspf_times = np.array([r.bspf_ms for r in results], dtype=float)
    cheb_times = np.array([r.cheb_ms for r in results], dtype=float)
    bspf_err = np.array([r.bspf_ms_std for r in results], dtype=float)
    cheb_err = np.array([r.cheb_ms_std for r in results], dtype=float)

    # Plot with error bars; set log scales explicitly
    plt.errorbar(
        n_refs,
        bspf_times,
        yerr=bspf_err,
        fmt='o-',
        label='BSPF',
        linewidth=2,
        markersize=6,
        capsize=3,
    )
    plt.errorbar(
        n_refs,
        cheb_times,
        yerr=cheb_err,
        fmt='s-',
        label='Chebyshev',
        linewidth=2,
        markersize=6,
        capsize=3,
    )
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('N (grid points)', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.title('Performance Benchmark: BSPF vs Chebyshev', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('figs/fig2.pdf', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()

