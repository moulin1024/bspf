"""
Investigate why rhs_build occasionally becomes slow.

This script breaks down rhs_build into its components and measures each separately
to identify which part causes the occasional slowdown.
"""

import numpy as np
import time
from bspf1d_profiling import bspf1d

def main():
    # Domain and grid
    a, b = 0.0, 2.0 * np.pi
    n = 10000
    x = np.linspace(a, b, n, endpoint=True)
    
    # Test function
    f = np.sin(x / (1.01 + np.cos(x)))
    
    # BSPF operator
    degree = 7
    op = bspf1d.from_grid(degree=degree, x=x, n_basis=4*degree, 
                          use_clustering=True, clustering_factor=2.0, use_gpu=False)
    
    # Warmup
    _ = op.differentiate_1_2(f)
    
    # Break down rhs_build into components
    n_runs = 200
    print(f"Running {n_runs} iterations to investigate rhs_build variability...\n")
    
    times_bw = []
    times_bnd = []
    times_concat = []
    times_total = []
    
    for i in range(n_runs):
        # Ensure f is contiguous
        f_cont = np.ascontiguousarray(f, dtype=np.float64)
        
        t_start = time.perf_counter()
        
        # Component 1: BW @ f
        t1 = time.perf_counter()
        rhs_2bw = 2.0 * (op.BW @ f_cont)
        t2 = time.perf_counter()
        times_bw.append(t2 - t1)
        
        # Component 2: BND @ f
        t3 = time.perf_counter()
        dY = op.end.BND @ f_cont
        t4 = time.perf_counter()
        times_bnd.append(t4 - t3)
        
        # Component 3: concatenate
        t5 = time.perf_counter()
        rhs = np.concatenate((rhs_2bw, dY))
        t6 = time.perf_counter()
        times_concat.append(t6 - t5)
        
        t_end = time.perf_counter()
        times_total.append(t_end - t_start)
    
    # Statistics
    def stats(name, times):
        mean = np.mean(times)
        std = np.std(times)
        min_t = np.min(times)
        max_t = np.max(times)
        cv = std / mean if mean > 0 else 0  # coefficient of variation
        print(f"{name:15s}: mean={mean:8.6f}, std={std:8.6f}, min={min_t:8.6f}, max={max_t:8.6f}, CV={cv:6.2%}")
        return mean, std, min_t, max_t, cv
    
    print("Component breakdown:")
    print("-" * 80)
    mean_bw, std_bw, min_bw, max_bw, cv_bw = stats("BW @ f", times_bw)
    mean_bnd, std_bnd, min_bnd, max_bnd, cv_bnd = stats("BND @ f", times_bnd)
    mean_concat, std_concat, min_concat, max_concat, cv_concat = stats("concatenate", times_concat)
    mean_total, std_total, min_total, max_total, cv_total = stats("Total rhs_build", times_total)
    
    print("\n" + "=" * 80)
    print("Analysis:")
    print(f"  BW @ f:        {mean_bw/mean_total*100:5.1f}% of total, CV={cv_bw:6.2%}")
    print(f"  BND @ f:       {mean_bnd/mean_total*100:5.1f}% of total, CV={cv_bnd:6.2%}")
    print(f"  concatenate:   {mean_concat/mean_total*100:5.1f}% of total, CV={cv_concat:6.2%}")
    
    # Identify outliers
    print("\n" + "=" * 80)
    print("Outlier analysis (top 5 slowest runs):")
    sorted_indices = np.argsort(times_total)[-5:][::-1]
    for idx in sorted_indices:
        print(f"\nRun {idx}:")
        print(f"  Total:     {times_total[idx]:8.6f} ({times_total[idx]/mean_total:.2f}x mean)")
        print(f"  BW @ f:    {times_bw[idx]:8.6f} ({times_bw[idx]/mean_bw:.2f}x mean)")
        print(f"  BND @ f:   {times_bnd[idx]:8.6f} ({times_bnd[idx]/mean_bnd:.2f}x mean)")
        print(f"  concat:    {times_concat[idx]:8.6f} ({times_concat[idx]/mean_concat:.2f}x mean)")
    
    # Check matrix properties
    print("\n" + "=" * 80)
    print("Matrix properties:")
    print(f"  BW shape: {op.BW.shape}, dtype: {op.BW.dtype}, contiguous: {op.BW.flags['C_CONTIGUOUS']}")
    print(f"  BND shape: {op.end.BND.shape}, dtype: {op.end.BND.dtype}, contiguous: {op.end.BND.flags['C_CONTIGUOUS']}")
    print(f"  f shape: {f.shape}, dtype: {f.dtype}, contiguous: {f.flags['C_CONTIGUOUS']}")
    
    # Memory layout impact
    print("\n" + "=" * 80)
    print("Testing memory layout impact:")
    
    # Test with Fortran-order matrices
    BW_f = np.asfortranarray(op.BW)
    BND_f = np.asfortranarray(op.end.BND)
    
    times_bw_f = []
    times_bnd_f = []
    
    for i in range(50):
        f_cont = np.ascontiguousarray(f, dtype=np.float64)
        
        t1 = time.perf_counter()
        _ = 2.0 * (BW_f @ f_cont)
        t2 = time.perf_counter()
        times_bw_f.append(t2 - t1)
        
        t3 = time.perf_counter()
        _ = BND_f @ f_cont
        t4 = time.perf_counter()
        times_bnd_f.append(t4 - t3)
    
    mean_bw_f = np.mean(times_bw_f)
    mean_bnd_f = np.mean(times_bnd_f)
    
    print(f"  BW @ f (C-order):  {mean_bw:8.6f} ± {std_bw:8.6f}")
    print(f"  BW @ f (F-order):  {mean_bw_f:8.6f} ± {np.std(times_bw_f):8.6f}")
    print(f"  Speedup: {mean_bw/mean_bw_f:.2f}x")
    print(f"  BND @ f (C-order): {mean_bnd:8.6f} ± {std_bnd:8.6f}")
    print(f"  BND @ f (F-order): {mean_bnd_f:8.6f} ± {np.std(times_bnd_f):8.6f}")
    print(f"  Speedup: {mean_bnd/mean_bnd_f:.2f}x")


if __name__ == "__main__":
    main()

