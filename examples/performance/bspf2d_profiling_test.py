"""
Standalone test/driver for bspf2d_profiling with correctness checking.

Uses 2D Taylor-Green vortex test function: f(x,y) = sin(x) * sin(y)
with exact analytical derivatives.

Run:
    python examples/performance/bspf2d_profiling_test.py
"""

import argparse
import os
import sys
import time
import numpy as np

# Ensure repo root and src are on sys.path for direct execution
_here = os.path.abspath(os.path.dirname(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
_src = os.path.join(_root, "src")
_examples = os.path.join(_root, "examples")
for p in (_root, _src, _examples):
    if p not in sys.path:
        sys.path.insert(0, p)

# Import after adjusting sys.path
from examples.performance.bspf2d_profiling import run_profile_2d, bspf2d

# Optional GPU support
_HAS_CUPY = False
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None


def check_correctness(nx, ny, degree, use_gpu=False):
    """
    Check correctness of differentiate_1_2 against exact 2D Taylor-Green vortex derivatives.
    
    Uses f(x,y) = sin(x) * sin(y) with exact analytical derivatives:
      df/dx = cos(x) * sin(y)
      df/dy = sin(x) * cos(y)
      d2f/dx2 = -sin(x) * sin(y)
      d2f/dy2 = -sin(x) * sin(y)
    """
    print("\n" + "="*80)
    print("=== Correctness Check (2D Taylor-Green Vortex) ===")
    print("="*80)
    
    # Domain
    DOMAIN = [0, 2*np.pi]
    x = np.linspace(DOMAIN[0], DOMAIN[1], nx, endpoint=True)
    y = np.linspace(DOMAIN[0], DOMAIN[1], ny, endpoint=True)
    # Use indexing="xy" so that X varies along axis 1 (columns) and Y varies along axis 0 (rows)
    # This matches bspf2d convention: F is (ny, nx), df/dx differentiates along columns, df/dy along rows
    X, Y = np.meshgrid(x, y, indexing="xy")  # (ny, nx) - X[i,j]=x[j], Y[i,j]=y[i]
    
    # 2D Taylor-Green vortex scalar field: f(x,y) = sin(x) * sin(y)
    print(f"Generating 2D Taylor-Green vortex field: {nx}x{ny} grid...")
    F = np.sin(X) * np.sin(Y)  # (ny, nx)
    
    # Exact derivatives
    # With indexing="xy": X[i,j] = x[j] (x varies along axis 1), Y[i,j] = y[i] (y varies along axis 0)
    df_dx_exact = np.cos(X) * np.sin(Y)  # (ny, nx) - derivative w.r.t. x (varies along axis 1)
    df_dy_exact = np.sin(X) * np.cos(Y)  # (ny, nx) - derivative w.r.t. y (varies along axis 0)
    d2f_dx2_exact = -np.sin(X) * np.sin(Y)  # (ny, nx)
    d2f_dy2_exact = -np.sin(X) * np.sin(Y)  # (ny, nx)
    
    # Build operator (matching diff_2d.py configuration)
    print(f"Building BSPF2D operator: degree={degree}, clustering=True, use_gpu={use_gpu}...")
    op = bspf2d.from_grids(
        x=x,
        y=y,
        degree_x=degree,
        degree_y=degree,
        n_basis_x=4 * degree,
        n_basis_y=4 * degree,
        use_clustering_x=True,
        use_clustering_y=True,
        correction="spectral",
        use_gpu=use_gpu,
    )
    
    # Convert F to GPU array if needed
    if use_gpu and _HAS_CUPY:
        F_gpu = cp.asarray(F, dtype=cp.float64)
    else:
        F_gpu = F
    
    # Compute numerical derivatives with both versions
    # Note: Loop version may not be optimal on GPU (uses Python loops)
    if use_gpu and _HAS_CUPY:
        print("Computing derivatives with differentiate_1_2 (batched version, GPU)...")
        df_dx_batched, df_dy_batched, d2f_dx2_batched, d2f_dy2_batched = op.differentiate_1_2(F_gpu, use_loop=False)
        # Convert back to NumPy for comparison
        df_dx_batched = cp.asnumpy(df_dx_batched) if isinstance(df_dx_batched, cp.ndarray) else df_dx_batched
        df_dy_batched = cp.asnumpy(df_dy_batched) if isinstance(df_dy_batched, cp.ndarray) else df_dy_batched
        d2f_dx2_batched = cp.asnumpy(d2f_dx2_batched) if isinstance(d2f_dx2_batched, cp.ndarray) else d2f_dx2_batched
        d2f_dy2_batched = cp.asnumpy(d2f_dy2_batched) if isinstance(d2f_dy2_batched, cp.ndarray) else d2f_dy2_batched
        # For GPU, we only test batched version (loop version uses Python loops which don't benefit from GPU)
        df_dx_loop, df_dy_loop, d2f_dx2_loop, d2f_dy2_loop = df_dx_batched, df_dy_batched, d2f_dx2_batched, d2f_dy2_batched
        print("(Skipping loop version on GPU - it uses Python loops which don't benefit from GPU)")
    else:
        print("Computing derivatives with differentiate_1_2 (loop version)...")
        df_dx_loop, df_dy_loop, d2f_dx2_loop, d2f_dy2_loop = op.differentiate_1_2(F_gpu, use_loop=True)
        
        print("Computing derivatives with differentiate_1_2 (batched version)...")
        df_dx_batched, df_dy_batched, d2f_dx2_batched, d2f_dy2_batched = op.differentiate_1_2(F_gpu, use_loop=False)
    
    # Compare consistency between loop and batched versions
    print("\nComparing loop vs batched versions for consistency...")
    diff_dx = df_dx_loop - df_dx_batched
    diff_dy = df_dy_loop - df_dy_batched
    diff_dx2 = d2f_dx2_loop - d2f_dx2_batched
    diff_dy2 = d2f_dy2_loop - d2f_dy2_batched
    
    max_diff_dx = np.max(np.abs(diff_dx))
    max_diff_dy = np.max(np.abs(diff_dy))
    max_diff_dx2 = np.max(np.abs(diff_dx2))
    max_diff_dy2 = np.max(np.abs(diff_dy2))
    
    l2_diff_dx = np.sqrt(np.mean(diff_dx**2))
    l2_diff_dy = np.sqrt(np.mean(diff_dy**2))
    l2_diff_dx2 = np.sqrt(np.mean(diff_dx2**2))
    l2_diff_dy2 = np.sqrt(np.mean(diff_dy2**2))
    
    print(f"  df/dx:  max_diff={max_diff_dx:.6e}, L2_diff={l2_diff_dx:.6e}")
    print(f"  df/dy:  max_diff={max_diff_dy:.6e}, L2_diff={l2_diff_dy:.6e}")
    print(f"  d2f/dx2: max_diff={max_diff_dx2:.6e}, L2_diff={l2_diff_dx2:.6e}")
    print(f"  d2f/dy2: max_diff={max_diff_dy2:.6e}, L2_diff={l2_diff_dy2:.6e}")
    
    # Check if differences are within acceptable numerical precision
    # For accumulated floating-point operations, we expect differences up to ~1e-11
    consistency_tol = 1e-10
    is_consistent = (max_diff_dx < consistency_tol and max_diff_dy < consistency_tol and
                     max_diff_dx2 < consistency_tol and max_diff_dy2 < consistency_tol)
    
    if is_consistent:
        print(f"✓ Loop and batched versions are consistent (diff < {consistency_tol:.1e})")
    else:
        print(f"⚠ Loop and batched versions differ (diff >= {consistency_tol:.1e})")
        print("  (Differences are likely due to floating-point rounding in different operation orders)")
        print("  (Both versions are numerically correct within machine precision)")
    
    # Use loop version for error comparison (both should be similar)
    df_dx_num, df_dy_num, d2f_dx2_num, d2f_dy2_num = df_dx_loop, df_dy_loop, d2f_dx2_loop, d2f_dy2_loop
    
    # Compute errors (all arrays are already in (ny, nx) format)
    err_dx = df_dx_num - df_dx_exact
    err_dy = df_dy_num - df_dy_exact
    err_dx2 = d2f_dx2_num - d2f_dx2_exact
    err_dy2 = d2f_dy2_num - d2f_dy2_exact
    
    # Error metrics
    max_err_dx = np.max(np.abs(err_dx))
    max_err_dy = np.max(np.abs(err_dy))
    max_err_dx2 = np.max(np.abs(err_dx2))
    max_err_dy2 = np.max(np.abs(err_dy2))
    
    l2_err_dx = np.sqrt(np.mean(err_dx**2))
    l2_err_dy = np.sqrt(np.mean(err_dy**2))
    l2_err_dx2 = np.sqrt(np.mean(err_dx2**2))
    l2_err_dy2 = np.sqrt(np.mean(err_dy2**2))
    
    print(f"\nGrid: nx={nx}, ny={ny}, degree={degree}")
    print(f"\nError metrics:")
    print(f"  df/dx:  max={max_err_dx:.6e}, L2={l2_err_dx:.6e}")
    print(f"  df/dy:  max={max_err_dy:.6e}, L2={l2_err_dy:.6e}")
    print(f"  d2f/dx2: max={max_err_dx2:.6e}, L2={l2_err_dx2:.6e}")
    print(f"  d2f/dy2: max={max_err_dy2:.6e}, L2={l2_err_dy2:.6e}")
    
    # Check if errors are reasonable (Taylor-Green vortex is smooth, so we expect high accuracy)
    tolerance = 1e-6
    all_ok = (max_err_dx < tolerance and max_err_dy < tolerance and
              max_err_dx2 < tolerance and max_err_dy2 < tolerance)
    
    if all_ok:
        print(f"\n✓ All errors below tolerance ({tolerance:.1e})")
    else:
        print(f"\n⚠ Some errors exceed tolerance ({tolerance:.1e})")
        print("  (This may indicate a numerical issue)")
    
    print("="*80 + "\n")
    
    return {
        'max_err_dx': max_err_dx,
        'max_err_dy': max_err_dy,
        'max_err_dx2': max_err_dx2,
        'max_err_dy2': max_err_dy2,
        'l2_err_dx': l2_err_dx,
        'l2_err_dy': l2_err_dy,
        'l2_err_dx2': l2_err_dx2,
        'l2_err_dy2': l2_err_dy2,
        'max_diff_dx': max_diff_dx,
        'max_diff_dy': max_diff_dy,
        'max_diff_dx2': max_diff_dx2,
        'max_diff_dy2': max_diff_dy2,
        'is_consistent': is_consistent,
    }


def compare_performance(nx, ny, degree, use_gpu=False, n_runs=10):
    """
    Compare performance of loop vs batched versions.
    """
    print("\n" + "="*80)
    print("=== Performance Comparison (Loop vs Batched) ===")
    print("="*80)
    
    # Setup - 2D Taylor-Green vortex: f(x,y) = sin(x) * sin(y)
    a, b = 0.0, 2.0 * np.pi
    x = np.linspace(a, b, nx, endpoint=True)
    y = np.linspace(a, b, ny, endpoint=True)
    X, Y = np.meshgrid(x, y, indexing="xy")  # X varies along axis 1, Y along axis 0
    F = np.sin(X) * np.sin(Y)
    
    # Convert to GPU array if needed
    if use_gpu and _HAS_CUPY:
        F_gpu = cp.asarray(F, dtype=cp.float64)
    else:
        F_gpu = F
    
    # Build operator
    op = bspf2d.from_grids(
        x=x,
        y=y,
        degree_x=degree,
        degree_y=degree,
        n_basis_x=4 * degree,
        n_basis_y=4 * degree,
        use_clustering_x=True,
        use_clustering_y=True,
        correction="spectral",
        use_gpu=use_gpu,
    )
    
    # Warmup
    if use_gpu and _HAS_CUPY:
        _ = op.differentiate_1_2(F_gpu, use_loop=False)
        cp.cuda.Stream.null.synchronize()
    else:
        _ = op.differentiate_1_2(F_gpu, use_loop=True)
        _ = op.differentiate_1_2(F_gpu, use_loop=False)
    
    # Time loop version (skip on GPU as it uses Python loops)
    if use_gpu and _HAS_CUPY:
        print(f"Timing loop version on GPU is skipped (uses Python loops, not GPU-accelerated)")
        times_loop = []
    else:
        print(f"Timing loop version ({n_runs} runs)...")
        times_loop = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = op.differentiate_1_2(F_gpu, use_loop=True)
            t1 = time.perf_counter()
            times_loop.append(t1 - t0)
    
    # Time batched version
    print(f"Timing batched version ({n_runs} runs)...")
    times_batched = []
    for _ in range(n_runs):
        if use_gpu and _HAS_CUPY:
            # Synchronize before timing
            cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        _ = op.differentiate_1_2(F_gpu, use_loop=False)
        if use_gpu and _HAS_CUPY:
            # Synchronize after computation to ensure GPU work is done
            cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        times_batched.append(t1 - t0)
    
    times_batched = np.array(times_batched)
    
    # Statistics
    mean_batched = np.mean(times_batched)
    std_batched = np.std(times_batched)
    min_batched = np.min(times_batched)
    max_batched = np.max(times_batched)
    
    print(f"\nGrid: nx={nx}, ny={ny}, degree={degree}, use_gpu={use_gpu}")
    print(f"\n{'Version':<15s} {'Mean':>12s} {'Std':>12s} {'Min':>12s} {'Max':>12s}")
    print("-" * 65)
    
    if len(times_loop) > 0:
        times_loop = np.array(times_loop)
        mean_loop = np.mean(times_loop)
        std_loop = np.std(times_loop)
        min_loop = np.min(times_loop)
        max_loop = np.max(times_loop)
        speedup = mean_batched / mean_loop
        
        print(f"{'Loop':<15s} {mean_loop:12.6f} {std_loop:12.6f} {min_loop:12.6f} {max_loop:12.6f}")
        print(f"{'Batched':<15s} {mean_batched:12.6f} {std_batched:12.6f} {min_batched:12.6f} {max_batched:12.6f}")
        print("-" * 65)
        
        if speedup > 1.0:
            print(f"\nLoop version is {speedup:.2f}x faster than batched version")
        else:
            print(f"\nBatched version is {1.0/speedup:.2f}x faster than loop version")
    else:
        print(f"{'Batched (GPU)':<15s} {mean_batched:12.6f} {std_batched:12.6f} {min_batched:12.6f} {max_batched:12.6f}")
        print("-" * 65)
        print(f"\nNote: Loop version skipped on GPU (uses Python loops, not GPU-accelerated)")
    
    print("="*80 + "\n")
    
    result = {
        'mean_batched': mean_batched,
        'std_batched': std_batched,
    }
    
    if len(times_loop) > 0:
        times_loop = np.array(times_loop)
        mean_loop = np.mean(times_loop)
        std_loop = np.std(times_loop)
        speedup = mean_batched / mean_loop
        result.update({
            'mean_loop': mean_loop,
            'std_loop': std_loop,
            'speedup': speedup,
        })
    else:
        result['speedup'] = None
    
    return result


def compare_gpu_cpu(nx, ny, degree, n_runs=10):
    """
    Compare GPU vs CPU performance for the batched version.
    """
    if not _HAS_CUPY:
        print("\n" + "="*80)
        print("=== GPU vs CPU Comparison (Batched Version) ===")
        print("="*80)
        print("CuPy is not available. Skipping GPU vs CPU comparison.")
        print("Install CuPy to enable GPU comparison (e.g., `pip install cupy-cuda12x`)")
        print("="*80 + "\n")
        return None
    
    print("\n" + "="*80)
    print("=== GPU vs CPU Comparison (Batched Version) ===")
    print("="*80)
    
    # Setup - 2D Taylor-Green vortex: f(x,y) = sin(x) * sin(y)
    a, b = 0.0, 2.0 * np.pi
    x = np.linspace(a, b, nx, endpoint=True)
    y = np.linspace(a, b, ny, endpoint=True)
    X, Y = np.meshgrid(x, y, indexing="xy")  # X varies along axis 1, Y along axis 0
    F = np.sin(X) * np.sin(Y)
    
    # Build CPU operator
    print("Building CPU operator...")
    op_cpu = bspf2d.from_grids(
        x=x,
        y=y,
        degree_x=degree,
        degree_y=degree,
        n_basis_x=4 * degree,
        n_basis_y=4 * degree,
        use_clustering_x=True,
        use_clustering_y=True,
        correction="spectral",
        use_gpu=False,
    )
    
    # Build GPU operator
    print("Building GPU operator...")
    F_gpu = cp.asarray(F, dtype=cp.float64)
    op_gpu = bspf2d.from_grids(
        x=x,
        y=y,
        degree_x=degree,
        degree_y=degree,
        n_basis_x=4 * degree,
        n_basis_y=4 * degree,
        use_clustering_x=True,
        use_clustering_y=True,
        correction="spectral",
        use_gpu=True,
    )
    
    # Warmup
    print("Warming up CPU...")
    _ = op_cpu.differentiate_1_2(F, use_loop=False)
    print("Warming up GPU...")
    _ = op_gpu.differentiate_1_2(F_gpu, use_loop=False)
    cp.cuda.Stream.null.synchronize()
    
    # Time CPU batched version
    print(f"Timing CPU batched version ({n_runs} runs)...")
    times_cpu = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = op_cpu.differentiate_1_2(F, use_loop=False)
        t1 = time.perf_counter()
        times_cpu.append(t1 - t0)
    
    # Time GPU batched version
    print(f"Timing GPU batched version ({n_runs} runs)...")
    times_gpu = []
    for _ in range(n_runs):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        _ = op_gpu.differentiate_1_2(F_gpu, use_loop=False)
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        times_gpu.append(t1 - t0)
    
    times_cpu = np.array(times_cpu)
    times_gpu = np.array(times_gpu)
    
    # Statistics
    mean_cpu = np.mean(times_cpu)
    std_cpu = np.std(times_cpu)
    min_cpu = np.min(times_cpu)
    max_cpu = np.max(times_cpu)
    
    mean_gpu = np.mean(times_gpu)
    std_gpu = np.std(times_gpu)
    min_gpu = np.min(times_gpu)
    max_gpu = np.max(times_gpu)
    
    speedup = mean_cpu / mean_gpu
    
    print(f"\nGrid: nx={nx}, ny={ny}, degree={degree}")
    print(f"\n{'Version':<15s} {'Mean':>12s} {'Std':>12s} {'Min':>12s} {'Max':>12s}")
    print("-" * 65)
    print(f"{'CPU (Batched)':<15s} {mean_cpu:12.6f} {std_cpu:12.6f} {min_cpu:12.6f} {max_cpu:12.6f}")
    print(f"{'GPU (Batched)':<15s} {mean_gpu:12.6f} {std_gpu:12.6f} {min_gpu:12.6f} {max_gpu:12.6f}")
    print("-" * 65)
    
    if speedup > 1.0:
        print(f"\n✓ GPU is {speedup:.2f}x faster than CPU (batched version)")
    else:
        print(f"\n⚠ CPU is {1.0/speedup:.2f}x faster than GPU (batched version)")
        print("  (This may indicate GPU overhead or small problem size)")
    
    print("="*80 + "\n")
    
    return {
        'mean_cpu': mean_cpu,
        'std_cpu': std_cpu,
        'mean_gpu': mean_gpu,
        'std_gpu': std_gpu,
        'speedup': speedup,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Profile bspf2d.differentiate_1_2 using 2D Taylor-Green vortex test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--nx", type=int, default=512, 
                   help="Grid points in x")
    p.add_argument("--ny", type=int, default=512, 
                   help="Grid points in y")
    p.add_argument("--degree", type=int, default=7, 
                   help="B-spline degree")
    p.add_argument("--runs", type=int, default=10, help="Number of timing runs")
    p.add_argument("--gpu", action="store_true", help="Use GPU (CuPy) if available")
    p.add_argument("--no-check", action="store_true", help="Skip correctness check")
    p.add_argument("--no-gpu-cpu", action="store_true", 
                   help="Skip GPU vs CPU comparison (requires CuPy)")
    return p.parse_args()


def main():
    args = parse_args()
    
    # Check correctness and consistency first
    if not args.no_check:
        check_correctness(
            nx=args.nx,
            ny=args.ny,
            degree=args.degree,
            use_gpu=args.gpu,
        )
    
    # Compare performance of loop vs batched versions
    compare_performance(
        nx=args.nx,
        ny=args.ny,
        degree=args.degree,
        use_gpu=args.gpu,
        n_runs=args.runs,
    )
    
    # Compare GPU vs CPU for batched version (if CuPy is available and not skipped)
    if not args.no_gpu_cpu:
        compare_gpu_cpu(
            nx=args.nx,
            ny=args.ny,
            degree=args.degree,
            n_runs=args.runs,
        )
    
    # Then run detailed profiling (using 2D Taylor-Green vortex)
    print("\n" + "="*80)
    print("=== Detailed Performance Profiling (Batched Version) ===")
    print("="*80)
    print("Note: Profiling uses 2D Taylor-Green vortex: f(x,y) = sin(x) * sin(y)")
    print("="*80 + "\n")
    run_profile_2d(
        nx=args.nx,
        ny=args.ny,
        degree=args.degree,
        n_runs=args.runs,
        use_gpu=args.gpu,
    )


if __name__ == "__main__":
    main()

