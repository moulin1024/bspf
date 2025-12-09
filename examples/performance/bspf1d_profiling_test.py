"""
Standalone test for bspf1d differentiate_1_2 with timing breakdown.

Run:
    python examples/performance/bspf1d_profiling_test.py
    python examples/performance/bspf1d_profiling_test.py --gpu
    python examples/performance/bspf1d_profiling_test.py --complex
    python examples/performance/bspf1d_profiling_test.py --test-complex
    python examples/performance/bspf1d_profiling_test.py --test-complex --gpu
"""

import argparse
import numpy as np
import sympy as sp

# Optional GPU support
_HAS_CUPY = False
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None

from bspf1d_profiling import bspf1d


def test_complex_input(use_gpu=False, n=1024, degree=7):
    """
    Explicit test for complex128 input support.
    Tests with a simple complex function: f(x) = exp(i*x) = cos(x) + i*sin(x)
    """
    print(f"\n{'='*80}")
    print(f"=== Testing Complex Input Support ===")
    print(f"{'='*80}")
    print(f"use_gpu={use_gpu}, n={n}, degree={degree}")
    
    # Domain and grid
    a, b = 0.0, 2.0 * np.pi
    x = np.linspace(a, b, n, endpoint=True)
    
    # Complex test function: f(x) = exp(i*x) = cos(x) + i*sin(x)
    # Exact derivatives:
    #   df/dx = i*exp(i*x) = -sin(x) + i*cos(x)
    #   d2f/dx2 = -exp(i*x) = -cos(x) - i*sin(x)
    f = np.exp(1j * x).astype(np.complex128)
    f1_exact = 1j * np.exp(1j * x).astype(np.complex128)
    f2_exact = -np.exp(1j * x).astype(np.complex128)
    
    # Convert to GPU if needed
    if use_gpu and _HAS_CUPY:
        x = cp.asarray(x, dtype=cp.float64)
        f = cp.asarray(f, dtype=cp.complex128)
        # Keep exact derivatives on CPU for comparison
        f1_exact = f1_exact  # NumPy array
        f2_exact = f2_exact  # NumPy array
    else:
        if use_gpu:
            print("Warning: --gpu specified but CuPy is not available. Using CPU.")
    
    # Build operator
    op = bspf1d.from_grid(degree=degree, x=x, n_basis=4*degree, use_clustering=True, 
                          clustering_factor=2.0, use_gpu=use_gpu)
    
    # Compute derivatives
    print("Computing derivatives...")
    df1, df2, f_spline = op.differentiate_1_2(f)
    
    if use_gpu and _HAS_CUPY:
        cp.cuda.Stream.null.synchronize()
        df1 = cp.asnumpy(df1)
        df2 = cp.asnumpy(df2)
    
    # Compute errors
    err1 = df1 - f1_exact
    err2 = df2 - f2_exact
    max_err1 = np.max(np.abs(err1))
    max_err2 = np.max(np.abs(err2))
    l2_err1 = np.sqrt(np.mean(np.abs(err1)**2))
    l2_err2 = np.sqrt(np.mean(np.abs(err2)**2))
    
    # Check if results are complex
    is_complex_result = np.iscomplexobj(df1) and np.iscomplexobj(df2)
    
    print(f"\nResults:")
    print(f"  Input type: complex128")
    print(f"  Output type: {'complex128' if is_complex_result else 'float64 (ERROR!)'}")
    print(f"  df1 max abs error: {max_err1:.6e}")
    print(f"  df2 max abs error: {max_err2:.6e}")
    print(f"  df1 L2 error: {l2_err1:.6e}")
    print(f"  df2 L2 error: {l2_err2:.6e}")
    
    # Verify correctness
    tolerance = 1e-6
    passed = (max_err1 < tolerance and max_err2 < tolerance and is_complex_result)
    
    if passed:
        print(f"\n✓ PASSED: Complex input test (tolerance={tolerance})")
    else:
        print(f"\n✗ FAILED: Complex input test (tolerance={tolerance})")
        if not is_complex_result:
            print("  ERROR: Output is not complex128!")
        if max_err1 >= tolerance:
            print(f"  ERROR: df1 error {max_err1:.6e} >= {tolerance}")
        if max_err2 >= tolerance:
            print(f"  ERROR: df2 error {max_err2:.6e} >= {tolerance}")
    
    print("="*80 + "\n")
    
    return passed


def run_test(use_gpu, n, degree, n_runs, f_func, f1_func, f2_func, use_complex=False):
    """Run the test for a given configuration (CPU or GPU)."""
    # Domain and grid
    a, b = 0.0, 2.0 * np.pi
    x = np.linspace(a, b, n, endpoint=True)

    f = f_func(x)
    f1_exact = f1_func(x)
    f2_exact = f2_func(x)
    
    # Convert to complex if requested
    if use_complex:
        # Create a complex function: f_complex = f_real + i * f_real_shifted
        # Use a phase-shifted version for the imaginary part
        f_imag = f_func(x + np.pi/4)  # Phase-shifted version
        f = f.astype(np.complex128) + 1j * f_imag.astype(np.complex128)
        f1_exact = f1_exact.astype(np.complex128) + 1j * f1_func(x + np.pi/4).astype(np.complex128)
        f2_exact = f2_exact.astype(np.complex128) + 1j * f2_func(x + np.pi/4).astype(np.complex128)
    
    # Convert to GPU arrays if needed
    if use_gpu and _HAS_CUPY:
        x = cp.asarray(x, dtype=cp.float64)
        if use_complex:
            f = cp.asarray(f, dtype=cp.complex128)
        else:
            f = cp.asarray(f, dtype=cp.float64)
        # Keep exact derivatives on CPU for comparison
        f1_exact = f1_exact  # NumPy array
        f2_exact = f2_exact  # NumPy array

    # BSPF operator
    op = bspf1d.from_grid(degree=degree, x=x, n_basis=4*degree, use_clustering=True, clustering_factor=2.0, use_gpu=use_gpu)

    # Warmup run
    _ = op.differentiate_1_2(f)
    if use_gpu and _HAS_CUPY:
        cp.cuda.Stream.null.synchronize()

    # Profiling loop
    all_timings = []
    for i in range(n_runs):
        if use_gpu and _HAS_CUPY:
            cp.cuda.Stream.null.synchronize()
        df1, df2, f_spline = op.differentiate_1_2(f)
        if use_gpu and _HAS_CUPY:
            cp.cuda.Stream.null.synchronize()
        timings = getattr(op, "last_timing_d12", None)
        if timings:
            all_timings.append(timings)
    
    # Error metrics vs exact derivatives (use last run)
    # Convert GPU results to NumPy for comparison
    if use_gpu and _HAS_CUPY:
        df1 = cp.asnumpy(df1)
        df2 = cp.asnumpy(df2)
    
    err1 = df1 - f1_exact
    err2 = df2 - f2_exact
    # For complex arrays, use absolute value for error metrics
    max_err1 = np.max(np.abs(err1))
    max_err2 = np.max(np.abs(err2))
    # L2 error: sqrt(mean(|err|^2)) for complex, sqrt(mean(err^2)) for real
    if np.iscomplexobj(err1):
        l2_err1 = np.sqrt(np.mean(np.abs(err1)**2))
        l2_err2 = np.sqrt(np.mean(np.abs(err2)**2))
    else:
        l2_err1 = np.sqrt(np.mean(err1**2))
        l2_err2 = np.sqrt(np.mean(err2**2))
    
    # Compute timing statistics
    stats = {}
    if all_timings:
        timing_keys = all_timings[0].keys()
        for key in timing_keys:
            values = [t[key] for t in all_timings]
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            }
    
    return {
        'max_err1': max_err1,
        'max_err2': max_err2,
        'l2_err1': l2_err1,
        'l2_err2': l2_err2,
        'stats': stats,
        'all_timings': all_timings,
    }


def main():
    parser = argparse.ArgumentParser(description="Test bspf1d differentiate_1_2 with timing breakdown")
    parser.add_argument("--gpu", action="store_true", help="Use GPU (CuPy) if available")
    parser.add_argument("--n", type=int, default=2**16, help="Number of grid points (default: 65536)")
    parser.add_argument("--degree", type=int, default=7, help="B-spline degree (default: 7)")
    parser.add_argument("--runs", type=int, default=100, help="Number of profiling runs (default: 100)")
    parser.add_argument("--no-compare", action="store_true", help="Skip CPU vs GPU comparison (only run specified mode)")
    parser.add_argument("--complex", action="store_true", help="Use complex array input (f_real + i*f_imag)")
    parser.add_argument("--test-complex", action="store_true", help="Run explicit complex input test (exp(i*x))")
    args = parser.parse_args()
    
    # If --test-complex is specified, run the explicit test and exit
    if args.test_complex:
        test_complex_input(use_gpu=args.gpu, n=args.n, degree=args.degree)
        return
    
    use_gpu = args.gpu
    if use_gpu and not _HAS_CUPY:
        print("Warning: --gpu specified but CuPy is not available. Falling back to CPU.")
        print("Install CuPy to enable GPU (e.g., `pip install cupy-cuda12x`)")
        use_gpu = False
    
    # Domain and grid
    a, b = 0.0, 2.0 * np.pi
    n = args.n
    degree = args.degree
    n_runs = args.runs

    # Test function: f(x) = sin(x / (1.05 + cos x)) with Sympy exact derivatives
    t = sp.symbols("t")
    f_sym = sp.sin(t / (1.01 + sp.cos(t)))
    f1_sym = sp.diff(f_sym, t)
    f2_sym = sp.diff(f1_sym, t)
    f_func = sp.lambdify(t, f_sym, modules=["numpy"])
    f1_func = sp.lambdify(t, f1_sym, modules=["numpy"])
    f2_func = sp.lambdify(t, f2_sym, modules=["numpy"])
    
    # Determine which modes to run
    run_cpu = not use_gpu or (not args.no_compare and _HAS_CUPY)
    run_gpu = use_gpu or (not args.no_compare and _HAS_CUPY)
    
    results_cpu = None
    results_gpu = None
    
    # Run CPU test
    if run_cpu:
        print(f"\n{'='*80}")
        print(f"Running CPU test: N={n}, degree={degree}, runs={n_runs}, complex={args.complex}")
        print(f"{'='*80}")
        results_cpu = run_test(use_gpu=False, n=n, degree=degree, n_runs=n_runs, 
                              f_func=f_func, f1_func=f1_func, f2_func=f2_func,
                              use_complex=args.complex)
        
        print(f"\n=== bspf1d differentiate_1_2 (CPU) ===")
        print(f"grid: N={n}, degree={degree}, domain=({a},{b}), complex={args.complex}")
        print(f"df1 max abs: {results_cpu['max_err1']:.3e}, df2 max abs: {results_cpu['max_err2']:.3e}")
        print(f"df1 L2 err: {results_cpu['l2_err1']:.3e}, df2 L2 err: {results_cpu['l2_err2']:.3e}")
        
        if results_cpu['stats']:
            print(f"\nTiming statistics ({n_runs} runs, seconds):")
            print(f"{'Stage':<15s} {'Mean':>12s} {'Std':>12s} {'Min':>12s} {'Max':>12s} {'% of total':>12s}")
            print("-" * 75)
            
            total_mean = results_cpu['stats'].get('total', {}).get('mean', 1.0)
            for key in sorted(results_cpu['stats'].keys()):
                s = results_cpu['stats'][key]
                pct = 100.0 * s['mean'] / total_mean if total_mean > 0 else 0.0
                print(f"{key:<15s} {s['mean']:12.6f} {s['std']:12.6f} {s['min']:12.6f} {s['max']:12.6f} {pct:11.2f}%")
    
    # Run GPU test
    if run_gpu and _HAS_CUPY:
        print(f"\n{'='*80}")
        print(f"Running GPU test: N={n}, degree={degree}, runs={n_runs}, complex={args.complex}")
        print(f"{'='*80}")
        results_gpu = run_test(use_gpu=True, n=n, degree=degree, n_runs=n_runs, 
                              f_func=f_func, f1_func=f1_func, f2_func=f2_func,
                              use_complex=args.complex)
        
        print(f"\n=== bspf1d differentiate_1_2 (GPU) ===")
        print(f"grid: N={n}, degree={degree}, domain=({a},{b}), complex={args.complex}")
        print(f"df1 max abs: {results_gpu['max_err1']:.3e}, df2 max abs: {results_gpu['max_err2']:.3e}")
        print(f"df1 L2 err: {results_gpu['l2_err1']:.3e}, df2 L2 err: {results_gpu['l2_err2']:.3e}")
        
        if results_gpu['stats']:
            print(f"\nTiming statistics ({n_runs} runs, seconds):")
            print(f"{'Stage':<15s} {'Mean':>12s} {'Std':>12s} {'Min':>12s} {'Max':>12s} {'% of total':>12s}")
            print("-" * 75)
            
            total_mean = results_gpu['stats'].get('total', {}).get('mean', 1.0)
            for key in sorted(results_gpu['stats'].keys()):
                s = results_gpu['stats'][key]
                pct = 100.0 * s['mean'] / total_mean if total_mean > 0 else 0.0
                print(f"{key:<15s} {s['mean']:12.6f} {s['std']:12.6f} {s['min']:12.6f} {s['max']:12.6f} {pct:11.2f}%")
    
    # Compare CPU vs GPU if both were run
    if results_cpu and results_gpu and _HAS_CUPY:
        print(f"\n{'='*80}")
        print("=== CPU vs GPU Performance Comparison ===")
        print(f"{'='*80}")
        
        cpu_total = results_cpu['stats'].get('total', {}).get('mean', 0.0)
        gpu_total = results_gpu['stats'].get('total', {}).get('mean', 0.0)
        
        if cpu_total > 0 and gpu_total > 0:
            speedup = cpu_total / gpu_total
            print(f"\nTotal time:")
            print(f"  CPU: {cpu_total:.6f} seconds")
            print(f"  GPU: {gpu_total:.6f} seconds")
            print(f"  Speedup: {speedup:.2f}x {'(GPU faster)' if speedup > 1.0 else '(CPU faster)'}")
            
            # Compare individual stages
            print(f"\nStage-by-stage comparison:")
            print(f"{'Stage':<15s} {'CPU (s)':>12s} {'GPU (s)':>12s} {'Speedup':>12s}")
            print("-" * 55)
            
            cpu_stats = results_cpu['stats']
            gpu_stats = results_gpu['stats']
            
            # Get all stage keys (excluding 'total')
            all_stages = set(cpu_stats.keys()) | set(gpu_stats.keys())
            all_stages.discard('total')
            
            for stage in sorted(all_stages):
                cpu_time = cpu_stats.get(stage, {}).get('mean', 0.0)
                gpu_time = gpu_stats.get(stage, {}).get('mean', 0.0)
                if cpu_time > 0 and gpu_time > 0:
                    stage_speedup = cpu_time / gpu_time
                    print(f"{stage:<15s} {cpu_time:12.6f} {gpu_time:12.6f} {stage_speedup:12.2f}x")
                elif cpu_time > 0:
                    print(f"{stage:<15s} {cpu_time:12.6f} {'N/A':>12s} {'N/A':>12s}")
                elif gpu_time > 0:
                    print(f"{stage:<15s} {'N/A':>12s} {gpu_time:12.6f} {'N/A':>12s}")
        
        # Accuracy comparison
        print(f"\nAccuracy comparison:")
        print(f"  df1 max error: CPU={results_cpu['max_err1']:.3e}, GPU={results_gpu['max_err1']:.3e}")
        print(f"  df2 max error: CPU={results_cpu['max_err2']:.3e}, GPU={results_gpu['max_err2']:.3e}")
        print(f"  df1 L2 error:  CPU={results_cpu['l2_err1']:.3e}, GPU={results_gpu['l2_err1']:.3e}")
        print(f"  df2 L2 error:  CPU={results_cpu['l2_err2']:.3e}, GPU={results_gpu['l2_err2']:.3e}")


if __name__ == "__main__":
    main()

