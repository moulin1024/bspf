#!/usr/bin/env python3
"""
Benchmark CPU vs GPU speedup for BDF2 time stepper with bspf1d spatial differentiator.

Tests the performance of BDF2 implicit time stepping method when using bspf1d
for spatial differentiation. Compares CPU (NumPy) vs GPU (CuPy) performance
across different grid sizes.

Requirements:
- bspf package with GPU support
- cupy (for GPU runs): pip install cupy-cuda12x (or appropriate CUDA version)
"""

import time
import argparse
import numpy as np
from typing import Tuple, List, Dict

from bspf import bspf1d, TimeStepperState, time_step

try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False


def sync_gpu():
    """Synchronize GPU operations."""
    if _HAS_CUPY:
        cp.cuda.Stream.null.synchronize()


def build_laplacian_matrix_bspf(bspf_op, n, neumann_bc=(0.0, 0.0), use_gpu=False):
    """
    Build Laplacian matrix using native BSPF differentiate method.
    
    Parameters:
    -----------
    bspf_op : bspf1d
        B-spline operator
    n : int
        Grid size (number of points)
    neumann_bc : tuple
        Neumann boundary conditions (left_flux, right_flux)
    use_gpu : bool
        Whether to use GPU arrays
        
    Returns:
    --------
    L : array
        Laplacian matrix
    """
    # Select backend
    if use_gpu and _HAS_CUPY:
        xp = cp
    else:
        xp = np
    
    # Create identity matrix
    I = xp.eye(n, dtype=np.float64)
    
    # Apply differentiate to each column of identity matrix
    L_columns = [bspf_op.differentiate(I[:, k], k=2, neumann_bc=neumann_bc)[0] 
                 for k in range(n)]
    
    # Convert list of columns to array and transpose
    L = xp.array(L_columns, dtype=np.complex128).T
    
    return L


def create_test_rhs(bspf_op, g=0.0, neumann_bc=(0.0, 0.0)):
    """
    Create RHS function for test equation: i*ψ_t = ψ_xx + g*|ψ|²*ψ
    
    Parameters:
    -----------
    bspf_op : bspf1d
        BSPF operator
    g : float
        Nonlinear coupling constant
    neumann_bc : tuple
        Neumann boundary conditions
        
    Returns:
    --------
    rhs_func : callable
        RHS function
    """
    def rhs_func(psi):
        # Compute second derivative
        lap, _ = bspf_op.differentiate(psi, k=2, neumann_bc=neumann_bc)
        
        # Nonlinear term
        if g != 0.0:
            if _HAS_CUPY and isinstance(psi, cp.ndarray):
                nl = g * cp.abs(psi)**2 * psi
            else:
                nl = g * np.abs(psi)**2 * psi
            return 1j * (lap + nl)
        else:
            return 1j * lap
    
    return rhs_func


def create_test_jacobian(L_complex, g=0.0):
    """
    Create Jacobian function for BDF2.
    
    Parameters:
    -----------
    L_complex : array
        Precomputed Laplacian matrix
    g : float
        Nonlinear coupling constant
        
    Returns:
    --------
    jacobian_func : callable
        Jacobian function
    """
    def jacobian_func(psi):
        # Detect backend
        if _HAS_CUPY and isinstance(psi, cp.ndarray):
            xp = cp
        else:
            xp = np
        
        # Linear part: i·L
        J_linear = 1j * L_complex
        
        # Nonlinear part: diagonal matrix with 2i·g·|ψ|²
        J_nonlinear = xp.diag(2j * g * xp.abs(psi)**2)
        
        return J_linear + J_nonlinear
    
    return jacobian_func


def run_bdf2_benchmark(
    N: int,
    n_steps: int,
    dt: float,
    use_gpu: bool,
    degree: int = 5,
    g: float = -1.0,
    warmup_steps: int = 5,
    n_runs: int = 3
) -> Dict:
    """
    Run BDF2 benchmark for a given grid size.
    
    Parameters:
    -----------
    N : int
        Number of grid points
    n_steps : int
        Number of time steps
    dt : float
        Time step
    use_gpu : bool
        Use GPU if True
    degree : int
        B-spline degree
    g : float
        Nonlinear coupling constant
    warmup_steps : int
        Number of warmup steps
    n_runs : int
        Number of benchmark runs for averaging
        
    Returns:
    --------
    results : dict
        Benchmark results
    """
    if use_gpu and not _HAS_CUPY:
        raise RuntimeError("use_gpu=True but CuPy is not available")
    
    # Create grid
    x_min, x_max = -1.0, 1.0
    if use_gpu and _HAS_CUPY:
        x = cp.linspace(x_min, x_max, N+1)
        xp = cp
    else:
        x = np.linspace(x_min, x_max, N+1)
        xp = np
    
    # Create BSPF operator
    bspf_op = bspf1d.from_grid(
        degree=degree,
        x=x,
        use_clustering=True,
        clustering_factor=2.0,
        correction='spectral',
        use_gpu=use_gpu
    )
    
    # Initial condition: modulational instability
    psi_init = (1.0 + 0.1*xp.cos(4*x)).astype(np.complex128)
    
    # Precompute Laplacian matrix for Jacobian
    L_complex = build_laplacian_matrix_bspf(bspf_op, len(x), neumann_bc=(0.0, 0.0), use_gpu=use_gpu)
    
    # Create RHS and Jacobian functions
    rhs_func = create_test_rhs(bspf_op, g=g, neumann_bc=(0.0, 0.0))
    jacobian_func = create_test_jacobian(L_complex, g=g)
    
    # Warmup runs
    for _ in range(warmup_steps):
        state = TimeStepperState(psi_init.copy(), t_init=0.0, dt=dt, method='bdf2')
        for _ in range(min(10, n_steps)):
            _ = time_step(state, dt, rhs_func, method='bdf2', jacobian_func=jacobian_func)
        if use_gpu:
            sync_gpu()
    
    # Benchmark runs
    times = []
    for run in range(n_runs):
        # Reset initial condition
        psi = psi_init.copy()
        state = TimeStepperState(psi, t_init=0.0, dt=dt, method='bdf2')
        
        # Time the integration
        t_start = time.perf_counter()
        for step in range(n_steps):
            _ = time_step(state, dt, rhs_func, method='bdf2', jacobian_func=jacobian_func)
        if use_gpu:
            sync_gpu()
        t_end = time.perf_counter()
        
        times.append(t_end - t_start)
    
    # Compute statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    return {
        'N': N,
        'n_steps': n_steps,
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'time_per_step': mean_time / n_steps,
        'use_gpu': use_gpu
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark BDF2 time stepper with bspf1d (CPU vs GPU)"
    )
    parser.add_argument("--grid-sizes", type=int, nargs="+", 
                       default=[128, 256, 512, 1024, 2048],
                       help="Grid sizes to test (default: 128 256 512 1024 2048)")
    parser.add_argument("--n-steps", type=int, default=100,
                       help="Number of time steps (default: 100)")
    parser.add_argument("--dt", type=float, default=0.001,
                       help="Time step (default: 0.001)")
    parser.add_argument("--degree", type=int, default=5,
                       help="B-spline degree (default: 5)")
    parser.add_argument("--g", type=float, default=-1.0,
                       help="Nonlinear coupling constant (default: -1.0)")
    parser.add_argument("--warmup", type=int, default=3,
                       help="Number of warmup runs (default: 3)")
    parser.add_argument("--n-runs", type=int, default=5,
                       help="Number of benchmark runs per test (default: 5)")
    parser.add_argument("--skip-gpu", action="store_true",
                       help="Skip GPU benchmark (useful if CuPy not available)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("BDF2 Time Stepper + bspf1d: CPU vs GPU Benchmark")
    print("=" * 80)
    print(f"Time steps: {args.n_steps}")
    print(f"Time step size: {args.dt}")
    print(f"B-spline degree: {args.degree}")
    print(f"Nonlinear coupling: {args.g}")
    print(f"Warmup runs: {args.warmup}")
    print(f"Benchmark runs per test: {args.n_runs}")
    print()
    
    if not _HAS_CUPY and not args.skip_gpu:
        print("WARNING: CuPy not available. GPU benchmarks will be skipped.")
        print("Install CuPy with: pip install cupy-cuda12x (or appropriate CUDA version)")
        print()
        args.skip_gpu = True
    
    results_cpu = []
    results_gpu = []
    
    for N in args.grid_sizes:
        print(f"Grid size: {N} points")
        print("-" * 80)
        
        # CPU benchmark
        print("  CPU:")
        try:
            result_cpu = run_bdf2_benchmark(
                N=N, n_steps=args.n_steps, dt=args.dt,
                use_gpu=False, degree=args.degree, g=args.g,
                warmup_steps=args.warmup, n_runs=args.n_runs
            )
            results_cpu.append(result_cpu)
            print(f"    Total time:     {result_cpu['mean_time']*1000:.2f} ± {result_cpu['std_time']*1000:.2f} ms")
            print(f"    Time per step:  {result_cpu['time_per_step']*1000:.4f} ms")
            print(f"    Min time:       {result_cpu['min_time']*1000:.2f} ms")
            print(f"    Max time:       {result_cpu['max_time']*1000:.2f} ms")
        except Exception as e:
            print(f"    ERROR: {e}")
            results_cpu.append(None)
        
        # GPU benchmark
        if not args.skip_gpu:
            print("  GPU:")
            try:
                result_gpu = run_bdf2_benchmark(
                    N=N, n_steps=args.n_steps, dt=args.dt,
                    use_gpu=True, degree=args.degree, g=args.g,
                    warmup_steps=args.warmup, n_runs=args.n_runs
                )
                results_gpu.append(result_gpu)
                print(f"    Total time:     {result_gpu['mean_time']*1000:.2f} ± {result_gpu['std_time']*1000:.2f} ms")
                print(f"    Time per step:  {result_gpu['time_per_step']*1000:.4f} ms")
                print(f"    Min time:       {result_gpu['min_time']*1000:.2f} ms")
                print(f"    Max time:       {result_gpu['max_time']*1000:.2f} ms")
                
                # Compute speedup
                if result_cpu is not None:
                    speedup = result_cpu['mean_time'] / result_gpu['mean_time']
                    print(f"    Speedup:        {speedup:.2f}x")
            except Exception as e:
                print(f"    ERROR: {e}")
                results_gpu.append(None)
        
        print()
    
    # Summary table
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Grid Size':<12} {'CPU (ms)':<15} {'GPU (ms)':<15} {'Speedup':<12}")
    print("-" * 80)
    
    for i, N in enumerate(args.grid_sizes):
        cpu_result = results_cpu[i] if i < len(results_cpu) else None
        gpu_result = results_gpu[i] if i < len(results_gpu) else None
        
        cpu_str = f"{cpu_result['mean_time']*1000:.2f}" if cpu_result else "N/A"
        gpu_str = f"{gpu_result['mean_time']*1000:.2f}" if gpu_result else "N/A"
        
        if cpu_result and gpu_result:
            speedup = cpu_result['mean_time'] / gpu_result['mean_time']
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"
        
        print(f"{N:<12} {cpu_str:<15} {gpu_str:<15} {speedup_str:<12}")
    
    print("=" * 80)
    
    # Average speedup
    if not args.skip_gpu:
        speedups = []
        for i in range(len(args.grid_sizes)):
            if (i < len(results_cpu) and results_cpu[i] and 
                i < len(results_gpu) and results_gpu[i]):
                speedup = results_cpu[i]['mean_time'] / results_gpu[i]['mean_time']
                speedups.append(speedup)
        
        if speedups:
            avg_speedup = np.mean(speedups)
            print(f"\nAverage speedup (GPU vs CPU): {avg_speedup:.2f}x")
            print(f"Speedup range: {np.min(speedups):.2f}x - {np.max(speedups):.2f}x")


if __name__ == "__main__":
    main()

