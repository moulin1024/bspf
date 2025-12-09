"""
Performance Comparison: BSPF  vs Chebyshev Spectral Methods.

This script compares the computational performance of BSPF  and Chebyshev
spectral methods for computing derivatives. It measures execution time
and scaling behavior across different grid sizes.

It generates a figure showing:
    - Execution time vs grid size for both methods
    - Scaling analysis comparing to O(N log N) reference
    - Performance metrics and speedup ratios

Run from repository root:
    python examples/performance/bspf_vs_cheb.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'paper'))

import numpy as np
import matplotlib.pyplot as plt
import time
import sympy as sp

from bspf import bspf1d
from specderiv import cheb_deriv

# =============================================================================
# Configuration Parameters
# =============================================================================
# Domain
DOMAIN = (0.0, 2.0 * np.pi)

# BSPF parameters
DEGREE = 7
ORDER = DEGREE
NUM_BOUNDARY_POINTS = ORDER
N_BASIS = 4 * DEGREE
LAM = 0.01
USE_CLUSTERING = True
CLUSTERING_FACTOR = 2.0

# Test function: f(t) = sin( t / (1.01 + cos t) )
t_sym = sp.symbols("t")
f_sym = sp.sin(t_sym / (1.001 + sp.cos(t_sym)))
f_prime_sym = sp.diff(f_sym, t_sym)
f_second_sym = sp.diff(f_prime_sym, t_sym)

# Numeric callables
f_func = sp.lambdify(t_sym, f_sym, modules=["numpy"])
f_prime_func = sp.lambdify(t_sym, f_prime_sym, modules=["numpy"])
f_second_func = sp.lambdify(t_sym, f_second_sym, modules=["numpy"])

# Timing parameters
N_RUNS = 10


def time_bspf_cpu(N, n_runs=N_RUNS):
    """Time BSPF  application phase for size N"""
    # Setup
    a, b = DOMAIN
    x = np.linspace(a, b, N)
    
    # Initialize bspf1d model (CPU implementation)
    model = bspf1d.from_grid(
        degree=DEGREE,
        x=x,
        n_basis=N_BASIS,
        domain=DOMAIN,
        use_clustering=USE_CLUSTERING,
        clustering_factor=CLUSTERING_FACTOR,
        order=ORDER,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        correction="spectral",
        use_gpu=False
    )
    
    # Test function
    f = f_func(x)
    
    # Warmup
    _, _ = model.differentiate(f, k=1, lam=LAM)
    
    # Timing
    times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        _, _ = model.differentiate(f, k=1, lam=LAM)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return np.array(times)


def time_bspf_gpu(N, n_runs=N_RUNS):
    """Time BSPF (GPU) application phase for size N"""
    # Setup
    a, b = DOMAIN
    x = np.linspace(a, b, N)
    
    # Initialize bspf1d model (GPU implementation)
    model = bspf1d.from_grid(
        degree=DEGREE,
        x=x,
        n_basis=N_BASIS,
        domain=DOMAIN,
        use_clustering=USE_CLUSTERING,
        clustering_factor=CLUSTERING_FACTOR,
        order=ORDER,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        correction="spectral",
        use_gpu=True
    )
    
    # Test function - convert to GPU array
    try:
        import cupy as cp
        f = cp.asarray(np.tanh(TANH_ALPHA * (x - TANH_CENTER)))
    except ImportError:
        raise RuntimeError("CuPy is required for GPU benchmarking. Install cupy (e.g., `pip install cupy-cuda12x`)")
    
    # Warmup
    _, _ = model.differentiate(f, k=1, lam=LAM)
    
    # Synchronize before timing
    cp.cuda.Stream.null.synchronize()
    
    # Timing
    times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        _, _ = model.differentiate(f, k=1, lam=LAM)
        cp.cuda.Stream.null.synchronize()  # Synchronize to ensure GPU work is done
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return np.array(times)


def time_chebyshev(N, n_runs=N_RUNS):
    """Time Chebyshev derivative for size N"""
    # Setup Chebyshev-Gauss-Lobatto nodes (N intervals → N+1 nodes)
    a, b = DOMAIN
    n_intervals = N - 1
    t = np.cos(np.arange(n_intervals + 1) * np.pi / n_intervals)  # [-1, 1]
    x = (b - a) * t / 2.0 + (b + a) / 2.0
    f_vals = f_func(x)  # evaluate function at nodes
    
    # Warmup
    _ = cheb_deriv(f_vals, x, order=1)
    
    # Timing
    times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        _ = cheb_deriv(f_vals, x, order=1)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return np.array(times)


def compute_errors(N_bspf, N_cheb):
    """Compute errors for both BSPF and Chebyshev methods."""
    # Setup
    a, b = DOMAIN
    x_bspf = np.linspace(a, b, N_bspf)
    # Chebyshev-Gauss-Lobatto nodes (N intervals → N+1 nodes)
    n_intervals = N_cheb - 1
    t_cheb = np.cos(np.arange(n_intervals + 1) * np.pi / n_intervals)  # [-1,1]
    x_cheb = (b - a) * t_cheb / 2.0 + (b + a) / 2.0
    
    # Test function and its analytical derivatives via sympy lambdify
    f_bspf = f_func(x_bspf)
    f_cheb = f_func(x_cheb)
    
    df_exact_bspf = f_prime_func(x_bspf)
    df_exact_cheb = f_prime_func(x_cheb)
    
    # BSPF computation
    model = bspf1d.from_grid(
        degree=DEGREE,
        x=x_bspf,
        n_basis=N_BASIS,
        domain=DOMAIN,
        use_clustering=USE_CLUSTERING,
        clustering_factor=CLUSTERING_FACTOR,
        order=ORDER,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        correction="spectral",
        use_gpu=False
    )
    
    df_bspf, _ = model.differentiate(f_bspf, k=1, lam=LAM)
    
    # Chebyshev computation
    df_cheb = cheb_deriv(f_cheb, x_cheb, order=1)
    
    # Compute L2 errors
    error_bspf = np.sqrt(np.mean((df_bspf - df_exact_bspf)**2))
    error_cheb = np.sqrt(np.mean((df_cheb - df_exact_cheb)**2))
    
    return error_bspf, error_cheb


def run_comparison():
    """Run performance comparison between BSPF and Chebyshev methods."""
    
    # Now do the regular comparison across different sizes
    k_range = range(6, 18)  # 256 to 8192 points
    N_values_bspf = np.array([2**k for k in k_range])
    N_values_cheb = N_values_bspf + 1
    
    times_bspf_cpu = []
    times_cheb = []
    errors_bspf = []
    errors_cheb = []
    print("\nRunning size scaling comparison...")
    print("Testing sizes (points):", N_values_bspf)
    
    for N_bspf, N_cheb in zip(N_values_bspf, N_values_cheb):
        print(f"Testing N = {N_bspf}")
        times_bspf_cpu.append(np.mean(time_bspf_cpu(N_bspf)))
        times_cheb.append(np.mean(time_chebyshev(N_cheb)))
        # Compute errors
        err_bspf, err_cheb = compute_errors(N_bspf, N_cheb)
        errors_bspf.append(err_bspf)
        errors_cheb.append(err_cheb)
    times_bspf_cpu = np.array(times_bspf_cpu)
    times_cheb = np.array(times_cheb)
    errors_bspf = np.array(errors_bspf)
    errors_cheb = np.array(errors_cheb)
    
    # Set up global plotting parameters
    plt.rcParams.update({
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 16,
        'figure.titlesize': 20,
        'axes.grid': True,
        'grid.alpha': 0.5
    })
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Performance plot (left)
    nlogn = N_values_bspf * np.log(N_values_bspf)
    scale_factor = min(times_bspf_cpu[-1], times_cheb[-1]) / nlogn[-1]
    ax1.loglog(N_values_bspf, scale_factor * nlogn, 'k--', 
               label='$O(N \\log N)$ reference', linewidth=1.5)
    ax1.loglog(N_values_bspf, times_bspf_cpu, 'o-', label='BSPF ', linewidth=1.5, markersize=6)
    ax1.loglog(N_values_cheb, times_cheb, '^-', label='Chebyshev', linewidth=1.5, markersize=6)
    ax1.grid(True)
    ax1.set_xlabel('$N$')
    ax1.set_ylabel('Time per run (seconds)')
    ax1.set_title('Performance')
    ax1.legend()
    ax1.set_ylim(1e-5, 1e-2)
    
    # Convergence plot (right)
    ax2.loglog(N_values_bspf, errors_bspf, 'o-', label='BSPF ', linewidth=1.5, markersize=6)
    ax2.loglog(N_values_cheb, errors_cheb, '^-', label='Chebyshev', linewidth=1.5, markersize=6)
    ax2.grid(True)
    ax2.set_xlabel('$N$')
    ax2.set_ylabel('$L^2$ error')
    ax2.set_title('Convergence')
    ax2.legend()
    
    plt.tight_layout()
    
    # Calculate and print scaling ratios
    def calc_scaling(times):
        # Filter out NaN values
        valid_mask = ~np.isnan(times)
        if np.sum(valid_mask) < 2:
            return np.nan
        valid_times = times[valid_mask]
        valid_N = N_values_bspf[valid_mask]
        ratios = np.log2(valid_times[1:] / valid_times[:-1]) / np.log2(valid_N[1:] / valid_N[:-1])
        return np.mean(ratios)
    
    scaling_bspf_cpu = calc_scaling(times_bspf_cpu)
    scaling_cheb = calc_scaling(times_cheb)
    print("\nScaling Analysis:")
    print(f"BSPF  scaling factor:     {scaling_bspf_cpu:.2f}")
    print(f"Chebyshev scaling factor:       {scaling_cheb:.2f}")
    print("(1.0 = perfect N log N scaling)")
    
    print("\nPerformance at N =", N_values_bspf[-1])
    print(f"BSPF :     {1e3 * times_bspf_cpu[-1]:.2f} ms per run")
    print(f"Chebyshev:       {1e3 * times_cheb[-1]:.2f} ms per run")
    print(f"\nSpeedup (BSPF CPU vs Chebyshev):   {times_cheb[-1]/times_bspf_cpu[-1]:.2f}x")
    
    print("\nConvergence Analysis:")
    print(f"BSPF  error at N={N_values_bspf[-1]}:     {errors_bspf[-1]:.6e}")
    print(f"Chebyshev error at N={N_values_cheb[-1]}:       {errors_cheb[-1]:.6e}")
    
    # Calculate convergence rates
    def calc_convergence_rate(errors, N_values):
        """Calculate convergence rate from error data."""
        valid_mask = errors > 0
        if np.sum(valid_mask) < 2:
            return np.nan
        valid_errors = errors[valid_mask]
        valid_N = N_values[valid_mask]
        # Use log-log slope to estimate convergence rate
        log_errors = np.log(valid_errors)
        log_N = np.log(valid_N)
        # Fit line to last few points for stability
        n_fit = min(5, len(log_errors))
        if n_fit < 2:
            return np.nan
        coeffs = np.polyfit(log_N[-n_fit:], log_errors[-n_fit:], 1)
        return -coeffs[0]  # Negative slope is the convergence rate
    
    conv_rate_bspf = calc_convergence_rate(errors_bspf, N_values_bspf)
    conv_rate_cheb = calc_convergence_rate(errors_cheb, N_values_cheb)
    print(f"\nConvergence rates (estimated):")
    if not np.isnan(conv_rate_bspf):
        print(f"BSPF :     {conv_rate_bspf:.2f}")
    if not np.isnan(conv_rate_cheb):
        print(f"Chebyshev:       {conv_rate_cheb:.2f}")

    plt.show()
    # plt.savefig("figs/fig3.pdf", dpi=300, bbox_inches='tight')

    # write the times and errors of bspf  and cheb to a single file
    np.savetxt("timing_data.txt", np.column_stack((k_range, times_bspf_cpu, times_cheb, errors_bspf, errors_cheb)))

    return N_values_bspf, times_bspf_cpu, times_cheb, errors_bspf, errors_cheb


if __name__ == "__main__":
    run_comparison()

