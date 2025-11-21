"""
Unit test for BSPF vs Chebyshev performance comparison.

Tests that BSPF1D performance is within 10% of Chebyshev at N=131072 with the
configuration from examples/performance/bspf_vs_cheb_1d.py.
"""

import numpy as np
import time
import pytest

from bspf import bspf1d
from bspf.utils import chebyshev_derivative_from_values, construct_chebyshev_nodes


# Configuration from bspf_vs_cheb_1d.py
DOMAIN = (0.0, 2.0 * np.pi)
DEGREE = 5
ORDER = DEGREE
NUM_BOUNDARY_POINTS = ORDER
N_BASIS = 2 * DEGREE
LAM = 0.01
USE_CLUSTERING = True
CLUSTERING_FACTOR = 2.0
TANH_ALPHA = 100.0
TANH_CENTER = np.pi

# Timing parameters - use fewer runs for unit test to keep it fast
N_RUNS = 100


def time_bspf_cpu(N, n_runs=N_RUNS):
    """Time BSPF application phase for size N"""
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
    f = np.tanh(TANH_ALPHA * (x - TANH_CENTER))
    
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


def time_chebyshev(N, n_runs=N_RUNS):
    """Time Chebyshev derivative for size N"""
    # Setup
    a, b = DOMAIN
    
    # Pre-compute Chebyshev nodes once
    x, _ = construct_chebyshev_nodes(N-1, domain=DOMAIN)
    f_vals = np.tanh(TANH_ALPHA * (x - TANH_CENTER))  # evaluate function at nodes
    
    # Warmup
    _ = chebyshev_derivative_from_values(f_vals, x, domain=DOMAIN)
    
    # Timing
    times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        _ = chebyshev_derivative_from_values(f_vals, x, domain=DOMAIN)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return np.array(times)


def test_bspf_performance_within_10pct_of_chebyshev():
    """
    Test that BSPF1D performance is within 10% of Chebyshev at N=131072.
    
    Uses the same configuration as examples/performance/bspf_vs_cheb_1d.py:
    - DEGREE = 5
    - ORDER = 5
    - NUM_BOUNDARY_POINTS = 5
    - N_BASIS = 10
    - LAM = 0.01
    - USE_CLUSTERING = True
    - CLUSTERING_FACTOR = 2.0
    - Test function: tanh(100 * (x - pi))
    """
    N = 131072
    N_cheb = N + 1  # Chebyshev uses N+1 nodes
    
    # Time both methods
    times_bspf = time_bspf_cpu(N, n_runs=N_RUNS)
    times_cheb = time_chebyshev(N_cheb, n_runs=N_RUNS)
    
    # Compute mean times
    mean_time_bspf = np.mean(times_bspf)
    mean_time_cheb = np.mean(times_cheb)
    
    # Compute performance ratio (BSPF time / Chebyshev time)
    # BSPF should be within 10% of Chebyshev, meaning ratio <= 1.1
    performance_ratio = mean_time_bspf / mean_time_cheb
    
    # Assert performance requirement
    assert performance_ratio <= 1.1, (
        f"BSPF1D performance {mean_time_bspf*1000:.3f} ms is not within 10% of "
        f"Chebyshev {mean_time_cheb*1000:.3f} ms at N={N}. "
        f"Ratio: {performance_ratio:.3f} (required: <= 1.1)"
    )
    
    # Also check that BSPF is not significantly faster (optional sanity check)
    # This ensures we're comparing similar implementations
    assert performance_ratio >= 0.5, (
        f"BSPF1D appears too fast relative to Chebyshev. "
        f"Ratio: {performance_ratio:.3f} (suspicious if < 0.5)"
    )

