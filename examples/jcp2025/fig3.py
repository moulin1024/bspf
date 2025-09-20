import numpy as np
import matplotlib.pyplot as plt
import time
from bfpsm1d import bfpsm1d
from chebyshev import chebyshev_derivative_from_values, _construct_chebyshev_nodes

def time_bfpsm(N, n_runs=100):
    """Time BSLF application phase for size N"""
    # Setup
    a, b = 0.0, 2.0 * np.pi
    x = np.linspace(a, b, N)
    degree = 10
    order = degree
    num_boundary_points = order + 8
    n_basis = 2*(degree+1)*2
    lam = 0.01
    
    # Initialize bfpsm1d model
    model = bfpsm1d.from_grid(
        degree=degree,
        x=x,
        n_basis=n_basis,
        domain=(a, b),
        use_clustering=False,
        order=order,
        num_boundary_points=num_boundary_points,
        correction="spectral"
    )
    
    # Test function
    f = np.sin(100.5*x)
    
    # Warmup
    _, _ = model.differentiate(f, k=1, lam=lam)
    
    # Timing
    times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        _, _ = model.differentiate(f, k=1, lam=lam)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return np.array(times)

def time_chebyshev(N, n_runs=100):
    """Time Chebyshev derivative for size N"""
    # Setup
    a, b = 0.0, 2.0 * np.pi
    
    # Pre-compute Chebyshev nodes once
    x, _ = _construct_chebyshev_nodes(N-1, domain=(a,b))
    f_vals = np.sin(100.5*x)  # evaluate function at nodes
    
    # Warmup
    _ = chebyshev_derivative_from_values(f_vals, x, domain=(a,b))
    
    # Timing
    times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        _ = chebyshev_derivative_from_values(f_vals, x, domain=(a,b))
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return np.array(times)

from pade10 import Pade10NonPeriodic
def time_pade10(N, n_runs=100):
    """Time Pade10 derivative for size N"""
    a, b = 0.0, 2.0 * np.pi
    x = np.linspace(a, b, N)
    f = np.sin(100.5*x)
    op = Pade10NonPeriodic(N, 2*np.pi/(N-1), acc=10)
    times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        _ = op(f)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    return np.array(times)

def run_comparison():
    
    # Now do the regular comparison across different sizes
    k_range = range(6, 18)  # 256 to 8192 points
    N_values_bfpsm = np.array([2**k for k in k_range])
    N_values_cheb = N_values_bfpsm + 1
    
    times_bfpsm = []
    times_cheb = []
    times_pade10 = []
    print("\nRunning size scaling comparison...")
    print("Testing sizes (points):", N_values_bfpsm)
    
    for N_bfpsm, N_cheb in zip(N_values_bfpsm, N_values_cheb):
        print(f"Testing N = {N_bfpsm}")
        times_bfpsm.append(np.mean(time_bfpsm(N_bfpsm)))
        times_cheb.append(np.mean(time_chebyshev(N_cheb)))
        # times_pade10.append(np.mean(time_pade10(N_cheb)))
    times_bfpsm = np.array(times_bfpsm)
    times_cheb = np.array(times_cheb)
    # times_pade10 = np.array(times_pade10)
    
    # Scaling plot
    plt.figure(figsize=(10, 6))

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
    
    # N log N reference
    nlogn = N_values_bfpsm * np.log(N_values_bfpsm)
    scale_factor = min(times_bfpsm[-1], times_cheb[-1]) / nlogn[-1]
    plt.loglog(N_values_bfpsm, scale_factor * nlogn, 'k--', 
               label='$O(N log N)$ reference', linewidth=1.5)
    
    plt.loglog(N_values_bfpsm, times_bfpsm, 'o-', label='BSPF', linewidth=1.5)
    plt.loglog(N_values_cheb, times_cheb, 'o-', label='Chebyshev', linewidth=1.5)
    # plt.loglog(N_values_bfpsm, times_pade10, 'o-', label='Pade10', linewidth=1.5)
    plt.grid(True)
    plt.xlabel('$N$')
    plt.ylabel('Time per run (seconds)')
    plt.legend()
    plt.ylim(1e-5, 1e-2)
    
    # Calculate and print scaling ratios
    def calc_scaling(times):
        ratios = np.log2(times[1:] / times[:-1]) / np.log2(N_values_bfpsm[1:] / N_values_bfpsm[:-1])
        return np.mean(ratios)
    
    scaling_bfpsm = calc_scaling(times_bfpsm)
    scaling_cheb = calc_scaling(times_cheb)
    # scaling_pade10 = calc_scaling(times_pade10)
    print("\nScaling Analysis:")
    print(f"BSLF scaling factor:     {scaling_bfpsm:.2f}")
    print(f"Chebyshev scaling factor: {scaling_cheb:.2f}")
    # print(f"Pade10 scaling factor: {scaling_pade10:.2f}")
    print("(1.0 = perfect N log N scaling)")
    
    print("\nPerformance at N =", N_values_bfpsm[-1])
    print(f"BSLF:     {1e3 * times_bfpsm[-1]:.2f} ms per run")
    print(f"Chebyshev: {1e3 * times_cheb[-1]:.2f} ms per run")
    # print(f"Pade10: {1e3 * times_pade10[-1]:.2f} ms per run")
    print(f"Speedup:   {times_cheb[-1]/times_bfpsm[-1]:.2f}x")

    # plt.show()
    plt.savefig("figs/fig3.pdf", dpi=300, bbox_inches='tight')

    # write the times OF bfpsm and cheb to a single file
    np.savetxt("timing_data.txt", np.column_stack((k_range, times_bfpsm, times_cheb)))
    # np.savetxt("times_cheb.txt", times_cheb)
    # np.savetxt("times_pade10.txt", times_pade10)

    return N_values_bfpsm, times_bfpsm, times_cheb

if __name__ == "__main__":
    run_comparison()
