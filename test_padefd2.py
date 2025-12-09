"""
Test script for padefd2 - Compact Padé second derivative operator.

This script tests the padefd2 class for computing second derivatives
using compact Padé finite difference schemes.
"""

import numpy as np
import matplotlib.pyplot as plt
from bspf.utils.padefd import padefd2


def test_basic_functionality():
    """Test basic functionality with a simple polynomial."""
    print("=" * 60)
    print("Test 1: Basic Functionality")
    print("=" * 60)
    
    # Test function: u(x) = x^2, so u''(x) = 2
    N = 101
    x = np.linspace(0, 1, N)
    h = x[1] - x[0]
    u = x**2
    u_exact = 2.0 * np.ones_like(x)
    
    # Compute second derivative
    d2op = padefd2(N=N, h=h, order=4)
    u_pp = d2op(u)
    
    # Compute error
    error = np.abs(u_pp - u_exact)
    max_error = np.max(error)
    l2_error = np.sqrt(np.trapz(error**2, x))
    
    print(f"Grid points: {N}")
    print(f"Grid spacing: h = {h:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"L2 error: {l2_error:.6e}")
    
    if max_error < 1e-10:
        print("✓ PASS: Basic functionality test")
    else:
        print(f"✗ FAIL: Error too large: {max_error:.6e}")
    
    return max_error < 1e-10


def test_sine_function():
    """Test with sine function: u(x) = sin(2πx), u''(x) = -4π² sin(2πx)."""
    print("\n" + "=" * 60)
    print("Test 2: Sine Function")
    print("=" * 60)
    
    N = 101
    x = np.linspace(0, 1, N)
    h = x[1] - x[0]
    u = np.sin(2 * np.pi * x)
    u_exact = -4 * np.pi**2 * np.sin(2 * np.pi * x)
    
    # Compute second derivative
    d2op = padefd2(N=N, h=h, order=4)
    u_pp = d2op(u)
    
    # Compute error
    error = np.abs(u_pp - u_exact)
    max_error = np.max(error)
    l2_error = np.sqrt(np.trapz(error**2, x))
    
    print(f"Grid points: {N}")
    print(f"Grid spacing: h = {h:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"L2 error: {l2_error:.6e}")
    
    # For 4th order scheme, expect error ~ O(h^4)
    expected_error = h**4
    print(f"Expected error scale: ~{expected_error:.6e}")
    
    if max_error < 1e-3:
        print("✓ PASS: Sine function test")
    else:
        print(f"✗ FAIL: Error too large: {max_error:.6e}")
    
    return max_error < 1e-3


def test_convergence_rate():
    """Test convergence rate - should be 4th order."""
    print("\n" + "=" * 60)
    print("Test 3: Convergence Rate")
    print("=" * 60)
    
    # Test function: u(x) = sin(2πx)
    def u_func(x):
        return np.sin(2 * np.pi * x)
    
    def u_pp_exact(x):
        return -4 * np.pi**2 * np.sin(2 * np.pi * x)
    
    N_list = [21, 41, 81, 161, 321]
    errors = []
    h_list = []
    
    for N in N_list:
        x = np.linspace(0, 1, N)
        h = x[1] - x[0]
        u = u_func(x)
        u_exact = u_pp_exact(x)
        
        d2op = padefd2(N=N, h=h, order=4)
        u_pp = d2op(u)
        
        error = np.abs(u_pp - u_exact)
        max_error = np.max(error)
        errors.append(max_error)
        h_list.append(h)
    
    # Compute convergence rate
    rates = []
    for i in range(1, len(errors)):
        rate = np.log(errors[i-1] / errors[i]) / np.log(h_list[i-1] / h_list[i])
        rates.append(rate)
    
    print("Convergence rates:")
    for i, rate in enumerate(rates):
        print(f"  h: {h_list[i]:.6e} -> {h_list[i+1]:.6e}, rate: {rate:.3f}")
    
    avg_rate = np.mean(rates)
    print(f"\nAverage convergence rate: {avg_rate:.3f} (expected ~4.0)")
    
    if avg_rate > 3.5:
        print("✓ PASS: Convergence rate test (4th order confirmed)")
    else:
        print(f"✗ FAIL: Convergence rate too low: {avg_rate:.3f}")
    
    return avg_rate > 3.5


def test_boundary_conditions():
    """Test boundary handling."""
    print("\n" + "=" * 60)
    print("Test 4: Boundary Conditions")
    print("=" * 60)
    
    N = 101
    x = np.linspace(0, 1, N)
    h = x[1] - x[0]
    u = np.sin(2 * np.pi * x)
    u_exact = -4 * np.pi**2 * np.sin(2 * np.pi * x)
    
    d2op = padefd2(N=N, h=h, order=4)
    u_pp = d2op(u)
    
    error = np.abs(u_pp - u_exact)
    
    # Check boundary regions
    boundary_size = d2op.K
    interior_start = boundary_size
    interior_end = N - boundary_size
    
    boundary_error = np.max([
        np.max(error[:boundary_size]),
        np.max(error[-boundary_size:])
    ])
    interior_error = np.max(error[interior_start:interior_end])
    
    print(f"Boundary region size: {boundary_size}")
    print(f"Boundary error (max): {boundary_error:.6e}")
    print(f"Interior error (max): {interior_error:.6e}")
    print(f"Boundary/Interior error ratio: {boundary_error/interior_error:.3f}")
    
    if boundary_error < 1e-2:
        print("✓ PASS: Boundary conditions test")
    else:
        print(f"✗ FAIL: Boundary error too large: {boundary_error:.6e}")
    
    return boundary_error < 1e-2


def test_matrix_construction():
    """Test matrix construction and compare with direct application."""
    print("\n" + "=" * 60)
    print("Test 5: Matrix Construction")
    print("=" * 60)
    
    N = 21
    x = np.linspace(0, 1, N)
    h = x[1] - x[0]
    u = np.sin(2 * np.pi * x)
    
    d2op = padefd2(N=N, h=h, order=4)
    
    # Apply operator
    u_pp1 = d2op(u)
    
    # Build matrix by applying to unit vectors
    import scipy.sparse as sp
    D2_data = []
    D2_row = []
    D2_col = []
    
    I = np.eye(N)
    for j in range(N):
        result = d2op(I[:, j])
        for i in range(N):
            if abs(result[i]) > 1e-12:
                D2_row.append(i)
                D2_col.append(j)
                D2_data.append(result[i])
    
    D2 = sp.csr_matrix((D2_data, (D2_row, D2_col)), shape=(N, N))
    
    # Apply matrix
    u_pp2 = D2 @ u
    
    # Compare
    diff = np.abs(u_pp1 - u_pp2)
    max_diff = np.max(diff)
    
    print(f"Matrix size: {D2.shape}")
    print(f"Non-zero entries: {D2.nnz}")
    print(f"Max difference (operator vs matrix): {max_diff:.6e}")
    
    # Check bandwidth
    D2_coo = D2.tocoo()
    if len(D2_coo.row) > 0:
        bandwidth = np.max(np.abs(D2_coo.row - D2_coo.col))
        print(f"Bandwidth: {bandwidth}")
    
    if max_diff < 1e-12:
        print("✓ PASS: Matrix construction test")
    else:
        print(f"✗ FAIL: Matrix and operator don't match: {max_diff:.6e}")
    
    return max_diff < 1e-12


def visualize_results():
    """Visualize results for a test function."""
    print("\n" + "=" * 60)
    print("Visualization")
    print("=" * 60)
    
    N = 101
    x = np.linspace(0, 1, N)
    h = x[1] - x[0]
    u = np.sin(2 * np.pi * x)
    u_exact = -4 * np.pi**2 * np.sin(2 * np.pi * x)
    
    d2op = padefd2(N=N, h=h, order=4)
    u_pp = d2op(u)
    
    error = np.abs(u_pp - u_exact)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Solution comparison
    ax1 = axes[0]
    ax1.plot(x, u_exact, 'r-', label='Exact', linewidth=2, alpha=0.8)
    ax1.plot(x, u_pp, 'b--', label='Numerical', linewidth=2, alpha=0.8)
    ax1.set_xlabel('$x$', fontsize=12)
    ax1.set_ylabel('$u\'\'(x)$', fontsize=12)
    ax1.set_title('Second Derivative: $u(x) = \\sin(2\\pi x)$', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error
    ax2 = axes[1]
    ax2.semilogy(x, error, 'r-', linewidth=2, alpha=0.8)
    ax2.set_xlabel('$x$', fontsize=12)
    ax2.set_ylabel('$|Error|$', fontsize=12)
    ax2.set_title('Pointwise Error', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('padefd2_test.png', dpi=150, bbox_inches='tight')
    print("Plot saved to padefd2_test.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing padefd2 - Compact Padé Second Derivative")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Basic Functionality", test_basic_functionality()))
    results.append(("Sine Function", test_sine_function()))
    results.append(("Convergence Rate", test_convergence_rate()))
    results.append(("Boundary Conditions", test_boundary_conditions()))
    results.append(("Matrix Construction", test_matrix_construction()))
    
    # Visualization
    try:
        visualize_results()
    except Exception as e:
        print(f"\nWarning: Visualization failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result[1] for result in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED. ✗")
    print("=" * 60 + "\n")

