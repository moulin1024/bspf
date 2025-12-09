"""
Standalone test/driver for bspf3d_profiling with correctness checking.

Uses Taylor-Green vortex as the test function with exact derivatives.
Matches the implementation from examples/basic/diff_gpu_vs_cpu_3d.py.

Run:
    python examples/performance/bspf3d_profiling_test.py
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
from examples.performance.bspf3d_profiling import run_profile_3d, bspf3d

# Optional GPU support
_HAS_CUPY = False
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None




def check_correctness(nx, ny, nz, degree, use_gpu=False):
    """
    Check correctness of differentiate_1_2 against exact Taylor-Green vortex derivatives.
    Uses the same approach as examples/basic/diff_gpu_vs_cpu_3d.py.
    """
    print("\n" + "="*80)
    print("=== Correctness Check (Taylor-Green Vortex) ===")
    print("="*80)
    
    # Domain - matches diff_gpu_vs_cpu_3d.py
    DOMAIN = [0, 2*np.pi]
    x = np.linspace(DOMAIN[0], DOMAIN[1], nx, endpoint=True)
    y = np.linspace(DOMAIN[0], DOMAIN[1], ny, endpoint=True)
    z = np.linspace(DOMAIN[0], DOMAIN[1], nz, endpoint=True)
    
    # Create meshgrid using "xy" indexing like diff_gpu_vs_cpu_3d.py, then reshape to (nz, ny, nx)
    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")  # (ny, nx, nz)
    to_nz_ny_nx = lambda A: np.moveaxis(A, 2, 0)  # (ny, nx, nz) -> (nz, ny, nx)
    
    # Generate Taylor-Green vortex field: f(x,y,z) = sin(x) * sin(y) * sin(z)
    # This matches the scalar field test used in diff_gpu_vs_cpu_3d.py style
    print(f"Generating Taylor-Green vortex field: {nx}x{ny}x{nz} grid...")
    F = to_nz_ny_nx(np.sin(X) * np.sin(Y) * np.sin(Z))
    
    # Compute exact derivatives directly on the final (nz, ny, nx) array
    # This ensures the coordinate system matches exactly how bspf3d interprets the axes
    # In (nz, ny, nx): x varies along axis 2, y along axis 1, z along axis 0
    X_final = to_nz_ny_nx(X)  # (nz, ny, nx) - x coordinate varies along axis 2
    Y_final = to_nz_ny_nx(Y)  # (nz, ny, nx) - y coordinate varies along axis 1
    Z_final = to_nz_ny_nx(Z)  # (nz, ny, nx) - z coordinate varies along axis 0
    
    # Compute exact derivatives using the final coordinate arrays
    df_dx_exact = np.cos(X_final) * np.sin(Y_final) * np.sin(Z_final)  # derivative w.r.t. x (axis 2)
    df_dy_exact = np.sin(X_final) * np.cos(Y_final) * np.sin(Z_final)  # derivative w.r.t. y (axis 1)
    df_dz_exact = np.sin(X_final) * np.sin(Y_final) * np.cos(Z_final)  # derivative w.r.t. z (axis 0)
    
    d2f_dx2_exact = -np.sin(X_final) * np.sin(Y_final) * np.sin(Z_final)
    d2f_dy2_exact = -np.sin(X_final) * np.sin(Y_final) * np.sin(Z_final)
    d2f_dz2_exact = -np.sin(X_final) * np.sin(Y_final) * np.sin(Z_final)
    
    # Build operator
    print(f"Building BSPF3D operator: degree={degree}, clustering=True, use_gpu={use_gpu}...")
    op = bspf3d.from_grids(
        x=x,
        y=y,
        z=z,
        degree_x=degree,
        degree_y=degree,
        degree_z=degree,
        n_basis_x=4 * degree,
        n_basis_y=4 * degree,
        n_basis_z=4 * degree,
        use_clustering_x=True,
        use_clustering_y=True,
        use_clustering_z=True,
        correction="spectral",
        use_gpu=use_gpu,
    )
    
    # Convert to GPU array if needed
    if use_gpu and _HAS_CUPY:
        F_gpu = cp.asarray(F, dtype=cp.float64)
        print("Computing derivatives on GPU...")
        dF_dx, dF_dy, dF_dz, d2F_dx2, d2F_dy2, d2F_dz2 = op.differentiate_1_2(F_gpu, use_loop=False)
        # Convert back to NumPy for comparison
        dF_dx = cp.asnumpy(dF_dx)
        dF_dy = cp.asnumpy(dF_dy)
        dF_dz = cp.asnumpy(dF_dz)
        d2F_dx2 = cp.asnumpy(d2F_dx2)
        d2F_dy2 = cp.asnumpy(d2F_dy2)
        d2F_dz2 = cp.asnumpy(d2F_dz2)
    else:
        print("Computing derivatives on CPU...")
        dF_dx, dF_dy, dF_dz, d2F_dx2, d2F_dy2, d2F_dz2 = op.differentiate_1_2(F, use_loop=False)
    
    # Compare with exact derivatives
    print("\nComparing with exact derivatives...")
    
    # First derivatives
    err_dx = np.abs(dF_dx - df_dx_exact)
    err_dy = np.abs(dF_dy - df_dy_exact)
    err_dz = np.abs(dF_dz - df_dz_exact)
    
    max_err_dx = np.max(err_dx)
    max_err_dy = np.max(err_dy)
    max_err_dz = np.max(err_dz)
    
    l2_err_dx = np.sqrt(np.mean(err_dx**2))
    l2_err_dy = np.sqrt(np.mean(err_dy**2))
    l2_err_dz = np.sqrt(np.mean(err_dz**2))
    
    # Second derivatives
    err_d2x = np.abs(d2F_dx2 - d2f_dx2_exact)
    err_d2y = np.abs(d2F_dy2 - d2f_dy2_exact)
    err_d2z = np.abs(d2F_dz2 - d2f_dz2_exact)
    
    max_err_d2x = np.max(err_d2x)
    max_err_d2y = np.max(err_d2y)
    max_err_d2z = np.max(err_d2z)
    
    l2_err_d2x = np.sqrt(np.mean(err_d2x**2))
    l2_err_d2y = np.sqrt(np.mean(err_d2y**2))
    l2_err_d2z = np.sqrt(np.mean(err_d2z**2))
    
    # Print results
    print(f"\nFirst derivatives:")
    print(f"  ∂f/∂x: max_err={max_err_dx:.6e}, L2_err={l2_err_dx:.6e}")
    print(f"  ∂f/∂y: max_err={max_err_dy:.6e}, L2_err={l2_err_dy:.6e}")
    print(f"  ∂f/∂z: max_err={max_err_dz:.6e}, L2_err={l2_err_dz:.6e}")
    
    # Diagnostic: check where maximum errors occur
    if max_err_dy > 0.1 or max_err_dz > 0.1:
        print(f"\nDiagnostic (checking where maximum errors occur):")
        # Find where max error occurs
        max_idx_dy = np.unravel_index(np.argmax(err_dy), err_dy.shape)
        max_idx_dz = np.unravel_index(np.argmax(err_dz), err_dz.shape)
        print(f"  Max error df/dy at index {max_idx_dy} (z={z[max_idx_dy[0]]:.3f}, y={y[max_idx_dy[1]]:.3f}, x={x[max_idx_dy[2]]:.3f}):")
        print(f"    numerical = {dF_dy[max_idx_dy]:.6f}, exact = {df_dy_exact[max_idx_dy]:.6f}, error = {err_dy[max_idx_dy]:.6e}")
        print(f"  Max error df/dz at index {max_idx_dz} (z={z[max_idx_dz[0]]:.3f}, y={y[max_idx_dz[1]]:.3f}, x={x[max_idx_dz[2]]:.3f}):")
        print(f"    numerical = {dF_dz[max_idx_dz]:.6f}, exact = {df_dz_exact[max_idx_dz]:.6f}, error = {err_dz[max_idx_dz]:.6e}")
        # Check if maybe y and z are swapped
        err_dy_if_swapped = np.abs(dF_dy - df_dz_exact)
        err_dz_if_swapped = np.abs(dF_dz - df_dy_exact)
        max_err_dy_swapped = np.max(err_dy_if_swapped)
        max_err_dz_swapped = np.max(err_dz_if_swapped)
        print(f"  If y and z derivatives are swapped: max_err(df/dy)={max_err_dy_swapped:.6e}, max_err(df/dz)={max_err_dz_swapped:.6e}")
        if max_err_dy_swapped < max_err_dy and max_err_dz_swapped < max_err_dz:
            print(f"  -> y and z derivatives appear to be swapped!")
    
    print(f"\nSecond derivatives:")
    print(f"  ∂²f/∂x²: max_err={max_err_d2x:.6e}, L2_err={l2_err_d2x:.6e}")
    print(f"  ∂²f/∂y²: max_err={max_err_d2y:.6e}, L2_err={l2_err_d2y:.6e}")
    print(f"  ∂²f/∂z²: max_err={max_err_d2z:.6e}, L2_err={l2_err_d2z:.6e}")
    
    # Check consistency between loop and batched versions (CPU only)
    if not use_gpu:
        print("\nChecking consistency between loop and batched versions...")
        dF_dx_loop, dF_dy_loop, dF_dz_loop, d2F_dx2_loop, d2F_dy2_loop, d2F_dz2_loop = \
            op.differentiate_1_2(F, use_loop=True)
        
        consistency_tol = 1e-10
        diff_dx = np.abs(dF_dx - dF_dx_loop)
        diff_dy = np.abs(dF_dy - dF_dy_loop)
        diff_dz = np.abs(dF_dz - dF_dz_loop)
        diff_d2x = np.abs(d2F_dx2 - d2F_dx2_loop)
        diff_d2y = np.abs(d2F_dy2 - d2F_dy2_loop)
        diff_d2z = np.abs(d2F_dz2 - d2F_dz2_loop)
        
        max_diff_dx = np.max(diff_dx)
        max_diff_dy = np.max(diff_dy)
        max_diff_dz = np.max(diff_dz)
        max_diff_d2x = np.max(diff_d2x)
        max_diff_d2y = np.max(diff_d2y)
        max_diff_d2z = np.max(diff_d2z)
        
        print(f"  Loop vs Batched (first derivatives):")
        print(f"    max_diff(∂f/∂x)={max_diff_dx:.6e}")
        print(f"    max_diff(∂f/∂y)={max_diff_dy:.6e}")
        print(f"    max_diff(∂f/∂z)={max_diff_dz:.6e}")
        print(f"  Loop vs Batched (second derivatives):")
        print(f"    max_diff(∂²f/∂x²)={max_diff_d2x:.6e}")
        print(f"    max_diff(∂²f/∂y²)={max_diff_d2y:.6e}")
        print(f"    max_diff(∂²f/∂z²)={max_diff_d2z:.6e}")
        
        if (max_diff_dx < consistency_tol and max_diff_dy < consistency_tol and 
            max_diff_dz < consistency_tol and max_diff_d2x < consistency_tol and 
            max_diff_d2y < consistency_tol and max_diff_d2z < consistency_tol):
            print("  ✓ Loop and batched versions are consistent")
        else:
            print("  ⚠ Loop and batched versions differ (may indicate numerical issues)")
    
    # Tolerance for correctness check
    tolerance = 1e-6
    if (max_err_dx < tolerance and max_err_dy < tolerance and max_err_dz < tolerance and
        max_err_d2x < tolerance and max_err_d2y < tolerance and max_err_d2z < tolerance):
        print(f"\n✓ All derivatives are accurate (tolerance={tolerance})")
    else:
        print(f"\n⚠ Some derivatives exceed tolerance={tolerance}")
    
    print("="*80 + "\n")


def compare_performance(nx, ny, nz, degree, use_gpu=False, n_runs=10):
    """
    Compare performance of loop vs batched versions.
    """
    print("\n" + "="*80)
    print("=== Performance Comparison (Loop vs Batched) ===")
    print("="*80)
    
    # Setup
    a, b = 0.0, 2.0 * np.pi
    x = np.linspace(a, b, nx, endpoint=True)
    y = np.linspace(a, b, ny, endpoint=True)
    z = np.linspace(a, b, nz, endpoint=True)
    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")  # (ny, nx, nz) - matches working version
    F = np.sin(X) * np.sin(Y) * np.sin(Z)
    to_nz_ny_nx = lambda A: np.moveaxis(A, 2, 0)  # (ny, nx, nz) -> (nz, ny, nx)
    F = to_nz_ny_nx(F)
    
    # Build operator
    op = bspf3d.from_grids(
        x=x,
        y=y,
        z=z,
        degree_x=degree,
        degree_y=degree,
        degree_z=degree,
        n_basis_x=4 * degree,
        n_basis_y=4 * degree,
        n_basis_z=4 * degree,
        use_clustering_x=True,
        use_clustering_y=True,
        use_clustering_z=True,
        correction="spectral",
        use_gpu=use_gpu,
    )
    
    # Convert to GPU array if needed
    if use_gpu and _HAS_CUPY:
        F_gpu = cp.asarray(F, dtype=cp.float64)
    
    # Warmup
    _ = op.differentiate_1_2(F_gpu if (use_gpu and _HAS_CUPY) else F, use_loop=False)
    if use_gpu and _HAS_CUPY:
        cp.cuda.Stream.null.synchronize()
    
    # Time loop version (CPU only)
    times_loop = []
    if not use_gpu:
        print(f"Timing loop version ({n_runs} runs)...")
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = op.differentiate_1_2(F, use_loop=True)
            t1 = time.perf_counter()
            times_loop.append(t1 - t0)
    
    # Time batched version
    print(f"Timing batched version ({n_runs} runs)...")
    times_batched = []
    for _ in range(n_runs):
        if use_gpu and _HAS_CUPY:
            cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        _ = op.differentiate_1_2(F_gpu if (use_gpu and _HAS_CUPY) else F, use_loop=False)
        if use_gpu and _HAS_CUPY:
            cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        times_batched.append(t1 - t0)
    
    times_batched = np.array(times_batched)
    
    # Statistics
    mean_batched = np.mean(times_batched)
    std_batched = np.std(times_batched)
    min_batched = np.min(times_batched)
    max_batched = np.max(times_batched)
    
    print(f"\nGrid: nx={nx}, ny={ny}, nz={nz}, degree={degree}, use_gpu={use_gpu}")
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


def compare_gpu_cpu(nx, ny, nz, degree, n_runs=10):
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
    
    # Setup
    a, b = 0.0, 2.0 * np.pi
    x = np.linspace(a, b, nx, endpoint=True)
    y = np.linspace(a, b, ny, endpoint=True)
    z = np.linspace(a, b, nz, endpoint=True)
    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")  # (ny, nx, nz) - matches working version
    F = np.sin(X) * np.sin(Y) * np.sin(Z)
    to_nz_ny_nx = lambda A: np.moveaxis(A, 2, 0)  # (ny, nx, nz) -> (nz, ny, nx)
    F = to_nz_ny_nx(F)
    
    # Build CPU operator
    print("Building CPU operator...")
    op_cpu = bspf3d.from_grids(
        x=x,
        y=y,
        z=z,
        degree_x=degree,
        degree_y=degree,
        degree_z=degree,
        n_basis_x=4 * degree,
        n_basis_y=4 * degree,
        n_basis_z=4 * degree,
        use_clustering_x=True,
        use_clustering_y=True,
        use_clustering_z=True,
        correction="spectral",
        use_gpu=False,
    )
    
    # Build GPU operator
    print("Building GPU operator...")
    F_gpu = cp.asarray(F, dtype=cp.float64)
    op_gpu = bspf3d.from_grids(
        x=x,
        y=y,
        z=z,
        degree_x=degree,
        degree_y=degree,
        degree_z=degree,
        n_basis_x=4 * degree,
        n_basis_y=4 * degree,
        n_basis_z=4 * degree,
        use_clustering_x=True,
        use_clustering_y=True,
        use_clustering_z=True,
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
    
    print(f"\nGrid: nx={nx}, ny={ny}, nz={nz}, degree={degree}")
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
        description="Profile bspf3d.differentiate_1_2 using Taylor-Green vortex test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--nx", type=int, default=64, 
                   help="Grid points in x")
    p.add_argument("--ny", type=int, default=64, 
                   help="Grid points in y")
    p.add_argument("--nz", type=int, default=64, 
                   help="Grid points in z")
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
    
    # Check correctness first
    if not args.no_check:
        check_correctness(
            nx=args.nx,
            ny=args.ny,
            nz=args.nz,
            degree=args.degree,
            use_gpu=args.gpu,
        )
    
    # Compare performance of loop vs batched versions
    compare_performance(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        degree=args.degree,
        use_gpu=args.gpu,
        n_runs=args.runs,
    )
    
    # Compare GPU vs CPU for batched version (if CuPy is available and not skipped)
    if not args.no_gpu_cpu:
        compare_gpu_cpu(
            nx=args.nx,
            ny=args.ny,
            nz=args.nz,
            degree=args.degree,
            n_runs=args.runs,
        )
    
    # Then run detailed profiling
    print("\n" + "="*80)
    print("=== Detailed Performance Profiling (Batched Version) ===")
    print("="*80)
    print("Note: Profiling uses Taylor-Green vortex test function.")
    print("="*80 + "\n")
    run_profile_3d(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        degree=args.degree,
        n_runs=args.runs,
        use_gpu=args.gpu,
    )


if __name__ == "__main__":
    main()

