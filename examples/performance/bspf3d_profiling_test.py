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
if _src not in sys.path:
    sys.path.insert(0, _src)

# Import from package
from bspf import bspf3d

# Optional GPU support
_HAS_CUPY = False
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None




def check_correctness(nx, ny, nz, degree, use_gpu=False, use_complex=False):
    """
    Check correctness of differentiate_1_2_batched against exact Taylor-Green vortex derivatives.
    Uses the same approach as examples/basic/diff_gpu_vs_cpu_3d.py.
    """
    print("\n" + "="*80)
    title = "=== Correctness Check (Taylor-Green Vortex"
    if use_complex:
        title += ", Complex Input"
    title += ") ==="
    print(title)
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
    
    # Convert to complex if requested
    if use_complex:
        # Create complex field: F_complex = F_real + i * F_shifted
        F_shifted = to_nz_ny_nx(np.sin(X + np.pi/4.0) * np.sin(Y + np.pi/4.0) * np.sin(Z + np.pi/4.0))
        F = F.astype(np.complex128) + 1j * F_shifted.astype(np.complex128)
        df_dx_shifted = to_nz_ny_nx(np.cos(X + np.pi/4.0) * np.sin(Y + np.pi/4.0) * np.sin(Z + np.pi/4.0))
        df_dy_shifted = to_nz_ny_nx(np.sin(X + np.pi/4.0) * np.cos(Y + np.pi/4.0) * np.sin(Z + np.pi/4.0))
        df_dz_shifted = to_nz_ny_nx(np.sin(X + np.pi/4.0) * np.sin(Y + np.pi/4.0) * np.cos(Z + np.pi/4.0))
        d2f_dx2_shifted = to_nz_ny_nx(-np.sin(X + np.pi/4.0) * np.sin(Y + np.pi/4.0) * np.sin(Z + np.pi/4.0))
        d2f_dy2_shifted = to_nz_ny_nx(-np.sin(X + np.pi/4.0) * np.sin(Y + np.pi/4.0) * np.sin(Z + np.pi/4.0))
        d2f_dz2_shifted = to_nz_ny_nx(-np.sin(X + np.pi/4.0) * np.sin(Y + np.pi/4.0) * np.sin(Z + np.pi/4.0))
        df_dx_exact = df_dx_exact.astype(np.complex128) + 1j * df_dx_shifted.astype(np.complex128)
        df_dy_exact = df_dy_exact.astype(np.complex128) + 1j * df_dy_shifted.astype(np.complex128)
        df_dz_exact = df_dz_exact.astype(np.complex128) + 1j * df_dz_shifted.astype(np.complex128)
        d2f_dx2_exact = d2f_dx2_exact.astype(np.complex128) + 1j * d2f_dx2_shifted.astype(np.complex128)
        d2f_dy2_exact = d2f_dy2_exact.astype(np.complex128) + 1j * d2f_dy2_shifted.astype(np.complex128)
        d2f_dz2_exact = d2f_dz2_exact.astype(np.complex128) + 1j * d2f_dz2_shifted.astype(np.complex128)
    
    # Build operator (convert grids if GPU requested)
    print(f"Building BSPF3D operator: degree={degree}, clustering=True, use_gpu={use_gpu}...")
    if use_gpu:
        if not _HAS_CUPY:
            raise RuntimeError("CuPy is required for GPU check but is not available.")
        x_gpu = cp.asarray(x, dtype=cp.float64)
        y_gpu = cp.asarray(y, dtype=cp.float64)
        z_gpu = cp.asarray(z, dtype=cp.float64)
    else:
        x_gpu, y_gpu, z_gpu = x, y, z

    op = bspf3d.from_grids(
        x=x_gpu,
        y=y_gpu,
        z=z_gpu,
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
    if use_gpu:
        F_gpu = cp.asarray(F, dtype=cp.complex128 if use_complex else cp.float64)
        print("Computing derivatives on GPU...")
        dF_dx, dF_dy, dF_dz, d2F_dx2, d2F_dy2, d2F_dz2 = op.differentiate_1_2_batched(F_gpu)
        # Convert back to NumPy for comparison
        dF_dx = cp.asnumpy(dF_dx)
        dF_dy = cp.asnumpy(dF_dy)
        dF_dz = cp.asnumpy(dF_dz)
        d2F_dx2 = cp.asnumpy(d2F_dx2)
        d2F_dy2 = cp.asnumpy(d2F_dy2)
        d2F_dz2 = cp.asnumpy(d2F_dz2)
    else:
        print("Computing derivatives on CPU...")
        dF_dx, dF_dy, dF_dz, d2F_dx2, d2F_dy2, d2F_dz2 = op.differentiate_1_2_batched(F)
    
    # Compare with exact derivatives
    print("\nComparing with exact derivatives...")
    
    # First derivatives (L2 only)
    err_dx = dF_dx - df_dx_exact
    err_dy = dF_dy - df_dy_exact
    err_dz = dF_dz - df_dz_exact
    
    if np.iscomplexobj(err_dx):
        l2_err_dx = np.sqrt(np.mean(np.abs(err_dx)**2))
        l2_err_dy = np.sqrt(np.mean(np.abs(err_dy)**2))
        l2_err_dz = np.sqrt(np.mean(np.abs(err_dz)**2))
    else:
        l2_err_dx = np.sqrt(np.mean(err_dx**2))
        l2_err_dy = np.sqrt(np.mean(err_dy**2))
        l2_err_dz = np.sqrt(np.mean(err_dz**2))
    
    # Second derivatives (L2 only)
    err_d2x = d2F_dx2 - d2f_dx2_exact
    err_d2y = d2F_dy2 - d2f_dy2_exact
    err_d2z = d2F_dz2 - d2f_dz2_exact
    
    if np.iscomplexobj(err_d2x):
        l2_err_d2x = np.sqrt(np.mean(np.abs(err_d2x)**2))
        l2_err_d2y = np.sqrt(np.mean(np.abs(err_d2y)**2))
        l2_err_d2z = np.sqrt(np.mean(np.abs(err_d2z)**2))
    else:
        l2_err_d2x = np.sqrt(np.mean(err_d2x**2))
        l2_err_d2y = np.sqrt(np.mean(err_d2y**2))
        l2_err_d2z = np.sqrt(np.mean(err_d2z**2))
    
    # Print results
    print(f"\nFirst derivatives (L2 error):")
    print(f"  ∂f/∂x: L2_err={l2_err_dx:.6e}")
    print(f"  ∂f/∂y: L2_err={l2_err_dy:.6e}")
    print(f"  ∂f/∂z: L2_err={l2_err_dz:.6e}")
    
    print(f"\nSecond derivatives (L2 error):")
    print(f"  ∂²f/∂x²: L2_err={l2_err_d2x:.6e}")
    print(f"  ∂²f/∂y²: L2_err={l2_err_d2y:.6e}")
    print(f"  ∂²f/∂z²: L2_err={l2_err_d2z:.6e}")
    
    # Tolerance for correctness check (L2 only)
    tolerance = 1e-5
    if (l2_err_dx < tolerance and l2_err_dy < tolerance and l2_err_dz < tolerance and
        l2_err_d2x < tolerance and l2_err_d2y < tolerance and l2_err_d2z < tolerance):
        print(f"\n✓ All derivatives are accurate (L2 tolerance={tolerance})")
    else:
        print(f"\n⚠ Some derivatives exceed L2 tolerance={tolerance}")
    
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
    
    # Build GPU operator (convert grids to CuPy)
    print("Building GPU operator...")
    x_gpu = cp.asarray(x, dtype=cp.float64)
    y_gpu = cp.asarray(y, dtype=cp.float64)
    z_gpu = cp.asarray(z, dtype=cp.float64)
    F_gpu = cp.asarray(F, dtype=cp.float64)
    op_gpu = bspf3d.from_grids(
        x=x_gpu,
        y=y_gpu,
        z=z_gpu,
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
    _ = op_cpu.differentiate_1_2_batched(F)
    print("Warming up GPU...")
    _ = op_gpu.differentiate_1_2_batched(F_gpu)
    cp.cuda.Stream.null.synchronize()
    
    # Time CPU batched version
    print(f"Timing CPU batched version ({n_runs} runs)...")
    times_cpu = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = op_cpu.differentiate_1_2_batched(F)
        t1 = time.perf_counter()
        times_cpu.append(t1 - t0)
    
    # Time GPU batched version
    print(f"Timing GPU batched version ({n_runs} runs)...")
    times_gpu = []
    for _ in range(n_runs):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        _ = op_gpu.differentiate_1_2_batched(F_gpu)
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
        description="Profile bspf3d.differentiate_1_2_batched using Taylor-Green vortex test data",
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
    p.add_argument("--runs", type=int, default=100, help="Number of timing runs")
    p.add_argument("--gpu", action="store_true", help="Use GPU (CuPy) if available")
    p.add_argument("--complex", action="store_true", help="Use complex array input")
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
            use_complex=args.complex,
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


if __name__ == "__main__":
    main()

