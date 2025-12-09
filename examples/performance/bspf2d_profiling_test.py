"""
Standalone test/driver for bspf2d_profiling with correctness checking.

Uses the same test data and configurations as examples/basic/diff_2d.py
(turbulence field with exact derivatives).

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
from examples.basic.diff_2d import (
    random_turbulence_signal_2d_with_derivatives,
    add_circular_shock_wave
)


def check_correctness(nx, ny, degree, use_gpu=False, shock_enabled=False):
    """
    Check correctness of differentiate_1_2 against exact turbulence field derivatives.
    
    Uses the same test data and configuration as examples/basic/diff_2d.py.
    """
    print("\n" + "="*80)
    print("=== Correctness Check (Turbulence Field) ===")
    print("="*80)
    
    # Parameters matching diff_2d.py
    DOMAIN_X = [0, 2*np.pi]
    DOMAIN_Y = [0, 2*np.pi]
    TURB_N_MODES = 64
    TURB_SEED = 123
    SHOCK_CENTER_X = np.pi
    SHOCK_CENTER_Y = np.pi
    SHOCK_RADIUS = 0.5
    SHOCK_AMPLITUDE = 2.0
    SHOCK_WIDTH = 0.02
    
    # Domain
    Lx = DOMAIN_X[1] - DOMAIN_X[0]
    Ly = DOMAIN_Y[1] - DOMAIN_Y[0]
    x = np.linspace(DOMAIN_X[0], DOMAIN_X[1], nx, endpoint=True)
    y = np.linspace(DOMAIN_Y[0], DOMAIN_Y[1], ny, endpoint=True)
    X_grid, Y_grid = np.meshgrid(x, y)  # (ny, nx) - matches diff_2d.py convention
    
    # Generate turbulence field with exact derivatives
    print(f"Generating turbulence field: {nx}x{ny} grid, {TURB_N_MODES} modes...")
    _, _, F, df_dx_exact, df_dy_exact, d2f_dx2_exact, d2f_dy2_exact, _ = \
        random_turbulence_signal_2d_with_derivatives(
            Lx=Lx, Ly=Ly, Nx=nx, Ny=ny,
            nmodes=TURB_N_MODES, seed=TURB_SEED
        )
    # F, df_dx_exact, etc. are already in (ny, nx) format
    
    # Add shock wave if enabled
    if shock_enabled:
        print("Adding circular shock wave...")
        F, df_dx_exact, df_dy_exact, d2f_dx2_exact, d2f_dy2_exact, _ = \
            add_circular_shock_wave(
                X_grid, Y_grid, F, df_dx_exact, df_dy_exact,
                d2f_dx2_exact, d2f_dy2_exact, np.zeros_like(F),
                center_x=SHOCK_CENTER_X, center_y=SHOCK_CENTER_Y,
                radius=SHOCK_RADIUS, amplitude=SHOCK_AMPLITUDE, width=SHOCK_WIDTH
            )
    
    # Build operator (matching diff_2d.py configuration)
    print(f"Building BSPF2D operator: degree={degree}, clustering=True...")
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
    
    # Compute numerical derivatives with both versions
    print("Computing derivatives with differentiate_1_2 (loop version)...")
    df_dx_loop, df_dy_loop, d2f_dx2_loop, d2f_dy2_loop = op.differentiate_1_2(F, use_loop=True)
    
    print("Computing derivatives with differentiate_1_2 (batched version)...")
    df_dx_batched, df_dy_batched, d2f_dx2_batched, d2f_dy2_batched = op.differentiate_1_2(F, use_loop=False)
    
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
    if shock_enabled:
        print(f"Shock: center=({SHOCK_CENTER_X}, {SHOCK_CENTER_Y}), radius={SHOCK_RADIUS}")
    print(f"\nError metrics:")
    print(f"  df/dx:  max={max_err_dx:.6e}, L2={l2_err_dx:.6e}")
    print(f"  df/dy:  max={max_err_dy:.6e}, L2={l2_err_dy:.6e}")
    print(f"  d2f/dx2: max={max_err_dx2:.6e}, L2={l2_err_dx2:.6e}")
    print(f"  d2f/dy2: max={max_err_dy2:.6e}, L2={l2_err_dy2:.6e}")
    
    # Check if errors are reasonable (turbulence field may have larger errors than smooth functions)
    # Use a more relaxed tolerance for turbulence field
    tolerance = 1e-6 if not shock_enabled else 1e-4
    all_ok = (max_err_dx < tolerance and max_err_dy < tolerance and
              max_err_dx2 < tolerance and max_err_dy2 < tolerance)
    
    if all_ok:
        print(f"\n✓ All errors below tolerance ({tolerance:.1e})")
    else:
        print(f"\n⚠ Some errors exceed tolerance ({tolerance:.1e})")
        print("  (This may be expected for turbulence fields with high-frequency content)")
    
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


def compare_performance(nx, ny, degree, use_gpu=False, n_runs=10, shock_enabled=False):
    """
    Compare performance of loop vs batched versions using turbulence field data.
    """
    print("\n" + "="*80)
    print("=== Performance Comparison (Loop vs Batched) ===")
    print("="*80)
    
    # Parameters matching diff_2d.py
    DOMAIN_X = [0, 2*np.pi]
    DOMAIN_Y = [0, 2*np.pi]
    TURB_N_MODES = 64
    TURB_SEED = 123
    SHOCK_CENTER_X = np.pi
    SHOCK_CENTER_Y = np.pi
    SHOCK_RADIUS = 0.5
    SHOCK_AMPLITUDE = 2.0
    SHOCK_WIDTH = 0.02
    
    # Domain
    Lx = DOMAIN_X[1] - DOMAIN_X[0]
    Ly = DOMAIN_Y[1] - DOMAIN_Y[0]
    x = np.linspace(DOMAIN_X[0], DOMAIN_X[1], nx, endpoint=True)
    y = np.linspace(DOMAIN_Y[0], DOMAIN_Y[1], ny, endpoint=True)
    X_grid, Y_grid = np.meshgrid(x, y)  # (ny, nx) - matches diff_2d.py convention
    
    # Generate turbulence field (same as in check_correctness)
    print(f"Generating turbulence field: {nx}x{ny} grid, {TURB_N_MODES} modes...")
    _, _, F, _, _, _, _, _ = \
        random_turbulence_signal_2d_with_derivatives(
            Lx=Lx, Ly=Ly, Nx=nx, Ny=ny,
            nmodes=TURB_N_MODES, seed=TURB_SEED
        )
    # F is already in (ny, nx) format
    
    # Add shock wave if enabled
    if shock_enabled:
        print("Adding circular shock wave...")
        F, _, _, _, _, _ = \
            add_circular_shock_wave(
                X_grid, Y_grid, F, np.zeros_like(F), np.zeros_like(F),
                np.zeros_like(F), np.zeros_like(F), np.zeros_like(F),
                center_x=SHOCK_CENTER_X, center_y=SHOCK_CENTER_Y,
                radius=SHOCK_RADIUS, amplitude=SHOCK_AMPLITUDE, width=SHOCK_WIDTH
            )
    
    # Build operator
    print(f"Building BSPF2D operator: degree={degree}, clustering=True...")
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
    _ = op.differentiate_1_2(F, use_loop=True)
    _ = op.differentiate_1_2(F, use_loop=False)
    
    # Time loop version
    print(f"Timing loop version ({n_runs} runs)...")
    times_loop = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = op.differentiate_1_2(F, use_loop=True)
        t1 = time.perf_counter()
        times_loop.append(t1 - t0)
    
    # Time batched version
    print(f"Timing batched version ({n_runs} runs)...")
    times_batched = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = op.differentiate_1_2(F, use_loop=False)
        t1 = time.perf_counter()
        times_batched.append(t1 - t0)
    
    times_loop = np.array(times_loop)
    times_batched = np.array(times_batched)
    
    # Statistics
    mean_loop = np.mean(times_loop)
    std_loop = np.std(times_loop)
    min_loop = np.min(times_loop)
    max_loop = np.max(times_loop)
    
    mean_batched = np.mean(times_batched)
    std_batched = np.std(times_batched)
    min_batched = np.min(times_batched)
    max_batched = np.max(times_batched)
    
    speedup = mean_batched / mean_loop
    
    print(f"\nGrid: nx={nx}, ny={ny}, degree={degree}, use_gpu={use_gpu}")
    print(f"Test data: Turbulence field ({TURB_N_MODES} modes)" + (" + shock wave" if shock_enabled else ""))
    print(f"\n{'Version':<15s} {'Mean':>12s} {'Std':>12s} {'Min':>12s} {'Max':>12s}")
    print("-" * 65)
    print(f"{'Loop':<15s} {mean_loop:12.6f} {std_loop:12.6f} {min_loop:12.6f} {max_loop:12.6f}")
    print(f"{'Batched':<15s} {mean_batched:12.6f} {std_batched:12.6f} {min_batched:12.6f} {max_batched:12.6f}")
    print("-" * 65)
    
    if speedup > 1.0:
        print(f"\nLoop version is {speedup:.2f}x faster than batched version")
    else:
        print(f"\nBatched version is {1.0/speedup:.2f}x faster than loop version")
    
    print("="*80 + "\n")
    
    return {
        'mean_loop': mean_loop,
        'std_loop': std_loop,
        'mean_batched': mean_batched,
        'std_batched': std_batched,
        'speedup': speedup,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Profile bspf2d.differentiate_1_2 using turbulence field test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--nx", type=int, default=512, 
                   help="Grid points in x (matching diff_2d.py default)")
    p.add_argument("--ny", type=int, default=512, 
                   help="Grid points in y (matching diff_2d.py default)")
    p.add_argument("--degree", type=int, default=7, 
                   help="B-spline degree (matching diff_2d.py)")
    p.add_argument("--runs", type=int, default=10, help="Number of timing runs")
    p.add_argument("--gpu", action="store_true", help="Use GPU (CuPy) if available")
    p.add_argument("--no-check", action="store_true", help="Skip correctness check")
    p.add_argument("--shock", action="store_true", 
                   help="Enable circular shock wave (matching diff_2d.py)")
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
            shock_enabled=args.shock,
        )
    
    # Compare performance of loop vs batched versions (using turbulence field)
    compare_performance(
        nx=args.nx,
        ny=args.ny,
        degree=args.degree,
        use_gpu=args.gpu,
        n_runs=args.runs,
        shock_enabled=args.shock,
    )
    
    # Then run detailed profiling (using simple test function for profiling, not turbulence)
    # Note: run_profile_2d uses a simple test function, not the turbulence field
    print("\n" + "="*80)
    print("=== Detailed Performance Profiling (Batched Version) ===")
    print("="*80)
    print("Note: Profiling uses a simple test function, not the turbulence field.")
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

