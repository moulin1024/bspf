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
import matplotlib.pyplot as plt

# Ensure local src is preferred over installed package
_here = os.path.abspath(os.path.dirname(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


# Import after adjusting sys.path
from bspf2d import bspf2d

# Optional GPU support
_HAS_CUPY = False
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None


def check_correctness(nx, ny, degree, use_gpu=False, use_complex=False):
    """
    Check correctness of differentiate_1_2 against a synthetic 2D turbulence-like field.

    The test field is constructed as a superposition of a few Fourier modes:
        f(x, y) = Σ a_{mn} sin(m x) sin(n y)
    with fixed coefficients and wavenumbers (m, n), so that all derivatives
    have closed-form expressions obtained mode-by-mode.
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
    
    # Synthetic 2D turbulence-like scalar field: superposition of Fourier modes
    print(f"Generating synthetic 2D turbulence-like field: {nx}x{ny} grid...")
    
    # Randomly generate modes with random phase shifts
    # Set seed for reproducibility (can be changed)
    rng = np.random.RandomState(42)
    n_modes = 100  # Number of modes to generate
    max_kx = 20
    max_ky = 20
    # Generate random modes: (amplitude, kx, ky, phi_x, phi_y)
    # Amplitude: random between 0.2 and 1.0
    # Wavenumbers: random integers between 1 and 6
    # Phase shifts: random between 0 and 2*pi
    modes = []
    for _ in range(n_modes):
        amp = rng.uniform(0.2, 1.0)
        kx = rng.randint(1, max_kx + 1)
        ky = rng.randint(1, max_ky + 1)
        phi_x = rng.uniform(0, 2 * np.pi)
        phi_y = rng.uniform(0, 2 * np.pi)
        modes.append((amp, kx, ky, phi_x, phi_y))
    
    print(f"Generated {n_modes} random modes with phase shifts")
    for i, (amp, kx, ky, phi_x, phi_y) in enumerate(modes):
        print(f"  Mode {i+1}: amp={amp:.3f}, kx={kx}, ky={ky}, phi_x={phi_x:.3f}, phi_y={phi_y:.3f}")

    F_real = np.zeros_like(X)
    df_dx_real = np.zeros_like(X)
    df_dy_real = np.zeros_like(X)
    d2f_dx2_real = np.zeros_like(X)
    d2f_dy2_real = np.zeros_like(X)

    for amp, kx, ky, phi_x, phi_y in modes:
        # Mode with phase shifts: sin(kx*X + phi_x) * sin(ky*Y + phi_y)
        arg_x = kx * X + phi_x
        arg_y = ky * Y + phi_y
        s = np.sin(arg_x) * np.sin(arg_y)
        F_real += amp * s
        df_dx_real += amp * kx * np.cos(arg_x) * np.sin(arg_y)
        df_dy_real += amp * ky * np.sin(arg_x) * np.cos(arg_y)
        d2f_dx2_real += -amp * (kx ** 2) * np.sin(arg_x) * np.sin(arg_y)
        d2f_dy2_real += -amp * (ky ** 2) * np.sin(arg_x) * np.sin(arg_y)

    if use_complex:
        # For complex case, use different random phase shifts for imaginary part
        rng_imag = np.random.RandomState(123)  # Different seed for imaginary part
        F_imag = np.zeros_like(X)
        df_dx_imag = np.zeros_like(X)
        df_dy_imag = np.zeros_like(X)
        d2f_dx2_imag = np.zeros_like(X)
        d2f_dy2_imag = np.zeros_like(X)

        for amp, kx, ky, phi_x, phi_y in modes:
            # Use different phase shifts for imaginary part
            phi_x_im = rng_imag.uniform(0, 2 * np.pi)
            phi_y_im = rng_imag.uniform(0, 2 * np.pi)
            arg_x_im = kx * X + phi_x_im
            arg_y_im = ky * Y + phi_y_im
            s_im = np.sin(arg_x_im) * np.sin(arg_y_im)
            F_imag += amp * s_im
            df_dx_imag += amp * kx * np.cos(arg_x_im) * np.sin(arg_y_im)
            df_dy_imag += amp * ky * np.sin(arg_x_im) * np.cos(arg_y_im)
            d2f_dx2_imag += -amp * (kx ** 2) * np.sin(arg_x_im) * np.sin(arg_y_im)
            d2f_dy2_imag += -amp * (ky ** 2) * np.sin(arg_x_im) * np.sin(arg_y_im)

        F = F_real + 1j * F_imag
    else:
        F = F_real
    
    if use_complex:
        df_dx_exact = df_dx_real + 1j * df_dx_imag
        df_dy_exact = df_dy_real + 1j * df_dy_imag
        d2f_dx2_exact = d2f_dx2_real + 1j * d2f_dx2_imag
        d2f_dy2_exact = d2f_dy2_real + 1j * d2f_dy2_imag
    else:
        df_dx_exact = df_dx_real
        df_dy_exact = df_dy_real
        d2f_dx2_exact = d2f_dx2_real
        d2f_dy2_exact = d2f_dy2_real
    
    # Plot the synthetic flow field
    F_plot = F_real if not use_complex else np.abs(F)
    if use_gpu and _HAS_CUPY and isinstance(F_plot, cp.ndarray):
        F_plot = cp.asnumpy(F_plot)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, F_plot, levels=20, cmap='RdBu_r')
    plt.colorbar(label='Field value')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Synthetic 2D Turbulence-like Flow Field', fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('figs/turbulence_field.png', dpi=150, bbox_inches='tight')
    print(f"Saved contour plot to figs/turbulence_field.png")
    plt.close()
    
    # Build operator (matching diff_2d.py configuration)
    print(f"Building BSPF2D operator: degree={degree}, clustering=True, use_gpu={use_gpu}, complex={use_complex}...")
    if use_gpu and _HAS_CUPY:
        x_backend = cp.asarray(x, dtype=cp.float64)
        y_backend = cp.asarray(y, dtype=cp.float64)
    else:
        x_backend, y_backend = x, y
    op = bspf2d.from_grids(
        x=x_backend,
        y=y_backend,
        degree_x=degree,
        degree_y=degree,
        use_clustering_x=True,
        use_clustering_y=True,
        correction="spectral",
        use_gpu=use_gpu,
    )
    
    # Convert F to GPU array if needed
    if use_gpu and _HAS_CUPY:
        dtype = cp.complex128 if use_complex else cp.float64
        F_gpu = cp.asarray(F, dtype=dtype)
    else:
        F_gpu = F
    
    # Compute numerical derivatives (with timing)
    print("Computing derivatives with differentiate_1_2...")
    if use_gpu and _HAS_CUPY:
        cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    df_dx_num, df_dy_num, d2f_dx2_num, d2f_dy2_num = op.differentiate_1_2(
        F_gpu
    )
    if use_gpu and _HAS_CUPY:
        cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    diff12_time = t1 - t0
    
    # Convert back to NumPy for comparison if on GPU
    if use_gpu and _HAS_CUPY:
        df_dx_num = cp.asnumpy(df_dx_num) if isinstance(df_dx_num, cp.ndarray) else df_dx_num
        df_dy_num = cp.asnumpy(df_dy_num) if isinstance(df_dy_num, cp.ndarray) else df_dy_num
        d2f_dx2_num = cp.asnumpy(d2f_dx2_num) if isinstance(d2f_dx2_num, cp.ndarray) else d2f_dx2_num
        d2f_dy2_num = cp.asnumpy(d2f_dy2_num) if isinstance(d2f_dy2_num, cp.ndarray) else d2f_dy2_num
    
    # Compute errors (all arrays are already in (ny, nx) format)
    err_dx = df_dx_num - df_dx_exact
    err_dy = df_dy_num - df_dy_exact
    err_dx2 = d2f_dx2_num - d2f_dx2_exact
    err_dy2 = d2f_dy2_num - d2f_dy2_exact
    
    abs_fn = np.abs if not use_gpu else np.abs  # both NumPy/CuPy
    # L2 errors
    l2_err_dx = np.sqrt(np.mean(abs_fn(err_dx)**2))
    l2_err_dy = np.sqrt(np.mean(abs_fn(err_dy)**2))
    l2_err_dx2 = np.sqrt(np.mean(abs_fn(err_dx2)**2))
    l2_err_dy2 = np.sqrt(np.mean(abs_fn(err_dy2)**2))
    # L_infty errors (absolute)
    linf_err_dx = float(np.max(abs_fn(err_dx)))
    linf_err_dy = float(np.max(abs_fn(err_dy)))
    linf_err_dx2 = float(np.max(abs_fn(err_dx2)))
    linf_err_dy2 = float(np.max(abs_fn(err_dy2)))
    # Relative L_infty errors: max|e| / max|exact|
    max_dx = float(np.max(abs_fn(df_dx_exact)))
    max_dy = float(np.max(abs_fn(df_dy_exact)))
    max_dx2 = float(np.max(abs_fn(d2f_dx2_exact)))
    max_dy2 = float(np.max(abs_fn(d2f_dy2_exact)))
    rel_linf_dx = linf_err_dx / max_dx if max_dx > 0 else 0.0
    rel_linf_dy = linf_err_dy / max_dy if max_dy > 0 else 0.0
    rel_linf_dx2 = linf_err_dx2 / max_dx2 if max_dx2 > 0 else 0.0
    rel_linf_dy2 = linf_err_dy2 / max_dy2 if max_dy2 > 0 else 0.0
    
    # Create 4x2 grid plot: original field (2x2), derivatives (1x2), errors (1x2)
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    
    # Convert to NumPy for plotting if needed
    def to_numpy(arr):
        if use_gpu and _HAS_CUPY and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return arr if isinstance(arr, np.ndarray) else np.asarray(arr)
    
    F_plot_real = to_numpy(F_real)
    df_dx_num_plot = to_numpy(df_dx_num)
    df_dy_num_plot = to_numpy(df_dy_num)
    d2f_dx2_num_plot = to_numpy(d2f_dx2_num)
    d2f_dy2_num_plot = to_numpy(d2f_dy2_num)
    err_dx_plot = to_numpy(err_dx)
    err_dy_plot = to_numpy(err_dy)
    err_dx2_plot = to_numpy(err_dx2)
    err_dy2_plot = to_numpy(err_dy2)
    
    # Row 1-2: Original field (2x2 grid)
    # Top-left: Real part (or absolute if complex)
    if use_complex:
        F_abs = to_numpy(np.abs(F))
        im1 = axes[0, 0].contourf(X, Y, F_abs, levels=20, cmap='RdBu_r')
        axes[0, 0].set_title('Original Field (|F|)', fontsize=12)
        plt.colorbar(im1, ax=axes[0, 0])
        
        F_imag_plot = to_numpy(F_imag)
        im2 = axes[0, 1].contourf(X, Y, F_imag_plot, levels=20, cmap='RdBu_r')
        axes[0, 1].set_title('Original Field (Imag)', fontsize=12)
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Second row: could show phase or another view
        F_phase = to_numpy(np.angle(F))
        im3 = axes[1, 0].contourf(X, Y, F_phase, levels=20, cmap='hsv')
        axes[1, 0].set_title('Original Field (Phase)', fontsize=12)
        plt.colorbar(im3, ax=axes[1, 0])
        
        im4 = axes[1, 1].contourf(X, Y, F_abs, levels=20, cmap='viridis')
        axes[1, 1].set_title('Original Field (|F|, alt)', fontsize=12)
        plt.colorbar(im4, ax=axes[1, 1])
    else:
        # Real case: show same field in all 4 positions or different views
        im1 = axes[0, 0].contourf(X, Y, F_plot_real, levels=20, cmap='RdBu_r')
        axes[0, 0].set_title('Original Field', fontsize=12)
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].contourf(X, Y, F_plot_real, levels=20, cmap='viridis')
        axes[0, 1].set_title('Original Field (alt colormap)', fontsize=12)
        plt.colorbar(im2, ax=axes[0, 1])
        
        im3 = axes[1, 0].contourf(X, Y, F_plot_real, levels=20, cmap='plasma')
        axes[1, 0].set_title('Original Field (alt colormap 2)', fontsize=12)
        plt.colorbar(im3, ax=axes[1, 0])
        
        im4 = axes[1, 1].contourf(X, Y, F_plot_real, levels=20, cmap='coolwarm')
        axes[1, 1].set_title('Original Field (alt colormap 3)', fontsize=12)
        plt.colorbar(im4, ax=axes[1, 1])
    
    # Row 3: First derivatives
    im5 = axes[2, 0].contourf(X, Y, np.abs(df_dx_num_plot) if use_complex else df_dx_num_plot, 
                               levels=20, cmap='RdBu_r')
    axes[2, 0].set_title('df/dx (numerical)', fontsize=12)
    plt.colorbar(im5, ax=axes[2, 0])
    
    im6 = axes[2, 1].contourf(X, Y, np.abs(df_dy_num_plot) if use_complex else df_dy_num_plot,
                               levels=20, cmap='RdBu_r')
    axes[2, 1].set_title('df/dy (numerical)', fontsize=12)
    plt.colorbar(im6, ax=axes[2, 1])
    
    # Row 4: Second derivatives
    im7 = axes[3, 0].contourf(X, Y, np.abs(d2f_dx2_num_plot) if use_complex else d2f_dx2_num_plot,
                               levels=20, cmap='RdBu_r')
    axes[3, 0].set_title('d²f/dx² (numerical)', fontsize=12)
    plt.colorbar(im7, ax=axes[3, 0])
    
    im8 = axes[3, 1].contourf(X, Y, np.abs(d2f_dy2_num_plot) if use_complex else d2f_dy2_num_plot,
                               levels=20, cmap='RdBu_r')
    axes[3, 1].set_title('d²f/dy² (numerical)', fontsize=12)
    plt.colorbar(im8, ax=axes[3, 1])
    
    # Set labels for all subplots
    for i in range(4):
        for j in range(2):
            axes[i, j].set_xlabel('x', fontsize=10)
            axes[i, j].set_ylabel('y', fontsize=10)
            axes[i, j].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('figs/derivatives_grid.png', dpi=150, bbox_inches='tight')
    print(f"Saved 4x2 grid plot to figs/derivatives_grid.png")
    plt.close()
    
    # Create separate 4x2 grid for errors
    fig2, axes2 = plt.subplots(4, 2, figsize=(14, 16))
    
    # Row 1-2: Original field (same as above, 2x2)
    if use_complex:
        im_e1 = axes2[0, 0].contourf(X, Y, F_abs, levels=20, cmap='RdBu_r')
        axes2[0, 0].set_title('Original Field (|F|)', fontsize=12)
        plt.colorbar(im_e1, ax=axes2[0, 0])
        
        im_e2 = axes2[0, 1].contourf(X, Y, F_imag_plot, levels=20, cmap='RdBu_r')
        axes2[0, 1].set_title('Original Field (Imag)', fontsize=12)
        plt.colorbar(im_e2, ax=axes2[0, 1])
        
        im_e3 = axes2[1, 0].contourf(X, Y, F_phase, levels=20, cmap='hsv')
        axes2[1, 0].set_title('Original Field (Phase)', fontsize=12)
        plt.colorbar(im_e3, ax=axes2[1, 0])
        
        im_e4 = axes2[1, 1].contourf(X, Y, F_abs, levels=20, cmap='viridis')
        axes2[1, 1].set_title('Original Field (|F|, alt)', fontsize=12)
        plt.colorbar(im_e4, ax=axes2[1, 1])
    else:
        im_e1 = axes2[0, 0].contourf(X, Y, F_plot_real, levels=20, cmap='RdBu_r')
        axes2[0, 0].set_title('Original Field', fontsize=12)
        plt.colorbar(im_e1, ax=axes2[0, 0])
        
        im_e2 = axes2[0, 1].contourf(X, Y, F_plot_real, levels=20, cmap='viridis')
        axes2[0, 1].set_title('Original Field (alt)', fontsize=12)
        plt.colorbar(im_e2, ax=axes2[0, 1])
        
        im_e3 = axes2[1, 0].contourf(X, Y, F_plot_real, levels=20, cmap='plasma')
        axes2[1, 0].set_title('Original Field (alt 2)', fontsize=12)
        plt.colorbar(im_e3, ax=axes2[1, 0])
        
        im_e4 = axes2[1, 1].contourf(X, Y, F_plot_real, levels=20, cmap='coolwarm')
        axes2[1, 1].set_title('Original Field (alt 3)', fontsize=12)
        plt.colorbar(im_e4, ax=axes2[1, 1])
    
    # Row 3: First derivative errors
    err_dx_abs = np.abs(err_dx_plot)
    im_e5 = axes2[2, 0].contourf(X, Y, err_dx_abs, levels=20, cmap='hot')
    axes2[2, 0].set_title(f'Error df/dx (max={np.max(err_dx_abs):.2e})', fontsize=12)
    plt.colorbar(im_e5, ax=axes2[2, 0])
    
    err_dy_abs = np.abs(err_dy_plot)
    im_e6 = axes2[2, 1].contourf(X, Y, err_dy_abs, levels=20, cmap='hot')
    axes2[2, 1].set_title(f'Error df/dy (max={np.max(err_dy_abs):.2e})', fontsize=12)
    plt.colorbar(im_e6, ax=axes2[2, 1])
    
    # Row 4: Second derivative errors
    err_dx2_abs = np.abs(err_dx2_plot)
    im_e7 = axes2[3, 0].contourf(X, Y, err_dx2_abs, levels=20, cmap='hot')
    axes2[3, 0].set_title(f'Error d²f/dx² (max={np.max(err_dx2_abs):.2e})', fontsize=12)
    plt.colorbar(im_e7, ax=axes2[3, 0])
    
    err_dy2_abs = np.abs(err_dy2_plot)
    im_e8 = axes2[3, 1].contourf(X, Y, err_dy2_abs, levels=20, cmap='hot')
    axes2[3, 1].set_title(f'Error d²f/dy² (max={np.max(err_dy2_abs):.2e})', fontsize=12)
    plt.colorbar(im_e8, ax=axes2[3, 1])
    
    # Set labels for all subplots
    for i in range(4):
        for j in range(2):
            axes2[i, j].set_xlabel('x', fontsize=10)
            axes2[i, j].set_ylabel('y', fontsize=10)
            axes2[i, j].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('figs/errors_grid.png', dpi=150, bbox_inches='tight')
    print(f"Saved 4x2 error grid plot to figs/errors_grid.png")
    plt.close()
    
    print(f"\nGrid: nx={nx}, ny={ny}, degree={degree}, complex={use_complex}")
    print(f"\nL2 error:")
    print(f"  df/dx:   L2={l2_err_dx:.6e}")
    print(f"  df/dy:   L2={l2_err_dy:.6e}")
    print(f"  d2f/dx2: L2={l2_err_dx2:.6e}")
    print(f"  d2f/dy2: L2={l2_err_dy2:.6e}")
    print(f"\nRelative L_infty error (max|e| / max|exact|):")
    print(f"  df/dx:   rel L_inf={rel_linf_dx:.6e}")
    print(f"  df/dy:   rel L_inf={rel_linf_dy:.6e}")
    print(f"  d2f/dx2: rel L_inf={rel_linf_dx2:.6e}")
    print(f"  d2f/dy2: rel L_inf={rel_linf_dy2:.6e}")
    print(f"\nTiming:")
    print(f"  differentiate_1_2: {diff12_time:.6f} s (single call)")
    
    # Check if errors are reasonable (Taylor-Green vortex is smooth, so we expect high accuracy)
    tolerance = 1e-6
    all_ok = (l2_err_dx < tolerance and l2_err_dy < tolerance and
              l2_err_dx2 < tolerance and l2_err_dy2 < tolerance)
    
    if all_ok:
        print(f"\n✓ All L2 errors below tolerance ({tolerance:.1e})")
    else:
        print(f"\n⚠ Some L2 errors exceed tolerance ({tolerance:.1e})")
        print("  (This may indicate a numerical issue)")
    
    print("="*80 + "\n")
    
    return {
        'l2_err_dx': l2_err_dx,
        'l2_err_dy': l2_err_dy,
        'l2_err_dx2': l2_err_dx2,
        'l2_err_dy2': l2_err_dy2,
        't_diff12': diff12_time,
    }


def compare_performance(nx, ny, degree, use_gpu=False, n_runs=10, use_complex=False):
    """
    Benchmark performance of differentiate_1_2.
    """
    print("\n" + "="*80)
    print("=== Performance Benchmark ===")
    print("="*80)
    
    # Setup - 2D Taylor-Green vortex: f(x,y) = sin(x) * sin(y)
    a, b = 0.0, 2.0 * np.pi
    x = np.linspace(a, b, nx, endpoint=True)
    y = np.linspace(a, b, ny, endpoint=True)
    X, Y = np.meshgrid(x, y, indexing="xy")  # X varies along axis 1, Y along axis 0
    F_real = np.sin(X) * np.sin(Y)
    if use_complex:
        shift = np.pi / 4.0
        F_imag = np.sin(X + shift) * np.sin(Y + shift)
        F = F_real + 1j * F_imag
    else:
        F = F_real
    
    # Convert to GPU array if needed
    if use_gpu and _HAS_CUPY:
        dtype = cp.complex128 if use_complex else cp.float64
        F_gpu = cp.asarray(F, dtype=dtype)
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
    _ = op.differentiate_1_2(F_gpu)
    if use_gpu and _HAS_CUPY:
        cp.cuda.Stream.null.synchronize()
    
    # Time differentiate_1_2
    print(f"Timing differentiate_1_2 ({n_runs} runs)...")
    times = []
    for _ in range(n_runs):
        if use_gpu and _HAS_CUPY:
            # Synchronize before timing
            cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        _ = op.differentiate_1_2(F_gpu)
        if use_gpu and _HAS_CUPY:
            # Synchronize after computation to ensure GPU work is done
            cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    times = np.array(times)
    
    # Statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\nGrid: nx={nx}, ny={ny}, degree={degree}, use_gpu={use_gpu}, complex={use_complex}")
    print(f"\n{'Metric':<15s} {'Mean':>12s} {'Std':>12s} {'Min':>12s} {'Max':>12s}")
    print("-" * 65)
    print(f"{'Time (s)':<15s} {mean_time:12.6f} {std_time:12.6f} {min_time:12.6f} {max_time:12.6f}")
    print("="*80 + "\n")
    
    return {
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
    }


def compare_gpu_cpu(nx, ny, degree, n_runs=10, use_complex=False):
    """
    Compare GPU vs CPU performance.
    """
    if not _HAS_CUPY:
        print("\n" + "="*80)
        print("=== GPU vs CPU Comparison ===")
        print("="*80)
        print("CuPy is not available. Skipping GPU vs CPU comparison.")
        print("Install CuPy to enable GPU comparison (e.g., `pip install cupy-cuda12x`)")
        print("="*80 + "\n")
        return None
    
    print("\n" + "="*80)
    print("=== GPU vs CPU Comparison ===")
    print("="*80)
    
    # Setup - 2D Taylor-Green vortex: f(x,y) = sin(x) * sin(y)
    a, b = 0.0, 2.0 * np.pi
    x = np.linspace(a, b, nx, endpoint=True)
    y = np.linspace(a, b, ny, endpoint=True)
    X, Y = np.meshgrid(x, y, indexing="xy")  # X varies along axis 1, Y along axis 0
    F_real = np.sin(X) * np.sin(Y)
    if use_complex:
        shift = np.pi / 4.0
        F_imag = np.sin(X + shift) * np.sin(Y + shift)
        F = F_real + 1j * F_imag
    else:
        F = F_real
    
    # Build CPU operator
    print("Building CPU operator...")
    op_cpu = bspf2d.from_grids(
        x=x,
        y=y,
        degree_x=degree,
        degree_y=degree,
        n_basis_x=4 * degree,
        n_basis_y=4 * degree,
        num_boundary_points_x = degree + 5,
        num_boundary_points_y = degree + 5,
        use_clustering_x=True,
        use_clustering_y=True,
        correction="spectral",
        use_gpu=False,
    )
    
    # Build GPU operator
    print("Building GPU operator...")
    if not _HAS_CUPY:
        raise RuntimeError("CuPy is required for GPU comparison but is not available.")

    # Convert inputs to CuPy to satisfy strict backend checking
    x_gpu = cp.asarray(x, dtype=cp.float64)
    y_gpu = cp.asarray(y, dtype=cp.float64)
    F_gpu = cp.asarray(F, dtype=cp.complex128 if use_complex else cp.float64)
    op_gpu = bspf2d.from_grids(
        x=x_gpu,
        y=y_gpu,
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
    _ = op_cpu.differentiate_1_2(F)
    print("Warming up GPU...")
    _ = op_gpu.differentiate_1_2(F_gpu)
    cp.cuda.Stream.null.synchronize()
    
    # Time CPU version
    print(f"Timing CPU version ({n_runs} runs)...")
    times_cpu = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = op_cpu.differentiate_1_2(F)
        t1 = time.perf_counter()
        times_cpu.append(t1 - t0)
    
    # Time GPU version
    print(f"Timing GPU version ({n_runs} runs)...")
    times_gpu = []
    for _ in range(n_runs):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        _ = op_gpu.differentiate_1_2(F_gpu)
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
    
    print(f"\nGrid: nx={nx}, ny={ny}, degree={degree}, complex={use_complex}")
    print(f"\n{'Version':<15s} {'Mean':>12s} {'Std':>12s} {'Min':>12s} {'Max':>12s}")
    print("-" * 65)
    print(f"{'CPU':<15s} {mean_cpu:12.6f} {std_cpu:12.6f} {min_cpu:12.6f} {max_cpu:12.6f}")
    print(f"{'GPU':<15s} {mean_gpu:12.6f} {std_gpu:12.6f} {min_gpu:12.6f} {max_gpu:12.6f}")
    print("-" * 65)
    
    if speedup > 1.0:
        print(f"\n✓ GPU is {speedup:.2f}x faster than CPU")
    else:
        print(f"\n⚠ CPU is {1.0/speedup:.2f}x faster than GPU")
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
    p.add_argument("--nx", type=int, default=1024, 
                   help="Grid points in x")
    p.add_argument("--ny", type=int, default=1024, 
                   help="Grid points in y")
    p.add_argument("--degree", type=int, default=8, 
                   help="B-spline degree")
    p.add_argument("--runs", type=int, default=100, help="Number of timing runs")
    p.add_argument("--gpu", action="store_true", help="Use GPU (CuPy) if available")
    p.add_argument("--complex", action="store_true", help="Use complex input field")
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
            use_complex=args.complex,
        )
    
    # Compare GPU vs CPU for batched version (if CuPy is available and not skipped)
    if not args.no_gpu_cpu:
        compare_gpu_cpu(
            nx=args.nx,
            ny=args.ny,
            degree=args.degree,
            n_runs=args.runs,
            use_complex=args.complex,
        )


if __name__ == "__main__":
    main()

