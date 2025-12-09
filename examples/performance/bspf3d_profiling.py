from __future__ import annotations
import os
import sys
import time

# Set CUDA_PATH for NVHPC SDK before importing CuPy
# This ensures CuPy can find CUDA headers during JIT compilation
if 'CUDA_PATH' not in os.environ:
    nvhpc_cuda_path = '/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6'
    if os.path.exists(nvhpc_cuda_path):
        os.environ['CUDA_PATH'] = nvhpc_cuda_path
        os.environ['CUDA_HOME'] = nvhpc_cuda_path

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
import numpy as np
import numpy.typing as npt
from scipy import linalg as sla  # CPU fallback

# Optional GPU backend
_HAS_CUPY = False
try:
    import cupy as cp
    import cupyx.scipy.linalg as cpla
    _HAS_CUPY = True
except Exception:
    cp = None
    cpla = None

# Support running as a script or as part of the package
# Prefer optimized profiling version, fall back to regular version
try:
    from .bspf1d_profiling import bspf1d  # optimized profiling version
except ImportError:
    try:
        from .bspf1d import bspf1d  # when imported as package
    except ImportError:
        # when executed as a script: add repository src to sys.path
        import sys
        _here = os.path.abspath(os.path.dirname(__file__))
        _root = os.path.abspath(os.path.join(_here, "..", "..", "src"))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        try:
            from bspf.bspf1d import bspf1d  # type: ignore
        except ImportError:
            # Fallback: try importing from examples/performance
            _perf_dir = os.path.abspath(os.path.dirname(__file__))
            if _perf_dir not in sys.path:
                sys.path.insert(0, _perf_dir)
            from bspf1d_profiling import bspf1d  # optimized profiling version

# Patch bspf package to use profiling version of bspf1d before importing bspf3d
try:
    import bspf
    bspf.bspf1d = bspf1d  # Replace with profiling version
except ImportError:
    pass

# Import bspf3d (it will now use the profiling version of bspf1d)
_bspf3d_base = None
try:
    from bspf.bspf3d import bspf3d as _bspf3d_base
except ImportError:
    # when executed as a script: add repository src to sys.path
    _here = os.path.abspath(os.path.dirname(__file__))
    _root = os.path.abspath(os.path.join(_here, "..", "..", "src"))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    # Patch before importing
    try:
        import bspf
        bspf.bspf1d = bspf1d
    except ImportError:
        pass
    from bspf.bspf3d import bspf3d as _bspf3d_base  # type: ignore

if _bspf3d_base is None:
    raise ImportError("Failed to import bspf3d from bspf.bspf3d")

# Create a wrapper class that fixes GPU array handling
class bspf3d(_bspf3d_base):
    """Wrapper around bspf3d that fixes GPU array handling in differentiate_1_2."""
    
    def differentiate_1_2(
        self,
        F,
        *,
        lam_x: float = 0.0,
        lam_y: float = 0.0,
        lam_z: float = 0.0,
        uniform_bc_x: bool = False,
        uniform_bc_y: bool = False,
        uniform_bc_z: bool = False,
        bc_x=None,
        bc_y=None,
        bc_z=None,
        use_loop: bool = False,
    ):
        """
        Override differentiate_1_2 to fix GPU array handling.
        The parent method has a bug where it tries to use np.asarray() on CuPy arrays.
        This wrapper avoids unnecessary conversions by implementing the batched path directly
        when GPU arrays are used and batched method is available.
        """
        # Check if input is GPU array
        is_gpu_array = _HAS_CUPY and cp is not None and isinstance(F, cp.ndarray)
        
        # Check if batched method is available
        has_batched = hasattr(self.x_model, 'differentiate_1_2_batched') if not use_loop else False
        
        # If using batched operations with GPU arrays, implement directly to avoid conversions
        if is_gpu_array and not use_loop and has_batched:
            # Use batched method directly with GPU arrays - no conversion needed!
            F_gpu = F
            nz, ny, nx = F_gpu.shape
            
            # For x-direction: differentiate along axis=2 (columns)
            # Reshape to (nx, nz*ny) for batched differentiation
            F_reshaped_x = F_gpu.reshape(nz * ny, nx).T  # (nx, nz*ny)
            dF_dx_T, d2F_dx2_T, _ = self.x_model.differentiate_1_2_batched(F_reshaped_x, lam=lam_x)
            dF_dx = dF_dx_T.T.reshape(nz, ny, nx)
            d2F_dx2 = d2F_dx2_T.T.reshape(nz, ny, nx)
            
            # For y-direction: differentiate along axis=1 (rows)
            # Reshape to (ny, nz*nx) for batched differentiation
            F_reshaped_y = F_gpu.transpose(0, 2, 1).reshape(nz * nx, ny).T  # (ny, nz*nx)
            dF_dy_T, d2F_dy2_T, _ = self.y_model.differentiate_1_2_batched(F_reshaped_y, lam=lam_y)
            dF_dy_T_reshaped = dF_dy_T.T.reshape(nz, nx, ny)
            dF_dy = dF_dy_T_reshaped.transpose(0, 2, 1)
            d2F_dy2_T_reshaped = d2F_dy2_T.T.reshape(nz, nx, ny)
            d2F_dy2 = d2F_dy2_T_reshaped.transpose(0, 2, 1)
            
            # For z-direction: differentiate along axis=0 (depth)
            # Reshape to (nz, ny*nx) for batched differentiation
            F_reshaped_z = F_gpu.reshape(nz, ny * nx)  # (nz, ny*nx)
            dF_dz, d2F_dz2, _ = self.z_model.differentiate_1_2_batched(F_reshaped_z, lam=lam_z)
            dF_dz = dF_dz.reshape(nz, ny, nx)
            d2F_dz2 = d2F_dz2.reshape(nz, ny, nx)
            
            return (dF_dx, dF_dy, dF_dz, d2F_dx2, d2F_dy2, d2F_dz2)
        
        # For loop operations or when batched is not available, convert to NumPy
        # (parent method has bugs with GPU arrays in these paths)
        if is_gpu_array:
            F_np = cp.asnumpy(F).astype(np.float64)
        else:
            F_np = np.asarray(F, dtype=np.float64)
        
        # Call parent method with NumPy array
        result = _bspf3d_base.differentiate_1_2(
            self,
            F_np,
            lam_x=lam_x, lam_y=lam_y, lam_z=lam_z,
            uniform_bc_x=uniform_bc_x, uniform_bc_y=uniform_bc_y, uniform_bc_z=uniform_bc_z,
            bc_x=bc_x, bc_y=bc_y, bc_z=bc_z,
            use_loop=use_loop,
        )
        
        # Convert back to GPU if input was GPU (only needed for loop/fallback paths)
        if is_gpu_array:
            result = tuple(cp.asarray(r, dtype=cp.float64) for r in result)
        
        return result

Array = npt.NDArray[np.float64]


# =============================================================================
# Profiling helpers
# =============================================================================
def _stats(arr: np.ndarray) -> Tuple[float, float, float, float]:
    """Return mean, std, min, max for a 1D array."""
    return float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))


def run_profile_3d(
    nx: int = 64,
    ny: int = 64,
    nz: int = 64,
    degree: int = 7,
    n_runs: int = 50,
    use_gpu: bool = False,
) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Benchmark bspf3d.differentiate_1_2 over Taylor-Green vortex test function.
    Returns a dict of timing stats (mean, std, min, max) in seconds.
    """
    a, b = 0.0, 2.0 * np.pi
    x = np.linspace(a, b, nx, endpoint=True)
    y = np.linspace(a, b, ny, endpoint=True)
    z = np.linspace(a, b, nz, endpoint=True)
    # Use same indexing as working version (diff_gpu_vs_cpu_3d.py)
    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")  # (ny, nx, nz)
    to_nz_ny_nx = lambda A: np.moveaxis(A, 2, 0)  # (ny, nx, nz) -> (nz, ny, nx)

    # Taylor-Green vortex test function: f(x,y,z) = sin(x) * sin(y) * sin(z)
    F = to_nz_ny_nx(np.sin(X) * np.sin(Y) * np.sin(Z))
    
    # Convert to GPU array if needed
    if use_gpu and _HAS_CUPY:
        F = cp.asarray(F, dtype=cp.float64)

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

    # Warmup
    _ = op.differentiate_1_2(F)
    if use_gpu and _HAS_CUPY:
        cp.cuda.Stream.null.synchronize()

    # With batched operations, we can only measure total time per direction
    t_total = []
    t_x_batch = []  # Time for x-direction batched operations (k=1 + k=2)
    t_y_batch = []  # Time for y-direction batched operations (k=1 + k=2)
    t_z_batch = []  # Time for z-direction batched operations (k=1 + k=2)
    t_x_k1 = []     # Time for x-direction k=1 only
    t_x_k2 = []     # Time for x-direction k=2 only
    t_y_k1 = []     # Time for y-direction k=1 only
    t_y_k2 = []     # Time for y-direction k=2 only
    t_z_k1 = []     # Time for z-direction k=1 only
    t_z_k2 = []     # Time for z-direction k=2 only

    for _ in range(n_runs):
        if use_gpu and _HAS_CUPY:
            cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()

        # X direction: batched differentiation along axis=2 (columns)
        # Reshape to (nx, nz*ny) for batched differentiation
        t_x0 = time.perf_counter()
        F_np = np.asarray(F, dtype=np.float64) if not (use_gpu and _HAS_CUPY) else cp.asnumpy(F)
        nz, ny, nx = F_np.shape
        F_reshaped_x = F_np.reshape(nz * ny, nx).T  # (nx, nz*ny)
        if use_gpu and _HAS_CUPY:
            F_reshaped_x = cp.asarray(F_reshaped_x, dtype=cp.float64)
        # Check if batched method exists (from profiling version)
        if hasattr(op.x_model, 'differentiate_1_2_batched'):
            dF_dx_T, d2F_dx2_T, _ = op.x_model.differentiate_1_2_batched(F_reshaped_x, lam=0.0)
        else:
            # Fallback: use loop (shouldn't happen with profiling version, but handle gracefully)
            dF_dx_list = []
            d2F_dx2_list = []
            for col in range(F_reshaped_x.shape[1]):
                df1, df2, _ = op.x_model.differentiate_1_2(F_reshaped_x[:, col], lam=0.0)
                dF_dx_list.append(df1)
                d2F_dx2_list.append(df2)
            dF_dx_T = np.array(dF_dx_list).T if not (use_gpu and _HAS_CUPY) else cp.array(dF_dx_list).T
            d2F_dx2_T = np.array(d2F_dx2_list).T if not (use_gpu and _HAS_CUPY) else cp.array(d2F_dx2_list).T
        if use_gpu and _HAS_CUPY:
            cp.cuda.Stream.null.synchronize()
        t_x1 = time.perf_counter()
        t_x_batch.append(t_x1 - t_x0)
        t_x_k1.append((t_x1 - t_x0) / 2.0)
        t_x_k2.append((t_x1 - t_x0) / 2.0)

        # Y direction: batched differentiation along axis=1 (rows)
        t_y0 = time.perf_counter()
        F_reshaped_y = F_np.transpose(0, 2, 1).reshape(nz * nx, ny).T  # (ny, nz*nx)
        if use_gpu and _HAS_CUPY:
            F_reshaped_y = cp.asarray(F_reshaped_y, dtype=cp.float64)
        if hasattr(op.y_model, 'differentiate_1_2_batched'):
            dF_dy_T, d2F_dy2_T, _ = op.y_model.differentiate_1_2_batched(F_reshaped_y, lam=0.0)
        else:
            # Fallback: use loop
            dF_dy_list = []
            d2F_dy2_list = []
            for col in range(F_reshaped_y.shape[1]):
                df1, df2, _ = op.y_model.differentiate_1_2(F_reshaped_y[:, col], lam=0.0)
                dF_dy_list.append(df1)
                d2F_dy2_list.append(df2)
            dF_dy_T = np.array(dF_dy_list).T if not (use_gpu and _HAS_CUPY) else cp.array(dF_dy_list).T
            d2F_dy2_T = np.array(d2F_dy2_list).T if not (use_gpu and _HAS_CUPY) else cp.array(d2F_dy2_list).T
        if use_gpu and _HAS_CUPY:
            cp.cuda.Stream.null.synchronize()
        t_y1 = time.perf_counter()
        t_y_batch.append(t_y1 - t_y0)
        t_y_k1.append((t_y1 - t_y0) / 2.0)
        t_y_k2.append((t_y1 - t_y0) / 2.0)

        # Z direction: batched differentiation along axis=0 (depth)
        t_z0 = time.perf_counter()
        F_reshaped_z = F_np.reshape(nz, ny * nx)  # (nz, ny*nx)
        if use_gpu and _HAS_CUPY:
            F_reshaped_z = cp.asarray(F_reshaped_z, dtype=cp.float64)
        if hasattr(op.z_model, 'differentiate_1_2_batched'):
            dF_dz, d2F_dz2, _ = op.z_model.differentiate_1_2_batched(F_reshaped_z, lam=0.0)
        else:
            # Fallback: use loop
            dF_dz_list = []
            d2F_dz2_list = []
            for col in range(F_reshaped_z.shape[1]):
                df1, df2, _ = op.z_model.differentiate_1_2(F_reshaped_z[:, col], lam=0.0)
                dF_dz_list.append(df1)
                d2F_dz2_list.append(df2)
            dF_dz = np.array(dF_dz_list).T if not (use_gpu and _HAS_CUPY) else cp.array(dF_dz_list).T
            d2F_dz2 = np.array(d2F_dz2_list).T if not (use_gpu and _HAS_CUPY) else cp.array(d2F_dz2_list).T
        if use_gpu and _HAS_CUPY:
            cp.cuda.Stream.null.synchronize()
        t_z1 = time.perf_counter()
        t_z_batch.append(t_z1 - t_z0)
        t_z_k1.append((t_z1 - t_z0) / 2.0)
        t_z_k2.append((t_z1 - t_z0) / 2.0)

        t1 = time.perf_counter()
        t_total.append(t1 - t0)

    t_total = np.asarray(t_total)
    t_x_batch = np.asarray(t_x_batch)
    t_y_batch = np.asarray(t_y_batch)
    t_z_batch = np.asarray(t_z_batch)
    t_x_k1 = np.asarray(t_x_k1)
    t_x_k2 = np.asarray(t_x_k2)
    t_y_k1 = np.asarray(t_y_k1)
    t_y_k2 = np.asarray(t_y_k2)
    t_z_k1 = np.asarray(t_z_k1)
    t_z_k2 = np.asarray(t_z_k2)

    # Compute overhead
    t_x_overhead = t_x_batch - (t_x_k1 + t_x_k2)
    t_y_overhead = t_y_batch - (t_y_k1 + t_y_k2)
    t_z_overhead = t_z_batch - (t_z_k1 + t_z_k2)

    # Print detailed breakdown
    print("\n" + "="*80)
    print("=== bspf3d differentiate_1_2 profiling (batched, non-overlapping) ===")
    print("="*80)
    print(f"grid: nx={nx}, ny={ny}, nz={nz}, degree={degree}, runs={n_runs}, use_gpu={use_gpu}")
    print(f"\n{'Component':20s} {'Mean':>12s} {'Std':>12s} {'Min':>12s} {'Max':>12s} {'% of total':>12s}")
    print("-" * 80)
    
    total_mean = np.mean(t_total)
    
    # X-direction breakdown
    print("\nX-direction (batched, nz*ny={} columns):".format(nz * ny))
    mean, std, tmin, tmax = _stats(t_x_batch)
    pct = 100.0 * mean / total_mean if total_mean > 0 else 0.0
    print(f"{'  x_batch_total':20s}: {mean:12.6f} {std:12.6f} {tmin:12.6f} {tmax:12.6f} {pct:11.2f}%")
    mean, std, tmin, tmax = _stats(t_x_k1)
    pct = 100.0 * mean / total_mean if total_mean > 0 else 0.0
    print(f"{'  x_k1 (df/dx)':20s}: {mean:12.6f} {std:12.6f} {tmin:12.6f} {tmax:12.6f} {pct:11.2f}%")
    mean, std, tmin, tmax = _stats(t_x_k2)
    pct = 100.0 * mean / total_mean if total_mean > 0 else 0.0
    print(f"{'  x_k2 (d2f/dx2)':20s}: {mean:12.6f} {std:12.6f} {tmin:12.6f} {tmax:12.6f} {pct:11.2f}%")
    mean, _, _, _ = _stats(t_x_overhead)
    pct = 100.0 * mean / total_mean if total_mean > 0 else 0.0
    print(f"{'  x_overhead':20s}: {mean:12.6f} {'-'*8:>12s} {'-'*8:>12s} {'-'*8:>12s} {pct:11.2f}%")
    
    # Y-direction breakdown
    print("\nY-direction (batched, nz*nx={} columns):".format(nz * nx))
    mean, std, tmin, tmax = _stats(t_y_batch)
    pct = 100.0 * mean / total_mean if total_mean > 0 else 0.0
    print(f"{'  y_batch_total':20s}: {mean:12.6f} {std:12.6f} {tmin:12.6f} {tmax:12.6f} {pct:11.2f}%")
    mean, std, tmin, tmax = _stats(t_y_k1)
    pct = 100.0 * mean / total_mean if total_mean > 0 else 0.0
    print(f"{'  y_k1 (df/dy)':20s}: {mean:12.6f} {std:12.6f} {tmin:12.6f} {tmax:12.6f} {pct:11.2f}%")
    mean, std, tmin, tmax = _stats(t_y_k2)
    pct = 100.0 * mean / total_mean if total_mean > 0 else 0.0
    print(f"{'  y_k2 (d2f/dy2)':20s}: {mean:12.6f} {std:12.6f} {tmin:12.6f} {tmax:12.6f} {pct:11.2f}%")
    mean, _, _, _ = _stats(t_y_overhead)
    pct = 100.0 * mean / total_mean if total_mean > 0 else 0.0
    print(f"{'  y_overhead':20s}: {mean:12.6f} {'-'*8:>12s} {'-'*8:>12s} {'-'*8:>12s} {pct:11.2f}%")
    
    # Z-direction breakdown
    print("\nZ-direction (batched, ny*nx={} columns):".format(ny * nx))
    mean, std, tmin, tmax = _stats(t_z_batch)
    pct = 100.0 * mean / total_mean if total_mean > 0 else 0.0
    print(f"{'  z_batch_total':20s}: {mean:12.6f} {std:12.6f} {tmin:12.6f} {tmax:12.6f} {pct:11.2f}%")
    mean, std, tmin, tmax = _stats(t_z_k1)
    pct = 100.0 * mean / total_mean if total_mean > 0 else 0.0
    print(f"{'  z_k1 (df/dz)':20s}: {mean:12.6f} {std:12.6f} {tmin:12.6f} {tmax:12.6f} {pct:11.2f}%")
    mean, std, tmin, tmax = _stats(t_z_k2)
    pct = 100.0 * mean / total_mean if total_mean > 0 else 0.0
    print(f"{'  z_k2 (d2f/dz2)':20s}: {mean:12.6f} {std:12.6f} {tmin:12.6f} {tmax:12.6f} {pct:11.2f}%")
    mean, _, _, _ = _stats(t_z_overhead)
    pct = 100.0 * mean / total_mean if total_mean > 0 else 0.0
    print(f"{'  z_overhead':20s}: {mean:12.6f} {'-'*8:>12s} {'-'*8:>12s} {'-'*8:>12s} {pct:11.2f}%")
    
    # Overall (non-overlapping top-level)
    print("\nOverall:")
    mean, std, tmin, tmax = _stats(t_total)
    print(f"{'total':20s}: {mean:12.6f} {std:12.6f} {tmin:12.6f} {tmax:12.6f} {'100.00':>12s}%")
    print("="*80 + "\n")
    print("Note: Using batched differentiate_1_2 which computes both derivatives together.")
    print("      This reuses RHS build, KKT solve, and spline evaluation for efficiency.")
    print("      k=1 and k=2 times are approximated by dividing total time in half.")
    print("="*80 + "\n")

    # Return summary for programmatic access
    summary = {
        "x_batch_total": _stats(t_x_batch),
        "x_k1": _stats(t_x_k1),
        "x_k2": _stats(t_x_k2),
        "x_overhead": _stats(t_x_overhead),
        "y_batch_total": _stats(t_y_batch),
        "y_k1": _stats(t_y_k1),
        "y_k2": _stats(t_y_k2),
        "y_overhead": _stats(t_y_overhead),
        "z_batch_total": _stats(t_z_batch),
        "z_k1": _stats(t_z_k1),
        "z_k2": _stats(t_z_k2),
        "z_overhead": _stats(t_z_overhead),
        "total": _stats(t_total),
    }
    
    return summary


def main():
    # Default CPU profiling run
    run_profile_3d()


if __name__ == "__main__":
    main()

