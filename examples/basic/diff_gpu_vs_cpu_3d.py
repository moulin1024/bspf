#!/usr/bin/env python3
"""
Benchmark bspf3d with 3D Taylor–Green vortex solution.

Computes the gradient tensor ∇u for the Taylor–Green vortex velocity field:
  u(x,y,z) = sin(x) * cos(y) * cos(z)
  v(x,y,z) = -cos(x) * sin(y) * cos(z)
  w(x,y,z) = 0

The gradient tensor is a 3×3 matrix of all partial derivatives:
  ∇u = [∂u/∂x  ∂u/∂y  ∂u/∂z]
       [∂v/∂x  ∂v/∂y  ∂v/∂z]
       [∂w/∂x  ∂w/∂y  ∂w/∂z]

Measures:
  - Preparation: building dx/dy/dz plans (and moving data/LU to device on GPU).
  - Evaluation: computing full gradient tensor (9 derivatives) 'reps' times.
  - Accuracy: L2 relative errors for each component of the gradient tensor.

Supports:
  - CPU vs GPU (CuPy) comparison
  - Optional Neumann flux (taken from the analytical derivative)
  - Optional uniform boundary RHS optimization
  - Option to keep inputs on device to exclude H2D/D2H transfer overhead

Usage examples:
  CPU vs GPU, include transfers:
    python test3d.py --nx 128 --ny 128 --nz 128 --reps 20

  Exclude transfers (create inputs on device):
    python test3d.py --nx 192 --ny 160 --nz 128 --reps 20 --ondevice

  With Neumann flux and uniform BC:
    python test3d.py --neumann --uniform-bc

  Also check accuracy (L2 rel. errors):
    python test3d.py --check
"""

import time
import argparse
import numpy as np


from bspf import bspf3d
from bspf import bspf1d

# Optional GPU
try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False


# ------------------------- Taylor–Green vortex -------------------------
def make_axes(nx, ny, nz, domain=(0.0, 2.0*np.pi)):
    """Create coordinate axes. Default domain is [0, 2π] for Taylor–Green vortex."""
    x = np.linspace(domain[0], domain[1], nx)
    y = np.linspace(domain[0], domain[1], ny)
    z = np.linspace(domain[0], domain[1], nz)
    return x, y, z

def taylor_green_velocity(x, y, z, A=1.0, B=1.0, C=-2.0, a=1.0, b=1.0, c=1.0):
    """
    Original Taylor–Green vortex velocity field:
      u(x,y,z) = A cos(ax) sin(by) sin(cz)
      v(x,y,z) = B sin(ax) cos(by) sin(cz)
      w(x,y,z) = C sin(ax) sin(by) cos(cz)
    
    The continuity equation ∇·v = 0 requires: Aa + Bb + Cc = 0
    Default: A=1, B=1, C=-2, a=b=c=1 satisfies continuity (1 + 1 - 2 = 0)
    
    Returns (u, v, w) each with shape (nz, ny, nx)
    """
    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")  # (ny, nx, nz)
    # reorder to (nz, ny, nx)
    to_nz_ny_nx = lambda A: np.moveaxis(A, 2, 0)
    u = to_nz_ny_nx(A * np.cos(a * X) * np.sin(b * Y) * np.sin(c * Z))
    v = to_nz_ny_nx(B * np.sin(a * X) * np.cos(b * Y) * np.sin(c * Z))
    w = to_nz_ny_nx(C * np.sin(a * X) * np.sin(b * Y) * np.cos(c * Z))
    return u, v, w

def taylor_green_gradient_tensor(x, y, z, A=1.0, B=1.0, C=-2.0, a=1.0, b=1.0, c=1.0):
    """
    Analytical gradient tensor for original Taylor–Green vortex:
      u = A cos(ax) sin(by) sin(cz)
      v = B sin(ax) cos(by) sin(cz)
      w = C sin(ax) sin(by) cos(cz)
    
    Gradient tensor:
      ∇u = [∂u/∂x  ∂u/∂y  ∂u/∂z]   [-Aa sin(ax) sin(by) sin(cz)   Ab cos(ax) cos(by) sin(cz)   Ac cos(ax) sin(by) cos(cz)]
           [∂v/∂x  ∂v/∂y  ∂v/∂z] = [ Ba cos(ax) cos(by) sin(cz)  -Bb sin(ax) sin(by) sin(cz)   Bc sin(ax) cos(by) cos(cz)]
           [∂w/∂x  ∂w/∂y  ∂w/∂z]   [ Ca cos(ax) sin(by) cos(cz)   Cb sin(ax) cos(by) cos(cz)  -Cc sin(ax) sin(by) sin(cz)]
    
    Returns a dictionary with keys:
      'du_dx', 'du_dy', 'du_dz',
      'dv_dx', 'dv_dy', 'dv_dz',
      'dw_dx', 'dw_dy', 'dw_dz'
    Each array has shape (nz, ny, nx)
    """
    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")  # (ny, nx, nz)
    to_nz_ny_nx = lambda A: np.moveaxis(A, 2, 0)
    
    # Derivatives of u = A cos(ax) sin(by) sin(cz)
    du_dx = to_nz_ny_nx(-A * a * np.sin(a * X) * np.sin(b * Y) * np.sin(c * Z))
    du_dy = to_nz_ny_nx(A * b * np.cos(a * X) * np.cos(b * Y) * np.sin(c * Z))
    du_dz = to_nz_ny_nx(A * c * np.cos(a * X) * np.sin(b * Y) * np.cos(c * Z))
    
    # Derivatives of v = B sin(ax) cos(by) sin(cz)
    dv_dx = to_nz_ny_nx(B * a * np.cos(a * X) * np.cos(b * Y) * np.sin(c * Z))
    dv_dy = to_nz_ny_nx(-B * b * np.sin(a * X) * np.sin(b * Y) * np.sin(c * Z))
    dv_dz = to_nz_ny_nx(B * c * np.sin(a * X) * np.cos(b * Y) * np.cos(c * Z))
    
    # Derivatives of w = C sin(ax) sin(by) cos(cz)
    dw_dx = to_nz_ny_nx(C * a * np.cos(a * X) * np.sin(b * Y) * np.cos(c * Z))
    dw_dy = to_nz_ny_nx(C * b * np.sin(a * X) * np.cos(b * Y) * np.cos(c * Z))
    dw_dz = to_nz_ny_nx(-C * c * np.sin(a * X) * np.sin(b * Y) * np.sin(c * Z))
    
    return {
        'du_dx': du_dx, 'du_dy': du_dy, 'du_dz': du_dz,
        'dv_dx': dv_dx, 'dv_dy': dv_dy, 'dv_dz': dv_dz,
        'dw_dx': dw_dx, 'dw_dy': dw_dy, 'dw_dz': dw_dz,
    }

def compute_dirichlet_bc_vectors(x, y, z, b3d):
    """
    Compute Dirichlet boundary condition vectors from analytical Taylor-Green vortex solution.
    
    For each direction, computes the BC vector by averaging over all boundary slices.
    Since uniform_bc=True uses the same BC for all slices, we average the BC vectors
    computed from all boundary slices to get a representative BC vector.
    
    Returns a dictionary with BC vectors for u, v, w in each direction.
    """
    # Create analytical solution arrays using original Taylor-Green definition
    # Default: A=1, B=1, C=-2, a=b=c=1
    A, B, C, a, b, c = 1.0, 1.0, -2.0, 1.0, 1.0, 1.0
    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")  # (ny, nx, nz)
    to_nz_ny_nx = lambda A: np.moveaxis(A, 2, 0)
    
    u_analytical = to_nz_ny_nx(A * np.cos(a * X) * np.sin(b * Y) * np.sin(c * Z))  # (nz, ny, nx)
    v_analytical = to_nz_ny_nx(B * np.sin(a * X) * np.cos(b * Y) * np.sin(c * Z))  # (nz, ny, nx)
    w_analytical = to_nz_ny_nx(C * np.sin(a * X) * np.sin(b * Y) * np.cos(c * Z))  # (nz, ny, nx)
    
    # For x-direction: average BC vectors from all slices at x=0 and x=2π
    nz, ny, nx = u_analytical.shape
    bc_u_x_list = []
    bc_v_x_list = []
    bc_w_x_list = []
    for k_z in range(nz):
        # At x=0 boundary
        u_slice = u_analytical[k_z, :, 0]  # (ny,)
        bc_u_x_list.append(b3d.x_model.end.BND @ u_slice)
        v_slice = v_analytical[k_z, :, 0]
        bc_v_x_list.append(b3d.x_model.end.BND @ v_slice)
        w_slice = w_analytical[k_z, :, 0]
        bc_w_x_list.append(b3d.x_model.end.BND @ w_slice)
    # Average over all slices
    bc_u_x = np.mean(bc_u_x_list, axis=0)
    bc_v_x = np.mean(bc_v_x_list, axis=0)
    bc_w_x = np.mean(bc_w_x_list, axis=0)
    
    # For y-direction: average BC vectors from all slices at y=0 and y=2π
    bc_u_y_list = []
    bc_v_y_list = []
    bc_w_y_list = []
    for k_z in range(nz):
        # At y=0 boundary
        u_slice = u_analytical[k_z, 0, :]  # (nx,)
        bc_u_y_list.append(b3d.y_model.end.BND @ u_slice)
        v_slice = v_analytical[k_z, 0, :]
        bc_v_y_list.append(b3d.y_model.end.BND @ v_slice)
        w_slice = w_analytical[k_z, 0, :]
        bc_w_y_list.append(b3d.y_model.end.BND @ w_slice)
    # Average over all slices
    bc_u_y = np.mean(bc_u_y_list, axis=0)
    bc_v_y = np.mean(bc_v_y_list, axis=0)
    bc_w_y = np.mean(bc_w_y_list, axis=0)
    
    # For z-direction: average BC vectors from all slices at z=0 and z=2π
    bc_u_z_list = []
    bc_v_z_list = []
    bc_w_z_list = []
    for k_y in range(ny):
        # At z=0 boundary
        u_slice = u_analytical[0, k_y, :]  # (nx,)
        bc_u_z_list.append(b3d.z_model.end.BND @ u_slice)
        v_slice = v_analytical[0, k_y, :]
        bc_v_z_list.append(b3d.z_model.end.BND @ v_slice)
        w_slice = w_analytical[0, k_y, :]
        bc_w_z_list.append(b3d.z_model.end.BND @ w_slice)
    # Average over all slices
    bc_u_z = np.mean(bc_u_z_list, axis=0)
    bc_v_z = np.mean(bc_v_z_list, axis=0)
    bc_w_z = np.mean(bc_w_z_list, axis=0)
    
    return {
        'u_x': bc_u_x,
        'v_x': bc_v_x,
        'w_x': bc_w_x,
        'u_y': bc_u_y,
        'v_y': bc_v_y,
        'w_y': bc_w_y,
        'u_z': bc_u_z,
        'v_z': bc_v_z,
        'w_z': bc_w_z,
    }

# ------------------------------- utils -------------------------------
def sync_gpu():
    if _HAS_CUPY:
        cp.cuda.Stream.null.synchronize()

def l2rel(a, b):
    num = np.linalg.norm((a - b).ravel())
    den = np.linalg.norm(b.ravel()) + 1e-30
    return num / den

def compute_error_metrics(numerical, analytical, name=""):
    """Compute comprehensive error metrics."""
    err = numerical - analytical
    err_flat = err.ravel()
    analytical_flat = analytical.ravel()
    
    # L2 relative error
    l2_rel = l2rel(numerical, analytical)
    
    # L∞ (max) absolute error
    linf_abs = np.max(np.abs(err_flat))
    
    # L∞ relative error (only where analytical is significant)
    # Use a threshold to avoid division by very small numbers
    threshold = np.max(np.abs(analytical_flat)) * 1e-10
    mask = np.abs(analytical_flat) > threshold
    if np.any(mask):
        linf_rel = np.max(np.abs(err_flat[mask]) / (np.abs(analytical_flat[mask]) + 1e-30))
    else:
        linf_rel = linf_abs / (np.max(np.abs(analytical_flat)) + 1e-30)
    
    # Mean absolute error
    mae = np.mean(np.abs(err_flat))
    
    # Mean relative error (only where analytical is significant)
    if np.any(mask):
        mre = np.mean(np.abs(err_flat[mask]) / (np.abs(analytical_flat[mask]) + 1e-30))
    else:
        mre = mae / (np.max(np.abs(analytical_flat)) + 1e-30)
    
    return {
        'l2_rel': l2_rel,
        'linf_abs': linf_abs,
        'linf_rel': linf_rel,
        'mae': mae,
        'mre': mre,
        'name': name
    }


# ------------------------------- core -------------------------------
def build_facade(nx, ny, nz, degx, degy, degz, ordx, ordy, ordz, use_gpu, domain=(0.0, 2.0*np.pi)):
    """
    Build bspf3d facade. Note: ordx/ordy/ordz are derivative orders (passed to make_plan_dx),
    not model orders. Model order defaults to degree-1 which is sufficient for Neumann BC.
    """
    x, y, z = make_axes(nx, ny, nz, domain=domain)
    return bspf3d.from_grids(
        x=x, y=y, z=z,
        degree_x=degx, degree_y=degy, degree_z=degz,
        # Don't pass order_x/order_y/order_z - let it default to degree-1 for Neumann BC support
        correction="spectral",
        use_gpu=use_gpu,
    ), x, y, z

def prepare_plans_3(b3d, *, ox, oy, oz, lamx, lamy, lamz, neumann, uniform_bc, bc_dict=None):
    """
    Prepare derivative plans for x, y, z directions.
    
    Parameters
    ----------
    bc_dict : dict, optional
        Dictionary with BC vectors for each component and direction.
        Keys: 'u_x', 'v_x', 'w_x', 'u_y', 'v_y', 'w_y', 'u_z', 'v_z', 'w_z'
        If uniform_bc=True and bc_dict is provided, creates separate plans for each component.
        Otherwise creates a single plan per direction.
    
    Returns
    -------
    If bc_dict provided: dict with keys 'px_u', 'px_v', 'px_w', 'py_u', 'py_v', 'py_w', 'pz_u', 'pz_v', 'pz_w'
    Otherwise: (px, py, pz, prep_time)
    """
    t0 = time.perf_counter()
    
    if uniform_bc and bc_dict:
        # Create separate plans for each component with their respective BC vectors
        plans = {}
        # X-direction plans
        plans['px_u'] = b3d.make_plan_dx(order=ox, lam=lamx, neumann=neumann, uniform_bc=True, bc=bc_dict.get('u_x', 0.0))
        plans['px_v'] = b3d.make_plan_dx(order=ox, lam=lamx, neumann=neumann, uniform_bc=True, bc=bc_dict.get('v_x', 0.0))
        plans['px_w'] = b3d.make_plan_dx(order=ox, lam=lamx, neumann=neumann, uniform_bc=True, bc=bc_dict.get('w_x', 0.0))
        # Y-direction plans
        plans['py_u'] = b3d.make_plan_dy(order=oy, lam=lamy, neumann=neumann, uniform_bc=True, bc=bc_dict.get('u_y', 0.0))
        plans['py_v'] = b3d.make_plan_dy(order=oy, lam=lamy, neumann=neumann, uniform_bc=True, bc=bc_dict.get('v_y', 0.0))
        plans['py_w'] = b3d.make_plan_dy(order=oy, lam=lamy, neumann=neumann, uniform_bc=True, bc=bc_dict.get('w_y', 0.0))
        # Z-direction plans
        plans['pz_u'] = b3d.make_plan_dz(order=oz, lam=lamz, neumann=neumann, uniform_bc=True, bc=bc_dict.get('u_z', 0.0))
        plans['pz_v'] = b3d.make_plan_dz(order=oz, lam=lamz, neumann=neumann, uniform_bc=True, bc=bc_dict.get('v_z', 0.0))
        plans['pz_w'] = b3d.make_plan_dz(order=oz, lam=lamz, neumann=neumann, uniform_bc=True, bc=bc_dict.get('w_z', 0.0))
        
        if b3d.use_gpu:
            sync_gpu()
        return plans, time.perf_counter() - t0
    else:
        # Single plan per direction (shared by all components)
        bc_x = bc_dict.get('u_x', 0.0) if (uniform_bc and bc_dict) else (0.0 if uniform_bc else None)
        bc_y = bc_dict.get('u_y', 0.0) if (uniform_bc and bc_dict) else (0.0 if uniform_bc else None)
        bc_z = bc_dict.get('u_z', 0.0) if (uniform_bc and bc_dict) else (0.0 if uniform_bc else None)
        
        px = b3d.make_plan_dx(order=ox, lam=lamx, neumann=neumann, uniform_bc=uniform_bc, bc=bc_x)
        py = b3d.make_plan_dy(order=oy, lam=lamy, neumann=neumann, uniform_bc=uniform_bc, bc=bc_y)
        pz = b3d.make_plan_dz(order=oz, lam=lamz, neumann=neumann, uniform_bc=uniform_bc, bc=bc_z)
        
        if b3d.use_gpu:
            sync_gpu()
        return px, py, pz, time.perf_counter() - t0

def compute_gradient_tensor(px_or_plans, py_or_none, pz_or_none, u, v, w, *, neumann, flux_dict, use_gpu):
    """
    Compute the full gradient tensor for velocity field (u, v, w).
    
    Parameters
    ----------
    px_or_plans : _AxisPlan3D or dict
        Either a single x-direction plan, or a dict with component-specific plans
        (keys: 'px_u', 'px_v', 'px_w', 'py_u', 'py_v', 'py_w', 'pz_u', 'pz_v', 'pz_w')
    py_or_none : _AxisPlan3D or None
        y-direction plan (None if px_or_plans is a dict)
    pz_or_none : _AxisPlan3D or None
        z-direction plan (None if px_or_plans is a dict)
    
    Returns
    -------
    Dictionary with keys: 'du_dx', 'du_dy', 'du_dz', 'dv_dx', 'dv_dy', 'dv_dz', 'dw_dx', 'dw_dy', 'dw_dz'
    """
    # Check if we have component-specific plans
    if isinstance(px_or_plans, dict):
        plans = px_or_plans
        # Use component-specific plans
        du_dx = plans['px_u'].apply(u, flux=(0.0, 0.0))
        du_dy = plans['py_u'].apply(u, flux=(0.0, 0.0))
        du_dz = plans['pz_u'].apply(u, flux=(0.0, 0.0))
        
        dv_dx = plans['px_v'].apply(v, flux=(0.0, 0.0))
        dv_dy = plans['py_v'].apply(v, flux=(0.0, 0.0))
        dv_dz = plans['pz_v'].apply(v, flux=(0.0, 0.0))
        
        dw_dx = plans['px_w'].apply(w, flux=(0.0, 0.0))
        dw_dy = plans['py_w'].apply(w, flux=(0.0, 0.0))
        dw_dz = plans['pz_w'].apply(w, flux=(0.0, 0.0))
    else:
        # Use shared plans
        px, py, pz = px_or_plans, py_or_none, pz_or_none
        if neumann:
            flux_u_x = flux_dict.get('u_x', (0.0, 0.0))
            flux_u_y = flux_dict.get('u_y', (0.0, 0.0))
            flux_u_z = flux_dict.get('u_z', (0.0, 0.0))
            flux_v_x = flux_dict.get('v_x', (0.0, 0.0))
            flux_v_y = flux_dict.get('v_y', (0.0, 0.0))
            flux_v_z = flux_dict.get('v_z', (0.0, 0.0))
            flux_w_x = flux_dict.get('w_x', (0.0, 0.0))
            flux_w_y = flux_dict.get('w_y', (0.0, 0.0))
            flux_w_z = flux_dict.get('w_z', (0.0, 0.0))
        else:
            flux_u_x = flux_u_y = flux_u_z = (0.0, 0.0)
            flux_v_x = flux_v_y = flux_v_z = (0.0, 0.0)
            flux_w_x = flux_w_y = flux_w_z = (0.0, 0.0)
        
        # Compute all 9 derivatives
        du_dx = px.apply(u, flux=flux_u_x)
        du_dy = py.apply(u, flux=flux_u_y)
        du_dz = pz.apply(u, flux=flux_u_z)
        
        dv_dx = px.apply(v, flux=flux_v_x)
        dv_dy = py.apply(v, flux=flux_v_y)
        dv_dz = pz.apply(v, flux=flux_v_z)
        
        dw_dx = px.apply(w, flux=flux_w_x)
        dw_dy = py.apply(w, flux=flux_w_y)
        dw_dz = pz.apply(w, flux=flux_w_z)
    
    return {
        'du_dx': du_dx, 'du_dy': du_dy, 'du_dz': du_dz,
        'dv_dx': dv_dx, 'dv_dy': dv_dy, 'dv_dz': dv_dz,
        'dw_dx': dw_dx, 'dw_dy': dw_dy, 'dw_dz': dw_dz,
    }

def eval_gradient_tensor(px_or_plans, py_or_none, pz_or_none, u_list, v_list, w_list, *, neumann, flux_dict, use_gpu):
    """
    Benchmark computing the gradient tensor for multiple velocity fields.
    Returns (elapsed_time, per_derivative_ms)
    """
    # warm-up
    _ = compute_gradient_tensor(px_or_plans, py_or_none, pz_or_none, u_list[0], v_list[0], w_list[0],
                                neumann=neumann, flux_dict=flux_dict, use_gpu=use_gpu)
    if use_gpu:
        sync_gpu()

    t0 = time.perf_counter()
    for u, v, w in zip(u_list, v_list, w_list):
        _ = compute_gradient_tensor(px_or_plans, py_or_none, pz_or_none, u, v, w,
                                    neumann=neumann, flux_dict=flux_dict, use_gpu=use_gpu)
    if use_gpu:
        sync_gpu()
    elapsed = time.perf_counter() - t0
    per_derivative_ms = (1000.0 * elapsed) / (9 * len(u_list))  # 9 derivatives total
    return elapsed, per_derivative_ms


def check_correctness(metrics_cpu, metrics_gpu=None, *, 
                     l2_rel_tol=1e-4, linf_abs_tol=1e-3, linf_rel_tol=1e-3):
    """
    Check correctness of numerical results against analytical solutions.
    
    Parameters
    ----------
    metrics_cpu : dict
        Dictionary of error metrics for CPU results
    metrics_gpu : dict, optional
        Dictionary of error metrics for GPU results
    l2_rel_tol : float, default 1e-4
        Tolerance for L2 relative error
    linf_abs_tol : float, default 1e-3
        Tolerance for L∞ absolute error
    linf_rel_tol : float, default 1e-3
        Tolerance for L∞ relative error
    
    Returns
    -------
    passed : bool
        True if all checks pass, False otherwise
    failures : list
        List of failure messages
    """
    components = ['du_dx', 'du_dy', 'du_dz', 'dv_dx', 'dv_dy', 'dv_dz', 'dw_dx', 'dw_dy', 'dw_dz']
    failures = []
    
    # Check CPU results
    for comp in components:
        m = metrics_cpu[comp]
        if m['l2_rel'] > l2_rel_tol:
            failures.append(f"CPU {comp}: L2 rel error {m['l2_rel']:.6e} > {l2_rel_tol:.6e}")
        if m['linf_abs'] > linf_abs_tol:
            failures.append(f"CPU {comp}: L∞ abs error {m['linf_abs']:.6e} > {linf_abs_tol:.6e}")
        if m['linf_rel'] > linf_rel_tol:
            failures.append(f"CPU {comp}: L∞ rel error {m['linf_rel']:.6e} > {linf_rel_tol:.6e}")
    
    # Check GPU results if provided
    if metrics_gpu is not None:
        for comp in components:
            m = metrics_gpu[comp]
            if m['l2_rel'] > l2_rel_tol:
                failures.append(f"GPU {comp}: L2 rel error {m['l2_rel']:.6e} > {l2_rel_tol:.6e}")
            if m['linf_abs'] > linf_abs_tol:
                failures.append(f"GPU {comp}: L∞ abs error {m['linf_abs']:.6e} > {linf_abs_tol:.6e}")
            if m['linf_rel'] > linf_rel_tol:
                failures.append(f"GPU {comp}: L∞ rel error {m['linf_rel']:.6e} > {linf_rel_tol:.6e}")
    
    return len(failures) == 0, failures


def main():
    ap = argparse.ArgumentParser(description="3D Taylor–Green vortex gradient tensor benchmark (CPU vs GPU).")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nz", type=int, default=128)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--degree-x", type=int, default=7)
    ap.add_argument("--degree-y", type=int, default=7)
    ap.add_argument("--degree-z", type=int, default=7)
    ap.add_argument("--order-x", type=int, default=1)
    ap.add_argument("--order-y", type=int, default=1)
    ap.add_argument("--order-z", type=int, default=1)
    ap.add_argument("--lam-x", type=float, default=0.0)
    ap.add_argument("--lam-y", type=float, default=0.0)
    ap.add_argument("--lam-z", type=float, default=0.0)
    ap.add_argument("--check", action="store_true", help="Enable correctness check")
    ap.add_argument("--l2-rel-tol", type=float, default=1e-4, help="Tolerance for L2 relative error (default: 1e-4)")
    ap.add_argument("--linf-abs-tol", type=float, default=1e-3, help="Tolerance for L∞ absolute error (default: 1e-3)")
    ap.add_argument("--linf-rel-tol", type=float, default=1e-3, help="Tolerance for L∞ relative error (default: 1e-3)")
    ap.add_argument("--fail-on-error", action="store_true", help="Exit with error code if correctness check fails")
    args = ap.parse_args()

    if args.order_x != 1 or args.order_y != 1 or args.order_z != 1:
        raise ValueError("Gradient tensor computation requires first-order derivatives (order=1)")

    degy = args.degree_y if args.degree_y is not None else args.degree_x
    degz = args.degree_z if args.degree_z is not None else args.degree_x

    domain = (0.0, 2.0*np.pi)
    print(f"\nTaylor–Green Vortex Gradient Tensor Benchmark")
    print(f"=" * 80)
    print(f"Grid: nz={args.nz}, ny={args.ny}, nx={args.nx} | reps={args.reps}")
    print(f"Domain: [{domain[0]:.2f}, {domain[1]:.2f}]")
    print(f"Degrees: (dx={args.degree_x}, dy={degy}, dz={degz})")
    print(f"lams: (lx={args.lam_x}, ly={args.lam_y}, lz={args.lam_z})")
    print("-" * 80)

    # ---------- Build Taylor–Green velocity field and analytical gradient tensor ----------
    x, y, z = make_axes(args.nx, args.ny, args.nz, domain=domain)
    u_cpu, v_cpu, w_cpu = taylor_green_velocity(x, y, z)

    # Always compute analytical gradient tensor for accuracy reporting
    grad_tensor_analytical = taylor_green_gradient_tensor(x, y, z)

    # ---------------- CPU run ----------------
    b3d_cpu, x_cpu, y_cpu, z_cpu = build_facade(args.nx, args.ny, args.nz,
                                                args.degree_x, degy, degz,
                                                args.order_x, args.order_y, args.order_z,
                                                use_gpu=False, domain=domain)
    
    # For Dirichlet BC, use uniform_bc=False and manually enforce boundary values
    # The BND @ FT2 will compute boundary conditions from function values,
    # and we'll manually correct the boundary values in the results to match analytical solution
    plans_result = prepare_plans_3(
        b3d_cpu, ox=args.order_x, oy=args.order_y, oz=args.order_z,
        lamx=args.lam_x, lamy=args.lam_y, lamz=args.lam_z,
        neumann=False, uniform_bc=False, bc_dict=None
    )
    
    # Handle both return formats: (plans_dict, prep_time) or (px, py, pz, prep_time)
    if isinstance(plans_result[0], dict):
        plans_cpu, prep_cpu = plans_result
        px_cpu = py_cpu = pz_cpu = None  # Not used
    else:
        px_cpu, py_cpu, pz_cpu, prep_cpu = plans_result
        plans_cpu = None

    u_list_cpu = [u_cpu] * args.reps
    v_list_cpu = [v_cpu] * args.reps
    w_list_cpu = [w_cpu] * args.reps
    
    # Use plans dict if available, otherwise use individual plans
    px_arg = plans_cpu if plans_cpu else px_cpu
    py_arg = None if plans_cpu else py_cpu
    pz_arg = None if plans_cpu else pz_cpu
    
    eval_cpu, per_derivative_cpu = eval_gradient_tensor(
        px_arg, py_arg, pz_arg, u_list_cpu, v_list_cpu, w_list_cpu,
        neumann=False, flux_dict={}, use_gpu=False
    )

    # Always compute CPU accuracy
    grad_tensor_cpu = compute_gradient_tensor(
        px_arg, py_arg, pz_arg, u_cpu, v_cpu, w_cpu,
        neumann=False, flux_dict={}, use_gpu=False
    )
    metrics_cpu = {}
    for key in grad_tensor_analytical.keys():
        metrics_cpu[key] = compute_error_metrics(
            grad_tensor_cpu[key], grad_tensor_analytical[key], key
        )

    # ---------------- GPU run ----------------
    if not _HAS_CUPY:
        print("CuPy not found; skipping GPU run. Install cupy to enable GPU timing.\n")

        print("PERFORMANCE (CPU)")
        print("=" * 80)
        print(f"{'Case':<10} | {'Prep (s)':<12} | {'Eval (s)':<12} | {'Per-derivative (ms)':<20}")
        print("-" * 80)
        print(f"{'CPU':<10} | {prep_cpu:12.4f} | {eval_cpu:12.4f} | {per_derivative_cpu:20.4f}")
        
        # Always print accuracy report
        print("\n" + "="*80)
        print("ACCURACY: NUMERICAL vs ANALYTICAL GRADIENT TENSOR (CPU)")
        print("="*80)
        components = ['du_dx', 'du_dy', 'du_dz', 'dv_dx', 'dv_dy', 'dv_dz', 'dw_dx', 'dw_dy', 'dw_dz']
        print(f"{'Component':<12} | {'L2 rel error':<15} | {'L∞ abs error':<15} | {'L∞ rel error':<15} | {'Mean abs error':<15} | {'Mean rel error':<15}")
        print("-"*80)
        for comp in components:
            m = metrics_cpu[comp]
            print(f"{comp:<12} | {m['l2_rel']:15.6e} | {m['linf_abs']:15.6e} | {m['linf_rel']:15.6e} | {m['mae']:15.6e} | {m['mre']:15.6e}")
        
        # Summary statistics
        l2_errors = [metrics_cpu[comp]['l2_rel'] for comp in components]
        linf_errors = [metrics_cpu[comp]['linf_abs'] for comp in components]
        print("-"*80)
        print(f"{'Summary':<12} | {'Max L2 rel':<15} | {'Max L∞ abs':<15} | {'Mean L2 rel':<15} | {'Mean L∞ abs':<15} | {'':<15}")
        print(f"{'':<12} | {max(l2_errors):15.6e} | {max(linf_errors):15.6e} | {np.mean(l2_errors):15.6e} | {np.mean(linf_errors):15.6e} | {'':<15}")
        
        # Correctness check
        if args.check:
            print("\n" + "="*80)
            print("CORRECTNESS CHECK")
            print("="*80)
            passed, failures = check_correctness(
                metrics_cpu, 
                l2_rel_tol=args.l2_rel_tol,
                linf_abs_tol=args.linf_abs_tol,
                linf_rel_tol=args.linf_rel_tol
            )
            if passed:
                print("✓ PASSED: All error metrics within tolerances")
                print(f"  L2 rel tolerance: {args.l2_rel_tol:.2e}")
                print(f"  L∞ abs tolerance: {args.linf_abs_tol:.2e}")
                print(f"  L∞ rel tolerance: {args.linf_rel_tol:.2e}")
            else:
                print("✗ FAILED: Some error metrics exceed tolerances")
                print(f"  L2 rel tolerance: {args.l2_rel_tol:.2e}")
                print(f"  L∞ abs tolerance: {args.linf_abs_tol:.2e}")
                print(f"  L∞ rel tolerance: {args.linf_rel_tol:.2e}")
                print(f"\nFailures ({len(failures)}):")
                for failure in failures:
                    print(f"  - {failure}")
                if args.fail_on_error:
                    return 1
        return 0

    b3d_gpu, x_gpu, y_gpu, z_gpu = build_facade(args.nx, args.ny, args.nz,
                                                args.degree_x, degy, degz,
                                                args.order_x, args.order_y, args.order_z,
                                                use_gpu=True, domain=domain)
    
    # Compute Dirichlet BC vectors from analytical solution using CPU model (BND is same structure)
    # Then convert to GPU arrays
    bc_dict_cpu = compute_dirichlet_bc_vectors(x, y, z, b3d_cpu)
    bc_dict_gpu = {}
    for key, bc_vec in bc_dict_cpu.items():
        bc_dict_gpu[key] = cp.asarray(bc_vec, dtype=cp.float64)

    plans_result_gpu = prepare_plans_3(
        b3d_gpu, ox=args.order_x, oy=args.order_y, oz=args.order_z,
        lamx=args.lam_x, lamy=args.lam_y, lamz=args.lam_z,
        neumann=False, uniform_bc=True, bc_dict=bc_dict_gpu
    )
    
    # Handle both return formats
    if isinstance(plans_result_gpu[0], dict):
        plans_gpu, prep_gpu = plans_result_gpu
        px_gpu = py_gpu = pz_gpu = None  # Not used
    else:
        px_gpu, py_gpu, pz_gpu, prep_gpu = plans_result_gpu
        plans_gpu = None

    # Always build the velocity field directly on device to exclude transfers
    # Using original Taylor-Green: A=1, B=1, C=-2, a=b=c=1
    A, B, C, a, b, c = 1.0, 1.0, -2.0, 1.0, 1.0, 1.0
    xg = cp.linspace(domain[0], domain[1], args.nx, dtype=cp.float64)
    yg = cp.linspace(domain[0], domain[1], args.ny, dtype=cp.float64)
    zg = cp.linspace(domain[0], domain[1], args.nz, dtype=cp.float64)
    Xg, Yg, Zg = cp.meshgrid(xg, yg, zg, indexing="xy")
    ug = A * cp.cos(a * Xg) * cp.sin(b * Yg) * cp.sin(c * Zg)
    vg = B * cp.sin(a * Xg) * cp.cos(b * Yg) * cp.sin(c * Zg)
    wg = C * cp.sin(a * Xg) * cp.sin(b * Yg) * cp.cos(c * Zg)
    ug = cp.moveaxis(ug, 2, 0)  # (nz, ny, nx)
    vg = cp.moveaxis(vg, 2, 0)
    wg = cp.moveaxis(wg, 2, 0)
    u_list_gpu = [ug] * args.reps
    v_list_gpu = [vg] * args.reps
    w_list_gpu = [wg] * args.reps

    # Use plans dict if available, otherwise use individual plans
    px_arg_gpu = plans_gpu if plans_gpu else px_gpu
    py_arg_gpu = None if plans_gpu else py_gpu
    pz_arg_gpu = None if plans_gpu else pz_gpu

    eval_gpu, per_derivative_gpu = eval_gradient_tensor(
        px_arg_gpu, py_arg_gpu, pz_arg_gpu, u_list_gpu, v_list_gpu, w_list_gpu,
        neumann=False, flux_dict={}, use_gpu=True
    )

    # Always compute GPU accuracy
    grad_tensor_gpu = compute_gradient_tensor(
        px_arg_gpu, py_arg_gpu, pz_arg_gpu, u_list_gpu[0], v_list_gpu[0], w_list_gpu[0],
        neumann=False, flux_dict={}, use_gpu=True
    )
    # Convert to CPU for comparison
    metrics_gpu = {}
    for key in grad_tensor_analytical.keys():
        gpu_val = grad_tensor_gpu[key]
        cpu_val = np.asanyarray(gpu_val.get() if hasattr(gpu_val, "get") else gpu_val)
        metrics_gpu[key] = compute_error_metrics(
            cpu_val, grad_tensor_analytical[key], key
        )

    # ---------------- Report ----------------
    print("\nPERFORMANCE")
    print("=" * 80)
    print(f"{'Case':<10} | {'Prep (s)':<12} | {'Eval (s)':<12} | {'Per-derivative (ms)':<20}")
    print("-" * 80)
    print(f"{'CPU':<10} | {prep_cpu:12.4f} | {eval_cpu:12.4f} | {per_derivative_cpu:20.4f}")
    print(f"{'GPU':<10} | {prep_gpu:12.4f} | {eval_gpu:12.4f} | {per_derivative_gpu:20.4f}")
    print("-" * 80)
    if prep_gpu > 0:
        print(f"Prep speedup (CPU/GPU):     {prep_cpu / prep_gpu:8.2f}×")
    if eval_gpu > 0:
        print(f"Eval speedup (CPU/GPU):     {eval_cpu / eval_gpu:8.2f}×")
        print(f"Per-derivative speedup:     {per_derivative_cpu / per_derivative_gpu:8.2f}×")

    # Always print accuracy report
    print("\n" + "="*80)
    print("ACCURACY: NUMERICAL vs ANALYTICAL GRADIENT TENSOR")
    print("="*80)
    components = ['du_dx', 'du_dy', 'du_dz', 'dv_dx', 'dv_dy', 'dv_dz', 'dw_dx', 'dw_dy', 'dw_dz']
    
    # Detailed accuracy table
    print(f"{'Component':<12} | {'L2 rel (CPU)':<15} | {'L2 rel (GPU)':<15} | {'L∞ abs (CPU)':<15} | {'L∞ abs (GPU)':<15} | {'L∞ rel (CPU)':<15} | {'L∞ rel (GPU)':<15}")
    print("-"*80)
    for comp in components:
        mc = metrics_cpu[comp]
        mg = metrics_gpu[comp]
        print(f"{comp:<12} | {mc['l2_rel']:15.6e} | {mg['l2_rel']:15.6e} | {mc['linf_abs']:15.6e} | {mg['linf_abs']:15.6e} | {mc['linf_rel']:15.6e} | {mg['linf_rel']:15.6e}")
    
    # Summary statistics
    l2_errors_cpu = [metrics_cpu[comp]['l2_rel'] for comp in components]
    l2_errors_gpu = [metrics_gpu[comp]['l2_rel'] for comp in components]
    linf_errors_cpu = [metrics_cpu[comp]['linf_abs'] for comp in components]
    linf_errors_gpu = [metrics_gpu[comp]['linf_abs'] for comp in components]
    print("-"*80)
    print(f"{'Summary':<12} | {'Max L2 rel (CPU)':<15} | {'Max L2 rel (GPU)':<15} | {'Max L∞ abs (CPU)':<15} | {'Max L∞ abs (GPU)':<15} | {'Mean L2 rel (CPU)':<15} | {'Mean L2 rel (GPU)':<15}")
    print(f"{'':<12} | {max(l2_errors_cpu):15.6e} | {max(l2_errors_gpu):15.6e} | {max(linf_errors_cpu):15.6e} | {max(linf_errors_gpu):15.6e} | {np.mean(l2_errors_cpu):15.6e} | {np.mean(l2_errors_gpu):15.6e}")
    
    # Correctness check
    if args.check:
        print("\n" + "="*80)
        print("CORRECTNESS CHECK")
        print("="*80)
        passed, failures = check_correctness(
            metrics_cpu, 
            metrics_gpu=metrics_gpu,
            l2_rel_tol=args.l2_rel_tol,
            linf_abs_tol=args.linf_abs_tol,
            linf_rel_tol=args.linf_rel_tol
        )
        if passed:
            print("✓ PASSED: All error metrics within tolerances")
            print(f"  L2 rel tolerance: {args.l2_rel_tol:.2e}")
            print(f"  L∞ abs tolerance: {args.linf_abs_tol:.2e}")
            print(f"  L∞ rel tolerance: {args.linf_rel_tol:.2e}")
        else:
            print("✗ FAILED: Some error metrics exceed tolerances")
            print(f"  L2 rel tolerance: {args.l2_rel_tol:.2e}")
            print(f"  L∞ abs tolerance: {args.linf_abs_tol:.2e}")
            print(f"  L∞ rel tolerance: {args.linf_rel_tol:.2e}")
            print(f"\nFailures ({len(failures)}):")
            for failure in failures:
                print(f"  - {failure}")
            if args.fail_on_error:
                return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
