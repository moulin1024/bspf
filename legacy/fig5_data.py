from __future__ import annotations
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ============================================================
# 0. 依赖处理 (兼容你的本地库，如果没有则使用替补方案)
# ============================================================
try:
    # 尝试导入你的 BSPF 库
    from bspf1d import bspf1d
    HAS_BSPF = True
except ImportError:
    HAS_BSPF = False
    print("Warning: 'bspf1d' not found. Switching to 4th-order Finite Difference for Solver 1.")

# ============================================================
# 1. 核心参数设置
# ============================================================
L_domain = 100.0      # 域长
nx = 100             # 空间分辨率 (为了验证高精度，建议设为 1024 或更高)
T = 50.0              # 模拟总时间
dt = 2e-2             # 时间步长 (固定步长 RK4 需要较小的 dt)

# --- 理论孤子参数 ---
c_target = 1.0        
A_target = 3.0 * c_target
kappa = np.sqrt(c_target) / 2.0 
x0_phase = -10.0      # 孤子中心初始位置 (在造波机左侧)

# --- 海绵层 ---
sponge_width = 20.0
sponge_strength = 20.0
nu_sponge = 5.0       # 稍微降低一点粘性，避免过度阻尼

# ============================================================
# 2. 网格与初始化
# ============================================================
x = np.linspace(0.0, L_domain, nx)
dx = x[1] - x[0]

# 准备 BSPF 算子 (如果可用)
if HAS_BSPF:
    bf = bspf1d.from_grid(degree=8, x=x)

# 构建海绵权重 w(x)
w_sponge = np.zeros_like(x)
mask_sponge = x > (L_domain - sponge_width)
w_sponge[mask_sponge] = ((x[mask_sponge] - (L_domain - sponge_width)) / sponge_width)**4

# ============================================================
# 3. 理论工具函数
# ============================================================
def compute_theoretical_solution(t, x_val):
    """KdV 单孤子解析解"""
    theta = kappa * (x_val - c_target * t - x0_phase)
    return 3.0 * c_target * (1.0 / np.cosh(theta))**2

def get_boundary_forcing(t):
    """造波机边界动作 u(0,t)"""
    return compute_theoretical_solution(t, 0.0)

# ============================================================
# [关键修改 1] 初始化：热启动 (Hot Start)
# ============================================================
# 不使用全 0，而是使用 t=0 的理论尾巴。
# 这消除了 t=0 时的微小阶跃，防止产生高频背景噪声。
u0 = compute_theoretical_solution(0.0, x)

# ============================================================
# 4. 手写固定步长积分器 (确保不受 atol/rtol 限制)
# ============================================================
def rk4_step(u, t, dt, rhs_func):
    """标准的显式 RK4 积分步"""
    k1 = rhs_func(u, t)
    k2 = rhs_func(u + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs_func(u + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs_func(u + dt * k3, t + dt)
    return u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ============================================================
# 5. 求解器 A: High Order (BSPF or 4th-Order FD)
# ============================================================
print(f"Running Solver 1: High Order Method...")

def fd4_diff(u, dx):
    """
    Standard 4th-order centered finite differences.
    Returns: (du_dx, du_dxx, du_dxxx)
    Handles boundaries with one-sided differences.
    """
    n = len(u)
    if n < 5:
        raise ValueError(f"Need at least 5 points for 4th-order FD, got {n}")
    
    # Check for invalid values
    if np.any(~np.isfinite(u)):
        raise ValueError("Input contains NaN or Inf values")
    
    du_dx = np.zeros_like(u)
    du_dxx = np.zeros_like(u)
    du_dxxx = np.zeros_like(u)
    
    # First derivative: 4th-order centered
    # (u[i-2] - 8*u[i-1] + 8*u[i+1] - u[i+2]) / (12*dx)
    if n > 4:
        du_dx[2:-2] = (u[:-4] - 8*u[1:-3] + 8*u[3:-1] - u[4:]) / (12.0 * dx)
    
    # Boundary: use 4th-order one-sided stencils
    # Left boundary: forward differences
    if n >= 5:
        du_dx[0] = (-25*u[0] + 48*u[1] - 36*u[2] + 16*u[3] - 3*u[4]) / (12*dx)
        du_dx[1] = (-3*u[0] - 10*u[1] + 18*u[2] - 6*u[3] + u[4]) / (12*dx)
    # Right boundary: backward differences
    if n >= 5:
        du_dx[-1] = (25*u[-1] - 48*u[-2] + 36*u[-3] - 16*u[-4] + 3*u[-5]) / (12*dx)
        du_dx[-2] = (3*u[-1] + 10*u[-2] - 18*u[-3] + 6*u[-4] - u[-5]) / (12*dx)
    
    # Second derivative: 4th-order centered
    # (-u[i-2] + 16*u[i-1] - 30*u[i] + 16*u[i+1] - u[i+2]) / (12*dx^2)
    if n > 4:
        du_dxx[2:-2] = (-u[:-4] + 16*u[1:-3] - 30*u[2:-2] + 16*u[3:-1] - u[4:]) / (12.0 * dx**2)
    
    # Boundary: use 4th-order one-sided stencils
    if n >= 5:
        du_dxx[0] = (35*u[0] - 104*u[1] + 114*u[2] - 56*u[3] + 11*u[4]) / (12*dx**2)
        du_dxx[1] = (11*u[0] - 20*u[1] + 6*u[2] + 4*u[3] - u[4]) / (12*dx**2)
        du_dxx[-1] = (35*u[-1] - 104*u[-2] + 114*u[-3] - 56*u[-4] + 11*u[-5]) / (12*dx**2)
        du_dxx[-2] = (11*u[-1] - 20*u[-2] + 6*u[-3] + 4*u[-4] - u[-5]) / (12*dx**2)
    
    # Third derivative: apply 4th-order first derivative to second derivative
    if n > 4:
        du_dxxx[2:-2] = (du_dxx[:-4] - 8*du_dxx[1:-3] + 8*du_dxx[3:-1] - du_dxx[4:]) / (12.0 * dx)
    
    # Boundary: use one-sided differences for third derivative
    if n >= 5:
        du_dxxx[0] = (-25*du_dxx[0] + 48*du_dxx[1] - 36*du_dxx[2] + 16*du_dxx[3] - 3*du_dxx[4]) / (12*dx)
        du_dxxx[1] = (-3*du_dxx[0] - 10*du_dxx[1] + 18*du_dxx[2] - 6*du_dxx[3] + du_dxx[4]) / (12*dx)
        du_dxxx[-1] = (25*du_dxx[-1] - 48*du_dxx[-2] + 36*du_dxx[-3] - 16*du_dxx[-4] + 3*du_dxx[-5]) / (12*dx)
        du_dxxx[-2] = (3*du_dxx[-1] + 10*du_dxx[-2] - 18*du_dxx[-3] + 6*du_dxx[-4] - du_dxx[-5]) / (12*dx)
    
    # Replace any invalid values with 0 (safety check)
    du_dx = np.nan_to_num(du_dx, nan=0.0, posinf=0.0, neginf=0.0)
    du_dxx = np.nan_to_num(du_dxx, nan=0.0, posinf=0.0, neginf=0.0)
    du_dxxx = np.nan_to_num(du_dxxx, nan=0.0, posinf=0.0, neginf=0.0)
    
    return du_dx, du_dxx, du_dxxx

def rhs_func_high_order(u, t):
    # 1. 强制 Dirichlet 边界
    boundary_val = get_boundary_forcing(t)
    u_bc = u.copy()
    u_bc[0] = boundary_val
    
    # 2. 计算导数
    du_dx, du_dxx, du_dxxx, _ = bf.differentiate_1_2_3(u_bc)

    # 3. KdV 方程 + 海绵层
    rhs_kdv = -u_bc * du_dx - du_dxxx
    rhs_sponge = -sponge_strength * (u_bc - 0.0) + nu_sponge * du_dxx
    
    du_dt = (1.0 - w_sponge) * rhs_kdv + w_sponge * rhs_sponge
    
    # 4. 锁定边界导数为 0 (Dirichlet 强约束)
    du_dt[0] = 0.0
    du_dt[-1] = 0.0
    return du_dt

# ============================================================
# 8. Mesh convergence study utilities
# ============================================================
def run_single_resolution(nx_val: int, dt_fixed: float | None = None, T_total: float = T,
                          error_x_max: float = 80.0, save_snapshots: bool = False,
                          outfile_prefix: str = None) -> tuple[float, float]:
    """
    Run the KdV solver for a single spatial resolution `nx_val` with
    fixed time step `dt_fixed` and total time `T_total`.

    Parameters
    ----------
    nx_val : int
        Spatial resolution.
    dt_fixed : float, optional
        Fixed time step. If None, uses CFL-like scaling.
    T_total : float
        Total simulation time.
    error_x_max : float
        Maximum x for error computation region (default: 80.0).
    save_snapshots : bool
        If True, save 1000 time snapshots of the solution to .npy files (default: False).
    outfile_prefix : str, optional
        Prefix for output files when save_snapshots=True. If None, uses f"solution_nx{nx_val}".

    Returns
    -------
    tuple[float, float]
        (relative_Linf_error_at_final_state, wall_time)
    """
    import time
    # --- grid and operators for this resolution ---
    x_local = np.linspace(0.0, L_domain, nx_val)
    dx_local = x_local[1] - x_local[0]

    if HAS_BSPF:
        bf_local = bspf1d.from_grid(degree=8, x=x_local)
    else:
        bf_local = None

    # sponge profile
    w_sponge_local = np.zeros_like(x_local)
    mask_sponge_local = x_local > (L_domain - sponge_width)
    w_sponge_local[mask_sponge_local] = (
        (x_local[mask_sponge_local] - (L_domain - sponge_width)) / sponge_width
    ) ** 4

    # local analytic tools
    def compute_theoretical_solution_local(t, x_val):
        theta = kappa * (x_val - c_target * t - x0_phase)
        return 3.0 * c_target * (1.0 / np.cosh(theta)) ** 2

    def get_boundary_forcing_local(t):
        return compute_theoretical_solution_local(t, 0.0)

    # --- choose (estimated) time step for this resolution ---
    # Reference CFL-like scaling for KdV (dt ~ dx^3 to keep stability/accuracy comparable)
    nx_ref = 100
    dx_ref = L_domain / (nx_ref - 1)
    dt_ref = 5e-2  # reference dt used near nx_ref
    if dt_fixed is None:
        dt_local = dt_ref * (dx_local / dx_ref) ** 3
        print(f"[Mesh convergence] Estimated dt_local = {dt_local:.3e} at nx_local = {nx_val}")
    else:
        dt_local = dt_fixed

    # initial condition (hot start)
    u_curr = compute_theoretical_solution_local(0.0, x_local)
    curr_time = 0.0

    # Spatial mask for error region: x in [0, error_x_max]
    mask_error_region = x_local <= error_x_max

    # Calculate number of time steps
    steps = int(T_total / dt_local)

    # Storage for solution snapshots (if save_snapshots=True)
    solution_snapshots = []  # List of solution arrays
    exact_snapshots = []      # List of exact solution arrays
    time_snapshots = []  # List of time values
    snapshot_interval = None
    if save_snapshots:
        snapshot_interval = max(1, steps // 100)  # Save 1000 snapshots evenly distributed
        print(f"[Snapshot saving] Will save 100 snapshots (every {snapshot_interval} steps)")
        # Save initial condition
        solution_snapshots.append(u_curr.copy())
        exact_snapshots.append(compute_theoretical_solution_local(0.0, x_local))
        time_snapshots.append(curr_time)

    # local RHS using closure over local grid / operators
    def rhs_func_high_order_local(u, t):
        u_bc = u.copy()
        u_bc[0] = get_boundary_forcing_local(t)

        if HAS_BSPF and bf_local is not None:
            du_dx, du_dxx, du_dxxx, _ = bf_local.differentiate_1_2_3(u_bc)
        else:
            du_dx, du_dxx, du_dxxx = fd4_diff(u_bc, dx_local)

        rhs_kdv = -u_bc * du_dx - du_dxxx
        rhs_sponge = -sponge_strength * (u_bc - 0.0) + nu_sponge * du_dxx

        du_dt = (1.0 - w_sponge_local) * rhs_kdv + w_sponge_local * rhs_sponge
        du_dt[0] = 0.0
        du_dt[-1] = 0.0
        return du_dt

    # Start timing
    start_time = time.time()
    
    step_iter = (
        tqdm(range(steps), desc=f"BSPF nx={nx_val}", leave=False)
        if HAS_TQDM
        else range(steps)
    )
    for step_idx in step_iter:
        u_curr = rk4_step(u_curr, curr_time, dt_local, rhs_func_high_order_local)
        curr_time += dt_local
        u_curr[0] = get_boundary_forcing_local(curr_time)

        # Store solution snapshots if requested
        if save_snapshots and step_idx % snapshot_interval == 0:
            solution_snapshots.append(u_curr.copy())
            exact_snapshots.append(compute_theoretical_solution_local(curr_time, x_local))
            time_snapshots.append(curr_time)
    
    # End timing
    wall_time = time.time() - start_time

    # Save final state if saving snapshots
    if save_snapshots:
        # Only save if it's not already saved (i.e., if final step wasn't a snapshot step)
        if len(solution_snapshots) == 0 or time_snapshots[-1] < curr_time - dt_local * 0.5:
            solution_snapshots.append(u_curr.copy())
            exact_snapshots.append(compute_theoretical_solution_local(curr_time, x_local))
            time_snapshots.append(curr_time)

    # ============================================================
    # Compute L∞ error at final state only
    # ============================================================
    u_exact_final = compute_theoretical_solution_local(curr_time, x_local)
    err_final = np.abs(u_curr - u_exact_final)
    err_final_region = err_final[mask_error_region]
    
    # Absolute L∞ error
    linf_abs = np.max(err_final_region)
    
    # Relative L∞ error: normalize by maximum absolute value of exact solution in error region
    u_exact_final_region = u_exact_final[mask_error_region]
    linf_exact = np.max(np.abs(u_exact_final_region))
    linf_rel = linf_abs / linf_exact if linf_exact > 0 else np.inf
    
    print(f"[Error report BSPF] Final state (t = {curr_time:.2f}) - Abs. L∞ error = {linf_abs:.6e}, Rel. L∞ error = {linf_rel:.6e}, Wall time = {wall_time:.2f} s")
    
    # Save solution snapshots to .npy files if requested
    if save_snapshots and solution_snapshots:
        if outfile_prefix is None:
            outfile_prefix = f"solution_nx{nx_val}"
        
        # Convert lists to numpy arrays
        solution_array = np.array(solution_snapshots)  # Shape: (n_snapshots, nx_val)
        exact_array = np.array(exact_snapshots)        # Shape: (n_snapshots, nx_val)
        time_array = np.array(time_snapshots)  # Shape: (n_snapshots,)
        
        # Save files
        solution_file = f"data/{outfile_prefix}_u.npy"
        exact_file = f"data/{outfile_prefix}_exact_u.npy"
        time_file = f"data/{outfile_prefix}_t.npy"
        x_file = f"data/{outfile_prefix}_x.npy"
        
        np.save(solution_file, solution_array)
        np.save(exact_file, exact_array)
        np.save(time_file, time_array)
        np.save(x_file, x_local)
        
        print(f"[Snapshot saving] Saved {len(solution_snapshots)} snapshots:")
        print(f"  Solution: {solution_file} (shape: {solution_array.shape})")
        print(f"  Exact:    {exact_file} (shape: {exact_array.shape})")
        print(f"  Times:    {time_file} (shape: {time_array.shape})")
        print(f"  Grid:     {x_file} (shape: {x_local.shape})")
    
    return linf_rel, wall_time


def run_single_resolution_fd(nx_val: int, dt_fixed: float | None = None, T_total: float = T,
                             error_x_max: float = 80.0, save_snapshots: bool = False,
                             outfile_prefix: str = None) -> tuple[float, float]:
    """
    Run the KdV solver using 4th-order finite differences for a single spatial resolution.
    
    Parameters
    ----------
    nx_val : int
        Spatial resolution.
    dt_fixed : float, optional
        Fixed time step. If None, uses CFL-like scaling.
    T_total : float
        Total simulation time.
    error_x_max : float
        Maximum x for error computation region (default: 80.0).
    save_snapshots : bool
        If True, save 1000 time snapshots of the solution to .npy files (default: False).
    outfile_prefix : str, optional
        Prefix for output files when save_snapshots=True. If None, uses f"solution_nx{nx_val}".

    Returns
    -------
    tuple[float, float]
        (relative_Linf_error_at_final_state, wall_time)
    """
    import time
    
    # --- grid and operators for this resolution ---
    x_local = np.linspace(0.0, L_domain, nx_val)
    dx_local = x_local[1] - x_local[0]

    # sponge profile
    w_sponge_local = np.zeros_like(x_local)
    mask_sponge_local = x_local > (L_domain - sponge_width)
    w_sponge_local[mask_sponge_local] = (
        (x_local[mask_sponge_local] - (L_domain - sponge_width)) / sponge_width
    ) ** 4

    # local analytic tools
    def compute_theoretical_solution_local(t, x_val):
        theta = kappa * (x_val - c_target * t - x0_phase)
        return 3.0 * c_target * (1.0 / np.cosh(theta)) ** 2

    def get_boundary_forcing_local(t):
        return compute_theoretical_solution_local(t, 0.0)

    # --- choose (estimated) time step for this resolution ---
    # Reference CFL-like scaling for KdV (dt ~ dx^3 to keep stability/accuracy comparable)
    nx_ref = 100
    dx_ref = L_domain / (nx_ref - 1)
    dt_ref = 5e-2  # reference dt for FD4 (smaller than BSPF)
    if dt_fixed is None:
        dt_local = dt_ref * (dx_local / dx_ref) ** 3
        print(f"[Mesh convergence FD] Estimated dt_local = {dt_local:.3e} at nx_local = {nx_val}")
    else:
        dt_local = dt_fixed

    # initial condition (hot start)
    u_curr = compute_theoretical_solution_local(0.0, x_local)
    curr_time = 0.0

    # Spatial mask for error region: x in [0, error_x_max]
    mask_error_region = x_local <= error_x_max

    # Calculate number of time steps
    steps = int(T_total / dt_local)

    # Storage for solution snapshots (if save_snapshots=True)
    solution_snapshots = []  # List of solution arrays
    time_snapshots = []  # List of time values
    snapshot_interval = None
    if save_snapshots:
        snapshot_interval = max(1, steps // 1000)  # Save 1000 snapshots evenly distributed
        print(f"[Snapshot saving] Will save 1000 snapshots (every {snapshot_interval} steps)")
        # Save initial condition
        solution_snapshots.append(u_curr.copy())
        time_snapshots.append(curr_time)

    # local RHS using finite differences
    def rhs_func_fd_local(u, t):
        u_bc = u.copy()
        u_bc[0] = get_boundary_forcing_local(t)
        
        try:
            du_dx, du_dxx, du_dxxx = fd4_diff(u_bc, dx_local)
        except (ValueError, RuntimeError) as e:
            raise RuntimeError(f"FD4 differentiation failed at t={t:.6f}: {e}")
        
        rhs_kdv = -u_bc * du_dx - du_dxxx
        rhs_sponge = -sponge_strength * (u_bc - 0.0) + nu_sponge * du_dxx
        
        du_dt = (1.0 - w_sponge_local) * rhs_kdv + w_sponge_local * rhs_sponge
        du_dt[0] = 0.0
        du_dt[-1] = 0.0
        
        # Check result for invalid values
        if np.any(~np.isfinite(du_dt)):
            du_dt = np.nan_to_num(du_dt, nan=0.0, posinf=0.0, neginf=0.0)
        
        return du_dt

    # Start timing
    start_time = time.time()
    
    step_iter = (
        tqdm(range(steps), desc=f"FD4 nx={nx_val}", leave=False)
        if HAS_TQDM
        else range(steps)
    )
    for step_idx in step_iter:
        try:
            u_curr = rk4_step(u_curr, curr_time, dt_local, rhs_func_fd_local)
            curr_time += dt_local
            u_curr[0] = get_boundary_forcing_local(curr_time)
            
            # Check for instability
            if np.any(~np.isfinite(u_curr)):
                print(f"\nWarning: Solution became unstable at step {step_idx}, t={curr_time:.6f}")
                break
            
            # Store solution snapshots if requested (exact solution saved for plotting, not error computation)
            if save_snapshots and step_idx % snapshot_interval == 0:
                solution_snapshots.append(u_curr.copy())
                time_snapshots.append(curr_time)
        except RuntimeError as e:
            print(f"\nError in simulation: {e}")
            break
    
    # End timing
    wall_time = time.time() - start_time

    # Save final state if saving snapshots
    if save_snapshots:
        # Only save if it's not already saved (i.e., if final step wasn't a snapshot step)
        if len(solution_snapshots) == 0 or time_snapshots[-1] < curr_time - dt_local * 0.5:
            solution_snapshots.append(u_curr.copy())
            time_snapshots.append(curr_time)

    # ============================================================
    # Compute L∞ error at final state only
    # ============================================================
    u_exact_final = compute_theoretical_solution_local(curr_time, x_local)
    err_final = np.abs(u_curr - u_exact_final)
    err_final_region = err_final[mask_error_region]
    
    # Absolute L∞ error
    linf_abs = np.max(err_final_region)
    
    # Relative L∞ error: normalize by maximum absolute value of exact solution in error region
    u_exact_final_region = u_exact_final[mask_error_region]
    linf_exact = np.max(np.abs(u_exact_final_region))
    linf_rel = linf_abs / linf_exact if linf_exact > 0 else np.inf
    
    print(f"[Error report FD] Final state (t = {curr_time:.2f}) - Abs. L∞ error = {linf_abs:.6e}, Rel. L∞ error = {linf_rel:.6e}, Wall time = {wall_time:.2f} s")
    
    # Save solution snapshots to .npy files if requested
    if save_snapshots and solution_snapshots:
        if outfile_prefix is None:
            outfile_prefix = f"solution_fd_nx{nx_val}"
        
        # Convert lists to numpy arrays
        solution_array = np.array(solution_snapshots)  # Shape: (n_snapshots, nx_val)
        time_array = np.array(time_snapshots)  # Shape: (n_snapshots,)
        
        # Save files
        solution_file = f"{outfile_prefix}_u.npy"
        time_file = f"{outfile_prefix}_t.npy"
        x_file = f"{outfile_prefix}_x.npy"
        
        np.save(solution_file, solution_array)
        np.save(time_file, time_array)
        np.save(x_file, x_local)
        
        print(f"[Snapshot saving] Saved {len(solution_snapshots)} snapshots:")
        print(f"  Solution: {solution_file} (shape: {solution_array.shape})")
        print(f"  Times:    {time_file} (shape: {time_array.shape})")
        print(f"  Grid:     {x_file} (shape: {x_local.shape})")
    
    return linf_rel, wall_time


def run_mesh_convergence(
    nx_min: int = 100,
    nx_max: int = 1000,
    nx_step: int = 100,
    dt_fixed: float | None = None,
    outfile_prefix: str = "mesh_convergence",
    method: str = "both",
):
    """
    Perform a mesh convergence study by varying nx from nx_min to nx_max
    (inclusive, with step nx_step) at fixed dt, and write final state errors and wall times to a .npz file.

    The error is computed at the final time state only, over the spatial region x ∈ [0, error_x_max].

    Parameters
    ----------
    method : str
        Method to use: "bspf", "fd", or "both" (default: "both")
    
    Saved file:
      - f"{outfile_prefix}.npz": NumPy archive containing:
        - 'nx': array of nx values
        - 'errors_linf_bspf': relative L∞ errors for BSPF (if method includes "bspf")
        - 'walltime_bspf': wall times for BSPF (if method includes "bspf")
        - 'errors_linf_fd': relative L∞ errors for FD4 (if method includes "fd")
        - 'walltime_fd': wall times for FD4 (if method includes "fd")
    """
    import time
    
    nx_values = np.arange(nx_min, nx_max + 1, nx_step, dtype=int)
    
    errors_linf_bspf = []
    errors_linf_fd = []
    walltime_bspf = []
    walltime_fd = []

    for nx_val in nx_values:
        print(f"\n[Mesh convergence] Running resolution nx = {nx_val}")
        
        if method in ["bspf", "both"]:
            print(f"[Mesh convergence] Running BSPF method...")
            linf_err_bspf, wall_time_bspf = run_single_resolution(nx_val, dt_fixed=dt_fixed, T_total=T)
            errors_linf_bspf.append(linf_err_bspf)
            walltime_bspf.append(wall_time_bspf)
            print(f"[Mesh convergence] BSPF: nx = {nx_val}, Rel. L∞ error = {linf_err_bspf:.6e}, Wall time = {wall_time_bspf:.2f} s")
        
        if method in ["fd", "both"]:
            print(f"[Mesh convergence] Running FD4 method...")
            linf_err_fd, wall_time_fd = run_single_resolution_fd(nx_val, dt_fixed=dt_fixed, T_total=T)
            errors_linf_fd.append(linf_err_fd)
            walltime_fd.append(wall_time_fd)
            print(f"[Mesh convergence] FD4: nx = {nx_val}, Rel. L∞ error = {linf_err_fd:.6e}, Wall time = {wall_time_fd:.2f} s")

    nx_values = np.asarray(nx_values, dtype=int)
    
    # Prepare data dictionary for npz file
    data_dict = {'nx': nx_values}
    
    if method in ["bspf", "both"]:
        errors_linf_bspf = np.asarray(errors_linf_bspf, dtype=float)
        walltime_bspf = np.asarray(walltime_bspf, dtype=float)
        data_dict['errors_linf_bspf'] = errors_linf_bspf
        data_dict['walltime_bspf'] = walltime_bspf
    
    if method in ["fd", "both"]:
        errors_linf_fd = np.asarray(errors_linf_fd, dtype=float)
        walltime_fd = np.asarray(walltime_fd, dtype=float)
        data_dict['errors_linf_fd'] = errors_linf_fd
        data_dict['walltime_fd'] = walltime_fd
    
    # Save to npz file
    npz_file = f"{outfile_prefix}.npz"
    np.savez(npz_file, **data_dict)
    
    print(f"\nMesh convergence data saved to '{npz_file}':")
    print(f"  Keys: {list(data_dict.keys())}")
    for key, value in data_dict.items():
        print(f"    {key}: shape={value.shape}, dtype={value.dtype}")


def generate_exact_solution(nx_val: int = 1400, T_total: float = 50.0, 
                            outfile_prefix: str = None):
    """
    Generate and save exact solution for comparison.
    
    Parameters
    ----------
    nx_val : int
        Spatial resolution (default: 1400).
    T_total : float
        Total simulation time (default: 50.0).
    outfile_prefix : str, optional
        Prefix for output files. If None, uses f"exact_solution_nx{nx_val}".
    """
    # Create grid
    x_local = np.linspace(0.0, L_domain, nx_val)
    
    # Local analytic function
    def compute_theoretical_solution_local(t, x_val):
        theta = kappa * (x_val - c_target * t - x0_phase)
        return 3.0 * c_target * (1.0 / np.cosh(theta)) ** 2
    
    # Use same time step as would be used in run_single_resolution
    dx_local = x_local[1] - x_local[0]
    nx_ref = 100
    dx_ref = L_domain / (nx_ref - 1)
    dt_ref = 5e-2
    dt_local = dt_ref * (dx_local / dx_ref) ** 3
    
    # Calculate number of time steps
    steps = int(T_total / dt_local)
    
    # Generate same snapshot intervals as run_single_resolution
    snapshot_interval = max(1, steps // 1000)  # Save 1000 snapshots
    
    # Storage for exact solution snapshots
    exact_snapshots = []
    time_snapshots = []
    
    # Save initial condition
    exact_snapshots.append(compute_theoretical_solution_local(0.0, x_local))
    time_snapshots.append(0.0)
    
    # Generate snapshots at same intervals as numerical solution
    curr_time = 0.0
    for step_idx in range(steps):
        curr_time += dt_local
        
        if step_idx % snapshot_interval == 0:
            exact_snapshots.append(compute_theoretical_solution_local(curr_time, x_local))
            time_snapshots.append(curr_time)
    
    # Save final state if not already saved
    if len(time_snapshots) == 0 or time_snapshots[-1] < curr_time - dt_local * 0.5:
        exact_snapshots.append(compute_theoretical_solution_local(curr_time, x_local))
        time_snapshots.append(curr_time)
    
    # Convert to numpy arrays
    exact_array = np.array(exact_snapshots)  # Shape: (n_snapshots, nx_val)
    time_array = np.array(time_snapshots)    # Shape: (n_snapshots,)
    
    # Save files
    if outfile_prefix is None:
        outfile_prefix = f"exact_solution_nx{nx_val}"
    
    exact_file = f"data/{outfile_prefix}_u.npy"
    time_file = f"data/{outfile_prefix}_t.npy"
    x_file = f"data/{outfile_prefix}_x.npy"
    
    np.save(exact_file, exact_array)
    np.save(time_file, time_array)
    np.save(x_file, x_local)
    
    print(f"[Exact solution] Saved exact solution for nx={nx_val}:")
    print(f"  Exact:    {exact_file} (shape: {exact_array.shape})")
    print(f"  Times:    {time_file} (shape: {time_array.shape})")
    print(f"  Grid:     {x_file} (shape: {x_local.shape})")
    print(f"  Total snapshots: {len(exact_snapshots)}")
    print(f"  Time range: [0.0, {curr_time:.2f}]")


if __name__ == "__main__":
    # Run mesh convergence study when executed as a script
    # run_mesh_convergence(nx_min=100, nx_max=400, nx_step=100)
    linf_err = run_single_resolution(nx_val=1400, T_total=50.0, save_snapshots=True, outfile_prefix="solution_nx1400")
    # print(f"\nFinal error report:")
    # print(f"  Relative L∞ error = {linf_err:.6e}")
    
