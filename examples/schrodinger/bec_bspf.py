import numpy as np
import matplotlib.pyplot as plt
import time
from bspf import bspf2d
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Simple progress bar fallback
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 100)
            self.n = 0
        def update(self, n=1):
            self.n += n
        def set_postfix(self, **kwargs):
            pass
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

# ===========================
# 参数设置
# ===========================
nx, ny = 256, 256          # 网格点数
Lx, Ly = 20.0, 20.0        # 物理尺寸
dx, dy = Lx / nx, Ly / ny

g = 10.0                    # 非线性系数
omega_trap = 1.0           # trap 频率（无量纲）
Omega = 0.9                # 旋转角速度（0 < Omega < omega_trap，一般）
dt_imag = 0.001           # initial imaginary-time 步长
tau_max = 6              # maximum imaginary-time to evolve
dt_real = 0.001            # real-time 步长
n_real_steps = 5000        # real-time 总步数
output_every = 1000         # 每隔多少步画一次图
degree = 5                 # B-spline 阶数

# Adaptive timestep parameters
rtol = 1e-6                # relative tolerance
atol = 1e-8                # absolute tolerance
safety = 0.9               # safety factor for timestep adjustment
dt_min = 1e-6              # minimum timestep
dt_max = 0.01              # maximum timestep
dt_imag = min(dt_imag, dt_max)  # ensure initial dt is within bounds

np.random.seed(0)

# ===========================
# 网格与势
# ===========================
x = (np.arange(nx) - nx // 2) * dx
y = (np.arange(ny) - ny // 2) * dy
X, Y = np.meshgrid(x, y, indexing='ij')

# 谐振 trap 势
V_trap = 0.5 * omega_trap**2 * (X**2 + Y**2)

# ===========================
# 初始条件：trap 内随机小扰动
# ===========================
sigma = Lx / 5.0
psi = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
psi = psi * (1.0 + 0.1 * (np.random.rand(nx, ny) - 0.5))
psi = psi.astype(np.complex128)

# ===========================
# 计时统计
# ===========================
timing_stats = {
    'laplacian': 0.0,
    'Lz_psi': 0.0,
    'H_psi': 0.0,
    'rk23_step': 0.0,
    'norm': 0.0,
    'rhs_computation': 0.0,
    'total_step': 0.0
}
timing_counts = {
    'laplacian': 0,
    'Lz_psi': 0,
    'H_psi': 0,
    'rk23_step': 0,
    'norm': 0,
    'rhs_computation': 0,
    'total_step': 0
}

# 计算初始粒子数
t_norm_start = time.perf_counter()
N_target = np.sum(np.abs(psi)**2) * dx * dy
t_norm_end = time.perf_counter()
timing_stats['norm'] += (t_norm_end - t_norm_start)
timing_counts['norm'] += 1

# ===========================
# 创建 BSPF 算子
# ===========================
print("Creating BSPF operator...")
bspf_op = bspf2d.from_grids(
    x=x, y=y,
    degree_x=degree,
    degree_y=degree,
    use_clustering_x=True,
    use_clustering_y=True,
    correction='spectral'
)

def print_timing_stats():
    """Print timing statistics"""
    print("\n" + "="*60)
    print("TIMING STATISTICS (non-overlapping measurements)")
    print("="*60)
    
    # Separate total_step from component times
    total_step_time = timing_stats.get('total_step', 0.0)
    component_keys = [k for k in timing_stats.keys() if k != 'total_step']
    
    # Sum of component times
    component_sum = sum(timing_stats[k] for k in component_keys if timing_counts.get(k, 0) > 0)
    
    # Print component times
    if component_sum > 0:
        for key in sorted(component_keys):
            if timing_counts.get(key, 0) > 0:
                avg_time = timing_stats[key] / timing_counts[key]
                percentage = 100.0 * timing_stats[key] / total_step_time if total_step_time > 0 else 0.0
                print(f"{key:20s}: {timing_stats[key]:10.4f}s total, "
                      f"{avg_time:8.4f}s avg, {percentage:6.2f}% "
                      f"({timing_counts[key]} calls)")
        
        print("-" * 60)
        print(f"{'Sum of components':20s}: {component_sum:10.4f}s")
    
    # Print total step time
    if total_step_time > 0 and timing_counts.get('total_step', 0) > 0:
        avg_total = total_step_time / timing_counts['total_step']
        print(f"{'total_step (all ops)':20s}: {total_step_time:10.4f}s total, "
              f"{avg_total:8.4f}s avg "
              f"({timing_counts['total_step']} calls)")
        
        # Show overhead (difference between total and sum of components)
        overhead = total_step_time - component_sum
        if overhead > 0:
            overhead_pct = 100.0 * overhead / total_step_time
            print(f"{'Overhead (BCs, etc)':20s}: {overhead:10.4f}s ({overhead_pct:5.2f}%)")
    
    print("="*60 + "\n")

# ===========================
# 1. imaginary-time 演化：求带旋转的稳态（涡旋晶格）
# ===========================
print("Imaginary-time evolution (finding rotating trapped state)...")

# Enforce Dirichlet BCs on initial condition
psi[0, :] = 0.0      # left boundary
psi[-1, :] = 0.0     # right boundary
psi[:, 0] = 0.0      # bottom boundary
psi[:, -1] = 0.0     # top boundary

# Helper function to compute RHS (inline, for reuse in RK23 stages)
def compute_rhs_imag_inline(psi_in):
    # Enforce BCs before differentiation
    psi_in[0, :] = 0.0; psi_in[-1, :] = 0.0; psi_in[:, 0] = 0.0; psi_in[:, -1] = 0.0
    
    # Compute all derivatives together using differentiate_1_2 (more efficient)
    t_lap_start = time.perf_counter()
    psi_real = np.real(psi_in)
    psi_imag = np.imag(psi_in)
    
    # Use differentiate_1_2 to compute all derivatives at once
    dpsi_dx_real, dpsi_dy_real, d2psi_dx2_real, d2psi_dy2_real = bspf_op.differentiate_1_2(psi_real)
    dpsi_dx_imag, dpsi_dy_imag, d2psi_dx2_imag, d2psi_dy2_imag = bspf_op.differentiate_1_2(psi_imag)
    
    # Compute Laplacian from second derivatives
    lap = (d2psi_dx2_real + 1j * d2psi_dx2_imag) + (d2psi_dy2_real + 1j * d2psi_dy2_imag)
    t_lap_end = time.perf_counter()
    timing_stats['laplacian'] += (t_lap_end - t_lap_start)
    timing_counts['laplacian'] += 1
    
    # Compute Lz_psi using first derivatives (already computed above)
    t_Lz_start = time.perf_counter()
    dpsi_dx = dpsi_dx_real + 1j * dpsi_dx_imag
    dpsi_dy = dpsi_dy_real + 1j * dpsi_dy_imag
    Lzpsi = -1j * (X * dpsi_dy - Y * dpsi_dx)
    t_Lz_end = time.perf_counter()
    timing_stats['Lz_psi'] += (t_Lz_end - t_Lz_start)
    timing_counts['Lz_psi'] += 1
    
    # Combine to get H_psi (only the combination step, not including laplacian/Lz)
    t_H_start = time.perf_counter()
    Hpsi = -0.5 * lap + V_trap * psi_in + g * np.abs(psi_in)**2 * psi_in - Omega * Lzpsi
    t_H_end = time.perf_counter()
    timing_stats['H_psi'] += (t_H_end - t_H_start)
    timing_counts['H_psi'] += 1
    
    # RHS for imaginary-time: ∂τ ψ = -H ψ
    t_rhs_start = time.perf_counter()
    rhs = -Hpsi
    t_rhs_end = time.perf_counter()
    timing_stats['rhs_computation'] += (t_rhs_end - t_rhs_start)
    timing_counts['rhs_computation'] += 1
    
    return rhs

# Adaptive RK23 time stepping
tau = 0.0  # current imaginary-time
step = 0
n_accepted = 0
n_rejected = 0
N_now = N_target  # Initialize for printing

# Create progress bar
with tqdm(total=tau_max, desc="Imaginary-time", unit="tau", 
          bar_format='{l_bar}{bar}| {n:.4f}/{total:.4f} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
    while tau < tau_max:
        step += 1
        t_step_start = time.perf_counter()
        
        # RK23 stages (Bogacki-Shampine method)
        t_rk23_start = time.perf_counter()
        
        # Stage 1: k1 = rhs(psi)
        k1 = compute_rhs_imag_inline(psi.copy())
        
        # Stage 2: k2 = rhs(psi + 0.5*dt*k1)
        psi_stage2 = psi + 0.5 * dt_imag * k1
        psi_stage2[0, :] = 0.0; psi_stage2[-1, :] = 0.0; psi_stage2[:, 0] = 0.0; psi_stage2[:, -1] = 0.0
        k2 = compute_rhs_imag_inline(psi_stage2)
        
        # Stage 3: k3 = rhs(psi + 0.75*dt*k2)
        psi_stage3 = psi + 0.75 * dt_imag * k2
        psi_stage3[0, :] = 0.0; psi_stage3[-1, :] = 0.0; psi_stage3[:, 0] = 0.0; psi_stage3[:, -1] = 0.0
        k3 = compute_rhs_imag_inline(psi_stage3)
        
        # Stage 4: k4 = rhs(psi + (2/9)*dt*k1 + (1/3)*dt*k2 + (4/9)*dt*k3)
        psi_stage4 = psi + (2.0/9.0) * dt_imag * k1 + (1.0/3.0) * dt_imag * k2 + (4.0/9.0) * dt_imag * k3
        psi_stage4[0, :] = 0.0; psi_stage4[-1, :] = 0.0; psi_stage4[:, 0] = 0.0; psi_stage4[:, -1] = 0.0
        k4 = compute_rhs_imag_inline(psi_stage4)
        
        # 3rd order solution (Bogacki-Shampine)
        psi_3rd = psi + dt_imag * ((2.0/9.0) * k1 + (1.0/3.0) * k2 + (4.0/9.0) * k3)
        
        # 2nd order embedded solution (for error estimation)
        psi_2nd = psi + dt_imag * ((7.0/24.0) * k1 + (1.0/4.0) * k2 + (1.0/3.0) * k3 + (1.0/8.0) * k4)
        
        # Error estimate: difference between 3rd and 2nd order solutions
        error = np.abs(psi_3rd - psi_2nd)
        # Scale by tolerance: error_scale = error / (rtol * |psi| + atol)
        scale = rtol * np.abs(psi_3rd) + atol
        error_norm = np.sqrt(np.mean((error / scale)**2))
        
        # Accept or reject step
        if error_norm <= 1.0:
            # Accept step
            psi = psi_3rd.copy()
            tau += dt_imag
            n_accepted += 1
            
            # Enforce Dirichlet BCs: ψ = 0 at boundaries
            psi[0, :] = 0.0      # left boundary
            psi[-1, :] = 0.0     # right boundary
            psi[:, 0] = 0.0      # bottom boundary
            psi[:, -1] = 0.0     # top boundary
            
            # Normalization
            t_norm_start = time.perf_counter()
            N_now = np.sum(np.abs(psi)**2) * dx * dy
            psi *= np.sqrt(N_target / (N_now + 1e-16))
            
            # Enforce Dirichlet BCs after normalization
            psi[0, :] = 0.0      # left boundary
            psi[-1, :] = 0.0     # right boundary
            psi[:, 0] = 0.0      # bottom boundary
            psi[:, -1] = 0.0     # top boundary
            t_norm_end = time.perf_counter()
            timing_stats['norm'] += (t_norm_end - t_norm_start)
            timing_counts['norm'] += 1
        else:
            # Reject step
            n_rejected += 1
        
        # Adjust timestep for next iteration
        if error_norm > 0:
            # Optimal timestep scaling factor
            factor = safety * (1.0 / error_norm) ** (1.0 / 3.0)  # 3rd order method
            dt_imag = np.clip(factor * dt_imag, dt_min, dt_max)
        else:
            # If error is zero, increase timestep
            dt_imag = min(dt_imag * 1.5, dt_max)
        
        # Ensure we don't overshoot tau_max
        if tau + dt_imag > tau_max:
            dt_imag = tau_max - tau
        
        t_rk23_end = time.perf_counter()
        timing_stats['rk23_step'] += (t_rk23_end - t_rk23_start)
        timing_counts['rk23_step'] += 1
        
        t_step_end = time.perf_counter()
        timing_stats['total_step'] += (t_step_end - t_step_start)
        timing_counts['total_step'] += 1

        # Update progress bar
        pbar.n = tau
        pbar.refresh()
        
        # Update progress bar postfix with current information
        if n_accepted > 0:
            pbar.set_postfix(
                step=step,
                dt=f"{dt_imag:.2e}",
                N=f"{N_now:.4e}",
                err=f"{error_norm:.2e}",
                acc=n_accepted,
                rej=n_rejected
            )
        else:
            pbar.set_postfix(
                step=step,
                dt=f"{dt_imag:.2e}",
                err=f"{error_norm:.2e}",
                acc=n_accepted,
                rej=n_rejected
            )
        
        if step % 1000 == 0:
            print_timing_stats()
        
        # Safety check: if dt becomes too small, something is wrong
        if dt_imag < dt_min:
            pbar.write(f"Warning: timestep reached minimum ({dt_min}). Stopping.")
            break

# 此时 psi 应该已经形成带有涡旋的旋转稳态（类似涡旋晶格）

# Plot final imaginary-time state
print("Plotting final imaginary-time state...")
N_final_imag = np.sum(np.abs(psi)**2) * dx * dy
density_imag = np.abs(psi)**2
real_part_imag = np.real(psi)
imag_part_imag = np.imag(psi)
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.imshow(density_imag.T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap='RdBu')
plt.title(f'Density (tau={tau:.4f}, step={step}, N={N_final_imag:.4e})')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(real_part_imag.T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap='RdBu')
plt.title('Real Part')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.subplot(1, 3, 3)
plt.imshow(imag_part_imag.T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap='RdBu')
plt.title('Imaginary Part')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.tight_layout()
plt.show()

# # ===========================
# # 2. real-time 演化：在这个涡旋晶格上演化，涡旋长期存在
# #    你也可以在这里加扰动，制造湍流
# # ===========================
# print("Real-time evolution (vortex lattice / turbulence)...")
# # Reset timing stats for real-time evolution
# for key in timing_stats:
#     timing_stats[key] = 0.0
#     timing_counts[key] = 0

# for step in range(1, n_real_steps + 1):
#     t_step_start = time.perf_counter()
    
#     # ===== Explicit Euler step for real-time =====
#     # RHS: i ψ_t = H ψ → ψ_t = -i H ψ
#     
#     # Enforce BCs before differentiation
#     psi[0, :] = 0.0; psi[-1, :] = 0.0; psi[:, 0] = 0.0; psi[:, -1] = 0.0
#     
#     # Compute all derivatives together using differentiate_1_2 (more efficient)
#     t_lap_start = time.perf_counter()
#     psi_real = np.real(psi)
#     psi_imag = np.imag(psi)
#     
#     # Use differentiate_1_2 to compute all derivatives at once
#     dpsi_dx_real, dpsi_dy_real, d2psi_dx2_real, d2psi_dy2_real = bspf_op.differentiate_1_2(psi_real)
#     dpsi_dx_imag, dpsi_dy_imag, d2psi_dx2_imag, d2psi_dy2_imag = bspf_op.differentiate_1_2(psi_imag)
#     
#     # Compute Laplacian from second derivatives
#     lap = (d2psi_dx2_real + 1j * d2psi_dx2_imag) + (d2psi_dy2_real + 1j * d2psi_dy2_imag)
#     t_lap_end = time.perf_counter()
#     timing_stats['laplacian'] += (t_lap_end - t_lap_start)
#     timing_counts['laplacian'] += 1
#     
#     # Compute Lz_psi using first derivatives (already computed above)
#     t_Lz_start = time.perf_counter()
#     dpsi_dx = dpsi_dx_real + 1j * dpsi_dx_imag
#     dpsi_dy = dpsi_dy_real + 1j * dpsi_dy_imag
#     Lzpsi = -1j * (X * dpsi_dy - Y * dpsi_dx)
#     t_Lz_end = time.perf_counter()
#     timing_stats['Lz_psi'] += (t_Lz_end - t_Lz_start)
#     timing_counts['Lz_psi'] += 1
#     
#     # Combine to get H_psi (only the combination step, not including laplacian/Lz)
#     t_H_start = time.perf_counter()
#     Hpsi = -0.5 * lap + V_trap * psi + g * np.abs(psi)**2 * psi - Omega * Lzpsi
#     t_H_end = time.perf_counter()
#     timing_stats['H_psi'] += (t_H_end - t_H_start)
#     timing_counts['H_psi'] += 1
#     
#     # RHS for real-time: ψ_t = -i H ψ
#     t_rhs_start = time.perf_counter()
#     rhs = -1j * Hpsi
#     t_rhs_end = time.perf_counter()
#     timing_stats['rhs_computation'] += (t_rhs_end - t_rhs_start)
#     timing_counts['rhs_computation'] += 1
#     
#     # Euler step: ψ_{n+1} = ψ_n + dt * rhs (only the update and BC enforcement)
#     t_euler_start = time.perf_counter()
#     psi = psi + dt_real * rhs
#     
#     # Enforce Dirichlet BCs: ψ = 0 at boundaries
#     psi[0, :] = 0.0      # left boundary
#     psi[-1, :] = 0.0     # right boundary
#     psi[:, 0] = 0.0      # bottom boundary
#     psi[:, -1] = 0.0     # top boundary
#     t_euler_end = time.perf_counter()
#     timing_stats['euler_step'] += (t_euler_end - t_euler_start)
#     timing_counts['euler_step'] += 1
#     
#     t_step_end = time.perf_counter()
#     timing_stats['total_step'] += (t_step_end - t_step_start)
#     timing_counts['total_step'] += 1

#     if step % 100 == 0:
#         t_norm_start = time.perf_counter()
#         N_now = np.sum(np.abs(psi)**2) * dx * dy
#         t_norm_end = time.perf_counter()
#         timing_stats['norm'] += (t_norm_end - t_norm_start)
#         timing_counts['norm'] += 1
#         print(f"[Real] step {step}/{n_real_steps}, N = {N_now:.4e}")
#     
#     if step % 1000 == 0:
#         print_timing_stats()

# # Plot final real-time state
# print("Plotting final real-time state...")
# N_final_real = np.sum(np.abs(psi)**2) * dx * dy
# density_real = np.abs(psi)**2
# real_part_real = np.real(psi)
# imag_part_real = np.imag(psi)
# plt.figure(figsize=(15, 4))
# plt.subplot(1, 3, 1)
# plt.imshow(density_real.T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap='hot')
# plt.title(f'Density (step={n_real_steps}, N={N_final_real:.4e})')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.colorbar()
# plt.subplot(1, 3, 2)
# plt.imshow(real_part_real.T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap='RdBu')
# plt.title('Real Part')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.colorbar()
# plt.subplot(1, 3, 3)
# plt.imshow(imag_part_real.T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap='RdBu')
# plt.title('Imaginary Part')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.colorbar()
# plt.tight_layout()
# plt.show()

# Print final timing statistics
print("\n" + "="*60)
print("FINAL TIMING STATISTICS")
print("="*60)
print_timing_stats()
