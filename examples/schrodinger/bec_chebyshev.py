import numpy as np
import matplotlib.pyplot as plt
import time
from specderiv import cheb_deriv

# ===========================
# 参数设置
# ===========================
nx, ny = 128, 128          # 网格点数
Lx, Ly = 40.0, 40.0        # 物理尺寸

g = 1.0                    # 非线性系数
omega_trap = 1.0           # trap 频率（无量纲）
Omega = 0.9                # 旋转角速度（0 < Omega < omega_trap，一般）
dt_imag = 0.00005           # imaginary-time 步长
n_imag_steps = 5000        # imaginary-time 总步数
dt_real = 0.001            # real-time 步长
n_real_steps = 5000        # real-time 总步数
output_every = 500         # 每隔多少步画一次图

np.random.seed(0)

# ===========================
# Chebyshev 网格与势
# ===========================
# Create Chebyshev-Gauss-Lobatto points on canonical interval [-1, 1]
# x_n = cos(π * n / N) for n = 0, ..., N
x_canonical = np.cos(np.arange(nx) * np.pi / (nx - 1))  # [-1, 1]
y_canonical = np.cos(np.arange(ny) * np.pi / (ny - 1))  # [-1, 1]

# Map to physical domain [-Lx/2, Lx/2] and [-Ly/2, Ly/2]
# x = (b-a)/2 * x_canonical + (a+b)/2
x = x_canonical * (Lx / 2.0)  # Maps [-1, 1] to [-Lx/2, Lx/2]
y = y_canonical * (Ly / 2.0)  # Maps [-1, 1] to [-Ly/2, Ly/2]

# Create 2D grid
X, Y = np.meshgrid(x, y, indexing='ij')  # Shape: (nx, ny)

# Chebyshev-Gauss-Lobatto quadrature weights for integration
# w[0] = w[N] = π/(2N), w[i] = π/N for i = 1, ..., N-1
def chebyshev_weights(n):
    """Compute Chebyshev-Gauss-Lobatto quadrature weights"""
    w = np.ones(n) * np.pi / (n - 1)
    w[0] = np.pi / (2 * (n - 1))
    w[-1] = np.pi / (2 * (n - 1))
    return w

# Scale weights for physical domain
wx = chebyshev_weights(nx) * (Lx / 2.0)  # Scale by domain half-length
wy = chebyshev_weights(ny) * (Ly / 2.0)
# Create 2D weight matrix for integration: dA = wx[i] * wy[j]
W = np.outer(wx, wy)  # Shape: (nx, ny)

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
    'rk4_step': 0.0,
    'norm': 0.0,
    'rhs_computation': 0.0,
    'total_step': 0.0
}
timing_counts = {
    'laplacian': 0,
    'Lz_psi': 0,
    'H_psi': 0,
    'rk4_step': 0,
    'norm': 0,
    'rhs_computation': 0,
    'total_step': 0
}

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

# 计算初始粒子数（在应用边界条件之后）
# Use Chebyshev quadrature: ∫ f dx dy ≈ Σ w[i,j] * f[i,j]
t_norm_start = time.perf_counter()
N_target = np.sum(np.abs(psi)**2 * W)
t_norm_end = time.perf_counter()
timing_stats['norm'] += (t_norm_end - t_norm_start)
timing_counts['norm'] += 1

for step in range(1, n_imag_steps + 1):
    t_step_start = time.perf_counter()
    
    # ===== RK4 step for imaginary-time =====
    # RHS: ∂τ ψ = -H ψ
    
    # Helper function to compute RHS (inline, for reuse in RK4 stages)
    def compute_rhs_imag_inline(psi_in):
        # Enforce BCs before differentiation
        psi_in[0, :] = 0.0; psi_in[-1, :] = 0.0; psi_in[:, 0] = 0.0; psi_in[:, -1] = 0.0
        
        # Compute Laplacian using Chebyshev spectral differentiation
        t_lap_start = time.perf_counter()
        
        # Split into real and imaginary parts for Chebyshev differentiation
        psi_real = np.real(psi_in)
        psi_imag = np.imag(psi_in)
        
        # Compute second derivatives using Chebyshev method
        # axis=1 for x-derivative (along columns), axis=0 for y-derivative (along rows)
        d2psi_dx2_real = cheb_deriv(psi_real, x, order=2, axis=1)
        d2psi_dx2_imag = cheb_deriv(psi_imag, x, order=2, axis=1)
        d2psi_dy2_real = cheb_deriv(psi_real, y, order=2, axis=0)
        d2psi_dy2_imag = cheb_deriv(psi_imag, y, order=2, axis=0)
        
        # Combine to get Laplacian
        lap = (d2psi_dx2_real + 1j * d2psi_dx2_imag) + (d2psi_dy2_real + 1j * d2psi_dy2_imag)
        
        # Enforce Dirichlet BCs on the result
        lap[0, :] = 0.0; lap[-1, :] = 0.0; lap[:, 0] = 0.0; lap[:, -1] = 0.0
        t_lap_end = time.perf_counter()
        timing_stats['laplacian'] += (t_lap_end - t_lap_start)
        timing_counts['laplacian'] += 1
        
        # Compute Lz_psi using first derivatives
        t_Lz_start = time.perf_counter()
        
        # Compute first derivatives using Chebyshev method
        dpsi_dx_real = cheb_deriv(psi_real, x, order=1, axis=1)
        dpsi_dx_imag = cheb_deriv(psi_imag, x, order=1, axis=1)
        dpsi_dy_real = cheb_deriv(psi_real, y, order=1, axis=0)
        dpsi_dy_imag = cheb_deriv(psi_imag, y, order=1, axis=0)
        
        # Combine to get complex derivatives
        dpsi_dx = dpsi_dx_real + 1j * dpsi_dx_imag
        dpsi_dy = dpsi_dy_real + 1j * dpsi_dy_imag
        
        # Lz ψ = -i (x ∂yψ - y ∂xψ)
        Lzpsi = -1j * (X * dpsi_dy - Y * dpsi_dx)
        
        # Enforce Dirichlet BCs on result
        Lzpsi[0, :] = 0.0; Lzpsi[-1, :] = 0.0; Lzpsi[:, 0] = 0.0; Lzpsi[:, -1] = 0.0
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
    
    # RK4 stages (classical Runge-Kutta 4th order)
    t_rk4_start = time.perf_counter()
    
    # Stage 1: k1 = rhs(psi)
    k1 = compute_rhs_imag_inline(psi.copy())
    
    # Stage 2: k2 = rhs(psi + 0.5*dt*k1)
    psi_stage2 = psi + 0.5 * dt_imag * k1
    psi_stage2[0, :] = 0.0; psi_stage2[-1, :] = 0.0; psi_stage2[:, 0] = 0.0; psi_stage2[:, -1] = 0.0
    k2 = compute_rhs_imag_inline(psi_stage2)
    
    # Stage 3: k3 = rhs(psi + 0.5*dt*k2)
    psi_stage3 = psi + 0.5 * dt_imag * k2
    psi_stage3[0, :] = 0.0; psi_stage3[-1, :] = 0.0; psi_stage3[:, 0] = 0.0; psi_stage3[:, -1] = 0.0
    k3 = compute_rhs_imag_inline(psi_stage3)
    
    # Stage 4: k4 = rhs(psi + dt*k3)
    psi_stage4 = psi + dt_imag * k3
    psi_stage4[0, :] = 0.0; psi_stage4[-1, :] = 0.0; psi_stage4[:, 0] = 0.0; psi_stage4[:, -1] = 0.0
    k4 = compute_rhs_imag_inline(psi_stage4)
    
    # 4th order solution (classical RK4)
    psi = psi + (dt_imag / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    
    # Enforce Dirichlet BCs: ψ = 0 at boundaries
    psi[0, :] = 0.0      # left boundary
    psi[-1, :] = 0.0     # right boundary
    psi[:, 0] = 0.0      # bottom boundary
    psi[:, -1] = 0.0     # top boundary
    t_rk4_end = time.perf_counter()
    timing_stats['rk4_step'] += (t_rk4_end - t_rk4_start)
    timing_counts['rk4_step'] += 1
    
    # Normalization using Chebyshev quadrature
    t_norm_start = time.perf_counter()
    N_now = np.sum(np.abs(psi)**2 * W)
    psi *= np.sqrt(N_target / (N_now + 1e-16))
    
    # Enforce Dirichlet BCs after normalization
    psi[0, :] = 0.0      # left boundary
    psi[-1, :] = 0.0     # right boundary
    psi[:, 0] = 0.0      # bottom boundary
    psi[:, -1] = 0.0     # top boundary
    t_norm_end = time.perf_counter()
    timing_stats['norm'] += (t_norm_end - t_norm_start)
    timing_counts['norm'] += 1
    
    t_step_end = time.perf_counter()
    timing_stats['total_step'] += (t_step_end - t_step_start)
    timing_counts['total_step'] += 1

    print(f"[Imag] step {step}/{n_imag_steps}, N = {N_now:.4e}")
    if step % 1000 == 0:
        print_timing_stats()

# 此时 psi 应该已经形成带有涡旋的旋转稳态（类似涡旋晶格）

# Plot final imaginary-time state
print("Plotting final imaginary-time state...")
N_final_imag = np.sum(np.abs(psi)**2 * W)
density_imag = np.abs(psi)**2
real_part_imag = np.real(psi)
imag_part_imag = np.imag(psi)
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.imshow(density_imag.T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap='hot')
plt.title(f'Density (step={n_imag_steps}, N={N_final_imag:.4e})')
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

# ===========================
# 2. real-time 演化：在这个涡旋晶格上演化，涡旋长期存在
#    你也可以在这里加扰动，制造湍流
# ===========================
print("Real-time evolution (vortex lattice / turbulence)...")
# Reset timing stats for real-time evolution
for key in timing_stats:
    timing_stats[key] = 0.0
    timing_counts[key] = 0

for step in range(1, n_real_steps + 1):
    t_step_start = time.perf_counter()
    
    # ===== RK4 step for real-time =====
    # RHS: i ψ_t = H ψ → ψ_t = -i H ψ
    
    # Helper function to compute RHS (inline, for reuse in RK4 stages)
    def compute_rhs_real_inline(psi_in):
        # Enforce BCs before differentiation
        psi_in[0, :] = 0.0; psi_in[-1, :] = 0.0; psi_in[:, 0] = 0.0; psi_in[:, -1] = 0.0
        
        # Compute Laplacian using Chebyshev spectral differentiation
        t_lap_start = time.perf_counter()
        
        # Split into real and imaginary parts for Chebyshev differentiation
        psi_real = np.real(psi_in)
        psi_imag = np.imag(psi_in)
        
        # Compute second derivatives using Chebyshev method
        d2psi_dx2_real = cheb_deriv(psi_real, x, order=2, axis=1)
        d2psi_dx2_imag = cheb_deriv(psi_imag, x, order=2, axis=1)
        d2psi_dy2_real = cheb_deriv(psi_real, y, order=2, axis=0)
        d2psi_dy2_imag = cheb_deriv(psi_imag, y, order=2, axis=0)
        
        # Combine to get Laplacian
        lap = (d2psi_dx2_real + 1j * d2psi_dx2_imag) + (d2psi_dy2_real + 1j * d2psi_dy2_imag)
        
        # Enforce Dirichlet BCs on the result
        lap[0, :] = 0.0; lap[-1, :] = 0.0; lap[:, 0] = 0.0; lap[:, -1] = 0.0
        t_lap_end = time.perf_counter()
        timing_stats['laplacian'] += (t_lap_end - t_lap_start)
        timing_counts['laplacian'] += 1
        
        # Compute Lz_psi using first derivatives
        t_Lz_start = time.perf_counter()
        
        # Compute first derivatives using Chebyshev method
        dpsi_dx_real = cheb_deriv(psi_real, x, order=1, axis=1)
        dpsi_dx_imag = cheb_deriv(psi_imag, x, order=1, axis=1)
        dpsi_dy_real = cheb_deriv(psi_real, y, order=1, axis=0)
        dpsi_dy_imag = cheb_deriv(psi_imag, y, order=1, axis=0)
        
        # Combine to get complex derivatives
        dpsi_dx = dpsi_dx_real + 1j * dpsi_dx_imag
        dpsi_dy = dpsi_dy_real + 1j * dpsi_dy_imag
        
        # Lz ψ = -i (x ∂yψ - y ∂xψ)
        Lzpsi = -1j * (X * dpsi_dy - Y * dpsi_dx)
        
        # Enforce Dirichlet BCs on result
        Lzpsi[0, :] = 0.0; Lzpsi[-1, :] = 0.0; Lzpsi[:, 0] = 0.0; Lzpsi[:, -1] = 0.0
        t_Lz_end = time.perf_counter()
        timing_stats['Lz_psi'] += (t_Lz_end - t_Lz_start)
        timing_counts['Lz_psi'] += 1
        
        # Combine to get H_psi (only the combination step, not including laplacian/Lz)
        t_H_start = time.perf_counter()
        Hpsi = -0.5 * lap + V_trap * psi_in + g * np.abs(psi_in)**2 * psi_in - Omega * Lzpsi
        t_H_end = time.perf_counter()
        timing_stats['H_psi'] += (t_H_end - t_H_start)
        timing_counts['H_psi'] += 1
        
        # RHS for real-time: ψ_t = -i H ψ
        t_rhs_start = time.perf_counter()
        rhs = -1j * Hpsi
        t_rhs_end = time.perf_counter()
        timing_stats['rhs_computation'] += (t_rhs_end - t_rhs_start)
        timing_counts['rhs_computation'] += 1
        
        return rhs
    
    # RK4 stages (classical Runge-Kutta 4th order)
    t_rk4_start = time.perf_counter()
    
    # Stage 1: k1 = rhs(psi)
    k1 = compute_rhs_real_inline(psi.copy())
    
    # Stage 2: k2 = rhs(psi + 0.5*dt*k1)
    psi_stage2 = psi + 0.5 * dt_real * k1
    psi_stage2[0, :] = 0.0; psi_stage2[-1, :] = 0.0; psi_stage2[:, 0] = 0.0; psi_stage2[:, -1] = 0.0
    k2 = compute_rhs_real_inline(psi_stage2)
    
    # Stage 3: k3 = rhs(psi + 0.5*dt*k2)
    psi_stage3 = psi + 0.5 * dt_real * k2
    psi_stage3[0, :] = 0.0; psi_stage3[-1, :] = 0.0; psi_stage3[:, 0] = 0.0; psi_stage3[:, -1] = 0.0
    k3 = compute_rhs_real_inline(psi_stage3)
    
    # Stage 4: k4 = rhs(psi + dt*k3)
    psi_stage4 = psi + dt_real * k3
    psi_stage4[0, :] = 0.0; psi_stage4[-1, :] = 0.0; psi_stage4[:, 0] = 0.0; psi_stage4[:, -1] = 0.0
    k4 = compute_rhs_real_inline(psi_stage4)
    
    # 4th order solution (classical RK4)
    psi = psi + (dt_real / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    
    # Enforce Dirichlet BCs: ψ = 0 at boundaries
    psi[0, :] = 0.0      # left boundary
    psi[-1, :] = 0.0     # right boundary
    psi[:, 0] = 0.0      # bottom boundary
    psi[:, -1] = 0.0     # top boundary
    t_rk4_end = time.perf_counter()
    timing_stats['rk4_step'] += (t_rk4_end - t_rk4_start)
    timing_counts['rk4_step'] += 1
    
    t_step_end = time.perf_counter()
    timing_stats['total_step'] += (t_step_end - t_step_start)
    timing_counts['total_step'] += 1

    if step % 100 == 0:
        t_norm_start = time.perf_counter()
        N_now = np.sum(np.abs(psi)**2 * W)
        t_norm_end = time.perf_counter()
        timing_stats['norm'] += (t_norm_end - t_norm_start)
        timing_counts['norm'] += 1
        print(f"[Real] step {step}/{n_real_steps}, N = {N_now:.4e}")
    
    if step % 1000 == 0:
        print_timing_stats()

# Plot final real-time state
print("Plotting final real-time state...")
N_final_real = np.sum(np.abs(psi)**2 * W)
density_real = np.abs(psi)**2
real_part_real = np.real(psi)
imag_part_real = np.imag(psi)
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.imshow(density_real.T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap='hot')
plt.title(f'Density (step={n_real_steps}, N={N_final_real:.4e})')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(real_part_real.T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap='RdBu')
plt.title('Real Part')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.subplot(1, 3, 3)
plt.imshow(imag_part_real.T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap='RdBu')
plt.title('Imaginary Part')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.tight_layout()
plt.show()

# Print final timing statistics
print("\n" + "="*60)
print("FINAL TIMING STATISTICS")
print("="*60)
print_timing_stats()

