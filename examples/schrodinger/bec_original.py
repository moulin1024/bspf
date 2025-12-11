import numpy as np
import matplotlib.pyplot as plt

# ===========================
# 参数设置
# ===========================
nx, ny = 256, 256          # 网格点数
Lx, Ly = 40.0, 40.0        # 物理尺寸
dx, dy = Lx / nx, Ly / ny

g = 1.0                    # 非线性系数
omega_trap = 1.0           # trap 频率（无量纲）
Omega = 0.9                # 旋转角速度（0 < Omega < omega_trap，一般）
dt_imag = 0.0005           # imaginary-time 步长
n_imag_steps = 5000        # imaginary-time 总步数
dt_real = 0.001            # real-time 步长
n_real_steps = 5000        # real-time 总步数
output_every = 500         # 每隔多少步画一次图

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
def init_guess():
    # 一个略带噪声的高斯作为初始猜测
    sigma = Lx / 5.0
    psi0 = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    psi0 = psi0 * (1.0 + 0.1 * (np.random.rand(nx, ny) - 0.5))
    return psi0.astype(np.complex128)

psi = init_guess()

# 目标粒子数（可以是任意归一化常数）
def norm(psi):
    return np.sum(np.abs(psi)**2) * dx * dy

# ===========================
# Dirichlet 边界条件：ψ = 0 在边界
# ===========================
def apply_dirichlet_bc(psi):
    """Enforce zero Dirichlet boundary conditions: ψ = 0 at boundaries"""
    psi_bc = psi.copy()
    # Set all boundary points to zero
    psi_bc[0, :] = 0.0      # left boundary
    psi_bc[-1, :] = 0.0     # right boundary
    psi_bc[:, 0] = 0.0      # bottom boundary
    psi_bc[:, -1] = 0.0     # top boundary
    return psi_bc

psi = apply_dirichlet_bc(psi)

# 计算初始粒子数（在应用边界条件之后）
N_target = norm(psi)

# ===========================
# 拉普拉斯算子（有限差分，Dirichlet 边界）
# ===========================
def laplacian(psi, dx, dy):
    """
    Compute Laplacian with zero Dirichlet boundary conditions.
    Boundary points are set to 0, interior uses standard finite differences.
    """
    lap = np.zeros_like(psi, dtype=np.complex128)

    # Ensure Dirichlet BCs are satisfied
    psi = apply_dirichlet_bc(psi)
    
    # Internal points: standard second-order finite difference
    # ∇²ψ = ∂²ψ/∂x² + ∂²ψ/∂y²
    lap[1:-1, 1:-1] = (
        (psi[2:, 1:-1] - 2.0 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dx**2 +
        (psi[1:-1, 2:] - 2.0 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / dy**2
    )

    # Boundary points: use one-sided differences with boundary value = 0
    # Left boundary (i=0): use forward difference, left neighbor is 0
    lap[0, 1:-1] = (
        (psi[1, 1:-1] - 2.0 * psi[0, 1:-1] + 0.0) / dx**2 +
        (psi[0, 2:] - 2.0 * psi[0, 1:-1] + psi[0, :-2]) / dy**2
    )
    # Right boundary (i=-1): use backward difference, right neighbor is 0
    lap[-1, 1:-1] = (
        (0.0 - 2.0 * psi[-1, 1:-1] + psi[-2, 1:-1]) / dx**2 +
        (psi[-1, 2:] - 2.0 * psi[-1, 1:-1] + psi[-1, :-2]) / dy**2
    )
    # Bottom boundary (j=0): use forward difference, bottom neighbor is 0
    lap[1:-1, 0] = (
        (psi[2:, 0] - 2.0 * psi[1:-1, 0] + psi[:-2, 0]) / dx**2 +
        (psi[1:-1, 1] - 2.0 * psi[1:-1, 0] + 0.0) / dy**2
    )
    # Top boundary (j=-1): use backward difference, top neighbor is 0
    lap[1:-1, -1] = (
        (psi[2:, -1] - 2.0 * psi[1:-1, -1] + psi[:-2, -1]) / dx**2 +
        (0.0 - 2.0 * psi[1:-1, -1] + psi[1:-1, -2]) / dy**2
    )

    # Corner points: both neighbors in one direction are 0
    lap[0, 0] = (psi[1, 0] - 2.0 * psi[0, 0] + 0.0) / dx**2 + (psi[0, 1] - 2.0 * psi[0, 0] + 0.0) / dy**2
    lap[0, -1] = (psi[1, -1] - 2.0 * psi[0, -1] + 0.0) / dx**2 + (0.0 - 2.0 * psi[0, -1] + psi[0, -2]) / dy**2
    lap[-1, 0] = (0.0 - 2.0 * psi[-1, 0] + psi[-2, 0]) / dx**2 + (psi[-1, 1] - 2.0 * psi[-1, 0] + 0.0) / dy**2
    lap[-1, -1] = (0.0 - 2.0 * psi[-1, -1] + psi[-2, -1]) / dx**2 + (0.0 - 2.0 * psi[-1, -1] + psi[-1, -2]) / dy**2
    
    # Enforce Dirichlet BCs on the result (boundary Laplacian is also 0)
    lap = apply_dirichlet_bc(lap)

    return lap

# ===========================
# 角动量算符 Lz ψ = -i ( x ∂yψ - y ∂xψ )
# ===========================
def Lz_psi(psi):
    """
    Compute angular momentum operator with zero Dirichlet boundary conditions.
    """
    # Ensure Dirichlet BCs are satisfied
    psi = apply_dirichlet_bc(psi.copy())

    dpsi_dx = np.zeros_like(psi, dtype=np.complex128)
    dpsi_dy = np.zeros_like(psi, dtype=np.complex128)

    # Center difference for interior points
    dpsi_dx[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2 * dx)
    dpsi_dy[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2 * dy)

    # Boundary points: derivatives are 0 (since psi = 0 at boundaries)
    # For boundary points, we can use one-sided differences or set to 0
    # Setting to 0 is consistent with Dirichlet BCs
    dpsi_dx[0, :] = 0.0
    dpsi_dx[-1, :] = 0.0
    dpsi_dy[:, 0] = 0.0
    dpsi_dy[:, -1] = 0.0

    # Lz ψ = -i (x ∂yψ - y ∂xψ)
    Lzpsi = -1j * (X * dpsi_dy - Y * dpsi_dx)
    
    # Enforce Dirichlet BCs on result
    Lzpsi = apply_dirichlet_bc(Lzpsi)
    
    return Lzpsi

# ===========================
# 哈密顿算符作用 H psi
# H = -1/2 ∇² + V_trap + g|ψ|² - Ω Lz
# ===========================
def H_psi(psi):
    # Ensure Dirichlet BCs are satisfied
    psi = apply_dirichlet_bc(psi)
    
    # Compute Laplacian and Lz (both enforce Dirichlet BCs internally)
    lap = laplacian(psi, dx, dy)
    Lzpsi = Lz_psi(psi)
    
    Hpsi = -0.5 * lap + V_trap * psi + g * np.abs(psi)**2 * psi - Omega * Lzpsi
    
    # Enforce Dirichlet BCs on result
    Hpsi = apply_dirichlet_bc(Hpsi)
    
    return Hpsi

# ===========================
# real-time RHS: i ψ_t = H ψ → ψ_t = -i H ψ
# ===========================
def rhs_real(psi):
    return -1j * H_psi(psi)

# ===========================
# imaginary-time RHS: ∂τ ψ = -H ψ
# ===========================
def rhs_imag(psi):
    return -H_psi(psi)

# ===========================
# RK4 步进（real / imag 通用）
# ===========================
def rk4_step(psi, dt, rhs_func):
    k1 = rhs_func(psi)
    k2 = rhs_func(psi + 0.5 * dt * k1)
    k3 = rhs_func(psi + 0.5 * dt * k2)
    k4 = rhs_func(psi + dt * k3)
    psi_new = psi + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    # Enforce Dirichlet BCs explicitly
    psi_new = apply_dirichlet_bc(psi_new)
    return psi_new

# ===========================
# 可视化
# ===========================
def plot_state(psi, step, title_prefix=""):
    density = np.abs(psi)**2
    phase = np.angle(psi)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(density.T, origin='lower',
               extent=[x[0], x[-1], y[0], y[-1]])
    plt.title(f'{title_prefix} Density, step={step}')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(phase.T, origin='lower',
               extent=[x[0], x[-1], y[0], y[-1]],
               vmin=-np.pi, vmax=np.pi)
    plt.title('Phase')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

# ===========================
# 1. imaginary-time 演化：求带旋转的稳态（涡旋晶格）
# ===========================
print("Imaginary-time evolution (finding rotating trapped state)...")
# Enforce Dirichlet BCs on initial condition
psi = apply_dirichlet_bc(psi)

for step in range(1, n_imag_steps + 1):
    psi = rk4_step(psi, dt_imag, rhs_imag)
    # 每步做一次归一化，保持粒子数 N_target
    N_now = norm(psi)
    psi *= np.sqrt(N_target / (N_now + 1e-16))
    # Enforce Dirichlet BCs after normalization
    psi = apply_dirichlet_bc(psi)

    if step % output_every == 0:
        print(f"[Imag] step {step}/{n_imag_steps}, N = {N_now:.4e}")
        plot_state(psi, step, title_prefix="Imag-time")

# 此时 psi 应该已经形成带有涡旋的旋转稳态（类似涡旋晶格）

# ===========================
# 2. real-time 演化：在这个涡旋晶格上演化，涡旋长期存在
#    你也可以在这里加扰动，制造湍流
# ===========================
print("Real-time evolution (vortex lattice / turbulence)...")
for step in range(1, n_real_steps + 1):
    psi = rk4_step(psi, dt_real, rhs_real)

    if step % output_every == 0:
        N_now = norm(psi)
        print(f"[Real] step {step}/{n_real_steps}, N = {N_now:.4e}")
        plot_state(psi, step, title_prefix="Real-time")
