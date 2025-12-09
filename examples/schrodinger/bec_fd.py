import numpy as np
import matplotlib.pyplot as plt

# ===========================
# 参数设置
# ===========================
nx, ny = 256, 256          # 网格点数
Lx, Ly = 20.0, 20.0        # 物理尺寸 [-Lx/2, Lx/2] × [-Ly/2, Ly/2]
dx, dy = Lx / nx, Ly / ny

g = 800.0                  # 非线性强度（可以调大一些看更多涡旋）
omega_trap = 1.0           # trap 频率
Omega_target = 0.9         # 目标旋转频率
tau_max = 6.0              # imaginary time 总长度
tau_ramp = 3.0             # 在前 tau_ramp 内把 Omega 从 0 ramp 到 Omega_target
dt_imag = 1e-3             # imaginary time 步长
n_steps = int(tau_max / dt_imag)
output_every = 1000        # 每隔多少步画一次图（可改小看看演化过程）

np.random.seed(0)

# ===========================
# 网格与势
# ===========================
x = (np.arange(nx) - nx // 2) * dx
y = (np.arange(ny) - ny // 2) * dy
X, Y = np.meshgrid(x, y, indexing='ij')

V_trap = 0.5 * omega_trap**2 * (X**2 + Y**2)  # 圆对称谐振势

# ===========================
# 初始条件：高斯 + 轻微噪声
# ===========================
sigma = Lx / 4.0
psi = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
psi *= (1.0 + 0.01 * (np.random.rand(nx, ny) - 0.5))
psi = psi.astype(np.complex128)


# ===========================
# Dirichlet 边界: ψ=0 at box edges
# ===========================
def apply_dirichlet(psi):
    psi[0, :] = 0.0
    psi[-1, :] = 0.0
    psi[:, 0] = 0.0
    psi[:, -1] = 0.0
    return psi


apply_dirichlet(psi)

# 初始粒子数（可以看作固定的 N_target）
def norm(psi):
    return np.sum(np.abs(psi)**2) * dx * dy


N_target = norm(psi)

# ===========================
# 4 阶有限差分：二阶导（Laplacian）和一阶导（∇ψ）
# 内部用 4 阶中心差分，边界两层简单设为 0（云远离边界影响不大）
# ===========================
def laplacian_4th(psi, dx, dy):
    """
    4th-order central difference Laplacian.
    只在 index 2:-2 内严格 4 阶，边界附近近似。
    """
    lap = np.zeros_like(psi, dtype=np.complex128)

    # 二阶导数 w.r.t x
    lap[2:-2, :] += (
        -psi[4:, :] + 16.0 * psi[3:-1, :] - 30.0 * psi[2:-2, :]
        + 16.0 * psi[1:-3, :] - psi[0:-4, :]
    ) / (12.0 * dx**2)

    # 二阶导数 w.r.t y
    lap[:, 2:-2] += (
        -psi[:, 4:] + 16.0 * psi[:, 3:-1] - 30.0 * psi[:, 2:-2]
        + 16.0 * psi[:, 1:-3] - psi[:, 0:-4]
    ) / (12.0 * dy**2)

    return lap


def grad_4th(psi, dx, dy):
    """
    4th-order central difference for ∂ψ/∂x, ∂ψ/∂y.
    """
    dpsi_dx = np.zeros_like(psi, dtype=np.complex128)
    dpsi_dy = np.zeros_like(psi, dtype=np.complex128)

    # ∂/∂x
    dpsi_dx[2:-2, :] = (
        psi[0:-4, :] - 8.0 * psi[1:-3, :] + 8.0 * psi[3:-1, :] - psi[4:, :]
    ) / (12.0 * dx)

    # ∂/∂y
    dpsi_dy[:, 2:-2] = (
        psi[:, 0:-4] - 8.0 * psi[:, 1:-3] + 8.0 * psi[:, 3:-1] - psi[:, 4:]
    ) / (12.0 * dy)

    return dpsi_dx, dpsi_dy


# ===========================
# 哈密顿算符 Hψ = -1/2 ∇²ψ + V ψ + g|ψ|²ψ - Ω Lz ψ
# Lzψ = -i (x ∂yψ - y ∂xψ)
# ===========================
def H_psi(psi, Omega):
    # 确保边界满足 Dirichlet
    psi = apply_dirichlet(psi.copy())

    lap = laplacian_4th(psi, dx, dy)
    dpsi_dx, dpsi_dy = grad_4th(psi, dx, dy)
    Lzpsi = -1j * (X * dpsi_dy - Y * dpsi_dx)

    Hpsi = -0.5 * lap + V_trap * psi + g * np.abs(psi)**2 * psi - Omega * Lzpsi
    return Hpsi


# ===========================
# imaginary-time RHS: ∂τ ψ = -H ψ
# ===========================
def rhs_imag(psi, Omega):
    return -H_psi(psi, Omega)


# ===========================
# RK4 step in imaginary time
# ===========================
def rk4_step_imag(psi, dt, Omega):
    k1 = rhs_imag(psi, Omega)
    k2 = rhs_imag(psi + 0.5 * dt * k1, Omega)
    k3 = rhs_imag(psi + 0.5 * dt * k2, Omega)
    k4 = rhs_imag(psi + dt * k3, Omega)

    psi_new = psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    apply_dirichlet(psi_new)
    return psi_new


# ===========================
# 可视化函数
# ===========================
def plot_state(psi, title=""):
    density = np.abs(psi)**2
    phase = np.angle(psi)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(
        density.T,
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
    )
    plt.title(title + " | Density")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(
        phase.T,
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        vmin=-np.pi,
        vmax=np.pi,
    )
    plt.title(title + " | Phase")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()

    plt.tight_layout()
    plt.show()


# ===========================
# Imaginary-time 演化：求旋转基态 / 涡旋晶格
# ===========================
tau = 0.0
Omega = 0.0

print("Imaginary-time evolution (4th-order FD + RK4)...")
print(f"Total steps: {n_steps}, dt = {dt_imag}")

for n in range(1, n_steps + 1):
    # Ω ramp: 从 0 线性升到 Omega_target
    if tau < tau_ramp:
        Omega = Omega_target * (tau / tau_ramp)
    else:
        Omega = Omega_target

    psi = rk4_step_imag(psi, dt_imag, Omega)

    # 归一化
    N_now = norm(psi)
    psi *= np.sqrt(N_target / (N_now + 1e-16))
    apply_dirichlet(psi)

    tau += dt_imag

    if n % output_every == 0 or n == 1:
        print(f"step {n}/{n_steps}, tau = {tau:.3f}, Omega = {Omega:.3f}, N = {N_now:.4e}")
        plot_state(psi, title=f"Imag-time τ={tau:.3f}")

print("Imaginary-time evolution finished.")
print(f"Final τ = {tau:.3f}, Omega = {Omega:.3f}, N = {norm(psi):.4e}")
plot_state(psi, title="Final state")
