import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ===========================
# 参数设置
# ===========================
nx, ny = 256, 256          # 网格点数
Lx, Ly = 30.0, 30.0        # 物理尺寸
dx, dy = Lx / nx, Ly / ny

g = 1.0                    # 非线性系数
omega_y = 0.5              # y 方向弱束缚频率
F = 0.15                   # x 方向“重力/梯度”强度，决定云被推向哪一侧

dt_imag = 0.0005           # imaginary-time 步长
n_imag_steps = 5000        # imaginary-time 步数
dt_real = 0.0005           # real-time 步长
n_real_steps = 12000       # real-time 步数
output_every = 400         # 每隔多少步画一次图

np.random.seed(0)

# ===========================
# 网格与势
# ===========================
x = (np.arange(nx) - nx // 2) * dx
y = (np.arange(ny) - ny // 2) * dy
X, Y = np.meshgrid(x, y, indexing='ij')

# 倾斜盒子势：x 方向线性势 + y 方向弱谐振
V_trap = F * X + 0.5 * omega_y**2 * Y**2

# ===========================
# 初始：一个平滑云 + 轻微噪声
# ===========================
def init_guess():
    sigma_x = Lx / 3.0
    sigma_y = Ly / 3.0
    psi0 = np.exp(-(X**2 / (2 * sigma_x**2) + Y**2 / (2 * sigma_y**2)))
    psi0 = psi0 * (1.0 + 0.01 * (np.random.rand(nx, ny) - 0.5))
    return psi0.astype(np.complex128)

psi = init_guess()

def norm(psi):
    return np.sum(np.abs(psi)**2) * dx * dy

N_target = norm(psi)

# ===========================
# Neumann 边界：∂ψ/∂n = 0（zero flux 壁面）
# ===========================
def apply_neumann_bc(psi):
    psi[0, :] = psi[1, :]
    psi[-1, :] = psi[-2, :]
    psi[:, 0] = psi[:, 1]
    psi[:, -1] = psi[:, -2]
    return psi

psi = apply_neumann_bc(psi)

# ===========================
# 拉普拉斯（有限差分 + Neumann）
# ===========================
def laplacian(psi, dx, dy):
    lap = np.zeros_like(psi, dtype=np.complex128)

    # 内部点
    lap[1:-1, 1:-1] = (
        (psi[2:, 1:-1] - 2.0 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dx**2 +
        (psi[1:-1, 2:] - 2.0 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / dy**2
    )

    # 边界：Neumann（镜像）
    # 左右
    lap[0, 1:-1] = (
        (psi[1, 1:-1] - 2.0 * psi[0, 1:-1] + psi[1, 1:-1]) / dx**2 +
        (psi[0, 2:] - 2.0 * psi[0, 1:-1] + psi[0, :-2]) / dy**2
    )
    lap[-1, 1:-1] = (
        (psi[-2, 1:-1] - 2.0 * psi[-1, 1:-1] + psi[-2, 1:-1]) / dx**2 +
        (psi[-1, 2:] - 2.0 * psi[-1, 1:-1] + psi[-1, :-2]) / dy**2
    )
    # 上下
    lap[1:-1, 0] = (
        (psi[2:, 0] - 2.0 * psi[1:-1, 0] + psi[:-2, 0]) / dx**2 +
        (psi[1:-1, 1] - 2.0 * psi[1:-1, 0] + psi[1:-1, 1]) / dy**2
    )
    lap[1:-1, -1] = (
        (psi[2:, -1] - 2.0 * psi[1:-1, -1] + psi[:-2, -1]) / dx**2 +
        (psi[1:-1, -2] - 2.0 * psi[1:-1, -1] + psi[1:-1, -2]) / dy**2
    )

    # 四个角
    lap[0, 0] = (
        (psi[1, 0] - 2.0 * psi[0, 0] + psi[1, 0]) / dx**2 +
        (psi[0, 1] - 2.0 * psi[0, 0] + psi[0, 1]) / dy**2
    )
    lap[0, -1] = (
        (psi[1, -1] - 2.0 * psi[0, -1] + psi[1, -1]) / dx**2 +
        (psi[0, -2] - 2.0 * psi[0, -1] + psi[0, -2]) / dy**2
    )
    lap[-1, 0] = (
        (psi[-2, 0] - 2.0 * psi[-1, 0] + psi[-2, 0]) / dx**2 +
        (psi[-1, 1] - 2.0 * psi[-1, 0] + psi[-1, 1]) / dy**2
    )
    lap[-1, -1] = (
        (psi[-2, -1] - 2.0 * psi[-1, -1] + psi[-2, -1]) / dx**2 +
        (psi[-1, -2] - 2.0 * psi[-1, -1] + psi[-1, -2]) / dy**2
    )

    return lap

# ===========================
# 搅拌势：移动高斯障碍 V_obs(x,y,t)
# 模拟蓝失谐激光勺子，在高密度一侧横向扫过
# ===========================
V0_obs = 3.0        # 障碍高度（足够大才能激发涡旋）
sigma_obs = 1.0     # 障碍宽度
y_obs = 0.0         # 障碍轨迹在 y=0 附近
# 起点和速度：从云内部偏左侧出发，向右扫
x_start = -Lx * 0.3
x_end   =  Lx * 0.1
T_stir  =  (n_real_steps * dt_real) * 0.5   # 在前半段时间完成一次扫动
v_obs   = (x_end - x_start) / T_stir        # 线速度

def obstacle_potential(t):
    """
    t: 物理时间
    返回一个与 X,Y 同尺寸的 V_obs(x,y,t)
    只在 0 < t < T_stir 时运动，其余时间关闭或保持在终点
    """
    if t < 0.0:
        return np.zeros_like(X)
    elif t < T_stir:
        x0 = x_start + v_obs * t
    else:
        # 搅拌结束后，把障碍移开或直接关闭
        return np.zeros_like(X)
        # 如果想保留静止障碍，可改为：x0 = x_end

    r2 = (X - x0)**2 + (Y - y_obs)**2
    Vobs = V0_obs * np.exp(-r2 / (2 * sigma_obs**2))
    return Vobs

# ===========================
# 哈密顿 H ψ
# imaginary-time 阶段只用 V_trap
# real-time 阶段用 V_trap + V_obs(x,y,t)
# ===========================
def H_psi_imag(psi):
    psi = apply_neumann_bc(psi)
    lap = laplacian(psi, dx, dy)
    Hpsi = -0.5 * lap + V_trap * psi + g * np.abs(psi)**2 * psi
    return Hpsi

def H_psi_real(psi, t):
    psi = apply_neumann_bc(psi)
    lap = laplacian(psi, dx, dy)
    Vobs = obstacle_potential(t)
    V_tot = V_trap + Vobs
    Hpsi = -0.5 * lap + V_tot * psi + g * np.abs(psi)**2 * psi
    return Hpsi

def rhs_imag(psi):
    # imaginary time: ∂τ ψ = -H ψ
    return -H_psi_imag(psi)

def rhs_real(psi, t):
    # real time: i ψ_t = H ψ → ψ_t = -i H ψ
    return -1j * H_psi_real(psi, t)

# ===========================
# RK4 步进
# ===========================
def rk4_step_imag(psi, dt):
    k1 = rhs_imag(psi)
    k2 = rhs_imag(psi + 0.5 * dt * k1)
    k3 = rhs_imag(psi + 0.5 * dt * k2)
    k4 = rhs_imag(psi + dt * k3)
    psi_new = psi + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    psi_new = apply_neumann_bc(psi_new)
    return psi_new

def rk4_step_real(psi, dt, t):
    k1 = rhs_real(psi, t)
    k2 = rhs_real(psi + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs_real(psi + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs_real(psi + dt * k3, t + dt)
    psi_new = psi + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    psi_new = apply_neumann_bc(psi_new)
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
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(phase.T, origin='lower',
               extent=[x[0], x[-1], y[0], y[-1]],
               vmin=-np.pi, vmax=np.pi)
    plt.title('Phase')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

# ===========================
# 1. imaginary-time：得到倾斜盒子里的 BEC 基态（无涡旋）
# ===========================
print("Imaginary-time evolution to get tilted-box ground state...")
for step in range(1, n_imag_steps + 1):
    psi = rk4_step_imag(psi, dt_imag)
    N_now = norm(psi)
    psi *= np.sqrt(N_target / (N_now + 1e-16))

    if step % 1000 == 0:
        print(f"[Imag] step {step}/{n_imag_steps}, N = {N_now:.4e}")

print("Ground state obtained.")
plot_state(psi, step=0, title_prefix="Ground state")

# ===========================
# 2. real-time：用移动障碍搅拌 → 涡旋自然产生
# ===========================
print("Real-time evolution with stirring obstacle (natural vortex nucleation)...")
t = 0.0
states = []
steps_recorded = []
times_recorded = []

for step in range(1, n_real_steps + 1):
    psi = rk4_step_real(psi, dt_real, t)
    t += dt_real

    if step % output_every == 0:
        print(f"[Real] step {step}/{n_real_steps}, t = {t:.3f}")
        states.append(psi.copy())
        steps_recorded.append(step)
        times_recorded.append(t)

print(f"Real-time evolution complete. Collected {len(states)} frames for animation.")

# ===========================
# 创建 real-time 演化动画：只显示 density
# ===========================
if len(states) > 0:
    fig, ax = plt.subplots(figsize=(8, 8))

    # 计算全局颜色范围
    all_densities = [np.abs(state)**2 for state in states]
    density_min, density_max = np.min([np.min(d) for d in all_densities]), np.max([np.max(d) for d in all_densities])

    # 初始化图像
    density_plot = ax.imshow(all_densities[0].T, origin='lower',
                             extent=[x[0], x[-1], y[0], y[-1]],
                             vmin=density_min, vmax=density_max, animated=True,cmap='hot')
    ax.set_title(f'Real-time Density, step={steps_recorded[0]}, t={times_recorded[0]:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    cbar = fig.colorbar(density_plot, ax=ax)

    plt.tight_layout()

    def animate(frame):
        """更新动画帧"""
        density = np.abs(states[frame])**2
        density_plot.set_array(density.T)
        ax.set_title(f'Real-time Density, step={steps_recorded[frame]}, t={times_recorded[frame]:.3f}')
        return density_plot,

    # 创建动画
    anim = FuncAnimation(fig, animate, frames=len(states), interval=100, blit=True, repeat=True)

    print("Animation created. Displaying real-time density evolution...")
    plt.show()
