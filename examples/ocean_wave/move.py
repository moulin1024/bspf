import numpy as np
import matplotlib.pyplot as plt
from bspf import bspf1d

# ============================================================
# 物理参数（线性水波方程）
# ============================================================
g = 9.81
h = 1.0
c = np.sqrt(g * h)  # wave speed

# ============================================================
# 计算域
# ============================================================
L = 10.0
N = 512
x = np.linspace(0, L, N)
dx = x[1] - x[0]

# ============================================================
# 移动墙参数
# ============================================================
wall_thickness = 0.5    # 厚度 d
X0 = 5.0                # 初始中心
A = 0.8                 # 振幅
omega = 0.7             # 频率

# ============================================================
# 初值：给一点扰动
# ============================================================
eta = np.exp(-80 * (x - 3)**2)   # 初始水面
v   = np.zeros_like(x)           # eta_t = 0

# ============================================================
# 时间步长（CFL）
# ============================================================
dt = 0.5 * dx / c
Tfinal = 15
steps = int(Tfinal / dt)

# 记录动画用
history = []

# ============================================================
# 辅助：从 d1 中抽取 jump
# ============================================================
def extract_derivative_jump(d1, a, b, x):
    """
    给 d1 (eta_x 数组) 与墙位置 a,b，返回
    eta_x 左/右极限值。
    """
    # 找到最接近 a,b 的网格下标
    ia = np.searchsorted(x, a)
    ib = np.searchsorted(x, b)

    # 左右侧导数
    eta_x_left_a  = d1[ia - 1]
    eta_x_right_a = d1[ia]

    eta_x_left_b  = d1[ib - 1]
    eta_x_right_b = d1[ib]

    # jump
    jump_a = eta_x_right_a - eta_x_left_a
    jump_b = eta_x_right_b - eta_x_left_b

    return jump_a, jump_b

# ============================================================
# 时间推进
# ============================================================
for n in range(steps):

    t = n * dt

    # -------------------------------------------
    # 1. 计算墙的位置
    # -------------------------------------------
    X = X0 + A * np.sin(omega * t)
    a = X - wall_thickness / 2
    b = X + wall_thickness / 2

    # -------------------------------------------
    # 2. 重新构造 bspf1d（每次更新内部 jump）
    #    注意：不修改 bspf1d，只重建
    # -------------------------------------------
    # 插入内部 knot：重复 degree+1 次
    degree = 3
    # 基于均匀网格自动确定 knots
    base = bspf1d.from_grid(degree=degree, x=x)
    knots = list(base.knots)

    # 在 a,b 插入 degree+1 次
    for _ in range(degree):
        knots.append(a)
        knots.append(b)
    knots = np.sort(np.array(knots))

    # 生成含内部 jump 的 spline
    bsp = bspf1d(grid=base.grid, degree=degree, knots=knots,
                 correction="spectral")

    # -------------------------------------------
    # 3. 用 bspf 求导数（它内部会进行 KKT 与分段谱校正）
    # -------------------------------------------
    d1, d2, eta_spline = bsp.differentiate_1_2(eta)

    # -------------------------------------------
    # 4. 用 jump 求界面力 f_L, f_R
    # -------------------------------------------
    jump_a, jump_b = extract_derivative_jump(d1, a, b, x)

    f_L = c**2 * jump_a
    f_R = -c**2 * jump_b

    # -------------------------------------------
    # 5. 把界面力加入 PDE（delta forcing）
    #    使用简单的 regularized delta（可改进）
    # -------------------------------------------
    F = np.zeros_like(x)
    ia = np.searchsorted(x, a)
    ib = np.searchsorted(x, b)

    # 简单 1st-order delta spread（可换成 Peskin 4pt）
    if 1 <= ia < N-1:
        F[ia] += f_L / dx
    if 1 <= ib < N-1:
        F[ib] += f_R / dx

    # -------------------------------------------
    # 6. 时间推进
    # -------------------------------------------
    eta_tt = c**2 * d2 + F

    v   += dt * eta_tt
    eta += dt * v

    # 记录动画
    if n % 20 == 0:
        history.append(eta.copy())

# ============================================================
# 可视化：显示波形随时间的传播
# ============================================================
plt.figure(figsize=(10,4))
plt.plot(x, history[0], label="t=0")
plt.plot(x, history[len(history)//2], label="middle")
plt.plot(x, history[-1], label="final")
plt.legend()
plt.xlabel("x")
plt.ylabel("η(x,t)")
plt.title("1D 水波 + 移动墙（IBM + bspf1d）")
plt.show()
