import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid

def strict_whitham_reconstruction(t, x_grid, x0=0.0):
    """
    严格重构 KdV 方程 Dispersive Shock Wave (A=1 -> 0) 的渐进解析解。
    基于 Gurevich-Pitaevskii 理论的一阶渐进项。
    
    参数:
        t: 演化时间 (float)
        x_grid: 空间网格点 (numpy array)
        x0: 初始阶跃位置 (float, default=0.0)，用于平移解析解以匹配数值结果
    """
    # 1. 理论边界 (A=1)，相对于初始阶跃位置 x0
    # Leading edge (Solitons): m -> 1, V = 4A = 4
    x_lead = x0 + 4.0 * t
    # Trailing edge (Linear waves): m -> 0, V = -6A = -6
    x_trail = x0 - 6.0 * t
    
    # 准备结果数组
    u_exact = np.zeros_like(x_grid)
    
    # 区域掩码
    mask_left = x_grid < x_trail  # 尚未受到扰动的左侧
    mask_right = x_grid > x_lead  # 尚未受到扰动的右侧
    mask_dsw = (~mask_left) & (~mask_right) # DSW 内部
    
    # 填充常数区
    u_exact[mask_left] = 1.0
    u_exact[mask_right] = 0.0
    
    # 如果没有点在 DSW 内，直接返回
    if not np.any(mask_dsw):
        return u_exact
        
    x_dsw = x_grid[mask_dsw]
    
    # === 步骤 A: 逐点精确求解模数 m ===
    # Whitham 方程: x/t - Vg(m) = 0
    # 我们定义残差函数
    def velocity_residual(m, xi_val):
        # 边界保护
        if m <= 1e-12: m = 1e-12
        if m >= 1-1e-12: m = 1-1e-12
        
        K = sp.ellipk(m)
        E = sp.ellipe(m)
        # 严格的群速度公式 Vg(m)
        # 对应 u_t + 6uu_x + u_xxx = 0, A=1
        term = (2 * m * (1 - m) * K) / (E - (1 - m) * K)
        Vg = 2 * (1 + m - term)
        return Vg - xi_val

    # 对 DSW 区域内的每个 x 求解 m
    # 为了速度，这里可以用插值优化，但为了"严格"，我们做逐点求解
    m_values = np.zeros_like(x_dsw)
    # 相对于初始阶跃位置 x0 计算无量纲坐标
    xi_values = (x_dsw - x0) / t
    
    # 利用 Vg 的单调性，我们可以利用前一个解作为初值，但 brentq 需要区间
    # Vg(m) 是单调增函数，从 -6 到 4
    for i, xi in enumerate(xi_values):
        try:
            # 在 (0, 1) 区间内求解
            m_sol = brentq(velocity_residual, 1e-10, 1-1e-10, args=(xi,))
            m_values[i] = m_sol
        except ValueError:
            # 极少数边缘情况处理
            m_values[i] = 1.0 if xi > 0 else 0.0

    # === 步骤 B: 严格的相位积分 ===
    # 局部物理波数 k(x)
    K_m = sp.ellipk(m_values)
    k_local = np.pi / (np.sqrt(6) * K_m)
    
    # 关键：从 Leading Edge (右侧) 向左积分
    # 原因：Leading edge 是孤子列，相位锁定最强。Trailing edge 是线性波，相位容易弥散。
    # 我们将积分反转：Flip array -> Integrate -> Flip back
    # 注意 dx 在反向时为负，或者我们手动处理 accum
    
    # x_dsw 是从左到右递增的。
    # 我们需要 int_{x_lead}^{x} k dx
    # = - int_{x}^{x_lead} k dx
    
    # 使用 cumulative_trapezoid 从右向左积分
    k_reversed = k_local[::-1]
    x_reversed = x_dsw[::-1]
    
    # 积分得到相位 (注意 cumulative_trapezoid 默认 initial=0 意味着第一个点相位为0)
    # 我们希望最右边的点相位接近 0 (第一个孤子峰值)
    phase_reversed = cumulative_trapezoid(k_reversed, x_reversed, initial=0)
    
    # 翻转回正常顺序
    phase = phase_reversed[::-1]
    
    # === 步骤 C: 构造雅可比 dn 解 ===
    # 椭圆函数参数 u = (K / pi) * phase
    # 注意：这里的 phase 已经是物理相位 (integral k dx)
    # 周期归一化需要乘 K/pi，但这里有一个倍数 2 的陷阱
    # Whitham 理论中，Theta = kx - wt，全相位变化 2pi 对应一个波长
    # ellipj 的 dn(u|m) 函数周期是 2K
    # 因此变数变换为: Arg = (Phase / 2pi) * 2K = Phase * K / pi
    
    args = phase * K_m / np.pi
    
    # 计算 dn
    # scipy.special.ellipj 返回 sn, cn, dn, ph
    _, _, dn_val, _ = sp.ellipj(args, m_values)
    
    # 组装波形: u = 2 * dn^2 - (1-m)
    u_dsw_sol = 2 * (dn_val**2) - (1 - m_values)
    
    u_exact[mask_dsw] = u_dsw_sol
    
    return u_exact, x_lead, x_trail

# === 执行验证 ===
if __name__ == "__main__":
    # 设置
    t_target = 20.0   # 时间足够大以显现清晰结构
    L = 1000.0          # 空间范围
    dx = 0.01          # 必须足够小以解析靠近 x_lead 的高频孤子
    x = np.arange(-L, L, dx)
    
    print(f"正在计算 t={t_target} 的严格 Whitham 解析解，可能需要几秒钟...")
    u, xl, xt = strict_whitham_reconstruction(t_target, x)
    
    plt.figure(figsize=(12, 6), dpi=120)
    
    # 绘制波形
    plt.plot(x, u, 'k-', linewidth=0.8, label='Strict Analytical Solution')
    
    # 绘制包络线 (用于检查计算正确性)
    # 重新计算一下 m 对应的包络
    # 简易方法：只画边界线
    plt.axvline(xl, color='r', linestyle='--', label=f'Soliton Edge (x={xl:.1f})')
    plt.axvline(xt, color='b', linestyle='--', label=f'Harmonic Edge (x={xt:.1f})')
    plt.axhline(2.0, color='r', linestyle=':', alpha=0.5, label='Max Amplitude (2.0)')
    
    # 局部放大图：展示最右侧的前导孤子
    # 这里的结构应该是完美的 sech^2 形状
    ax_ins = plt.axes([0.2, 0.5, 0.25, 0.25])
    
    # 找到最右侧部分
    idx_zoom = (x > xl - 30) & (x < xl + 5)
    ax_ins.plot(x[idx_zoom], u[idx_zoom], 'k-')
    ax_ins.set_title("Leading Solitons Structure")
    ax_ins.grid(True, alpha=0.3)
    
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f'Strict Gurevich-Pitaevskii Solution for KdV DSW (t={t_target})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([-700, 500])
    plt.show()