import numpy as np
import math

def riemann_sod(gamma=1.4):
    # Sod 初值
    rhoL, uL, pL = 1.0, 0.0, 1.0
    rhoR, uR, pR = 0.125, 0.0, 0.1
    cL = math.sqrt(gamma * pL / rhoL)
    cR = math.sqrt(gamma * pR / rhoR)

    # 求星区压力 p*
    def pressure_function(p):
        if p <= 0.0:
            return 1e20, 1.0
        # 左
        if p > pL:
            AL = 2.0 / ((gamma + 1.0) * rhoL)
            BL = pL * (gamma - 1.0) / (gamma + 1.0)
            fL = (p - pL) * math.sqrt(AL / (p + BL))
            dfL = math.sqrt(AL / (p + BL)) * (1.0 - 0.5 * (p - pL) / (p + BL))
        else:
            fL = (2.0 * cL / (gamma - 1.0)) * ((p / pL) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
            dfL = (1.0 / (rhoL * cL)) * (p / pL) ** (-(gamma + 1.0) / (2.0 * gamma))
        # 右
        if p > pR:
            AR = 2.0 / ((gamma + 1.0) * rhoR)
            BR = pR * (gamma - 1.0) / (gamma + 1.0)
            fR = (p - pR) * math.sqrt(AR / (p + BR))
            dfR = math.sqrt(AR / (p + BR)) * (1.0 - 0.5 * (p - pR) / (p + BR))
        else:
            fR = (2.0 * cR / (gamma - 1.0)) * ((p / pR) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
            dfR = (1.0 / (rhoR * cR)) * (p / pR) ** (-(gamma + 1.0) / (2.0 * gamma))

        f = fL + fR + (uR - uL)
        df = dfL + dfR
        return f, df

    # 初始压力猜测
    pPV = 0.5 * (pL + pR) - 0.125 * (uR - uL) * (rhoL + rhoR) * (cL + cR)
    p = max(1e-6, pPV)

    for _ in range(50):
        f, df = pressure_function(p)
        dp = -f / df
        p = p + dp
        if abs(dp) / (p + 1e-16) < 1e-6:
            break

    p_star = p

    # 计算 fL, fR
    def f_side(p, rho, u, p_i, c_i):
        if p > p_i:
            A = 2.0 / ((gamma + 1.0) * rho)
            B = p_i * (gamma - 1.0) / (gamma + 1.0)
            return (p - p_i) * math.sqrt(A / (p + B))
        else:
            return (2.0 * c_i / (gamma - 1.0)) * ((p / p_i) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)

    fL = f_side(p_star, rhoL, uL, pL, cL)
    fR = f_side(p_star, rhoR, uR, pR, cR)
    u_star = 0.5 * (uL + uR + fR - fL)

    # 星区密度
    if p_star > pL:
        rho_star_L = rhoL * ((p_star / pL + (gamma - 1.0) / (gamma + 1.0)) /
                             ((gamma - 1.0) / (gamma + 1.0) * p_star / pL + 1.0))
    else:
        rho_star_L = rhoL * (p_star / pL) ** (1.0 / gamma)

    if p_star > pR:
        rho_star_R = rhoR * ((p_star / pR + (gamma - 1.0) / (gamma + 1.0)) /
                             ((gamma - 1.0) / (gamma + 1.0) * p_star / pR + 1.0))
    else:
        rho_star_R = rhoR * (p_star / pR) ** (1.0 / gamma)

    return (rhoL, uL, pL), (rhoR, uR, pR), p_star, u_star, rho_star_L, rho_star_R

def sample_sod_exact(x, t, x0=0.5, gamma=1.4):
    # x: numpy 数组
    # t: 标量时间 > 0
    (rhoL, uL, pL), (rhoR, uR, pR), p_star, u_star, rho_star_L, rho_star_R = riemann_sod(gamma)
    cL = math.sqrt(gamma * pL / rhoL)
    cR = math.sqrt(gamma * pR / rhoR)
    c_star_L = math.sqrt(gamma * p_star / rho_star_L)

    # 各个波速
    S_HL = uL - cL                 # 稀疏波头
    S_TL = u_star - c_star_L       # 稀疏波尾
    S_C  = u_star                  # 接触间断
    # 右激波速度（Rankine-Hugoniot）
    S_R  = (rho_star_R * u_star - rhoR * uR) / (rho_star_R - rhoR)

    xi = (x - x0) / t

    rho = np.zeros_like(x)
    u = np.zeros_like(x)
    p = np.zeros_like(x)

    # 区域 1：左常状态
    mask1 = xi <= S_HL
    rho[mask1] = rhoL
    u[mask1] = uL
    p[mask1] = pL

    # 区域 2：左稀疏波扇形
    mask2 = (xi > S_HL) & (xi <= S_TL)
    xi2 = xi[mask2]
    c = (2.0 * cL / (gamma + 1.0) +
         (gamma - 1.0) / (gamma + 1.0) * (uL - xi2))
    u[mask2] = xi2 + c
    rho[mask2] = rhoL * (c / cL) ** (2.0 / (gamma - 1.0))
    p[mask2] = pL * (c / cL) ** (2.0 * gamma / (gamma - 1.0))

    # 区域 3：左星区
    mask3 = (xi > S_TL) & (xi <= S_C)
    rho[mask3] = rho_star_L
    u[mask3] = u_star
    p[mask3] = p_star

    # 区域 4：右星区
    mask4 = (xi > S_C) & (xi <= S_R)
    rho[mask4] = rho_star_R
    u[mask4] = u_star
    p[mask4] = p_star

    # 区域 5：右常状态
    mask5 = xi > S_R
    rho[mask5] = rhoR
    u[mask5] = uR
    p[mask5] = pR

    return rho, u, p

if __name__ == "__main__":
    # 示例：在 t = 0.2 处计算 Sod 激波管解
    nx = 400
    x = np.linspace(0.0, 1.0, nx)
    t = 0.2
    rho, u, p = sample_sod_exact(x, t)

    print("rho min/max:", rho.min(), rho.max())
    # 需要画图的话：
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, rho)
    plt.xlabel("x"); plt.ylabel("rho")
    plt.show()
