import numpy as np
import matplotlib.pyplot as plt
from sod_exact import sample_sod_exact, riemann_sod
import math

GAMMA = 1.4

# ========== 基本转换 ==========
def prim_to_cons(rho, u, p, gamma=GAMMA):
    E = p / (gamma - 1.0) + 0.5 * rho * u**2
    return np.stack([rho, rho * u, E], axis=0)  # shape (3, N)

def cons_to_prim(U, gamma=GAMMA):
    rho = U[0]
    mom = U[1]
    E   = U[2]
    u = mom / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * u**2)
    return rho, u, p

def euler_flux(U, gamma=GAMMA):
    rho, u, p = cons_to_prim(U, gamma=gamma)
    F = np.zeros_like(U)
    F[0] = rho * u
    F[1] = rho * u**2 + p
    F[2] = (U[2] + p) * u  # (E + p) u
    return F

# ========== Exact Riemann Solver ==========
def exact_riemann_solver(rhoL, uL, pL, rhoR, uR, pR, gamma=GAMMA):
    """
    Solve the exact Riemann problem for arbitrary left and right states.
    Returns: p_star, u_star, rho_star_L, rho_star_R
    """
    cL = math.sqrt(gamma * pL / rhoL)
    cR = math.sqrt(gamma * pR / rhoR)

    # Pressure function for Newton iteration
    def pressure_function(p):
        if p <= 0.0:
            return 1e20, 1.0
        # Left side
        if p > pL:
            AL = 2.0 / ((gamma + 1.0) * rhoL)
            BL = pL * (gamma - 1.0) / (gamma + 1.0)
            fL = (p - pL) * math.sqrt(AL / (p + BL))
            dfL = math.sqrt(AL / (p + BL)) * (1.0 - 0.5 * (p - pL) / (p + BL))
        else:
            fL = (2.0 * cL / (gamma - 1.0)) * ((p / pL) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
            dfL = (1.0 / (rhoL * cL)) * (p / pL) ** (-(gamma + 1.0) / (2.0 * gamma))
        # Right side
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

    # Initial pressure guess (PVRS - Primitive Variable Riemann Solver)
    pPV = 0.5 * (pL + pR) - 0.125 * (uR - uL) * (rhoL + rhoR) * (cL + cR)
    p = max(1e-6, pPV)

    # Newton iteration
    for _ in range(50):
        f, df = pressure_function(p)
        dp = -f / df
        p = p + dp
        if abs(dp) / (p + 1e-16) < 1e-6:
            break

    p_star = p

    # Compute u_star
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

    # Compute star region densities
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

    return p_star, u_star, rho_star_L, rho_star_R

def sample_riemann_solution(rhoL, uL, pL, rhoR, uR, pR, xi, gamma=GAMMA):
    """
    Sample the exact Riemann solution at xi = x/t.
    Returns: rho, u, p at the given xi value(s).
    """
    p_star, u_star, rho_star_L, rho_star_R = exact_riemann_solver(rhoL, uL, pL, rhoR, uR, pR, gamma)
    
    cL = math.sqrt(gamma * pL / rhoL)
    cR = math.sqrt(gamma * pR / rhoR)
    c_star_L = math.sqrt(gamma * p_star / rho_star_L)
    
    # Wave speeds
    S_HL = uL - cL  # Left rarefaction head
    S_TL = u_star - c_star_L  # Left rarefaction tail
    S_C = u_star  # Contact discontinuity
    S_R = (rho_star_R * u_star - rhoR * uR) / (rho_star_R - rhoR)  # Shock speed
    
    # Handle scalar or array xi
    xi = np.asarray(xi)
    is_scalar = xi.ndim == 0
    if is_scalar:
        xi = np.array([xi])
    
    rho = np.zeros_like(xi)
    u = np.zeros_like(xi)
    p = np.zeros_like(xi)
    
    # Region 1: Left constant state
    mask1 = xi <= S_HL
    rho[mask1] = rhoL
    u[mask1] = uL
    p[mask1] = pL
    
    # Region 2: Left rarefaction fan
    mask2 = (xi > S_HL) & (xi <= S_TL)
    xi2 = xi[mask2]
    c = (2.0 * cL / (gamma + 1.0) + (gamma - 1.0) / (gamma + 1.0) * (uL - xi2))
    u[mask2] = xi2 + c
    rho[mask2] = rhoL * (c / cL) ** (2.0 / (gamma - 1.0))
    p[mask2] = pL * (c / cL) ** (2.0 * gamma / (gamma - 1.0))
    
    # Region 3: Left star region
    mask3 = (xi > S_TL) & (xi <= S_C)
    rho[mask3] = rho_star_L
    u[mask3] = u_star
    p[mask3] = p_star
    
    # Region 4: Right star region
    mask4 = (xi > S_C) & (xi <= S_R)
    rho[mask4] = rho_star_R
    u[mask4] = u_star
    p[mask4] = p_star
    
    # Region 5: Right constant state
    mask5 = xi > S_R
    rho[mask5] = rhoR
    u[mask5] = uR
    p[mask5] = pR
    
    if is_scalar:
        return rho[0], u[0], p[0]
    return rho, u, p

def exact_riemann_flux(UL, UR, gamma=GAMMA):
    """
    Compute flux using exact Riemann solver at interface.
    UL, UR: shape (3, N_interfaces) or (3,) vector
    Returns flux at interface (x/t = 0) shape (3, ...)
    """
    rhoL, uL, pL = cons_to_prim(UL, gamma=gamma)
    rhoR, uR, pR = cons_to_prim(UR, gamma=gamma)
    
    # Handle both scalar and vector cases
    if UL.ndim == 1:
        # Single interface
        rho_interface, u_interface, p_interface = sample_riemann_solution(
            rhoL, uL, pL, rhoR, uR, pR, 0.0, gamma=gamma
        )
        U_interface = prim_to_cons(rho_interface, u_interface, p_interface, gamma=gamma)
        F = euler_flux(U_interface, gamma=gamma)
    else:
        # Multiple interfaces - process one by one
        n_interfaces = UL.shape[1]
        F = np.zeros_like(UL)
        for i in range(n_interfaces):
            rho_i, u_i, p_i = sample_riemann_solution(
                rhoL[i], uL[i], pL[i], rhoR[i], uR[i], pR[i], 0.0, gamma=gamma
            )
            U_i = prim_to_cons(rho_i, u_i, p_i, gamma=gamma)
            F_i = euler_flux(U_i, gamma=gamma)
            F[:, i] = F_i.flatten()  # Ensure it's 1D
    
    return F

def max_wave_speed(U, gamma=GAMMA):
    rho, u, p = cons_to_prim(U, gamma=gamma)
    c = np.sqrt(gamma * p / rho)
    return np.max(np.abs(u) + c)

# ========== Sod 初始条件 ==========
def sod_initial(nx, xL=0.0, xR=1.0, x0=0.5, gamma=GAMMA):
    # cell centers
    dx = (xR - xL) / nx
    x = np.linspace(xL + 0.5*dx, xR - 0.5*dx, nx)

    rho = np.where(x < x0, 1.0, 0.125)
    u   = np.zeros_like(x)
    p   = np.where(x < x0, 1.0, 0.1)

    U = prim_to_cons(rho, u, p, gamma=gamma)
    return x, U

# ========== 一阶 Godunov 步进 ==========
def godunov_step(U, dx, dt, gamma=GAMMA):
    """
    单步 FV 更新，使用精确 Riemann 求解器，一阶 Godunov 型。
    U: shape (3, N)
    """
    # 左右状态
    UL = U[:, :-1]  # shape (3, N-1)
    UR = U[:,  1:]  # shape (3, N-1)

    F_int = exact_riemann_flux(UL, UR, gamma=gamma)  # shape (3, N-1)

    U_new = U.copy()
    # 内部 cell 更新： i=1..N-2
    U_new[:, 1:-1] -= dt / dx * (F_int[:, 1:] - F_int[:, :-1])

    # 边界：简单的"外推"：copy 邻居
    U_new[:, 0]  = U_new[:, 1]
    U_new[:, -1] = U_new[:, -2]
    return U_new

def run_godunov_sod(nx=400, gamma=GAMMA):
    """Initialize the Sod problem and return grid and initial state."""
    x, U = sod_initial(nx, gamma=gamma)
    return x, U

# ========== breakpoint 检测：shock & contact ==========
def find_shock_index(rho, p):
    """
    非常简单粗暴的 shock 检测：
    - 在压强跳最大的接口附近找 密度跳 最大的位置
    """
    drho = np.abs(rho[1:] - rho[:-1])  # N-1
    dp   = np.abs(p[1:]   - p[:-1])    # N-1

    # shock 处压强跳一般最大
    i_dp_max = int(np.argmax(dp))

    # 在压强较大的区域里，再用 drho refine 一下
    dp_max = dp[i_dp_max]
    mask = dp > 0.5 * dp_max
    cand = np.where(mask)[0]
    if cand.size > 0:
        i_s = int(cand[np.argmax(drho[cand])])
    else:
        i_s = i_dp_max

    return i_s  # 对应接口 i_s+1/2

def find_contact_index(rho, p):
    """
    粗糙接触检测：
    - 接触：密度跳大，但压强跳小
    """
    drho = np.abs(rho[1:] - rho[:-1])
    dp   = np.abs(p[1:]   - p[:-1])

    # 压强最大跳（shock）先去掉一个
    i_shock = int(np.argmax(dp))
    dp_shock = dp[i_shock]

    # 接触处 dp 很小
    small_dp_mask = dp < 0.1 * dp_shock

    cand = np.where(small_dp_mask)[0]
    if cand.size == 0:
        return None

    # 在这些压强变化小的接口中，找密度跳最大的
    i_c = int(cand[np.argmax(drho[cand])])
    return i_c  # 接口 i_c+1/2

# ========== demo：跑一遍 Sod 然后标出 breakpoint ==========
if __name__ == "__main__":
    nx    = 2000
    t_end = 0.01
    cfl   = 0.9

    # Initialize
    x, U = run_godunov_sod(nx=nx)
    dx = x[1] - x[0]
    
    # Time loop
    t = 0.0
    # while t < t_end:
    for i in range(1):
        amax = max_wave_speed(U, gamma=GAMMA)
        dt = cfl * dx / amax
        if t + dt > t_end:
            dt = t_end - t
        U = godunov_step(U, dx, dt, gamma=GAMMA)
        t += dt
    
    rho, u, p = cons_to_prim(U)

    # 找 shock index & contact index
    i_shock   = find_shock_index(rho, p)
    i_contact = find_contact_index(rho, p)

    dx = x[1] - x[0]
    # 接口位置在 cell 中心之间
    if i_shock is not None:
        x_shock = 0.5 * (x[i_shock] + x[i_shock + 1])
    else:
        x_shock = None

    if i_contact is not None:
        x_contact = 0.5 * (x[i_contact] + x[i_contact + 1])
    else:
        x_contact = None

    print(f"t = {t_end}")
    if x_shock is not None:
        print(f"Estimated shock at interface index {i_shock}, x ≈ {x_shock:.6f}")
    else:
        print("Shock not detected")

    if x_contact is not None:
        print(f"Estimated contact at interface index {i_contact}, x ≈ {x_contact:.6f}")
    else:
        print("Contact not detected")

    # 计算精确解
    x_exact = x  # 使用相同的网格点
    rho_exact, u_exact, p_exact = sample_sod_exact(x_exact, t_end, x0=0.5, gamma=GAMMA)

    # 计算理论波速和位置
    x0 = 0.5
    (rhoL, uL, pL), (rhoR, uR, pR), p_star, u_star, rho_star_L, rho_star_R = riemann_sod(GAMMA)
    cL = math.sqrt(GAMMA * pL / rhoL)
    cR = math.sqrt(GAMMA * pR / rhoR)
    c_star_L = math.sqrt(GAMMA * p_star / rho_star_L)
    
    # 波速
    S_C = u_star  # 接触间断速度
    S_R = (rho_star_R * u_star - rhoR * uR) / (rho_star_R - rhoR)  # 激波速度
    
    # 理论位置（在时间 t_end）
    x_shock_theory = x0 + S_R * t_end
    x_contact_theory = x0 + S_C * t_end

    print(f"\nTheoretical wave positions at t = {t_end}:")
    print(f"  Shock (theoretical):     x = {x_shock_theory:.6f}")
    print(f"  Contact (theoretical):   x = {x_contact_theory:.6f}")
    if x_shock is not None:
        print(f"  Shock (numerical):       x = {x_shock:.6f} (error: {abs(x_shock - x_shock_theory):.6f})")
    if x_contact is not None:
        print(f"  Contact (numerical):     x = {x_contact:.6f} (error: {abs(x_contact - x_contact_theory):.6f})")

    # 画图：数值解 vs 精确解
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # Plot density
    axes[0].plot(x, rho, label="rho (Exact Riemann)", lw=1.5, alpha=0.7)
    axes[0].plot(x_exact, rho_exact, label="rho (exact)", lw=1.5, linestyle="--", alpha=0.8)
    # Theoretical positions
    axes[0].axvline(x_shock_theory, color="r", linestyle="-", linewidth=2, label="shock (theoretical)", alpha=0.8)
    axes[0].axvline(x_contact_theory, color="g", linestyle="-", linewidth=2, label="contact (theoretical)", alpha=0.8)
    # Numerical breakpoints
    if x_shock is not None:
        axes[0].axvline(x_shock, color="r", linestyle=":", linewidth=1.5, label="shock (numerical)", alpha=0.6)
    if x_contact is not None:
        axes[0].axvline(x_contact, color="g", linestyle=":", linewidth=1.5, label="contact (numerical)", alpha=0.6)
    axes[0].set_ylabel("rho")
    axes[0].set_title(f"Sod 1D, t = {t_end}, first-order Godunov (Exact Riemann) vs Exact")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot velocity
    axes[1].plot(x, u, label="u (Exact Riemann)", lw=1.5, alpha=0.7)
    axes[1].plot(x_exact, u_exact, label="u (exact)", lw=1.5, linestyle="--", alpha=0.8)
    # Theoretical positions
    axes[1].axvline(x_shock_theory, color="r", linestyle="-", linewidth=2, label="shock (theoretical)", alpha=0.8)
    axes[1].axvline(x_contact_theory, color="g", linestyle="-", linewidth=2, label="contact (theoretical)", alpha=0.8)
    # Numerical breakpoints
    if x_shock is not None:
        axes[1].axvline(x_shock, color="r", linestyle=":", linewidth=1.5, label="shock (numerical)", alpha=0.6)
    if x_contact is not None:
        axes[1].axvline(x_contact, color="g", linestyle=":", linewidth=1.5, label="contact (numerical)", alpha=0.6)
    axes[1].set_ylabel("u")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot pressure
    axes[2].plot(x, p, label="p (Exact Riemann)", lw=1.5, alpha=0.7)
    axes[2].plot(x_exact, p_exact, label="p (exact)", lw=1.5, linestyle="--", alpha=0.8)
    # Theoretical positions
    axes[2].axvline(x_shock_theory, color="r", linestyle="-", linewidth=2, label="shock (theoretical)", alpha=0.8)
    axes[2].axvline(x_contact_theory, color="g", linestyle="-", linewidth=2, label="contact (theoretical)", alpha=0.8)
    # Numerical breakpoints
    if x_shock is not None:
        axes[2].axvline(x_shock, color="r", linestyle=":", linewidth=1.5, label="shock (numerical)", alpha=0.6)
    if x_contact is not None:
        axes[2].axvline(x_contact, color="g", linestyle=":", linewidth=1.5, label="contact (numerical)", alpha=0.6)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("p")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
