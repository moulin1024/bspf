import numpy as np
import matplotlib.pyplot as plt

# ----------------- Physical constants -----------------
GAMMA = 1.4
RHO_FLOOR = 1e-8
P_FLOOR   = 1e-8

# ----------------- Euler utilities -----------------
def cons_to_prim(U, gamma=GAMMA):
    """
    U: shape (3, Nx) conservative variables [rho, rho*u, E]
    Returns: rho, u, p (each shape (Nx,))
    """
    rho = np.maximum(U[0], RHO_FLOOR)
    m   = U[1]
    E   = U[2]
    u = m / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * u**2)
    p = np.maximum(p, P_FLOOR)
    return rho, u, p

def prim_to_cons(rho, u, p, gamma=GAMMA):
    """
    rho, u, p: shape (Nx,) (or scalar), returns U: shape (3, Nx)
    """
    E = p / (gamma - 1.0) + 0.5 * rho * u**2
    m = rho * u
    return np.stack([rho, m, E], axis=0)

def euler_flux(U, gamma=GAMMA):
    """
    Physical flux F(U) for 1D Euler.
    U: (3, Nx)
    F: (3, Nx)
    """
    rho, u, p = cons_to_prim(U, gamma=gamma)
    F = np.zeros_like(U)
    F[0] = rho * u
    F[1] = rho * u**2 + p
    F[2] = (U[2] + p) * u
    return F

def enforce_physical(U, gamma=GAMMA):
    """
    Clamp rho, p to be positive; mildly limit u.
    This breaks strict conservation but keeps things from blowing up.
    """
    rho, u, p = cons_to_prim(U, gamma=gamma)
    umax = 10.0
    u = np.clip(u, -umax, umax)
    rho = np.maximum(rho, RHO_FLOOR)
    p   = np.maximum(p, P_FLOOR)
    return prim_to_cons(rho, u, p, gamma=gamma)

# ----------------- Rankine–Hugoniot shock speed -----------------
def shock_speed_RH(UL, UR, gamma=GAMMA):
    """
    Compute shock speed from Rankine–Hugoniot using the mass jump condition:
        s (rho_R - rho_L) = rho_R u_R - rho_L u_L
        => s = (rho_R u_R - rho_L u_L) / (rho_R - rho_L)
    UL, UR: shape (3,) left/right conservative states (at the shock)
    """
    rhoL, uL, pL = cons_to_prim(UL[:, None], gamma=gamma)
    rhoR, uR, pR = cons_to_prim(UR[:, None], gamma=gamma)
    rhoL = rhoL[0]; uL = uL[0]
    rhoR = rhoR[0]; uR = uR[0]

    if abs(rhoR - rhoL) < 1e-12:
        return 0.0
    s = (rhoR * uR - rhoL * uL) / (rhoR - rhoL)
    return s

# ----------------- 2nd-order FD RHS (smooth regions only) -----------------
def rhs_euler_fd(U, x, Xs, dx, gamma=GAMMA, nu_visc=0.1):
    """
    Compute RHS dU/dt = -F_x + artificial viscosity, 
    but **avoid crossing the tracked shock** at Xs.

    U: (3, Nx)
    x: (Nx,)
    Xs: float, shock position
    dx: grid spacing
    gamma: adiabatic index
    nu_visc: dimensionless viscosity coefficient

    We:
    - use 2nd-order central difference for F_x in smooth regions,
    - skip stencils that would cross the shock,
    - add simple Laplacian viscosity in smooth regions.
    """
    Nx = U.shape[1]
    F = euler_flux(U, gamma=gamma)
    dU_dt = np.zeros_like(U)

    # locate cell index k such that x[k] <= Xs < x[k+1]
    k = np.searchsorted(x, Xs) - 1
    k = np.clip(k, 1, Nx-3)

    # ---- flux derivative: central differences away from shock ----
    # left smooth region: i = 1..k-1
    for i in range(1, k):
        dF_dx = (F[:, i+1] - F[:, i-1]) / (2.0 * dx)
        dU_dt[:, i] = -dF_dx

    # right smooth region: i = k+2..Nx-2
    for i in range(k+2, Nx-1):
        dF_dx = (F[:, i+1] - F[:, i-1]) / (2.0 * dx)
        dU_dt[:, i] = -dF_dx

    # boundaries: keep dU_dt = 0 (Dirichlet BC will fix U)
    dU_dt[:, 0]  = 0.0
    dU_dt[:, -1] = 0.0

    # ---- simple artificial viscosity in smooth regions ----
    # U_xx ≈ (U[i+1] - 2U[i] + U[i-1]) / dx^2
    if nu_visc > 0.0:
        for i in range(2, k-1):
            U_xx = (U[:, i+1] - 2*U[:, i] + U[:, i-1]) / (dx*dx)
            dU_dt[:, i] += nu_visc * U_xx
        for i in range(k+3, Nx-2):
            U_xx = (U[:, i+1] - 2*U[:, i] + U[:, i-1]) / (dx*dx)
            dU_dt[:, i] += nu_visc * U_xx

    return dU_dt

# ----------------- Sod initial condition -----------------
def sod_initial(x, x0=0.5, gamma=GAMMA):
    """
    Sod shock tube initial condition on grid x.
    Left:  rho=1,   u=0, p=1
    Right: rho=0.125, u=0, p=0.1
    """
    rhoL, uL, pL   = 1.0,   0.0, 1.0
    rhoR, uR, pR   = 0.125, 0.0, 0.1
    rho = np.where(x < x0, rhoL, rhoR)
    u   = np.zeros_like(x)
    p   = np.where(x < x0, pL, pR)
    U   = prim_to_cons(rho, u, p, gamma=gamma)
    return U, (rhoL, uL, pL), (rhoR, uR, pR)

# ----------------- Main shock-fitting solver -----------------
def solve_sod_shock_fitting_euler(
        Nx=401,
        xL=0.0,
        xR=1.0,
        x0=0.5,
        t_end=0.2,
        CFL=0.4,
        gamma=GAMMA
    ):
    """
    1D Euler "mock" shock-fitting for Sod:
    - Track the primary right-going shock using Rankine–Hugoniot (mass jump).
    - Inside each step:
        * use 2nd-order FD to update smooth regions away from the shock
        * enforce a sharp jump at the tracked shock position

    NOTE:
    - This is an educational mock, not a high-fidelity scheme.
    - We only treat the right-moving shock specially; rarefaction + contact
      are still handled by the crude FD + viscosity.
    """
    # grid
    x = np.linspace(xL, xR, Nx)
    dx = x[1] - x[0]

    # initial condition
    U, left_state, right_state = sod_initial(x, x0=x0, gamma=gamma)
    rhoL0, uL0, pL0 = left_state
    rhoR0, uR0, pR0 = right_state

    # initial shock location ~ discontinuity location
    Xs = x0

    # initial guess for left/right shock states:
    # start from left/right constant states
    UL_s = prim_to_cons(rhoL0, uL0, pL0, gamma=gamma).flatten()
    UR_s = prim_to_cons(rhoR0, uR0, pR0, gamma=gamma).flatten()

    t = 0.0
    step = 0

    history = [(t, Xs, U.copy())]

    while t < t_end - 1e-12:
        # compute max wave speed for CFL
        rho, u, p = cons_to_prim(U, gamma=gamma)
        c = np.sqrt(gamma * p / rho)
        amax = np.max(np.abs(u) + c)
        dt = CFL * dx / amax
        if t + dt > t_end:
            dt = t_end - t

        # locate which cell contains the shock
        k = np.searchsorted(x, Xs) - 1
        k = np.clip(k, 0, Nx-2)

        # take left/right states at the shock from the grid
        UL = U[:, k].copy()
        UR = U[:, k+1].copy()

        # compute shock speed via Rankine–Hugoniot
        s = shock_speed_RH(UL, UR, gamma=gamma)

        # FD update in smooth regions
        dU_dt = rhs_euler_fd(U, x, Xs, dx, gamma=gamma, nu_visc=0.1)
        U_new = U + dt * dU_dt

        # enforce Dirichlet BCs (same as initial left/right states)
        UL_bc = prim_to_cons(rhoL0, uL0, pL0, gamma=gamma).flatten()
        UR_bc = prim_to_cons(rhoR0, uR0, pR0, gamma=gamma).flatten()
        U_new[:, 0]  = UL_bc
        U_new[:, -1] = UR_bc

        U_new = enforce_physical(U_new, gamma=gamma)

        # update shock position
        Xs_new = Xs + s * dt

        # re-impose a sharp jump at the new shock location:
        #   find new cell index k_new, set U[:,k_new] ~ left state,
        #   U[:,k_new+1] ~ right state.
        k_new = np.searchsorted(x, Xs_new) - 1
        k_new = np.clip(k_new, 0, Nx-2)

        # use UL, UR computed before to "sharpen" the jump
        U_new[:, k_new]   = UL
        U_new[:, k_new+1] = UR

        # update state
        U = U_new
        Xs = Xs_new
        t += dt
        step += 1

        history.append((t, Xs, U.copy()))

        if step % 20 == 0:
            print(f"step={step}, t={t:.4f}, dt={dt:.3e}, Xs={Xs:.4f}, s={s:.4f}")

    return x, history

# ----------------- Demo run -----------------
if __name__ == "__main__":
    x, history = solve_sod_shock_fitting_euler(
        Nx=401,
        xL=0.0,
        xR=1.0,
        x0=0.5,
        t_end=0.2,
        CFL=0.4
    )

    t_final, Xs_final, U_final = history[-1]
    rho, u, p = cons_to_prim(U_final)

    print(f"\nFinal time t = {t_final:.3f}")
    print(f"Tracked shock position X_s(t) ≈ {Xs_final:.4f}")

    plt.figure(figsize=(10, 6))
    plt.subplot(3,1,1)
    plt.plot(x, rho, "-k", lw=1.5)
    plt.axvline(Xs_final, color="r", linestyle="--", label="tracked shock")
    plt.ylabel("rho")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(3,1,2)
    plt.plot(x, u, "-k", lw=1.5)
    plt.axvline(Xs_final, color="r", linestyle="--")
    plt.ylabel("u")
    plt.grid(alpha=0.3)

    plt.subplot(3,1,3)
    plt.plot(x, p, "-k", lw=1.5)
    plt.axvline(Xs_final, color="r", linestyle="--")
    plt.ylabel("p")
    plt.xlabel("x")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
