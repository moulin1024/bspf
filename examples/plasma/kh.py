import numpy as np
import matplotlib.pyplot as plt

# ------------------ constants ------------------
GAMMA = 5.0 / 3.0
P_FLOOR = 1e-6
RHO_FLOOR = 1e-8

# U = [rho, mx, my, E, Bx, By]


# ------------------ cons -> primitive ------------------
def cons_to_prim(U, gamma=GAMMA):
    rho = U[..., 0]
    rho = np.maximum(rho, RHO_FLOOR)
    mx  = U[..., 1]
    my  = U[..., 2]
    E   = U[..., 3]
    Bx  = U[..., 4]
    By  = U[..., 5]

    vx = mx / rho
    vy = my / rho
    v2 = vx**2 + vy**2
    B2 = Bx**2 + By**2

    p = (gamma - 1.0) * (E - 0.5 * rho * v2 - 0.5 * B2)
    p = np.maximum(p, P_FLOOR)

    return rho, vx, vy, p, Bx, By


# ------------------ physical fluxes ------------------
def flux_x(U, gamma=GAMMA):
    rho, vx, vy, p, Bx, By = cons_to_prim(U, gamma)
    E   = U[..., 3]
    B2  = Bx**2 + By**2
    pt  = p + 0.5 * B2
    vdotB = vx * Bx + vy * By

    F = np.zeros_like(U)
    F[..., 0] = rho * vx
    F[..., 1] = rho * vx * vx + pt - Bx * Bx
    F[..., 2] = rho * vx * vy - Bx * By
    F[..., 3] = (E + pt) * vx - Bx * vdotB
    F[..., 4] = 0.0
    F[..., 5] = vx * By - vy * Bx

    return F, rho, vx, vy, p, Bx, By


def flux_y(U, gamma=GAMMA):
    rho, vx, vy, p, Bx, By = cons_to_prim(U, gamma)
    E   = U[..., 3]
    B2  = Bx**2 + By**2
    pt  = p + 0.5 * B2
    vdotB = vx * Bx + vy * By

    G = np.zeros_like(U)
    G[..., 0] = rho * vy
    G[..., 1] = rho * vx * vy - Bx * By
    G[..., 2] = rho * vy * vy + pt - By * By
    G[..., 3] = (E + pt) * vy - By * vdotB
    G[..., 4] = vy * Bx - vx * By
    G[..., 5] = 0.0

    return G, rho, vx, vy, p, Bx, By


# ------------------ fast magnetosonic speed (for HLL) ------------------
def fast_speed_x(rho, p, Bx, By, gamma=GAMMA):
    """
    Fast speed in x-direction (normal B = Bx).
    """
    a2  = gamma * p / rho
    B2  = Bx**2 + By**2
    vA2 = B2 / rho
    vAn2 = Bx**2 / rho

    term = a2 + vA2
    disc = term**2 - 4.0 * a2 * vAn2
    disc = np.maximum(disc, 0.0)
    cf2  = 0.5 * (term + np.sqrt(disc))
    cf2  = np.maximum(cf2, 0.0)
    return np.sqrt(cf2)


def fast_speed_y(rho, p, Bx, By, gamma=GAMMA):
    """
    Fast speed in y-direction (normal B = By).
    """
    a2  = gamma * p / rho
    B2  = Bx**2 + By**2
    vA2 = B2 / rho
    vAn2 = By**2 / rho

    term = a2 + vA2
    disc = term**2 - 4.0 * a2 * vAn2
    disc = np.maximum(disc, 0.0)
    cf2  = 0.5 * (term + np.sqrt(disc))
    cf2  = np.maximum(cf2, 0.0)
    return np.sqrt(cf2)


# ------------------ HLL (1st order, no MUSCL) ------------------
def hll_flux_x(U, gamma=GAMMA):
    """
    First-order HLL flux in x:
      interface at i+1/2: UL = U[i], UR = U[i+1]
    """
    UL = U[:-1, :, :]
    UR = U[ 1:, :, :]

    FL, rhoL, vxL, vyL, pL, BxL, ByL = flux_x(UL, gamma)
    FR, rhoR, vxR, vyR, pR, BxR, ByR = flux_x(UR, gamma)

    cfL = fast_speed_x(rhoL, pL, BxL, ByL, gamma)
    cfR = fast_speed_x(rhoR, pR, BxR, ByR, gamma)

    SL = np.minimum(vxL - cfL, vxR - cfR)
    SR = np.maximum(vxL + cfL, vxR + cfR)

    denom = SR - SL
    denom = np.where(denom < 1e-8, 1e-8, denom)

    F_mid = (SR[..., None] * FL - SL[..., None] * FR +
             SL[..., None] * SR[..., None] * (UR - UL)) / denom[..., None]

    F_HLL = np.where(SL[..., None] >= 0.0, FL,
             np.where(SR[..., None] <= 0.0, FR, F_mid))

    return F_HLL   # (Nx+1, Ny+2, 6)


def hll_flux_y(U, gamma=GAMMA):
    """
    First-order HLL flux in y:
      interface at j+1/2: UL = U[:,j], UR = U[:,j+1]
    """
    UL = U[:, :-1, :]
    UR = U[:,  1:, :]

    GL, rhoL, vxL, vyL, pL, BxL, ByL = flux_y(UL, gamma)
    GR, rhoR, vxR, vyR, pR, BxR, ByR = flux_y(UR, gamma)

    cfL = fast_speed_y(rhoL, pL, BxL, ByL, gamma)
    cfR = fast_speed_y(rhoR, pR, BxR, ByR, gamma)

    SL = np.minimum(vyL - cfL, vyR - cfR)
    SR = np.maximum(vyL + cfL, vyR + cfR)

    denom = SR - SL
    denom = np.where(denom < 1e-8, 1e-8, denom)

    G_mid = (SR[..., None] * GL - SL[..., None] * GR +
             SL[..., None] * SR[..., None] * (UR - UL)) / denom[..., None]

    G_HLL = np.where(SL[..., None] >= 0.0, GL,
             np.where(SR[..., None] <= 0.0, GR, G_mid))

    return G_HLL   # (Nx+2, Ny+1, 6)


# ------------------ boundary conditions (vectorized, no for) ------------------
def apply_bc(U, gamma=GAMMA):
    """
    1 ghost layer:
      i = 0, Nx+1 : left/right ghosts
      j = 0, Ny+1 : bottom/top ghosts

    Left/right: outflow (zero-gradient)
    Top/bottom: reflecting wall (vy -> -vy)
    """
    Nx_tot, Ny_tot, nvar = U.shape
    Nx = Nx_tot - 2
    Ny = Ny_tot - 2

    # left/right outflow
    U[0,      :, :] = U[1,     :, :]
    U[Nx+1,   :, :] = U[Nx,    :, :]

    # bottom wall from interior j=1
    Ui = U[:, 1, :].copy()    # shape (Nx+2, 6)
    rho = np.maximum(Ui[:, 0], RHO_FLOOR)
    mx  = Ui[:, 1]
    my  = Ui[:, 2]
    E   = Ui[:, 3]
    Bx  = Ui[:, 4]
    By  = Ui[:, 5]

    vx = mx / rho
    vy = my / rho
    v2 = vx**2 + vy**2
    B2 = Bx**2 + By**2
    p  = (gamma - 1.0) * (E - 0.5 * rho * v2 - 0.5 * B2)
    p  = np.maximum(p, P_FLOOR)

    vx_g = vx
    vy_g = -vy
    rho_g = rho
    p_g   = p
    Bx_g  = Bx
    By_g  = By

    v2_g = vx_g**2 + vy_g**2
    B2_g = Bx_g**2 + By_g**2
    mx_g = rho_g * vx_g
    my_g = rho_g * vy_g
    E_g  = p_g/(gamma - 1.0) + 0.5 * rho_g * v2_g + 0.5 * B2_g

    U[:, 0, 0] = rho_g
    U[:, 0, 1] = mx_g
    U[:, 0, 2] = my_g
    U[:, 0, 3] = E_g
    U[:, 0, 4] = Bx_g
    U[:, 0, 5] = By_g

    # top wall from interior j=Ny
    Ui = U[:, Ny, :].copy()
    rho = np.maximum(Ui[:, 0], RHO_FLOOR)
    mx  = Ui[:, 1]
    my  = Ui[:, 2]
    E   = Ui[:, 3]
    Bx  = Ui[:, 4]
    By  = Ui[:, 5]

    vx = mx / rho
    vy = my / rho
    v2 = vx**2 + vy**2
    B2 = Bx**2 + By**2
    p  = (gamma - 1.0) * (E - 0.5 * rho * v2 - 0.5 * B2)
    p  = np.maximum(p, P_FLOOR)

    vx_g = vx
    vy_g = -vy
    rho_g = rho
    p_g   = p
    Bx_g  = Bx
    By_g  = By

    v2_g = vx_g**2 + vy_g**2
    B2_g = Bx_g**2 + By_g**2
    mx_g = rho_g * vx_g
    my_g = rho_g * vy_g
    E_g  = p_g/(gamma - 1.0) + 0.5 * rho_g * v2_g + 0.5 * B2_g

    U[:, Ny+1, 0] = rho_g
    U[:, Ny+1, 1] = mx_g
    U[:, Ny+1, 2] = my_g
    U[:, Ny+1, 3] = E_g
    U[:, Ny+1, 4] = Bx_g
    U[:, Ny+1, 5] = By_g

    return U


# ------------------ RHS (finite-volume) ------------------
def compute_rhs(U, dx, dy, gamma=GAMMA):
    U = apply_bc(U, gamma)

    Nx_tot, Ny_tot, _ = U.shape
    Nx = Nx_tot - 2
    Ny = Ny_tot - 2

    Fx = hll_flux_x(U, gamma)   # (Nx+1, Ny+2, 6)
    Gy = hll_flux_y(U, gamma)   # (Nx+2, Ny+1, 6)

    dFdx = (Fx[1:, :, :] - Fx[:-1, :, :]) / dx      # (Nx,   Ny+2, 6)
    dGdy = (Gy[:, 1:, :] - Gy[:, :-1, :]) / dy      # (Nx+2, Ny,   6)

    RHS = np.zeros_like(U)
    RHS[1:Nx+1, 1:Ny+1, :] = - dFdx[:, 1:Ny+1, :] - dGdy[1:Nx+1, :, :]

    return RHS


# ------------------ CFL dt ------------------
def compute_dt(U, dx, dy, CFL=0.4, gamma=GAMMA):
    rho, vx, vy, p, Bx, By = cons_to_prim(U, gamma)
    # 简化 fast 速度估计，用 a^2 + B^2/rho
    B2 = Bx**2 + By**2
    a  = np.sqrt(gamma * p / rho)
    vA = np.sqrt(B2 / rho)
    cfast = np.sqrt(a*a + vA*vA)

    sx = np.abs(vx) + cfast
    sy = np.abs(vy) + cfast
    smax = max(np.max(sx), np.max(sy))

    if smax <= 0.0:
        return 1e-3
    return CFL * min(dx, dy) / smax


# ------------------ grid & KH initial condition ------------------
Nx, Ny = 400, 200
Lx, Ly = 2.0, 1.0
x = np.linspace(0.0, Lx, Nx)
y = np.linspace(0.0, Ly, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y, indexing="ij")

rho0 = 1.0
p0   = 1.0
U0 = np.zeros((Nx, Ny, 6))
U0[..., 0] = rho0

# shear flow vx(y)
U_shear = 0.1
a_shear = 0.01 * Ly
vx = U_shear * np.tanh((Y - 0.5 * Ly) / a_shear)

# KH perturb in vy (zero at walls)
eps = 0.1
kx  = 2.0 * np.pi / Lx
vy = eps * np.sin(kx * X) * np.sin(np.pi * Y / Ly)

# uniform Bx
Bx = 0.5 * np.ones_like(vx)
By = np.zeros_like(vx)

v2 = vx**2 + vy**2
B2 = Bx**2 + By**2
E  = p0 / (GAMMA - 1.0) + 0.5 * rho0 * v2 + 0.5 * B2

U0[..., 1] = rho0 * vx
U0[..., 2] = rho0 * vy
U0[..., 3] = E
U0[..., 4] = Bx
U0[..., 5] = By

# add ghost cells
U = np.zeros((Nx+2, Ny+2, 6))
U[1:-1, 1:-1, :] = U0


# ------------------ time integration: RK2 ------------------
t = 0.0
t_end = 1.0
step = 0
max_steps = 50000

while t < t_end and step < max_steps:
    dt = compute_dt(U[1:-1, 1:-1, :], dx, dy, CFL=0.5, gamma=GAMMA)
    if t + dt > t_end:
        dt = t_end - t

    RHS1 = compute_rhs(U, dx, dy, GAMMA)
    U1   = U + dt * RHS1

    RHS2 = compute_rhs(U1, dx, dy, GAMMA)
    U    = U + 0.5 * dt * (RHS1 + RHS2)

    t    += dt
    step += 1

    if step % 20 == 0 or abs(t - t_end) < 1e-10:
        print(f"step {step}, t = {t:.4f}, dt = {dt:.3e}")

print("done. t = ", t, " steps =", step)


# ------------------ plots ------------------
rho, vx, vy, p, Bx, By = cons_to_prim(U[1:-1, 1:-1, :], GAMMA)

plt.figure(figsize=(8, 3))

plt.subplot(1, 2, 1)
plt.imshow(rho.T, origin="lower", extent=[0, Lx, 0, Ly], aspect="auto")
plt.colorbar(label=r"$\rho$")
plt.xlabel("x"); plt.ylabel("y")
plt.title("Density (final)")

plt.subplot(1, 2, 2)
plt.imshow(vx.T, origin="lower", extent=[0, Lx, 0, Ly], aspect="auto")
plt.colorbar(label=r"$v_x$")
plt.xlabel("x"); plt.ylabel("y")
plt.title(r"$v_x$ (final)")

plt.tight_layout()
plt.show()
