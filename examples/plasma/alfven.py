#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

gamma = 5.0 / 3.0
p_floor = 1e-8


# ---------------------------------------------------------------------
# Primitive/conserved conversions
# ---------------------------------------------------------------------
def prim_from_cons(U):
    '''Primitive variables from a single-cell conserved state U[8].'''
    rho = U[0]
    vx = U[1] / rho
    vy = U[2] / rho
    vz = U[3] / rho
    Bx = U[4]
    By = U[5]
    Bz = U[6]
    E = U[7]

    v2 = vx * vx + vy * vy + vz * vz
    B2 = Bx * Bx + By * By + Bz * Bz

    p = (gamma - 1.0) * (E - 0.5 * rho * v2 - 0.5 * B2)
    if p < p_floor:
        p = p_floor

    return rho, vx, vy, vz, Bx, By, Bz, p


def cons_to_prim(U):
    '''Vectorised: conserved -> primitive for full 2D array.'''
    rho = U[..., 0]
    vx = U[..., 1] / rho
    vy = U[..., 2] / rho
    vz = U[..., 3] / rho
    Bx = U[..., 4]
    By = U[..., 5]
    Bz = U[..., 6]
    E = U[..., 7]

    v2 = vx * vx + vy * vy + vz * vz
    B2 = Bx * Bx + By * By + Bz * Bz

    p = (gamma - 1.0) * (E - 0.5 * rho * v2 - 0.5 * B2)
    p = np.maximum(p, p_floor)

    return rho, vx, vy, vz, Bx, By, Bz, p


# ---------------------------------------------------------------------
# Single-state fluxes (for HLL)
# ---------------------------------------------------------------------
def flux_x_single(U):
    rho, vx, vy, vz, Bx, By, Bz, p = prim_from_cons(U)
    B2 = Bx * Bx + By * By + Bz * Bz
    ptot = p + 0.5 * B2
    Bdotv = Bx * vx + By * vy + Bz * vz
    E = U[7]

    Fx = np.zeros_like(U)
    Fx[0] = rho * vx
    Fx[1] = rho * vx * vx + ptot - Bx * Bx
    Fx[2] = rho * vx * vy - Bx * By
    Fx[3] = rho * vx * vz - Bx * Bz
    Fx[4] = 0.0
    Fx[5] = vx * By - Bx * vy
    Fx[6] = vx * Bz - Bx * vz
    Fx[7] = (E + ptot) * vx - Bx * Bdotv
    return Fx


def flux_y_single(U):
    rho, vx, vy, vz, Bx, By, Bz, p = prim_from_cons(U)
    B2 = Bx * Bx + By * By + Bz * Bz
    ptot = p + 0.5 * B2
    Bdotv = Bx * vx + By * vy + Bz * vz
    E = U[7]

    Fy = np.zeros_like(U)
    Fy[0] = rho * vy
    Fy[1] = rho * vy * vx - By * Bx
    Fy[2] = rho * vy * vy + ptot - By * By
    Fy[3] = rho * vy * vz - By * Bz
    Fy[4] = vy * Bx - By * vx
    Fy[5] = 0.0
    Fy[6] = vy * Bz - By * vz
    Fy[7] = (E + ptot) * vy - By * Bdotv
    return Fy


# ---------------------------------------------------------------------
# Single-state fast magnetosonic speeds
# ---------------------------------------------------------------------
def fast_speed_x_single(U):
    rho, vx, vy, vz, Bx, By, Bz, p = prim_from_cons(U)
    B2 = Bx * Bx + By * By + Bz * Bz
    a2 = B2 / rho
    cs2 = gamma * p / rho
    if cs2 < 0.0:
        cs2 = 0.0
    a_n2 = Bx * Bx / rho
    term = (a2 + cs2) ** 2 - 4.0 * a_n2 * cs2
    if term < 0.0:
        term = 0.0
    cf2 = 0.5 * ((a2 + cs2) + np.sqrt(term))
    if cf2 < 0.0:
        cf2 = 0.0
    cf = np.sqrt(cf2)
    return vx, cf


def fast_speed_y_single(U):
    rho, vx, vy, vz, Bx, By, Bz, p = prim_from_cons(U)
    B2 = Bx * Bx + By * By + Bz * Bz
    a2 = B2 / rho
    cs2 = gamma * p / rho
    if cs2 < 0.0:
        cs2 = 0.0
    a_n2 = By * By / rho
    term = (a2 + cs2) ** 2 - 4.0 * a_n2 * cs2
    if term < 0.0:
        term = 0.0
    cf2 = 0.5 * ((a2 + cs2) + np.sqrt(term))
    if cf2 < 0.0:
        cf2 = 0.0
    cf = np.sqrt(cf2)
    return vy, cf


# ---------------------------------------------------------------------
# HLL fluxes
# ---------------------------------------------------------------------
def hll_flux_x(UL, UR):
    FL = flux_x_single(UL)
    FR = flux_x_single(UR)
    vxL, cfL = fast_speed_x_single(UL)
    vxR, cfR = fast_speed_x_single(UR)

    sL = min(vxL - cfL, vxR - cfR, 0.0)
    sR = max(vxL + cfL, vxR + cfR, 0.0)

    if sL >= 0.0:
        return FL
    if sR <= 0.0:
        return FR

    denom = sR - sL
    if abs(denom) < 1e-12:
        return 0.5 * (FL + FR)

    return (sR * FL - sL * FR + sL * sR * (UR - UL)) / denom


def hll_flux_y(UL, UR):
    FL = flux_y_single(UL)
    FR = flux_y_single(UR)
    vyL, cfL = fast_speed_y_single(UL)
    vyR, cfR = fast_speed_y_single(UR)

    sL = min(vyL - cfL, vyR - cfR, 0.0)
    sR = max(vyL + cfL, vyR + cfR, 0.0)

    if sL >= 0.0:
        return FL
    if sR <= 0.0:
        return FR

    denom = sR - sL
    if abs(denom) < 1e-12:
        return 0.5 * (FL + FR)

    return (sR * FL - sL * FR + sL * sR * (UR - UL)) / denom


# ---------------------------------------------------------------------
# Wave speeds for CFL
# ---------------------------------------------------------------------
def compute_wave_speeds(rho, vx, vy, Bx, By, Bz, p):
    '''Fast magnetosonic speeds in x and y for CFL estimate.'''
    B2 = Bx ** 2 + By ** 2 + Bz ** 2
    a2 = B2 / rho
    cs2 = gamma * p / rho
    cs2 = np.maximum(cs2, 0.0)

    # x-direction
    a_n2_x = Bx ** 2 / rho
    term_x = (a2 + cs2) ** 2 - 4.0 * a_n2_x * cs2
    term_x = np.maximum(term_x, 0.0)
    cf2_x = 0.5 * ((a2 + cs2) + np.sqrt(term_x))
    cf2_x = np.maximum(cf2_x, 0.0)
    cf_x = np.sqrt(cf2_x)
    s_x = np.abs(vx) + cf_x

    # y-direction
    a_n2_y = By ** 2 / rho
    term_y = (a2 + cs2) ** 2 - 4.0 * a_n2_y * cs2
    term_y = np.maximum(term_y, 0.0)
    cf2_y = 0.5 * ((a2 + cs2) + np.sqrt(term_y))
    cf2_y = np.maximum(cf2_y, 0.0)
    cf_y = np.sqrt(cf2_y)
    s_y = np.abs(vy) + cf_y

    return s_x, s_y


# ---------------------------------------------------------------------
# Boundary conditions (Harris sheet reconnection)
# ---------------------------------------------------------------------
def apply_bc_reconnection(U, params):
    '''
    Non-periodic BCs for Harris current sheet reconnection.
    Top/bottom: fixed upstream Harris equilibrium.
    Left/right: outflow (zero-gradient).
    '''
    B0 = params["B0"]
    Bg = params["Bg"]
    rho0 = params["rho0"]
    p_inf = params["p_inf"]
    Ly = params["Ly"]
    a = params["a"]

    Ny_tot, Nx_tot, _ = U.shape
    Ny = Ny_tot - 2

    # Left/right: copy nearest interior
    U[1:-1, 0, :] = U[1:-1, 1, :]
    U[1:-1, -1, :] = U[1:-1, -2, :]

    # Bottom (y = -Ly)
    yb = -Ly
    tanh_b = np.tanh(yb / a)
    sech2_b = 1.0 / np.cosh(yb / a) ** 2
    Bx_b = B0 * tanh_b
    p_b = p_inf + 0.5 * B0 ** 2 * sech2_b
    rho_b = rho0
    B2_b = Bx_b ** 2 + Bg ** 2
    E_b = p_b / (gamma - 1.0) + 0.5 * rho_b * 0.0 + 0.5 * B2_b

    U[0, :, 0] = rho_b
    U[0, :, 1] = 0.0
    U[0, :, 2] = 0.0
    U[0, :, 3] = 0.0
    U[0, :, 4] = Bx_b
    U[0, :, 5] = 0.0
    U[0, :, 6] = Bg
    U[0, :, 7] = E_b

    # Top (y = +Ly)
    yt = Ly
    tanh_t = np.tanh(yt / a)
    sech2_t = 1.0 / np.cosh(yt / a) ** 2
    Bx_t = B0 * tanh_t
    p_t = p_inf + 0.5 * B0 ** 2 * sech2_t
    rho_t = rho0
    B2_t = Bx_t ** 2 + Bg ** 2
    E_t = p_t / (gamma - 1.0) + 0.5 * rho_t * 0.0 + 0.5 * B2_t

    U[-1, :, 0] = rho_t
    U[-1, :, 1] = 0.0
    U[-1, :, 2] = 0.0
    U[-1, :, 3] = 0.0
    U[-1, :, 4] = Bx_t
    U[-1, :, 5] = 0.0
    U[-1, :, 6] = Bg
    U[-1, :, 7] = E_t


# ---------------------------------------------------------------------
# RHS with HLL flux
# ---------------------------------------------------------------------
def rhs(U, dx, dy, params):
    '''Compute dU/dt using HLL fluxes and reconnection BCs (vectorized).'''
    apply_bc_reconnection(U, params)

    Ny_tot, Nx_tot, _ = U.shape
    Ny = Ny_tot - 2
    Nx = Nx_tot - 2

    dUdt = np.zeros_like(U)

    # Interior slice indices
    iy_slice = slice(1, Ny + 1)
    ix_slice = slice(1, Nx + 1)

    # Vectorized HLL flux computation for x-direction interfaces
    # Left interfaces: between [iy, ix-1] and [iy, ix] for all interior cells
    UL_x_L = U[iy_slice, 0:Nx, :]      # shape (Ny, Nx, 8)
    UR_x_L = U[iy_slice, 1:Nx+1, :]    # shape (Ny, Nx, 8)
    F_x_L = np.array([hll_flux_x(UL_x_L[iy, ix, :], UR_x_L[iy, ix, :]) 
                      for iy in range(Ny) for ix in range(Nx)]).reshape(Ny, Nx, 8)
    
    # Right interfaces: between [iy, ix] and [iy, ix+1] for all interior cells
    UL_x_R = U[iy_slice, 1:Nx+1, :]    # shape (Ny, Nx, 8)
    UR_x_R = U[iy_slice, 2:Nx+2, :]    # shape (Ny, Nx, 8)
    F_x_R = np.array([hll_flux_x(UL_x_R[iy, ix, :], UR_x_R[iy, ix, :]) 
                      for iy in range(Ny) for ix in range(Nx)]).reshape(Ny, Nx, 8)

    # Vectorized HLL flux computation for y-direction interfaces
    # Down interfaces: between [iy-1, ix] and [iy, ix] for all interior cells
    UL_y_D = U[0:Ny, ix_slice, :]      # shape (Ny, Nx, 8)
    UR_y_D = U[1:Ny+1, ix_slice, :]   # shape (Ny, Nx, 8)
    F_y_D = np.array([hll_flux_y(UL_y_D[iy, ix, :], UR_y_D[iy, ix, :]) 
                      for iy in range(Ny) for ix in range(Nx)]).reshape(Ny, Nx, 8)
    
    # Up interfaces: between [iy, ix] and [iy+1, ix] for all interior cells
    UL_y_U = U[1:Ny+1, ix_slice, :]   # shape (Ny, Nx, 8)
    UR_y_U = U[2:Ny+2, ix_slice, :]   # shape (Ny, Nx, 8)
    F_y_U = np.array([hll_flux_y(UL_y_U[iy, ix, :], UR_y_U[iy, ix, :]) 
                      for iy in range(Ny) for ix in range(Nx)]).reshape(Ny, Nx, 8)

    # Compute divergence of fluxes (vectorized)
    dUdt[iy_slice, ix_slice, :] = -(F_x_R - F_x_L) / dx - (F_y_U - F_y_D) / dy

    return dUdt


def rk2_step(U, dt, dx, dy, params):
    k1 = rhs(U, dx, dy, params)
    U_tmp = U + dt * k1
    k2 = rhs(U_tmp, dx, dy, params)
    return U + 0.5 * dt * (k1 + k2)


# ---------------------------------------------------------------------
# Initial Harris sheet + perturbation
# ---------------------------------------------------------------------
def init_harris(x, y, params, A1):
    B0 = params["B0"]
    Bg = params["Bg"]
    rho0 = params["rho0"]
    p_inf = params["p_inf"]
    Lx = params["Lx"]
    Ly = params["Ly"]
    a = params["a"]

    Nx = x.size
    Ny = y.size
    X, Y = np.meshgrid(x, y)  # (Ny, Nx)

    # Harris sheet
    Bx = B0 * np.tanh(Y / a)
    By = np.zeros_like(Bx)
    Bz = Bg * np.ones_like(Bx)

    sech2 = 1.0 / np.cosh(Y / a) ** 2
    p = p_inf + 0.5 * B0 ** 2 * sech2
    rho = rho0 * np.ones_like(Bx)
    vx = np.zeros_like(Bx)
    vy = np.zeros_like(Bx)
    vz = np.zeros_like(Bx)

    # Perturbation via flux function A_z
    kx_p = np.pi / (2.0 * Lx)
    ky_p = np.pi / (2.0 * Ly)
    A_z = A1 * np.cos(kx_p * X) * np.cos(ky_p * Y) * np.exp(-(Y / a) ** 2)

    dA_dx = -A1 * kx_p * np.sin(kx_p * X) * np.cos(ky_p * Y) * np.exp(-(Y / a) ** 2)
    dA_dy = A1 * np.cos(kx_p * X) * (
        -ky_p * np.sin(ky_p * Y) * np.exp(-(Y / a) ** 2)
        + np.cos(ky_p * Y) * np.exp(-(Y / a) ** 2) * (-2.0 * Y / a ** 2)
    )

    dBx = -dA_dy
    dBy = dA_dx
    Bx += dBx
    By += dBy

    # Build U with ghost cells
    U = np.zeros((Ny + 2, Nx + 2, 8))
    Ui = U[1:-1, 1:-1, :]

    Ui[..., 0] = rho
    Ui[..., 1] = rho * vx
    Ui[..., 2] = rho * vy
    Ui[..., 3] = rho * vz
    Ui[..., 4] = Bx
    Ui[..., 5] = By
    Ui[..., 6] = Bz

    v2 = vx * vx + vy * vy + vz * vz
    B2 = Bx * Bx + By * By + Bz * Bz
    Ui[..., 7] = p / (gamma - 1.0) + 0.5 * rho * v2 + 0.5 * B2

    apply_bc_reconnection(U, params)
    return U


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------
def total_energy(U, dx, dy):
    Uint = U[1:-1, 1:-1, :]
    rho, vx, vy, vz, Bx, By, Bz, p = cons_to_prim(Uint)
    v2 = vx * vx + vy * vy + vz * vz
    B2 = Bx * Bx + By * By + Bz * Bz
    e_int = p / (gamma - 1.0)
    e_kin = 0.5 * rho * v2
    e_mag = 0.5 * B2
    e_tot = e_int + e_kin + e_mag
    return np.sum(e_tot) * dx * dy


def compute_Jz(Bx, By, dx, dy):
    '''Compute Jz = ∂x By - ∂y Bx (vectorized).'''
    Ny, Nx = Bx.shape
    Jz = np.zeros_like(Bx)
    
    # Central differences in interior (vectorized)
    dBy_dx = (By[1:Ny-1, 2:Nx] - By[1:Ny-1, 0:Nx-2]) / (2.0 * dx)  # shape (Ny-2, Nx-2)
    dBx_dy = (Bx[2:Ny, 1:Nx-1] - Bx[0:Ny-2, 1:Nx-1]) / (2.0 * dy)  # shape (Ny-2, Nx-2)
    Jz[1:Ny-1, 1:Nx-1] = dBy_dx - dBx_dy

    # Copy neighbors to boundaries (vectorized)
    Jz[0, :] = Jz[1, :]
    Jz[-1, :] = Jz[-2, :]
    Jz[:, 0] = Jz[:, 1]
    Jz[:, -1] = Jz[:, -2]
    return Jz


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------
def main():
    # Domain and grid
    Lx = 10.0
    Ly = 5.0
    Nx = 512
    Ny = 256

    dx = 2.0 * Lx / Nx
    dy = 2.0 * Ly / Ny

    x = np.linspace(-Lx + 0.5 * dx, Lx - 0.5 * dx, Nx)
    y = np.linspace(-Ly + 0.5 * dy, Ly - 0.5 * dy, Ny)

    # Harris sheet parameters
    B0 = 1.0
    Bg = 0.0
    rho0 = 1.0
    p_inf = 0.05
    a = 0.3
    A1 = 0.05

    params = dict(B0=B0, Bg=Bg, rho0=rho0, p_inf=p_inf,
                  Lx=Lx, Ly=Ly, a=a)

    U = init_harris(x, y, params, A1)

    CFL = 0.2
    t = 0.0
    Tf = 40.0

    # Storage for animation (store more frequently for smooth animation)
    snapshot_interval = 1.0  # Store snapshot every 1.0 time units
    stored_Bx = []
    stored_Jz = []
    stored_t = []

    Uint = U[1:-1, 1:-1, :]
    rho_i, vx_i, vy_i, vz_i, Bx_i, By_i, Bz_i, p_i = cons_to_prim(Uint)
    stored_Bx.append(Bx_i.copy())
    stored_Jz.append(compute_Jz(Bx_i, By_i, dx, dy))
    stored_t.append(t)

    next_snapshot_time = snapshot_interval
    step = 0
    E0 = total_energy(U, dx, dy)
    print(f"Initial total energy: {E0:.6f}")
    print(f"Nx={Nx}, Ny={Ny}, dx={dx:.3e}, dy={dy:.3e}")

    # Time loop
    while t < Tf - 1e-12:
        Uint = U[1:-1, 1:-1, :]
        rho_i, vx_i, vy_i, vz_i, Bx_i, By_i, Bz_i, p_i = cons_to_prim(Uint)
        s_x, s_y = compute_wave_speeds(rho_i, vx_i, vy_i, Bx_i, By_i, Bz_i, p_i)
        s_max = max(np.max(s_x), np.max(s_y))
        dt = CFL * min(dx, dy) / s_max
        if t + dt > Tf:
            dt = Tf - t

        U = rk2_step(U, dt, dx, dy, params)
        t += dt
        step += 1

        if step % 20 == 0:
            Etot = total_energy(U, dx, dy)
            print(f"step {step:5d}, t={t:.3f}, E={Etot:.6f}, dE/E0={(Etot-E0)/E0:.3e}")

        # Store snapshots for animation
        if t >= next_snapshot_time - 1e-12:
            Uint = U[1:-1, 1:-1, :]
            rho_i, vx_i, vy_i, vz_i, Bx_i, By_i, Bz_i, p_i = cons_to_prim(Uint)
            stored_Bx.append(Bx_i.copy())
            stored_Jz.append(compute_Jz(Bx_i, By_i, dx, dy))
            stored_t.append(t)
            next_snapshot_time += snapshot_interval

    print(f"Finished at t={t:.3f}, steps={step}")
    Ef = total_energy(U, dx, dy)
    print(f"Final total energy: {Ef:.6f}, dE/E0={(Ef-E0)/E0:.3e}")

    # Plot Bx and Jz snapshots
    fig, axes = plt.subplots(2, len(stored_t),
                             figsize=(4 * len(stored_t), 6),
                             sharex=True, sharey=True)

    for idx, (Bx_snap, Jz_snap, ts) in enumerate(zip(stored_Bx, stored_Jz, stored_t)):
        ax1 = axes[0, idx] if len(stored_t) > 1 else axes[0]
        im1 = ax1.imshow(Bx_snap, origin="lower",
                         extent=[-Lx, Lx, -Ly, Ly], aspect="equal")
        ax1.set_title(f"B_x, t={ts:.2f}")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = axes[1, idx] if len(stored_t) > 1 else axes[1]
        im2 = ax2.imshow(Jz_snap, origin="lower",
                         extent=[-Lx, Lx, -Ly, Ly], aspect="equal")
        ax2.set_title(f"J_z, t={ts:.2f}")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    for ax in axes[1, :]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
