#!/usr/bin/env python3
"""
2D nonlinear ideal MHD with non-periodic (outflow) boundary conditions.

State per cell:
    U = [ rho,
          rho*v_x, rho*v_y, rho*v_z,
          B_x, B_y, B_z,
          E ]

Conservative form:
    ∂_t U + ∂_x F_x(U) + ∂_y F_y(U) = 0

Numerics:
    - Finite volume in 2D
    - Rusanov (local Lax–Friedrichs) flux in x, y
    - 2nd-order Runge–Kutta (Heun) in time
    - **Outflow (zero-gradient) boundary conditions** via ghost cells
      (no periodic np.roll)

Initial condition:
    Divergence-free Alfvén-like perturbation from vector potential:

      A_z = A0 cos(kx x) cos(ky y)
      => B_x =  A0 ky sin(ky y) cos(kx x)
         B_y = -A0 kx sin(kx x) cos(ky y)
         B_z = B0 (guide field)

      v_perp ≈ - B_perp / sqrt(rho0)
      rho = rho0, p = p0

You should see a smooth Alfvén-like pattern that propagates /
distorts and interacts with outflow boundaries, with no periodic wrap.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

gamma = 5.0 / 3.0
p_floor = 1e-8


# --------------------------------------------------------------------
# Primitive / flux / wave-speeds
# --------------------------------------------------------------------
def cons_to_prim(U):
    """
    U[...,0] = rho
    U[...,1] = rho*vx
    U[...,2] = rho*vy
    U[...,3] = rho*vz
    U[...,4] = Bx
    U[...,5] = By
    U[...,6] = Bz
    U[...,7] = E
    """
    rho = U[..., 0]
    vx = U[..., 1] / rho
    vy = U[..., 2] / rho
    vz = U[..., 3] / rho
    Bx = U[..., 4]
    By = U[..., 5]
    Bz = U[..., 6]
    E = U[..., 7]

    v2 = vx**2 + vy**2 + vz**2
    B2 = Bx**2 + By**2 + Bz**2

    p = (gamma - 1.0) * (E - 0.5 * rho * v2 - 0.5 * B2)
    p = np.maximum(p, p_floor)

    return rho, vx, vy, vz, Bx, By, Bz, p


def compute_fluxes(U):
    """
    Ideal MHD fluxes F_x, F_y on the whole grid (including ghosts).
    """
    rho, vx, vy, vz, Bx, By, Bz, p = cons_to_prim(U)
    B2 = Bx**2 + By**2 + Bz**2
    v2 = vx**2 + vy**2 + vz**2

    ptot = p + 0.5 * B2
    Bdotv = Bx * vx + By * vy + Bz * vz
    E = U[..., 7]

    Fx = np.zeros_like(U)
    Fy = np.zeros_like(U)

    # Flux in x-direction
    Fx[..., 0] = rho * vx
    Fx[..., 1] = rho * vx * vx + ptot - Bx * Bx
    Fx[..., 2] = rho * vx * vy - Bx * By
    Fx[..., 3] = rho * vx * vz - Bx * Bz
    Fx[..., 4] = 0.0
    Fx[..., 5] = vx * By - Bx * vy
    Fx[..., 6] = vx * Bz - Bx * vz
    Fx[..., 7] = (E + ptot) * vx - Bx * Bdotv

    # Flux in y-direction
    Fy[..., 0] = rho * vy
    Fy[..., 1] = rho * vy * vx - By * Bx
    Fy[..., 2] = rho * vy * vy + ptot - By * By
    Fy[..., 3] = rho * vy * vz - By * Bz
    Fy[..., 4] = vy * Bx - By * vx
    Fy[..., 5] = 0.0
    Fy[..., 6] = vy * Bz - By * vz
    Fy[..., 7] = (E + ptot) * vy - By * Bdotv

    return Fx, Fy, (rho, vx, vy, vz, Bx, By, Bz, p)


def compute_wave_speeds(rho, vx, vy, Bx, By, Bz, p):
    """
    Local fast magnetosonic speeds in x and y for Rusanov flux.
    """
    B2 = Bx**2 + By**2 + Bz**2
    a2 = B2 / rho                 # total Alfvén speed^2
    cs2 = gamma * p / rho         # sound speed^2
    cs2 = np.maximum(cs2, 0.0)

    # x-direction normal field
    a_n2_x = Bx**2 / rho
    term_x = (a2 + cs2)**2 - 4.0 * a_n2_x * cs2
    term_x = np.maximum(term_x, 0.0)
    cf2_x = 0.5 * (a2 + cs2 + np.sqrt(term_x))
    cf_x = np.sqrt(cf2_x)
    s_x = np.abs(vx) + cf_x

    # y-direction normal field
    a_n2_y = By**2 / rho
    term_y = (a2 + cs2)**2 - 4.0 * a_n2_y * cs2
    term_y = np.maximum(term_y, 0.0)
    cf2_y = 0.5 * (a2 + cs2 + np.sqrt(term_y))
    cf_y = np.sqrt(cf2_y)
    s_y = np.abs(vy) + cf_y

    return s_x, s_y


# --------------------------------------------------------------------
# Boundary conditions, RHS, RK2
# --------------------------------------------------------------------
def apply_bc_outflow(U):
    """
    Zero-gradient (outflow) boundary conditions using 1-layer ghost cells.

    U has shape (Ny+2, Nx+2, 8), interior at [1:-1, 1:-1].
    """
    # Copy nearest interior cell into ghost layer
    # Left/right
    U[:, 0, :] = U[:, 1, :]
    U[:, -1, :] = U[:, -2, :]

    # Bottom/top
    U[0, :, :] = U[1, :, :]
    U[-1, :, :] = U[-2, :, :]


def rhs(U, dx, dy):
    """
    Compute dU/dt on the whole grid (ghosts included, but only
    interior gets nonzero updates), using Rusanov flux and
    outflow boundaries.
    Vectorized version (no loops).
    """
    # Enforce BCs before computing fluxes
    apply_bc_outflow(U)

    Fx, Fy, prim = compute_fluxes(U)
    rho, vx, vy, vz, Bx, By, Bz, p = prim
    s_x, s_y = compute_wave_speeds(rho, vx, vy, Bx, By, Bz, p)

    Ny_tot, Nx_tot, _ = U.shape
    Ny = Ny_tot - 2   # interior count
    Nx = Nx_tot - 2

    dUdt = np.zeros_like(U)

    # Interior slice indices
    iy_slice = slice(1, Ny + 1)  # interior y indices: [1, 2, ..., Ny]
    ix_slice = slice(1, Nx + 1)  # interior x indices: [1, 2, ..., Nx]

    # --- x-direction interfaces (vectorized) ---
    # For interior cells at (iy, ix), left interface is between (iy, ix-1) and (iy, ix)
    # Right interface is between (iy, ix) and (iy, ix+1)
    
    # Left interfaces: between cells [iy, 0:Nx] and [iy, 1:Nx+1] for all iy in interior
    UL_x_L = U[iy_slice, 0:Nx, :]      # shape (Ny, Nx, 8)
    UR_x_L = U[iy_slice, 1:Nx+1, :]    # shape (Ny, Nx, 8)
    FL_x_L = Fx[iy_slice, 0:Nx, :]
    FR_x_L = Fx[iy_slice, 1:Nx+1, :]
    sl_x_L = s_x[iy_slice, 0:Nx]       # shape (Ny, Nx)
    sr_x_L = s_x[iy_slice, 1:Nx+1]
    smax_x_L = np.maximum(sl_x_L, sr_x_L)[..., None]  # shape (Ny, Nx, 1)
    F_hat_x_L = 0.5 * (FL_x_L + FR_x_L) - 0.5 * smax_x_L * (UR_x_L - UL_x_L)

    # Right interfaces: between cells [iy, 1:Nx+1] and [iy, 2:Nx+2] for all iy in interior
    UL_x_R = U[iy_slice, 1:Nx+1, :]    # shape (Ny, Nx, 8)
    UR_x_R = U[iy_slice, 2:Nx+2, :]    # shape (Ny, Nx, 8)
    FL_x_R = Fx[iy_slice, 1:Nx+1, :]
    FR_x_R = Fx[iy_slice, 2:Nx+2, :]
    sl_x_R = s_x[iy_slice, 1:Nx+1]     # shape (Ny, Nx)
    sr_x_R = s_x[iy_slice, 2:Nx+2]
    smax_x_R = np.maximum(sl_x_R, sr_x_R)[..., None]  # shape (Ny, Nx, 1)
    F_hat_x_R = 0.5 * (FL_x_R + FR_x_R) - 0.5 * smax_x_R * (UR_x_R - UL_x_R)

    # --- y-direction interfaces (vectorized) ---
    # For interior cells at (iy, ix), down interface is between (iy-1, ix) and (iy, ix)
    # Up interface is between (iy, ix) and (iy+1, ix)
    
    # Down interfaces: between cells [0:Ny, ix] and [1:Ny+1, ix] for all ix in interior
    UL_y_D = U[0:Ny, ix_slice, :]      # shape (Ny, Nx, 8)
    UR_y_D = U[1:Ny+1, ix_slice, :]    # shape (Ny, Nx, 8)
    FL_y_D = Fy[0:Ny, ix_slice, :]
    FR_y_D = Fy[1:Ny+1, ix_slice, :]
    sl_y_D = s_y[0:Ny, ix_slice]       # shape (Ny, Nx)
    sr_y_D = s_y[1:Ny+1, ix_slice]
    smax_y_D = np.maximum(sl_y_D, sr_y_D)[..., None]  # shape (Ny, Nx, 1)
    F_hat_y_D = 0.5 * (FL_y_D + FR_y_D) - 0.5 * smax_y_D * (UR_y_D - UL_y_D)

    # Up interfaces: between cells [1:Ny+1, ix] and [2:Ny+2, ix] for all ix in interior
    UL_y_U = U[1:Ny+1, ix_slice, :]    # shape (Ny, Nx, 8)
    UR_y_U = U[2:Ny+2, ix_slice, :]    # shape (Ny, Nx, 8)
    FL_y_U = Fy[1:Ny+1, ix_slice, :]
    FR_y_U = Fy[2:Ny+2, ix_slice, :]
    sl_y_U = s_y[1:Ny+1, ix_slice]     # shape (Ny, Nx)
    sr_y_U = s_y[2:Ny+2, ix_slice]
    smax_y_U = np.maximum(sl_y_U, sr_y_U)[..., None]  # shape (Ny, Nx, 1)
    F_hat_y_U = 0.5 * (FL_y_U + FR_y_U) - 0.5 * smax_y_U * (UR_y_U - UL_y_U)

    # Compute divergence of fluxes (vectorized)
    # For each interior cell (iy, ix):
    #   dF_x/dx = (F_right(ix+1/2) - F_left(ix-1/2)) / dx
    #   dF_y/dy = (F_up(iy+1/2) - F_down(iy-1/2)) / dy
    dUdt[iy_slice, ix_slice, :] = - (F_hat_x_R - F_hat_x_L) / dx - (F_hat_y_U - F_hat_y_D) / dy

    return dUdt


def rk2_step(U, dt, dx, dy):
    """Heun's RK2 step."""
    k1 = rhs(U, dx, dy)
    U_tmp = U + dt * k1
    k2 = rhs(U_tmp, dx, dy)
    U_new = U + 0.5 * dt * (k1 + k2)
    return U_new


# --------------------------------------------------------------------
# Initial condition & diagnostics (non-periodic box)
# --------------------------------------------------------------------
def init_alfven_box(Nx, Ny, Lx, Ly, rho0, p0, B0, deltaB, mx, my):
    """
    Divergence-free Alfvén-like perturbation in a finite 2D box.

    Cells are centered at x_i, y_j:

        x_i = (i + 0.5) * dx,  i=0..Nx-1
        y_j = (j + 0.5) * dy,  j=0..Ny-1

    A_z = A0 cos(kx x) cos(ky y)
      => B_x =  A0 ky sin(ky y) cos(kx x)
         B_y = -A0 kx sin(kx x) cos(ky y)
         B_z = B0

    v_perp ≈ - B_perp / sqrt(rho0)
    """
    dx = Lx / Nx
    dy = Ly / Ny

    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing="xy")

    kx = 2.0 * np.pi * mx / Lx
    ky = 2.0 * np.pi * my / Ly

    A0 = deltaB
    Bx = A0 * ky * np.sin(ky * Y) * np.cos(kx * X)
    By = -A0 * kx * np.sin(kx * X) * np.cos(ky * Y)
    Bz = B0 * np.ones_like(X)

    rho = rho0 * np.ones_like(X)

    vx = -Bx / np.sqrt(rho0)
    vy = -By / np.sqrt(rho0)
    vz = np.zeros_like(X)

    v2 = vx**2 + vy**2 + vz**2
    B2 = Bx**2 + By**2 + Bz**2

    E = p0 / (gamma - 1.0) + 0.5 * rho * v2 + 0.5 * B2

    # Embed interior in ghost-cell array
    U = np.zeros((Ny + 2, Nx + 2, 8))
    U[1:-1, 1:-1, 0] = rho
    U[1:-1, 1:-1, 1] = rho * vx
    U[1:-1, 1:-1, 2] = rho * vy
    U[1:-1, 1:-1, 3] = rho * vz
    U[1:-1, 1:-1, 4] = Bx
    U[1:-1, 1:-1, 5] = By
    U[1:-1, 1:-1, 6] = Bz
    U[1:-1, 1:-1, 7] = E

    return x, y, U

def init_orszag_tang_2d(Nx, Ny, Lx, Ly, rho0, p0, B0, v0):
    """
    Orszag–Tang-like 2D MHD initial condition on [0,Lx]x[0,Ly].

    - Divergence-free B from vector potential A_z = B0 * cos(x) * cos(y):
        Bx =  ∂_y A_z = -B0 * cos(x) * sin(y)
        By = -∂_x A_z =  B0 * sin(x) * cos(y)
        Bz = 0

    - Velocity is a cellular vortex:
        v_x = -v0 * sin(y)
        v_y =  v0 * sin(x)
        v_z = 0

    - Uniform density and pressure: rho = rho0, p = p0.
    """
    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Magnetic field from vector potential A_z = B0 cos(x) cos(y)
    Bx = -B0 * np.cos(X) * np.sin(Y)
    By =  B0 * np.sin(X) * np.cos(Y)
    Bz = np.zeros_like(X)

    rho = rho0 * np.ones_like(X)

    # Velocity field: vortex pattern
    vx = -v0 * np.sin(Y)
    vy =  v0 * np.sin(X)
    vz = np.zeros_like(X)

    v2 = vx**2 + vy**2 + vz**2
    B2 = Bx**2 + By**2 + Bz**2

    E = p0 / (gamma - 1.0) + 0.5 * rho * v2 + 0.5 * B2

    U = np.zeros((Ny, Nx, 8))
    U[..., 0] = rho
    U[..., 1] = rho * vx
    U[..., 2] = rho * vy
    U[..., 3] = rho * vz
    U[..., 4] = Bx
    U[..., 5] = By
    U[..., 6] = Bz
    U[..., 7] = E

    return x, y, U

def total_energy(U, dx, dy):
    """Total energy ∫ e dA over interior cells."""
    Uint = U[1:-1, 1:-1, :]
    rho, vx, vy, vz, Bx, By, Bz, p = cons_to_prim(Uint)
    v2 = vx**2 + vy**2 + vz**2
    B2 = Bx**2 + By**2 + Bz**2
    e_int = p / (gamma - 1.0)
    e_kin = 0.5 * rho * v2
    e_mag = 0.5 * B2
    e_tot = e_int + e_kin + e_mag
    return np.sum(e_tot) * dx * dy


# --------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------
def main():
    # Box and grid
    Lx, Ly = 2.0 * np.pi, 2.0 * np.pi
    Nx, Ny = 256, 256     # higher resolution shows more structure

    rho0 = 1.0
    p0   = 0.1
    B0   = 1.0
    v0   = 1.0

    dx = Lx / Nx
    dy = Ly / Ny

    x, y, U = init_orszag_tang_2d(Nx, Ny, Lx, Ly, rho0, p0, B0, v0)

    CFL = 0.3
    t   = 0.0
    Tf  = 10.0

    # Storage for animation (store more frequently for smooth animation)
    snapshot_interval = 0.1  # Store snapshot every 0.1 time units
    stored_Bx = []
    stored_vx = []
    stored_t = []

    # Initial interior fields
    rho_i, vx_i, vy_i, vz_i, Bx_i, By_i, Bz_i, p_i = cons_to_prim(U[1:-1, 1:-1, :])
    stored_Bx.append(Bx_i.copy())
    stored_vx.append(vx_i.copy())
    stored_t.append(t)

    next_snapshot_time = snapshot_interval
    step = 0

    E0 = total_energy(U, dx, dy)
    print(f"Initial total energy: {E0:.6f}")

    # Time loop
    while t < Tf - 1e-12:
        # CFL from interior
        rho_i, vx_i, vy_i, vz_i, Bx_i, By_i, Bz_i, p_i = cons_to_prim(U[1:-1, 1:-1, :])
        s_x_i, s_y_i = compute_wave_speeds(rho_i, vx_i, vy_i, Bx_i, By_i, Bz_i, p_i)
        s_max = max(np.max(s_x_i), np.max(s_y_i))
        dt = CFL * min(dx, dy) / s_max
        if t + dt > Tf:
            dt = Tf - t

        U = rk2_step(U, dt, dx, dy)
        t += dt
        step += 1

        if step % 20 == 0:
            Etot = total_energy(U, dx, dy)
            print(f"step {step:5d}, t = {t:.3f}, "
                  f"E = {Etot:.6f}, ΔE/E0 ≈ {(Etot - E0)/E0:.3e}")

        # Store snapshots for animation
        if t >= next_snapshot_time - 1e-12:
            rho_i, vx_i, vy_i, vz_i, Bx_i, By_i, Bz_i, p_i = cons_to_prim(U[1:-1, 1:-1, :])
            stored_Bx.append(Bx_i.copy())
            stored_vx.append(vx_i.copy())
            stored_t.append(t)
            next_snapshot_time += snapshot_interval

    print(f"Finished at t = {t:.3f}, steps = {step}")
    Ef = total_energy(U, dx, dy)
    print(f"Final total energy: {Ef:.6f}, ΔE/E0 ≈ {(Ef - E0)/E0:.3e}")
    print(f"Stored {len(stored_t)} snapshots for animation")

    # ----------------------------------------------------------------
    # Animation: Bx and vx evolution
    # ----------------------------------------------------------------
    # Find global min/max for consistent color scales
    Bx_min, Bx_max = np.min([np.min(Bx) for Bx in stored_Bx]), np.max([np.max(Bx) for Bx in stored_Bx])
    vx_min, vx_max = np.min([np.min(vx) for vx in stored_vx]), np.max([np.max(vx) for vx in stored_vx])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Initialize images
    im1 = axes[0].imshow(stored_Bx[0], origin="lower", 
                         extent=[0.0, Lx, 0.0, Ly],
                         aspect="equal", vmin=Bx_min, vmax=Bx_max,
                         cmap='RdBu')
    axes[0].set_title(f"B_x, t={stored_t[0]:.2f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(stored_vx[0], origin="lower",
                         extent=[0.0, Lx, 0.0, Ly],
                         aspect="equal", vmin=vx_min, vmax=vx_max,
                         cmap='RdBu')
    axes[1].set_title(f"v_x, t={stored_t[0]:.2f}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    def animate(frame):
        """Update animation frame."""
        im1.set_array(stored_Bx[frame])
        axes[0].set_title(f"B_x, t={stored_t[frame]:.2f}")
        
        im2.set_array(stored_vx[frame])
        axes[1].set_title(f"v_x, t={stored_t[frame]:.2f}")
        
        return [im1, im2]
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(stored_t), 
                        interval=100, blit=True, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    # Optionally save animation (uncomment to save)
    # print("Saving animation...")
    # anim.save('mhd_orszag_tang_2d.gif', writer='pillow', fps=10)
    # print("Animation saved to mhd_orszag_tang_2d.gif")


if __name__ == "__main__":
    main()
