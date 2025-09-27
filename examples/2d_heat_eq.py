# test_bspf2d_heat_neumann.py
from __future__ import annotations
import time
from typing import Tuple, Optional

import numpy as np

from bspf import bspf2d  # <-- the class you just built

try:
    from tqdm import tqdm
    def _iter(it, total): return tqdm(it, total=total)
except Exception:
    def _iter(it, total): return it


def exact_one_mode_neumann(x: np.ndarray, y: np.ndarray, t: float, nu: float,
                           Lx: float, Ly: float, A: float, mean: float = 1.0) -> np.ndarray:
    """
    Exact solution for IC: u(x,y,0) = mean + A cos(pi x/Lx) cos(pi y/Ly) with zero-flux BCs.
    u(x,y,t) = mean + A * exp(-nu*((pi/Lx)^2 + (pi/Ly)^2)*t) * cos(pi x/Lx) cos(pi y/Ly)
    """
    kx = np.pi / Lx
    ky = np.pi / Ly
    decay = np.exp(-nu * ((kx**2) + (ky**2)) * t)
    X, Y = np.meshgrid(x, y)  # y rows, x cols (ny, nx)
    return mean + A * decay * np.cos(kx * X) * np.cos(ky * Y)


def rk4_heat_step(U: np.ndarray,
                  plan_x2, plan_y2,
                  dt: float, nu: float,
                  flux_x: Tuple[float, float] = (0.0, 0.0),
                  flux_y: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """One RK4 step: U_t = nu*(U_xx + U_yy), using precomputed ∂²/∂x² and ∂²/∂y² plans with Neumann BC."""
    def rhs(V: np.ndarray) -> np.ndarray:
        Vxx = plan_x2.apply(V, flux=flux_x)
        Vyy = plan_y2.apply(V, flux=flux_y)
        return nu * (Vxx + Vyy)

    k1 = rhs(U)
    k2 = rhs(U + 0.5 * dt * k1)
    k3 = rhs(U + 0.5 * dt * k2)
    k4 = rhs(U + dt * k3)
    return U + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def main():
    # ------------------ grid & params ------------------
    Lx, Ly = 1.0, 1.0
    nx, ny = 64, 64
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    nu = 1e-2
    Tfinal = 0.2
    
    dt = 0.1 * (min(dx, dy) ** 2) / nu
    nsteps = int(np.ceil(Tfinal / dt))
    dt = Tfinal / nsteps  # end exactly at Tfinal

    # ------------------ initial condition (one mode) ------------------
    A = 0.5
    mean = 1.0
    X, Y = np.meshgrid(x, y)  # (ny, nx)
    U0 = mean + A * np.cos(np.pi * X / Lx) * np.cos(np.pi * Y / Ly)

    # ------------------ operator & plans ------------------
    # Use degree high enough so model.end.order >= 2 (Neumann requires order>=2).
    # bspf1d defaults order = degree-1, so degree >= 3 is fine; we pick 8 for comfort.
    op = bspf2d.from_grids(x=x, y=y, degree_x=8, degree_y=8, correction="spectral")

    # Precompute plans for second derivatives with Neumann flux at the boundary
    plan_x2 = op.make_plan_dx(order=2, lam=0.0, neumann=True)
    plan_y2 = op.make_plan_dy(order=2, lam=0.0, neumann=True)

    # Also make first-derivative plans for periodic flux checks (they enforce 0 flux too)
    plan_x1 = op.make_plan_dx(order=1, lam=0.0, neumann=True)
    plan_y1 = op.make_plan_dy(order=1, lam=0.0, neumann=True)

    # ------------------ time integration ------------------
    U = U0.copy()
    flux_tol = 1e-8
    check_stride = max(1, nsteps // 20)

    # Mass (heat) should be conserved for zero-flux BCs
    cell_area = dx * dy
    mass0 = float(np.sum(U0) * cell_area)

    t0 = time.time()
    for step in _iter(range(1, nsteps + 1), total=nsteps):
        U = rk4_heat_step(U, plan_x2, plan_y2, dt, nu, flux_x=(0.0, 0.0), flux_y=(0.0, 0.0))

        # check boundary flux occasionally
        if (step % check_stride == 0) or (step == nsteps):
            Ux = plan_x1.apply(U, flux=(0.0, 0.0))
            Uy = plan_y1.apply(U, flux=(0.0, 0.0))
            # endpoints: x-direction -> first/last columns; y-direction -> first/last rows
            max_flux = max(
                float(np.max(np.abs(Ux[:, 0]))),
                float(np.max(np.abs(Ux[:, -1]))),
                float(np.max(np.abs(Uy[0, :]))),
                float(np.max(np.abs(Uy[-1, :]))),
            )
            if max_flux > flux_tol:
                print(f"[warn] flux check @ step {step}: |grad·n|_max = {max_flux:.2e} > {flux_tol:.2e}")

    wall = time.time() - t0

    # ------------------ diagnostics ------------------
    mass_final = float(np.sum(U) * cell_area)
    mass_err = abs(mass_final - mass0)

    U_exact = exact_one_mode_neumann(x, y, Tfinal, nu, Lx, Ly, A, mean)
    L2 = float(np.sqrt(np.mean((U - U_exact) ** 2)))
    Linf = float(np.max(np.abs(U - U_exact)))

    print("\n--- run summary ---")
    print(f"grid           : ny={ny}, nx={nx}")
    print(f"dt, steps      : {dt:.3e}, {nsteps}")
    print(f"runtime        : {wall:.2f} s")
    print(f"mass initial   : {mass0:.12e}")
    print(f"mass final     : {mass_final:.12e}")
    print(f"|Δmass|        : {mass_err:.3e}")
    print(f"L2 error       : {L2:.3e}")
    print(f"L∞ error       : {Linf:.3e}")

    # quick plot (optional)
    try:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        im0 = axs[0].imshow(U0, origin='lower', extent=[0, Lx, 0, Ly]); axs[0].set_title('U0')
        im1 = axs[1].imshow(U, origin='lower', extent=[0, Lx, 0, Ly]); axs[1].set_title('U (num)')
        im2 = axs[2].imshow(U_exact, origin='lower', extent=[0, Lx, 0, Ly]); axs[2].set_title('U (exact)')
        for ax in axs:
            ax.set_xlabel('x'); ax.set_ylabel('y')
        for im in (im0, im1, im2):
            fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.85)
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
