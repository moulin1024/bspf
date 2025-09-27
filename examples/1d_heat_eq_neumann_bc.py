from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

from bspf import bspf1d

Array = npt.NDArray[np.float64]

def heat_neumann_cosx(x, t, kappa=1.0):
    """
    Solution of u_t = kappa*u_xx on [0, 2π] with Neumann BCs
    and initial condition u(x,0) = cos(x).
    """
    return np.cos(x) * np.exp(-kappa * t)

def heat_rhs(t: float, u: Array) -> Array:
    """Heat equation RHS with homogeneous Neumann boundary conditions."""
    _, d2u_dx2, _ = bf.differentiate_1_2(u, neumann_bc=(0.0, 0.0))
    return nu * d2u_dx2

# ---------- parameters ----------
nu = 1e-1            # diffusivity
L = 2 * np.pi        # domain length
nx = 101             # grid points (including endpoints)
T = 5                # final time
nt = 101             # number of saved frames

# ---------- grid, operator, IC ----------
x = np.linspace(0.0, L, nx)
bf = bspf1d.from_grid(degree=9, n_basis=20, x=x)

# Initial condition compatible with homogeneous Neumann BC (u_x=0 at ends)
u0 = np.cos(x)

# ---------- output times & storage ----------
t_out = np.linspace(0.0, T, nt)
U = np.empty((nt, nx), dtype=np.float64)
U[0] = u0.copy()

# ---------- integrate in time with SciPy ----------
solution = solve_ivp(
    heat_rhs,
    (t_out[0], t_out[-1]),
    u0,
    t_eval=t_out,
    method="RK45",
    rtol=1e-10,
    atol=1e-10,
)

U[:] = solution.y.T
u_exact = heat_neumann_cosx(x[None, :], t_out[:, None], kappa=nu)

# ---------- quick BC check ----------
du_dx_end, _, _ = bf.differentiate_1_2(U[-1], neumann_bc=(0.0, 0.0))
print(f"Neumann check: u_x(0)={du_dx_end[0]: .3e}, u_x(L)={du_dx_end[-1]: .3e}")

# ---------- (optional) visualize ----------
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, U[0], label="t=0 s")
    plt.plot(x, U[-1], ".", label=f"t={T} s")
    plt.plot(x, u_exact[-1],'k--', label="exact t=5 s")
    plt.legend()
    plt.title("1D Heat Equation with Neumann BC")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.subplot(1, 2, 2)
    plt.plot(x, np.abs(U[-1]-u_exact[-1]), label="error at t=5s")
    plt.title("|Error|")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.tight_layout()
    plt.show()
except Exception as exc:  # pragma: no cover - visualization is optional
    print("Plot skipped:", exc)
