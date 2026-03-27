import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from bspf1d import bspf1d
from filter import apply_filter_dct
# from chebyshev_burgers1d import mesh_convergence_study_chebyshev  # optional, used in the convergence section

# ----------------------------
# Exact / manufactured solution
# ----------------------------
def smooth_step_solution(x, t, nu, alpha=0.4, beta=0.6, gamma=1.0*np.pi):
    """
    Analytical solution for a smooth step-like shock wave.
    Returns u(x,t) evaluated on (x,t). Handles scalar or array t.
    """
    if np.isscalar(t):
        t = np.array([t])
    eta = (alpha/nu) * (x - beta*t.reshape(-1, 1) - gamma)
    u = (alpha + beta + (beta - alpha) * np.exp(eta)) / (1 + np.exp(eta))
    return u.squeeze()

# ----------------------------
# IVP RHS with boundary control
# ----------------------------
def burgers_rhs_ivp(t, u, bspf_op, nu, u_bc_func):
    """
    RHS for solve_ivp (RK45) with exact Dirichlet BCs enforced.

    Steps:
      1) Overwrite boundary values of the working copy with exact BCs at time t.
      2) Compute spatial derivatives using BSPF on this corrected vector.
      3) Set du/dt = 0 at boundaries so they remain fixed during integration.
    """
    u_ext = u.copy()
    bc = u_bc_func(t)
    u_ext[0]  = bc[0]
    u_ext[-1] = bc[-1]

    u_ext = apply_filter_dct(u_ext)
    du_dx, d2u_dx2, _ = bspf_op.differentiate_1_2(u_ext)
    rhs = nu * d2u_dx2 - u_ext * du_dx

    rhs[0] = 0.0
    rhs[-1] = 0.0
    return rhs

# ----------------------------
# Solver using explicit RK4 time stepping
# ----------------------------
def solve_burgers_equation(nu=0.01, nx=101, nt=1001, Border=5, L=1.0, T=1.0):
    """
    Solve the 1D Burgers' equation u_t + u u_x = nu u_xx on [0,L]x[0,T]
    with spatial derivatives from BSPF and RK45 time stepping.
    Boundary values are pinned to the exact smooth-step solution.
    """
    # grids
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    dx = L / (nx - 1)

    dt = t[1] - t[0]

    # BSPF operator
    bspf_op = bspf1d.from_grid(
        degree=Border,
        order=Border,
        n_basis=4*Border,
        num_boundary_points=Border+5,
        x=x,
        use_clustering=True,
        clustering_factor=2.0
        )

    # exact solution on the output time grid (for ICs/boundaries/plots)
    u_exact = np.zeros((nt, nx))
    for i, ti in enumerate(t):
        u_exact[i, :] = smooth_step_solution(x, ti, nu)

    u0 = u_exact[0, :].copy()
    u_bc_func = lambda ti: smooth_step_solution(x, ti, nu)

    # integrate using explicit RK4 with fixed time step dt
    start_time = time.time()
    U = np.zeros((nt, nx), dtype=u0.dtype)
    U[0, :] = u0
    u = u0.copy()
    for n in tqdm(range(nt - 1), desc="RK4 time stepping", unit="step"):
        tn = t[n]
        k1 = burgers_rhs_ivp(tn, u, bspf_op, nu, u_bc_func)
        k2 = burgers_rhs_ivp(tn + 0.5 * dt, u + 0.5 * dt * k1, bspf_op, nu, u_bc_func)
        k3 = burgers_rhs_ivp(tn + 0.5 * dt, u + 0.5 * dt * k2, bspf_op, nu, u_bc_func)
        k4 = burgers_rhs_ivp(tn + dt,       u + dt * k3,       bspf_op, nu, u_bc_func)
        u = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        U[n + 1, :] = u
    time_integration_time = time.time() - start_time

    return x, t, U, u_exact, time_integration_time

# ----------------------------
# Plotting utilities
# ----------------------------
def plot_results(x, t, U, u_exact, nu, plot_times=None):
    """
    Plot numerical vs exact and pointwise error at a few times.
    """
    if plot_times is None:
        plot_times = np.linspace(0, t[-1], 3)

    plt.rcParams.update({
        'font.size': 20
    })

    t_idx = [np.abs(t - pt).argmin() for pt in plot_times]

    fig = plt.figure(figsize=(16, 6))

    # left: solution comparison
    plt.subplot(1, 2, 1)
    for i in t_idx:
        label = f'Sim. ({t[i]:.1f} s)'
        plt.plot(x, U[i, :], '-', label=label, markersize=4)
    # reset color cycle, then plot exact with markers
    plt.gca().set_prop_cycle(None)
    for i in t_idx:
        label = f'Exact ({t[i]:.1f} s)'
        plt.plot(x, u_exact[i, :], 'o', label=label, markersize=4)
    plt.xlabel('$x$')
    plt.ylabel('$u(x,t)$')
    plt.grid(True)
    plt.legend(loc='lower left')

    # right: error curves
    plt.subplot(1, 2, 2)
    for i in t_idx:
        label = f't = {t[i]:.1f} s'
        plt.semilogy(x, np.abs(U[i, :] - u_exact[i, :]), '-', label=label, markersize=4)
    plt.xlabel('$x$')
    plt.ylabel('|Error|')
    plt.grid(True)
    plt.legend(loc='best')

    plt.tight_layout()
    return fig

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # Parameters
    nu = 0.01     # viscosity
    Border = 8    # B-spline order
    nx = 1000     # spatial points
    dt = 5e-4
    L = 2.0 * np.pi
    T = 2.0
    nt = int(T / dt) + 1

    # Solve
    x, t, U, u_exact, time_integration_time = solve_burgers_equation(
        nu=nu,
        nx=nx,
        nt=nt,
        Border=Border,
        L=L,
        T=T,
    )

    plot_results(x, t, U, u_exact, nu)
    # plt.savefig(f'./figs/fig3_nu{nu}.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    # Save U and u_exact to npy
    np.save(f'./data/U_nu{nu}.npy', U)
    np.save(f'./data/u_exact_nu{nu}.npy', u_exact)