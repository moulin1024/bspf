import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
from bspf import bspf1d

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
      2) Compute spatial derivatives using bspf on this corrected vector.
      3) Set du/dt = 0 at boundaries so they remain fixed during integration.
    """
    u_ext = u.copy()
    bc = u_bc_func(t)
    u_ext[0]  = bc[0]
    u_ext[-1] = bc[-1]
    du_dx, d2u_dx2, _ = bspf_op.differentiate_1_2(u_ext)
    rhs = nu * d2u_dx2 - u_ext * du_dx

    rhs[0] = 0.0
    rhs[-1] = 0.0
    return rhs

# ----------------------------
# Solver using solve_ivp (RK45)
# ----------------------------
def solve_burgers_equation(nu=0.01, nx=101, nt=1001, degree=5, L=1.0, T=1.0,method="RK45",
                           rtol=1e-8, atol=1e-8, max_step=None):
    """
    Solve the 1D Burgers' equation u_t + u u_x = nu u_xx on [0,L]x[0,T]
    with spatial derivatives from bspf and RK45 time stepping.
    Boundary values are pinned to the exact smooth-step solution.
    """
    # grids
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    dx = L / (nx - 1)

    dt = t[1] - t[0]

    # bspf operator
    bspf_op = bspf1d.from_grid(
        degree=degree,
        order=degree,
        n_basis=4*degree,
        num_boundary_points=degree,
        x=x
        )

    # exact solution on the output time grid (for ICs/boundaries/plots)
    u_exact = np.zeros((nt, nx))
    for i, ti in enumerate(t):
        u_exact[i, :] = smooth_step_solution(x, ti, nu)

    u0 = u_exact[0, :].copy()
    u_bc_func = lambda ti: smooth_step_solution(x, ti, nu)

    # integrate
    start_time = time.time()
    sol = solve_ivp(
        fun=lambda ti, ui: burgers_rhs_ivp(ti, ui, bspf_op, nu, u_bc_func),
        t_span=(0.0, T),
        y0=u0,
        method=method,
        rtol=1e-11,
        atol=1e-11,
        t_eval=t  # This ensures evaluation at your predefined time points
    )
    time_integration_time = time.time() - start_time

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    U = sol.y.T

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
        'font.size': 16
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
        plt.plot(x, np.abs(U[i, :] - u_exact[i, :]), '-', label=label, markersize=4)
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
    degree = 7    # B-spline order
    nx = 500     # spatial points
    nt = 500     # output time samples
    L = 2.0 * np.pi
    T = 2.0

    # Solve
    x, t, U, u_exact, time_integration_time = solve_burgers_equation(
        nu=nu,
        nx=nx,
        nt=nt,
        degree=degree,
        L=L,
        T=T,
        method="RK45"
    )

    plot_results(x, t, U, u_exact, nu, plot_times=[0.0, 0.5*T, T])
    plt.show()