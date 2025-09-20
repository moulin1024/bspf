import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
from bfpsm1d import bfpsm1d
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
def burgers_rhs_ivp(t, u, bfpsm_op, nu, u_bc_func):
    """
    RHS for solve_ivp (RK45) with exact Dirichlet BCs enforced.

    Steps:
      1) Overwrite boundary values of the working copy with exact BCs at time t.
      2) Compute spatial derivatives using BFPSM on this corrected vector.
      3) Set du/dt = 0 at boundaries so they remain fixed during integration.
    """
    u_ext = u.copy()
    bc = u_bc_func(t)
    u_ext[0]  = bc[0]
    u_ext[-1] = bc[-1]
    from filter import apply_filter_dct
    u_ext = apply_filter_dct(u_ext)
    du_dx, d2u_dx2, _ = bfpsm_op.differentiate_1_2(u_ext)
    rhs = nu * d2u_dx2 - u_ext * du_dx

    rhs[0] = 0.0
    rhs[-1] = 0.0
    return rhs

# ----------------------------
# Solver using solve_ivp (RK45)
# ----------------------------
def solve_burgers_equation(nu=0.01, nx=101, nt=1001, Border=5, L=1.0, T=1.0,method="RK45",
                           rtol=1e-8, atol=1e-8, max_step=None):
    """
    Solve the 1D Burgers' equation u_t + u u_x = nu u_xx on [0,L]x[0,T]
    with spatial derivatives from BFPSM and RK45 time stepping.
    Boundary values are pinned to the exact smooth-step solution.
    """
    # grids
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    dx = L / (nx - 1)

    dt = t[1] - t[0]

    # BFPSM operator
    bfpsm_op = bfpsm1d.from_grid(
        degree=Border,
        order=Border,
        n_basis=4*Border,
        num_boundary_points=Border,
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
        fun=lambda ti, ui: burgers_rhs_ivp(ti, ui, bfpsm_op, nu, u_bc_func),
        t_span=(0.0, T),
        y0=u0,
        method=method,
        rtol=rtol,
        atol=atol,
        t_eval=t  # This ensures evaluation at your predefined time points
    )
    time_integration_time = time.time() - start_time

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    U = sol.y.T

    # # Re-pin boundaries exactly at output times (nice for comparisons)
    # U[:, 0] = u_exact[:, 0]
    # U[:, -1] = u_exact[:, -1]

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
# Convergence study (optional)
# ----------------------------
def compute_convergence_study(nu=0.01, mesh_sizes=None, Border=8, L=2.0*np.pi, T=2.0,method="BDF", rtol=1e-12, atol=1e-12):
    """
    Run multiple meshes and return L2/L∞ errors at final time and timing.
    """
    if mesh_sizes is None:
        mesh_sizes = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

    l2_errors = np.zeros(len(mesh_sizes))
    linf_errors = np.zeros(len(mesh_sizes))
    time_measure = np.zeros(len(mesh_sizes))

    for i, nx in enumerate(mesh_sizes):
        nt = int(2.0 * nx)  # reasonable number of output times for RK45

        x, t, U, u_exact, time_integration_time = solve_burgers_equation(
            nu=nu,
            nx=nx + 1,
            nt=nt,
            Border=Border,
            L=L,
            T=T,
            method=method,
            rtol=rtol,
            atol=atol
        )

        # errors at final time
        error = np.abs(U[-1, :] - u_exact[-1, :])
        l2_errors[i] = np.sqrt(np.mean(error**2))
        linf_errors[i] = np.max(error)
        time_measure[i] = time_integration_time
        print(f"N={nx:4d} -> L2 error = {l2_errors[i]:.3e}, L∞ error = {linf_errors[i]:.3e}, time = {time_measure[i]:.2f} s, t_points = {t.shape[0]}")

    return mesh_sizes, l2_errors, linf_errors, time_measure

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    nu = 0.01
    Border = 8
    L = 2.0 * np.pi
    T = 2.0
    
    # -------- Optional: Convergence study ----------
    mesh_sizes = np.arange(100, 1001, 100)#np.array([200, 400, 800, 1600])
    print("Running BFPSM+RK45...")
    mesh_sizes, l2_errors_RK45, linf_errors_RK45, time_measure_RK45 = compute_convergence_study(
        nu=nu, mesh_sizes=mesh_sizes, Border=Border, L=L, T=2.0,method="RK45", rtol=2e-11, atol=2e-11
    )

    print("Running BFPSM+BDF...")
    mesh_sizes, l2_errors_BDF, linf_errors_BDF, time_measure_BDF = compute_convergence_study(
        nu=nu, mesh_sizes=mesh_sizes, Border=Border, L=L, T=2.0,method="BDF", rtol=1e-12, atol=1e-12
    )

    print("Running Chebyshev+BDF...")
    # If you have the Chebyshev reference:
    from chebyshev_burgers1d import mesh_convergence_study_chebyshev
    l2_errors_chebyshev, timings_chebyshev = mesh_convergence_study_chebyshev(
        mesh_sizes, nu=nu, t_span=(0, 2.0), nt=1001, method="BDF", metric="L2_space_time"
    )

    mesh_sizes_short = [100, 200, 300]
    print("Running Chebyshev+RK45...")
    l2_errors_cheb_RK45, time_measure_cheb_RK45 = mesh_convergence_study_chebyshev(
        mesh_sizes_short, nu=nu, t_span=(0, 2.0), nt=1001, method="RK45", metric="L2_space_time"
    )

    
    np.savez(
        f'burgers1d_benchmark_nu{nu}.npz',
        l2_errors_chebyshev_BDF=l2_errors_chebyshev,
        timings_chebyshev_BDF=timings_chebyshev,
        l2_errors_bfpsm_RK45=l2_errors_RK45,
        time_measure_bfpsm_RK45=time_measure_RK45,
        l2_errors_bfpsm_BDF=l2_errors_BDF,
        time_measure_bfpsm_BDF=time_measure_BDF,
        l2_errors_chebyshev_RK45=l2_errors_cheb_RK45,
        time_measure_chebyshev_RK45=time_measure_cheb_RK45
    )

    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    plt.figure(figsize=(16, 6))

    # Set up global plotting parameters
    plt.rcParams.update({
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 16,
        'figure.titlesize': 24,
        'axes.grid': True,
        'grid.alpha': 0.5
    })

    plt.subplot(1, 2, 1)
    plt.loglog(mesh_sizes, l2_errors_BDF, 'o-', label='BFPSM+BDF', linewidth=1.5, markersize=8, color=default_colors[0])
    plt.loglog(mesh_sizes, l2_errors_chebyshev, 'o-', label='Chebyshev+BDF', linewidth=1.5, markersize=8, color=default_colors[1])
    plt.loglog(mesh_sizes, l2_errors_RK45, 's--', label='BFPSM+RK45', linewidth=1.5, markersize=8, color=default_colors[0])
    plt.loglog(mesh_sizes_short, l2_errors_cheb_RK45, 'o-', label='Chebyshev+RK45', linewidth=1.5, markersize=8, color=default_colors[1])
    plt.xlabel('$N$')
    plt.ylabel('$\|Error\|_2$')
    plt.title('(a)', loc='left', x = -0.1, fontsize=24)
    plt.grid(True, which='both')
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.loglog(mesh_sizes, time_measure_BDF, 'o-', label='BFPSM+BDF', linewidth=2, markersize=8, color=default_colors[0])
    plt.loglog(mesh_sizes, timings_chebyshev, 'o-', label='Chebyshev+BDF', linewidth=2, markersize=8, color=default_colors[1])
    plt.loglog(mesh_sizes, time_measure_RK45, 's--', label='BFPSM+RK45', linewidth=2, markersize=8, color=default_colors[0])
    plt.loglog(mesh_sizes_short, time_measure_cheb_RK45, 'o-', label='Chebyshev+RK45', linewidth=2, markersize=8, color=default_colors[1])
    plt.xlabel('$N$')
    plt.ylabel('Wall time [s]')
    plt.title('(b)', loc='left', x = -0.1, fontsize=24)
    plt.grid(True, which='both')
    plt.legend(loc='best')
    plt.ylim(1e-1, 2*1e1)
    plt.tight_layout()
    plt.savefig(f'figs/fig6.pdf', dpi=300, bbox_inches='tight')
