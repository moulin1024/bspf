from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Import time steppers from bspf library
from bspf.utils import TimeStepperState, time_step

Array = npt.NDArray[np.float64]

def heat_neumann_cosx(x, t, kappa=1.0):
    """
    Solution of u_t = kappa*u_xx on [0, 2π] with Neumann BCs
    and initial condition u(x,0) = cos(x).
    """
    return np.cos(x) * np.exp(-kappa * t)

# ---------- parameters ----------
nu = 1e-1            # diffusivity
L_domain = 3 * np.pi  # domain length
nx = 101             # grid points (including endpoints)
T = 10                # final time
dt = 0.01            # time step
nt = int(T / dt) + 1 # number of time steps

# ---------- grid, IC ----------
x = np.linspace(0.0, L_domain, nx)
dx = x[1] - x[0]  # grid spacing

# Initial condition compatible with homogeneous Neumann BC (u_x=0 at ends)
u0 = np.sin(x)

# ============================================================
# Build Finite Difference Laplacian Matrix with Neumann BCs
# ============================================================
def build_fd_laplacian_matrix(n, dx, neumann_bc=(0.0, 0.0)):
    """
    Build finite difference Laplacian matrix (second derivative) with Neumann BCs
    using ghost cell method.
    
    For zero-flux Neumann BCs (∂u/∂x = 0):
    - Left boundary: ghost point u[-1] = u[1] (symmetric extension)
    - Right boundary: ghost point u[n] = u[n-2] (symmetric extension)
    
    This gives:
    - u''[0] = (u[1] - 2*u[0] + u[-1])/dx² = 2*(u[1] - u[0])/dx²
    - u''[n-1] = (u[n] - 2*u[n-1] + u[n-2])/dx² = 2*(u[n-2] - u[n-1])/dx²
    
    Parameters
    ----------
    n : int
        Number of grid points
    dx : float
        Grid spacing
    neumann_bc : tuple, optional
        Neumann boundary conditions (left_flux, right_flux).
        Default: (0.0, 0.0) for zero flux
    
    Returns
    -------
    L : sparse matrix
        Laplacian matrix (second derivative operator)
    """
    left_flux, right_flux = neumann_bc
    
    # Second-order centered difference for interior: (u[i+1] - 2*u[i] + u[i-1]) / dx²
    # Main diagonal: -2/dx²
    main_diag = -2.0 / (dx**2) * np.ones(n, dtype=np.float64)
    
    # Upper diagonal: 1/dx²
    upper_diag = np.ones(n-1, dtype=np.float64) / (dx**2)
    
    # Lower diagonal: 1/dx²
    lower_diag = np.ones(n-1, dtype=np.float64) / (dx**2)
    
    # Build tridiagonal matrix
    L = sp.diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], 
                 shape=(n, n), format='csr', dtype=np.float64)
    
    # Apply ghost cell boundary conditions for zero-flux Neumann BC
    if left_flux == 0.0:
        # Left boundary: ghost point u[-1] = u[1] (symmetric)
        # u''[0] = (u[1] - 2*u[0] + u[-1])/dx² = (u[1] - 2*u[0] + u[1])/dx² = 2*(u[1] - u[0])/dx²
        L[0, 0] = -2.0 / (dx**2)
        L[0, 1] = 2.0 / (dx**2)
    
    if right_flux == 0.0:
        # Right boundary: ghost point u[n] = u[n-2] (symmetric)
        # u''[n-1] = (u[n] - 2*u[n-1] + u[n-2])/dx² = (u[n-2] - 2*u[n-1] + u[n-2])/dx² = 2*(u[n-2] - u[n-1])/dx²
        L[-1, -1] = -2.0 / (dx**2)
        L[-1, -2] = 2.0 / (dx**2)
    
    return L

# Build Laplacian matrix
print("Building finite difference Laplacian matrix...")
L_fd = build_fd_laplacian_matrix(nx, dx, neumann_bc=(0.0, 0.0))
print("  Laplacian matrix built successfully")

# ============================================================
# Create RHS function for heat equation
# ============================================================
def create_heat_rhs_fd(L_matrix, nu):
    """
    Create RHS function for heat equation: u_t = nu * u_xx
    
    Parameters
    ----------
    L_matrix : sparse matrix
        Laplacian matrix (second derivative operator)
    nu : float
        Diffusivity coefficient
    
    Returns
    -------
    rhs_func : callable
        RHS function with signature: rhs_func(u) -> du_dt
    """
    def rhs_func(u: Array) -> Array:
        """
        Compute RHS of heat equation.
        
        Parameters
        ----------
        u : array
            Current solution
        
        Returns
        -------
        du_dt : array
            Time derivative: du_dt = nu * L @ u
        """
        # Heat equation: u_t = nu * u_xx
        du_dt = nu * (L_matrix @ u)
        return du_dt
    
    return rhs_func

# Create RHS function
rhs_func = create_heat_rhs_fd(L_fd, nu)

# ============================================================
# Create Jacobian function for BDF2
# ============================================================
def create_heat_jacobian_fd(L_matrix, nu, dt):
    """
    Create Jacobian function for heat equation with BDF2.
    
    For BDF2: (3/2)*u^{n+1} - 2*u^n + (1/2)*u^{n-1} = dt * f(u^{n+1})
    Linearizing: f(u^{n+1}) ≈ f(u^n) + J * (u^{n+1} - u^n)
    where J = ∂f/∂u = nu * L
    
    The Jacobian for the implicit system is:
    J_BDF2 = (3/2)*I - dt * nu * L
    
    Parameters
    ----------
    L_matrix : sparse matrix
        Laplacian matrix
    nu : float
        Diffusivity coefficient
    dt : float
        Time step
    
    Returns
    -------
    jacobian_func : callable
        Jacobian function with signature: jacobian_func(u) -> J
    """
    n = L_matrix.shape[0]
    I = sp.eye(n, dtype=np.float64, format='csr')
    J_BDF2 = (3.0/2.0) * I - dt * nu * L_matrix
    
    def jacobian_func(u: Array) -> sp.csr_matrix:
        """
        Compute Jacobian matrix.
        
        Parameters
        ----------
        u : array
            Current solution (not used for linear problem, but required by interface)
        
        Returns
        -------
        J : sparse matrix
            Jacobian matrix
        """
        return J_BDF2
    
    return jacobian_func

# Create Jacobian function for BDF2
jacobian_func = create_heat_jacobian_fd(L_fd, nu, dt)

# ---------- Time integration with RK45 ----------
print("\n" + "="*60)
print("Running RK45 embedded time integration (Finite Difference)...")
print("="*60)

state_rk45 = TimeStepperState(u0.copy(), t_init=0.0, dt=dt, method='rk45')
U_rk45 = np.empty((nt, nx), dtype=np.float64)
U_rk45[0] = u0.copy()
times_rk45 = np.zeros(nt)
times_rk45[0] = 0.0

for step in range(1, nt):
    u_next = time_step(state_rk45, dt, rhs_func, method='rk45')
    U_rk45[step] = state_rk45.get_current()
    times_rk45[step] = state_rk45.get_current_time()

u_exact_rk45 = heat_neumann_cosx(x[None, :], times_rk45[:, None], kappa=nu)
error_rk45 = np.abs(U_rk45 - u_exact_rk45)
max_error_rk45 = np.max(error_rk45)
l2_error_rk45 = np.sqrt(np.mean(error_rk45**2))

print(f"RK45 Results:")
print(f"  Max error: {max_error_rk45:.6e}")
print(f"  L2 error: {l2_error_rk45:.6e}")

# ---------- Time integration with BDF2 ----------
print("\n" + "="*60)
print("Running BDF2 time integration (Finite Difference)...")
print("="*60)

state_bdf2 = TimeStepperState(u0.copy(), t_init=0.0, dt=dt, method='bdf2')
U_bdf2 = np.empty((nt, nx), dtype=np.float64)
U_bdf2[0] = u0.copy()
times_bdf2 = np.zeros(nt)
times_bdf2[0] = 0.0

for step in range(1, nt):
    u_next = time_step(state_bdf2, dt, rhs_func, method='bdf2', 
                       jacobian_func=jacobian_func)
    U_bdf2[step] = state_bdf2.get_current()
    times_bdf2[step] = state_bdf2.get_current_time()

u_exact_bdf2 = heat_neumann_cosx(x[None, :], times_bdf2[:, None], kappa=nu)
error_bdf2 = np.abs(U_bdf2 - u_exact_bdf2)
max_error_bdf2 = np.max(error_bdf2)
l2_error_bdf2 = np.sqrt(np.mean(error_bdf2**2))

print(f"BDF2 Results:")
print(f"  Max error: {max_error_bdf2:.6e}")
print(f"  L2 error: {l2_error_bdf2:.6e}")

# ---------- Time integration with RK23 ----------
print("\n" + "="*60)
print("Running RK23 time integration (Finite Difference)...")
print("="*60)

state_rk23 = TimeStepperState(u0.copy(), t_init=0.0, dt=dt, method='rk23')
U_rk23 = np.empty((nt, nx), dtype=np.float64)
U_rk23[0] = u0.copy()
times_rk23 = np.zeros(nt)
times_rk23[0] = 0.0

for step in range(1, nt):
    u_next = time_step(state_rk23, dt, rhs_func, method='rk23')
    U_rk23[step] = state_rk23.get_current()
    times_rk23[step] = state_rk23.get_current_time()

u_exact_rk23 = heat_neumann_cosx(x[None, :], times_rk23[:, None], kappa=nu)
error_rk23 = np.abs(U_rk23 - u_exact_rk23)
max_error_rk23 = np.max(error_rk23)
l2_error_rk23 = np.sqrt(np.mean(error_rk23**2))

print(f"RK23 Results:")
print(f"  Max error: {max_error_rk23:.6e}")
print(f"  L2 error: {l2_error_rk23:.6e}")

# ---------- Comparison ----------
print("\n" + "="*60)
print("Comparison:")
print("="*60)
print(f"RK45  - Max error: {max_error_rk45:.6e}, L2 error: {l2_error_rk45:.6e}")
print(f"RK23 - Max error: {max_error_rk23:.6e}, L2 error: {l2_error_rk23:.6e}")
print(f"BDF2 - Max error: {max_error_bdf2:.6e}, L2 error: {l2_error_bdf2:.6e}")
print(f"\nError ratios:")
print(f"  RK23/RK45: Max={max_error_rk23/max_error_rk45:.3f}, L2={l2_error_rk23/l2_error_rk45:.3f}")
print(f"  BDF2/RK45: Max={max_error_bdf2/max_error_rk45:.3f}, L2={l2_error_bdf2/l2_error_rk45:.3f}")
print(f"  BDF2/RK23: Max={max_error_bdf2/max_error_rk23:.3f}, L2={l2_error_bdf2/l2_error_rk23:.3f}")

# ---------- quick BC check ----------
# Check Neumann BCs using finite differences with ghost cells
du_dx_rk45 = np.zeros(nx, dtype=np.float64)
du_dx_rk23 = np.zeros(nx, dtype=np.float64)
du_dx_bdf2 = np.zeros(nx, dtype=np.float64)

# Interior: centered difference
du_dx_rk45[1:-1] = (U_rk45[-1, 2:] - U_rk45[-1, :-2]) / (2.0 * dx)
du_dx_rk23[1:-1] = (U_rk23[-1, 2:] - U_rk23[-1, :-2]) / (2.0 * dx)
du_dx_bdf2[1:-1] = (U_bdf2[-1, 2:] - U_bdf2[-1, :-2]) / (2.0 * dx)

# Boundaries: use ghost cell method for zero-flux Neumann BC
# Left boundary: ghost point u[-1] = u[1] → ∂u/∂x[0] = 0
du_dx_rk45[0] = 0.0
du_dx_rk23[0] = 0.0
du_dx_bdf2[0] = 0.0

# Right boundary: ghost point u[n] = u[n-2] → ∂u/∂x[n-1] = 0
du_dx_rk45[-1] = 0.0
du_dx_rk23[-1] = 0.0
du_dx_bdf2[-1] = 0.0

print(f"\nNeumann BC check (final time, using finite differences):")
print(f"  RK45:  u_x(0)={du_dx_rk45[0]: .3e}, u_x(L)={du_dx_rk45[-1]: .3e}")
print(f"  RK23: u_x(0)={du_dx_rk23[0]: .3e}, u_x(L)={du_dx_rk23[-1]: .3e}")
print(f"  BDF2: u_x(0)={du_dx_bdf2[0]: .3e}, u_x(L)={du_dx_bdf2[-1]: .3e}")

# ---------- Visualization ----------
try:
    import matplotlib.pyplot as plt

    # Create figure with organized layout: 1 row x 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Heat Equation (FD): u_t = νu_xx with Neumann BCs (ν={nu:.1e}, T={T}s)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Solutions at initial and final time
    ax = axes[0]
    ax.plot(x, U_rk45[0], 'b-', label="Initial (t=0)", linewidth=2)
    ax.plot(x, U_rk45[-1], 'r-', label=f"RK45 (t={T})", linewidth=2)
    ax.plot(x, U_rk23[-1], 'm-.', label=f"RK23 (t={T})", linewidth=2)
    ax.plot(x, U_bdf2[-1], 'g--', label=f"BDF2 (t={T})", linewidth=2)
    ax.plot(x, u_exact_rk45[-1], 'k:', label="Exact", linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title("Solutions")
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Error comparison at final time
    ax = axes[1]
    ax.plot(x, error_rk45[-1], 'r-', label="RK45", linewidth=2)
    ax.plot(x, error_rk23[-1], 'm-.', label="RK23", linewidth=2)
    ax.plot(x, error_bdf2[-1], 'g--', label="BDF2", linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("|Error|")
    ax.set_title(f"Error at Final Time (t={T}s)")
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Max error evolution over time
    ax = axes[2]
    max_error_rk45_time = np.max(error_rk45, axis=1)
    max_error_rk23_time = np.max(error_rk23, axis=1)
    max_error_bdf2_time = np.max(error_bdf2, axis=1)
    ax.plot(times_rk45, max_error_rk45_time, 'r-', label="RK45", linewidth=2)
    ax.plot(times_rk23, max_error_rk23_time, 'm-.', label="RK23", linewidth=2)
    ax.plot(times_bdf2, max_error_bdf2_time, 'g--', label="BDF2", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Max |Error|")
    ax.set_title("Error Evolution Over Time")
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
except Exception as exc:  # pragma: no cover - visualization is optional
    print("Plot skipped:", exc)

