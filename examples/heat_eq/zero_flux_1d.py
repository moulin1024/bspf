from __future__ import annotations

import numpy as np
import numpy.typing as npt
import sys
import os

# Add src to path to use local bspf code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Import time steppers from bspf library
from bspf.utils import TimeStepperState, time_step
from bspf import bspf1d

Array = npt.NDArray[np.float64]

def heat_neumann_cosx(x, t, kappa=1.0):
    """
    Solution of u_t = kappa*u_xx on [0, 2π] with Neumann BCs
    and initial condition u(x,0) = cos(x).
    """
    return np.cos(x) * np.exp(-kappa * t)

# ---------- parameters ----------
nu = 1e-1            # diffusivity
L_domain = 3 * np.pi  # domain length (renamed to avoid conflict with Laplacian)
nx = 101             # grid points (including endpoints)
T = 10                # final time
dt = 0.01            # time step
nt = int(T / dt) + 1 # number of time steps

# ---------- grid, operator, IC ----------
x = np.linspace(0.0, L_domain, nx)
dx = x[1] - x[0]  # grid spacing
bf = bspf1d.from_grid(degree=9, x=x)

# Initial condition compatible with homogeneous Neumann BC (u_x=0 at ends)
u0 = np.cos(x)

# Note: Neumann BCs are enforced via BSPF operator during differentiation
# No explicit BC enforcement needed as BSPF handles it through KKT system

# ============================================================
# Create RHS function (inline implementation for debugging)
# ============================================================
def create_heat_rhs_1d(bspf_op, nu, neumann_bc=(0.0, 0.0)):
    """
    Create RHS function for the 1D heat equation.
    
    The equation: ∂u/∂t = ν*∂²u/∂x²
    
    Parameters
    ----------
    bspf_op : bspf1d
        BSPF operator for spatial differentiation
    nu : float
        Diffusion coefficient (ν > 0)
    neumann_bc : tuple, optional
        Neumann boundary conditions (left_flux, right_flux).
        Default: (0.0, 0.0) for zero flux (homogeneous Neumann)
    
    Returns
    -------
    rhs_func : callable
        RHS function with signature: rhs_func(u, t) -> du_dt
        where:
            u : array, current solution
            t : float, current time (not used but kept for interface consistency)
            du_dt : array, time derivative
    """
    def rhs_func(u: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        Compute RHS of heat equation: du/dt = nu * d²u/dx²
        
        Parameters
        ----------
        u : array
            Current solution
        t : float, optional
            Current time (not used, kept for interface consistency)
        
        Returns
        -------
        du_dt : array
            Time derivative
        """
        # Compute second derivative with Neumann BC
        _, d2u_dx2, _ = bspf_op.differentiate_1_2(u, neumann_bc=neumann_bc)
        return nu * d2u_dx2
    
    return rhs_func

# Create RHS function
rhs_func = create_heat_rhs_1d(bf, nu, neumann_bc=(0.0, 0.0))

# ---------- Time integration with RK45 ----------
print("\n" + "="*60)
print("Running RK45 embedded time integration...")
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

# ---------- Time integration with RK23 ----------
print("\n" + "="*60)
print("Running RK23 time integration...")
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

# ---------- Time integration with RK45 + enforced_zero_flux ----------
print("\n" + "="*60)
print("Running RK45 with enforced_zero_flux boundary correction...")
print("="*60)

# Create RHS function that applies enforced_zero_flux at each step
def create_heat_rhs_with_enforced_zero_flux(bspf_op, nu):
    """
    Create RHS function that applies enforced_zero_flux boundary correction.
    """
    def rhs_func(u: np.ndarray, t: float = 0.0) -> np.ndarray:
        # Apply enforced_zero_flux to correct boundary values
        u_corrected = u.copy()
        f_left_corrected, f_right_corrected = bspf_op.enforced_zero_flux(u)
        u_corrected[0] = f_left_corrected
        u_corrected[-1] = f_right_corrected
        
        # Compute second derivative (no neumann_bc needed since boundaries are corrected)
        _, d2u_dx2, _ = bspf_op.differentiate_1_2(u_corrected)
        return nu * d2u_dx2
    
    return rhs_func

rhs_func_enforced = create_heat_rhs_with_enforced_zero_flux(bf, nu)

state_rk45_enforced = TimeStepperState(u0.copy(), t_init=0.0, dt=dt, method='rk45')
U_rk45_enforced = np.empty((nt, nx), dtype=np.float64)
U_rk45_enforced[0] = u0.copy()
times_rk45_enforced = np.zeros(nt)
times_rk45_enforced[0] = 0.0

for step in range(1, nt):
    u_next = time_step(state_rk45_enforced, dt, rhs_func_enforced, method='rk45')
    U_rk45_enforced[step] = state_rk45_enforced.get_current()
    times_rk45_enforced[step] = state_rk45_enforced.get_current_time()

u_exact_rk45_enforced = heat_neumann_cosx(x[None, :], times_rk45_enforced[:, None], kappa=nu)
error_rk45_enforced = np.abs(U_rk45_enforced - u_exact_rk45_enforced)
max_error_rk45_enforced = np.max(error_rk45_enforced)
l2_error_rk45_enforced = np.sqrt(np.mean(error_rk45_enforced**2))

print(f"RK45 + enforced_zero_flux Results:")
print(f"  Max error: {max_error_rk45_enforced:.6e}")
print(f"  L2 error: {l2_error_rk45_enforced:.6e}")

# Check boundary corrections at final time
f_left_final, f_right_final = bf.enforced_zero_flux(U_rk45_enforced[-1])
print(f"  Final boundary corrections:")
print(f"    Left:  original={U_rk45_enforced[-1][0]:.8f}, corrected={f_left_final:.8f}, diff={f_left_final-U_rk45_enforced[-1][0]:.8e}")
print(f"    Right: original={U_rk45_enforced[-1][-1]:.8f}, corrected={f_right_final:.8f}, diff={f_right_final-U_rk45_enforced[-1][-1]:.8e}")

# ---------- quick BC check ----------
du_dx_rk45, _, _ = bf.differentiate_1_2(U_rk45[-1], neumann_bc=(0.0, 0.0))
du_dx_rk23, _, _ = bf.differentiate_1_2(U_rk23[-1], neumann_bc=(0.0, 0.0))
du_dx_rk45_enforced, _, _ = bf.differentiate_1_2(U_rk45_enforced[-1])
print(f"\nNeumann BC check (final time):")
print(f"  RK45:  u_x(0)={du_dx_rk45[0]: .3e}, u_x(L)={du_dx_rk45[-1]: .3e}")
print(f"  RK23: u_x(0)={du_dx_rk23[0]: .3e}, u_x(L)={du_dx_rk23[-1]: .3e}")
print(f"  RK45+enforced: u_x(0)={du_dx_rk45_enforced[0]: .3e}, u_x(L)={du_dx_rk45_enforced[-1]: .3e}")

# ---------- Comparison ----------
print("\n" + "="*60)
print("Comparison:")
print("="*60)
print(f"RK45              - Max error: {max_error_rk45:.6e}, L2 error: {l2_error_rk45:.6e}")
print(f"RK23              - Max error: {max_error_rk23:.6e}, L2 error: {l2_error_rk23:.6e}")
print(f"RK45+enforced_zero_flux - Max error: {max_error_rk45_enforced:.6e}, L2 error: {l2_error_rk45_enforced:.6e}")
print(f"\nError ratios:")
print(f"  RK23/RK45: Max={max_error_rk23/max_error_rk45:.3f}, L2={l2_error_rk23/l2_error_rk45:.3f}")
print(f"  RK45_enforced/RK45: Max={max_error_rk45_enforced/max_error_rk45:.3f}, L2={l2_error_rk45_enforced/l2_error_rk45:.3f}")

# ---------- Visualization ----------
try:
    import matplotlib.pyplot as plt

    # Create figure with organized layout: 2 rows x 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Heat Equation: u_t = νu_xx with Neumann BCs (ν={nu:.1e}, T={T}s)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Solutions at initial and final time
    ax = axes[0, 0]
    ax.plot(x, U_rk45[0], 'b-', label="Initial (t=0)", linewidth=2)
    ax.plot(x, U_rk45[-1], 'r-', label=f"RK45 (t={T})", linewidth=2)
    ax.plot(x, U_rk23[-1], 'm-.', label=f"RK23 (t={T})", linewidth=2)
    ax.plot(x, U_rk45_enforced[-1], 'g--', label=f"RK45+enforced (t={T})", linewidth=2)
    ax.plot(x, u_exact_rk45[-1], 'k:', label="Exact", linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title("Solutions")
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Error comparison at final time
    ax = axes[0, 1]
    ax.plot(x, error_rk45[-1], 'r-', label="RK45", linewidth=2)
    ax.plot(x, error_rk23[-1], 'm-.', label="RK23", linewidth=2)
    ax.plot(x, error_rk45_enforced[-1], 'g--', label="RK45+enforced", linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("|Error|")
    ax.set_title(f"Error at Final Time (t={T}s)")
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Max error evolution over time
    ax = axes[0, 2]
    max_error_rk45_time = np.max(error_rk45, axis=1)
    max_error_rk23_time = np.max(error_rk23, axis=1)
    max_error_rk45_enforced_time = np.max(error_rk45_enforced, axis=1)
    ax.plot(times_rk45, max_error_rk45_time, 'r-', label="RK45", linewidth=2)
    ax.plot(times_rk23, max_error_rk23_time, 'm-.', label="RK23", linewidth=2)
    ax.plot(times_rk45_enforced, max_error_rk45_enforced_time, 'g--', label="RK45+enforced", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Max |Error|")
    ax.set_title("Error Evolution Over Time")
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Boundary value corrections over time
    ax = axes[1, 0]
    boundary_corrections_left = []
    boundary_corrections_right = []
    for step in range(0, nt, max(1, nt//50)):  # Sample every ~50 steps
        f_left, f_right = bf.enforced_zero_flux(U_rk45_enforced[step])
        boundary_corrections_left.append(f_left - U_rk45_enforced[step][0])
        boundary_corrections_right.append(f_right - U_rk45_enforced[step][-1])
    steps_sampled = np.arange(0, nt, max(1, nt//50))[:len(boundary_corrections_left)]
    ax.plot(times_rk45_enforced[steps_sampled], boundary_corrections_left, 'b-', label="Left boundary", linewidth=2)
    ax.plot(times_rk45_enforced[steps_sampled], boundary_corrections_right, 'r-', label="Right boundary", linewidth=2)
    ax.axhline(0.0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Boundary Correction")
    ax.set_title("Boundary Value Corrections Over Time")
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Boundary region zoom (left)
    ax = axes[1, 1]
    zoom_width = 0.5
    mask = x <= x[0] + zoom_width
    ax.plot(x[mask], U_rk45[-1][mask], 'r-', label="RK45", linewidth=2)
    ax.plot(x[mask], U_rk45_enforced[-1][mask], 'g--', label="RK45+enforced", linewidth=2)
    ax.plot(x[mask], u_exact_rk45[-1][mask], 'k:', label="Exact", linewidth=2)
    ax.axvline(x[0], color='k', linestyle='--', linewidth=1, alpha=0.5, label='Boundary')
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title(f"Left Boundary Region (t={T}s)")
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Boundary region zoom (right)
    ax = axes[1, 2]
    mask = x >= x[-1] - zoom_width
    ax.plot(x[mask], U_rk45[-1][mask], 'r-', label="RK45", linewidth=2)
    ax.plot(x[mask], U_rk45_enforced[-1][mask], 'g--', label="RK45+enforced", linewidth=2)
    ax.plot(x[mask], u_exact_rk45[-1][mask], 'k:', label="Exact", linewidth=2)
    ax.axvline(x[-1], color='k', linestyle='--', linewidth=1, alpha=0.5, label='Boundary')
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title(f"Right Boundary Region (t={T}s)")
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
except Exception as exc:  # pragma: no cover - visualization is optional
    print("Plot skipped:", exc)
