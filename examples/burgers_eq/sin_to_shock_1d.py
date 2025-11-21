"""
1D Burgers' equation solver using BSPF spatial discretization and RK45 time stepping.

Solves: u_t + u*u_x = nu*u_xx on [0,L]x[0,T] with zero Dirichlet boundary conditions.
Initial condition: u(x,0) = sin(x)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from bspf import bspf1d
from bspf.utils import TimeStepperState, time_step

# ============================================================================
# Configuration Parameters
# ============================================================================
# Physical parameters
nu = 0.01          # viscosity
L = 2.0 * np.pi    # domain length
T = 15.0            # final time

# Spatial discretization
nx = 512           # number of grid points
degree = 9         # B-spline degree
n_basis = 2 * degree  # number of basis functions
num_boundary_points = degree  # boundary points

# Time discretization
dt = T / 5000       # time step
nt = int(T / dt) + 1  # number of time steps

# Plotting
plot_times = [0.0, 0.5*T, T]  # times to plot

# ============================================================================
# Grid and BSPF Operator Setup
# ============================================================================
x = np.linspace(0, L, nx)
dx = L / (nx - 1)

# Create BSPF operator
bspf_op = bspf1d.from_grid(
    degree=degree,
    order=degree,
    n_basis=n_basis,
    num_boundary_points=num_boundary_points,
    x=x
)

# ============================================================================
# Initial Condition
# ============================================================================
# Initial condition: u(x,0) = sin(x)
u0 = np.sin(0.75*x)

# ============================================================================
# RHS Function
# ============================================================================
def burgers_rhs(u, bspf_op, nu):
    """
    RHS for Burgers' equation: u_t = nu*u_xx - u*u_x
    
    Enforces zero Dirichlet BCs by:
      1) Setting boundary values to zero
      2) Computing spatial derivatives
      3) Setting du/dt = 0 at boundaries to keep them fixed
    """
    u_work = u.copy()
    
    # Enforce zero Dirichlet BCs
    u_work[0] = 0.0
    u_work[-1] = -1.0
    
    # Compute spatial derivatives
    du_dx, d2u_dx2, _ = bspf_op.differentiate_1_2(u_work)
    
    # Burgers' equation RHS: u_t = nu*u_xx - u*u_x
    rhs = nu * d2u_dx2 - u_work * du_dx
    
    # Set RHS to zero at boundaries to keep them fixed
    rhs[0] = 0.0
    rhs[-1] = 0.0
    
    return rhs

# ============================================================================
# Time Integration
# ============================================================================
print("="*60)
print("Solving Burgers' equation with BSPF + RK45")
print("="*60)
print(f"Parameters:")
print(f"  nu = {nu}")
print(f"  nx = {nx}, degree = {degree}")
print(f"  dt = {dt:.6e}, nt = {nt}")
print(f"  T = {T}")
print()

# Initialize time stepper state
state = TimeStepperState(u0.copy(), t_init=0.0, dt=dt, method='rk45')

# Create RHS function with bound arguments
def rhs_func(u):
    """RHS function for time stepper (takes only u as argument)."""
    return burgers_rhs(u, bspf_op, nu)

# Storage arrays
U = np.empty((nt, nx), dtype=np.float64)
U[0] = u0.copy()
times = np.zeros(nt)
times[0] = 0.0

# Time integration loop
start_time = time.time()
for step in tqdm(range(1, nt), desc="Time integration", unit="step", ncols=100):
    u_next = time_step(state, dt, rhs_func, method='rk45')
    U[step] = state.get_current()
    times[step] = state.get_current_time()

time_integration_time = time.time() - start_time
print(f"\nTime integration completed in {time_integration_time:.4f} seconds")
print(f"Average time per step: {time_integration_time/(nt-1):.6e} seconds")

# Enforce zero Dirichlet BCs on final solution
U[:, 0] = 0.0
U[:, -1] = -1.0

# ============================================================================
# Visualization
# ============================================================================
plt.rcParams.update({'font.size': 16})

fig = plt.figure(figsize=(16, 6))

# Left: solution evolution
plt.subplot(1, 2, 1)
t_idx = [np.abs(times - pt).argmin() for pt in plot_times]
for i in t_idx:
    label = f't = {times[i]:.2f}'
    plt.plot(x, U[i, :], '-', label=label, linewidth=2)
plt.xlabel('$x$')
plt.ylabel('$u(x,t)$')
plt.title('Solution Evolution')
plt.grid(True, alpha=0.3)
plt.legend(loc='best')

# Right: space-time plot
plt.subplot(1, 2, 2)
im = plt.imshow(U.T, aspect='auto', origin='lower', 
                extent=[times[0], times[-1], x[0], x[-1]],
                cmap='viridis', interpolation='bilinear')
plt.colorbar(im, label='$u(x,t)$')
plt.xlabel('Time $t$')
plt.ylabel('Space $x$')
plt.title('Space-Time Evolution')

plt.tight_layout()
plt.show()
