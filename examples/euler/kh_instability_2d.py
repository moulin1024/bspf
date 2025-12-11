"""
2D Compressible Kelvin-Helmholtz Instability using BSPF.

The Kelvin-Helmholtz instability occurs when two fluid layers with different
velocities are in contact, creating a shear layer. A small perturbation grows
exponentially, leading to vortex roll-up and mixing.

This implementation uses the compressible Euler equations with:
- Periodic boundary conditions in x-direction
- Periodic boundary conditions in y-direction (can be modified to walls)
- Smooth initial velocity shear profile
- Small sinusoidal perturbation to trigger instability
"""

import numpy as np
import matplotlib.pyplot as plt
from bspf import bspf2d, TimeStepperState, time_step

GAMMA = 1.4

# Small floors to avoid weird states
RHO_FLOOR = 1e-10
P_FLOOR = 1e-10

# ========== Basic conversions ==========
def prim_to_cons(rho, u, v, p, gamma=GAMMA):
    """Convert primitive to conservative variables."""
    E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)
    return np.stack([rho, rho * u, rho * v, E], axis=0)  # shape (4, ny, nx)

def cons_to_prim(U, gamma=GAMMA):
    """Convert conservative to primitive variables."""
    rho = U[0]
    mom_x = U[1]
    mom_y = U[2]
    E = U[3]
    u = mom_x / (rho + RHO_FLOOR)
    v = mom_y / (rho + RHO_FLOOR)
    p = (gamma - 1.0) * (E - 0.5 * rho * (u**2 + v**2))
    p = np.maximum(p, P_FLOOR)
    return rho, u, v, p

def euler_flux_x(U, gamma=GAMMA):
    """x-direction flux F(U)."""
    rho, u, v, p = cons_to_prim(U, gamma=gamma)
    F = np.zeros_like(U)
    F[0] = rho * u
    F[1] = rho * u**2 + p
    F[2] = rho * u * v
    F[3] = (U[3] + p) * u
    return F

def euler_flux_y(U, gamma=GAMMA):
    """y-direction flux G(U)."""
    rho, u, v, p = cons_to_prim(U, gamma=gamma)
    G = np.zeros_like(U)
    G[0] = rho * v
    G[1] = rho * u * v
    G[2] = rho * v**2 + p
    G[3] = (U[3] + p) * v
    return G

def max_wave_speed(U, gamma=GAMMA):
    """Compute maximum wave speed for CFL condition."""
    rho, u, v, p = cons_to_prim(U, gamma=gamma)
    rho = np.maximum(rho, RHO_FLOOR)
    p = np.maximum(p, P_FLOOR)
    c = np.sqrt(gamma * p / rho)
    a = np.sqrt(u**2 + v**2) + c
    return np.max(a)

# ========== Initial conditions ==========
def kh_initial_condition(x, y, Lx, Ly, 
                        u_top=0.5, u_bottom=-0.5,
                        rho_top=2.0, rho_bottom=1.0,
                        p0=1.0, gamma=GAMMA,
                        shear_width=0.05, pert_amplitude=0.01,
                        pert_wavenumber=2.0):
    """
    Create initial condition for Kelvin-Helmholtz instability.
    
    Parameters:
    -----------
    x, y : arrays
        Grid coordinates (2D meshgrids)
    Lx, Ly : float
        Domain size
    u_top, u_bottom : float
        Velocities in top and bottom layers
    rho_top, rho_bottom : float
        Densities in top and bottom layers
    p0 : float
        Initial pressure (constant)
    gamma : float
        Adiabatic index
    shear_width : float
        Width of the shear layer (as fraction of Ly)
    pert_amplitude : float
        Amplitude of initial perturbation
    pert_wavenumber : float
        Wavenumber of initial perturbation (in units of 2π/Lx)
    
    Returns:
    --------
    rho, u, v, p : arrays
        Primitive variables
    """
    # Normalized y coordinate (0 to 1)
    y_norm = (y - y.min()) / (y.max() - y.min())
    
    # Center of shear layer
    y_center = 0.5
    
    # Smooth velocity profile using tanh
    # u(y) transitions from u_bottom to u_top across the shear layer
    u_shear = 0.5 * (u_top + u_bottom) + 0.5 * (u_top - u_bottom) * np.tanh((y_norm - y_center) / shear_width)
    
    # Smooth density profile
    rho_shear = 0.5 * (rho_top + rho_bottom) + 0.5 * (rho_top - rho_bottom) * np.tanh((y_norm - y_center) / shear_width)
    
    # Initial perturbation: sinusoidal in x, localized in y
    # Perturbation affects the interface location
    kx = pert_wavenumber * 2.0 * np.pi / Lx
    pert_y = pert_amplitude * np.sin(kx * x) * np.exp(-((y_norm - y_center) / (2.0 * shear_width))**2)
    
    # Perturbed interface location
    y_interface = y_center + pert_y
    
    # Recompute with perturbed interface
    u = 0.5 * (u_top + u_bottom) + 0.5 * (u_top - u_bottom) * np.tanh((y_norm - y_interface) / shear_width)
    rho = 0.5 * (rho_top + rho_bottom) + 0.5 * (rho_top - rho_bottom) * np.tanh((y_norm - y_interface) / shear_width)
    
    # Initial vertical velocity: small perturbation
    v = pert_amplitude * 0.1 * np.sin(kx * x) * np.exp(-((y_norm - y_center) / (2.0 * shear_width))**2)
    
    # Constant pressure
    p = np.full_like(rho, p0)
    
    return rho, u, v, p

# ========== RHS function ==========
def euler_rhs_2d_bspf(U, bspf_op, gamma=GAMMA, nu=0.0, dx=None, dy=None):
    """
    RHS for 2D Euler equations using BSPF.
    
    Parameters:
    -----------
    U : array, shape (4, ny, nx)
        Conservative variables [rho, rho*u, rho*v, E]
    bspf_op : bspf2d
        BSPF operator for 2D spatial differentiation
    gamma : float
        Adiabatic index
    nu : float
        Artificial viscosity coefficient (optional, for stability)
    dx, dy : float
        Grid spacing (for viscosity scaling)
    
    Returns:
    --------
    dU_dt : array, shape (4, ny, nx)
        Time derivative of conservative variables
    """
    # Compute fluxes
    F = euler_flux_x(U, gamma=gamma)  # shape (4, ny, nx)
    G = euler_flux_y(U, gamma=gamma)  # shape (4, ny, nx)
    
    # Compute flux derivatives
    dF_dx = np.zeros_like(F)
    dG_dy = np.zeros_like(G)
    
    for i in range(4):  # For each conservative variable
        dF_dx[i] = bspf_op.partial_dx(F[i], order=1)
        dG_dy[i] = bspf_op.partial_dy(G[i], order=1)
    
    # RHS: -∂F/∂x - ∂G/∂y
    dU_dt = -dF_dx - dG_dy
    
    # Add artificial viscosity if requested (helps with stability)
    if nu > 0.0 and dx is not None and dy is not None:
        # Reference wave speed for scaling
        rho, u, v, p = cons_to_prim(U, gamma=gamma)
        rho = np.maximum(rho, RHO_FLOOR)
        p = np.maximum(p, P_FLOOR)
        c = np.sqrt(gamma * p / rho)
        a_ref = np.max(np.sqrt(u**2 + v**2) + c)
        
        # Constant viscosity scaled by grid and wave speed
        nu_local = nu * min(dx, dy) * a_ref
        
        for i in range(4):
            d2U_dx2 = bspf_op.partial_dxx(U[i])
            d2U_dy2 = bspf_op.partial_dyy(U[i])
            dU_dt[i] += nu_local * (d2U_dx2 + d2U_dy2)
    
    return dU_dt

def enforce_periodic_bc(U):
    """
    Enforce periodic boundary conditions.
    
    For periodic BCs, we ensure the boundaries match by averaging.
    This ensures continuity across periodic boundaries.
    
    Parameters:
    -----------
    U : array, shape (4, ny, nx)
        Conservative variables
    
    Returns:
    --------
    U_bc : array
        U with periodic BCs enforced
    """
    U_bc = U.copy()
    
    # Periodic in x: average values at x=0 and x=Lx
    avg_x = 0.5 * (U_bc[:, :, 0] + U_bc[:, :, -1])
    U_bc[:, :, 0] = avg_x
    U_bc[:, :, -1] = avg_x
    
    # Periodic in y: average values at y=0 and y=Ly
    avg_y = 0.5 * (U_bc[:, 0, :] + U_bc[:, -1, :])
    U_bc[:, 0, :] = avg_y
    U_bc[:, -1, :] = avg_y
    
    return U_bc

# ========== Main solver ==========
def solve_kh_instability_2d(nx=256, ny=256, Lx=1.0, Ly=1.0,
                            t_end=2.0, cfl=0.5, degree=5,
                            method='rk4', use_gpu=False, nu=0.0,
                            u_top=0.5, u_bottom=-0.5,
                            rho_top=2.0, rho_bottom=1.0,
                            p0=1.0, shear_width=0.05,
                            pert_amplitude=0.01, pert_wavenumber=2.0,
                            save_interval=10):
    """
    Solve 2D Kelvin-Helmholtz instability using BSPF.
    
    Parameters:
    -----------
    nx, ny : int
        Number of grid points in x and y
    Lx, Ly : float
        Domain size
    t_end : float
        Final time
    cfl : float
        CFL number
    degree : int
        B-spline degree
    method : str
        Time stepping method
    use_gpu : bool
        Use GPU acceleration
    nu : float
        Artificial viscosity coefficient (for stability)
    u_top, u_bottom : float
        Velocities in top and bottom layers
    rho_top, rho_bottom : float
        Densities in top and bottom layers
    p0 : float
        Initial pressure
    shear_width : float
        Width of shear layer
    pert_amplitude : float
        Amplitude of initial perturbation
    pert_wavenumber : float
        Wavenumber of initial perturbation
    save_interval : int
        Save solution every N steps
    
    Returns:
    --------
    x, y : arrays
        Grid coordinates
    t : array
        Time points
    U_history : list
        Solution history
    """
    # Grid
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    X, Y = np.meshgrid(x, y)  # (ny, nx)
    
    # Initial condition
    rho0, u0, v0, p0 = kh_initial_condition(
        X, Y, Lx, Ly,
        u_top=u_top, u_bottom=u_bottom,
        rho_top=rho_top, rho_bottom=rho_bottom,
        p0=p0, gamma=GAMMA,
        shear_width=shear_width,
        pert_amplitude=pert_amplitude,
        pert_wavenumber=pert_wavenumber
    )
    U0 = prim_to_cons(rho0, u0, v0, p0, gamma=GAMMA)
    
    # Enforce periodic BCs on initial condition
    U0 = enforce_periodic_bc(U0)
    
    # Create BSPF operator
    bspf_op = bspf2d.from_grids(x=x, y=y, degree_x=degree, degree_y=degree,
                                correction="spectral", use_gpu=use_gpu)
    
    # Create RHS function
    def rhs_func(U_flat):
        """RHS function for time stepper."""
        U_reshaped = U_flat.reshape(4, ny, nx)
        # Enforce periodic BCs before computing RHS
        U_reshaped = enforce_periodic_bc(U_reshaped)
        dU_dt = euler_rhs_2d_bspf(U_reshaped, bspf_op, gamma=GAMMA,
                                  nu=nu, dx=dx, dy=dy)
        # Enforce periodic BCs on RHS
        dU_dt = enforce_periodic_bc(dU_dt)
        return dU_dt.flatten()
    
    # Time stepping
    U_flat = U0.flatten()
    t = 0.0
    dt = 0.0
    U_history = [U0.copy()]
    times = [0.0]
    U = U0.copy()  # Current solution state
    
    with TimeStepperState(U_flat, t_init=0.0, dt=dt, method=method,
                          t_final=t_end, show_progress=True) as state:
        step = 0
        while t < t_end:
            # Compute time step from CFL condition
            amax = max_wave_speed(U, gamma=GAMMA)
            dt = cfl * min(dx, dy) / amax
            if t + dt > t_end:
                dt = t_end - t
            
            # Update state dt
            state.dt = dt
            
            # Time step
            U_flat_next = time_step(state, dt, rhs_func, method=method)
            U_flat = state.get_current()
            t = state.get_current_time()
            
            # Reshape for storage
            U = U_flat.reshape(4, ny, nx)
            
            # Enforce periodic BCs after each step
            U = enforce_periodic_bc(U)
            U_flat = U.flatten()
            state.psi_now = U_flat.copy()
            
            # Store solution
            if step % save_interval == 0:
                U_history.append(U.copy())
                times.append(t)
            
            step += 1
            if step % 50 == 0:
                print(f"Step {step}, t = {t:.6f}, dt = {dt:.6e}, max wave speed = {amax:.6f}")
    
    return x, y, np.array(times), U_history

# ========== Visualization ==========
def plot_kh_solution(x, y, U_history, times, save_prefix='kh_instability'):
    """
    Create visualization of KH instability evolution.
    
    Parameters:
    -----------
    x, y : arrays
        Grid coordinates
    U_history : list
        Solution history
    times : array
        Time points
    save_prefix : str
        Prefix for saved files
    """
    X, Y = np.meshgrid(x, y)
    
    # Create figure with subplots for different times
    n_times = len(U_history)
    n_plot = min(6, n_times)  # Plot up to 6 snapshots
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i in range(n_plot):
        idx = int(i * (n_times - 1) / (n_plot - 1)) if n_plot > 1 else 0
        U = U_history[idx]
        rho, u, v, p = cons_to_prim(U)
        
        # Plot density (shows the mixing)
        im = axes[i].contourf(X, Y, rho, levels=30, cmap='viridis')
        axes[i].set_title(f'Density at t = {times[idx]:.3f}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        axes[i].set_aspect('equal')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_density_evolution.png', dpi=150)
    print(f"Saved density evolution to {save_prefix}_density_evolution.png")
    
    # Create velocity magnitude plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i in range(n_plot):
        idx = int(i * (n_times - 1) / (n_plot - 1)) if n_plot > 1 else 0
        U = U_history[idx]
        rho, u, v, p = cons_to_prim(U)
        vel_mag = np.sqrt(u**2 + v**2)
        
        im = axes[i].contourf(X, Y, vel_mag, levels=30, cmap='plasma')
        axes[i].set_title(f'Velocity magnitude at t = {times[idx]:.3f}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        axes[i].set_aspect('equal')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_velocity_evolution.png', dpi=150)
    print(f"Saved velocity evolution to {save_prefix}_velocity_evolution.png")
    
    # Create vorticity plot (shows the vortex structures)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i in range(n_plot):
        idx = int(i * (n_times - 1) / (n_plot - 1)) if n_plot > 1 else 0
        U = U_history[idx]
        rho, u, v, p = cons_to_prim(U)
        
        # Compute vorticity: ω = ∂v/∂x - ∂u/∂y
        # Use finite differences for simplicity
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        dv_dx = np.gradient(v, dx, axis=1)
        du_dy = np.gradient(u, dy, axis=0)
        vorticity = dv_dx - du_dy
        
        im = axes[i].contourf(X, Y, vorticity, levels=30, cmap='RdBu_r')
        axes[i].set_title(f'Vorticity at t = {times[idx]:.3f}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        axes[i].set_aspect('equal')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_vorticity_evolution.png', dpi=150)
    print(f"Saved vorticity evolution to {save_prefix}_vorticity_evolution.png")
    
    plt.show()

# ========== Main ==========
if __name__ == "__main__":
    # Parameters
    nx, ny = 256, 256
    Lx, Ly = 1.0, 1.0
    t_end = 0.1
    cfl = 0.1
    degree = 5
    nu = 0.0  # No artificial viscosity (can add small amount for stability)
    
    # KH instability parameters
    u_top = 0.5      # Top layer velocity
    u_bottom = -0.5  # Bottom layer velocity
    rho_top = 2.0    # Top layer density
    rho_bottom = 1.0 # Bottom layer density
    p0 = 1.0         # Initial pressure
    shear_width = 0.05  # Width of shear layer
    pert_amplitude = 0.01  # Initial perturbation amplitude
    pert_wavenumber = 2.0  # Wavenumber of perturbation
    
    print("=" * 60)
    print("2D Compressible Kelvin-Helmholtz Instability")
    print("=" * 60)
    print(f"Grid: {nx} x {ny}")
    print(f"Domain: [{0}, {Lx}] x [{0}, {Ly}]")
    print(f"B-spline degree: {degree}")
    print(f"CFL: {cfl}")
    print(f"Viscosity: nu = {nu:.2e}")
    print(f"Velocity: u_top = {u_top}, u_bottom = {u_bottom}")
    print(f"Density: rho_top = {rho_top}, rho_bottom = {rho_bottom}")
    print(f"Perturbation: amplitude = {pert_amplitude}, wavenumber = {pert_wavenumber}")
    print()
    
    # Solve
    x, y, times, U_history = solve_kh_instability_2d(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly, t_end=t_end, cfl=cfl, degree=degree,
        method='rk4', use_gpu=False, nu=nu,
        u_top=u_top, u_bottom=u_bottom,
        rho_top=rho_top, rho_bottom=rho_bottom,
        p0=p0, shear_width=shear_width,
        pert_amplitude=pert_amplitude, pert_wavenumber=pert_wavenumber,
        save_interval=10
    )
    
    print(f"\nSimulation complete!")
    print(f"Final time: {times[-1]:.6f}")
    print(f"Number of saved snapshots: {len(U_history)}")
    
    # Visualize
    plot_kh_solution(x, y, U_history, times, save_prefix='kh_instability_2d')

