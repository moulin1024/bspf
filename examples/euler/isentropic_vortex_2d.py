"""
2D Isentropic Vortex test case for Euler equations using BSPF.

The isentropic vortex is a smooth, exact solution that advects with the mean flow.
This is an excellent test case for high-order methods since it has no shocks.
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

# ========== Isentropic vortex exact solution ==========
def isentropic_vortex_exact(x, y, t, x0=0.0, y0=0.0, beta=5.0, 
                            u_inf=1.0, v_inf=0.0, rho_inf=1.0, p_inf=1.0, gamma=GAMMA):
    """
    Exact solution for 2D isentropic vortex.
    
    The vortex is centered at (x0, y0) at t=0 and advects with mean flow (u_inf, v_inf).
    
    Parameters:
    -----------
    x, y : arrays
        Grid coordinates (can be 2D meshgrids)
    t : float
        Time
    x0, y0 : float
        Initial vortex center position
    beta : float
        Vortex strength parameter
    u_inf, v_inf : float
        Mean flow velocity
    rho_inf, p_inf : float
        Far-field density and pressure
    gamma : float
        Adiabatic index
    
    Returns:
    --------
    rho, u, v, p : arrays
        Primitive variables at time t
    """
    # Advected vortex center
    xc = x0 + u_inf * t
    yc = y0 + v_inf * t
    
    # Distance from vortex center
    r_sq = (x - xc)**2 + (y - yc)**2
    
    # Vortex perturbation (decays with distance)
    f = 1.0 - (beta**2 / (8.0 * np.pi**2)) * np.exp(1.0 - r_sq)
    
    # Temperature (from isentropic relation)
    T = f
    
    # Density (from isentropic relation: p/rho^gamma = const)
    rho = rho_inf * (T / (p_inf / rho_inf))**(1.0 / (gamma - 1.0))
    
    # Pressure (from ideal gas law and isentropic relation)
    p = p_inf * (T / (p_inf / rho_inf))**(gamma / (gamma - 1.0))
    
    # Velocity components (vortex + mean flow)
    # Vortex velocity: tangential, decays with distance
    r = np.sqrt(r_sq + 1e-12)  # Avoid division by zero
    v_theta = (beta / (2.0 * np.pi)) * np.exp(0.5 * (1.0 - r_sq))
    
    # Convert to Cartesian
    u = u_inf - v_theta * (y - yc) / r
    v = v_inf + v_theta * (x - xc) / r
    
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
        Artificial viscosity coefficient (optional)
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
    
    # Add artificial viscosity if requested
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

# ========== Main solver ==========
def solve_isentropic_vortex_2d(nx=128, ny=128, Lx=10.0, Ly=10.0, 
                               t_end=1.0, cfl=0.5, degree=5,
                               method='rk4', use_gpu=False, nu=0.0,
                               x0=5.0, y0=5.0, beta=5.0,
                               u_inf=1.0, v_inf=0.0):
    """
    Solve 2D isentropic vortex problem using BSPF.
    
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
        Artificial viscosity coefficient
    x0, y0 : float
        Initial vortex center
    beta : float
        Vortex strength
    u_inf, v_inf : float
        Mean flow velocity
    
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
    
    # Initial condition (exact solution at t=0)
    rho0, u0, v0, p0 = isentropic_vortex_exact(
        X, Y, 0.0, x0=x0, y0=y0, beta=beta,
        u_inf=u_inf, v_inf=v_inf
    )
    U0 = prim_to_cons(rho0, u0, v0, p0, gamma=GAMMA)
    
    # Create BSPF operator
    bspf_op = bspf2d.from_grids(x=x, y=y, degree_x=degree, degree_y=degree,
                                correction="spectral", use_gpu=use_gpu)
    
    # Create RHS function
    def rhs_func(U_flat):
        """RHS function for time stepper."""
        U_reshaped = U_flat.reshape(4, ny, nx)
        dU_dt = euler_rhs_2d_bspf(U_reshaped, bspf_op, gamma=GAMMA, 
                                  nu=nu, dx=dx, dy=dy)
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
            
            # Store solution
            U_history.append(U.copy())
            times.append(t)
            
            step += 1
            if step % 10 == 0:
                print(f"Step {step}, t = {t:.6f}, dt = {dt:.6e}, max wave speed = {amax:.6f}")
    
    return x, y, np.array(times), U_history

# ========== Main ==========
if __name__ == "__main__":
    # Parameters
    nx, ny = 128, 128
    Lx, Ly = 10.0, 10.0
    t_end = 0.05
    cfl = 0.5
    degree = 5
    nu = 0.0  # No viscosity for smooth solution
    
    # Vortex parameters
    x0, y0 = 5.0, 5.0  # Initial vortex center
    beta = 5.0  # Vortex strength
    u_inf, v_inf = 1.0, 0.0  # Mean flow
    
    print("=" * 60)
    print("2D Isentropic Vortex: BSPF")
    print("=" * 60)
    print(f"Grid: {nx} x {ny}")
    print(f"Domain: [{0}, {Lx}] x [{0}, {Ly}]")
    print(f"B-spline degree: {degree}")
    print(f"CFL: {cfl}")
    print(f"Viscosity: nu = {nu:.2e}")
    print(f"Vortex: center=({x0}, {y0}), beta={beta}")
    print(f"Mean flow: u={u_inf}, v={v_inf}")
    print()
    
    # Solve
    x, y, times, U_history = solve_isentropic_vortex_2d(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly, t_end=t_end, cfl=cfl, degree=degree,
        method='rk4', nu=nu, x0=x0, y0=y0, beta=beta, u_inf=u_inf, v_inf=v_inf
    )
    
    # Get final solution
    U_final = U_history[-1]
    rho, u, v, p = cons_to_prim(U_final)
    
    # Exact solution at final time
    X, Y = np.meshgrid(x, y)
    rho_exact, u_exact, v_exact, p_exact = isentropic_vortex_exact(
        X, Y, t_end, x0=x0, y0=y0, beta=beta, u_inf=u_inf, v_inf=v_inf
    )
    
    # Compute errors
    lmax_error_rho = np.max(np.abs(rho - rho_exact))
    lmax_error_u = np.max(np.abs(u - u_exact))
    lmax_error_v = np.max(np.abs(v - v_exact))
    lmax_error_p = np.max(np.abs(p - p_exact))
    
    print(f"\nL_infinity Errors:")
    print(f"  Density:  {lmax_error_rho:.6e}")
    print(f"  u-velocity: {lmax_error_u:.6e}")
    print(f"  v-velocity: {lmax_error_v:.6e}")
    print(f"  Pressure: {lmax_error_p:.6e}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Density
    im0 = axes[0, 0].contourf(X, Y, rho, levels=20, cmap='viridis')
    axes[0, 0].set_title(f'Density (numerical) at t = {t_end:.3f}')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].contourf(X, Y, rho_exact, levels=20, cmap='viridis')
    axes[0, 1].set_title('Density (exact)')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Pressure
    im2 = axes[1, 0].contourf(X, Y, p, levels=20, cmap='plasma')
    axes[1, 0].set_title('Pressure (numerical)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1, 0])
    
    im3 = axes[1, 1].contourf(X, Y, p_exact, levels=20, cmap='plasma')
    axes[1, 1].set_title('Pressure (exact)')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('isentropic_vortex_2d.png', dpi=150)
    print(f"\nSolution saved to isentropic_vortex_2d.png")
    
    plt.show()

