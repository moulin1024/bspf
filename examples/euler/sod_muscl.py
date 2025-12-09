"""
Sod Shock Tube Problem using MUSCL-HLL with Reflective Walls - Animation.

This script solves the 1D Sod shock tube problem in a confined chamber using:
- MUSCL-HLL finite volume scheme (2nd-order)
- Reflective wall boundary conditions (u = 0 at boundaries)
- Animation of solution evolution over time

Run from repository root:
    python examples/euler/sod_muscl_animation.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Constants
# ============================================================================

GAMMA = 1.4
RHO_FLOOR = 1e-12
P_FLOOR = 1e-12

# ============================================================================
# Basic Euler Equations Utilities
# ============================================================================

def prim_to_cons(rho, u, p, gamma=GAMMA):
    E = p / (gamma - 1.0) + 0.5 * rho * u**2
    return np.stack([rho, rho * u, E], axis=0)  # shape (3, N)

def cons_to_prim(U, gamma=GAMMA):
    rho = U[0]
    mom = U[1]
    E   = U[2]
    u = mom / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * u**2)
    return rho, u, p

def euler_flux(U, gamma=GAMMA):
    rho, u, p = cons_to_prim(U, gamma=gamma)
    F = np.zeros_like(U)
    F[0] = rho * u
    F[1] = rho * u**2 + p
    F[2] = (U[2] + p) * u  # (E + p) u
    return F

def max_wave_speed(U, gamma=GAMMA):
    rho, u, p = cons_to_prim(U, gamma=gamma)
    rho = np.maximum(rho, RHO_FLOOR)
    p   = np.maximum(p,   P_FLOOR)
    c = np.sqrt(gamma * p / rho)
    return np.max(np.abs(u) + c)

# ============================================================================
# MUSCL-HLL Solver (Finite Volume)
# ============================================================================

def hll_flux(UL, UR, gamma=GAMMA):
    """
    HLL approximate Riemann solver for 1D Euler (vectorized).
    
    UL, UR : arrays of shape (3,) or (3, N_interfaces)
        Left and right conservative states.
    
    Returns
    -------
    FHLL : array, shape (3,) or (3, N_interfaces)
        Numerical flux at the interface(s).
    """
    UL = np.asarray(UL)
    UR = np.asarray(UR)
    
    if UL.ndim == 1:
        UL = UL[:, None]
        UR = UR[:, None]
        squeeze_output = True
    else:
        squeeze_output = False
    
    n_interfaces = UL.shape[1]
    
    rhoL, uL, pL = cons_to_prim(UL, gamma=gamma)
    rhoR, uR, pR = cons_to_prim(UR, gamma=gamma)
    
    rhoL = np.maximum(rhoL, RHO_FLOOR)
    rhoR = np.maximum(rhoR, RHO_FLOOR)
    pL   = np.maximum(pL,   P_FLOOR)
    pR   = np.maximum(pR,   P_FLOOR)
    
    cL = np.sqrt(gamma * pL / rhoL)
    cR = np.sqrt(gamma * pR / rhoR)
    
    FL = euler_flux(UL, gamma=gamma)
    FR = euler_flux(UR, gamma=gamma)
    
    SL = np.minimum(uL - cL, uR - cR)
    SR = np.maximum(uL + cL, uR + cR)
    
    FHLL = np.zeros_like(UL)
    
    mask1 = SL >= 0.0
    FHLL[:, mask1] = FL[:, mask1]
    
    mask2 = SR <= 0.0
    FHLL[:, mask2] = FR[:, mask2]
    
    mask3 = ~(mask1 | mask2)
    if np.any(mask3):
        denom = SR[mask3] - SL[mask3]
        FHLL[:, mask3] = (
            SR[mask3] * FL[:, mask3] - SL[mask3] * FR[:, mask3] +
            SL[mask3] * SR[mask3] * (UR[:, mask3] - UL[:, mask3])
        ) / denom
    
    if squeeze_output:
        return FHLL[:, 0]
    return FHLL

def minmod_vec(a, b):
    """Vectorised minmod for arrays a, b."""
    res = np.zeros_like(a)
    mask_same = (a * b) > 0.0
    res[mask_same] = np.where(
        np.abs(a[mask_same]) < np.abs(b[mask_same]),
        a[mask_same],
        b[mask_same],
    )
    return res

def apply_bc_muscl_reflective(U, ng=2):
    """
    Apply reflective wall boundary conditions for MUSCL.
    
    Reflective walls: u = 0 at boundaries.
    For ghost cells, we mirror the interior state but flip the velocity sign.
    This ensures that when we compute the flux at the boundary, the velocity
    component is zero (reflective condition).
    """
    nvars, N_tot = U.shape
    N = N_tot - 2*ng
    
    # Left boundary: reflective wall
    # Mirror interior state but flip velocity (u -> -u)
    # Ghost cell i mirrors interior cell (ng + i - 1)
    for i in range(ng):
        # Copy density and energy from interior (mirror)
        U[0, ng - 1 - i] = U[0, ng + i]  # rho
        U[2, ng - 1 - i] = U[2, ng + i]  # E
        # Flip momentum (velocity sign) for reflection
        U[1, ng - 1 - i] = -U[1, ng + i]  # rho*u -> -rho*u
    
    # Right boundary: reflective wall
    # Mirror interior state but flip velocity (u -> -u)
    # Ghost cell (ng + N + i) mirrors interior cell (ng + N - 1 - i)
    for i in range(ng):
        # Copy density and energy from interior (mirror)
        U[0, ng + N + i] = U[0, ng + N - 1 - i]  # rho
        U[2, ng + N + i] = U[2, ng + N - 1 - i]  # E
        # Flip momentum (velocity sign) for reflection
        U[1, ng + N + i] = -U[1, ng + N - 1 - i]  # rho*u -> -rho*u

def muscl_rhs(U, dx, gamma=GAMMA, ng=2):
    """Compute RHS dU/dt for MUSCL–HLL finite volume scheme with reflective walls."""
    nvars, N_tot = U.shape
    N = N_tot - 2*ng
    
    # Apply reflective wall boundary conditions
    apply_bc_muscl_reflective(U, ng=ng)
    
    slopes = np.zeros_like(U)
    dU_left  = (U[:, 1:N_tot-1] - U[:, 0:N_tot-2]) / dx
    dU_right = (U[:, 2:N_tot]   - U[:, 1:N_tot-1]) / dx
    slopes[:, 1:N_tot-1] = minmod_vec(dU_left, dU_right)
    
    UL_face = U[:, :N_tot-1] + 0.5*dx*slopes[:, :N_tot-1]
    UR_face = U[:, 1:N_tot] - 0.5*dx*slopes[:, 1:N_tot]
    
    F_face = hll_flux(UL_face, UR_face, gamma=gamma)
    
    dUdt = np.zeros_like(U)
    dUdt[:, ng:ng+N] = -(F_face[:, ng:ng+N] - F_face[:, ng-1:ng+N-1]) / dx
    
    return dUdt

def rk2_step_muscl(U, dt, dx, gamma=GAMMA, ng=2):
    """One TVD RK2 step for MUSCL–HLL scheme with reflective walls."""
    nvars, N_tot = U.shape
    N = N_tot - 2*ng
    
    k1 = muscl_rhs(U.copy(), dx, gamma=gamma, ng=ng)
    U1 = U.copy()
    U1[:, ng:ng+N] += dt * k1[:, ng:ng+N]
    
    k2 = muscl_rhs(U1.copy(), dx, gamma=gamma, ng=ng)
    U_new = U.copy()
    U_new[:, ng:ng+N] = 0.5*(U[:, ng:ng+N] + U1[:, ng:ng+N] + dt * k2[:, ng:ng+N])
    
    return U_new

def solve_sod_muscl_reflective(nx=400, t_end=0.2, cfl=0.4, gamma=GAMMA):
    """
    Solve Sod problem using 2nd–order MUSCL–HLL with reflective wall BCs.
    
    Parameters
    ----------
    nx : int
        Number of grid points
    t_end : float
        Final time
    cfl : float
        CFL number
    gamma : float
        Adiabatic index
    
    Returns
    -------
    x_centers : array, shape (N,)
        Cell centers
    U_history : list
        Solution history at each time step
    times : list
        Time points
    """
    # Use cell-centered grid for finite volume method
    x_centers = np.linspace(0.0 + 0.5*(1.0-0.0)/nx,
                           1.0 - 0.5*(1.0-0.0)/nx,
                           nx)
    rho = np.where(x_centers < 0.5, 1.0, 0.125)
    u   = np.zeros_like(x_centers)
    p   = np.where(x_centers < 0.5, 1.0, 0.1)
    U0 = prim_to_cons(rho, u, p, gamma=gamma)
    
    dx = x_centers[1] - x_centers[0]
    
    ng = 2
    N = nx
    N_tot = N + 2*ng
    
    U = np.zeros((3, N_tot))
    U[:, ng:ng+N] = U0
    
    # Apply initial reflective wall BCs
    apply_bc_muscl_reflective(U, ng=ng)
    
    t = 0.0
    step = 0
    
    # Store solution history - save every time step
    U_history = [U[:, ng:ng+N].copy()]
    times = [0.0]
    
    print("MUSCL–HLL Sod solver with Reflective Walls")
    print(f"  nx   = {nx}")
    print(f"  dx   = {dx:.4e}")
    print(f"  CFL  = {cfl}")
    print(f"  t_end= {t_end}")
    print(f"  BC:  Reflective Wall (u=0 at boundaries)")
    print(f"  Note: Saving every time step (no sampling)")
    
    while t < t_end:
        amax = max_wave_speed(U[:, ng:ng+N], gamma=gamma)
        dt = cfl * dx / amax
        if t + dt > t_end:
            dt = t_end - t
        
        U = rk2_step_muscl(U, dt, dx, gamma=gamma, ng=ng)
        
        # Enforce reflective wall BCs after time step
        # Set momentum to zero at boundaries (u = 0)
        U[1, ng] = 0.0      # Left boundary: u = 0
        U[1, ng+N-1] = 0.0  # Right boundary: u = 0
        
        t += dt
        step += 1
        
        # Store every time step (no sampling)
        U_history.append(U[:, ng:ng+N].copy())
        times.append(t)
        
        if step % 20 == 0:
            print(f"step {step:5d}, t = {t:.6f}, dt = {dt:.3e}, amax = {amax:.3f}, frames = {len(U_history)}")
    
    print(f"Finished at t = {t:.6f}, steps = {step}")
    
    return x_centers, U_history, times

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Parameters
    nx = 1000
    t_end = 1.0
    cfl = 0.5
    
    print("=" * 60)
    print("Sod Problem: MUSCL-HLL with Reflective Walls - Space-Time Plots")
    print("=" * 60)
    print(f"Grid points: {nx}")
    print(f"CFL: {cfl}")
    print(f"Final time: {t_end}")
    print(f"Boundary conditions: Reflective Wall (u=0 at boundaries)\n")
    
    # Solve
    x, U_history, times = solve_sod_muscl_reflective(nx=nx, t_end=t_end, cfl=cfl, gamma=GAMMA)
    
    # Convert to primitive variables for all frames
    n_frames_total = len(U_history)
    print(f"\nTotal frames stored: {n_frames_total}")
    
    # Prepare data for space-time plots
    rho_data = []
    u_data = []
    p_data = []
    
    for U_frame in U_history:
        rho_frame, u_frame, p_frame = cons_to_prim(U_frame, gamma=GAMMA)
        rho_data.append(rho_frame)
        u_data.append(u_frame)
        p_data.append(p_frame)
    
    # Convert to 2D arrays for space-time plots
    # Shape: (n_time, n_space)
    rho_space_time = np.array(rho_data)
    u_space_time = np.array(u_data)
    p_space_time = np.array(p_data)
    
    # Time array
    times_array = np.array(times)
    
    # Create space-time meshgrid
    X, T = np.meshgrid(x, times_array)
    
    # Create figure for space-time plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Sod Problem: MUSCL-HLL with Reflective Walls - Space-Time Plots', 
                 fontsize=16, fontweight='bold')
    
    # Density space-time plot
    ax_rho = axes[0]
    im_rho = ax_rho.contourf(X, T, rho_space_time, levels=50, cmap='viridis')
    ax_rho.set_xlabel('x (spatial position)', fontsize=12)
    ax_rho.set_ylabel('t (time)', fontsize=12)
    ax_rho.set_title('Density ρ(x,t)', fontsize=14, fontweight='bold')
    ax_rho.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(im_rho, ax=ax_rho, label='ρ')
    
    # Velocity space-time plot
    ax_u = axes[1]
    im_u = ax_u.contourf(X, T, u_space_time, levels=50, cmap='RdBu_r')
    ax_u.set_xlabel('x (spatial position)', fontsize=12)
    ax_u.set_ylabel('t (time)', fontsize=12)
    ax_u.set_title('Velocity u(x,t)', fontsize=14, fontweight='bold')
    ax_u.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(im_u, ax=ax_u, label='u')
    
    # Pressure space-time plot
    ax_p = axes[2]
    im_p = ax_p.contourf(X, T, p_space_time, levels=50, cmap='plasma')
    ax_p.set_xlabel('x (spatial position)', fontsize=12)
    ax_p.set_ylabel('t (time)', fontsize=12)
    ax_p.set_title('Pressure p(x,t)', fontsize=14, fontweight='bold')
    ax_p.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(im_p, ax=ax_p, label='p')
    
    plt.tight_layout()
    plt.savefig('sod_muscl_reflective_spacetime.png', dpi=150, bbox_inches='tight')
    print(f"\nSpace-time plots saved to sod_muscl_reflective_spacetime.png")
    plt.show()

