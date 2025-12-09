"""
Sod Shock Tube Problem using BSPF + Spectral Filtering (DCT).

This script solves the 1D Sod shock tube problem using:
- B-spline pseudo-spectral differentiation (bspf1d) for spatial derivatives
- Outflow boundary conditions (zero-gradient extrapolation)
- DCT-based exponential filtering after each time step as an implicit
  artificial viscosity / spectral viscosity mechanism
- Comparison with MUSCL-HLL (finite volume) and TVD Lax-Wendroff (nodal finite difference)

Run from repository root:
    python examples/euler_eq/sod_bspf_filtered.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
from scipy.interpolate import interp1d

from sod_exact import sample_sod_exact
from bspf import bspf1d, TimeStepperState, time_step

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
    # Simple protection against negative values
    rho = np.maximum(rho, RHO_FLOOR)
    p   = np.maximum(p,   P_FLOOR)
    c = np.sqrt(gamma * p / rho)
    return np.max(np.abs(u) + c)

# ============================================================================
# Boundary Conditions
# ============================================================================
def create_reflective_wall_bc_enforcer():
    """
    Create a reflective wall BC enforcer for Euler equations.
    
    A reflective wall enforces u = 0 at boundaries (impermeable wall).
    This means:
    - Momentum (rho*u) = 0 at boundaries
    - Density and energy evolve naturally
    - The wall reflects waves back into the domain
    
    Returns
    -------
    enforce_bc : callable
        Function that enforces reflective wall BCs: enforce_bc(U) -> U_bc
    """
    def enforce_bc(U):
        """
        Enforce reflective wall BCs: u = 0 at boundaries.
        
        Parameters
        ----------
        U : array, shape (3, N)
            Conservative variables [rho, rho*u, E]
        
        Returns
        -------
        U_bc : array, shape (3, N)
            Conservative variables with reflective wall BCs applied
        """
        U_bc = U.copy()
        
        # Enforce u = 0 at boundaries by setting momentum to zero
        # Left boundary: u[0] = 0
        U_bc[1, 0] = 0.0  # momentum = rho*u = 0
        
        # Right boundary: u[-1] = 0
        U_bc[1, -1] = 0.0  # momentum = rho*u = 0
        
        # Note: Density and energy are left unchanged
        # The wall is impermeable but allows pressure/density to vary
        
        return U_bc
    
    return enforce_bc

# ============================================================================
# Spectral Filtering
# ============================================================================
def apply_global_dct_filter(U, alpha=36.0, p=8, gamma=GAMMA):
    """
    Apply DCT-based spectral filter globally to the entire solution.

    Parameters
    ----------
    U : array, shape (3, N)
        Conservative solution [rho, rho*u, E].
    alpha : float
        Filter strength parameter (larger = stronger damping of high modes).
    p : int
        Filter order (even integer; larger = sharper transition).
    gamma : float
        Adiabatic index
    
    Returns
    -------
    U_filtered : array, shape (3, N)
        Filtered solution
    
    We filter only interior points 1:-1 to avoid mangling boundary values.
    """
    nvars, N = U.shape
    if N <= 2:
        return U.copy()  # nothing meaningful to do

    i0 = 1           # first interior index
    i1 = N - 1       # last interior index (exclusive in slicing)
    M  = i1 - i0     # number of interior points

    if M <= 1:
        return U.copy()
    
    U_filtered = U.copy()
    
    # Create filter for interior points
    k = np.arange(M, dtype=float)
    if M > 1:
        eta = k / (M - 1)  # normalized wavenumber in [0,1]
    else:
        eta = np.zeros_like(k)
    
    # Exponential filter in spectral space
    sigma = np.exp(-alpha * eta**p)
    sigma[0] = 1.0  # preserve the mean
    
    # Apply filter to each conservative variable
    for m in range(nvars):
        v_interior = U[m, i0:i1].copy()
        
        # DCT-II / IDCT-II with orthonormal scaling
        c = dct(v_interior, type=2, norm='ortho')
        c_filtered = sigma * c
        v_filtered = idct(c_filtered, type=2, norm='ortho')
        
        # Update interior points only (boundaries remain unchanged)
        U_filtered[m, i0:i1] = v_filtered
    
    return U_filtered

# ============================================================================
# BSPF Solver
# ============================================================================
def euler_rhs_bspf(U, bspf_op, gamma=GAMMA, nu=0.0, dx=None, bc_type='outflow'):
    """
    RHS for Euler equations using BSPF with optional artificial viscosity.
    
    Zero-flux Neumann boundary conditions are explicitly enforced using
    bspf_op.enforced_zero_flux() for each conservative variable before differentiation.
    
    Parameters
    ----------
    U : array, shape (3, N)
        Conservative variables [rho, rho*u, E]
    bspf_op : bspf1d
        BSPF operator for spatial differentiation
    gamma : float
        Adiabatic index
    nu : float, optional
        Artificial viscosity coefficient C_visc (dimensionless).
        The local viscosity is:
            nu_local(x) = nu * dx * a_ref * S(x)
        where S(x) is a shock sensor in [0,1].
        If nu <= 0, no artificial viscosity is added.
    dx : float, optional
        Grid spacing (assumed roughly uniform). Needed for scaling nu_local.
    
    Returns
    -------
    dU_dt : array, shape (3, N)
        Time derivative of conservative variables
    """
    # For outflow BCs: explicitly enforce zero-flux boundary conditions
    if bc_type == 'outflow':
        U_corrected = U.copy()
        for i in range(3):  # For each conservative variable
            f_left, f_right = bspf_op.enforced_zero_flux(U[i, :])
            U_corrected[i, 0] = f_left
            U_corrected[i, -1] = f_right
        
        # Compute flux F(U) using corrected boundary values
        F = euler_flux(U_corrected, gamma=gamma)  # shape (3, N)
        
        # Also enforce zero-flux on flux itself before differentiation
        F_corrected = F.copy()
        for i in range(3):  # For each flux component
            f_left, f_right = bspf_op.enforced_zero_flux(F[i, :])
            F_corrected[i, 0] = f_left
            F_corrected[i, -1] = f_right
        
        # Compute flux derivative: dF/dx with zero-flux Neumann BCs
        dF_dx = np.zeros_like(F)
        for i in range(3):  # For each conservative variable
            dF_dx[i], _ = bspf_op.differentiate(F_corrected[i], k=1, neumann_bc=(0.0, 0.0))
    else:
        # For reflective wall: enforce u=0 at boundaries before computing flux
        # This mimics MUSCL's approach: ensure boundary conditions are satisfied
        # before computing fluxes
        U_reflective = U.copy()
        # Set momentum to zero at boundaries (u = 0)
        U_reflective[1, 0] = 0.0   # Left boundary: u = 0
        U_reflective[1, -1] = 0.0  # Right boundary: u = 0
        
        # Compute flux F(U) with u=0 at boundaries
        F = euler_flux(U_reflective, gamma=gamma)  # shape (3, N)
        
        # For reflective wall, at boundaries (u=0):
        # - Mass flux F[0] = rho*u = 0
        # - Momentum flux F[1] = rho*u^2 + p = p
        # - Energy flux F[2] = (E+p)*u = 0
        # These are automatically satisfied when u=0, but we ensure F[0] and F[2] are exactly 0
        F[0, 0] = 0.0   # Mass flux = 0 at left boundary
        F[0, -1] = 0.0  # Mass flux = 0 at right boundary
        F[2, 0] = 0.0   # Energy flux = 0 at left boundary
        F[2, -1] = 0.0  # Energy flux = 0 at right boundary
        
        # Compute flux derivative
        # For reflective wall, we use natural boundary conditions (no zero-flux)
        # The flux values at boundaries are already set correctly above
        dF_dx = np.zeros_like(F)
        for i in range(3):  # For each conservative variable
            dF_dx[i], _ = bspf_op.differentiate(F[i], k=1)
    
    # RHS from flux: -dF/dx
    dU_dt = -dF_dx
    
    # For reflective wall: ensure dU_dt respects boundary conditions
    # At boundaries, u=0 means momentum should not change (dU_dt[1] = 0)
    if bc_type == 'reflective_wall':
        dU_dt[1, 0] = 0.0   # Left boundary: momentum derivative = 0
        dU_dt[1, -1] = 0.0  # Right boundary: momentum derivative = 0
    
    # ---------- Artificial viscosity: ∂x( nu(x) ∂x U ) ----------
    # nu is now a *coefficient* (C_visc); nu(x) is computed via shock sensor.
    if nu > 0.0 and dx is not None:
        # Primitive variables for sensor and local wave speed
        rho, u, p = cons_to_prim(U, gamma=gamma)
        rho = np.maximum(rho, RHO_FLOOR)
        p   = np.maximum(p,   P_FLOOR)

        # Shock sensor based on |∂x rho|
        if bc_type == 'outflow':
            # Explicitly enforce zero-flux on rho before differentiation
            rho_corrected = rho.copy()
            f_left, f_right = bspf_op.enforced_zero_flux(rho)
            rho_corrected[0] = f_left
            rho_corrected[-1] = f_right
            rho_x, _ = bspf_op.differentiate(rho_corrected, k=1, neumann_bc=(0.0, 0.0))
        else:
            # For reflective wall: no special BC treatment
            rho_x, _ = bspf_op.differentiate(rho, k=1)
        sensor_raw = np.abs(rho_x)

        s_max = np.max(sensor_raw)
        if s_max > 0.0:
            # Normalize to [0,1], square to sharpen localization
            sensor = (sensor_raw / s_max)**2
        else:
            sensor = np.zeros_like(sensor_raw)

        # Zero sensor at boundaries (outflow, we don't want to smear them)
        sensor[0]  = 0.0
        sensor[-1] = 0.0

        # Reference wave speed a_ref
        c = np.sqrt(gamma * p / rho)
        a_loc = np.abs(u) + c
        a_ref = np.max(a_loc)

        # Local viscosity profile: nu_local has dimensions L^2 / T
        nu_local = nu * dx * a_ref * sensor   # shape (N,)

        # Add ∂x( nu(x) ∂x U ) for each conservative component
        for i in range(3):
            if bc_type == 'outflow':
                # Explicitly enforce zero-flux on U before differentiation
                U_i_corrected = U[i, :].copy()
                f_left, f_right = bspf_op.enforced_zero_flux(U_i_corrected)
                U_i_corrected[0] = f_left
                U_i_corrected[-1] = f_right
                U_x, _ = bspf_op.differentiate(U_i_corrected, k=1, neumann_bc=(0.0, 0.0))
            else:
                # For reflective wall: no special BC treatment
                U_x, _ = bspf_op.differentiate(U[i, :], k=1)
            
            visc_flux = nu_local * U_x
            
            if bc_type == 'outflow':
                # Enforce zero-flux on viscous flux before differentiation
                f_left, f_right = bspf_op.enforced_zero_flux(visc_flux)
                visc_flux[0] = f_left
                visc_flux[-1] = f_right
                visc_flux_x, _ = bspf_op.differentiate(visc_flux, k=1, neumann_bc=(0.0, 0.0))
            else:
                # For reflective wall: no special BC treatment
                visc_flux_x, _ = bspf_op.differentiate(visc_flux, k=1)
            
            dU_dt[i] += visc_flux_x
    
    return dU_dt

# ============================================================================
# Initial Conditions
# ============================================================================
def sod_initial(nx, xL=0.0, xR=1.0, x0=0.5, gamma=GAMMA, smooth_width=None):
    """
    Initialize Sod problem initial condition with a true step jump.
    
    Parameters
    ----------
    nx : int
        Number of grid points
    xL, xR : float
        Domain boundaries
    x0 : float
        Location of the initial discontinuity
    gamma : float
        Adiabatic index
    smooth_width : float, optional
        DEPRECATED: This parameter is ignored. The function now always uses
        a true step function (discontinuous jump).
    """
    x = np.linspace(xL, xR, nx, endpoint=True)
    
    # True step function (discontinuous jump)
    # Left state: rho=1.0, p=1.0 for x <= x0
    # Right state: rho=0.125, p=0.1 for x > x0
    rho = np.where(x <= x0, 1.0, 0.125)
    u   = np.zeros_like(x)
    p   = np.where(x <= x0, 1.0, 0.1)
    
    U = prim_to_cons(rho, u, p, gamma=gamma)
    return x, U

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

def apply_bc_muscl(U, ng=2, U_left=None, U_right=None):
    """
    Boundary conditions for MUSCL.
    If U_left and U_right are provided, uses Dirichlet BCs (matching BSPF).
    Otherwise, uses outflow (zero-gradient) BCs.
    """
    nvars, N_tot = U.shape
    N = N_tot - 2*ng
    
    if U_left is not None and U_right is not None:
        # Dirichlet BCs: fix ghost cells to boundary values
        U[:, :ng] = U_left[:, None]
        U[:, ng+N:] = U_right[:, None]
    else:
        # Outflow (zero-gradient) BCs: copy interior to ghost cells
        U[:, :ng] = U[:, ng][:, None]
        U[:, ng+N:] = U[:, ng+N-1][:, None]

def muscl_rhs(U, dx, gamma=GAMMA, ng=2, U_left=None, U_right=None):
    """Compute RHS dU/dt for MUSCL–HLL finite volume scheme."""
    nvars, N_tot = U.shape
    N = N_tot - 2*ng
    
    apply_bc_muscl(U, ng=ng, U_left=U_left, U_right=U_right)
    
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

def rk2_step_muscl(U, dt, dx, gamma=GAMMA, ng=2, U_left=None, U_right=None):
    """One TVD RK2 step for MUSCL–HLL scheme."""
    nvars, N_tot = U.shape
    N = N_tot - 2*ng
    
    k1 = muscl_rhs(U.copy(), dx, gamma=gamma, ng=ng, U_left=U_left, U_right=U_right)
    U1 = U.copy()
    U1[:, ng:ng+N] += dt * k1[:, ng:ng+N]
    
    k2 = muscl_rhs(U1.copy(), dx, gamma=gamma, ng=ng, U_left=U_left, U_right=U_right)
    U_new = U.copy()
    U_new[:, ng:ng+N] = 0.5*(U[:, ng:ng+N] + U1[:, ng:ng+N] + dt * k2[:, ng:ng+N])
    
    return U_new

def solve_sod_muscl(nx=400, t_end=0.2, cfl=0.4, gamma=GAMMA, use_dirichlet_bc=False):
    """
    Solve Sod problem using 2nd–order MUSCL–HLL finite volume scheme.
    
    Parameters
    ----------
    use_dirichlet_bc : bool
        If True, use Dirichlet BCs (matching BSPF). If False, use outflow BCs.
    
    Returns
    -------
    x_centers : array, shape (N,)
        Cell centers
    U_int : array, shape (3, N)
        Final conservative solution in physical cells
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
    
    # Store boundary values for Dirichlet BCs if requested
    U_left = None
    U_right = None
    if use_dirichlet_bc:
        U_left = U0[:, 0].copy()
        U_right = U0[:, -1].copy()
    
    apply_bc_muscl(U, ng=ng, U_left=U_left, U_right=U_right)
    
    t = 0.0
    step = 0
    
    print("MUSCL–HLL Sod solver")
    print(f"  nx   = {nx}")
    print(f"  dx   = {dx:.4e}")
    print(f"  CFL  = {cfl}")
    print(f"  t_end= {t_end}")
    if use_dirichlet_bc:
        print(f"  BC:  Dirichlet")
    else:
        print(f"  BC:  Outflow (zero-gradient, matching BSPF)")
    
    while t < t_end:
        amax = max_wave_speed(U[:, ng:ng+N], gamma=gamma)
        dt = cfl * dx / amax
        if t + dt > t_end:
            dt = t_end - t
        
        U = rk2_step_muscl(U, dt, dx, gamma=gamma, ng=ng, U_left=U_left, U_right=U_right)
        t += dt
        step += 1
        
        if step % 20 == 0:
            print(f"step {step:5d}, t = {t:.6f}, dt = {dt:.3e}, amax = {amax:.3f}")
    
    print(f"Finished at t = {t:.6f}, steps = {step}")
    
    U_int = U[:, ng:ng+N]
    return x_centers, U_int

# ============================================================================
# TVD Lax-Wendroff Solver (Nodal Finite Difference)
# ============================================================================
def solve_sod_tvd_lw(nx=1000, t_end=0.2, cfl=0.8, gamma=GAMMA, smooth_width=None):
    """
    Solve Sod problem using TVD Lax-Wendroff finite difference scheme.
    
    This is a nodal finite difference method (like BSPF), making it a fair comparison.
    Uses a flux limiter to prevent oscillations near shocks.
    
    Parameters:
    -----------
    nx : int
        Number of grid points
    t_end : float
        Final time
    cfl : float
        CFL number for time stepping
    gamma : float
        Adiabatic index
    smooth_width : float, optional
        Width of smoothing transition for initial condition.
        If None, uses true step function (like original TVD code).
    
    Returns:
    --------
    x : array
        Grid points
    U : array, shape (3, nx)
        Final conservative solution
    """
    # Use nodal grid (like BSPF)
    x = np.linspace(0.0, 1.0, nx, endpoint=True)
    dx = x[1] - x[0]
    
    # Initial condition
    if smooth_width is None:
        # True step function (like original TVD code)
        mid_x = 0.5
        rho = np.where(x <= mid_x, 1.0, 0.125)
        u = np.zeros_like(x)
        p = np.where(x <= mid_x, 1.0, 0.1)
    else:
        # Smoothed initial condition (matching BSPF)
        x0 = 0.5
        transition = 0.5 * (1.0 - np.tanh((x - x0) / smooth_width))
        rho = 1.0 * transition + 0.125 * (1.0 - transition)
        u = np.zeros_like(x)
        p = 1.0 * transition + 0.1 * (1.0 - transition)
    
    U = prim_to_cons(rho, u, p, gamma=gamma)  # shape (3, nx)
    
    # For outflow BCs, we don't need to store boundary values
    # (boundaries will be extrapolated from interior)
    
    # Reshape to (nx, 3) for TVD code compatibility
    U_nd = U.T.copy()  # shape (nx, 3)
    
    t = 0.0
    step = 0
    
    print("TVD Lax-Wendroff Sod solver (nodal finite difference)")
    print(f"  nx   = {nx}")
    print(f"  dx   = {dx:.4e}")
    print(f"  CFL  = {cfl}")
    print(f"  t_end= {t_end}")
    print(f"  BC:  Outflow (zero-gradient, matching BSPF)")
    
    while t < t_end:
        # Compute time step from CFL condition
        rho_prim, u_prim, p_prim = cons_to_prim(U, gamma=gamma)
        rho_prim = np.maximum(rho_prim, RHO_FLOOR)
        p_prim = np.maximum(p_prim, P_FLOOR)
        c = np.sqrt(gamma * p_prim / rho_prim)
        a = np.max(np.abs(u_prim) + c)
        dt = cfl * dx / a
        if t + dt > t_end:
            dt = t_end - t
        
        # TVD Lax-Wendroff step
        U_nd = tvd_lw_step(U_nd, dx, dt, gamma=gamma)
        
        # Convert back to (3, nx) format
        U = U_nd.T.copy()
        
        t += dt
        step += 1
        
        if step % 20 == 0:
            print(f"step {step:5d}, t = {t:.6f}, dt = {dt:.3e}, amax = {a:.3f}")
    
    print(f"Finished at t = {t:.6f}, steps = {step}")
    
    return x, U

def tvd_lw_step(U, dx, dt, gamma=GAMMA):
    """
    TVD Lax-Wendroff step with outflow BCs.
    
    U : array, shape (Nx, 3) - conservative variables
    Returns updated U in same shape.
    """
    Nx = U.shape[0]
    
    # Outflow BCs: zero-gradient (extrapolation)
    # Applied before and after the step
    
    # Physical flux
    F = flux_tvd(U, gamma=gamma)  # shape (Nx, 3)
    
    # Primitive variables for wave speed
    rho, u, p = cons_to_prim_tvd(U, gamma=gamma)
    rho = np.maximum(rho, RHO_FLOOR)
    p = np.maximum(p, P_FLOOR)
    c = np.sqrt(gamma * p / rho)
    a = np.abs(u) + c  # Maximum characteristic speed
    
    # First-order Rusanov flux F^(1)
    a_half = np.maximum(a[:-1], a[1:])  # (Nx-1,)
    FL = F[:-1]  # (Nx-1, 3)
    FR = F[1:]   # (Nx-1, 3)
    UL = U[:-1]  # (Nx-1, 3)
    UR = U[1:]   # (Nx-1, 3)
    
    F1 = 0.5 * (FL + FR) - 0.5 * a_half[:, None] * (UR - UL)  # (Nx-1, 3)
    
    # Lax-Wendroff high-order flux F^LW
    # Predictor: U*_{i+1/2} = 0.5(U_i + U_{i+1}) - 0.5*(dt/dx)*(F_{i+1} - F_i)
    U_star = 0.5 * (UL + UR) - 0.5 * (dt / dx) * (FR - FL)
    U_star = enforce_physical_tvd(U_star, gamma=gamma)
    F_LW = flux_tvd(U_star, gamma=gamma)  # (Nx-1, 3)
    
    # Flux limiter coefficient psi
    rho_full, _, _ = cons_to_prim_tvd(U, gamma=gamma)
    dr = rho_full[1:] - rho_full[:-1]  # Δρ_{i+1/2}, (Nx-1,)
    
    eps = 1e-12
    psi = np.zeros(Nx - 1)
    
    # Only use limiter at interior interfaces 1..Nx-2
    j = np.arange(1, Nx - 2)  # corresponds to interfaces 1..Nx-3
    
    a_half_mid = a_half[j]
    
    # Upwind direction positive: look at left slope ratio
    pos = a_half_mid >= 0
    # Upwind direction negative: look at right slope ratio
    neg = ~pos
    
    r = np.zeros_like(a_half_mid)
    
    # r = Δρ_{i-1/2} / (Δρ_{i+1/2} + eps) (upwind on left)
    r[pos] = dr[j[pos] - 1] / (dr[j[pos]] + eps)
    
    # r = Δρ_{i+3/2} / (Δρ_{i+1/2} + eps) (upwind on right)
    r[neg] = dr[j[neg] + 1] / (dr[j[neg]] + eps)
    
    # minmod limiter: phi(r) = max(0, min(1, r))
    phi = np.maximum(0.0, np.minimum(1.0, r))
    psi[j] = phi
    
    # Final TVD flux
    F_hat = F1 + psi[:, None] * (F_LW - F1)
    
    # Update U
    U_new = U.copy()
    # Interior points i=1..Nx-2
    U_new[1:-1] = U[1:-1] - (dt / dx) * (F_hat[1:] - F_hat[:-1])
    
    # Outflow BCs: extrapolate from interior
    U_new[0] = U_new[1]
    U_new[-1] = U_new[-2]
    
    U_new = enforce_physical_tvd(U_new, gamma=gamma)
    return U_new

def flux_tvd(U, gamma=GAMMA):
    """Physical flux for TVD code (U shape: (Nx, 3))."""
    rho, u, p = cons_to_prim_tvd(U, gamma=gamma)
    m = rho * u
    F = np.zeros_like(U)
    F[:, 0] = m
    F[:, 1] = m * u + p
    F[:, 2] = u * (U[:, 2] + p)
    return F

def cons_to_prim_tvd(U, gamma=GAMMA):
    """Convert conservative to primitive (U shape: (Nx, 3))."""
    rho = np.maximum(U[:, 0], RHO_FLOOR)
    m = U[:, 1]
    E = U[:, 2]
    u = m / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * u**2)
    p = np.maximum(p, P_FLOOR)
    return rho, u, p

def enforce_physical_tvd(U, gamma=GAMMA):
    """Enforce physical bounds (U shape: (Nx, 3))."""
    rho, u, p = cons_to_prim_tvd(U, gamma=gamma)
    umax = 20.0
    u = np.clip(u, -umax, umax)
    rho = np.maximum(rho, RHO_FLOOR)
    p = np.maximum(p, P_FLOOR)
    return prim_to_cons_tvd(rho, u, p, gamma=gamma)

def prim_to_cons_tvd(rho, u, p, gamma=GAMMA):
    """Convert primitive to conservative (all shape: (Nx,))."""
    E = p / (gamma - 1.0) + 0.5 * rho * u**2
    m = rho * u
    return np.stack([rho, m, E], axis=-1)  # shape (Nx, 3)

# ============================================================================
# BSPF Main Solver
# ============================================================================
def solve_sod_bspf(nx=1000, t_end=0.1, cfl=0.01, degree=5, 
                   method='rk4', use_gpu=False, smooth_width=None,
                   filter_alpha=36.0, filter_order=8, nu=0.0,
                   bc_type='outflow'):
    """
    Solve Sod problem using BSPF + DCT spectral filtering.
    
    Parameters
    ----------
    nx : int
        Number of grid points
    t_end : float
        Final time
    cfl : float
        CFL number for time stepping
    degree : int
        B-spline degree for BSPF
    method : str
        Time stepping method ('rk4', 'rk45', 'rk23')
    use_gpu : bool
        Whether to use GPU acceleration (unused here)
    smooth_width : float, optional
        DEPRECATED: This parameter is ignored. The function now always uses
        a true step function (discontinuous jump).
    filter_alpha : float
        Exponential filter strength parameter.
    filter_order : int
        Exponential filter order (even integer).
    nu : float
        Artificial viscosity coefficient C_visc (dimensionless).
        Local viscosity is nu_local(x) = nu * dx * a_ref * sensor(x).
        If nu <= 0, no artificial viscosity is added.
    bc_type : str, optional
        Boundary condition type: 'outflow' (default) or 'reflective_wall'.
        - 'outflow': Zero-flux Neumann BCs (waves exit domain)
        - 'reflective_wall': u = 0 at boundaries (waves reflect)
    
    Returns
    -------
    x : array
        Grid points
    t : array
        Time points
    U_history : list
        Solution history
    """
    # Initialize with smoothed initial condition
    x, U = sod_initial(nx, gamma=GAMMA, smooth_width=smooth_width)
    dx = x[1] - x[0]
    
    # Create BSPF operator
    bspf_op = bspf1d.from_grid(degree, x, n_basis=3*degree, num_boundary_points=degree, use_clustering=True, clustering_factor=2.0)
    
    # Create boundary condition enforcer
    if bc_type == 'reflective_wall':
        enforce_bc = create_reflective_wall_bc_enforcer()
    else:
        enforce_bc = None  # Outflow uses zero-flux Neumann in differentiation
    
    # Create RHS function (with artificial viscosity)
    def rhs_func(U_flat):
        """RHS function for time stepper (expects flattened array)"""
        U_reshaped = U_flat.reshape(3, nx)
        dU_dt = euler_rhs_bspf(U_reshaped, bspf_op, gamma=GAMMA, nu=nu, dx=dx, bc_type=bc_type)
        return dU_dt.flatten()
    
    # Time stepping
    U_flat = U.flatten()
    t = 0.0
    dt = 0.0
    U_history = [U.copy()]
    times = [0.0]
    
    with TimeStepperState(U_flat, t_init=0.0, dt=dt, method=method,
                          t_final=t_end, show_progress=True) as state:
        step = 0
        while t < t_end:
            # Compute time step from CFL condition
            amax = max_wave_speed(U, gamma=GAMMA)
            dt = cfl * dx / amax
            if t + dt > t_end:
                dt = t_end - t
            
            # Update state dt
            state.dt = dt
            
            # Time step
            _ = time_step(state, dt, rhs_func, method=method)
            U_flat = state.get_current()
            t = state.get_current_time()
            
            # Reshape for storage and filtering
            U = U_flat.reshape(3, nx)
            
            # Explicitly enforce zero-flux boundary conditions using enforced_zero_flux
            # (for outflow BCs; reflective wall uses different BC enforcer)
            if bc_type == 'outflow':
                for i in range(3):  # For each conservative variable
                    f_left, f_right = bspf_op.enforced_zero_flux(U[i, :])
                    U[i, 0] = f_left
                    U[i, -1] = f_right
            
            # Apply boundary conditions (if reflective wall)
            # For reflective wall: enforce u=0 at boundaries
            if bc_type == 'reflective_wall':
                U[1, 0] = 0.0   # Left boundary: u = 0
                U[1, -1] = 0.0  # Right boundary: u = 0
            
            # Apply global DCT-based spectral filter
            U = apply_global_dct_filter(U, alpha=filter_alpha, p=filter_order, gamma=GAMMA)
            
            # Re-apply zero-flux boundary conditions after filtering (for outflow)
            if bc_type == 'outflow':
                for i in range(3):  # For each conservative variable
                    f_left, f_right = bspf_op.enforced_zero_flux(U[i, :])
                    U[i, 0] = f_left
                    U[i, -1] = f_right
            
            # Re-apply reflective wall boundary conditions after filtering
            if bc_type == 'reflective_wall':
                U[1, 0] = 0.0   # Left boundary: u = 0
                U[1, -1] = 0.0  # Right boundary: u = 0
            
            # Update state with filtered solution
            U_flat = U.flatten()
            state.psi_now = U_flat.copy()
            
            # Store solution
            U_history.append(U.copy())
            times.append(t)
            
            step += 1
            if step % 10 == 0:
                print(f"Step {step}, t = {t:.6f}, dt = {dt:.6e}, max wave speed = {amax:.6f}")
    
    return x, np.array(times), U_history

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    # Parameters
    nx = 1000
    t_end = 1.0
    cfl = 0.5  # CFL for stability
    degree = 5
    
    print("=" * 60)
    print("Sod Problem: BSPF with DCT Spectral Filtering")
    print("=" * 60)
    print(f"Grid points: {nx}")
    print(f"B-spline degree: {degree}")
    print(f"CFL: {cfl}")
    # Boundary condition type: 'outflow' or 'reflective_wall'
    bc_type = 'reflective_wall'  # Use 'reflective_wall' to see wave reflections
    if bc_type == 'reflective_wall':
        print(f"Boundary conditions: Reflective Wall (u=0 at boundaries)")
    else:
        print(f"Boundary conditions: Outflow (zero-flux Neumann via BSPF)")
    dx = 1.0 / nx
    print(f"Initial condition: true step jump (discontinuous)")
    
    # Filter parameters
    filter_alpha = 36.0   # try 20–60
    filter_order = 8      # even, e.g. 8, 10, 16
    print(f"Global spectral filter: alpha = {filter_alpha}, order = {filter_order}")
    print("Filtering is applied globally to the entire domain.")
    
    # Artificial viscosity parameter (dimensionless)
    C_visc = 0.5  # Artificial viscosity coefficient
    print(f"Artificial viscosity coefficient C_visc = {C_visc:.3f}")
    print("Local nu(x) = C_visc * dx * a_ref * sensor(x)\n")
    
    # Solve
    x, times, U_history = solve_sod_bspf(
        nx=nx, t_end=t_end, cfl=cfl, degree=degree,
        method='rk23', smooth_width=None, use_gpu=False,
        filter_alpha=filter_alpha, filter_order=filter_order,
        nu=C_visc, bc_type=bc_type
    )
    
    # Get final solution
    U_final = U_history[-1]
    rho, u, p = cons_to_prim(U_final)
    
    # Exact solution
    rho_exact, u_exact, p_exact = sample_sod_exact(x, t_end, x0=0.5, gamma=GAMMA)
    
    # Also solve with MUSCL-HLL scheme for comparison (finite volume)
    print("\n" + "=" * 60)
    print("Solving with MUSCL-HLL scheme for comparison...")
    print("=" * 60)
    x_muscl, U_muscl = solve_sod_muscl(nx=nx, t_end=t_end, cfl=0.5, gamma=GAMMA, use_dirichlet_bc=False)
    rho_muscl, u_muscl, p_muscl = cons_to_prim(U_muscl, gamma=GAMMA)
    
    # Exact solution on MUSCL grid
    rho_exact_muscl, u_exact_muscl, p_exact_muscl = sample_sod_exact(x_muscl, t_end, x0=0.5, gamma=GAMMA)
    
    # Also solve with TVD Lax-Wendroff scheme for comparison (nodal FD, like BSPF)
    print("\n" + "=" * 60)
    print("Solving with TVD Lax-Wendroff scheme for comparison...")
    print("=" * 60)
    x_tvd, U_tvd = solve_sod_tvd_lw(nx=nx, t_end=t_end, cfl=0.8, gamma=GAMMA, smooth_width=None)
    rho_tvd, u_tvd, p_tvd = cons_to_prim(U_tvd, gamma=GAMMA)
    
    # Exact solution on TVD grid
    rho_exact_tvd, u_exact_tvd, p_exact_tvd = sample_sod_exact(x_tvd, t_end, x0=0.5, gamma=GAMMA)
    
    # Interpolate MUSCL results to BSPF grid for comparison
    rho_muscl_interp = interp1d(x_muscl, rho_muscl, kind='linear',
                                bounds_error=False, fill_value='extrapolate')(x)
    u_muscl_interp = interp1d(x_muscl, u_muscl, kind='linear',
                              bounds_error=False, fill_value='extrapolate')(x)
    p_muscl_interp = interp1d(x_muscl, p_muscl, kind='linear',
                               bounds_error=False, fill_value='extrapolate')(x)
    
    # Compute errors
    error_rho = rho - rho_exact
    error_u = u - u_exact
    error_p = p - p_exact
    
    # Plot: 3 rows x 2 columns (solution on left, error on right)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex='col')
    
    # Compute MUSCL errors (interpolated to BSPF grid)
    error_rho_muscl = rho_muscl_interp - rho_exact
    error_u_muscl = u_muscl_interp - u_exact
    error_p_muscl = p_muscl_interp - p_exact
    
    # Compute TVD errors (using exact solution on TVD grid)
    error_rho_tvd = rho_tvd - rho_exact_tvd
    error_u_tvd = u_tvd - u_exact_tvd
    error_p_tvd = p_tvd - p_exact_tvd
    
    # Density - Solution
    axes[0, 0].plot(x, rho, 'b-', lw=2, label='BSPF + Filter + Visc', alpha=0.8)
    axes[0, 0].plot(x, rho_muscl_interp, 'm-', lw=2, label='MUSCL-HLL (FV)', alpha=0.8)
    axes[0, 0].plot(x_tvd, rho_tvd, 'g-', lw=2, label='TVD Lax-Wendroff (FD)', alpha=0.8)
    axes[0, 0].plot(x, rho_exact, 'r--', lw=2, label='Exact', alpha=0.8)
    axes[0, 0].set_ylabel('Density ρ')
    axes[0, 0].set_title(f'Sod Problem at t = {t_end:.3f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Density - Error
    axes[1, 0].semilogy(x, np.abs(error_rho), 'b-', lw=2, label='BSPF error', alpha=0.8)
    axes[1, 0].semilogy(x, np.abs(error_rho_muscl), 'm-', lw=2, label='MUSCL error', alpha=0.8)
    axes[1, 0].semilogy(x_tvd, np.abs(error_rho_tvd), 'g-', lw=2, label='TVD LW error', alpha=0.8)
    axes[1, 0].axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 0].set_ylabel('Error in Density')
    axes[1, 0].set_title('Density Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Velocity - Solution
    axes[0, 1].plot(x, u, 'b-', lw=2, label='BSPF + Filter + Visc', alpha=0.8)
    axes[0, 1].plot(x, u_muscl_interp, 'm-', lw=2, label='MUSCL-HLL (FV)', alpha=0.8)
    axes[0, 1].plot(x_tvd, u_tvd, 'g-', lw=2, label='TVD Lax-Wendroff (FD)', alpha=0.8)
    axes[0, 1].plot(x, u_exact, 'r--', lw=2, label='Exact', alpha=0.8)
    axes[0, 1].set_ylabel('Velocity u')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Velocity - Error
    axes[1, 1].semilogy(x, np.abs(error_u), 'b-', lw=2, label='BSPF error', alpha=0.8)
    axes[1, 1].semilogy(x, np.abs(error_u_muscl), 'm-', lw=2, label='MUSCL error', alpha=0.8)
    axes[1, 1].semilogy(x_tvd, np.abs(error_u_tvd), 'g-', lw=2, label='TVD LW error', alpha=0.8)
    axes[1, 1].axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 1].set_ylabel('Error in Velocity')
    axes[1, 1].set_title('Velocity Error')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Pressure - Solution
    axes[0, 2].plot(x, p, 'b-', lw=2, label='BSPF + Filter + Visc', alpha=0.8)
    axes[0, 2].plot(x, p_muscl_interp, 'm-', lw=2, label='MUSCL-HLL (FV)', alpha=0.8)
    axes[0, 2].plot(x_tvd, p_tvd, 'g-', lw=2, label='TVD Lax-Wendroff (FD)', alpha=0.8)
    axes[0, 2].plot(x, p_exact, 'r--', lw=2, label='Exact', alpha=0.8)
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('Pressure p')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Pressure - Error
    axes[1, 2].semilogy(x, np.abs(error_p), 'b-', lw=2, label='BSPF error', alpha=0.8)
    axes[1, 2].semilogy(x, np.abs(error_p_muscl), 'm-', lw=2, label='MUSCL error', alpha=0.8)
    axes[1, 2].semilogy(x_tvd, np.abs(error_p_tvd), 'g-', lw=2, label='TVD LW error', alpha=0.8)
    axes[1, 2].axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('Error in Pressure')
    axes[1, 2].set_title('Pressure Error')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sod_bspf_filtered.png', dpi=150)
    print(f"\nSolution saved to sod_bspf_filtered.png")
    
    # ========================================================================
    # Create Space-Time Plots
    # ========================================================================
    print("\n" + "=" * 60)
    print("Creating space-time plots...")
    print("=" * 60)
    
    # Use all stored time steps (no sampling)
    n_history = len(U_history)
    print(f"Total time steps stored: {n_history}")
    
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
    fig_st, axes_st = plt.subplots(1, 3, figsize=(18, 6))
    fig_st.suptitle('Sod Problem: BSPF + Filter + Visc - Space-Time Plots', 
                    fontsize=16, fontweight='bold')
    
    # Density space-time plot
    ax_rho = axes_st[0]
    im_rho = ax_rho.contourf(X, T, rho_space_time, levels=50, cmap='viridis')
    ax_rho.set_xlabel('x (spatial position)', fontsize=12)
    ax_rho.set_ylabel('t (time)', fontsize=12)
    ax_rho.set_title('Density ρ(x,t)', fontsize=14, fontweight='bold')
    ax_rho.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(im_rho, ax=ax_rho, label='ρ')
    
    # Velocity space-time plot
    ax_u = axes_st[1]
    im_u = ax_u.contourf(X, T, u_space_time, levels=50, cmap='RdBu_r')
    ax_u.set_xlabel('x (spatial position)', fontsize=12)
    ax_u.set_ylabel('t (time)', fontsize=12)
    ax_u.set_title('Velocity u(x,t)', fontsize=14, fontweight='bold')
    ax_u.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(im_u, ax=ax_u, label='u')
    
    # Pressure space-time plot
    ax_p = axes_st[2]
    im_p = ax_p.contourf(X, T, p_space_time, levels=50, cmap='plasma')
    ax_p.set_xlabel('x (spatial position)', fontsize=12)
    ax_p.set_ylabel('t (time)', fontsize=12)
    ax_p.set_title('Pressure p(x,t)', fontsize=14, fontweight='bold')
    ax_p.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(im_p, ax=ax_p, label='p')
    
    plt.tight_layout()
    plt.savefig('sod_bspf_filtered_spacetime.png', dpi=150, bbox_inches='tight')
    print(f"Space-time plots saved to sod_bspf_filtered_spacetime.png")
    
    # Compute errors
    l2_error_rho = np.sqrt(np.trapz((rho - rho_exact)**2, x))
    l2_error_u   = np.sqrt(np.trapz((u   - u_exact)**2,   x))
    l2_error_p   = np.sqrt(np.trapz((p   - p_exact)**2,   x))
    
    # Compute MUSCL errors (on BSPF grid after interpolation)
    l2_error_rho_muscl = np.sqrt(np.trapz((rho_muscl_interp - rho_exact)**2, x))
    l2_error_u_muscl   = np.sqrt(np.trapz((u_muscl_interp   - u_exact)**2,   x))
    l2_error_p_muscl   = np.sqrt(np.trapz((p_muscl_interp   - p_exact)**2,   x))
    
    # Compute TVD errors
    l2_error_rho_tvd = np.sqrt(np.trapz((rho_tvd - rho_exact_tvd)**2, x_tvd))
    l2_error_u_tvd   = np.sqrt(np.trapz((u_tvd   - u_exact_tvd)**2,   x_tvd))
    l2_error_p_tvd   = np.sqrt(np.trapz((p_tvd   - p_exact_tvd)**2,   x_tvd))
    
    print(f"\nL2 Errors - BSPF + Filter + Visc:")
    print(f"  Density:  {l2_error_rho:.6e}")
    print(f"  Velocity: {l2_error_u:.6e}")
    print(f"  Pressure: {l2_error_p:.6e}")
    
    print(f"\nL2 Errors - MUSCL-HLL (finite volume):")
    print(f"  Density:  {l2_error_rho_muscl:.6e}")
    print(f"  Velocity: {l2_error_u_muscl:.6e}")
    print(f"  Pressure: {l2_error_p_muscl:.6e}")
    
    print(f"\nL2 Errors - TVD Lax-Wendroff (nodal FD):")
    print(f"  Density:  {l2_error_rho_tvd:.6e}")
    print(f"  Velocity: {l2_error_u_tvd:.6e}")
    print(f"  Pressure: {l2_error_p_tvd:.6e}")
    
    # Show static plots and animation
    plt.show()  # This will display both the static plots and the animation
