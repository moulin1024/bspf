from __future__ import annotations

import numpy as np
import numpy.typing as npt
import sys
import os

# Import time steppers and Schrödinger solver from bspf library
from bspf1d import bspf1d 
from time_steppers import TimeStepperState, time_step

Array = npt.NDArray[np.complex128]


def enforce_zero_flux_neumann_bc(psi: Array, dx: float, order: int = 2) -> Array:
    """
    Enforce zero-flux Neumann boundary conditions explicitly using finite difference.
    
    For zero flux: ∂ψ/∂x = 0 at boundaries
    Uses one-sided finite difference stencils of specified order.
    
    This function is useful when you need to explicitly enforce Neumann BCs
    on a solution array, for example after time stepping or when using
    methods that don't automatically enforce BCs.
    
    Parameters:
    -----------
    psi : array (complex or float)
        Solution array to enforce BCs on. Can be 1D array of any numeric type.
    dx : float
        Grid spacing
    order : int
        Order of accuracy for finite difference approximation (1, 2, 3, 4, or 5)
        Should match or be close to the BSPF degree for consistency.
        Default is 2.
        
    Returns:
    --------
    psi_bc : array
        Solution array with enforced Neumann BCs (same type as input)
    
    Notes:
    -----
    The function uses one-sided finite difference stencils to approximate
    ∂ψ/∂x = 0 at boundaries. Higher order stencils require more interior
    points, so the function automatically falls back to lower-order stencils
    if there aren't enough grid points.
    
    Examples:
    --------
    >>> import numpy as np
    >>> from bspf import enforce_zero_flux_neumann_bc
    >>> x = np.linspace(0, 1, 101)
    >>> dx = x[1] - x[0]
    >>> psi = np.sin(np.pi * x)
    >>> psi_bc = enforce_zero_flux_neumann_bc(psi, dx, order=2)
    """
    psi_bc = psi.copy()
    n = len(psi)
    
    if order == 1:
        # 1st order: ψ[0] = ψ[1], ψ[-1] = ψ[-2]
        psi_bc[0] = psi_bc[1]
        psi_bc[-1] = psi_bc[-2]
        
    elif order == 2:
        # 2nd order forward difference at left boundary:
        # ∂ψ/∂x ≈ (-3*ψ[0] + 4*ψ[1] - ψ[2])/(2*dx) = 0
        # => ψ[0] = (4*ψ[1] - ψ[2])/3
        if n >= 3:
            psi_bc[0] = (4.0 * psi_bc[1] - psi_bc[2]) / 3.0
        else:
            psi_bc[0] = psi_bc[1]
        
        # 2nd order backward difference at right boundary:
        # ∂ψ/∂x ≈ (3*ψ[-1] - 4*ψ[-2] + ψ[-3])/(2*dx) = 0
        # => ψ[-1] = (4*ψ[-2] - ψ[-3])/3
        if n >= 3:
            psi_bc[-1] = (4.0 * psi_bc[-2] - psi_bc[-3]) / 3.0
        else:
            psi_bc[-1] = psi_bc[-2]
            
    elif order == 3:
        # 3rd order forward difference at left boundary:
        # ∂ψ/∂x ≈ (-11*ψ[0] + 18*ψ[1] - 9*ψ[2] + 2*ψ[3])/(6*dx) = 0
        # => ψ[0] = (18*ψ[1] - 9*ψ[2] + 2*ψ[3])/11
        if n >= 4:
            psi_bc[0] = (18.0 * psi_bc[1] - 9.0 * psi_bc[2] + 2.0 * psi_bc[3]) / 11.0
        elif n >= 3:
            psi_bc[0] = (4.0 * psi_bc[1] - psi_bc[2]) / 3.0
        else:
            psi_bc[0] = psi_bc[1]
        
        # 3rd order backward difference at right boundary:
        # ∂ψ/∂x ≈ (11*ψ[-1] - 18*ψ[-2] + 9*ψ[-3] - 2*ψ[-4])/(6*dx) = 0
        # => ψ[-1] = (18*ψ[-2] - 9*ψ[-3] + 2*ψ[-4])/11
        if n >= 4:
            psi_bc[-1] = (18.0 * psi_bc[-2] - 9.0 * psi_bc[-3] + 2.0 * psi_bc[-4]) / 11.0
        elif n >= 3:
            psi_bc[-1] = (4.0 * psi_bc[-2] - psi_bc[-3]) / 3.0
        else:
            psi_bc[-1] = psi_bc[-2]
            
    elif order == 4:
        # 4th order forward difference at left boundary:
        # ∂ψ/∂x ≈ (-25*ψ[0] + 48*ψ[1] - 36*ψ[2] + 16*ψ[3] - 3*ψ[4])/(12*dx) = 0
        # => ψ[0] = (48*ψ[1] - 36*ψ[2] + 16*ψ[3] - 3*ψ[4])/25
        if n >= 5:
            psi_bc[0] = (48.0 * psi_bc[1] - 36.0 * psi_bc[2] + 16.0 * psi_bc[3] - 3.0 * psi_bc[4]) / 25.0
        elif n >= 4:
            psi_bc[0] = (18.0 * psi_bc[1] - 9.0 * psi_bc[2] + 2.0 * psi_bc[3]) / 11.0
        elif n >= 3:
            psi_bc[0] = (4.0 * psi_bc[1] - psi_bc[2]) / 3.0
        else:
            psi_bc[0] = psi_bc[1]
        
        # 4th order backward difference at right boundary:
        # ∂ψ/∂x ≈ (25*ψ[-1] - 48*ψ[-2] + 36*ψ[-3] - 16*ψ[-4] + 3*ψ[-5])/(12*dx) = 0
        # => ψ[-1] = (48*ψ[-2] - 36*ψ[-3] + 16*ψ[-4] - 3*ψ[-5])/25
        if n >= 5:
            psi_bc[-1] = (48.0 * psi_bc[-2] - 36.0 * psi_bc[-3] + 16.0 * psi_bc[-4] - 3.0 * psi_bc[-5]) / 25.0
        elif n >= 4:
            psi_bc[-1] = (18.0 * psi_bc[-2] - 9.0 * psi_bc[-3] + 2.0 * psi_bc[-4]) / 11.0
        elif n >= 3:
            psi_bc[-1] = (4.0 * psi_bc[-2] - psi_bc[-3]) / 3.0
        else:
            psi_bc[-1] = psi_bc[-2]
            
    elif order == 5:
        # 5th order forward difference at left boundary:
        # ∂ψ/∂x ≈ (-137*ψ[0] + 300*ψ[1] - 300*ψ[2] + 200*ψ[3] - 75*ψ[4] + 12*ψ[5])/(60*dx) = 0
        # => ψ[0] = (300*ψ[1] - 300*ψ[2] + 200*ψ[3] - 75*ψ[4] + 12*ψ[5])/137
        if n >= 6:
            psi_bc[0] = (300.0 * psi_bc[1] - 300.0 * psi_bc[2] + 200.0 * psi_bc[3] - 75.0 * psi_bc[4] + 12.0 * psi_bc[5]) / 137.0
        elif n >= 5:
            psi_bc[0] = (48.0 * psi_bc[1] - 36.0 * psi_bc[2] + 16.0 * psi_bc[3] - 3.0 * psi_bc[4]) / 25.0
        elif n >= 4:
            psi_bc[0] = (18.0 * psi_bc[1] - 9.0 * psi_bc[2] + 2.0 * psi_bc[3]) / 11.0
        elif n >= 3:
            psi_bc[0] = (4.0 * psi_bc[1] - psi_bc[2]) / 3.0
        else:
            psi_bc[0] = psi_bc[1]
        
        # 5th order backward difference at right boundary:
        # ∂ψ/∂x ≈ (137*ψ[-1] - 300*ψ[-2] + 300*ψ[-3] - 200*ψ[-4] + 75*ψ[-5] - 12*ψ[-6])/(60*dx) = 0
        # => ψ[-1] = (300*ψ[-2] - 300*ψ[-3] + 200*ψ[-4] - 75*ψ[-5] + 12*ψ[-6])/137
        if n >= 6:
            psi_bc[-1] = (300.0 * psi_bc[-2] - 300.0 * psi_bc[-3] + 200.0 * psi_bc[-4] - 75.0 * psi_bc[-5] + 12.0 * psi_bc[-6]) / 137.0
        elif n >= 5:
            psi_bc[-1] = (48.0 * psi_bc[-2] - 36.0 * psi_bc[-3] + 16.0 * psi_bc[-4] - 3.0 * psi_bc[-5]) / 25.0
        elif n >= 4:
            psi_bc[-1] = (18.0 * psi_bc[-2] - 9.0 * psi_bc[-3] + 2.0 * psi_bc[-4]) / 11.0
        elif n >= 3:
            psi_bc[-1] = (4.0 * psi_bc[-2] - psi_bc[-3]) / 3.0
        else:
            psi_bc[-1] = psi_bc[-2]
    else:
        # Default to 2nd order if order not in [1,2,3,4,5]
        if n >= 3:
            psi_bc[0] = (4.0 * psi_bc[1] - psi_bc[2]) / 3.0
            psi_bc[-1] = (4.0 * psi_bc[-2] - psi_bc[-3]) / 3.0
        else:
            psi_bc[0] = psi_bc[1]
            psi_bc[-1] = psi_bc[-2]
    
    return psi_bc

def create_schrodinger_rhs(bspf_op: bspf1d, V: np.ndarray, g: float = 0.0,
                           enforce_bc: Optional[Callable] = None,
                           neumann_bc: Optional[tuple] = None) -> Callable:
    """
    Create RHS function for Schrödinger equation.
    
    The equation in non-dimensional form (ℏ = m = 1):
        i*∂ψ/∂t = -(1/2)*∂²ψ/∂x² + V(x)*ψ + g*|ψ|²*ψ
    
    Parameters:
    -----------
    bspf_op : bspf1d
        BSPF operator for spatial differentiation
    V : array
        Potential V(x) (dimensionless, can be NumPy or CuPy array)
    g : float, optional
        Nonlinearity parameter. g=0 for linear case, g≠0 for nonlinear case.
        Default: 0.0 (linear Schrödinger equation)
    enforce_bc : callable, optional
        Function to enforce boundary conditions: enforce_bc(psi) -> psi_bc
        Used for Dirichlet BCs. If None, no BC enforcement is applied.
        Default: None
    neumann_bc : tuple, optional
        Neumann boundary conditions: (left_flux, right_flux) where
        left_flux = ∂ψ/∂x at left boundary, right_flux = ∂ψ/∂x at right boundary.
        If None, no Neumann BCs are applied.
        Default: None
    
    Returns:
    --------
    rhs_func : callable
        RHS function with signature: rhs_func(psi) -> dpsi_dt
    """
    # Detect backend from V array (GPU-aware)

    xp = np
    
    def schrodinger_rhs(psi: np.ndarray) -> np.ndarray:
        """
        RHS for Schrödinger equation.
        GPU-aware: automatically works with CuPy arrays if V is a GPU array.
        
        Parameters:
        -----------
        psi : complex array
            Complex wavefunction (dimensionless, NumPy or CuPy array)
        
        Returns:
        --------
        dpsi_dt : complex array
            Time derivative of wavefunction (dimensionless)
        """
        # Enforce Dirichlet boundary conditions if provided
        if enforce_bc is not None:
            psi_bc = enforce_bc(psi)
        else:
            psi_bc = psi.copy()
        
        # Compute second derivative using bspf (handles complex arrays and GPU)
        # Pass neumann_bc to differentiate() to enforce Neumann BCs during differentiation
        if neumann_bc is not None:
            d2psi_dx2, _ = bspf_op.differentiate(psi_bc, k=2, neumann_bc=neumann_bc)
        else:
            d2psi_dx2, _ = bspf_op.differentiate(psi_bc, k=2)
        
        # Enforce Dirichlet BCs on derivative if BC function provided
        if enforce_bc is not None:
            d2psi_dx2 = enforce_bc(d2psi_dx2)
        
        # Linear terms: kinetic energy + potential
        # In non-dimensional units (ℏ = m = 1):
        # ψ_t = (i/2)*ψ_xx - i*V(x)*ψ
        linear_term = 0.5j * d2psi_dx2 - 1j * V * psi_bc
        
        # Nonlinear term: g*|ψ|²*ψ (only if g ≠ 0)
        if g != 0.0:
            # In non-dimensional units: ψ_t += -i*g*|ψ|²*ψ
            # Use xp.abs for GPU-aware absolute value
            nonlinear_term = -1j * g * xp.abs(psi_bc)**2 * psi_bc
            dpsi_dt = linear_term + nonlinear_term
        else:
            dpsi_dt = linear_term
        
        # Enforce Dirichlet boundary conditions on RHS if provided
        # For Dirichlet BCs, dψ/dt = 0 at boundaries (since ψ is fixed)
        if enforce_bc is not None:
            dpsi_dt[0] = 0.0   # Left boundary: dψ/dt = 0
            dpsi_dt[-1] = 0.0  # Right boundary: dψ/dt = 0
        
        return dpsi_dt
    
    return schrodinger_rhs

# ============================================================
# Analytical Solutions: Breathers
# ============================================================
def peregrine_exact(x, t):
    """Peregrine breather (rational solution) for i ψ_t + 1/2 ψ_xx + |ψ|^2 ψ = 0."""
    x = np.asarray(x, dtype=float)
    denom = 1.0 + 4.0 * x**2 + 4.0 * t**2
    return (1.0 - 4.0 * (1.0 + 2.0j * t) / denom) * np.exp(1.0j * t)


def sfb_unified(x, t, a):
    """
    Unified soliton-on-finite-background family (SFB)
    from which Akhmediev, Kuznetsov–Ma, and Peregrine
    arise, for NLS:
        i ψ_t + 1/2 ψ_xx + |ψ|^2 ψ = 0.

    General form (Yang et al., etc.):

        ψ(t,x) = [ 1 +
            ( 2(1-2a) cosh(b t) + i b sinh(b t) )
            / ( sqrt(2a) cos(ω x) - cosh(b t) )
        ] e^{i t},

    where
        ω = 2 sqrt(1 - 2a),
        b = sqrt(8 a (1 - 2a)).

    For 0 < a < 1/2: Akhmediev breather (AB).
    For a -> 1/2: Peregrine (rational) limit.
    For 1/2 < a < 1: Kuznetsov–Ma (KM) breather (via
    analytic continuation: ω,b purely imaginary).

    Parameters
    ----------
    x : array_like
    t : float
    a : float in (0,1), a != 0.5

    Returns
    -------
    ψ(x,t;a) : complex ndarray
    """
    if not (0.0 < a < 1.0):
        raise ValueError("a must be in (0,1) for SFB family.")

    x = np.asarray(x, dtype=np.complex128)
    t_c = complex(t)
    a_c = complex(a)

    # Complex-safe sqrt for both AB (a<0.5) and KM (a>0.5)
    b = np.sqrt(8.0 * a_c * (1.0 - 2.0 * a_c))
    omega = 2.0 * np.sqrt(1.0 - 2.0 * a_c)

    num = 2.0 * (1.0 - 2.0 * a_c) * np.cosh(b * t_c) + 1.0j * b * np.sinh(b * t_c)
    den = np.sqrt(2.0 * a_c) * np.cos(omega * x) - np.cosh(b * t_c)

    psi = (1.0 + num / den) * np.exp(1.0j * t_c)
    return psi


def kuznetsov_ma_exact(x, t, a=0.75):
    """
    Kuznetsov–Ma breather:
      - localized in x, periodic in t
      - parameter range: 0.5 < a < 1
    """
    if not (0.5 < a < 1.0):
        raise ValueError("For Kuznetsov–Ma breather, use 0.5 < a < 1.")
    return sfb_unified(x, t, a=a)

# ============================================================
# Control Parameters
# ============================================================
# Domain parameters
L_domain = 100  # domain length
nx = 1024            # grid points (including endpoints)

# Time parameters
T = 2             # final time
dt = 0.001             # time step

# BSPF parameters
degree = 7            # B-spline degree

# Breather type selection
breather_type = 'kuznetsov_ma'  # Options: 'peregrine' or 'kuznetsov_ma'
a_km = 0.75  # Parameter for Kuznetsov–Ma breather (0.5 < a < 1)

# Initial condition: Breather
# Note: Our time starts from 0, but breathers use time starting from -1
# So we use breather_exact(x, t - 1) where t is our simulation time

# Boundary condition parameters
neumann_bc = (0.0, 0.0)  # (left_flux, right_flux) = (0, 0) for zero flux

# Potential (for linear Schrödinger, V = 0)
V_value = 0.0         # Potential value (constant potential)

# Nonlinearity parameters
g_focusing = -1.0     # Focusing NLS (g < 0)
g_defocusing = 1.0    # Defocusing NLS (g > 0)

# ============================================================
# Setup: Grid, Operator, Initial Condition
# ============================================================
# Create spatial grid
# Shift domain to center at 0 for peregrine_exact (which is centered at x=0)
x = np.linspace(-L_domain/2.0, L_domain/2.0, nx)
dx = x[1] - x[0]     # grid spacing

# Print parameters
print("="*60)
print("Control Parameters:")
print("="*60)
print(f"Domain: L = {L_domain:.3f}, nx = {nx}, dx = {dx:.6f}")
print(f"Time: T = {T:.3f}, dt = {dt:.6f}, dx² = {dx**2:.6f}")
print(f"BSPF: degree = {degree}")
print(f"Initial condition: {breather_type.upper()} breather")
if breather_type == 'kuznetsov_ma':
    print(f"  Parameter a = {a_km}")
print(f"  Domain centered at x=0: x ∈ [{x[0]:.3f}, {x[-1]:.3f}]")
print(f"  Simulation time starts at t=0, but breather uses t starting from -1")
print(f"Boundary conditions: Neumann (zero flux)")
print(f"Potential: V = {V_value:.3f}")
print(f"Focusing case: g = {g_focusing:.3f} (g < 0)")
print(f"Defocusing case: g = {g_defocusing:.3f} (g > 0)")
print("="*60)

# Number of time steps
nt = int(T / dt) + 1

# Create BSPF operator
bf = bspf1d.from_grid(degree=degree, x=x, use_clustering=True, clustering_factor=2.0)

# Initial condition: Breather
# Our simulation time starts at t=0, but breathers use time starting from -1
# So at t=0, we use breather_exact(x, -1)
if breather_type == 'peregrine':
    psi0 = peregrine_exact(x, -1.0).astype(np.complex128)
elif breather_type == 'kuznetsov_ma':
    psi0 = kuznetsov_ma_exact(x, -1.0, a=a_km).astype(np.complex128)
else:
    raise ValueError(f"Unknown breather type: {breather_type}")

# Enforce zero-flux Neumann BCs on initial condition
# Use order matching the BSPF degree (or closest available: 1, 2, 3, 4, or 5)
bc_order = min(degree, 5)  # Cap at 5th order (highest implemented)
print(f"  Using {bc_order}-th order finite difference for Neumann BC enforcement")
psi0 = enforce_zero_flux_neumann_bc(psi0, dx, order=bc_order)

# Create potential array
V = np.full(nx, V_value, dtype=np.float64)

# ============================================================
# Focusing Case (g < 0)
# ============================================================
print("\n" + "="*60)
print("Running FOCUSING case (g < 0)...")
print("="*60)

# Create RHS function for focusing case
rhs_func_focusing = create_schrodinger_rhs(bf, V, g=g_focusing, neumann_bc=neumann_bc)
print("  Created RHS function (focusing case: g < 0, Neumann BCs: zero flux)")

# Time integration
with TimeStepperState(psi0.copy(), t_init=0.0, dt=dt, method='rk45', t_final=T, show_progress=True) as state_focusing:
    Psi_focusing = np.empty((nt, nx), dtype=np.complex128)
    Psi_focusing[0] = psi0.copy()
    times_focusing = np.zeros(nt)
    times_focusing[0] = 0.0

    for step in range(1, nt):
        psi_next = time_step(state_focusing, dt, rhs_func_focusing, method='rk45')
        psi = state_focusing.get_current()
        
        # Enforce zero-flux Neumann BCs explicitly after each time step
        psi = enforce_zero_flux_neumann_bc(psi, dx, order=bc_order)
        state_focusing.psi_now = psi.copy()
        
        Psi_focusing[step] = psi.copy()
        times_focusing[step] = state_focusing.get_current_time()

print("Focusing case completed successfully!")

# ---------- BC check ----------
dpsi_dx_focusing, _, _ = bf.differentiate_1_2(Psi_focusing[-1], neumann_bc=(0.0, 0.0))
print(f"\nNeumann BC check (final time):")
print(f"  Focusing:  |ψ_x(0)|={np.abs(dpsi_dx_focusing[0]): .3e}, |ψ_x(L)|={np.abs(dpsi_dx_focusing[-1]): .3e}")

# ---------- Compute analytical solution for comparison ----------
print("\nComputing analytical solution for comparison...")
Psi_exact = np.empty((nt, nx), dtype=np.complex128)
for step in range(nt):
    # Our simulation time t corresponds to breather time (t - 1)
    t_sim = times_focusing[step]
    t_breather = t_sim - 1.0
    if breather_type == 'peregrine':
        Psi_exact[step] = peregrine_exact(x, t_breather)
    elif breather_type == 'kuznetsov_ma':
        Psi_exact[step] = kuznetsov_ma_exact(x, t_breather, a=a_km)

# ============================================================
# Convergence test: max error vs dx
# ============================================================
# We keep the same physical parameters (T, L_domain, degree) and vary nx.
# To maintain stability, we scale dt with dx^2 relative to the baseline dx.
dx_base = dx
run_convergence = True

def simulate_max_error(nx_local: int, dt_base: float) -> tuple[float, float, float]:
    """Run a single simulation at resolution nx_local and return (dx, max_abs_err, max_rel_err)."""
    x_local = np.linspace(-L_domain / 2.0, L_domain / 2.0, nx_local)
    dx_local = x_local[1] - x_local[0]
    # Scale time step with dx^2 relative to baseline to keep CFL-like behavior
    dt_local = dt_base * (dx_local / dx_base) ** 2
    nt_local = int(T / dt_local) + 1

    bf_local = bspf1d.from_grid(degree=degree, x=x_local, use_clustering=True, clustering_factor=2.0)

    # Initial condition at t = -1 in breather time
    if breather_type == 'peregrine':
        psi0_local = peregrine_exact(x_local, -1.0).astype(np.complex128)
    elif breather_type == 'kuznetsov_ma':
        psi0_local = kuznetsov_ma_exact(x_local, -1.0, a=a_km).astype(np.complex128)
    else:
        raise ValueError(f"Unknown breather type: {breather_type}")

    bc_order_local = min(degree, 5)
    psi0_local = enforce_zero_flux_neumann_bc(psi0_local, dx_local, order=bc_order_local)

    # Potential
    V_local = np.full(nx_local, V_value, dtype=np.float64)
    rhs_func_local = create_schrodinger_rhs(bf_local, V_local, g=g_focusing, neumann_bc=neumann_bc)

    # Time integration (store only current state)
    with TimeStepperState(psi0_local.copy(), t_init=0.0, dt=dt_local, method='rk45', t_final=T, show_progress=False) as state:
        for _ in range(1, nt_local):
            _ = time_step(state, dt_local, rhs_func_local, method='rk45')
            psi_curr = state.get_current()
            psi_curr = enforce_zero_flux_neumann_bc(psi_curr, dx_local, order=bc_order_local)
            state.psi_now = psi_curr.copy()

        psi_final = state.get_current()
        t_final = state.get_current_time()

    # Analytical at final time (breather time shift)
    t_breather_final = t_final - 1.0
    if breather_type == 'peregrine':
        psi_exact_final = peregrine_exact(x_local, t_breather_final)
    elif breather_type == 'kuznetsov_ma':
        psi_exact_final = kuznetsov_ma_exact(x_local, t_breather_final, a=a_km)

    abs_err = np.abs(psi_final - psi_exact_final)
    abs_exact = np.abs(psi_exact_final)
    rel_err = abs_err / np.maximum(abs_exact, 1e-12)

    max_abs_err = np.max(abs_err)
    max_rel_err = np.max(rel_err)
    return dx_local, max_abs_err, max_rel_err

# Run convergence study (lightweight resolutions to keep runtime reasonable)
if run_convergence:
    nx_list = np.geomspace(256, 1024, 10).astype(int)
    conv_results = []
    print("\n" + "=" * 60)
    print("Convergence study: max error vs dx")
    print("=" * 60)
    for nx_c in nx_list:
        dx_c, max_abs_err_c, max_rel_err_c = simulate_max_error(nx_c, dt)
        conv_results.append((dx_c, max_abs_err_c, max_rel_err_c))
        print(f"  nx = {nx_c:4d}, dx = {dx_c:.4f}, max_abs_err = {max_abs_err_c:.3e}, max_rel_err = {max_rel_err_c:.3e}")

    # Plot log-log of max_rel_err vs dx
    try:
        import matplotlib.pyplot as plt

        dx_vals = np.array([r[0] for r in conv_results])
        max_rel_vals = np.array([r[2] for r in conv_results])

        fig_conv, ax_conv = plt.subplots(1, 1, figsize=(6, 5))
        ax_conv.loglog(dx_vals, max_rel_vals, 'o-', label='Max relative error')
        ax_conv.set_xlabel('dx (log scale)')
        ax_conv.set_ylabel('Max relative error (log scale)')
        ax_conv.set_title('Convergence of max relative error vs dx')
        ax_conv.grid(True, which='both', ls='--', alpha=0.5)

        # Estimate slope
        if len(dx_vals) >= 2:
            coeffs = np.polyfit(np.log(dx_vals), np.log(max_rel_vals), 1)
            slope = coeffs[0]
            ax_conv.text(0.05, 0.95, f"Slope ≈ {slope:.2f}", transform=ax_conv.transAxes,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax_conv.legend()
        plt.tight_layout()
        plt.show()
    except ImportError as e:
        print(f"Convergence plot skipped (missing dependency): {e}")

# ---------- Visualization: space-time plots ----------
try:
    import matplotlib.pyplot as plt

    # Compute fields for plotting
    amp_num = np.abs(Psi_focusing)**2
    amp_exact = np.abs(Psi_exact)**2
    abs_error = np.abs(Psi_focusing - Psi_exact)
    abs_exact = np.abs(Psi_exact)
    rel_error = abs_error / np.maximum(abs_exact, 1e-12)

    breather_name = breather_type.replace('_', '-').title()
    if breather_type == 'kuznetsov_ma':
        breather_name += f' (a={a_km})'

    # Space-time plots: numerical amplitude and relative error
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle(f'{breather_name} Breather: Space-Time (Focusing Case)', fontsize=16, fontweight='bold')

    # Numerical |psi|^2
    im0 = axes[0].imshow(
        amp_num,
        extent=[x[0], x[-1], times_focusing[0], times_focusing[-1]],
        origin='lower',
        aspect='auto',
        cmap='viridis'
    )
    axes[0].set_title(r'$|\psi_{\mathrm{num}}|^2$')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    axes[0].set_xlim(-5,5)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Relative error
    im1 = axes[1].imshow(
        rel_error,
        extent=[x[0], x[-1], times_focusing[0], times_focusing[-1]],
        origin='lower',
        aspect='auto',
        cmap='magma',
        vmin=np.min(rel_error),
        vmax=np.max(rel_error)
    )
    axes[1].set_title(r'Relative error: $|\psi_{\mathrm{num}}-\psi_{\mathrm{exact}}|/|\psi_{\mathrm{exact}}|$')
    axes[1].set_xlabel('x')
    axes[1].set_xlim(-5,5)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Print error statistics
    print("\n" + "="*60)
    print("Error Analysis:")
    print("="*60)
    abs_error_final = np.abs(Psi_focusing[-1] - Psi_exact[-1])
    abs_exact_final = np.abs(Psi_exact[-1])
    rel_error_final = abs_error_final / np.maximum(abs_exact_final, 1e-12)

    max_abs_error = np.max(abs_error_final)
    l2_abs_error = np.sqrt(np.mean(abs_error_final**2))
    max_rel_error = np.max(rel_error_final)
    l2_rel_error = np.sqrt(np.mean(rel_error_final**2))

    print(f"Final time (t = {T:.3f}):")
    print(f"  Max absolute error: {max_abs_error:.6e}")
    print(f"  L2 absolute error: {l2_abs_error:.6e}")
    print(f"  Max relative error: {max_rel_error:.6e}")
    print(f"  L2 relative error: {l2_rel_error:.6e}")

    plt.tight_layout()
    plt.show()

except ImportError as e:
    print(f"Visualization skipped (missing dependency): {e}")
except Exception as exc:  # pragma: no cover - visualization is optional
    print("Visualization error:", exc)
    import traceback
    traceback.print_exc()
