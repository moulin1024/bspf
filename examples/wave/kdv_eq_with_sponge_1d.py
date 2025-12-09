from __future__ import annotations

import numpy as np
import numpy.typing as npt
import sys
import os

# Import time steppers and BSPF from bspf library
from bspf import TimeStepperState, time_step, bspf1d

Array = npt.NDArray[np.float64]

# ============================================================
# Control Parameters
# ============================================================
# Domain parameters
L_domain = 200  # domain length
nx = 256              # grid points (including endpoints)25

# Time parameters
T = 300            # final time
dt = 2e-2             # time step

# BSPF parameters
degree = 5            # B-spline degree

# Initial condition parameters
# Options for interesting phenomena:
# 1. "single_soliton": Single traveling soliton (stable, no change)
# 2. "two_solitons": Two solitons with different speeds (collision/interaction)
# 3. "gaussian": Gaussian profile (can evolve into solitons)
# 4. "step": Step function (wave breaking, dispersive shock waves)
# 5. "sine_wave": Sine wave (modulation instability, soliton formation)
ic_type = "gaussian"  # Options: "single_soliton", "two_solitons", "gaussian", "step", "sine_wave"

# Single soliton parameters
c = 1.0               # Soliton speed parameter (c > 0)
x0 = None             # Will be set to L_domain / 2.0 (soliton center)

# Two solitons parameters (for collision/interaction)
c1 = 2.0              # First soliton speed (faster)
c2 = 0.5              # Second soliton speed (slower)
x01 = L_domain * 0.3  # First soliton center
x02 = L_domain * 0.7  # Second soliton center

# Gaussian parameters
A_gaussian = 2.0      # Gaussian amplitude
sigma_gaussian = 5.0  # Gaussian width
x0_gaussian = L_domain / 4.0  # Gaussian center

# Step function parameters
u_left = 1.0          # Left side value
u_right = 0.0         # Right side value
x_step = L_domain / 2.0  # Step location
step_width = 1.0      # Transition width (for smooth step)

# Sponge zone parameters (for absorbing boundary conditions)
use_sponge_zone = True  # Use sponge zone instead of Dirichlet BCs
sponge_width = L_domain * 0.25  # Width of sponge zone at each boundary (10% of domain)
sponge_strength = 0.05  # Damping strength (sigma0) in sponge zone
nu_sponge = 1  # Artificial viscosity coefficient in sponge zone
# The sponge blends KdV into: u_t = -sigma0*u + nu*u_xx (damped-diffusion)

# Sine wave parameters
A_sine = 1.0          # Sine wave amplitude
k_sine = 0.5          # Wavenumber
x0_sine = L_domain / 2.0  # Sine wave center

# Boundary condition parameters
# Options:
# - use_sponge_zone = True: Sponge zone (absorbing BCs) - more physically consistent
# - use_sponge_zone = False: Dirichlet BCs (fixed values at boundaries)

# ============================================================
# Setup: Grid, Operator, Initial Condition
# ============================================================
# Set soliton center if not set
if x0 is None:
    x0 = L_domain / 2.0  # Center of domain

# Create spatial grid
x = np.linspace(0.0, L_domain, nx)
dx = x[1] - x[0]     # grid spacing

# KdV soliton width parameter: κ = √c/2
kappa = np.sqrt(c) / 2.0

# Print parameters
print("="*60)
print("Control Parameters:")
print("="*60)
print(f"Domain: L = {L_domain:.3f}, nx = {nx}, dx = {dx:.6f}")
print(f"Time: T = {T:.3f}, dt = {dt:.6f}, dx² = {dx**2:.6f}")
print(f"BSPF: degree = {degree}")
print(f"Initial condition type: {ic_type}")
if use_sponge_zone:
    print(f"Boundary conditions: Sponge zone (absorbing BCs)")
else:
    print(f"Boundary conditions: Dirichlet (u(0) = u(L) = 0)")
print("="*60)
print("Korteweg-de Vries (KdV) Equation:")
print("="*60)
print("  u_t + u*u_x + u_xxx = 0")
print("")
print("The KdV equation describes:")
print("  - Nonlinear wave propagation")
print("  - Soliton solutions")
print("  - Balance between nonlinearity and dispersion")
print("")
print("Boundary Conditions:")
print("="*60)
if use_sponge_zone:
    print("Using sponge zone (absorbing boundary conditions)")
    print("  - Physically consistent: absorbs outgoing waves without reflections")
    print("  - Prevents numerical instabilities from hard boundary conditions")
    if ic_type == "step":
        print(f"  - Left target: {u_left:.2f}, Right target: {u_right:.2f}")
    else:
        print(f"  - Target: 0.0 (both boundaries)")
else:
    if ic_type == "step":
        print(f"Using Dirichlet BCs: u(0,t) = {u_left:.2f}, u(L,t) = {u_right:.2f}")
        print("")
        print("Boundary values match the step function values for consistency.")
    else:
        print("Using Dirichlet BCs: u(0,t) = u(L,t) = 0")
        print("")
        print("This is a well-posed boundary condition for KdV on finite domains.")
        print("The Dirichlet conditions u(0) = u(L) = 0 provide 2 conditions,")
        print("and the equation structure provides the third condition implicitly.")
print("="*60)

# Number of time steps
nt = int(T / dt) + 1

# Create BSPF operator
bf = bspf1d.from_grid(degree=degree, x=x)

# Initial condition based on ic_type
if x0 is None:
    x0 = L_domain / 2.0  # Center of domain

if ic_type == "single_soliton":
    # Single KdV soliton: u(x,0) = 3c * sech²(κ*(x - x0))
    # where κ = √c/2
    kappa = np.sqrt(c) / 2.0
    u0 = 3.0 * c * (1.0 / np.cosh(kappa * (x - x0)))**2
    print(f"  Initial condition: Single soliton (c={c:.2f}, x0={x0:.2f})")
    
elif ic_type == "two_solitons":
    # Two solitons with different speeds (will collide/interact)
    kappa1 = np.sqrt(c1) / 2.0
    kappa2 = np.sqrt(c2) / 2.0
    soliton1 = 3.0 * c1 * (1.0 / np.cosh(kappa1 * (x - x01)))**2
    soliton2 = 3.0 * c2 * (1.0 / np.cosh(kappa2 * (x - x02)))**2
    u0 = soliton1 + soliton2
    print(f"  Initial condition: Two solitons")
    print(f"    Soliton 1: c={c1:.2f}, x0={x01:.2f} (faster, left)")
    print(f"    Soliton 2: c={c2:.2f}, x0={x02:.2f} (slower, right)")
    print(f"    They will collide and interact!")
    
elif ic_type == "gaussian":
    # Gaussian profile: can evolve into solitons
    u0 = A_gaussian * np.exp(-(x - x0_gaussian)**2 / (2.0 * sigma_gaussian**2))
    print(f"  Initial condition: Gaussian (A={A_gaussian:.2f}, σ={sigma_gaussian:.2f})")
    print(f"    May evolve into solitons or show dispersive behavior")
    
elif ic_type == "step":
    # Smooth step function: can lead to wave breaking and dispersive shock waves
    u0 = u_left + (u_right - u_left) * 0.5 * (1.0 + np.tanh((x - x_step) / step_width))
    # With sponge zone, we don't need aggressive boundary smoothing
    # The sponge will naturally handle waves near boundaries
    if not use_sponge_zone:
        # Only apply smoothing if using Dirichlet BCs (legacy code)
        boundary_smooth_width = 5  # Number of points to smooth
        for i in range(boundary_smooth_width):
            # Smooth left boundary: gradually transition to u_left
            smoothing = (boundary_smooth_width - i) / boundary_smooth_width
            u0[i] = u_left * smoothing + u0[i] * (1.0 - smoothing)
            # Smooth right boundary: gradually transition to u_right
            u0[nx - 1 - i] = u_right * smoothing + u0[nx - 1 - i] * (1.0 - smoothing)
    print(f"  Initial condition: Smooth step (u_left={u_left:.2f}, u_right={u_right:.2f})")
    if use_sponge_zone:
        print(f"    Sponge zone will absorb waves near boundaries (no need for boundary smoothing)")
    print(f"    May show wave breaking and dispersive shock waves")
    
elif ic_type == "sine_wave":
    # Sine wave: can show modulation instability and soliton formation
    u0 = A_sine * np.sin(k_sine * (x - x0_sine))
    # Make it localized (multiply by Gaussian envelope)
    envelope = np.exp(-(x - x0_sine)**2 / (2.0 * (L_domain/4.0)**2))
    u0 = u0 * envelope
    print(f"  Initial condition: Localized sine wave (A={A_sine:.2f}, k={k_sine:.2f})")
    print(f"    May show modulation instability and soliton formation")
    
else:
    raise ValueError(f"Unknown ic_type: {ic_type}")

u0 = u0.astype(np.float64)

# ============================================================
# Sponge zone weight function (blending function)
# ============================================================
def build_sponge_weight(x: Array, L: float, sponge_width: float) -> Array:
    """
    Construct a smooth weight w(x) in [0,1]:
        w = 0  : pure KdV region
        w -> 1 : deep sponge region
    
    Uses quintic smoothstep for C^2 continuity to minimize reflections.
    Sponge is applied on both left and right boundaries.
    """
    w = np.zeros_like(x)
    for i, xi in enumerate(x):
        # Distance from left boundary
        dist_left = xi
        # Distance from right boundary
        dist_right = L - xi
        
        # Check if in left sponge zone
        if dist_left < sponge_width:
            # eta: 1 at sponge start, 0 at boundary
            eta = dist_left / sponge_width
            # s: 0 at sponge start, 1 at boundary
            s = 1.0 - eta
            # Quintic smoothstep: s^3 (10 - 15 s + 6 s^2)
            w[i] = s**3 * (10.0 - 15.0*s + 6.0*s**2)
        # Check if in right sponge zone (only if not already in left)
        elif dist_right < sponge_width:
            # eta: 1 at sponge start, 0 at boundary
            eta = dist_right / sponge_width
            # s: 0 at sponge start, 1 at boundary
            s = 1.0 - eta
            # Quintic smoothstep: s^3 (10 - 15 s + 6 s^2)
            w[i] = s**3 * (10.0 - 15.0*s + 6.0*s**2)
        else:
            w[i] = 0.0
    return w

# Boundary condition setup
if use_sponge_zone:
    # Build sponge weight function (blending function)
    sponge_weight = build_sponge_weight(x, L_domain, sponge_width)
    n_sponge = int(sponge_width / dx)  # Number of grid points in sponge zone
    
    print(f"  Using sponge zone boundary conditions (viscosity + damping)")
    print(f"    Sponge width: {sponge_width:.2f} ({n_sponge} grid points at each end)")
    print(f"    Damping strength (sigma0): {sponge_strength:.2f}")
    print(f"    Viscosity coefficient (nu): {nu_sponge:.2f}")
    print(f"    Blended equation: u_t = (1-w)*(-u*u_x - u_xxx) + w*(-sigma0*u + nu*u_xx)")
    print(f"    Quintic smoothstep blending minimizes reflections")
else:
    # Traditional Dirichlet BCs - create dummy array (won't be used)
    sponge_weight = np.zeros(nx, dtype=np.float64)
    # Enforce Dirichlet BCs on initial condition
    if ic_type == "step":
        # For step case: match the step values at boundaries
        u0[0] = u_left
        u0[-1] = u_right
        print(f"  Enforced Dirichlet BCs on initial condition: u(0) = {u_left:.2f}, u(L) = {u_right:.2f}")
    else:
        # For other cases: zero Dirichlet BCs
        u0[0] = 0.0
        u0[-1] = 0.0
        print(f"  Enforced Dirichlet BCs on initial condition: u(0) = u(L) = 0")

# ============================================================
# KdV RHS Function
# ============================================================
def kdv_rhs(u: Array, bspf_op: bspf1d, ic_type: str, u_left: float, u_right: float,
             use_sponge: bool, sponge_weight: Array) -> Array:
    """
    Right-hand side of the KdV equation with blended sponge zone.
    
    In pure KdV region (w=0): u_t = -u*u_x - u_xxx
    In sponge zone (w->1): u_t = -sigma0*u + nu*u_xx (damped-diffusion)
    Blended: u_t = (1-w)*(-u*u_x - u_xxx) + w*(-sigma0*u + nu*u_xx)
    
    Parameters:
    -----------
    u : array
        Current solution
    bspf_op : bspf1d
        BSPF operator for spatial derivatives
    ic_type : str
        Initial condition type (to determine boundary values)
    u_left : float
        Left boundary value (for step case, if not using sponge)
    u_right : float
        Right boundary value (for step case, if not using sponge)
    use_sponge : bool
        Whether to use sponge zone (True) or Dirichlet BCs (False)
    sponge_weight : array
        Sponge zone blending weight w(x) in [0,1] (if use_sponge=True)
        
    Returns:
    --------
    du_dt : array
        Time derivative du/dt
    """
    # Handle boundary conditions
    u_bc = u.copy()
    if not use_sponge:
        # Traditional Dirichlet BCs
        if ic_type == "step":
            u_bc[0] = u_left
            u_bc[-1] = u_right
        else:
            u_bc[0] = 0.0
            u_bc[-1] = 0.0
    else:
        # With sponge zone, enforce left Dirichlet BC
        if ic_type == "step":
            u_bc[0] = u_left
        else:
            u_bc[0] = 0.0
        # Right boundary handled by sponge zone
    
    # Compute derivatives needed for KdV
    du_dx, _ = bspf_op.differentiate(u_bc, k=1)
    du_dxxx, _ = bspf_op.differentiate(u_bc, k=3)
    
    # Physical KdV part: u_t = -u*u_x - u_xxx
    rhs_kdv = -u_bc * du_dx - du_dxxx
    
    if not use_sponge:
        # Traditional Dirichlet BCs: fixed boundaries imply du/dt = 0 at boundaries
        du_dt = rhs_kdv
        if ic_type == "step":
            du_dt[0] = 0.0
            du_dt[-1] = 0.0
        else:
            du_dt[0] = 0.0
            du_dt[-1] = 0.0
        return du_dt
    
    # Sponge part: damped-diffusion u_t = -sigma0*u + nu*u_xx
    # Compute second derivative for viscosity term
    du_dxx, _ = bspf_op.differentiate(u_bc, k=2)
    rhs_sponge = -sponge_strength * u_bc + nu_sponge * du_dxx
    
    # Blend between KdV and sponge: u_t = (1-w)*rhs_kdv + w*rhs_sponge
    du_dt = (1.0 - sponge_weight) * rhs_kdv + sponge_weight * rhs_sponge
    
    # Left boundary Dirichlet -> du/dt = 0 there
    du_dt[0] = 0.0
    
    return du_dt

# Create RHS function with bound arguments
def rhs_func(u):
    return kdv_rhs(u, bf, ic_type, u_left, u_right, use_sponge_zone, sponge_weight)

if use_sponge_zone:
    print(f"  Created KdV RHS function (BSPF, Sponge zone BCs)")
else:
    if ic_type == "step":
        print(f"  Created KdV RHS function (BSPF, Dirichlet BCs: u(0) = {u_left:.2f}, u(L) = {u_right:.2f})")
    else:
        print("  Created KdV RHS function (BSPF, Dirichlet BCs: u(0) = u(L) = 0)")

# ---------- Time integration with rk23 ----------
print("\n" + "="*60)
print("Running BSPF-rk23 time integration...")
print("="*60)

with TimeStepperState(u0.copy(), t_init=0.0, dt=dt, method='rk23', t_final=T, show_progress=True) as state:
    U = np.empty((nt, nx), dtype=np.float64)
    U[0] = u0.copy()
    times = np.zeros(nt)
    times[0] = 0.0

    for step in range(1, nt):
        u_next = time_step(state, dt, rhs_func, method='rk23')
        u = state.get_current()
        
        # Blowup detection: check for NaN or very large values
        max_u = np.max(np.abs(u))
        if np.isnan(max_u) or max_u > 1e6:
            print(f"\nWARNING: Blowup detected at step {step}, t={state.get_current_time():.6f}")
            print(f"  Max |u| = {max_u:.6e}")
            print(f"  Stopping simulation early")
            # Truncate arrays to current step
            U = U[:step]
            times = times[:step]
            break
        
        # Handle boundary conditions after each time step
        if use_sponge_zone:
            # Sponge zone: enforce left Dirichlet BC, right boundary handled by sponge
            if ic_type == "step":
                u[0] = u_left
            else:
                u[0] = 0.0
            # Sponge zone in RHS naturally absorbs waves at right boundary
            # Just clip extreme values as safety check
            max_val = 10.0  # Reasonable maximum value
            u = np.clip(u, -max_val, max_val)
        else:
            # Traditional Dirichlet BCs
            if ic_type == "step":
                # For step case: match the step values at boundaries
                # Also apply smoothing near boundaries to prevent blowup from third derivative
                u[0] = u_left
                u[-1] = u_right
                # Smooth a few points near boundaries to reduce third derivative spikes
                boundary_smooth_width = 3
                for i in range(1, min(boundary_smooth_width + 1, nx // 2)):
                    # Smooth left boundary region: gradually transition to u_left
                    smoothing = (boundary_smooth_width - i + 1) / boundary_smooth_width
                    u[i] = u_left * (1.0 - smoothing) + u[i] * smoothing
                    # Smooth right boundary region: gradually transition to u_right
                    u[nx - 1 - i] = u_right * (1.0 - smoothing) + u[nx - 1 - i] * smoothing
            else:
                # For other cases: zero Dirichlet BCs
                u[0] = 0.0
                u[-1] = 0.0
        state.psi_now = u.copy()
        
        U[step] = u.copy()
        times[step] = state.get_current_time()

print("Simulation completed successfully!")

# ---------- BC check ----------
if use_sponge_zone:
    print(f"\nSponge zone check (final time):")
    print(f"  u(0)={U[-1][0]: .3e}, u(L)={U[-1][-1]: .3e}")
    print(f"  Sponge zone should have damped waves near boundaries")
    print(f"  Values near boundaries may be non-zero (this is expected with sponge zone)")
else:
    print(f"\nDirichlet BC check (final time):")
    print(f"  u(0)={U[-1][0]: .3e}, u(L)={U[-1][-1]: .3e}")

# ---------- Conservation check (KdV has conserved quantities) ----------
# KdV conserves: ∫ u dx (mass), ∫ u² dx (momentum), ∫ (u²/2 - u_xx) dx (energy)
mass_initial = np.trapz(U[0], x)
mass_final = np.trapz(U[-1], x)
momentum_initial = np.trapz(U[0]**2, x)
momentum_final = np.trapz(U[-1]**2, x)

print(f"\nConservation check:")
print(f"  Mass: initial = {mass_initial:.6f}, final = {mass_final:.6f}, error = {abs(mass_final - mass_initial):.6e}")
print(f"  Momentum: initial = {momentum_initial:.6f}, final = {momentum_final:.6f}, error = {abs(momentum_final - momentum_initial):.6e}")

# ---------- Visualization ----------
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: u at initial and final time
    ax1 = axes[0]
    ax1.plot(x, U[0], label="t=0 s", linewidth=2)
    ax1.plot(x, U[-1], label=f"t={T} s", linewidth=2)
    if use_sponge_zone:
        # Show sponge zone region (only at right boundary)
        ax1.axvspan(L_domain - sponge_width, L_domain, alpha=0.2, color='red', label='Sponge zone (right)')
    ax1.set_xlabel("x")
    ax1.set_ylabel("u(x,t)")
    title = "KdV (BSPF): Solution at Initial and Final Time"
    if use_sponge_zone:
        title += " [Sponge Zone BCs]"
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Space-time plot of u evolution
    ax2 = axes[1]
    im = ax2.imshow(U, aspect='auto', 
                    extent=[0, L_domain, 0, T], origin='lower',
                    cmap='plasma', interpolation='bilinear')
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.set_title("KdV (BSPF): Solution Evolution (Space-Time)")
    plt.colorbar(im, ax=ax2, label="u(x,t)")
    
    plt.tight_layout()
    plt.show()
except Exception as exc:  # pragma: no cover - visualization is optional
    print("Plot skipped:", exc)

