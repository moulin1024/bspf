from __future__ import annotations

import numpy as np
import numpy.typing as npt
import sys
import os

# Import time steppers and Schrödinger solver from bspf library
from bspf import TimeStepperState, time_step, bspf1d, create_schrodinger_rhs, enforce_zero_flux_neumann_bc

Array = npt.NDArray[np.complex128]

# ============================================================
# Control Parameters
# ============================================================
# Domain parameters
L_domain = 40  # domain length
nx = 512              # grid points (including endpoints)

# Time parameters
T = 5.0             # final time
dt = 0.001             # time step

# BSPF parameters
degree = 5            # B-spline degree

# Initial condition parameters (Gaussian wave packet)
x0 = None             # Will be set to L_domain / 2.0
sigma = L_domain/10.0          # Will be set to L_domain / 10.0

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
# Set initial condition parameters if not set
if x0 is None:
    x0 = L_domain / 2.0  # Center of domain
if sigma is None:
    sigma = L_domain / 10.0  # Width of Gaussian

# Create spatial grid
x = np.linspace(0.0, L_domain, nx)
dx = x[1] - x[0]     # grid spacing

# Print parameters
print("="*60)
print("Control Parameters:")
print("="*60)
print(f"Domain: L = {L_domain:.3f}, nx = {nx}, dx = {dx:.6f}")
print(f"Time: T = {T:.3f}, dt = {dt:.6f}, dx² = {dx**2:.6f}")
print(f"BSPF: degree = {degree}")
print(f"Initial condition: Gaussian wave packet")
print(f"  Center: x0 = {x0:.3f}, Width: σ = {sigma:.3f}")
print(f"Boundary conditions: Neumann (zero flux)")
print(f"Potential: V = {V_value:.3f}")
print(f"Focusing case: g = {g_focusing:.3f} (g < 0)")
print(f"Defocusing case: g = {g_defocusing:.3f} (g > 0)")
print("="*60)

# Number of time steps
nt = int(T / dt) + 1

# Create BSPF operator
bf = bspf1d.from_grid(degree=degree, x=x)

# Initial condition: Gaussian wave packet at domain center
# ψ(x,0) = exp(-(x-x0)²/(2σ²))
psi0 = np.exp(-(x - x0)**2 / (2.0 * sigma**2)).astype(np.complex128) + 1

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
with TimeStepperState(psi0.copy(), t_init=0.0, dt=dt, method='rk23', t_final=T, show_progress=True) as state_focusing:
    Psi_focusing = np.empty((nt, nx), dtype=np.complex128)
    Psi_focusing[0] = psi0.copy()
    times_focusing = np.zeros(nt)
    times_focusing[0] = 0.0

    for step in range(1, nt):
        psi_next = time_step(state_focusing, dt, rhs_func_focusing, method='rk23')
        psi = state_focusing.get_current()
        
        # Enforce zero-flux Neumann BCs explicitly after each time step
        psi = enforce_zero_flux_neumann_bc(psi, dx, order=bc_order)
        state_focusing.psi_now = psi.copy()
        
        Psi_focusing[step] = psi.copy()
        times_focusing[step] = state_focusing.get_current_time()

print("Focusing case completed successfully!")

# ============================================================
# Defocusing Case (g > 0)
# ============================================================
print("\n" + "="*60)
print("Running DEFOCUSING case (g > 0)...")
print("="*60)

# Create RHS function for defocusing case
rhs_func_defocusing = create_schrodinger_rhs(bf, V, g=g_defocusing, neumann_bc=neumann_bc)
print("  Created RHS function (defocusing case: g > 0, Neumann BCs: zero flux)")

# Time integration
with TimeStepperState(psi0.copy(), t_init=0.0, dt=dt, method='rk23', t_final=T, show_progress=True) as state_defocusing:
    Psi_defocusing = np.empty((nt, nx), dtype=np.complex128)
    Psi_defocusing[0] = psi0.copy()
    times_defocusing = np.zeros(nt)
    times_defocusing[0] = 0.0

    for step in range(1, nt):
        psi_next = time_step(state_defocusing, dt, rhs_func_defocusing, method='rk23')
        psi = state_defocusing.get_current()
        
        # Enforce zero-flux Neumann BCs explicitly after each time step
        psi = enforce_zero_flux_neumann_bc(psi, dx, order=bc_order)
        state_defocusing.psi_now = psi.copy()
        
        Psi_defocusing[step] = psi.copy()
        times_defocusing[step] = state_defocusing.get_current_time()

print("Defocusing case completed successfully!")

# ---------- BC check ----------
dpsi_dx_focusing, _, _ = bf.differentiate_1_2(Psi_focusing[-1], neumann_bc=(0.0, 0.0))
dpsi_dx_defocusing, _, _ = bf.differentiate_1_2(Psi_defocusing[-1], neumann_bc=(0.0, 0.0))
print(f"\nNeumann BC check (final time):")
print(f"  Focusing:  |ψ_x(0)|={np.abs(dpsi_dx_focusing[0]): .3e}, |ψ_x(L)|={np.abs(dpsi_dx_focusing[-1]): .3e}")
print(f"  Defocusing: |ψ_x(0)|={np.abs(dpsi_dx_defocusing[0]): .3e}, |ψ_x(L)|={np.abs(dpsi_dx_defocusing[-1]): .3e}")

# ---------- Visualization ----------
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Modulation Instability: Focusing vs Defocusing NLS', fontsize=16, fontweight='bold')
    
    # Plot 1: Focusing - |ψ|² at initial and final time
    ax1 = axes[0, 0]
    ax1.plot(x, np.abs(Psi_focusing[0])**2, label="t=0 s", linewidth=2)
    ax1.plot(x, np.abs(Psi_focusing[-1])**2, label=f"t={T} s", linewidth=2)
    ax1.set_xlabel("x")
    ax1.set_ylabel("|ψ|²")
    ax1.set_title("Focusing (g < 0): |ψ|² at Initial and Final Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Defocusing - |ψ|² at initial and final time
    ax2 = axes[0, 1]
    ax2.plot(x, np.abs(Psi_defocusing[0])**2, label="t=0 s", linewidth=2)
    ax2.plot(x, np.abs(Psi_defocusing[-1])**2, label=f"t={T} s", linewidth=2)
    ax2.set_xlabel("x")
    ax2.set_ylabel("|ψ|²")
    ax2.set_title("Defocusing (g > 0): |ψ|² at Initial and Final Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Focusing - Space-time plot of |ψ|² evolution
    ax3 = axes[1, 0]
    im3 = ax3.imshow(np.abs(Psi_focusing)**2, aspect='auto', 
                     extent=[0, L_domain, 0, T], origin='lower',
                     cmap='plasma', interpolation='bilinear')
    ax3.set_xlabel("x")
    ax3.set_ylabel("t")
    ax3.set_title("Focusing (g < 0): |ψ|² Evolution (Space-Time)")
    plt.colorbar(im3, ax=ax3, label="|ψ|²")
    
    # Plot 4: Defocusing - Space-time plot of |ψ|² evolution
    ax4 = axes[1, 1]
    im4 = ax4.imshow(np.abs(Psi_defocusing)**2, aspect='auto', 
                     extent=[0, L_domain, 0, T], origin='lower',
                     cmap='plasma', interpolation='bilinear')
    ax4.set_xlabel("x")
    ax4.set_ylabel("t")
    ax4.set_title("Defocusing (g > 0): |ψ|² Evolution (Space-Time)")
    plt.colorbar(im4, ax=ax4, label="|ψ|²")
    
    plt.tight_layout()
    plt.show()
except Exception as exc:  # pragma: no cover - visualization is optional
    print("Plot skipped:", exc)


