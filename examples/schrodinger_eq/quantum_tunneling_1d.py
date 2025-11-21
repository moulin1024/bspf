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
L_domain = 40.0  # domain length (dimensionless)
nx = 512        # grid points (including endpoints)

# Time parameters
T = 5.0         # final time (dimensionless)
dt = 0.001      # time step (dimensionless)

# BSPF parameters
degree = 5       # B-spline degree

# Potential barrier parameters
V0 = 10.0        # Barrier height (dimensionless)
bw = 0.5         # Barrier width (dimensionless)
barrier_center = None  # Will be set to L_domain / 2.0

# Initial wavefunction parameters
ke = 5.0         # Kinetic energy (dimensionless)
sig = 1.0        # Initial wavefunction spread (dimensionless)
x0 = None        # Will be set to position before barrier

# Boundary condition parameters
neumann_bc = (0.0, 0.0)  # (left_flux, right_flux) = (0, 0) for zero flux

# Potential (for linear Schrödinger, V = 0)
V_value = 0.0    # Potential value outside barrier

# Nonlinearity parameter (0 for linear, non-zero for nonlinear)
g = 0.0          # Nonlinearity parameter (0 for linear quantum tunneling)

# ============================================================
# Setup: Grid, Operator, Potential, Initial Condition
# ============================================================
# Set barrier center if not set
if barrier_center is None:
    barrier_center = L_domain / 2.0  # Center of domain

# Create spatial grid
x = np.linspace(0.0, L_domain, nx)
dx = x[1] - x[0]  # grid spacing

# Wave vector: k0 = sqrt(2*ke) in dimensionless units (ℏ = m = 1)
k0 = np.sqrt(2.0 * ke)

# Calculate barrier boundaries
bposgrid = int(nx / 2.0)
bw_half_grid = max(1, int(bw / (2.0 * dx)))
bl = max(0, bposgrid - bw_half_grid)
br = min(nx, bposgrid + bw_half_grid)
if br <= bl:
    br = bl + 1
    br = min(br, nx)

# Create potential array
V = np.full(nx, V_value, dtype=np.float64)
V[bl:br] = V0  # Set barrier

# Initial wavefunction: Gaussian wave packet with momentum
# Normalization: ∫|ψ|²dx = 1
ac = 1.0 / np.sqrt(np.sqrt(np.pi) * sig)
x0_init = bl * dx - 6 * sig  # Position before barrier
psigauss = ac * np.exp(-(x - x0_init)**2 / (2.0 * sig**2))
# Complex wavefunction: ψ = psigauss * exp(i*k0*x)
psi0 = (psigauss * np.cos(k0 * x) + 1j * psigauss * np.sin(k0 * x)).astype(np.complex128)

# Enforce zero-flux Neumann BCs on initial condition
# Use order matching the BSPF degree (or closest available: 1, 2, 3, 4, or 5)
bc_order = min(degree, 5)  # Cap at 5th order (highest implemented)
print(f"  Using {bc_order}-th order finite difference for Neumann BC enforcement")
psi0 = enforce_zero_flux_neumann_bc(psi0, dx, order=bc_order)

# Print parameters
print("="*60)
print("Control Parameters:")
print("="*60)
print(f"Domain: L = {L_domain:.3f}, nx = {nx}, dx = {dx:.6f}")
print(f"Time: T = {T:.3f}, dt = {dt:.6f}, dx² = {dx**2:.6f}")
print(f"BSPF: degree = {degree}")
print(f"Potential barrier:")
print(f"  Height: V0 = {V0:.3f}")
print(f"  Width: bw = {bw:.3f}")
print(f"  Center: x = {barrier_center:.3f}")
print(f"  Grid indices: {bl} to {br} ({br-bl} points)")
print(f"Initial wavefunction:")
print(f"  Kinetic energy: ke = {ke:.3f}")
print(f"  Wave vector: k0 = {k0:.3f}")
print(f"  Spread: σ = {sig:.3f}")
print(f"  Initial position: x0 = {x0_init:.3f}")
print(f"Boundary conditions: Neumann (zero flux)")
print(f"Potential: V_max = {np.max(V):.3f}, Nonlinearity: g = {g:.3f}")
print("="*60)

# Number of time steps
nt = int(T / dt) + 1

# Create BSPF operator
bf = bspf1d.from_grid(degree=degree, x=x)

# Create RHS function using built library function
# Schrödinger: i*∂ψ/∂t = -(1/2)*∂²ψ/∂x² + V*ψ + g*|ψ|²*ψ
# For quantum tunneling: g = 0 (linear case)
rhs_func = create_schrodinger_rhs(bf, V, g=g, neumann_bc=neumann_bc)
print("  Created RHS function (Neumann BCs: zero flux)")

# ---------- Time integration with rk23 ----------
print("\n" + "="*60)
print("Running BSPF-rk23 time integration...")
print("="*60)

with TimeStepperState(psi0.copy(), t_init=0.0, dt=dt, method='rk23', t_final=T, show_progress=True) as state:
    Psi = np.empty((nt, nx), dtype=np.complex128)
    Psi[0] = psi0.copy()
    times = np.zeros(nt)
    times[0] = 0.0

    for step in range(1, nt):
        psi_next = time_step(state, dt, rhs_func, method='rk23')
        psi = state.get_current()
        
        # Enforce zero-flux Neumann BCs explicitly after each time step
        # Use order matching the BSPF degree
        psi = enforce_zero_flux_neumann_bc(psi, dx, order=bc_order)
        state.psi_now = psi.copy()
        
        Psi[step] = psi.copy()
        times[step] = state.get_current_time()

print("Simulation completed successfully!")

# ---------- BC check ----------
dpsi_dx, _, _ = bf.differentiate_1_2(Psi[-1], neumann_bc=(0.0, 0.0))
print(f"\nNeumann BC check (final time):")
print(f"  |ψ_x(0)|={np.abs(dpsi_dx[0]): .3e}, |ψ_x(L)|={np.abs(dpsi_dx[-1]): .3e}")

# ---------- Visualization ----------
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: |ψ|² at initial and final time
    ax1 = axes[0, 0]
    ax1.plot(x, np.abs(Psi[0])**2, label="t=0", linewidth=2)
    ax1.plot(x, np.abs(Psi[-1])**2, label=f"t={T}", linewidth=2)
    # Plot potential barrier
    V_normalized = V / np.max(V) * np.max(np.abs(Psi)**2)
    ax1.fill_between(x, 0, V_normalized, where=(V > 0), 
                     color='red', alpha=0.3, label='Barrier')
    ax1.set_xlabel("x")
    ax1.set_ylabel("|ψ|²")
    ax1.set_title("Probability Density: |ψ|²")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Real and Imaginary parts at final time
    ax2 = axes[0, 1]
    max_psi = np.max(np.abs(Psi[-1]))
    if max_psi > 0:
        ax2.plot(x, Psi[-1].real / max_psi, label="Re(ψ)", linewidth=2, alpha=0.7)
        ax2.plot(x, Psi[-1].imag / max_psi, label="Im(ψ)", linewidth=2, alpha=0.7)
    ax2.fill_between(x, -1, 1, where=(V > 0), 
                     color='red', alpha=0.2)
    ax2.set_xlabel("x")
    ax2.set_ylabel("Normalized ψ")
    ax2.set_title("Wavefunction Components (t={:.2f})".format(T))
    ax2.set_ylim(-1.2, 1.2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Space-time plot of |ψ|² evolution
    ax3 = axes[1, 0]
    im = ax3.imshow(np.abs(Psi)**2, aspect='auto', 
                    extent=[0, L_domain, 0, T], origin='lower',
                    cmap='viridis', interpolation='bilinear')
    # Overlay barrier position
    ax3.axvline(x=barrier_center, color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label='Barrier center')
    ax3.set_xlabel("x")
    ax3.set_ylabel("t")
    ax3.set_title("|ψ|² Evolution (Space-Time)")
    ax3.legend()
    plt.colorbar(im, ax=ax3, label="|ψ|²")
    
    # Plot 4: Potential and initial condition
    ax4 = axes[1, 1]
    ax4.plot(x, V / np.max(V), 'r-', label='Potential (normalized)', linewidth=2)
    ax4.plot(x, np.abs(Psi[0])**2 / np.max(np.abs(Psi[0])**2), 
             'b-', label='|ψ|² at t=0 (normalized)', linewidth=2)
    ax4.set_xlabel("x")
    ax4.set_ylabel("Normalized magnitude")
    ax4.set_title("Potential Barrier and Initial Condition")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
except Exception as exc:  # pragma: no cover - visualization is optional
    print("Plot skipped:", exc)
