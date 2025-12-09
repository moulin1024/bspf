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
L_domain = 100  # domain length
nx = 1024              # grid points (including endpoints)

# Time parameters
T = 10            # final time
dt = 0.001             # time step

# BSPF parameters
degree = 7            # B-spline degree

# Initial condition parameters (Gaussian wave packet)
x0 = None             # Will be set to L_domain / 2.0
sigma = L_domain/20.0          # Will be set to L_domain / 10.0

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
bf = bspf1d.from_grid(degree=degree, x=x, use_clustering=True, clustering_factor=2.0)

# Initial condition: Peregrine breather
# Our simulation time starts at t=0, but peregrine_exact uses t starting from -1
# So at t=0, we use peregrine_exact(x, -1)
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

# ---------- BC check ----------
dpsi_dx_focusing, _, _ = bf.differentiate_1_2(Psi_focusing[-1], neumann_bc=(0.0, 0.0))
print(f"\nNeumann BC check (final time):")
print(f"  Focusing:  |ψ_x(0)|={np.abs(dpsi_dx_focusing[0]): .3e}, |ψ_x(L)|={np.abs(dpsi_dx_focusing[-1]): .3e}")

# ---------- Visualization and Animation ----------
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    # Subsample data for animation (every nth frame)
    anim_skip = max(1, nt // 200)  # Show ~200 frames max
    anim_frames = list(range(0, nt, anim_skip))
    
    print(f"\nCreating animation with {len(anim_frames)} frames...")
    
    # Create figure with single plot for focusing case
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle('Modulation Instability: Focusing Case (g < 0)', fontsize=16, fontweight='bold')
    
    # Initialize plot line
    line, = ax.plot([], [], 'b-', linewidth=2, label='|ψ|²')
    
    # Set up axes
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("|ψ|²", fontsize=12)
    ax.set_title("Focusing (g < 0)", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([x[0], x[-1]])
    
    # Find global y-limits for consistent scaling
    amp_focusing_max = np.max(np.abs(Psi_focusing)**2)
    y_max = amp_focusing_max * 1.1
    ax.set_ylim([0, y_max])
    
    # Add time text
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def animate(frame_idx):
        """Update animation frame."""
        # Get actual step index
        step = anim_frames[frame_idx]
        
        # Update plot
        amp = np.abs(Psi_focusing[step])**2
        line.set_data(x, amp)
        time_text.set_text(f'Time: t = {times_focusing[step]:.3f}\nStep: {step}/{nt-1}')
        
        return line, time_text
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(anim_frames), 
                        interval=50, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    # Optionally save animation
    save_animation = False  # Set to True to save
    if save_animation:
        try:
            output_file = 'modulation_instability_animation.gif'
            print(f"Saving animation to '{output_file}'...")
            anim.save(output_file, writer='pillow', fps=20, dpi=100)
            print(f"Animation saved to '{output_file}'")
        except Exception as e:
            print(f"Could not save animation: {e}")
    
except ImportError as e:
    print(f"Visualization skipped (missing dependency): {e}")
except Exception as exc:  # pragma: no cover - visualization is optional
    print("Visualization error:", exc)
    import traceback
    traceback.print_exc()


