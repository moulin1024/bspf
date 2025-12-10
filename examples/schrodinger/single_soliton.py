#!/usr/bin/env python3
"""
Test single soliton case with analytical solution
For NLSE: iE_t + p*E_xx + (V + q*|E|^2)E = 0

The single soliton is an exact traveling wave solution:
E(x,t) = sqrt(2)*a0*sech(a0*(x - x0 - v0*t)) * exp(i*(v0*x/2 - omega*t))

where omega = v0^2/4 - a0^2 (for p=1, q=1)

Uses RK4 time integration from time_steppers.py with BSPF (7th order) 
for spatial derivatives and Neumann boundary conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from bspf import bspf1d
from bspf.utils.time_steppers import rk4_step

# ============================================================================
# Parameters (defined at top of program)
# ============================================================================
# NLSE parameters
P = 1.0  # Dispersion coefficient
Q = 1.0  # Nonlinearity coefficient
V = 0.0  # External potential

# Soliton parameters
A0 = 1.0  # Soliton amplitude
V0 = 2.0  # Soliton velocity
X0_FRAC = 0.3  # Initial position (fraction of domain length)

# Domain and grid parameters
L = 100.0  # Domain length
DX = None  # Spatial step (None = auto: 0.001*L)
DT = 0.001  # Time step
T = 10.0  # Total simulation time

# BSPF parameters
BSPF_DEGREE = 7  # B-spline degree (7th order)

# Plotting parameters
PLOT_INTERVAL = 100  # Plot progress every N steps


def onesoliton_analytical(x, t, a0, v0, x0, p=1.0, q=1.0):
    """
    Analytical single soliton solution at time t.
    
    E(x,t) = sqrt(2)*a0*sech(a0*(x - x0 - v0*t)) * exp(i*(v0*x/2 - omega*t))
    where omega = v0^2/4 - a0^2
    
    Args:
        x: Spatial grid
        t: Time
        a0: Amplitude parameter
        v0: Velocity parameter
        x0: Initial center position
        p: Dispersion coefficient (default: 1.0)
        q: Nonlinearity coefficient (default: 1.0)
    
    Returns:
        Complex field E(x,t)
    """
    # Phase frequency: omega = v0^2/(4*p) - a0^2*q
    omega = (v0**2) / (4 * p) - a0**2 * q
    # Position at time t
    x_t = x0 + v0 * t
    # Soliton profile
    profile = np.sqrt(2) * a0 * np.cosh(a0 * (x - x_t))**(-1)
    # Phase factor
    phase = np.exp(1j * (v0 * x / 2 - omega * t))
    return profile * phase


def nlseqn_bspf(E, p, q, bspf_op, V=0.0):
    """
    Compute RHS of NLSE using BSPF for spatial derivatives with Neumann BCs.
    
    Equation: E_t = i*p*E_xx + i*(V + q*|E|^2)E
    
    Uses bspf1d for high-accuracy spatial derivatives.
    Enforces Neumann BC: E'(0) = E'(L) = 0
    
    Args:
        E: Field array (complex)
        p: Dispersion coefficient
        q: Nonlinearity coefficient
        bspf_op: bspf1d operator for computing derivatives
        V: External potential (default: 0)
    
    Returns:
        dE/dt (time derivative)
    """
    # Neumann BC: E'(0) = E'(L) = 0
    # bspf1d.differentiate supports complex arrays directly
    E_xx, _ = bspf_op.differentiate(E, k=2, neumann_bc=(0.0, 0.0))
    
    # RHS
    dEdt = 1j * p * E_xx + 1j * (V + q * np.abs(E)**2) * E
    
    # Enforce Neumann BC on dE/dt
    dEdt[0] = dEdt[1]
    dEdt[-1] = dEdt[-2]
    
    return dEdt


def main():
    # Setup grid (Neumann BCs: include boundaries)
    if DX is None:
        dx = 0.001 * L
    else:
        dx = DX
    
    # For Neumann BCs, include boundaries
    x = np.arange(0, L + dx, dx)
    nx = len(x)
    dx = x[1] - x[0]  # Actual dx
    
    # BSPF operator (7th order)
    print(f"  Creating bspf1d operator (nx={nx}, dx={dx:.6f}, degree={BSPF_DEGREE})...")
    bspf_op = bspf1d.from_grid(
        degree=BSPF_DEGREE,
        x=x,
        use_gpu=False
    )
    
    # Soliton parameters
    a0 = A0
    v0 = V0
    x0 = X0_FRAC * L
    
    # Initial condition (analytical solution at t=0)
    E_numerical = onesoliton_analytical(x, 0.0, a0, v0, x0, P, Q)
    
    E_0 = E_numerical.copy()
    
    # Time stepping
    dt = DT
    nsteps = int(T / dt)
    t = 0.0
    
    print("=" * 80)
    print("Single Soliton Test (with Analytical Solution)")
    print("=" * 80)
    print(f"  Method: RK4 + BSPF (degree={BSPF_DEGREE})")
    print(f"  Parameters: p={P}, q={Q}")
    print(f"  Soliton: a0={a0}, v0={v0}, x0={x0:.2f}")
    print(f"  Grid: nx={nx}, dx={dx:.6f}, L={L}")
    print(f"  Time: dt={dt:.6f}, T={T}, nsteps={nsteps}")
    print(f"  BC: Neumann (E'(0) = E'(L) = 0)")
    print("-" * 80)
    print("Starting simulation...")
    
    # Storage for space-time plots
    times_to_compare = [0.0, T/4, T/2, 3*T/4, T]
    comparison_data = []
    
    # Storage for space-time evolution (every PLOT_INTERVAL steps)
    time_history = []
    density_history = []  # |E|²
    error_history = []    # |E_num - E_ana|
    
    for ii in range(nsteps):
        # Time step using RK4 from time_steppers.py
        E_numerical = rk4_step(E_numerical, dt, nlseqn_bspf, P, Q, bspf_op, V)
        t += dt
        
        # Store data for space-time plots at every PLOT_INTERVAL step
        if (ii + 1) % PLOT_INTERVAL == 0 or ii == 0:
            E_analytical = onesoliton_analytical(x, t, a0, v0, x0, P, Q)
            error = np.abs(E_numerical - E_analytical)
            
            time_history.append(t)
            density_history.append(np.abs(E_numerical)**2)
            error_history.append(error)
            
            max_error = np.max(error)
            l2_error = np.sqrt(np.sum(error**2) * dx)
            print(f"  Step {ii+1}/{nsteps} (t={t:.4f}): max_error={max_error:.6e}, L2_error={l2_error:.6e}")
        
        # Check if we should compare at this time (for detailed comparison)
        if any(abs(t - t_comp) < dt/2 for t_comp in times_to_compare):
            E_analytical = onesoliton_analytical(x, t, a0, v0, x0, P, Q)
            
            # Compute error
            error = np.abs(E_numerical - E_analytical)
            max_error = np.max(error)
            l2_error = np.sqrt(np.sum(error**2) * dx)
            
            comparison_data.append({
                't': t,
                'E_num': E_numerical.copy(),
                'E_ana': E_analytical.copy(),
                'max_error': max_error,
                'l2_error': l2_error
            })
            
            print(f"  t={t:.4f}: max_error={max_error:.6e}, L2_error={l2_error:.6e}")
    
    print("-" * 80)
    print("Simulation complete!")
    print("=" * 80)
    
    # Convert history to arrays for plotting
    time_history = np.array(time_history)
    density_history = np.array(density_history)  # Shape: (n_times, n_x)
    error_history = np.array(error_history)      # Shape: (n_times, n_x)
    
    # Create space-time plots
    print("\nCreating space-time plots...")
    
    # Space-time plot 1: Density |E|²
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    X, T_mesh = np.meshgrid(x, time_history)
    im1 = ax1.pcolormesh(X, T_mesh, density_history, shading='gouraud', cmap='viridis')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('t', fontsize=12)
    ax1.set_title('Space-Time Evolution: |E|² (Numerical)', fontsize=14)
    plt.colorbar(im1, ax=ax1, label='|E|²')
    plt.tight_layout()
    plt.savefig('single_soliton_spacetime_density.png', dpi=150, bbox_inches='tight')
    print(f"Saved space-time density plot: single_soliton_spacetime_density.png")
    
    # Space-time plot 2: Error |E_num - E_ana|
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    im2 = ax2.pcolormesh(X, T_mesh, error_history, shading='gouraud', cmap='hot')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('t', fontsize=12)
    ax2.set_title('Space-Time Error: |E_num - E_ana|', fontsize=14)
    plt.colorbar(im2, ax=ax2, label='|Error|')
    ax2.set_yscale('linear')
    plt.tight_layout()
    plt.savefig('single_soliton_spacetime_error.png', dpi=150, bbox_inches='tight')
    print(f"Saved space-time error plot: single_soliton_spacetime_error.png")
    
    # Space-time plot 3: Error (log scale)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    # Use log scale for better visualization of small errors
    error_log = np.log10(error_history + 1e-15)  # Add small value to avoid log(0)
    im3 = ax3.pcolormesh(X, T_mesh, error_log, shading='gouraud', cmap='hot')
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('t', fontsize=12)
    ax3.set_title('Space-Time Error: log₁₀(|E_num - E_ana|)', fontsize=14)
    plt.colorbar(im3, ax=ax3, label='log₁₀(|Error|)')
    plt.tight_layout()
    plt.savefig('single_soliton_spacetime_error_log.png', dpi=150, bbox_inches='tight')
    print(f"Saved space-time error (log) plot: single_soliton_spacetime_error_log.png")
    
    # Create comparison plots
    n_comparisons = len(comparison_data)
    fig, axes = plt.subplots(2, n_comparisons, figsize=(5*n_comparisons, 10))
    if n_comparisons == 1:
        axes = axes.reshape(2, 1)
    
    for idx, data in enumerate(comparison_data):
        t = data['t']
        E_num = data['E_num']
        E_ana = data['E_ana']
        error = np.abs(E_num - E_ana)
        
        # Top row: Comparison of solutions
        ax = axes[0, idx]
        ax.plot(x, np.abs(E_num)**2, 'b-', label='Numerical |E|²', linewidth=2)
        ax.plot(x, np.abs(E_ana)**2, 'r--', label='Analytical |E|²', linewidth=2, alpha=0.7)
        ax.set_xlabel('x')
        ax.set_ylabel('|E|²')
        ax.set_title(f't = {t:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, L)
        
        # Bottom row: Error
        ax = axes[1, idx]
        ax.plot(x, error, 'k-', linewidth=1.5)
        ax.set_xlabel('x')
        ax.set_ylabel('|Error|')
        ax.set_title(f'Error (max={data["max_error"]:.2e})')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, L)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('single_soliton_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot: single_soliton_comparison.png")
    
    # Summary plot: evolution over time (at comparison times)
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    for idx, data in enumerate(comparison_data):
        t_comp = data['t']
        E_num = data['E_num']
        ax4.plot(x, np.abs(E_num)**2, label=f't={t_comp:.2f}', linewidth=2, alpha=0.8)
    ax4.set_xlabel('x')
    ax4.set_ylabel('|E|²')
    ax4.set_title('Soliton Evolution at Comparison Times (Numerical)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, L)
    plt.tight_layout()
    plt.savefig('single_soliton_evolution.png', dpi=150, bbox_inches='tight')
    print(f"Saved evolution plot: single_soliton_evolution.png")
    
    # Print error summary
    print("\n" + "=" * 80)
    print("Error Summary")
    print("=" * 80)
    print(f"{'Time':<10} {'Max Error':<15} {'L2 Error':<15}")
    print("-" * 80)
    for data in comparison_data:
        print(f"{data['t']:<10.4f} {data['max_error']:<15.6e} {data['l2_error']:<15.6e}")
    print("=" * 80)
    
    plt.show()


if __name__ == '__main__':
    main()
