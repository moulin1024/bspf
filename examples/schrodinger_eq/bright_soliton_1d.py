from __future__ import annotations

import numpy as np
import numpy.typing as npt
import sys
import os

# Add src to path to use local bspf code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Import time steppers and Schrödinger solver from bspf library
from bspf import TimeStepperState, time_step, bspf1d

Array = npt.NDArray[np.complex128]

# ============================================================
# Analytical Solution
# ============================================================
def bright_soliton_analytical(x, t, A, v, x0, g, L=None, bc_type="neumann"):
    """
    Analytical bright soliton solution at time t with boundary condition handling.
    
    For the NLSE: i*∂ψ/∂t = -(1/2)*∂²ψ/∂x² + V*ψ + g*|ψ|²*ψ
    The bright soliton solution is:
        ψ(x,t) = A * sech(κ*(x - x_t)) * exp(i*(v*(x - x0) - omega*t))
    
    where:
        κ = A * sqrt(-g)  (width parameter)
        omega = v²/2 - A²*g/2  (frequency)
        x_t = x0 + v*t  (position at time t, with reflections for Neumann BCs)
    
    For Neumann BCs: Soliton reflects at boundaries (reflective)
    
    Args:
        x: Spatial grid
        t: Time
        A: Soliton amplitude
        v: Soliton velocity
        x0: Initial center position
        g: Nonlinearity parameter (must be < 0 for bright soliton)
        L: Domain length (required for Neumann BCs)
        bc_type: Boundary condition type ("neumann" or None for free space)
    
    Returns:
        Complex field ψ(x,t)
    """
    # Width parameter
    kappa = A * np.sqrt(-g)
    
    # Frequency: omega = v²/2 - A²*g/2
    # For the NLSE form i*∂ψ/∂t = -(1/2)*∂²ψ/∂x² + V*ψ + g*|ψ|²*ψ
    # The dispersion coefficient is p = 1/2
    omega = (v**2) / 2.0 - (A**2 * g) / 2.0
    
    if bc_type == "neumann" and L is not None:
        # Neumann BC: Soliton reflects at boundaries
        # Position at time t (unwrapped, can be outside [0, L])
        x_t_raw = x0 + v * t
        
        # Use method of images for reflective boundaries
        # The soliton and its images are at: x_t_raw, -x_t_raw, 2L - x_t_raw, 2L + x_t_raw, etc.
        d_direct = np.abs(x - x_t_raw)
        d_image_neg = np.abs(x - (-x_t_raw))  # Image at -x_t_raw
        d_image_2L_minus = np.abs(x - (2 * L - x_t_raw))  # Image at 2L - x_t_raw
        d_image_2L_plus = np.abs(x - (2 * L + x_t_raw))  # Image at 2L + x_t_raw
        d_image_neg2L_plus = np.abs(x - (-2 * L + x_t_raw))  # Image at -2L + x_t_raw
        
        # Take the minimum distance (shortest path to any image)
        dx_reflective = np.minimum.reduce([
            d_direct,
            d_image_neg,
            d_image_2L_minus,
            d_image_2L_plus,
            d_image_neg2L_plus
        ])
        
        # Soliton profile using reflective distance
        profile = A * np.cosh(kappa * dx_reflective)**(-1)
        
        # Phase factor: Need to determine current velocity direction
        # Count number of boundary crossings to determine if velocity flipped
        x_t_shifted = x_t_raw - x0
        x_t_periodic = x_t_shifted % (2 * L)
        if x_t_periodic < 0:
            x_t_periodic += 2 * L
        
        # Fold back to [0, L] to get effective position
        if x_t_periodic > L:
            x_t_effective = 2 * L - x_t_periodic
        else:
            x_t_effective = x_t_periodic
        x_t_effective = x_t_effective + x0
        
        # Ensure in [0, L]
        if x_t_effective < 0:
            x_t_effective = -x_t_effective
        if x_t_effective > L:
            x_t_effective = 2 * L - x_t_effective
        
        # Count boundary crossings to determine velocity direction
        n_crossings = int((x_t_raw - x0) / L) if L > 0 and v > 0 else 0
        if v < 0:
            n_crossings = int((x0 - x_t_raw) / L) if L > 0 else 0
        
        # Velocity flips after each boundary crossing
        v_current = v if n_crossings % 2 == 0 else -v
        
        # Phase factor: Use current velocity direction
        # The phase is: exp(i*(v*(x - x0) - omega*t))
        # But we need to account for velocity direction changes
        phase = np.exp(1j * (v_current * (x - x0) - omega * t))
        
    else:
        # Free space: Standard traveling soliton
        # Position at time t
        x_t = x0 + v * t
        # Soliton profile
        profile = A * np.cosh(kappa * (x - x_t))**(-1)
        # Phase factor
        phase = np.exp(1j * (v * (x - x0) - omega * t))
    
    return profile * phase

# ============================================================
# Control Parameters
# ============================================================
# Domain parameters
L_domain = 100  # domain length
nx = 1000              # grid points (including endpoints)

# Time parameters
T = 0.5             # final time
dt = 0.001             # time step

# BSPF parameters
degree = 5            # B-spline degree

# Bright soliton parameters
A = 1.0               # Soliton amplitude
v = 5.0               # Soliton velocity (0 for stationary soliton)
x0 = L_domain * 0.75            # Will be set to L_domain / 2.0 (soliton center)

# Boundary condition parameters
neumann_bc = (0.0, 0.0)  # (left_flux, right_flux) = (0, 0) for zero flux

# Potential (for linear Schrödinger, V = 0)
V_value = 0.0         # Potential value (constant potential)

# Nonlinearity parameter (must be < 0 for bright soliton)
g = 0.0               # Nonlinearity parameter (focusing NLS: g < 0)

# ============================================================
# Setup: Grid, Operator, Initial Condition
# ============================================================
# Set soliton center if not set
if x0 is None:
    x0 = L_domain * 0.9  # Center of domain

# Create spatial grid
x = np.linspace(0.0, L_domain, nx)
dx = x[1] - x[0]     # grid spacing

# Check for aliasing errors
nyquist_wavenumber = np.pi / dx / 2.0
if v > nyquist_wavenumber:
    print("="*60)
    print("WARNING: Potential aliasing detected!")
    print("="*60)
    print(f"  Soliton velocity v = {v:.3f}")
    print(f"  Grid spacing dx = {dx:.6f}")
    print(f"  Nyquist wavenumber = π/dx = {nyquist_wavenumber:.3f}")
    print(f"  Condition: v ({v:.3f}) > Nyquist ({nyquist_wavenumber:.3f})")
    print(f"\n  The phase factor exp(i*v*(x-x0)) creates oscillations")
    print(f"  with wavenumber k = v = {v:.3f}.")
    print(f"  For accurate resolution, v should be < π/dx = {nyquist_wavenumber:.3f}")
    print(f"  (ideally v < π/(2*dx) = {nyquist_wavenumber/2:.3f} for safety).")
    print(f"\n  Recommended: Increase nx to at least {int(np.ceil(2 * v * L_domain / np.pi))}")
    print(f"  or reduce velocity v to < {nyquist_wavenumber:.3f}")
    print("="*60)
    print()
    # Pause and ask for user confirmation
    response = input("Continue anyway? (y/n): ").strip().lower()
    if response != 'y':
        print("Simulation aborted by user.")
        sys.exit(0)
    print()
elif v > nyquist_wavenumber / 2:
    print("="*60)
    print("WARNING: Marginal resolution for traveling soliton")
    print("="*60)
    print(f"  Soliton velocity v = {v:.3f}")
    print(f"  Grid spacing dx = {dx:.6f}")
    print(f"  Nyquist wavenumber = π/dx = {nyquist_wavenumber:.3f}")
    print(f"  Condition: v ({v:.3f}) > π/(2*dx) = {nyquist_wavenumber/2:.3f}")
    print(f"\n  For better accuracy, consider:")
    print(f"  - Increasing nx to at least {int(np.ceil(2 * v * L_domain / np.pi))}")
    print(f"  - Or reducing velocity v to < {nyquist_wavenumber/2:.3f}")
    print("="*60)
    print()

# Soliton width parameter: κ = A * sqrt(-g)
kappa = A * np.sqrt(-g)

# Print parameters
print("="*60)
print("Control Parameters:")
print("="*60)
print(f"Domain: L = {L_domain:.3f}, nx = {nx}, dx = {dx:.6f}")
print(f"Time: T = {T:.3f}, dt = {dt:.6f}, dx² = {dx**2:.6f}")
print(f"BSPF: degree = {degree}")
print(f"Initial condition: Bright soliton")
print(f"  Amplitude: A = {A:.3f}")
print(f"  Velocity: v = {v:.3f}")
print(f"  Center: x0 = {x0:.3f}")
print(f"  Width parameter: κ = {kappa:.3f}")
print(f"Boundary conditions: Neumann (zero flux)")
print(f"Potential: V = {V_value:.3f}, Nonlinearity: g = {g:.3f} (focusing)")
print("="*60)

# Number of time steps
nt = int(T / dt) + 1

# Create BSPF operator
bf = bspf1d.from_grid(degree=degree, x=x, n_basis=20, use_clustering=True, clustering_factor=2.0)

# Initial condition: Bright soliton
# For traveling soliton: ψ(x,0) = A * sech(κ*(x - x0)) * exp(i*v*(x - x0))
# where κ = A * sqrt(-g) is the soliton width parameter
# The phase factor exp(i*v*(x - x0)) gives the soliton its velocity
envelope = A * (1.0 / np.cosh(kappa * (x - x0)))
phase = np.exp(1j * v * (x - x0))  # Phase factor for traveling soliton
psi0 = (envelope * phase).astype(np.complex128)

# Enforce zero-flux Neumann BCs on initial condition
# Use order matching the BSPF degree (or closest available: 1, 2, 3, 4, or 5)
bc_order = 2  # Cap at 5th order (highest implemented)
print(f"  Using {bc_order}-th order finite difference for Neumann BC enforcement")
# psi0 = enforce_zero_flux_neumann_bc(psi0, dx, order=bc_order)

# Create potential array
V = np.full(nx, V_value, dtype=np.float64)

# ============================================================
# Internal linear Schrödinger solver using real BSPF
# Boundary conditions are enforced through bf.differentiate() with neumann_bc parameter
# ============================================================
def create_linear_schrodinger_rhs(bf_op, V_array, neumann_bc):
    """
    Create RHS function for linear Schrödinger equation using real BSPF.
    
    The equation: i*∂ψ/∂t = -(1/2)*∂²ψ/∂x² + V*ψ
    
    This function treats real and imaginary parts separately using real BSPF.
    Boundary conditions are enforced through bf.differentiate() with neumann_bc parameter.
    
    Parameters
    ----------
    bf_op : bspf1d
        BSPF operator (works with real arrays)
    V_array : array
        Potential V(x)
    neumann_bc : tuple
        Neumann boundary conditions (left_flux, right_flux)
    
    Returns
    -------
    rhs_func : callable
        RHS function with signature: rhs_func(psi) -> dpsi_dt
    """
    def rhs_func(psi: Array) -> Array:
        """
        Compute RHS of linear Schrödinger equation.
        
        Parameters
        ----------
        psi : complex array
            Complex wavefunction
        
        Returns
        -------
        dpsi_dt : complex array
            Time derivative: dpsi_dt = -i * [-(1/2)*∂²ψ/∂x² + V*ψ]
        """
        # Split into real and imaginary parts
        psi_real = np.real(psi)
        psi_imag = np.imag(psi)
        
        # Compute second derivative using BSPF differentiate (enforces Neumann BCs)
        # bf.differentiate() with neumann_bc parameter enforces BCs during differentiation
        d2psi_real_dx2, _ = bf_op.differentiate(psi_real, k=2, neumann_bc=neumann_bc)
        d2psi_imag_dx2, _ = bf_op.differentiate(psi_imag, k=2, neumann_bc=neumann_bc)
        
        # Combine back to complex
        d2psi_dx2 = d2psi_real_dx2 + 1j * d2psi_imag_dx2
        
        # Linear part: -(1/2)*∂²ψ/∂x² + V*ψ
        linear_part = -0.5 * d2psi_dx2 + V_array * psi
        
        # RHS: dψ/dt = -i * linear_part
        # For i*∂ψ/∂t = linear_part, we have ∂ψ/∂t = -i * linear_part
        dpsi_dt = -1j * linear_part
        
        return dpsi_dt
    
    return rhs_func


def create_nls_rhs(bf_op, V_array, g, neumann_bc):
    """
    Create RHS function for nonlinear Schrödinger equation using real BSPF.
    
    The equation: i*∂ψ/∂t = -(1/2)*∂²ψ/∂x² + V*ψ + g*|ψ|²*ψ
    
    This function treats real and imaginary parts separately using real BSPF.
    Boundary conditions are enforced through bf.differentiate() with neumann_bc parameter.
    
    Parameters
    ----------
    bf_op : bspf1d
        BSPF operator (works with real arrays)
    V_array : array
        Potential V(x)
    g : float
        Nonlinearity parameter
    neumann_bc : tuple
        Neumann boundary conditions (left_flux, right_flux)
    
    Returns
    -------
    rhs_func : callable
        RHS function with signature: rhs_func(psi) -> dpsi_dt
    """
    def rhs_func(psi: Array) -> Array:
        """
        Compute RHS of nonlinear Schrödinger equation.
        
        Parameters
        ----------
        psi : complex array
            Complex wavefunction
        
        Returns
        -------
        dpsi_dt : complex array
            Time derivative: dpsi_dt = -i * [-(1/2)*∂²ψ/∂x² + V*ψ + g*|ψ|²*ψ]
        """
        # Split into real and imaginary parts
        psi_real = np.real(psi)
        psi_imag = np.imag(psi)
        
        # Compute second derivative using BSPF differentiate (enforces Neumann BCs)
        # bf.differentiate() with neumann_bc parameter enforces BCs during differentiation
        d2psi_real_dx2, _ = bf_op.differentiate(psi_real, k=2, neumann_bc=neumann_bc)
        d2psi_imag_dx2, _ = bf_op.differentiate(psi_imag, k=2, neumann_bc=neumann_bc)
        
        # Combine back to complex
        d2psi_dx2 = d2psi_real_dx2 + 1j * d2psi_imag_dx2
        
        # Linear part: -(1/2)*∂²ψ/∂x² + V*ψ
        linear_part = -0.5 * d2psi_dx2 + V_array * psi
        
        # Nonlinear part: g*|ψ|²*ψ
        nonlinear_part = g * np.abs(psi)**2 * psi
        
        # Total RHS: linear + nonlinear
        total_rhs = linear_part + nonlinear_part
        
        # For i*∂ψ/∂t = total_rhs, we have ∂ψ/∂t = -i * total_rhs
        dpsi_dt = -1j * total_rhs
        
        return dpsi_dt
    
    return rhs_func

# Create RHS function using internal solver
# Nonlinear Schrödinger: i*∂ψ/∂t = -(1/2)*∂²ψ/∂x² + V*ψ + g*|ψ|²*ψ
# For bright soliton: g < 0 (focusing case)
# Boundary conditions are enforced through bf.differentiate() with neumann_bc parameter
rhs_func = create_nls_rhs(bf, V, g, neumann_bc)
print("  Created NLS RHS function using internal solver (real BSPF, focusing case: g < 0)")
print("  Neumann BCs enforced through bf.differentiate() with neumann_bc parameter")

# ---------- Energy computation function ----------
def compute_energy(psi, bf_op, V, g):
    """
    Compute energy (Hamiltonian) for nonlinear Schrödinger equation.
    
    Energy: E = ∫ [ (1/2)|∇ψ|² + V|ψ|² + (g/2)|ψ|⁴ ] dx
    
    Parameters
    ----------
    psi : complex array
        Wavefunction
    bf_op : bspf1d
        BSPF operator for differentiation and integration
    V : array
        Potential V(x)
    g : float
        Nonlinearity parameter
    
    Returns
    -------
    E : float
        Total energy
    """
    # Compute first derivative using real BSPF
    # Split into real and imaginary parts
    psi_real = np.real(psi)
    psi_imag = np.imag(psi)
    
    # Compute derivatives separately
    dpsi_real_dx, _ = bf_op.differentiate(psi_real, k=1, neumann_bc=neumann_bc)
    dpsi_imag_dx, _ = bf_op.differentiate(psi_imag, k=1, neumann_bc=neumann_bc)
    
    # Combine back to complex
    dpsi_dx = dpsi_real_dx + 1j * dpsi_imag_dx
    
    # Kinetic energy density: (1/2)|∂ψ/∂x|²
    kinetic_density = 0.5 * np.abs(dpsi_dx)**2
    
    # Potential energy density: V|ψ|²
    potential_density = V * np.abs(psi)**2
    
    # Nonlinear energy density: (g/2)|ψ|⁴
    nonlinear_density = (g / 2.0) * np.abs(psi)**4
    
    # Total energy density
    energy_density = kinetic_density + potential_density + nonlinear_density
    
    # Integrate using BSPF
    E = bf_op.definite_integral(energy_density)
    
    return E

# ---------- Time integration with RK45 ----------
print("\n" + "="*60)
print("Running BSPF-RK45 time integration...")
print("="*60)
print("  Using internal linear Schrödinger solver (real BSPF)")
print("  Time stepping: RK45 (adaptive)")
print("  Neumann BCs enforced through bf.differentiate() with neumann_bc parameter")

# Compute initial energy
energy_initial = compute_energy(psi0, bf, V, g)
print(f"Initial energy: E₀ = {energy_initial:.10e}")

# Track energy over time
energies = np.zeros(nt, dtype=np.float64)
energies[0] = energy_initial

# Track L∞ error over time
error_linf_over_time = np.zeros(nt, dtype=np.float64)
# Initial error should be zero (or very small due to BC enforcement)
psi_analytical_0 = bright_soliton_analytical(x, 0.0, A, v, x0, g, L=L_domain, bc_type="neumann")
density_num_0 = np.abs(psi0)**2
density_ana_0 = np.abs(psi_analytical_0)**2
error_linf_over_time[0] = np.max(np.abs(density_num_0 - density_ana_0))

with TimeStepperState(psi0.copy(), t_init=0.0, dt=dt, method='rk45', t_final=T, show_progress=True) as state:
    Psi = np.empty((nt, nx), dtype=np.complex128)
    Psi[0] = psi0.copy()
    times = np.zeros(nt)
    times[0] = 0.0

    for step in range(1, nt):
        psi_next = time_step(state, dt, rhs_func, method='rk45')
        psi = state.get_current()
        
        # Note: Neumann BCs are enforced through bf.differentiate() with neumann_bc parameter
        # No explicit BC enforcement needed
        state.psi_now = psi.copy()
        
        Psi[step] = psi.copy()
        times[step] = state.get_current_time()
        
        # Compute energy at this time step
        energies[step] = compute_energy(psi, bf, V, g)
        
        # Compute L∞ error at this time step
        psi_analytical_t = bright_soliton_analytical(x, times[step], A, v, x0, g, L=L_domain, bc_type="neumann")
        density_num_t = np.abs(psi)**2
        density_ana_t = np.abs(psi_analytical_t)**2
        error_linf_over_time[step] = np.max(np.abs(density_num_t - density_ana_t))

print("Simulation completed successfully!")

# ---------- Energy conservation analysis ----------
energy_final = energies[-1]
energy_error_abs = np.abs(energies - energy_initial)
energy_error_max = np.max(energy_error_abs)
energy_error_relative = energy_error_max / np.abs(energy_initial)

print("\n" + "="*60)
print("Energy Conservation Analysis:")
print("="*60)
print(f"Initial energy: E₀ = {energy_initial:.10e}")
print(f"Final energy:   E_T = {energy_final:.10e}")
print(f"Energy change:  ΔE = {energy_final - energy_initial:.10e}")
print(f"Max |ΔE|:       {energy_error_max:.10e}")
print(f"Relative error: {energy_error_relative:.6e}")
print("="*60)

# ---------- Error computation ----------
# Compare final solution with analytical solution at final time
t_final = times[-1]

# Compute analytical solution at final time
psi_analytical_final = bright_soliton_analytical(x, t_final, A, v, x0, g, L=L_domain, bc_type="neumann")

# Compute error in |ψ|² (density)
density_numerical = np.abs(Psi[-1])**2
density_analytical = np.abs(psi_analytical_final)**2
error_density_abs = np.abs(density_numerical - density_analytical)
error_density_linf = np.max(error_density_abs)
error_density_l2 = np.sqrt(np.sum(error_density_abs**2) * dx)

print("\n" + "="*60)
print("Error Analysis (Numerical vs Analytical):")
print("="*60)
print(f"Final time: t = {t_final:.6f} s")
print(f"Soliton traveled distance: v*T = {v * t_final:.6f}")
print(f"Initial soliton center: x = {x0:.6f}")
print(f"\nDensity error (||ψ|²_numerical - |ψ|²_analytical|):")
print(f"  Max error (L∞):  {error_density_linf:.6e}")
print(f"  L2 error:        {error_density_l2:.6e}")
print("="*60)

# ---------- BC check ----------
# Check Neumann BCs using real BSPF
psi_final_real = np.real(Psi[-1])
psi_final_imag = np.imag(Psi[-1])
dpsi_real_dx, _, _ = bf.differentiate_1_2(psi_final_real, neumann_bc=(0.0, 0.0))
dpsi_imag_dx, _, _ = bf.differentiate_1_2(psi_final_imag, neumann_bc=(0.0, 0.0))
dpsi_dx = dpsi_real_dx + 1j * dpsi_imag_dx
print(f"\nNeumann BC check (final time):")
print(f"  |ψ_x(0)|={np.abs(dpsi_dx[0]): .3e}, |ψ_x(L)|={np.abs(dpsi_dx[-1]): .3e}")
print(f"  (Note: BCs are enforced through bf.differentiate() with neumann_bc parameter)")

# ---------- Visualization ----------
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: |ψ|² at initial, final (numerical), and final (analytical)
    ax1 = axes[0, 0]
    ax1.plot(x, np.abs(Psi[0])**2, label="t=0 s (initial)", linewidth=2)
    ax1.plot(x, np.abs(Psi[-1])**2, label=f"t={T} s (numerical)", linewidth=2, linestyle='-')
    ax1.plot(x, np.abs(psi_analytical_final)**2, label=f"t={T} s (analytical)", 
             linewidth=2, linestyle='--', alpha=0.7)
    ax1.set_xlabel("x")
    ax1.set_ylabel("|ψ|²")
    ax1.set_title("Bright Soliton: |ψ|² Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Space-time plot of |ψ|² evolution
    ax2 = axes[0, 1]
    im = ax2.imshow(np.abs(Psi)**2, aspect='auto', 
                    extent=[0, L_domain, 0, T], origin='lower',
                    cmap='plasma', interpolation='bilinear')
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.set_title("Bright Soliton: |ψ|² Evolution (Space-Time)")
    plt.colorbar(im, ax=ax2, label="|ψ|²")
    
    # Plot 3: L∞ error over time
    ax3 = axes[1, 0]
    ax3.plot(times, error_linf_over_time, 'r-', linewidth=2, label="L∞ error")
    ax3.set_xlabel("Time t")
    ax3.set_ylabel("L∞ Error")
    ax3.set_title("L∞ Error Over Time")
    ax3.set_yscale('log')  # Log scale to show error evolution
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Energy error over time
    ax4 = axes[1, 1]
    energy_error = energies - energy_initial
    ax4.plot(times, energy_error, 'r-', linewidth=2)
    ax4.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel("Time t")
    ax4.set_ylabel("Energy Error ΔE = E(t) - E₀")
    ax4.set_title("Energy Conservation Error")
    ax4.set_yscale('symlog', linthresh=1e-12)  # Symmetric log scale to show small errors
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
except Exception as exc:  # pragma: no cover - visualization is optional
    print("Plot skipped:", exc)

