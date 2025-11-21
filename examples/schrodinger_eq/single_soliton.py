#!/usr/bin/env python3
"""
Test single soliton case with analytical solution
For NLSE: iE_t + p*E_xx + (V + q*|E|^2)E = 0

The single soliton is an exact traveling wave solution:
E(x,t) = sqrt(2)*a0*sech(a0*(x - x0 - v0*t)) * exp(i*(v0*x/2 - omega*t))

where omega = v0^2/4 - a0^2 (for p=1, q=1)

Uses rk45 + BSPF: 4th order time integration with B-spline spatial derivatives

Boundary conditions:
- Dirichlet: E(0) = E(L) = 0
- Neumann: E'(0) = E'(L) = 0
"""

import numpy as np
import matplotlib.pyplot as plt
try:
    from bspf import bspf1d
    from bspf.utils import TimeStepperState, time_step
    BSPF_AVAILABLE = True
except ImportError:
    BSPF_AVAILABLE = False
    print("Warning: bspf library not available. BSPF method will be disabled.")
    # Define dummy classes for type hints
    TimeStepperState = None
    time_step = None


def onesoliton_analytical(x, t, a0, v0, x0, p=1.0, q=1.0, L=None, bc_type="periodic"):
    """
    Analytical single soliton solution at time t with boundary condition handling.
    
    For periodic BCs: Soliton wraps around the domain
    For Neumann BCs: Soliton reflects at boundaries (reflective)
    For Dirichlet BCs: Standard traveling soliton (may not satisfy BCs exactly)
    
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
        L: Domain length (required for periodic/Neumann BCs)
        bc_type: Boundary condition type ("periodic", "neumann", or "dirichlet")
    
    Returns:
        Complex field E(x,t)
    """
    # Phase frequency: omega = v0^2/(4*p) - a0^2*q
    omega = (v0**2) / (4 * p) - a0**2 * q
    
    if bc_type == "periodic" and L is not None:
        # Periodic BC: Soliton wraps around the domain
        # Position at time t (unwrapped, can be outside [0, L))
        x_t_raw = x0 + v0 * t
        
        # Handle modulo for periodic wrapping
        # Ensure x_t is in [0, L) range for profile computation
        x_t = x_t_raw % L
        if x_t < 0:
            x_t += L
        
        # For periodic BCs, use periodic distance function
        # The periodic distance between x and x_t is:
        # d_periodic(x, x_t) = min(|x - x_t|, L - |x - x_t|)
        # This naturally handles wrapping without needing to sum over images
        dx = np.abs(x - x_t)
        dx_periodic = np.minimum(dx, L - dx)
        
        # Soliton profile using periodic distance
        # The periodic distance already handles wrapping correctly
        profile = np.sqrt(2) * a0 * np.cosh(a0 * dx_periodic)**(-1)
        
        # Note: The periodic distance function min(|x-x_t|, L-|x-x_t|) correctly handles
        # the case when the soliton wraps around. When x_t=0, points near x=L will have
        # small periodic distance (wrapping around), so the soliton appears on both sides.
        # This is the correct behavior for periodic BCs.
        
        # Phase factor: For periodic BCs, the phase must be periodic
        # The phase should be computed using the same periodic distance logic as the profile
        # This ensures E(0) = E(L) for periodic BCs
        
        # Compute phase using periodic distance with correct sign
        # Use the same logic as the profile: find the signed periodic distance
        dx_direct = x - x_t  # Direct distance (can be negative)
        
        # For periodic BCs, wrap to [-L/2, L/2] range
        # This ensures phase continuity: when x_t=0 and x=L-dx, dx_phase ≈ -dx
        dx_phase = dx_direct.copy()
        
        # Wrap distances > L/2 to the left
        mask_left = dx_phase > L / 2
        dx_phase[mask_left] = dx_phase[mask_left] - L
        
        # Wrap distances < -L/2 to the right
        mask_right = dx_phase < -L / 2
        dx_phase[mask_right] = dx_phase[mask_right] + L
        
        # Phase offset: v0 * dx_phase / 2
        # This ensures the phase is periodic and continuous at boundaries
        phase_offset = v0 * dx_phase / 2
        phase = np.exp(1j * (phase_offset - omega * t))
        
    elif bc_type == "neumann" and L is not None:
        # Neumann BC: Soliton reflects at boundaries (reflective)
        # Position at time t (unwrapped, can be outside [0, L])
        x_t_raw = x0 + v0 * t
        
        # For reflective BCs, use method of images
        # The reflective distance is the minimum distance to any image of x_t_raw
        # Images are at: x_t_raw, -x_t_raw, 2L - x_t_raw, 2L + x_t_raw, -2L + x_t_raw, ...
        # In closed form, we can compute this using the unfolded trajectory
        
        # Compute distances to the nearest images (sufficient for localized soliton)
        # Image positions in the unfolded space: ..., -2L + x_t_raw, -x_t_raw, x_t_raw, 2L - x_t_raw, 2L + x_t_raw, ...
        d_direct = np.abs(x - x_t_raw)
        d_image_neg = np.abs(x + x_t_raw)  # Image at -x_t_raw
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
        profile = np.sqrt(2) * a0 * np.cosh(a0 * dx_reflective)**(-1)
        
        # Phase factor: Need to determine current velocity direction
        # Count number of boundary crossings to determine if velocity flipped
        # Each time the soliton crosses x=0 or x=L, velocity flips
        
        # Calculate the effective position in [0, L] after reflections
        # This is used to determine the velocity direction
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
        n_crossings = int((x_t_raw - x0) / L) if L > 0 and v0 > 0 else 0
        if v0 < 0:
            n_crossings = int((x0 - x_t_raw) / L) if L > 0 else 0
        
        # Velocity flips after each boundary crossing
        v_current = v0 if n_crossings % 2 == 0 else -v0
        
        # Phase factor: Use current velocity direction
        phase = np.exp(1j * (v_current * x / 2 - omega * t))
        
    else:
        # Dirichlet or no BC handling: Standard traveling soliton
        # Position at time t
        x_t = x0 + v0 * t
        # Soliton profile
        profile = np.sqrt(2) * a0 * np.cosh(a0 * (x - x_t))**(-1)
        # Phase factor
        phase = np.exp(1j * (v0 * x / 2 - omega * t))
    
    return profile * phase




def nlseqn_bspf(E, p, q, bspf_op, V=0.0, bc_type="neumann"):
    """
    Compute RHS of NLSE using BSPF for spatial derivatives.
    
    Equation: E_t = i*p*E_xx + i*(V + q*|E|^2)E
    
    Uses bspf1d for high-accuracy spatial derivatives.
    For periodic BCs, falls back to finite differences.
    
    Args:
        E: Field array (complex)
        p: Dispersion coefficient
        q: Nonlinearity coefficient
        bspf_op: bspf1d operator for computing derivatives
        V: External potential (default: 0)
        bc_type: Boundary condition type ("periodic", "dirichlet", or "neumann")
    
    Returns:
        dE/dt (time derivative)
    """
    n = len(E)
    
    if bc_type == "periodic":
        # For periodic, use finite differences (bspf1d doesn't support periodic BCs)
        dx = bspf_op.grid.x[1] - bspf_op.grid.x[0]
        E_padded = np.concatenate([E[-2:], E, E[:2]])
        E_xx = (E_padded[3:n+3] - 2*E_padded[2:n+2] + E_padded[1:n+1]) / (dx**2)
        E_interior = E_padded[2:n+2]
        dEdt = 1j * p * E_xx + 1j * (V + q * np.abs(E_interior)**2) * E_interior
        
    elif bc_type == "neumann":
        # Neumann BC: E'(0) = E'(L) = 0
        E_real_work = np.real(E)
        E_imag_work = np.imag(E)

        f_left_real, f_right_real = bspf_op.enforced_zero_flux(E_real_work)
        f_left_imag, f_right_imag = bspf_op.enforced_zero_flux(E_imag_work)

        E_real_work[0] = f_left_real
        E_real_work[-1] = f_right_real
        E_imag_work[0] = f_left_imag
        E_imag_work[-1] = f_right_imag
        
        # Compute second derivative with Neumann BC enforcement
        E_xx_real, _ = bspf_op.differentiate(E_real_work, k=2, neumann_bc=(0.0, 0.0))
        E_xx_imag, _ = bspf_op.differentiate(E_imag_work, k=2, neumann_bc=(0.0, 0.0))
        E_xx = E_xx_real + 1j * E_xx_imag
        
        # RHS
        dEdt = 1j * p * E_xx + 1j * (V + q * np.abs(E)**2) * E
        
        # Enforce zero-flux Neumann BCs using enforced_zero_flux for real and imaginary parts
        # dEdt_real = np.real(dEdt)
        # dEdt_imag = np.imag(dEdt)
        
        # Get corrected boundary values (returns tuple: (left, right))
        # f_left_real, f_right_real = bspf_op.enforced_zero_flux(dEdt_real)
        # f_left_imag, f_right_imag = bspf_op.enforced_zero_flux(dEdt_imag)
        
        # Apply corrections to boundaries
        # dEdt_real[0] = #f_left_real
        # dEdt_real[-1] = #f_right_real
        # dEdt_imag[0] = #f_left_imag
        # dEdt_imag[-1] = #f_right_imag
        
        # Recombine to complex
        # dEdt = dEdt_real + 1j * dEdt_imag
        
    else:
        raise ValueError(f"Unknown boundary condition type: {bc_type}")
    
    return dEdt


def compute_energy_bspf(E, p, q, bspf_op, dx, V=0.0, bc_type="neumann"):
    """
    Compute energy (Hamiltonian) for NLSE using BSPF.
    
    For equation: iE_t + p*E_xx + (V + q*|E|^2)E = 0
    Energy: H = ∫ [ (p/2)|∇E|² + V|E|² + (q/2)|E|⁴ ] dx
    
    Parameters
    ----------
    E : complex array
        Complex field
    p : float
        Dispersion coefficient
    q : float
        Nonlinearity coefficient
    bspf_op : bspf1d
        BSPF operator for differentiation and integration
    dx : float
        Grid spacing
    V : float, optional
        External potential. Default: 0.0
    bc_type : str
        Boundary condition type
    
    Returns
    -------
    H : float
        Total energy (Hamiltonian)
    """
    # Compute first derivative
    if bc_type == "dirichlet":
        # Dirichlet BC: E(0) = E(L) = 0
        E_work = E.copy()
        E_work[0] = 0.0
        E_work[-1] = 0.0
        E_real_work = np.real(E_work)
        E_imag_work = np.imag(E_work)
        dE_dx_real, _ = bspf_op.differentiate(E_real_work, k=1)
        dE_dx_imag, _ = bspf_op.differentiate(E_imag_work, k=1)
        dE_dx = dE_dx_real + 1j * dE_dx_imag
    elif bc_type == "neumann":
        # Neumann BC: E'(0) = E'(L) = 0
        E_real_work = np.real(E)
        E_imag_work = np.imag(E)
        dE_dx_real, _ = bspf_op.differentiate(E_real_work, k=1, neumann_bc=(0.0, 0.0))
        dE_dx_imag, _ = bspf_op.differentiate(E_imag_work, k=1, neumann_bc=(0.0, 0.0))
        dE_dx = dE_dx_real + 1j * dE_dx_imag
    else:
        raise ValueError(f"Unknown boundary condition type: {bc_type}")
    
    # Kinetic energy density: (p/2)|∇E|²
    kinetic_density = (p / 2.0) * np.abs(dE_dx)**2
    
    # Potential energy density: V|E|²
    potential_density = V * np.abs(E)**2
    
    # Nonlinear energy density: (q/2)|E|⁴
    nonlinear_density = (q / 2.0) * np.abs(E)**4
    
    # Total energy density
    energy_density = kinetic_density + potential_density + nonlinear_density
    
    # Integrate using BSPF
    H = bspf_op.definite_integral(energy_density)
    
    return H


def main():
    # ========================================================================
    # Control Parameters - Set all parameters here
    # ========================================================================
    
    # Equation parameters
    p = 1.0  # Dispersion coefficient
    q = 1.0  # Nonlinearity coefficient
    
    # Domain parameters
    L = 100.0  # Domain length
    dx = None  # Spatial step (None = auto: 0.001*L)
    
    # Time parameters
    dt = 0.001  # Time step
    T = 10.0  # Total time
    
    # Soliton parameters
    a0 = 1.0  # Soliton amplitude
    v0 = 1.0  # Soliton velocity
    x0 = 0.5  # Initial position (fraction of L)
    
    # Numerical method parameters
    bspf_degree = 5  # B-spline degree for bspf1d
    bc = 'neumann'  # Boundary condition type: 'dirichlet' or 'neumann'
    
    # ========================================================================
    # End of Control Parameters
    # ========================================================================
    
    # Check if bspf is available
    if not BSPF_AVAILABLE:
        raise ImportError("bspf library is not available. Install it with: pip install bspf")
    
    # Setup grid
    if dx is None:
        dx = 0.001 * L
    
    # For Dirichlet/Neumann, include boundaries
    x = np.arange(0, L + dx, dx)
    nx = len(x)
    dx = x[1] - x[0]  # Actual dx
    
    # BSPF operator
    print(f"  Creating bspf1d operator (nx={nx}, dx={dx:.6f}, degree={bspf_degree})...")
    bspf_op = bspf1d.from_grid(
        degree=bspf_degree,
        x=x,
        order=bspf_degree,
        use_clustering=True,
        clustering_factor=2.0,
        correction='spectral',
        use_gpu=False
    )
    
    # Soliton parameters (convert x0 from fraction to absolute)
    x0_abs = x0 * L
    
    # BC type for analytical solution matches numerical
    bc_analytical = bc
    
    # Initial condition (analytical solution at t=0)
    E_numerical = onesoliton_analytical(x, 0.0, a0, v0, x0_abs, p, q, L=L, bc_type=bc_analytical)
    
    # Create RHS function for modular time stepper
    def rhs_func_bspf(E, p, q, bspf_op, V, bc_type):
        return nlseqn_bspf(E, p, q, bspf_op, V, bc_type)
    rhs_func = lambda E: rhs_func_bspf(E, p, q, bspf_op, V, bc)
    
    # Initialize time stepper state with progress bar
    state = TimeStepperState(E_numerical.copy(), t_init=0.0, dt=dt, method='rk45', t_final=T, show_progress=True)
    
    # Compute initial energy
    V = 0.0  # External potential
    energy_initial = compute_energy_bspf(E_numerical, p, q, bspf_op, dx, V=V, bc_type=bc)
    print(f"  Initial energy: H₀ = {energy_initial:.10e}")
    
    # Track energy over time
    energies = []
    times_energy = []
    energies.append(energy_initial)
    times_energy.append(0.0)
    
    # Storage for space-time plot
    nsteps = int(T / dt)
    n_save = min(500, nsteps)  # Save at most 500 snapshots for space-time plot
    save_interval = max(1, nsteps // n_save)
    E_history = []
    times_history = []
    E_history.append(E_numerical.copy())
    times_history.append(0.0)
    
    print("=" * 80)
    print("Single Soliton Test (with Analytical Solution)")
    print("=" * 80)
    print(f"  Method: rk45 + BSPF (degree={bspf_degree})")
    print(f"  Parameters: p={p}, q={q}")
    print(f"  Soliton: a0={a0}, v0={v0}, x0={x0_abs:.2f}")
    print(f"  Grid: nx={nx}, dx={dx:.6f}, L={L}")
    print(f"  Time: dt={dt:.6f}, T={T}, nsteps={nsteps}")
    print(f"  BC: {bc}")
    print("-" * 80)
    print("Starting simulation...")
    
    # Time stepping with progress bar
    with state:
        for ii in range(nsteps):
            # Time step
            E_numerical = time_step(state, dt, rhs_func, method='rk45')
            t = state.get_current_time()
            
            # Save snapshots for space-time plot
            if (ii + 1) % save_interval == 0 or ii == nsteps - 1:
                E_history.append(E_numerical.copy())
                times_history.append(t)
            
            # Compute energy at regular intervals
            if (ii + 1) % max(1, nsteps // 100) == 0 or ii == nsteps - 1:
                energy_current = compute_energy_bspf(E_numerical, p, q, bspf_op, dx, V=V, bc_type=bc)
                energies.append(energy_current)
                times_energy.append(t)
    
    print("-" * 80)
    print("Simulation complete!")
    print("=" * 80)
    
    # Final energy computation
    energy_final = compute_energy_bspf(E_numerical, p, q, bspf_op, dx, V=V, bc_type=bc)
    if len(energies) == 0 or times_energy[-1] < T:
        energies.append(energy_final)
        times_energy.append(T)
    
    # Energy conservation analysis
    energy_error_abs = np.abs(np.array(energies) - energy_initial)
    energy_error_max = np.max(energy_error_abs)
    energy_error_relative = energy_error_max / np.abs(energy_initial) if np.abs(energy_initial) > 0 else 0.0
    
    print("\n" + "=" * 80)
    print("Energy Conservation Analysis:")
    print("=" * 80)
    print(f"  Initial energy: H₀ = {energy_initial:.10e}")
    print(f"  Final energy:   H_T = {energy_final:.10e}")
    print(f"  Energy change:  ΔH = {energy_final - energy_initial:.10e}")
    print(f"  Max |ΔH|:       {energy_error_max:.10e}")
    print(f"  Relative error: {energy_error_relative:.6e}")
    print("=" * 80)
    
    # Compute final analytical solution for comparison
    E_analytical_final = onesoliton_analytical(x, T, a0, v0, x0_abs, p, q, L=L, bc_type=bc_analytical)
    error_final = np.abs(E_numerical - E_analytical_final)
    max_error_final = np.max(error_final)
    l2_error_final = np.sqrt(np.sum(error_final**2) * dx)
    
    print("\n" + "=" * 80)
    print("Final Error Analysis:")
    print("=" * 80)
    print(f"  Max error: {max_error_final:.6e}")
    print(f"  L2 error:  {l2_error_final:.6e}")
    print("=" * 80)
    
    # Create visualization: final state and space-time plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top left: Final state comparison
    ax = axes[0, 0]
    ax.plot(x, np.abs(E_numerical)**2, 'b-', label='Numerical |E|²', linewidth=2)
    ax.plot(x, np.abs(E_analytical_final)**2, 'r--', label='Analytical |E|²', linewidth=2, alpha=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel('|E|²')
    ax.set_title(f'Final State (t={T:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, L)
    
    # Top right: Final error
    ax = axes[0, 1]
    ax.plot(x, error_final, 'k-', linewidth=1.5)
    ax.set_xlabel('x')
    ax.set_ylabel('|Error|')
    ax.set_title(f'Final Error (max={max_error_final:.2e})')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, L)
    ax.set_yscale('log')
    
    # Bottom left: Space-time plot
    ax = axes[1, 0]
    E_history_array = np.array(E_history)
    density_history = np.abs(E_history_array)**2
    im = ax.imshow(density_history.T, aspect='auto', origin='lower',
                   extent=[0, T, 0, L], cmap='plasma', interpolation='bilinear')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Position x')
    ax.set_title('Space-Time Plot: |E|²')
    plt.colorbar(im, ax=ax, label='|E|²')
    
    # Bottom right: Energy conservation
    ax = axes[1, 1]
    ax.semilogy(times_energy, np.asarray(energies) - 5.3333333333e+01, 'b-', linewidth=2, label='Energy H(t)')
    # ax.axhline(energy_initial, color='r', linestyle='--', linewidth=1.5,
    #             label=f'H₀ = {energy_initial:.6e}')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Energy H(t)')
    ax.set_title('Energy Conservation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('single_soliton_results.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved results plot: single_soliton_results.png")
    
    plt.show()


if __name__ == '__main__':
    main()

