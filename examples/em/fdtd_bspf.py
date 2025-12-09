"""
BSPF-based FDTD solver for 1D Maxwell's equations.

This script solves the 1D Maxwell's equations using B-spline finite elements (BSPF)
instead of finite differences. It compares the numerical solution with the exact
analytical solution using the method of images.

Maxwell's equations in 1D:
    dE/dt = c * dH/dx
    dH/dt = c * dE/dx

with Perfect Electric Conductor (PEC) boundary conditions: E = 0 at boundaries.

The BSPF method uses high-order B-spline basis functions for spatial discretization,
which can provide better accuracy than finite differences, especially for smooth
solutions. Time stepping is done using RK4.

Boundary conditions:
- E-field: PEC (Dirichlet) - E = 0 at boundaries, enforced explicitly
- H-field: No explicit BC enforcement. H evolves naturally through Maxwell's
  equations. At PEC boundaries, E=0 but H can be non-zero (no zero-flux requirement).
  Enforcing zero-flux on H would create spurious left-propagating waves.

Run from repository root:
    python examples/em/fdtd_bspf.py              # Generate space-time plots
    python examples/em/fdtd_bspf.py --animate  # Generate animation GIF
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from bspf import bspf1d, TimeStepperState, time_step, enforce_zero_flux_neumann_bc

# --- 1. Parameters ---
SIZE = 200              # Grid points (x-axis)
DX = 1.0
C = 1.0
S = 0.9                 # Courant Number (< 1.0 for visible dispersion error)
DT = S * DX / C         
STEPS = 100             # Time steps (y-axis)
SIGMA = 10.0            # Pulse width
X_START = 50.0 * DX     # Pulse starting center

# BSPF parameters
DEGREE = 5              # B-spline degree
REG_PARAM = 1e-3        # Tikhonov regularization parameter

# --- 2. Analytical Solution Function (Method of Images) ---

def exact_solution(x_grid, t, dt, c, x_start, sigma, box_length):
    """Calculates the exact analytical E-field including one reflection."""
    current_time = t * dt
    
    # Primary Pulse (Moving Right)
    center_main = x_start + c * current_time
    pulse_main = np.exp(-0.5 * ((x_grid - center_main) / sigma) ** 2)
    
    # Reflected Pulse (Image Source moving left)
    wall_pos = box_length
    center_image = (2 * wall_pos - x_start) - c * current_time
    pulse_reflect = -1.0 * np.exp(-0.5 * ((x_grid - center_image) / sigma) ** 2)
    
    return pulse_main + pulse_reflect

# --- 3. BSPF Maxwell's Equations RHS ---

def create_maxwell_rhs(bspf_op_e, bspf_op_h, c, dx, enforce_bc_e, use_zero_flux=False):
    """
    Create RHS function for Maxwell's equations using BSPF.
    
    Maxwell's equations in 1D:
        dE/dt = c * dH/dx
        dH/dt = c * dE/dx
    
    Parameters:
    -----------
    bspf_op_e : bspf1d
        BSPF operator for E-field
    bspf_op_h : bspf1d
        BSPF operator for H-field
    c : float
        Speed of light
    dx : float
        Grid spacing (for zero-flux BC enforcement)
    enforce_bc_e : callable
        Function to enforce PEC boundary conditions on E-field
    use_zero_flux : bool
        If True, use zero-flux Neumann BCs for H-field derivatives (not recommended,
        as it can create spurious left-propagating waves)
    
    Returns:
    --------
    rhs_func : callable
        RHS function: rhs_func(state_flat) -> dstate_dt_flat
    """
    def rhs_func(state_flat):
        """
        Compute RHS for Maxwell's equations.
        
        Parameters:
        -----------
        state_flat : array
            Flattened state vector [E, H]
        
        Returns:
        --------
        dstate_dt_flat : array
            Flattened time derivative [dE/dt, dH/dt]
        """
        n = len(state_flat) // 2
        E = state_flat[:n]
        H = state_flat[n:]
        
        # Compute spatial derivatives using BSPF
        # dH/dx for E-field equation: no special BC needed
        # The derivative will be computed naturally, and boundary effects
        # will be handled through the PEC BC on E
        dH_dx, _ = bspf_op_h.differentiate(H, k=1, lam=REG_PARAM)
        
        # dE/dx for H-field equation: no special BC needed (E will be enforced separately)
        dE_dx, _ = bspf_op_e.differentiate(E, k=1, lam=REG_PARAM)
        
        # Maxwell's equations
        dE_dt = c * dH_dx
        dH_dt = c * dE_dx
        
        # Enforce PEC boundary conditions on dE/dt (since E = 0 at boundaries,
        # we must have dE/dt = 0 at boundaries to maintain the BC)
        dE_dt = enforce_bc_e(dE_dt)
        
        # Note: We do NOT enforce zero-flux BCs on dH/dt.
        # The H-field evolution is determined by dH/dt = c * dE/dx,
        # and at PEC boundaries, H can have non-zero values.
        # Enforcing zero-flux on H would create spurious left-propagating waves.
        
        # Combine into flattened state derivative
        dstate_dt_flat = np.concatenate([dE_dt, dH_dt])
        
        return dstate_dt_flat
    
    return rhs_func

def create_pec_bc_enforcer(n_grid):
    """
    Create boundary condition enforcer for PEC: E = 0 at boundaries.
    
    Parameters:
    -----------
    n_grid : int
        Number of grid points
    
    Returns:
    --------
    enforce_bc : callable
        Function that enforces PEC BCs: enforce_bc(E) -> E_bc
    """
    def enforce_bc(E):
        """Enforce PEC BCs: E = 0 at boundaries."""
        E_bc = E.copy()
        E_bc[0] = 0.0
        E_bc[-1] = 0.0
        return E_bc
    
    return enforce_bc

# --- 4. Simulation and Data Collection ---

def run_space_time_plot():
    
    # Initialization
    x_grid = np.arange(SIZE) * DX
    domain = (x_grid[0], x_grid[-1])
    
    # Initialize E-field
    ez = np.exp(-0.5 * ((x_grid - X_START) / SIGMA) ** 2)
    
    # Initialize H-field for One-Way (+x) Wave Propagation
    # For a one-way wave: H = -E (for c=1, right-propagating wave)
    # Since both fields are on the same grid (unlike staggered FDTD),
    # we initialize H to match E exactly: H = -E
    hy = -ez.copy()
    
    # Enforce PEC boundary conditions on initial E-field
    enforce_bc_e = create_pec_bc_enforcer(SIZE)
    ez = enforce_bc_e(ez)
    
    # Note: We do NOT enforce zero-flux BCs on H at initialization.
    # The H-field will evolve naturally through Maxwell's equations.
    # At PEC boundaries, E=0, but H can be non-zero (no zero-flux requirement).
    
    # Create BSPF operators for E and H fields
    bspf_op_e = bspf1d.from_grid(
        degree=DEGREE,
        x=x_grid,
        domain=domain,
        order=DEGREE
    )
    
    bspf_op_h = bspf1d.from_grid(
        degree=DEGREE,
        x=x_grid,
        domain=domain,
        order=DEGREE
    )
    
    # Create RHS function with zero-flux Neumann BCs for H-field
    rhs_func = create_maxwell_rhs(bspf_op_e, bspf_op_h, C, DX, enforce_bc_e, use_zero_flux=True)
    
    # Initialize state vector [E, H]
    state_flat = np.concatenate([ez, hy])
    
    # Data storage arrays
    E_history = []
    Error_history = []
    
    # Time stepping
    t = 0.0
    t_final = STEPS * DT
    
    with TimeStepperState(state_flat, t_init=0.0, dt=DT, method='rk4',
                          t_final=t_final, show_progress=False) as state:
        
        # Store initial state
        E_history.append(ez.copy())
        ez_exact = exact_solution(x_grid, 0, DT, C, X_START, SIGMA, SIZE * DX)
        error_field = ez - ez_exact
        Error_history.append(error_field.copy())
        
        # Main Loop
        for t_step in range(STEPS):
            
            # Time step
            state_flat_next = time_step(state, DT, rhs_func, method='rk4')
            state_flat = state.get_current()
            t = state.get_current_time()
            
            # Extract E and H fields
            ez = state_flat[:SIZE]
            hy = state_flat[SIZE:]
            
            # Enforce PEC boundary conditions on E-field after time step
            ez = enforce_bc_e(ez)
            
            # Note: H-field is NOT modified at boundaries.
            # It evolves naturally through Maxwell's equations.
            # Enforcing zero-flux on H would create spurious waves.
            
            # Update state
            state_flat[:SIZE] = ez
            state_flat[SIZE:] = hy
            state.psi_now = state_flat.copy()
            
            # Calculate Analytical Solution
            ez_exact = exact_solution(x_grid, t_step + 1, DT, C, X_START, SIGMA, SIZE * DX)
            
            # Calculate Error
            error_field = ez - ez_exact
            
            # Store Data
            E_history.append(ez.copy())
            Error_history.append(error_field.copy())
    
    # Convert lists to NumPy arrays for plotting
    E_matrix = np.array(E_history)
    Error_matrix = np.array(Error_history)
    
    # --- 5. Plotting ---
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot A: Electric Field (Ez) Space-Time Plot
    im1 = ax1.imshow(E_matrix, aspect='auto', 
                     extent=[0, SIZE*DX, STEPS*DT, 0], 
                     cmap='viridis', vmin=-1.0, vmax=1.0)
    
    ax1.set_title(f"BSPF Electric Field ($E_z$) Space-Time Plot (S={S}, degree={DEGREE})")
    ax1.set_ylabel("Time (arb. units)")
    ax1.axvline(X_START, color='w', linestyle='--', linewidth=1) # Initial source location
    fig.colorbar(im1, ax=ax1, label='Field Amplitude')
    
    # Plot B: Error Space-Time Plot
    max_err = np.max(np.abs(Error_matrix))
    im2 = ax2.imshow(Error_matrix, aspect='auto', 
                     extent=[0, SIZE*DX, STEPS*DT, 0], 
                     cmap='seismic', vmin=-max_err, vmax=max_err)

    ax2.set_title("Numerical Dispersion Error ($E_{BSPF} - E_{Exact}$)")
    ax2.set_ylabel("Time (arb. units)")
    ax2.set_xlabel("Space (x-axis)")
    fig.colorbar(im2, ax=ax2, label='Error Amplitude')
    
    fig.tight_layout()
    plt.savefig('fdtd_bspf_2d_spacetime_error_comparison.png', dpi=150)
    plt.close(fig)

    print("2D Space-Time plots saved to 'fdtd_bspf_2d_spacetime_error_comparison.png'")
    print(f"Max error: {max_err:.6e}")
    print(f"L2 error (final): {np.sqrt(np.mean(Error_matrix[-1]**2)):.6e}")

def run_animation():
    """
    Run simulation and create an animation showing the evolution of E-field
    compared with the exact solution.
    """
    
    # Initialization
    x_grid = np.arange(SIZE) * DX
    domain = (x_grid[0], x_grid[-1])
    
    # Initialize E-field
    ez = np.exp(-0.5 * ((x_grid - X_START) / SIGMA) ** 2)
    
    # Initialize H-field for One-Way (+x) Wave Propagation
    # For a one-way wave: H = -E (for c=1, right-propagating wave)
    hy = -ez.copy()
    
    # Enforce PEC boundary conditions on initial E-field
    enforce_bc_e = create_pec_bc_enforcer(SIZE)
    ez = enforce_bc_e(ez)
    
    # Note: We do NOT enforce zero-flux BCs on H at initialization.
    # The H-field will evolve naturally through Maxwell's equations.
    
    # Create BSPF operators for E and H fields
    bspf_op_e = bspf1d.from_grid(
        degree=DEGREE,
        x=x_grid,
        domain=domain,
        order=DEGREE
    )
    
    bspf_op_h = bspf1d.from_grid(
        degree=DEGREE,
        x=x_grid,
        domain=domain,
        order=DEGREE
    )
    
    # Create RHS function with zero-flux Neumann BCs for H-field
    rhs_func = create_maxwell_rhs(bspf_op_e, bspf_op_h, C, DX, enforce_bc_e, use_zero_flux=True)
    
    # Initialize state vector [E, H]
    state_flat = np.concatenate([ez, hy])
    
    # Data storage arrays
    E_history = []
    E_exact_history = []
    Error_history = []
    times_history = []
    
    # Time stepping
    t = 0.0
    t_final = STEPS * DT
    
    with TimeStepperState(state_flat, t_init=0.0, dt=DT, method='rk4',
                          t_final=t_final, show_progress=True) as state:
        
        # Store initial state
        ez_exact = exact_solution(x_grid, 0, DT, C, X_START, SIGMA, SIZE * DX)
        E_history.append(ez.copy())
        E_exact_history.append(ez_exact.copy())
        Error_history.append(ez - ez_exact)
        times_history.append(0.0)
        
        # Main Loop
        for t_step in range(STEPS):
            
            # Time step
            state_flat_next = time_step(state, DT, rhs_func, method='rk4')
            state_flat = state.get_current()
            t = state.get_current_time()
            
            # Extract E and H fields
            ez = state_flat[:SIZE]
            hy = state_flat[SIZE:]
            
            # Enforce PEC boundary conditions on E-field after time step
            ez = enforce_bc_e(ez)
            
            # Note: H-field is NOT modified at boundaries.
            # It evolves naturally through Maxwell's equations.
            # Enforcing zero-flux on H would create spurious waves.
            
            # Update state
            state_flat[:SIZE] = ez
            state_flat[SIZE:] = hy
            state.psi_now = state_flat.copy()
            
            # Calculate Analytical Solution
            ez_exact = exact_solution(x_grid, t_step + 1, DT, C, X_START, SIGMA, SIZE * DX)
            
            # Calculate Error
            error_field = ez - ez_exact
            
            # Store Data
            E_history.append(ez.copy())
            E_exact_history.append(ez_exact.copy())
            Error_history.append(error_field.copy())
            times_history.append(t)
    
    # Convert lists to NumPy arrays
    E_array = np.array(E_history)
    E_exact_array = np.array(E_exact_history)
    Error_array = np.array(Error_history)
    times_array = np.array(times_history)
    
    # Find global limits for consistent scaling
    E_min, E_max = -1.0, 1.0
    error_max = np.max(np.abs(Error_array))
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Top plot: E-field comparison
    line1, = ax1.plot([], [], 'b-', linewidth=2, label='BSPF $E_z$')
    line2, = ax1.plot([], [], 'r--', linewidth=2, alpha=0.7, label='Exact $E_z$')
    ax1.set_xlim(0, SIZE * DX)
    ax1.set_ylim(E_min, E_max)
    ax1.set_ylabel('Electric Field $E_z$')
    ax1.set_title('BSPF FDTD: Electric Field Evolution')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(X_START, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Initial position')
    ax1.axvline(SIZE * DX, color='red', linestyle='--', linewidth=1, alpha=0.5, label='PEC boundary')
    
    # Middle plot: Error
    line3, = ax2.plot([], [], 'r-', linewidth=2, label='Error')
    ax2.set_xlim(0, SIZE * DX)
    ax2.set_ylim(-error_max, error_max)
    ax2.set_ylabel('Error $E_{BSPF} - E_{Exact}$')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Bottom plot: Space-time plot (showing current time)
    im = ax3.imshow(E_array, aspect='auto', 
                    extent=[0, SIZE*DX, STEPS*DT, 0], 
                    cmap='viridis', vmin=E_min, vmax=E_max, origin='upper')
    ax3.axvline(X_START, color='w', linestyle='--', linewidth=1, alpha=0.7)
    time_line = ax3.axhline(0, color='yellow', linewidth=2, alpha=0.8, label='Current time')
    ax3.set_xlabel('Space $x$')
    ax3.set_ylabel('Time $t$')
    ax3.set_title('Space-Time Evolution')
    cbar = plt.colorbar(im, ax=ax3, label='$E_z$')
    ax3.legend(loc='upper right', fontsize=8)
    
    # Time text
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, 
                         fontsize=12, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    error_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes,
                          fontsize=12, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def animate(frame):
        """Update animation frame."""
        # Update E-field comparison
        line1.set_data(x_grid, E_array[frame])
        line2.set_data(x_grid, E_exact_array[frame])
        
        # Update error plot
        line3.set_data(x_grid, Error_array[frame])
        
        # Update time line in space-time plot
        current_time = times_array[frame]
        time_line.set_ydata([current_time, current_time])
        
        # Update text
        max_err_frame = np.max(np.abs(Error_array[frame]))
        l2_err_frame = np.sqrt(np.mean(Error_array[frame]**2))
        time_text.set_text(f'Time: $t = {current_time:.3f}$\nStep: {frame}/{STEPS}')
        error_text.set_text(f'Max Error: {max_err_frame:.4e}\nL2 Error: {l2_err_frame:.4e}')
        
        return line1, line2, line3, time_line, time_text, error_text
    
    # Create animation
    print("\nCreating animation...")
    anim = animation.FuncAnimation(fig, animate, frames=len(E_history), 
                                   interval=50, blit=False, repeat=True)
    
    # Save animation as GIF
    output_file = 'fdtd_bspf_animation.gif'
    print(f"Saving animation to '{output_file}'...")
    try:
        anim.save(output_file, writer='pillow', fps=20, dpi=100)
        print(f"Animation saved successfully to '{output_file}'")
    except Exception as e:
        print(f"Warning: Could not save GIF ({e}). Trying alternative method...")
        try:
            # Try with imagemagick writer
            anim.save(output_file, writer='imagemagick', fps=20)
            print(f"Animation saved successfully to '{output_file}'")
        except Exception as e2:
            print(f"Warning: Could not save animation ({e2})")
            print("Displaying animation instead (close window to stop)...")
            plt.show()
    
    return anim, E_array, E_exact_array, Error_array, times_array

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--animate':
        run_animation()
    else:
        run_space_time_plot()

