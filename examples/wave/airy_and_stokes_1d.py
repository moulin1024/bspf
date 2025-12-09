
import numpy as np
from bspf import bspf1d, TimeStepperState, time_step
import matplotlib.pyplot as plt

# ============================================================
# Control Parameters (Shared for both equations)
# ============================================================
# Physical parameters
g = 9.81              # gravitational acceleration (m/s²)
h = 6.0              # mean water depth (m)
c = np.sqrt(g*h)      # linear wave speed (m/s)

# Domain parameters
L = 100.0             # domain length (m)
N = 1024               # number of grid points
x = np.linspace(0, L, N)

# Initial condition parameters (Gaussian bump)
A_gaussian = 0.5          # Gaussian amplitude
sigma_gaussian = 5.0      # Gaussian width (standard deviation)
x0_gaussian = L / 2.0     # Gaussian center position

# Time parameters
dt = 0.5 * (L/N) / c  # CFL-safe time step
T = 10.0               # final time (s)
steps = int(T/dt)

# BSPF parameters
degree = 5

# Storage parameters
n_save = 200  # Number of time snapshots to save

# ============================================================
# Setup
# ============================================================
# Create BSPF operator
bspf = bspf1d.from_grid(degree=degree, x=x)

# Initial condition: Gaussian bump (same for both equations)
eta = A_gaussian * np.exp(-(x - x0_gaussian)**2 / (2.0 * sigma_gaussian**2))
v = np.zeros_like(x)  # eta_t = 0
eta0 = eta.copy()

# Pack/unpack state vector functions
def pack_state(eta, v):
    """Pack eta and v into a single state vector."""
    return np.concatenate([eta, v])

def unpack_state(q):
    """Unpack state vector into eta and v."""
    return q[:N], q[N:]

# ============================================================
# Airy Wave Equation RHS
# ============================================================
def airy_rhs(q, bspf_op, c):
    """
    Right-hand side for Airy wave equation (linear).
    
    System:
        eta_t = v
        v_t = c^2 * eta_xx
    
    With Neumann BCs: eta_x = 0 at boundaries (reflecting boundaries)
    """
    eta, v = unpack_state(q)
    
    # Compute second derivative with Neumann BCs
    d1, d2, _ = bspf_op.differentiate_1_2(eta, neumann_bc=(0.0, 0.0))
    
    # Time derivatives
    deta_dt = v
    dv_dt = c**2 * d2
    
    # Pack derivatives
    dq_dt = pack_state(deta_dt, dv_dt)
    
    return dq_dt

# ============================================================
# Stokes Wave Equation RHS
# ============================================================
def stokes_rhs(q, bspf_op, g, h, c):
    """
    Right-hand side for Stokes wave equation (nonlinear).
    
    The Stokes wave equation is a nonlinear extension of the Airy wave equation:
        eta_t = v
        v_t = c^2 * eta_xx - (3/2) * g * (eta^2)_xx  (nonlinear correction)
    
    With Neumann BCs: eta_x = 0 at boundaries (reflecting boundaries)
    """
    eta, v = unpack_state(q)
    
    # Compute derivatives with Neumann BCs
    d1, d2, _ = bspf_op.differentiate_1_2(eta, neumann_bc=(0.0, 0.0))
    
    # Linear part: c^2 * eta_xx
    linear_term = c**2 * d2
    
    # Nonlinear correction: -(3/2) * g * (eta^2)_xx
    # First compute eta^2, then its second derivative
    eta_squared = eta**2
    d1_eta2, d2_eta2, _ = bspf_op.differentiate_1_2(eta_squared, neumann_bc=(0.0, 0.0))
    nonlinear_term = -(3.0/2.0) * g * d2_eta2
    
    # Time derivatives
    deta_dt = v
    dv_dt = linear_term + nonlinear_term
    
    # Pack derivatives
    dq_dt = pack_state(deta_dt, dv_dt)
    
    return dq_dt

# ============================================================
# Time Integration Function
# ============================================================
def solve_wave_equation(rhs_func, name, q0, dt, T, steps, n_save):
    """
    Solve a wave equation and return solution history.
    
    Parameters:
    -----------
    rhs_func : callable
        RHS function for time stepper
    name : str
        Name of the equation (for progress display)
    q0 : array
        Initial state vector
    dt : float
        Time step
    T : float
        Final time
    steps : int
        Number of time steps
    n_save : int
        Number of snapshots to save
        
    Returns:
    --------
    eta_history : array
        Solution history (n_save, N)
    times_history : array
        Time points for history
    eta_final : array
        Final solution
    """
    print(f"\nSolving {name} wave equation...")
    
    # Initialize state
    state = TimeStepperState(q0.copy(), t_init=0.0, dt=dt, method='rk23', t_final=T, show_progress=True)
    
    # Storage for space-time plot
    save_interval = max(1, steps // n_save)
    eta_history = []
    times_history = []
    
    for step in range(steps):
        q_next = time_step(state, dt, rhs_func, method='rk23')
        q = state.get_current()
        eta, v = unpack_state(q)
        
        # Save solution at regular intervals for space-time plot
        if step % save_interval == 0 or step == steps - 1:
            eta_history.append(eta.copy())
            times_history.append(state.get_current_time())
    
    # Convert to arrays
    eta_history = np.array(eta_history)
    times_history = np.array(times_history)
    
    # Extract final solution
    q_final = state.get_current()
    eta_final, v_final = unpack_state(q_final)
    
    return eta_history, times_history, eta_final

# ============================================================
# Main Execution
# ============================================================
print("="*60)
print("Comparing Airy and Stokes Wave Equations")
print("="*60)
print(f"Parameters:")
print(f"  g = {g:.3f} m/s²")
print(f"  h = {h:.3f} m")
print(f"  c = {c:.3f} m/s")
print(f"  L = {L:.3f} m, N = {N}")
print(f"  Initial condition: Gaussian (A={A_gaussian:.3f} m, σ={sigma_gaussian:.3f} m)")
print(f"  dt = {dt:.6e}, T = {T:.3f}, steps = {steps}")
print(f"  Boundary conditions: Neumann (reflecting)")
print("="*60)

# Initial state
q0 = pack_state(eta, v)

# Create RHS functions
def airy_rhs_func(q):
    return airy_rhs(q, bspf, c)

def stokes_rhs_func(q):
    return stokes_rhs(q, bspf, g, h, c)

# Solve both equations
eta_history_airy, times_history_airy, eta_final_airy = solve_wave_equation(
    airy_rhs_func, "Airy", q0, dt, T, steps, n_save
)

eta_history_stokes, times_history_stokes, eta_final_stokes = solve_wave_equation(
    stokes_rhs_func, "Stokes", q0, dt, T, steps, n_save
)

print("\n" + "="*60)
print("Simulation completed successfully!")
print("="*60)

# ============================================================
# Visualization
# ============================================================
fig = plt.figure(figsize=(16, 10))

# Top row: Initial and final conditions comparison
ax1 = plt.subplot(2, 2, 1)
ax1.plot(x, eta0, 'k--', label='Initial condition', linewidth=2, alpha=0.7)
ax1.plot(x, eta_final_airy, 'b-', label='Airy (linear)', linewidth=2)
ax1.plot(x, eta_final_stokes, 'r-', label='Stokes (nonlinear)', linewidth=2)
ax1.set_xlabel('$x$ (m)')
ax1.set_ylabel('$\\eta(x,t)$ (m)')
ax1.set_title('Final Solutions Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Top right: Difference between Airy and Stokes
ax2 = plt.subplot(2, 2, 2)
difference = eta_final_stokes - eta_final_airy
ax2.plot(x, difference, 'g-', linewidth=2)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax2.set_xlabel('$x$ (m)')
ax2.set_ylabel('$\\eta_{Stokes} - \\eta_{Airy}$ (m)')
ax2.set_title('Nonlinear Effect (Stokes - Airy)')
ax2.grid(True, alpha=0.3)

# Bottom left: Airy wave space-time plot
ax3 = plt.subplot(2, 2, 3)
im1 = ax3.imshow(eta_history_airy, aspect='auto', origin='lower',
                 extent=[x[0], x[-1], times_history_airy[0], times_history_airy[-1]],
                 cmap='viridis', interpolation='bilinear')
ax3.set_xlabel('$x$ (m)')
ax3.set_ylabel('$t$ (s)')
ax3.set_title('Airy Wave: Space-Time Evolution (Linear)')
plt.colorbar(im1, ax=ax3, label='$\\eta(x,t)$ (m)')

# Bottom right: Stokes wave space-time plot
ax4 = plt.subplot(2, 2, 4)
im2 = ax4.imshow(eta_history_stokes, aspect='auto', origin='lower',
                 extent=[x[0], x[-1], times_history_stokes[0], times_history_stokes[-1]],
                 cmap='viridis', interpolation='bilinear')
ax4.set_xlabel('$x$ (m)')
ax4.set_ylabel('$t$ (s)')
ax4.set_title('Stokes Wave: Space-Time Evolution (Nonlinear)')
plt.colorbar(im2, ax=ax4, label='$\\eta(x,t)$ (m)')

plt.tight_layout()
plt.show()

# Print statistics
print("\n" + "="*60)
print("Solution Statistics (Final Time):")
print("="*60)
print(f"Airy wave:")
print(f"  Max |eta| = {np.max(np.abs(eta_final_airy)):.6f} m")
print(f"  Min eta = {np.min(eta_final_airy):.6f} m")
print(f"  Max eta = {np.max(eta_final_airy):.6f} m")
print(f"\nStokes wave:")
print(f"  Max |eta| = {np.max(np.abs(eta_final_stokes)):.6f} m")
print(f"  Min eta = {np.min(eta_final_stokes):.6f} m")
print(f"  Max eta = {np.max(eta_final_stokes):.6f} m")
print(f"\nDifference (Stokes - Airy):")
print(f"  Max |difference| = {np.max(np.abs(difference)):.6f} m")
print(f"  RMS difference = {np.sqrt(np.mean(difference**2)):.6f} m")
print("="*60)

