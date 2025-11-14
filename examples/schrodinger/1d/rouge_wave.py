import numpy as np
import matplotlib.pyplot as plt
import os
from bspf import bspf1d, TimeStepperState, time_step

# =========================================================
# 1D focusing NLS with modulational instability (BSPF)
# i ψ_t = ψ_xx + g |ψ|^2 ψ
# Modulational instability initial condition
# Spatial: BSPF with Neumann BC, time: bspf time stepper (RK4/RK45/RK23/BDF2)
# =========================================================

# ---------------- Parameters ----------------
Nx   = 256          # number of spatial points
x_min, x_max = -1.0, 1.0
x    = np.linspace(x_min, x_max, Nx, endpoint=True)  # endpoint=True for non-periodic BC
L    = x_max - x_min
dx   = x[1] - x[0]

dt        = 0.001   # time step
T_max     = 1.0     # total simulated time
Nt        = int(T_max / dt)

g = -1.0  # Nonlinear coupling constant (focusing NLS)

# ---------------- Figure saving parameters ----------------
save_every = 10     # save figure every N timesteps
output_dir = "rogue_wave_frames"  # directory to save figures
os.makedirs(output_dir, exist_ok=True)

# ---------------- BSPF setup with Neumann BC (reflective) ----------------
# Neumann BC: ∂ψ/∂x = 0 at both boundaries (reflective)
bspf_op = bspf1d.from_grid(
    degree=5,
    x=x
)
neumann_bc = (0.0, 0.0)  # (left_flux, right_flux) = 0 for reflective BC

# ---------------- Build Laplacian matrix for Jacobian ----------------
def build_laplacian_matrix_bspf(bspf_op, n):
    """
    Build Laplacian matrix using native BSPF differentiate method.
    
    The Laplacian matrix L is such that L @ psi gives the second derivative.
    We build it by applying differentiate to unit vectors.
    
    Parameters:
    -----------
    bspf_op : bspf1d
        B-spline operator
    n : int
        Grid size (number of points)
        
    Returns:
    --------
    L : array (complex, shape=(n, n))
        Laplacian matrix
    """
    # Create identity matrix once (all unit vectors)
    I = np.eye(n, dtype=np.float64)
    
    # Apply differentiate to each column of identity matrix
    # This builds the Laplacian matrix column by column
    L_columns = [bspf_op.differentiate(I[:, k], k=2, neumann_bc=neumann_bc)[0] 
                 for k in range(n)]
    
    # Convert list of columns to array and transpose
    L = np.array(L_columns, dtype=np.complex128).T
    
    return L

# ---------------- RHS function for NLS equation ----------------
# Equation: i ψ_t = ψ_xx + g |ψ|^2 ψ

def rhs_nlse_bspf(psi, bspf_op, g):
    """
    RHS for B-spline method with Neumann BC.
    Works on full grid, bspf handles BC internally.
    
    Parameters:
    -----------
    psi : array (complex)
        Full solution on grid (including boundaries)
    bspf_op : bspf1d
        B-spline operator for computing derivatives
    g : float
        Nonlinear coupling constant
    """
    # Compute second derivative with Neumann BC: ψ_x = 0 at both ends
    lap, _ = bspf_op.differentiate(psi, k=2, neumann_bc=neumann_bc)
    
    # Nonlinear term
    nl = g * np.abs(psi)**2 * psi
    
    return 1j * (lap + nl)

# ---------------- Analytical Jacobian for BDF2 ----------------
def jacobian_nlse_bspf(psi, L_complex, g):
    """
    Analytical Jacobian of RHS for B-spline method using precomputed Laplacian.
    
    For F(ψ) = i(ψ_xx + g|ψ|²ψ):
    - Linear part: i·L (where L is the precomputed Laplacian matrix)
    - Nonlinear part: i·g·∂(|ψ|²ψ)/∂ψ = i·g·(2|ψ|²) = 2i·g·|ψ|² (diagonal)
    
    Parameters:
    -----------
    psi : array (complex)
        Current solution state
    L_complex : array (complex, shape=(n, n))
        Precomputed Laplacian matrix
    g : float
        Nonlinear coupling constant
    
    Returns:
    --------
    J : array (complex, shape=(n, n))
        Jacobian matrix ∂F/∂ψ
    """
    # Linear part: i·L (using precomputed Laplacian)
    J_linear = 1j * L_complex
    
    # Nonlinear part: diagonal matrix with 2i·g·|ψ|²
    J_nonlinear = np.diag(2j * g * np.abs(psi)**2)
    
    return J_linear + J_nonlinear

# ---------------- Modulational instability initial condition ----------------
psi = (1.0 + 0.1*np.cos(4*x)).astype(np.complex128)

# =========================================================
# PLOT SETUP
# =========================================================
# Set DPI to ensure even pixel dimensions (100 DPI * 10 inches = 1000 pixels, even)
dpi = 100
fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
ax.set_xlim(x_min, x_max)
ax.set_ylim(0, 2.0)  # Adjust for MI case
ax.set_xlabel("x")
ax.set_ylabel("|ψ(x,t)|²")

# =========================================================
# TIME INTEGRATION + SAVE FIGURES
# =========================================================
# Time stepping method
time_method = 'bdf2'  # Options: 'rk4', 'rk45', 'rk23', 'bdf2'

# Precompute Laplacian matrix for Jacobian (only needed for BDF2)
L_complex = None
if time_method.lower() == 'bdf2':
    print("Building Laplacian matrix for BDF2 Jacobian...")
    L_complex = build_laplacian_matrix_bspf(bspf_op, Nx)
    print("  Laplacian matrix built successfully")

# Create RHS function with bspf_op and g bound
def rhs_func(psi):
    return rhs_nlse_bspf(psi, bspf_op, g)

# Create Jacobian function (only needed for BDF2)
jacobian_func = None
if time_method.lower() == 'bdf2' and L_complex is not None:
    def jacobian_func(psi):
        return jacobian_nlse_bspf(psi, L_complex, g)

# Initialize time stepper
T_final = T_max
state = TimeStepperState(
    psi_init=psi,
    t_init=0.0,
    dt=dt,
    method=time_method,
    t_final=T_final,
    show_progress=False  # Disable progress bar since we have our own printing
)

# Save initial frame
amp2 = np.abs(psi)**2
ax.plot(x, amp2, lw=2)
ax.set_title(f"1D NLS Modulational Instability (BSPF, Neumann BC, {time_method.upper()}), t = 0.000")
fig.tight_layout()  # Adjust layout before saving
frame_num = 0
filename = os.path.join(output_dir, f"frame_{frame_num:05d}.png")
# Save with fixed dimensions (no bbox_inches='tight') to ensure even pixel dimensions
# figsize=(10,5) * dpi=100 = 1000x500 pixels (both even)
fig.savefig(filename, dpi=dpi, bbox_inches=None)
print(f"Saved frame {frame_num} to {filename}")

# Time integration loop
for n in range(Nt):
    if n % save_every == 0:
        print(f"Time step {n} of {Nt}")
    
    # Physical time (for plotting) - time after the step
    t_phys = (n + 1) * dt

    # Time step using bspf time stepper
    psi = time_step(state, dt, rhs_func, method=time_method, jacobian_func=jacobian_func)

    # Save figure every save_every timesteps
    if (n + 1) % save_every == 0:
        amp2 = np.abs(psi)**2
        ax.clear()
        ax.plot(x, amp2, lw=2)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 2.0)
        ax.set_xlabel("x")
        ax.set_ylabel("|ψ(x,t)|²")
        ax.set_title(f"1D NLS Modulational Instability (BSPF, Neumann BC, {time_method.upper()}), t = {t_phys:6.3f}")
        
        # Save figure with zero-padded frame number
        frame_num = (n + 1) // save_every
        filename = os.path.join(output_dir, f"frame_{frame_num:05d}.png")
        fig.tight_layout()  # Adjust layout before saving
        # Save with fixed dimensions (no bbox_inches='tight') to ensure even pixel dimensions
        # figsize=(10,5) * dpi=100 = 1000x500 pixels (both even)
        fig.savefig(filename, dpi=dpi, bbox_inches=None)
        print(f"Saved frame {frame_num} to {filename}")

# Close time stepper (cleanup progress bar if any)
state.close_progress()

plt.close(fig)
total_frames = 1 + (Nt // save_every)  # Initial frame + frames during integration
print(f"Saved {total_frames} figures to {output_dir}/")
