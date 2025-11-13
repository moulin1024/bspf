import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Import bspf library components
from bspf import (bspf1d, create_schrodinger_rhs, create_dirichlet_bc_enforcer,
                  SchrodingerAnimator1D, create_comparison_plot_config)

# =========================================================
# PARAMETERS
# =========================================================
hbar = 1.0
m = 1.0

# Nonlinearity parameter: g < 0 for bright solitons (focusing), g > 0 for dark solitons (defocusing)
g = -1.0  # Focusing NLS (bright soliton)

# Grid setup - include endpoints for Dirichlet BCs
Nx = 512  # Number of grid points (including endpoints)
xmin, xmax = -30, 30  # Larger domain to minimize boundary effects
x = np.linspace(xmin, xmax, Nx)  # Include endpoints for Dirichlet BCs
dx = x[1] - x[0]

dt = 0.002  # Time step
Nt = 20000

# Time stepping method: 'rk4', 'rk45', or 'bdf2'
# Note: Split-step method requires FFT, so we use RK methods with bspf1d
time_method = 'rk45'

# Output parameters
save_interval = 100  # Save plot every N steps
output_folder = 'qm_plots'  # Output folder for plots

# BSPF parameters
degree = 6

# Soliton parameters
A = 1.0      # Amplitude
v = 0.5      # Velocity
x0 = -10.0   # Initial position

# =========================================================
# SETUP BSPF OPERATOR AND SCHRÖDINGER RHS
# =========================================================
print("Setting up BSPF operator...")
bspf_op = bspf1d.from_grid(degree=degree, x=x, use_clustering=True, clustering_factor=2.0, correction='spectral')
print(f"  BSPF setup complete: degree={degree}, grid_size={Nx}")

# Create potential (zero for free particle NLS)
V = np.zeros(Nx)

# Create BC enforcer and RHS function using library utilities
enforce_bc = create_dirichlet_bc_enforcer(Nx)
# Note: In non-dimensional units (ℏ = m = 1), the equation is:
# i*∂ψ/∂t = -(1/2)*∂²ψ/∂x² + g*|ψ|²*ψ
# The library function handles this automatically
nls_rhs = create_schrodinger_rhs(bspf_op, V, g=g, enforce_bc=enforce_bc)
print(f"  Created NLS RHS function (g={g:.2f}, {'nonlinear' if g != 0 else 'linear'})")

# =========================================================
# ANALYTICAL BRIGHT SOLITON SOLUTION
# =========================================================
def psi_soliton_analytical(x, t, A, v, x0, g, hbar, m):
    """
    Analytical bright soliton solution for focusing NLS (g < 0):
    
    i*ℏ*∂ψ/∂t = -ℏ²/(2m)*∂²ψ/∂x² + g*|ψ|²*ψ
    
    With ℏ=1, m=1: i*∂ψ/∂t = -(1/2)*∂²ψ/∂x² + g*|ψ|²*ψ
    
    For bright soliton (g < 0), the solution is:
    ψ(x,t) = A * sech(κ*(x - x0 - vt)) * 
             exp(i*(v*(x - x0) - ω*t))
    
    where:
    - κ = A * sqrt(-g) (soliton width parameter)
    - ω = v²/2 - A²*g/2 (frequency)
    
    Note: This solution is for infinite domain. With Dirichlet BCs,
    we'll enforce ψ = 0 at boundaries during time stepping.
    """
    # Check that g < 0 for bright soliton
    if g >= 0:
        raise ValueError("Bright soliton requires g < 0 (focusing NLS)")
    
    # Soliton width parameter: κ = A * sqrt(-g)
    kappa = A * np.sqrt(-g)
    
    # Position of soliton center
    x_center = x0 + v * t
    
    # Envelope (sech profile)
    envelope = A * (1.0 / np.cosh(kappa * (x - x_center)))
    
    # Phase factor: exp(i*(v*(x - x0) - ω*t))
    omega = v**2 / 2 - A**2 * g / 2
    phase = np.exp(1j * (v * (x - x0) - omega * t))
    
    return envelope * phase

# Initial condition from analytical solution
def psi_initial(x):
    psi_init = psi_soliton_analytical(x, 0.0, A, v, x0, g, hbar, m)
    # Enforce Dirichlet BCs: ψ = 0 at boundaries
    return enforce_bc(psi_init)

psi = psi_initial(x)

# Verify initial condition matches analytical solution at t=0
print("="*60)
print("INITIAL CONDITION CHECK (t=0)")
print("="*60)
psi0_ana = psi_soliton_analytical(x, 0.0, A, v, x0, g, hbar, m)
# Analytical solution doesn't satisfy Dirichlet BCs, so we compare
# the interior points (excluding boundaries)
ic_error = np.abs(psi[1:-1] - psi0_ana[1:-1])
ic_max_error = np.max(ic_error)
ic_l2_error = np.sqrt(np.mean(ic_error**2))
ic_relative_error = ic_max_error / (np.max(np.abs(psi0_ana[1:-1])) + 1e-15)

print(f"  Max error (interior):     |ψ_num - ψ_ana|_∞ = {ic_max_error:.2e}")
print(f"  L2 error (interior):      |ψ_num - ψ_ana|_2 = {ic_l2_error:.2e}")
print(f"  Relative error: {ic_relative_error:.2e}")
print(f"  Note: Boundaries set to 0 (Dirichlet BCs), analytical solution is for infinite domain")

# Check boundary values
print(f"\n  Boundary values:")
print(f"    ψ[0] = {psi[0]:.2e} (should be 0.0)")
print(f"    ψ[-1] = {psi[-1]:.2e} (should be 0.0)")
print(f"    |ψ_ana[0]| = {np.abs(psi0_ana[0]):.2e}")
print(f"    |ψ_ana[-1]| = {np.abs(psi0_ana[-1]):.2e}")

if ic_max_error > 1e-10:
    print(f"\n  WARNING: Large error at t=0!")
    print(f"  This suggests a potential issue with:")
    print(f"    - Analytical solution formula")
    print(f"    - Initial condition implementation")
    print(f"    - Boundary condition enforcement")
elif ic_max_error > 1e-12:
    print(f"  Note: Small error likely due to floating-point precision")
else:
    print(f"  ✓ Initial condition matches analytical solution (interior points)!")

# Verify analytical solution satisfies NLS equation at t=0 (interior)
print(f"\n  Verifying analytical solution satisfies NLS equation at t=0 (interior)...")
dt_check = 1e-6
psi_t0 = psi_soliton_analytical(x, 0.0, A, v, x0, g, hbar, m)
psi_t1 = psi_soliton_analytical(x, dt_check, A, v, x0, g, hbar, m)
dpsi_dt_ana = (psi_t1 - psi_t0) / dt_check

# Compute RHS of NLS using the library function (interior only for comparison)
psi_t0_bc = enforce_bc(psi_t0)
rhs_nls = nls_rhs(psi_t0_bc)

# Check residual (interior only)
residual = np.abs(dpsi_dt_ana[1:-1] - rhs_nls[1:-1])
residual_max = np.max(residual)
residual_l2 = np.sqrt(np.mean(residual**2))

print(f"    NLS equation residual (interior):")
print(f"      Max: {residual_max:.2e}")
print(f"      L2:  {residual_l2:.2e}")
if residual_max < 1e-5:
    print(f"    ✓ Analytical solution satisfies NLS equation (interior)!")
else:
    print(f"    ⚠ Large residual - may be due to boundary effects or formula")

print("="*60)

# Domain information
L_domain = xmax - xmin
print(f"\nDomain: [{xmin}, {xmax}], length L = {L_domain:.1f} (Dirichlet BCs: ψ=0 at boundaries)")
print(f"Soliton amplitude: A = {A:.2f}")
print(f"Soliton velocity: v = {v:.3f}")
print(f"Nonlinearity parameter: g = {g:.3f} (focusing NLS)")
print(f"Soliton width: ~{1/(A*np.sqrt(-g)):.3f}")
print(f"At final time t={Nt*dt:.3f}, soliton center will be at: {x0 + v*Nt*dt:.3f}")
print(f"\nUsing {time_method.upper()} time stepping with BSPF for spatial differentiation.")
print(f"Dirichlet BCs: ψ(x_min) = ψ(x_max) = 0")
print(f"Comparing numerical (Dirichlet BCs) vs analytical (infinite domain) solution.")
print(f"Errors will grow as soliton approaches boundaries due to BCs.")

# =========================================================
# PLOTTING SETUP
# =========================================================
# Create plot configuration with real, imaginary, and magnitude
plot_config = {
    'layout': 'single',
    'plots': [
        {
            'type': 'density',
            'ax': 0,
            'label': '|ψ|²',
            'color': 'black',
            'style': '-',
            'linewidth': 2,
            'ylim': (-1.2 * A, 1.2 * A**2),  # Accommodate real/imag parts (can be negative) and |ψ|²
            'xlabel': 'x',
            'ylabel': 'Magnitude',
            'title_template': f'{time_method.upper()}-BSPF NLS Solver (Dirichlet BCs, g={g:.2f})   t={{t:.3f}}'
        },
        {
            'type': 'custom',
            'ax': 0,
            'label': 'Re[ψ]',
            'color': 'green',
            'style': '-',
            'linewidth': 1.5,
            'alpha': 0.7,
            'callback': lambda t, psi, psi_ana, V, cd: psi.real
        },
        {
            'type': 'custom',
            'ax': 0,
            'label': 'Im[ψ]',
            'color': 'blue',
            'style': '-',
            'linewidth': 1.5,
            'alpha': 0.7,
            'callback': lambda t, psi, psi_ana, V, cd: psi.imag
        }
    ]
}

# Create animator
animator = SchrodingerAnimator1D(x, figsize=(10, 8), fps=30, dpi=100)
animator.setup_plot(plot_config)

# Analytical solution function
def psi_analytical_func(t):
    return psi_soliton_analytical(x, t, A, v, x0, g, hbar, m)

# Progress callback for error reporting
def progress_callback(n, t, psi, error):
    if n % 100 == 0 or n == 1:
        if error is not None:
            max_err = np.max(error[1:-1])  # Interior only
            l2_err = np.sqrt(np.mean(error[1:-1]**2))  # Interior only
            print(f"t={t:.3f}: max |ψ_num - ψ_ana| (interior) = {max_err:.2e}, L2 = {l2_err:.2e}")

# =========================================================
# RUN SIMULATION
# =========================================================
# Create output folder for saving plots
os.makedirs(output_folder, exist_ok=True)
print(f'\nPlots will be saved to "{output_folder}" folder every {save_interval} steps')
print('Starting simulation...')

# Use non-interactive backend for faster execution
plt.ioff()

# Run simulation with static plot saving
psi_final = animator.animate_simulation(
    rhs_func=nls_rhs,
    psi_init=psi,
    dt=dt,
    n_steps=Nt,
    time_method=time_method,
    output_path=None,  # No video, just static plots
    enforce_bc=enforce_bc,
    psi_analytical_func=psi_analytical_func,
    progress_callback=progress_callback,
    save_static=True,
    save_interval=save_interval,
    static_output_dir=output_folder
)

print('Simulation completed!')

print(f"\nFinal error statistics (interior points only):")
final_psi_a = psi_soliton_analytical(x, Nt*dt, A, v, x0, g, hbar, m)
final_err = np.abs(psi_final - final_psi_a)
print(f"  Max error: {np.max(final_err[1:-1]):.2e}")
print(f"  L2 error: {np.sqrt(np.mean(final_err[1:-1]**2)):.2e}")
print(f"\nNote: Boundaries are enforced to 0 (Dirichlet BCs), so boundary errors are not meaningful.")

