import numpy as np
import matplotlib.pyplot as plt

# --- 1. Parameters ---
SIZE = 200              # Grid points (x-axis)
DX = 1.0
C = 1.0
S = 0.9                 # Courant Number (< 1.0 for visible dispersion error)
DT = S * DX / C         
STEPS = 100             # Time steps (y-axis)
SIGMA = 10.0            # Pulse width
X_START = 50.0 * DX     # Pulse starting center

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

# --- 3. Simulation and Data Collection ---

def run_space_time_plot():
    
    # Initialization
    x_grid = np.arange(SIZE) * DX
    ez = np.zeros(SIZE)
    hy = np.zeros(SIZE)

    # 1. Initialize E-field
    ez = np.exp(-0.5 * ((x_grid - X_START) / SIGMA) ** 2)

    # 2. Initialize H-field for One-Way (+x) Wave Propagation (Corrected sign)
    x_grid_h = (np.arange(SIZE) + 0.5) * DX
    dist_h_init = C * (-0.5 * DT) 
    hy = -1.0 * np.exp(-0.5 * ((x_grid_h - (X_START + dist_h_init)) / SIGMA) ** 2)
    
    # Data storage arrays
    E_history = []
    Error_history = []
    
    # Main Loop
    for t_step in range(STEPS):
        
        # FDTD Updates
        hy[:-1] += S * (ez[1:] - ez[:-1]) # Update H
        ez[1:]  += S * (hy[1:] - hy[:-1]) # Update E
        
        # PEC Boundary Conditions
        ez[0] = 0
        ez[-1] = 0
        
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
    
    # --- 4. Plotting ---
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot A: Electric Field (Ez) Space-Time Plot
    im1 = ax1.imshow(E_matrix, aspect='auto', 
                     extent=[0, SIZE*DX, STEPS*DT, 0], 
                     cmap='viridis', vmin=-1.0, vmax=1.0)
    
    ax1.set_title(f"FDTD Electric Field ($E_z$) Space-Time Plot (S={S})")
    ax1.set_ylabel("Time (arb. units)")
    ax1.axvline(X_START, color='w', linestyle='--', linewidth=1) # Initial source location
    fig.colorbar(im1, ax=ax1, label='Field Amplitude')
    
    # Plot B: Error Space-Time Plot
    # Using 'seismic' to clearly show positive (red) and negative (blue) errors around zero.
    max_err = np.max(np.abs(Error_matrix))
    im2 = ax2.imshow(Error_matrix, aspect='auto', 
                     extent=[0, SIZE*DX, STEPS*DT, 0], 
                     cmap='seismic', vmin=-max_err, vmax=max_err)

    ax2.set_title("Numerical Dispersion Error ($E_{FDTD} - E_{Exact}$)")
    ax2.set_ylabel("Time (arb. units)")
    ax2.set_xlabel("Space (x-axis)")
    fig.colorbar(im2, ax=ax2, label='Error Amplitude')
    
    fig.tight_layout()
    plt.savefig('2d_spacetime_error_comparison.png')
    plt.close(fig)

    print("2D Space-Time plots saved to '2d_spacetime_error_comparison.png'")

if __name__ == "__main__":
    run_space_time_plot()