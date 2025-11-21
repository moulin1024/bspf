"""
1D Extrapolation with PiecewiseBSPF1D.

This script demonstrates high-order extrapolation using PiecewiseBSPF1D,
which is particularly useful for functions with discontinuities. It shows:
    - Extrapolation by half grid size beyond domain boundaries
    - Comparison with analytical solution
    - How piecewise reconstruction handles extrapolation near breakpoints
    - Convergence study of extrapolation accuracy

Run from repository root:
    python examples/basic/extrapolate_piecewise_1d.py
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.linalg as sla

from bspf import PiecewiseBSPF1D


# ============================================================================
# Parameters
# ============================================================================

# BSPF parameters
DEGREE = 5                    # B-spline polynomial degree
NUM_BOUNDARY_POINTS = DEGREE  # Number of boundary points
N_BASIS = 2 * DEGREE         # Number of basis functions
REG_PARAM = 1e-3             # Tikhonov regularization strength (lambda)

# Grid parameters
DOMAIN = [0, 2*np.pi]        # Domain [a, b]
NUM_POINTS = 2048            # Grid resolution for main computation

# Jump parameters
JUMP_LOCATION = np.pi        # Location of the jump discontinuity

# Breakpoints for piecewise reconstruction (must include jump location)
BREAKPOINTS = [JUMP_LOCATION]  # Discontinuity in test function

# Convergence study parameters
GRID_SIZES = [100,200,300]  # Grid sizes for convergence study
FIT_START, FIT_END = 10, 15  # Indices for convergence rate fitting

# Extrapolation parameters
EXTRAPOLATION_FACTOR = 0.5  # Extrapolate by half grid size


# ============================================================================
# Test Function Definition
# ============================================================================

def define_test_function():
    """
    Define a test function with sin functions on each side of a jump.
    
    The function has different sin behaviors on each side:
    - Left side (x < JUMP_LOCATION): sin(2x) - higher frequency
    - Right side (x >= JUMP_LOCATION): sin(x) + 1 - shifted lower frequency
    
    This creates a jump where both the function value and frequency change.
    
    Returns
    -------
    func : callable
        Function f(x) as a NumPy-compatible function
    func_deriv : callable
        Derivative f'(x) as a NumPy-compatible function (smooth within segments)
    """
    def func(x):
        """Sin function with jump at JUMP_LOCATION."""
        result = np.zeros_like(x)
        mask_left = x < JUMP_LOCATION
        mask_right = x >= JUMP_LOCATION
        
        # Left side: sin(2x) - higher frequency
        result[mask_left] = np.sin(2 * x[mask_left])
        
        # Right side: sin(x) + 1 - shifted lower frequency
        result[mask_right] = np.sin(x[mask_right]) + 1
        
        return result
    
    def func_deriv(x):
        """Derivative (smooth within each segment)."""
        result = np.zeros_like(x)
        mask_left = x < JUMP_LOCATION
        mask_right = x >= JUMP_LOCATION
        
        # Left side derivative: 2*cos(2x)
        result[mask_left] = 2 * np.cos(2 * x[mask_left])
        
        # Right side derivative: cos(x)
        result[mask_right] = np.cos(x[mask_right])
        
        return result
    
    return func, func_deriv


# ============================================================================
# Helper Functions
# ============================================================================

def get_spline_coefficients(model, y, lam=0.0):
    """
    Get B-spline coefficients P from a bspf1d model by solving the KKT system.
    
    Parameters
    ----------
    model : bspf1d
        BSPF model instance
    y : Array
        Function values on the grid
    lam : float
        Regularization parameter
        
    Returns
    -------
    P : Array
        B-spline coefficients
    """
    # Build RHS (same as in differentiate method)
    rhs_2bw = 2.0 * (model.BW @ y)
    dY = model.end.BND @ y
    rhs = np.concatenate((rhs_2bw, dY))
    
    # Solve KKT system
    lu, piv = model._kkt_lu(lam)
    sol = sla.lu_solve((lu, piv), rhs, overwrite_b=False)
    P = sol[:model.basis.B0.shape[0]]
    
    return P


def evaluate_spline_at_points(model, P, x_eval):
    """
    Evaluate B-spline at arbitrary points using coefficients.
    
    Parameters
    ----------
    model : bspf1d
        BSPF model instance
    P : Array
        B-spline coefficients
    x_eval : Array
        Points at which to evaluate the spline
        
    Returns
    -------
    y_eval : Array
        Spline values at x_eval
    """
    # Evaluate basis functions at x_eval
    B_eval = model.basis._evaluate_splines_vectorized(x_eval, deriv_order=0)
    
    # Evaluate spline: y = B^T @ P
    y_eval = B_eval.T @ P
    
    return y_eval


def evaluate_piecewise_spline_at_points(pw_model, f, x_eval, lam=0.0):
    """
    Evaluate piecewise B-spline at arbitrary points, including extrapolation.
    
    Parameters
    ----------
    pw_model : PiecewiseBSPF1D
        Piecewise BSPF model instance
    f : Array
        Function values on the full grid
    x_eval : Array
        Points at which to evaluate the spline
    lam : float
        Regularization parameter
        
    Returns
    -------
    y_eval : Array
        Spline values at x_eval
    """
    y_eval = np.zeros_like(x_eval)
    
    # For each evaluation point, find which segment it belongs to
    for i, xp in enumerate(x_eval):
        # Find the segment that contains or should handle this point
        # For extrapolation, use the nearest segment
        seg_idx = None
        
        # Check if point is in domain
        if xp < pw_model.x[0]:
            # Left extrapolation - use first segment
            seg_idx = 0
        elif xp > pw_model.x[-1]:
            # Right extrapolation - use last segment
            seg_idx = len(pw_model.segments) - 1
        else:
            # Find segment containing this point
            for idx, seg in enumerate(pw_model.segments):
                i0, i1 = seg["i0"], seg["i1"]
                if pw_model.x[i0] <= xp <= pw_model.x[i1]:
                    seg_idx = idx
                    break
        
        if seg_idx is not None:
            seg = pw_model.segments[seg_idx]
            op = seg["op"]
            i0, i1 = seg["i0"], seg["i1"]
            
            # Get function values for this segment
            f_seg = f[i0:i1 + 1]
            
            # Get coefficients for this segment
            P_seg = get_spline_coefficients(op, f_seg, lam=lam)
            
            # Evaluate at this point
            y_eval[i] = evaluate_spline_at_points(op, P_seg, np.array([xp]))[0]
    
    return y_eval


# ============================================================================
# Main Computation
# ============================================================================

def main():
    """Main computation and visualization."""
    
    # Define test function
    test_func, test_func_deriv = define_test_function()
    
    # Create grid
    x = np.linspace(DOMAIN[0], DOMAIN[1], NUM_POINTS, endpoint=True)
    dx = x[1] - x[0]
    
    # Initialize PiecewiseBSPF1D model
    pw_model = PiecewiseBSPF1D(
        degree=DEGREE,
        x=x,
        breakpoints=BREAKPOINTS,
        order=DEGREE,
        n_basis=N_BASIS,
        num_boundary_points=NUM_BOUNDARY_POINTS
    )
    
    # Compute function values
    y = test_func(x)
    y_deriv_exact = test_func_deriv(x)
    
    # Compute derivatives and spline approximations
    y_deriv_pw, _, y_spline_pw = pw_model.differentiate_1_2(y, lam=REG_PARAM)
    
    # ========================================================================
    # Extrapolation Around Jump
    # ========================================================================
    
    extrapolation_distance = EXTRAPOLATION_FACTOR * dx
    
    # Create extrapolation points around the jump
    x_jump_left_extrap = JUMP_LOCATION - extrapolation_distance
    x_jump_right_extrap = JUMP_LOCATION + extrapolation_distance
    
    x_jump_extrap = np.array([x_jump_left_extrap, x_jump_right_extrap])
    
    # Evaluate piecewise spline at jump extrapolation points
    y_jump_extrap_pw = evaluate_piecewise_spline_at_points(pw_model, y, x_jump_extrap, lam=REG_PARAM)
    
    # Get analytical values at extrapolation points
    y_jump_extrap_exact = test_func(x_jump_extrap)
    
    # Errors around jump
    error_pw_jump_left = np.abs(y_jump_extrap_pw[0] - y_jump_extrap_exact[0])
    error_pw_jump_right = np.abs(y_jump_extrap_pw[1] - y_jump_extrap_exact[1])
    
    print("=" * 70)
    print("Extrapolation Results Around Jump (half grid size):")
    print("=" * 70)
    print(f"  Jump location: x = {JUMP_LOCATION:.8f}")
    print(f"  Left of jump (x = {x_jump_left_extrap:.8f}):")
    print(f"    PiecewiseBSPF: {y_jump_extrap_pw[0]:.10f}, Error = {error_pw_jump_left:.6e}")
    print(f"    Exact:          {y_jump_extrap_exact[0]:.10f}")
    print(f"  Right of jump (x = {x_jump_right_extrap:.8f}):")
    print(f"    PiecewiseBSPF: {y_jump_extrap_pw[1]:.10f}, Error = {error_pw_jump_right:.6e}")
    print(f"    Exact:          {y_jump_extrap_exact[1]:.10f}")
    print("=" * 70)
    
    # ========================================================================
    # Convergence Study for Extrapolation
    # ========================================================================
    
    print("\nRunning extrapolation convergence study around jump...")
    errors_pw_jump_left = []
    errors_pw_jump_right = []
    
    for n_points in GRID_SIZES:
        # Create grid
        x_test = np.linspace(DOMAIN[0], DOMAIN[1], n_points, endpoint=True)
        dx_test = x_test[1] - x_test[0]
        
        # Compute function values
        y_test = test_func(x_test)
        
        # Piecewise BSPF
        pw_test = PiecewiseBSPF1D(
            degree=DEGREE,
            x=x_test,
            breakpoints=BREAKPOINTS,
            order=DEGREE,
            n_basis=N_BASIS,
            num_boundary_points=NUM_BOUNDARY_POINTS
        )
        
        # Calculate extrapolation points around jump
        extrap_dist_test = EXTRAPOLATION_FACTOR * dx_test
        x_jump_left_test = JUMP_LOCATION - extrap_dist_test
        x_jump_right_test = JUMP_LOCATION + extrap_dist_test
        
        x_jump_extrap_test = np.array([x_jump_left_test, x_jump_right_test])
        
        # Evaluate spline at jump extrapolation points
        y_jump_extrap_pw_test = evaluate_piecewise_spline_at_points(pw_test, y_test, x_jump_extrap_test, lam=REG_PARAM)
        
        # Get analytical values
        y_jump_extrap_exact_test = test_func(x_jump_extrap_test)
        
        # Compute jump errors
        error_pw_jump_left_test = np.abs(y_jump_extrap_pw_test[0] - y_jump_extrap_exact_test[0])
        error_pw_jump_right_test = np.abs(y_jump_extrap_pw_test[1] - y_jump_extrap_exact_test[1])
        
        errors_pw_jump_left.append(error_pw_jump_left_test)
        errors_pw_jump_right.append(error_pw_jump_right_test)
        
        print(f"N = {n_points:5d} | Jump: Left={error_pw_jump_left_test:.6e}, Right={error_pw_jump_right_test:.6e}")
    
    # Compute convergence rates
    # x_fit = GRID_SIZES[FIT_START:FIT_END]
    # y_fit_pw = np.array(errors_pw_jump_left[FIT_START:FIT_END])
    # log_x_fit = np.log(x_fit)
    # log_y_fit_pw = np.log(y_fit_pw)
    
    # coeffs_pw = np.polyfit(log_x_fit, log_y_fit_pw, 1)
    # slope_pw = coeffs_pw[0]
    # intercept_pw = coeffs_pw[1]
    
    # print(f"\nConvergence rate around jump (from indices {FIT_START}-{FIT_END}):")
    # print(f"  PiecewiseBSPF (left of jump): {slope_pw:.3f}")
    
    # ========================================================================
    # Visualization
    # ========================================================================
    
    # Set up plotting parameters
    plt.rcParams.update({
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
        'figure.titlesize': 20,
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    fig = plt.figure(figsize=(14, 10))
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Get segment ranges for visualization
    segment_ranges = []
    for seg in pw_model.segments:
        i0, i1 = seg["i0"], seg["i1"]
        x_left = pw_model.x[i0]
        x_right = pw_model.x[i1]
        segment_ranges.append((x_left, x_right))
    
    # Evaluate splines around jump region for visualization
    x_jump_zoom = np.linspace(JUMP_LOCATION - 5*extrapolation_distance,
                               JUMP_LOCATION + 5*extrapolation_distance, 1000)
    y_jump_pw = evaluate_piecewise_spline_at_points(pw_model, y, x_jump_zoom, lam=REG_PARAM)
    y_jump_exact = test_func(x_jump_zoom)
    
    # (a) Full view around jump with extrapolation
    ax1 = plt.subplot(2, 2, 1)
    # Plot domain points
    mask_domain = (x >= JUMP_LOCATION - 5*extrapolation_distance) & (x <= JUMP_LOCATION + 5*extrapolation_distance)
    ax1.plot(x[mask_domain], y[mask_domain], 'o', label='$f(x)$ (domain)', markersize=4, color='black', alpha=0.6)
    ax1.plot(x_jump_zoom, y_jump_exact, '-', label='Exact', linewidth=2, color='black')
    ax1.plot(x_jump_zoom, y_jump_pw, '--', label='PiecewiseBSPF', linewidth=1.5, alpha=0.8, color=default_colors[0])
    
    # Mark segment ranges with shaded regions (after plotting to get proper y-limits)
    for i, (x_left_seg, x_right_seg) in enumerate(segment_ranges):
        if i == 0:
            label_seg = 'Left BSPF range'
        elif i == len(segment_ranges) - 1:
            label_seg = 'Right BSPF range'
        else:
            label_seg = f'Segment {i+1}'
        ax1.axvspan(x_left_seg, x_right_seg, alpha=0.15, color=default_colors[i % len(default_colors)], 
                    label=label_seg if i < 2 else '', zorder=0)
    
    ax1.axvline(JUMP_LOCATION, linestyle='--', color='orange', linewidth=2, alpha=0.7, label='Jump')
    ax1.axvline(x_jump_left_extrap, linestyle=':', color='red', linewidth=1.5, alpha=0.7, label='Extrapolation points')
    ax1.axvline(x_jump_right_extrap, linestyle=':', color='red', linewidth=1.5, alpha=0.7)
    ax1.plot(x_jump_extrap, y_jump_extrap_pw, 'o', markersize=10, color=default_colors[0], zorder=5, label='Extrapolated')
    ax1.plot(x_jump_extrap, y_jump_extrap_exact, 'x', markersize=12, color='black', zorder=5, label='Exact extrap')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$f(x)$')
    ax1.legend(ncol=2, fontsize=10, loc='upper left')
    ax1.set_title('(a) Reconstruction Around Jump', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # (b) Very tight zoom around jump interface showing segment boundaries and extrapolation
    ax2 = plt.subplot(2, 2, 2)
    # Zoom in very close to the jump to show the interface - very tight zoom
    zoom_width = 0.6 * extrapolation_distance  # Very tight zoom around jump
    x_interface_zoom = np.linspace(JUMP_LOCATION - zoom_width,
                                    JUMP_LOCATION + zoom_width, 1000)
    y_interface_exact = test_func(x_interface_zoom)
    y_interface_pw = evaluate_piecewise_spline_at_points(pw_model, y, x_interface_zoom, lam=REG_PARAM)
    
    # Plot exact function
    ax2.plot(x_interface_zoom, y_interface_exact, '-', label='Exact', linewidth=2, color='black')
    ax2.plot(x_interface_zoom, y_interface_pw, '--', label='PiecewiseBSPF', linewidth=1.5, alpha=0.8, color=default_colors[0])
    
    # Mark segment ranges and their boundaries
    if len(segment_ranges) >= 2:
        # Left segment
        x_left_seg_start, x_left_seg_end = segment_ranges[0]
        # Right segment
        x_right_seg_start, x_right_seg_end = segment_ranges[-1]
        
        # Shaded regions for each segment
        ax2.axvspan(x_left_seg_start, x_left_seg_end, alpha=0.15, color=default_colors[0], 
                    label='Left BSPF domain', zorder=0)
        ax2.axvspan(x_right_seg_start, x_right_seg_end, alpha=0.15, color=default_colors[1], 
                    label='Right BSPF domain', zorder=0)
        
        # Mark segment boundaries
        ax2.axvline(x_left_seg_end, linestyle='-.', color=default_colors[0], linewidth=2, alpha=0.7, 
                    label='Left segment end')
        ax2.axvline(x_right_seg_start, linestyle='-.', color=default_colors[1], linewidth=2, alpha=0.7, 
                    label='Right segment start')
        
        # Show extrapolation from left segment end to interface (if there's a gap)
        if x_left_seg_end < JUMP_LOCATION:
            x_left_extrap_range = np.linspace(x_left_seg_end, JUMP_LOCATION, 200)
            y_left_extrap_range = evaluate_piecewise_spline_at_points(pw_model, y, x_left_extrap_range, lam=REG_PARAM)
            ax2.plot(x_left_extrap_range, y_left_extrap_range, linewidth=2.5, color=default_colors[0], 
                    alpha=0.7, label='Left BSPF extrapolation', zorder=4, linestyle=':')
        
        # Show extrapolation from interface to right segment start (if there's a gap)
        if JUMP_LOCATION < x_right_seg_start:
            x_right_extrap_range = np.linspace(JUMP_LOCATION, x_right_seg_start, 200)
            y_right_extrap_range = evaluate_piecewise_spline_at_points(pw_model, y, x_right_extrap_range, lam=REG_PARAM)
            ax2.plot(x_right_extrap_range, y_right_extrap_range, linewidth=2.5, color=default_colors[1], 
                    alpha=0.7, label='Right BSPF extrapolation', zorder=4, linestyle=':')
        
        # Mark the extrapolation points
        ax2.plot(x_jump_left_extrap, y_jump_extrap_pw[0], 'o', markersize=10, color=default_colors[0], 
                zorder=5, label='Left extrap point')
        ax2.plot(x_jump_right_extrap, y_jump_extrap_pw[1], 's', markersize=8, color=default_colors[1], 
                zorder=5, label='Right extrap point')
    
    # Mark the jump interface
    ax2.axvline(JUMP_LOCATION, linestyle='--', color='orange', linewidth=3, alpha=0.8, label='Jump interface')
    
    # Plot domain points in this region
    mask_interface = (x >= JUMP_LOCATION - zoom_width) & (x <= JUMP_LOCATION + zoom_width)
    ax2.plot(x[mask_interface], y[mask_interface], 'o', markersize=3, color='black', alpha=0.4, zorder=3)
    
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$f(x)$')
    ax2.legend(ncol=2, fontsize=9, loc='upper right')
    ax2.set_title('(b) Interface: Segment Boundaries & Extrapolation', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # (c) Very tight zoom around jump showing both segments meeting at interface
    ax3 = plt.subplot(2, 2, 3)
    if len(segment_ranges) >= 2:
        x_left_seg_start, x_left_seg_end = segment_ranges[0]
        x_right_seg_start, x_right_seg_end = segment_ranges[-1]
        
        # Very tight zoom around the jump
        tight_zoom_width = 0.8 * extrapolation_distance
        x_tight_zoom = np.linspace(JUMP_LOCATION - tight_zoom_width,
                                    JUMP_LOCATION + tight_zoom_width, 600)
        y_tight_exact = test_func(x_tight_zoom)
        y_tight_pw = evaluate_piecewise_spline_at_points(pw_model, y, x_tight_zoom, lam=REG_PARAM)
        
        # Plot exact function
        ax3.plot(x_tight_zoom, y_tight_exact, '-', label='Exact', linewidth=2.5, color='black')
        ax3.plot(x_tight_zoom, y_tight_pw, '--', label='PiecewiseBSPF', linewidth=2, alpha=0.8, color=default_colors[0])
        
        # Mark segment boundaries if they're in view
        if x_left_seg_end >= JUMP_LOCATION - tight_zoom_width:
            ax3.axvline(x_left_seg_end, linestyle='-.', color=default_colors[0], linewidth=2.5, alpha=0.8, 
                        label='Left segment end')
        if x_right_seg_start <= JUMP_LOCATION + tight_zoom_width:
            ax3.axvline(x_right_seg_start, linestyle='-.', color=default_colors[1], linewidth=2.5, alpha=0.8, 
                        label='Right segment start')
        
        # Show extrapolation regions if visible
        if x_left_seg_end < JUMP_LOCATION and x_left_seg_end >= JUMP_LOCATION - tight_zoom_width:
            x_left_extrap_vis = np.linspace(max(x_left_seg_end, JUMP_LOCATION - tight_zoom_width), JUMP_LOCATION, 200)
            y_left_extrap_vis = evaluate_piecewise_spline_at_points(pw_model, y, x_left_extrap_vis, lam=REG_PARAM)
            ax3.plot(x_left_extrap_vis, y_left_extrap_vis, linewidth=2.5, color=default_colors[0], 
                    alpha=0.7, label='Left extrapolation', zorder=4, linestyle=':')
        
        if x_right_seg_start > JUMP_LOCATION and x_right_seg_start <= JUMP_LOCATION + tight_zoom_width:
            x_right_extrap_vis = np.linspace(JUMP_LOCATION, min(x_right_seg_start, JUMP_LOCATION + tight_zoom_width), 200)
            y_right_extrap_vis = evaluate_piecewise_spline_at_points(pw_model, y, x_right_extrap_vis, lam=REG_PARAM)
            ax3.plot(x_right_extrap_vis, y_right_extrap_vis, linewidth=2.5, color=default_colors[1], 
                    alpha=0.7, label='Right extrapolation', zorder=4, linestyle=':')
        
        # Mark the jump interface prominently
        ax3.axvline(JUMP_LOCATION, linestyle='--', color='orange', linewidth=3, alpha=0.9, label='Jump interface')
        
        # Mark extrapolation points if in view
        if abs(x_jump_left_extrap - JUMP_LOCATION) <= tight_zoom_width:
            ax3.plot(x_jump_left_extrap, y_jump_extrap_pw[0], 'o', markersize=12, color=default_colors[0], 
                    zorder=6, label='Left extrap point', markeredgewidth=2, markeredgecolor='black')
            ax3.plot(x_jump_left_extrap, y_jump_extrap_exact[0], 'x', markersize=14, color='black', zorder=6, linewidth=2)
        if abs(x_jump_right_extrap - JUMP_LOCATION) <= tight_zoom_width:
            ax3.plot(x_jump_right_extrap, y_jump_extrap_pw[1], 's', markersize=10, color=default_colors[1], 
                    zorder=6, label='Right extrap point', markeredgewidth=2, markeredgecolor='black')
            ax3.plot(x_jump_right_extrap, y_jump_extrap_exact[1], 'x', markersize=14, color='black', zorder=6, linewidth=2)
        
        # Plot domain points in this tight region
        mask_tight = (x >= JUMP_LOCATION - tight_zoom_width) & (x <= JUMP_LOCATION + tight_zoom_width)
        ax3.plot(x[mask_tight], y[mask_tight], 'o', markersize=4, color='black', alpha=0.5, zorder=3)
        
        ax3.set_xlabel('$x$')
        ax3.set_ylabel('$f(x)$')
        ax3.legend(ncol=2, fontsize=8, loc='best')
        ax3.set_title('(c) Tight Zoom: Jump Interface', fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # (d) Extrapolation errors around jump
    ax4 = plt.subplot(2, 2, 4)
    ax4.semilogy(GRID_SIZES, errors_pw_jump_left, '.-', linewidth=1.5, 
                label='Left of jump', color=default_colors[0], alpha=1, markersize=6)
    ax4.semilogy(GRID_SIZES, errors_pw_jump_right, linewidth=1.5, 
                label='Right of jump', color=default_colors[0], alpha=0.6, markersize=6, linestyle='--', marker='.')
    ax4.axvline(NUM_POINTS, linestyle='--', color='gray', linewidth=1)
    ax4.set_xlabel('$N$')
    ax4.set_ylabel('$|Error|$')
    ax4.legend(ncol=1, fontsize=12)
    ax4.set_title('(d) Jump Extrapolation Error', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

