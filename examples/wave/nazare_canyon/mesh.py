import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, griddata, RegularGridInterpolator
from scipy.optimize import minimize, minimize_scalar
from typing import Tuple, Optional
import warnings
import os
import h5py

# =========================================================
# CONFIGURATION PARAMETERS
# =========================================================

# Input data
data_dir = 'data'
HDF5_FILE = os.path.join(data_dir, "nazare_data.h5")  # HDF5 file containing all data

# =========================================================
# Helper function for HDF5 data access
# =========================================================
def load_hdf5_data(hdf5_file, dataset_path):
    """
    Load a dataset from HDF5 file.
    
    Args:
        hdf5_file: Path to HDF5 file
        dataset_path: Path to dataset (e.g., 'coastline/coastline_smooth_scaled')
    
    Returns:
        numpy array with the data
    """
    with h5py.File(hdf5_file, 'r') as f:
        return f[dataset_path][:]

def get_hdf5_attrs(hdf5_file, group_path):
    """
    Get attributes from an HDF5 group.
    
    Args:
        hdf5_file: Path to HDF5 file
        group_path: Path to group (e.g., 'scaling')
    
    Returns:
        Dictionary of attributes
    """
    with h5py.File(hdf5_file, 'r') as f:
        return dict(f[group_path].attrs)

# Load initial data from HDF5 file for preview
try:
    bathy = load_hdf5_data(HDF5_FILE, 'bathymetry/bathy_interpolated')
    x_coords = load_hdf5_data(HDF5_FILE, 'coordinates/x_coords_scaled')
    y_coords = load_hdf5_data(HDF5_FILE, 'coordinates/y_coords_scaled')
    coastline = load_hdf5_data(HDF5_FILE, 'coastline/coastline_smooth_scaled')
    
    print(f"  Loaded interpolated bathymetry: shape {bathy.shape}, range {np.nanmin(bathy):.2f} to {np.nanmax(bathy):.2f} m")
    print(f"  Loaded X coordinates: shape {x_coords.shape}, range {np.nanmin(x_coords):.2f} to {np.nanmax(x_coords):.2f} m")
    print(f"  Loaded Y coordinates: shape {y_coords.shape}, range {np.nanmin(y_coords):.2f} to {np.nanmax(y_coords):.2f} m")
except Exception as e:
    print(f"  Warning: Could not load HDF5 data: {e}")
    bathy = None
    x_coords = None
    y_coords = None
    coastline = None



# Mesh parameters
NX = 256              # Grid points along coast (ξ direction)
NY = 256             # Grid points normal to coast (η direction)
CLUSTERING_FACTOR = 3  # Clustering towards coast (0 = uniform, >0 = clustered)
XI_CLUSTERING_FACTOR = 0  # Horizontal clustering along ξ direction (0 = uniform, >0 = clustered)
XI_CLUSTERING_TARGET_X = 0.4  # Target x location (in 0-1 scaled coordinates) to cluster towards
L_NORMAL = 0.1       # Normal extension depth (not used in mapping, kept for compatibility)
OUTER_OFFSET = 0.6    # Vertical offset for outer boundary (fraction of coordinate range if < 1.0, absolute if >= 1.0)

# Spline parameters
SPLINE_SMOOTHING = 0.0  # Spline smoothing parameter (0 = interpolation, >0 = smoothing)
SPLINE_DEGREE = 3       # B-spline degree (cubic)

# Outer curve parameters
# =========================================================
# Top boundary rotation control
# =========================================================
# OUTER_ANGLE: Rotation angle of the outer boundary (top line) in degrees
#   - 0.0   = horizontal (parallel to x-axis)
#   - > 0   = counterclockwise rotation (top boundary slopes upward to the right)
#   - < 0   = clockwise rotation (top boundary slopes downward to the right)
#   - ±90   = vertical (perpendicular to x-axis)
# 
# This controls the orientation of the outer boundary curve, which affects
# the mesh quality and Jacobian determinants. Use optimize_outer_angle()
# to automatically find the best angle for your coastline.
# =========================================================
OUTER_ANGLE = 00.0  # Rotation angle of outer curve (top boundary) in degrees
OUTER_HORIZONTAL_SHIFT = 0.0  # Horizontal shift of outer boundary (fraction of coordinate range if < 1.0, absolute if >= 1.0)
                              # Positive = shift right, negative = shift left
                              # Example: 0.1 = shift 10% of coordinate range to the right
OUTER_VERTICAL_SHIFT = 0.0  # Vertical shift of outer boundary (fraction of coordinate range if < 1.0, absolute if >= 1.0)
                            # Positive = shift up, negative = shift down
                            # Example: 0.1 = shift 10% of coordinate range upward

# Plotting parameters
SHOW_CLUSTERING_DETAILS = False  # Show clustering visualization
SAVE_FILENAME = 'coastline_mesh.png'  # Filename to save the plot (None to not save)
PLOT_BATHYMETRY = True  # Whether to overlay and plot bathymetry in parametric space
BATHY_VMIN = -300  # Minimum depth for colormap (meters)
BATHY_VMAX = 0     # Maximum depth for colormap (meters)

# =========================================================
# Core modular function for coastline mesh generation
# =========================================================
def generate_coastline_mesh(
    coast_x: np.ndarray,
    coast_y: np.ndarray,
    nx: int = 50,
    ny: int = 30,
    clustering_factor: float = 0.0,
    xi_clustering_factor: float = 0.0,
    xi_clustering_target_x: float = 0.5,
    L_normal: float = 0.1,
    outer_offset: float = 0.8,
    outer_angle: float = 0.0,  # Angle in degrees (0 = horizontal, positive = counterclockwise) - used as initial guess
    outer_horizontal_shift: float = 0.0,  # Horizontal shift of outer boundary (fraction if < 1.0, absolute if >= 1.0)
    outer_vertical_shift: float = 0.0,  # Vertical shift of outer boundary (fraction if < 1.0, absolute if >= 1.0)
    outer_curve_params: Optional[np.ndarray] = None,  # Control points for flexible outer curve (None = use straight line)
    n_outer_control: int = 3,  # Number of control points for flexible outer curve
    spline_smoothing: float = 0.0,
    spline_degree: int = 3,
    normal_relaxation: float = 0.3  # Relaxation factor: 0 = strict normal, 1 = fully relaxed (blend with outer curve direction)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Generate a coastline mesh with normal-based expansion.
    
    Args:
        coast_x: X coordinates of coastline points
        coast_y: Y coordinates of coastline points
        nx: Number of grid points in ξ direction (along coast)
        ny: Number of grid points in η direction (normal to coast)
        clustering_factor: Clustering factor for physical space (0 = uniform, >0 = cluster towards coast)
        L_normal: Normal extension depth (thickness of normal layer)
        outer_offset: Vertical offset for outer boundary
        outer_angle: Rotation angle of outer boundary in degrees
        outer_horizontal_shift: Horizontal shift of outer boundary (fraction if < 1.0, absolute if >= 1.0)
        outer_vertical_shift: Vertical shift of outer boundary (fraction if < 1.0, absolute if >= 1.0)
        spline_smoothing: Spline smoothing parameter (0 = interpolation, >0 = smoothing)
        spline_degree: B-spline degree (default 3 for cubic)
        normal_relaxation: Relaxation factor (0 = strict normal, 1 = fully relaxed). 
                          Higher values blend normal direction with outer curve direction to avoid negative Jacobians.
    
    Returns:
        X: Physical X coordinates (ny, nx)
        Y: Physical Y coordinates (ny, nx)
        XI: Parameter space ξ coordinates (ny, nx)
        ETA: Parameter space η coordinates (ny, nx)
        X_xi: Jacobian component ∂X/∂ξ (ny, nx, 2)
        X_eta: Jacobian component ∂X/∂η (ny, nx, 2)
        info: Dictionary with additional information (coastline functions, etc.)
    """
    # =========================================================
    # 1. Construct B-spline from coastline points
    # =========================================================
    tck, u = splprep([coast_x, coast_y], s=spline_smoothing, k=spline_degree)
    
    # Find target_xi: the ξ value that corresponds to xi_clustering_target_x
    # We need to find which ξ gives C(ξ)[0] ≈ xi_clustering_target_x
    if xi_clustering_factor > 0.0:
        # Sample the coastline to find target_xi
        xi_test = np.linspace(0, 1, 1000)
        C_test = np.array(splev(xi_test, tck)).T
        x_test = C_test[:, 0]
        # Find closest match
        idx_closest = np.argmin(np.abs(x_test - xi_clustering_target_x))
        target_xi = xi_test[idx_closest]
    else:
        target_xi = 0.5  # Default, not used
    
    def C(xi):
        """海岸线位置 C(ξ)"""
        x, y = splev(xi, tck)
        return np.stack([x, y], axis=-1)  # (...,2)
    
    def C1(xi):
        """一阶导 C'(ξ)"""
        dx, dy = splev(xi, tck, der=1)
        return np.stack([dx, dy], axis=-1)
    
    def C2(xi):
        """二阶导 C''(ξ)"""
        ddx, ddy = splev(xi, tck, der=2)
        return np.stack([ddx, ddy], axis=-1)
    
    def tangent_normal(xi):
        """单位切向 t_hat, 单位法向 n_hat"""
        Cp = C1(xi)
        speed = np.linalg.norm(Cp, axis=-1, keepdims=True)
        t_hat = Cp / speed
        n_hat = np.stack([-t_hat[...,1], t_hat[...,0]], axis=-1)
        return t_hat, n_hat
    
    def normal_derivative(xi):
        """n'(ξ) 的解析表达式（需要 C'(ξ), C''(ξ)）"""
        Cp  = C1(xi)
        Cpp = C2(xi)
        k   = np.linalg.norm(Cp, axis=-1, keepdims=True)  # |C'|
        dot = np.sum(Cp * Cpp, axis=-1, keepdims=True)   # C'·C''
        # t_hat' = C''/|C'| - C' (C'·C'')/|C'|^3
        t_prime = Cpp / k - Cp * (dot / (k**3))
        # n = R t, n' = R t'
        n_prime = np.stack([-t_prime[...,1], t_prime[...,0]], axis=-1)
        return n_prime
    
    # =========================================================
    # 2. Define outer curve
    # =========================================================
    # Get coastline endpoints for reference
    C_start = C(0.0)
    C_end = C(1.0)
    x_start, y_start = C_start[0], C_start[1]
    x_end, y_end = C_end[0], C_end[1]
    
    # Reference point: midpoint of coastline endpoints, offset upward
    y_max = np.max(coast_y)
    y_min = np.min(coast_y)
    x_max = np.max(coast_x)
    x_min = np.min(coast_x)
    # Calculate coordinate ranges for relative offsets
    y_range = y_max - y_min
    x_range = x_max - x_min
    # Use outer_offset as a fraction of the coordinate range
    # If outer_offset < 1, treat as fraction; if >= 1, treat as absolute
    if outer_offset < 1.0:
        offset_absolute = outer_offset * max(y_range, x_range)
    else:
        offset_absolute = outer_offset
    
    # Calculate vertical shift
    # If outer_vertical_shift < 1, treat as fraction of y_range; if >= 1, treat as absolute
    if abs(outer_vertical_shift) < 1.0:
        vertical_shift_absolute = outer_vertical_shift * y_range
    else:
        vertical_shift_absolute = outer_vertical_shift
    
    y_ref = y_max + offset_absolute + vertical_shift_absolute
    
    # Calculate horizontal shift
    # If outer_horizontal_shift < 1, treat as fraction of x_range; if >= 1, treat as absolute
    if abs(outer_horizontal_shift) < 1.0:
        horizontal_shift_absolute = outer_horizontal_shift * x_range
    else:
        horizontal_shift_absolute = outer_horizontal_shift
    
    x_ref = (x_start + x_end) / 2.0 + horizontal_shift_absolute
    
    # Initialize outer_tck for info dictionary
    outer_tck = None
    
    # Define outer curve - either flexible B-spline or straight line
    if outer_curve_params is not None:
        # Flexible B-spline outer curve
        # outer_curve_params: array of shape (n_outer_control, 2) with (x, y) control points
        # Or: array of shape (n_outer_control,) with y-offsets relative to baseline
        if outer_curve_params.ndim == 1:
            # Interpret as y-offsets relative to a baseline curve
            # Create control points: x follows coastline, y is offset
            u_control = np.linspace(0, 1, n_outer_control)
            C_control = C(u_control)
            x_control = C_control[:, 0]
            # y_control = baseline_y + offset
            # Calculate offset relative to coordinate range
            if outer_offset < 1.0:
                offset_absolute = outer_offset * max(y_range, x_range)
            else:
                offset_absolute = outer_offset
            
            # Calculate vertical shift
            if abs(outer_vertical_shift) < 1.0:
                vertical_shift_absolute = outer_vertical_shift * y_range
            else:
                vertical_shift_absolute = outer_vertical_shift
            
            baseline_y = y_max + offset_absolute + vertical_shift_absolute
            
            # Apply horizontal shift to control points
            if abs(outer_horizontal_shift) < 1.0:
                horizontal_shift_absolute = outer_horizontal_shift * x_range
            else:
                horizontal_shift_absolute = outer_horizontal_shift
            x_control = x_control + horizontal_shift_absolute
            
            y_control = baseline_y + outer_curve_params
            control_points = np.column_stack([x_control, y_control])
        else:
            # Full control points provided - apply horizontal and vertical shifts
            control_points = outer_curve_params.copy()
            if abs(outer_horizontal_shift) < 1.0:
                horizontal_shift_absolute = outer_horizontal_shift * x_range
            else:
                horizontal_shift_absolute = outer_horizontal_shift
            if abs(outer_vertical_shift) < 1.0:
                vertical_shift_absolute = outer_vertical_shift * y_range
            else:
                vertical_shift_absolute = outer_vertical_shift
            control_points[:, 0] = control_points[:, 0] + horizontal_shift_absolute
            control_points[:, 1] = control_points[:, 1] + vertical_shift_absolute
        
        # Fit B-spline to control points
        outer_tck, _ = splprep([control_points[:, 0], control_points[:, 1]], 
                               s=0, k=min(spline_degree, len(control_points)-1))
        
        def outer_curve(xi):
            """Flexible B-spline outer curve"""
            x_outer, y_outer = splev(xi, outer_tck)
            # Handle both scalar and array inputs
            if np.isscalar(xi):
                return np.array([x_outer, y_outer])
            else:
                return np.stack([x_outer, y_outer], axis=-1)
        
        def outer_curve_derivative(xi):
            """Derivative of outer curve"""
            dx, dy = splev(xi, outer_tck, der=1)
            if np.isscalar(xi):
                return np.array([dx, dy])
            else:
                return np.stack([dx, dy], axis=-1)
        
        use_flexible_curve = True
    else:
        # Straight line outer curve (original behavior)
        angle_rad = np.deg2rad(outer_angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        def outer_curve(xi):
            """
            外层引导曲线 O(ξ)：
            Straight line at specified angle, passing through reference point.
            For each xi, finds the point on the line corresponding to the coastline's x-coordinate.
            """
            Cpos = C(xi)
            x_coast = Cpos[...,0]
            
            # For each xi, find the point on the angled line with the same x-coordinate
            # Line equation: y = y_ref + tan(angle) * (x - x_ref)
            if abs(cos_angle) > 1e-10:  # Not vertical (angle not near ±90°)
                # Use x-coast as x-coordinate, compute corresponding y on the line
                x_proj = x_coast
                y_proj = y_ref + np.tan(angle_rad) * (x_proj - x_ref)
            else:  # Vertical line (angle near ±90°)
                # For vertical line, x is fixed, y varies
                x_proj = np.full_like(x_coast, x_ref)
                # Use coastline y-coordinate to determine position along vertical line
                y_coast = Cpos[...,1]
                # Use offset_absolute that was calculated earlier
                y_proj = y_ref + (y_coast - y_start) * np.sign(sin_angle) * offset_absolute / (y_max - y_min + 1e-10)
            
            return np.stack([x_proj, y_proj], axis=-1)
        
        def outer_curve_derivative(xi):
            """Derivative of straight line outer curve"""
            Cp = C1(xi)
            angle_rad = np.deg2rad(outer_angle)
            cos_angle = np.cos(angle_rad)
            if abs(cos_angle) > 1e-10:
                tan_angle = np.tan(angle_rad)
                return np.stack([Cp[:,0], tan_angle * Cp[:,0]], axis=-1)
            else:
                return np.stack([np.zeros_like(Cp[:,0]), Cp[:,1]], axis=-1)
        
        use_flexible_curve = False
    
    # =========================================================
    # 3. Generate parameter space grid (always uniform)
    # =========================================================
    # Parameter space should always be uniform [0,1] for visualization
    # Clustering is applied during physical mapping, not in parameter space
    xis_uniform = np.linspace(0.0, 1.0, nx)
    etas_uniform = np.linspace(0.0, 1.0, ny)
    
    # Create uniform parameter space grid
    XI, ETA = np.meshgrid(xis_uniform, etas_uniform)
    
    # =========================================================
    # 4. Generate physical mesh and Jacobian
    # =========================================================
    # Pass uniform parameter space and clustering parameters
    # Clustering will be applied during physical mapping
    X, Y = _map_grid_internal(xis_uniform, etas_uniform, C, tangent_normal, outer_curve, 
                              L_normal, clustering_factor, normal_relaxation,
                              xi_clustering_factor, target_xi)
    X_xi, X_eta = _jacobian_internal(xis_uniform, etas_uniform, C, C1, tangent_normal, normal_derivative,
                                     outer_curve, outer_curve_derivative, L_normal, 
                                     clustering_factor, normal_relaxation,
                                     xi_clustering_factor, target_xi)
    
    # =========================================================
    # 5. Package additional info
    # =========================================================
    info = {
        'C': C,
        'C1': C1,
        'C2': C2,
        'tangent_normal': tangent_normal,
        'normal_derivative': normal_derivative,
        'outer_curve': outer_curve,
        'outer_curve_derivative': outer_curve_derivative,
        'use_flexible_curve': use_flexible_curve,
        'coast_x': coast_x,
        'coast_y': coast_y,
        'tck': tck,
        'y_max': y_max
    }
    if use_flexible_curve:
        info['outer_tck'] = outer_tck
        info['outer_curve_params'] = outer_curve_params
    
    return X, Y, XI, ETA, X_xi, X_eta, info


# =========================================================
# Internal helper functions
# =========================================================
def eta_clustering_transform(eta, clustering_factor=0.0):
    """
    将均匀的 η ∈ [0,1] 转换为聚类的 η_clustered ∈ [0,1]
    用于在物理空间产生聚类，同时保持参数空间均匀
    """
    if clustering_factor == 0.0:
        return eta, np.ones_like(eta)
    
    exp_factor = np.exp(clustering_factor)
    eta_clustered = (np.exp(clustering_factor * eta) - 1.0) / (exp_factor - 1.0)
    deta_clustered_deta = (clustering_factor * np.exp(clustering_factor * eta)) / (exp_factor - 1.0)
    
    return eta_clustered, deta_clustered_deta

def xi_clustering_transform(xi, clustering_factor=0.0, target_xi=0.5):
    """
    Transform uniform ξ ∈ [0,1] to clustered ξ_clustered ∈ [0,1],
    clustering points around target_xi.

    clustering_factor:
        0   -> uniform
        >0  -> increasing clustering strength

    target_xi:
        point in [0,1] where we want the highest grid density
    """
    xi = np.asarray(xi, dtype=float)

    # No clustering
    if clustering_factor <= 0.0:
        return xi, np.ones_like(xi)

    # Clamp target_xi into [0,1]
    t = float(target_xi)
    t = np.clip(t, 0.0, 1.0)

    # Shape parameters:
    # alpha controls how strong the clustering is (0 -> none, ->1 strong)
    # p > 1 gives a "power law" clustering profile
    k = clustering_factor
    alpha = k / (1.0 + k)   # in (0,1)
    p = 2.0                 # fixed exponent; you can tweak if you like

    xi_clustered = np.empty_like(xi)
    dxi_dxi = np.empty_like(xi)
    eps = 1e-12

    # --- Special cases: cluster at edges if target_xi ~ 0 or ~ 1 ---
    if t <= eps:
        # Cluster near left edge (0)
        v = xi                     # v ∈ [0,1]
        G = (1.0 - alpha) * v + alpha * (v ** p)
        xi_clustered = G
        dxi_dxi = (1.0 - alpha) + alpha * p * (v ** (p - 1))
        return xi_clustered, dxi_dxi

    if t >= 1.0 - eps:
        # Cluster near right edge (1)
        w = 1.0 - xi               # w ∈ [0,1], w=0 at right edge
        G = (1.0 - alpha) * w + alpha * (w ** p)
        xi_clustered = 1.0 - G
        dxi_dxi = (1.0 - alpha) + alpha * p * (w ** (p - 1))
        return xi_clustered, dxi_dxi

    # --- General case: cluster around interior target t ∈ (0,1) ---
    mask_left = xi <= t
    mask_right = ~mask_left

    # Left side: xi ∈ [0, t]  → more points near xi = t
    if np.any(mask_left):
        xL = xi[mask_left]
        u = (t - xL) / t          # u = 0 at target, 1 at left edge
        G = (1.0 - alpha) * u + alpha * (u ** p)
        xi_clustered[mask_left] = t - t * G
        # d(xi_clustered)/d(xi)
        dxi_dxi[mask_left] = (1.0 - alpha) + alpha * p * (u ** (p - 1))

    # Right side: xi ∈ [t, 1] → more points near xi = t
    if np.any(mask_right):
        xR = xi[mask_right]
        v = (xR - t) / (1.0 - t)  # v = 0 at target, 1 at right edge
        G = (1.0 - alpha) * v + alpha * (v ** p)
        xi_clustered[mask_right] = t + (1.0 - t) * G
        dxi_dxi[mask_right] = (1.0 - alpha) + alpha * p * (v ** (p - 1))

    return xi_clustered, dxi_dxi

def _map_grid_internal(xis, etas, C, tangent_normal, outer_curve, L_normal, clustering_factor, normal_relaxation=0.3,
                       xi_clustering_factor=0.0, target_xi=0.5):
    """
    Simple transfinite interpolation: P(ξ, η) = (1-η) * C(ξ) + η * O(ξ)
    This guarantees no negative Jacobians if curves don't cross.
    
    Clustering is applied here: uniform parameter space (ξ, η) is transformed to clustered
    values before evaluating C and O, but parameter space itself remains uniform.
    """
    XI, ETA = np.meshgrid(xis, etas)
    xi_flat = XI.ravel()  # Uniform parameter space
    eta_flat = ETA.ravel()  # Uniform parameter space
    
    # Apply clustering transformations to get clustered parameter values
    # These clustered values are used for physical mapping, but XI/ETA remain uniform
    if xi_clustering_factor > 0.0:
        xi_clustered_flat, _ = xi_clustering_transform(xi_flat, xi_clustering_factor, target_xi)
    else:
        xi_clustered_flat = xi_flat
    
    eta_clustered, _ = eta_clustering_transform(eta_flat, clustering_factor)
    
    # Evaluate coastline and outer curve using CLUSTERED xi values
    Cpos = C(xi_clustered_flat)
    O = outer_curve(xi_clustered_flat)
    
    # Simple linear interpolation: guaranteed valid if curves don't cross
    X_flat = (1.0 - eta_clustered[:, None]) * Cpos + eta_clustered[:, None] * O
    
    X = X_flat[:,0].reshape(XI.shape)
    Y = X_flat[:,1].reshape(XI.shape)
    return X, Y


def _jacobian_internal(xis, etas, C, C1, tangent_normal, normal_derivative, 
                       outer_curve, outer_curve_derivative, L_normal, clustering_factor, normal_relaxation=0.3,
                       xi_clustering_factor=0.0, target_xi=0.5):
    """
    Jacobian for simple transfinite interpolation: P(ξ, η) = (1-η) * C(ξ_clustered) + η * O(ξ_clustered)
    where ξ_clustered = clustering_transform(ξ)
    
    ∂P/∂ξ = (1-η) * C'(ξ_clustered) * d(ξ_clustered)/d(ξ) + η * O'(ξ_clustered) * d(ξ_clustered)/d(ξ)
    ∂P/∂η = O(ξ_clustered) - C(ξ_clustered)
    """
    XI, ETA = np.meshgrid(xis, etas)
    xi_flat = XI.ravel()  # Uniform parameter space
    eta_flat = ETA.ravel()  # Uniform parameter space
    
    # Apply clustering transformations
    if xi_clustering_factor > 0.0:
        xi_clustered_flat, dxi_clustered_dxi = xi_clustering_transform(xi_flat, xi_clustering_factor, target_xi)
    else:
        xi_clustered_flat = xi_flat
        dxi_clustered_dxi = np.ones_like(xi_flat)
    
    eta_clustered, deta_clustered_deta = eta_clustering_transform(eta_flat, clustering_factor)
    
    # Evaluate using CLUSTERED xi values
    Cpos = C(xi_clustered_flat)
    Cp = C1(xi_clustered_flat)  # C'(ξ_clustered)
    O = outer_curve(xi_clustered_flat)
    O_prime = outer_curve_derivative(xi_clustered_flat)  # O'(ξ_clustered)
    
    # ∂P/∂ξ = (1-η) * C'(ξ_clustered) * d(ξ_clustered)/d(ξ) + η * O'(ξ_clustered) * d(ξ_clustered)/d(ξ)
    # Chain rule: dP/dξ = dP/d(ξ_clustered) * d(ξ_clustered)/d(ξ)
    X_xi_flat = ((1.0 - eta_clustered[:, None]) * Cp + eta_clustered[:, None] * O_prime) * dxi_clustered_dxi[:, None]
    
    # ∂P/∂η = O(ξ_clustered) - C(ξ_clustered)
    # Note: This is in terms of eta_clustered, so we need to multiply by d(eta_clustered)/d(eta)
    D = O - Cpos
    X_eta_flat = D * deta_clustered_deta[:, None]
    
    shape = XI.shape + (2,)
    return X_xi_flat.reshape(shape), X_eta_flat.reshape(shape)

# =========================================================
# Plotting function
# =========================================================
def plot_coastline_mesh(X, Y, XI, ETA, X_xi, X_eta, info, clustering_factor=0.0, 
                        show_clustering_details=True, save_filename=None,
                        bathy_resampled=None, bathy_vmin=-350, bathy_vmax=0,
                        hdf5_file=None):
    """
    Plot the coastline mesh visualization.
    
    Args:
        X, Y: Physical coordinates (ny, nx)
        XI, ETA: Parameter space coordinates (ny, nx)
        X_xi, X_eta: Jacobian components (ny, nx, 2)
        info: Dictionary with additional information
        clustering_factor: Clustering factor used
        show_clustering_details: Whether to show clustering visualization
        save_filename: Optional filename to save the plot
        bathy_resampled: Resampled bathymetry on mesh points (ny, nx) or None
        bathy_vmin: Minimum depth for colormap
        bathy_vmax: Maximum depth for colormap
    """
    ny, nx = X.shape
    # Jacobian determinant: det(J) = (∂x/∂ξ)(∂y/∂η) - (∂x/∂η)(∂y/∂ξ)
    # Where X_xi = [∂x/∂ξ, ∂y/∂ξ] and X_eta = [∂x/∂η, ∂y/∂η]
    det_J = X_xi[...,0] * X_eta[...,1] - X_xi[...,1] * X_eta[...,0]
    
    # Load scaling factors from HDF5 to transform back to original coordinates
    scaling_factors = None
    if hdf5_file is not None:
        try:
            scaling_factors = get_hdf5_attrs(hdf5_file, 'scaling')
            print(f"\n  Loaded scaling factors from HDF5:")
            print(f"    X range: [{scaling_factors['x_min']:.2f}, {scaling_factors['x_max']:.2f}] m")
            print(f"    Y range: [{scaling_factors['y_min']:.2f}, {scaling_factors['y_max']:.2f}] m")
        except Exception as e:
            print(f"  Warning: Could not load scaling factors: {e}")
            scaling_factors = None
    
    # Transform coordinates back to original if scaling factors are available
    if scaling_factors is not None:
        X_original = X * scaling_factors['x_range'] + scaling_factors['x_min']
        Y_original = Y * scaling_factors['y_range'] + scaling_factors['y_min']
        # Transform coastline coordinates
        coast_x_original = info['coast_x'] * scaling_factors['x_range'] + scaling_factors['x_min']
        coast_y_original = info['coast_y'] * scaling_factors['y_range'] + scaling_factors['y_min']
        # Transform spline coastline
        C_plot_scaled = info['C'](np.linspace(0,1,200))
        C_plot_original = C_plot_scaled.copy()
        C_plot_original[:, 0] = C_plot_scaled[:, 0] * scaling_factors['x_range'] + scaling_factors['x_min']
        C_plot_original[:, 1] = C_plot_scaled[:, 1] * scaling_factors['y_range'] + scaling_factors['y_min']
        
        # Transform Jacobian components to original coordinates
        # X_xi and X_eta are in scaled coordinates, need to scale by coordinate ranges
        X_xi_original = X_xi.copy()
        X_xi_original[..., 0] *= scaling_factors['x_range']  # ∂x_original/∂ξ
        X_xi_original[..., 1] *= scaling_factors['y_range']  # ∂y_original/∂ξ
        
        X_eta_original = X_eta.copy()
        X_eta_original[..., 0] *= scaling_factors['x_range']  # ∂x_original/∂η
        X_eta_original[..., 1] *= scaling_factors['y_range']  # ∂y_original/∂η
        
        # Construct full Jacobian matrix J: (ξ,η) -> (x,y)
        # J = [∂x/∂ξ  ∂x/∂η]
        #     [∂y/∂ξ  ∂y/∂η]
        J = np.zeros((ny, nx, 2, 2))
        J[:, :, 0, 0] = X_xi_original[:, :, 0]  # ∂x/∂ξ
        J[:, :, 0, 1] = X_eta_original[:, :, 0]  # ∂x/∂η
        J[:, :, 1, 0] = X_xi_original[:, :, 1]  # ∂y/∂ξ
        J[:, :, 1, 1] = X_eta_original[:, :, 1]  # ∂y/∂η
        
        # Jacobian determinant in original coordinates
        det_J_original = X_xi_original[...,0] * X_eta_original[...,1] - X_xi_original[...,1] * X_eta_original[...,0]
    else:
        # Use scaled coordinates if no scaling factors available
        X_original = X
        Y_original = Y
        coast_x_original = info['coast_x']
        coast_y_original = info['coast_y']
        C_plot_original = info['C'](np.linspace(0,1,200))
        
        # Construct Jacobian in scaled coordinates
        J = np.zeros((ny, nx, 2, 2))
        J[:, :, 0, 0] = X_xi[:, :, 0]  # ∂x/∂ξ
        J[:, :, 0, 1] = X_eta[:, :, 0]  # ∂x/∂η
        J[:, :, 1, 0] = X_xi[:, :, 1]  # ∂y/∂ξ
        J[:, :, 1, 1] = X_eta[:, :, 1]  # ∂y/∂η
        det_J_original = det_J
    
    # Detailed investigation of why Jacobian might be negative
    if np.all(det_J < 0) or np.any(det_J < 0):
        print(f"\n" + "=" * 60)
        print("INVESTIGATING JACOBIAN SIGN ISSUE")
        print("=" * 60)
        
        # Sample a few points to analyze
        sample_indices = [(0, 0), (0, nx//2), (0, nx-1), (ny//2, 0), (ny-1, 0)]
        
        print(f"\nSample points analysis:")
        for j, i in sample_indices:
            if j < ny and i < nx:
                xi_val = XI[j, i]
                eta_val = ETA[j, i]
                C_pos = info['C'](xi_val)
                O_pos = info['outer_curve'](xi_val)
                
                print(f"\n  Point (j={j}, i={i}): ξ={xi_val:.3f}, η={eta_val:.3f}")
                print(f"    Coastline C(ξ) = [{C_pos[0]:.6f}, {C_pos[1]:.6f}]")
                print(f"    Outer curve O(ξ) = [{O_pos[0]:.6f}, {O_pos[1]:.6f}]")
                print(f"    Vector O-C = [{O_pos[0]-C_pos[0]:.6f}, {O_pos[1]-C_pos[1]:.6f}]")
                print(f"    ∂P/∂ξ = X_xi[{j},{i}] = [{X_xi[j,i,0]:.6f}, {X_xi[j,i,1]:.6f}]")
                print(f"    ∂P/∂η = X_eta[{j},{i}] = [{X_eta[j,i,0]:.6f}, {X_eta[j,i,1]:.6f}]")
                print(f"    det(J) = {det_J[j,i]:.6f}")
                
                # Check cross product direction
                cross_z = X_xi[j,i,0] * X_eta[j,i,1] - X_xi[j,i,1] * X_eta[j,i,0]
                print(f"    Cross product (X_xi × X_eta)_z = {cross_z:.6f}")
                
                # Check if vectors form a right-handed system
                # For right-handed: if ξ points right and η points up, cross should be positive
                xi_dir = "right" if X_xi[j,i,0] > 0 else "left"
                eta_dir = "up" if X_eta[j,i,1] > 0 else "down"
                print(f"    ∂P/∂ξ points mostly {xi_dir} (x={X_xi[j,i,0]:.6f})")
                print(f"    ∂P/∂η points mostly {eta_dir} (y={X_eta[j,i,1]:.6f})")
                
                # Expected: if ξ increases along coast (left to right) and η increases from coast to outer (down to up)
                # Then for right-handed system, det should be positive
                # If det is negative, either:
                # 1. ξ direction is reversed (coast goes right to left)
                # 2. η direction is reversed (outer is below coast)
                # 3. Coordinate system is left-handed
                
        print(f"\n  Checking coordinate system orientation:")
        print(f"    Coastline X range: [{info['coast_x'].min():.6f}, {info['coast_x'].max():.6f}]")
        print(f"    Coastline Y range: [{info['coast_y'].min():.6f}, {info['coast_y'].max():.6f}]")
        
        # Check if outer curve is above or below coastline
        C_sample = info['C'](np.linspace(0, 1, 10))
        O_sample = info['outer_curve'](np.linspace(0, 1, 10))
        avg_C_y = np.mean(C_sample[:, 1])
        avg_O_y = np.mean(O_sample[:, 1])
        print(f"    Average coastline Y: {avg_C_y:.6f}")
        print(f"    Average outer curve Y: {avg_O_y:.6f}")
        if avg_O_y > avg_C_y:
            print(f"    → Outer curve is ABOVE coastline (η increases upward)")
        else:
            print(f"    → Outer curve is BELOW coastline (η increases downward)")
        
        # Check coastline direction
        C_start = info['C'](0.0)
        C_end = info['C'](1.0)
        coast_dx = C_end[0] - C_start[0]
        print(f"    Coastline from ξ=0 to ξ=1: dx = {coast_dx:.6f}")
        if coast_dx > 0:
            print(f"    → Coastline goes LEFT to RIGHT (ξ increases rightward)")
        else:
            print(f"    → Coastline goes RIGHT to LEFT (ξ increases leftward)")
        
        print(f"\n  For a RIGHT-HANDED coordinate system:")
        print(f"    If ξ increases rightward and η increases upward → det(J) should be POSITIVE")
        print(f"    If ξ increases leftward or η increases downward → det(J) will be NEGATIVE")
        
        if np.all(det_J < 0):
            print(f"\n  ⚠️  ALL Jacobians are negative!")
            print(f"     This suggests the coordinate system orientation is reversed.")
            print(f"     Possible fixes:")
            print(f"     1. Reverse ξ direction (flip coastline parameterization)")
            print(f"     2. Reverse η direction (swap C and O in mapping)")
            print(f"     3. Use left-handed coordinate system (accept negative det)")
    
    # Create three-panel figure
    fig = plt.figure(figsize=(10, 5))
    
    # Also create separate figure for Jacobian matrix components
    fig_jacobian = True
    
    # Left panel: Physical mesh with bathymetry overlay
    ax1 = plt.subplot(1, 2, 1)
    
    # Plot bathymetry background if available (use original coordinates)
    if bathy_resampled is not None:
        im1 = ax1.pcolormesh(X_original, Y_original, bathy_resampled, shading='gouraud', 
                            cmap='terrain', vmin=bathy_vmin, vmax=bathy_vmax, alpha=0.7)
        # plt.colorbar(im1, ax=ax1, label='Depth (m)')
    
    # Overlay mesh grid (use original coordinates)
    for j in range(0, ny, 2):  # Show every other line for clarity
        ax1.plot(X_original[j,:], Y_original[j,:], "b-", lw=0.5, alpha=0.2)
    for i in range(0, nx, 2):
        ax1.plot(X_original[:,i], Y_original[:,i], "b-", lw=0.5, alpha=0.2)
    
    # Plot coastline points and spline (use original coordinates)
    # ax1.plot(coast_x_original, coast_y_original, "ko", ms=4, label="input points")
    # Spline coastline (in original coordinates)
    ax1.plot(C_plot_original[:,0], C_plot_original[:,1], "r-", lw=1, label="coastline")
    ax1.legend()
    ax1.set_title("Physical Mesh (x, y)", fontsize=14, fontweight='bold')
    
    # Set axis labels based on whether we have scaling factors
    if scaling_factors is not None:
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        # Set limits based on actual mesh coordinates in original space
        x_min, x_max = X_original.min(), X_original.max()
        y_min, y_max = Y_original.min(), Y_original.max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        margin_x = x_range * 0.05
        margin_y = y_range * 0.05
        ax1.set_xlim(x_min - margin_x, x_max + margin_x)
        ax1.set_ylim(y_min - margin_y, y_max + margin_y)
    else:
        ax1.set_xlabel("X (scaled 0-1)")
        ax1.set_ylabel("Y (scaled 0-1)")
        # Set limits for scaled coordinates (0-1 range with small margin)
        margin = 0.05
        ax1.set_xlim(-margin, 1 + margin)
        ax1.set_ylim(-margin, 1 + margin)
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.3)
    
    # Middle panel: Parameter space with bathymetry
    ax2 = plt.subplot(1, 2, 2)
    
    if bathy_resampled is not None:
        # Plot bathymetry in parametric space
        im2 = ax2.pcolormesh(XI, ETA, bathy_resampled, shading='gouraud', 
                             cmap='terrain', vmin=bathy_vmin, vmax=bathy_vmax)
        plt.colorbar(im2, ax=ax2, label='Depth (m)')
        
        # Overlay mesh grid
        for j in range(0, ny, 5):
            ax2.plot(XI[j,:], ETA[j,:], "b-", lw=0.3, alpha=0.2)
        for i in range(0, nx, 5):
            ax2.plot(XI[:,i], ETA[:,i], "b-", lw=0.3, alpha=0.2)
        
        ax2.set_title("Parameter Space (ξ, η)", fontsize=14, fontweight='bold')
    else:
        # Just show mesh grid
        for j in range(ny):
            ax2.plot(XI[j,:], ETA[j,:], "g-", lw=0.6, alpha=0.7)
        for i in range(nx):
            ax2.plot(XI[:,i], ETA[:,i], "g-", lw=0.6, alpha=0.7)
        ax2.set_title("Parameter Space (ξ, η)", fontsize=14, fontweight='bold')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.set_xlabel("ξ")
    ax2.set_ylabel("η")
    # ax2.grid(True, alpha=0.3)
    
    # # Third panel: Jacobian determinant
    # ax3 = plt.subplot(1, 3, 3)
    # im3 = ax3.imshow(det_J_original, origin='lower', aspect='auto', 
    #                 cmap='RdYlGn', interpolation='bilinear')
    # plt.colorbar(im3, ax=ax3, label='Jacobian Determinant')
    
    # # Overlay contour at det_J = 0 to highlight invalid regions
    # if np.any(det_J_original < 0):
    #     contour = ax3.contour(det_J_original, levels=[0.0], colors='black', linewidths=2, linestyles='--')
    #     ax3.clabel(contour, inline=True, fontsize=8, fmt='det=0')
    
    # ax3.set_title("Jacobian Determinant\n(red=negative, green=positive)", 
    #               fontsize=14, fontweight='bold')
    # ax3.set_xlabel("ξ index")
    # ax3.set_ylabel("η index")
    # ax3.grid(True, alpha=0.3)
    
    # # Add text with statistics
    # n_negative = np.sum(det_J_original < 0)
    # stats_text = f"Min: {det_J_original.min():.4f}\nMax: {det_J_original.max():.4f}\nMean: {det_J_original.mean():.4f}\nNegative: {n_negative}"
    # ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
    #         fontsize=10, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    if save_filename:
        plt.savefig(save_filename, dpi=150)
    plt.show()
    
    # Diagnostics
    X_eta_boundary = X_eta[0,:,:]
    X_xi_boundary = X_xi[0,:,:]
    dot = np.sum(X_eta_boundary * X_xi_boundary, axis=-1)
    print("Max |X_eta·X_xi| at boundary (should be ~0):", np.abs(dot).max())
    print(f"Jacobian determinant range: [{det_J.min():.4f}, {det_J.max():.4f}]")
    print(f"Mean |det(J)|: {np.abs(det_J).mean():.4f}")
    
    # Print Jacobian statistics (no warning)
    n_negative = np.sum(det_J < 0)
    print(f"\nJacobian determinant statistics:")
    print(f"  Minimum: {det_J.min():.6f}")
    print(f"  Maximum: {det_J.max():.6f}")
    print(f"  Mean: {det_J.mean():.6f}")
    print(f"  Negative values: {n_negative} ({100.0 * n_negative / det_J.size:.2f}%)")
    if n_negative > 0:
        print(f"  ⚠️  Negative Jacobians indicate invalid mesh regions (see plot panel 3)")
    else:
        print(f"  ✓ All Jacobian determinants are positive (mesh is valid)")
    
    # Clustering information
    if clustering_factor > 0 and show_clustering_details:
        etas = ETA[:, 0]  # Get eta values from parameter space
        eta_clustered, _ = eta_clustering_transform(etas, clustering_factor)
        
        print(f"\nClustering information:")
        print(f"  Factor: {clustering_factor}")
        print(f"  Parameter space: UNIFORM (eta = {etas[:3]} ... {etas[-3:]})")
        print(f"  Transformed eta_clustered: {eta_clustered[:3]} ... {eta_clustered[-3:]}")
        
        # Calculate physical spacing along eta direction (normal to coast)
        physical_spacing_coast = np.mean([np.linalg.norm([X[1,i]-X[0,i], Y[1,i]-Y[0,i]]) for i in range(nx)])
        physical_spacing_mid = np.mean([np.linalg.norm([X[ny//2+1,i]-X[ny//2,i], Y[ny//2+1,i]-Y[ny//2,i]]) for i in range(nx)])
        physical_spacing_outer = np.mean([np.linalg.norm([X[-1,i]-X[-2,i], Y[-1,i]-Y[-2,i]]) for i in range(nx)])
        
        print(f"  Physical spacing near coast (η=0): {physical_spacing_coast:.6f}")
        print(f"  Physical spacing at middle (η=0.5): {physical_spacing_mid:.6f}")
        print(f"  Physical spacing near outer (η=1): {physical_spacing_outer:.6f}")
        
        if physical_spacing_coast > 0:
            ratio_outer_coast = physical_spacing_outer / physical_spacing_coast
            ratio_mid_coast = physical_spacing_mid / physical_spacing_coast
            print(f"  Ratio (outer/coast): {ratio_outer_coast:.2f}x")
            print(f"  Ratio (mid/coast): {ratio_mid_coast:.2f}x")
            
            # Expected ratio based on clustering
            # With clustering, eta_clustered near 0 should be smaller, near 1 should be larger
            # The derivative deta_clustered_deta shows the compression factor
            deta_coast = (clustering_factor * np.exp(clustering_factor * 0.0)) / (np.exp(clustering_factor) - 1.0)
            deta_outer = (clustering_factor * np.exp(clustering_factor * 1.0)) / (np.exp(clustering_factor) - 1.0)
            expected_ratio = deta_outer / deta_coast if deta_coast > 0 else 1.0
            print(f"  Expected ratio from clustering: {expected_ratio:.2f}x")
            
            if abs(ratio_outer_coast - expected_ratio) > 0.1:
                print(f"  ⚠️  WARNING: Physical spacing ratio ({ratio_outer_coast:.2f}x) doesn't match expected ({expected_ratio:.2f}x)")
                print(f"     This may indicate the distance between curves varies significantly along ξ")
        
        # Clustering visualization
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(etas, np.arange(len(etas)), 'b-o', markersize=4, label='Parameter space (uniform)')
        eta_clustered, _ = eta_clustering_transform(etas, clustering_factor)
        ax1.plot(eta_clustered, np.arange(len(eta_clustered)), 'r--', label='Transformed (for mapping)', lw=2, alpha=0.7)
        ax1.set_xlabel('η')
        ax1.set_ylabel('Grid point index')
        ax1.set_title(f'Parameter Space: Uniform\nPhysical Space: Clustered (factor={clustering_factor})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
        
        ax2.plot(X[0, :], Y[0, :], 'r-', lw=3, label='Coast (η=0)')
        for j in [0, ny//4, ny//2, 3*ny//4, ny-1]:
            ax2.plot(X[j, :], Y[j, :], 'b-', lw=1, alpha=0.6)
        ax2.set_aspect('equal')
        ax2.set_title('Physical Mesh (showing clustering)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_filename:
            base_name = save_filename.rsplit('.', 1)[0]
            plt.savefig(f'{base_name}_clustering.png', dpi=150)
        plt.show()


# =========================================================
# Function to find safe L_normal that ensures positive Jacobian
# =========================================================
def find_safe_L_normal(coast_x, coast_y, nx=50, ny=30, clustering_factor=0.0,
                       L_normal_initial=0.1, outer_offset=0.8, outer_angle=0.0,
                       spline_smoothing=0.0, spline_degree=3,
                       min_jacobian_threshold=1e-6, max_iterations=50, verbose=True):
    """
    Find the maximum safe L_normal value that ensures all Jacobian determinants are positive.
    
    Uses binary search to find the largest L_normal that doesn't produce negative Jacobians.
    
    Args:
        coast_x, coast_y: Coastline coordinates
        nx, ny: Mesh resolution
        clustering_factor: Clustering factor
        L_normal_initial: Initial guess for L_normal
        outer_offset: Vertical offset for outer boundary
        outer_angle: Angle for outer curve (degrees)
        spline_smoothing: Spline smoothing parameter
        spline_degree: B-spline degree
        min_jacobian_threshold: Minimum acceptable Jacobian value
        max_iterations: Maximum binary search iterations
        verbose: Print progress
    
    Returns:
        safe_L_normal: Maximum safe L_normal value
        min_jacobian: Minimum Jacobian at safe_L_normal
    """
    def check_jacobian(L_test):
        """Check if L_test produces valid mesh"""
        try:
            X, Y, XI, ETA, X_xi, X_eta, info = generate_coastline_mesh(
                coast_x=coast_x,
                coast_y=coast_y,
                nx=nx,
                ny=ny,
                clustering_factor=clustering_factor,
                L_normal=L_test,
                outer_offset=outer_offset,
                outer_angle=outer_angle,
                outer_curve_params=None,
                spline_smoothing=spline_smoothing,
                spline_degree=spline_degree
            )
            
            det_J = X_xi[...,0] * X_eta[...,1] - X_xi[...,1] * X_eta[...,0]
            min_det = det_J.min()
            n_negative = np.sum(det_J < 0)
            
            return min_det >= min_jacobian_threshold and n_negative == 0, min_det, n_negative
        except:
            return False, -1e10, 999999
    
    if verbose:
        print(f"\nFinding safe L_normal (binary search)...")
        print(f"  Initial L_normal: {L_normal_initial}")
        print(f"  Target: min Jacobian >= {min_jacobian_threshold}, no negative Jacobians")
    
    # Binary search bounds
    L_min = 0.0
    L_max = L_normal_initial * 2.0  # Start with 2x initial value
    
    # First, find an upper bound that's definitely too large
    is_valid, min_det, n_neg = check_jacobian(L_max)
    if is_valid:
        # L_max is valid, try larger values
        while is_valid and L_max < L_normal_initial * 10:
            L_max *= 2.0
            is_valid, min_det, n_neg = check_jacobian(L_max)
    
    # Now binary search between L_min and L_max
    for iteration in range(max_iterations):
        L_test = (L_min + L_max) / 2.0
        is_valid, min_det, n_neg = check_jacobian(L_test)
        
        if verbose and iteration < 5:
            print(f"  Iteration {iteration+1}: L_normal={L_test:.6f}, valid={is_valid}, min_det={min_det:.6f}, neg={n_neg}")
        
        if is_valid:
            L_min = L_test  # This value works, try larger
        else:
            L_max = L_test  # This value fails, try smaller
        
        if (L_max - L_min) / L_max < 1e-6:  # Convergence
            break
    
    safe_L_normal = L_min
    final_valid, final_min_det, final_n_neg = check_jacobian(safe_L_normal)
    
    if verbose:
        print(f"\nSafe L_normal found: {safe_L_normal:.6f}")
        print(f"  Minimum Jacobian: {final_min_det:.6f}")
        print(f"  Negative Jacobians: {final_n_neg}")
        if safe_L_normal < L_normal_initial:
            print(f"  WARNING: Safe L_normal ({safe_L_normal:.6f}) is smaller than initial ({L_normal_initial:.6f})")
    
    return safe_L_normal, final_min_det


# =========================================================
# Function to optimize outer angle (top line rotation)
# =========================================================
def optimize_outer_angle(coast_x, coast_y, nx=50, ny=30, clustering_factor=0.0,
                         L_normal=0.1, outer_offset=0.8,
                         spline_smoothing=0.0, spline_degree=3,
                         angle_range=(-45.0, 45.0), verbose=True):
    """
    Optimize the rotation angle of the outer curve (top line) to maximize mesh quality.
    
    This function finds the angle that maximizes the minimum Jacobian determinant,
    ensuring the best possible mesh quality.
    
    Args:
        coast_x, coast_y: Coastline coordinates
        nx, ny: Mesh resolution
        clustering_factor: Clustering factor
        L_normal: Normal extension depth
        outer_offset: Vertical offset for outer boundary
        spline_smoothing: Spline smoothing parameter
        spline_degree: B-spline degree
        angle_range: Tuple (min_angle, max_angle) in degrees for search range
        verbose: Print optimization progress
    
    Returns:
        best_angle: Optimal angle in degrees
        best_quality: Quality metric value (minimum Jacobian determinant)
        results: Dictionary with optimization results
    """
    def evaluate_angle(angle_deg):
        """Evaluate mesh quality for given angle"""
        try:
            X, Y, XI, ETA, X_xi, X_eta, info = generate_coastline_mesh(
                coast_x=coast_x,
                coast_y=coast_y,
                nx=nx,
                ny=ny,
                clustering_factor=clustering_factor,
                L_normal=L_normal,
                outer_offset=outer_offset,
                outer_angle=angle_deg,
                outer_curve_params=None,
                spline_smoothing=spline_smoothing,
                spline_degree=spline_degree
            )
            
            # Compute Jacobian determinant
            det_J = X_xi[...,0] * X_eta[...,1] - X_xi[...,1] * X_eta[...,0]
            
            # Quality metric: minimum Jacobian determinant (we want to maximize this)
            min_det = det_J.min()
            
            # Also check for negative Jacobians (penalize heavily)
            n_negative = np.sum(det_J < 0)
            if n_negative > 0:
                # Heavy penalty for negative Jacobians
                quality = min_det - 1000.0 * n_negative
            else:
                quality = min_det
            
            return quality, min_det, det_J.min(), det_J.max(), n_negative
        except Exception as e:
            if verbose:
                print(f"  Error with angle {angle_deg:.2f}°: {e}")
            return -1e10, -1e10, -1e10, -1e10, 999999
    
    if verbose:
        print(f"\nOptimizing outer curve angle (top line rotation)...")
        print(f"  Angle range: [{angle_range[0]}°, {angle_range[1]}°]")
        print(f"  Mesh resolution: {nx} x {ny}")
    
    # Objective function (negate because we want to maximize)
    def obj_func(angle_deg):
        quality, _, _, _, _ = evaluate_angle(angle_deg)
        return -quality  # Negate because minimize finds minimum
    
    # First, do a coarse grid search to find a good starting point
    if verbose:
        print(f"  Performing coarse grid search...")
    n_grid = 9  # Test 9 angles initially
    angles_grid = np.linspace(angle_range[0], angle_range[1], n_grid)
    qualities_grid = []
    best_grid_angle = angle_range[0]
    best_grid_quality = -1e10
    
    for angle in angles_grid:
        quality, min_det, _, _, n_neg = evaluate_angle(angle)
        qualities_grid.append(quality)
        if quality > best_grid_quality:
            best_grid_quality = quality
            best_grid_angle = angle
        if verbose:
            print(f"    Angle {angle:6.2f}°: quality={quality:8.6f}, min_det={min_det:8.6f}, neg={n_neg}")
    
    if verbose:
        print(f"  Best from grid search: {best_grid_angle:.2f}° (quality: {best_grid_quality:.6f})")
    
    # Now use minimize_scalar for fine optimization around the best grid point
    if verbose:
        print(f"  Refining with minimize_scalar...")
    
    # Use a smaller range around the best grid point
    grid_spacing = (angle_range[1] - angle_range[0]) / (n_grid - 1)
    refine_range = (max(angle_range[0], best_grid_angle - grid_spacing),
                    min(angle_range[1], best_grid_angle + grid_spacing))
    
    result = minimize_scalar(
        obj_func,
        bounds=refine_range,
        method='bounded',
        options={'xatol': 0.1, 'maxiter': 50}  # 0.1 degree tolerance
    )
    
    best_angle = result.x
    best_quality, best_min_det, best_min, best_max, best_n_neg = evaluate_angle(best_angle)
    
    if verbose:
        print(f"\nOptimization complete:")
        print(f"  Best angle: {best_angle:.2f}°")
        print(f"  Minimum Jacobian: {best_min_det:.6f}")
        print(f"  Jacobian range: [{best_min:.6f}, {best_max:.6f}]")
        print(f"  Negative Jacobians: {best_n_neg}")
        print(f"  Optimization success: {result.success}")
        print(f"  Number of iterations: {result.nfev}")
        
        # Compare with initial (0 degrees)
        initial_quality, initial_min_det, _, _, initial_n_neg = evaluate_angle(0.0)
        improvement = best_quality - initial_quality
        print(f"  Quality improvement: {improvement:.6f} (from {initial_quality:.6f} to {best_quality:.6f})")
        print(f"  Improvement vs 0°: {improvement:.6f}")
    
    results = {
        'best_angle': best_angle,
        'best_quality': best_quality,
        'min_jacobian': best_min_det,
        'jacobian_range': (best_min, best_max),
        'n_negative': best_n_neg,
        'optimization_result': result,
        'grid_search_results': list(zip(angles_grid, qualities_grid))
    }
    
    return best_angle, best_quality, results


# =========================================================
# Optimization function for flexible outer curve
# =========================================================
def optimize_outer_curve(coast_x, coast_y, nx=50, ny=30, clustering_factor=0.0,
                          L_normal=0.1, outer_offset=0.8, n_control=7,
                          offset_range=(-0.5, 0.5), spline_smoothing=0.0, 
                          spline_degree=3, verbose=True):
    """
    Optimize the shape of the outer curve (flexible B-spline) to improve mesh quality.
    
    This function optimizes the y-offset parameters of control points to maximize
    the minimum Jacobian determinant, ensuring better mesh quality.
    
    Args:
        coast_x, coast_y: Coastline coordinates
        nx, ny: Mesh resolution
        clustering_factor: Clustering factor
        L_normal: Normal extension depth
        outer_offset: Vertical offset for outer boundary baseline
        n_control: Number of control points for flexible curve (more = more flexible)
        offset_range: Tuple (min_offset, max_offset) for y-offset bounds
        spline_smoothing: Spline smoothing parameter
        spline_degree: B-spline degree
        verbose: Print optimization progress
    
    Returns:
        best_params: Optimal y-offset parameters (array of shape (n_control,))
        best_quality: Quality metric value (minimum Jacobian determinant)
        results: Dictionary with optimization results
    """
    def evaluate_params(params):
        """Evaluate mesh quality for given curve parameters"""
        try:
            X, Y, XI, ETA, X_xi, X_eta, info = generate_coastline_mesh(
                coast_x=coast_x,
                coast_y=coast_y,
                nx=nx,
                ny=ny,
                clustering_factor=clustering_factor,
                L_normal=L_normal,
                outer_offset=outer_offset,
                outer_curve_params=params,  # y-offsets
                n_outer_control=n_control,
                spline_smoothing=spline_smoothing,
                spline_degree=spline_degree
            )
            
            # Compute Jacobian determinant
            det_J = X_xi[...,0] * X_eta[...,1] - X_xi[...,1] * X_eta[...,0]
            
            # Quality metric: minimum Jacobian determinant (we want to maximize this)
            min_det = det_J.min()
            
            # Also check for negative Jacobians (penalize heavily)
            n_negative = np.sum(det_J < 0)
            if n_negative > 0:
                # Heavy penalty for negative Jacobians
                quality = min_det - 1000.0 * n_negative
            else:
                quality = min_det
            
            return quality, min_det, det_J.min(), det_J.max(), n_negative
        except Exception as e:
            if verbose:
                print(f"  Error with params {params}: {e}")
            return -1e10, -1e10, -1e10, -1e10, 999999
    
    if verbose:
        print(f"\nOptimizing flexible outer curve...")
        print(f"  Number of control points: {n_control}")
        print(f"  Offset range: [{offset_range[0]}, {offset_range[1]}]")
        print(f"  Mesh resolution: {nx} x {ny}")
    
    # Evaluate initial guess (all zeros - straight line)
    x0 = np.zeros(n_control)
    initial_quality, initial_min_det, _, _, initial_n_neg = evaluate_params(x0)
    if verbose:
        print(f"  Initial guess quality: {initial_quality:.6f} (min Jacobian: {initial_min_det:.6f})")
    
    # Try a few random initial guesses to break symmetry and find better starting point
    best_x0 = x0.copy()
    best_initial_quality = initial_quality
    
    if verbose:
        print(f"  Testing random initial guesses...")
    for trial in range(5):
        # Random initial guess within bounds
        trial_x0 = np.random.uniform(offset_range[0] * 0.3, offset_range[1] * 0.3, n_control)
        trial_quality, _, _, _, _ = evaluate_params(trial_x0)
        if trial_quality > best_initial_quality:
            best_x0 = trial_x0
            best_initial_quality = trial_quality
            if verbose:
                print(f"    Trial {trial+1}: Found better initial guess (quality: {trial_quality:.6f})")
    
    x0 = best_x0
    if verbose:
        print(f"  Using initial guess with quality: {best_initial_quality:.6f}")
    
    # Test if objective function is sensitive to parameter changes
    if verbose:
        print(f"  Testing objective function sensitivity...")
        test_params = x0.copy()
        if n_control > 0:
            test_params[0] += 0.1 * (offset_range[1] - offset_range[0])
            test_quality, _, _, _, _ = evaluate_params(test_params)
            if verbose:
                print(f"    Quality change with parameter perturbation: {test_quality - best_initial_quality:.6f}")
                if abs(test_quality - best_initial_quality) < 1e-8:
                    print(f"    WARNING: Objective function appears insensitive to parameter changes!")
    
    # Bounds for each control point offset
    bounds = [offset_range] * n_control
    
    # Objective function with callback for progress
    iteration_count = [0]
    def obj_func(params):
        iteration_count[0] += 1
        quality = evaluate_params(params)[0]
        obj_val = -quality  # Negate because we want to maximize
        if verbose and iteration_count[0] <= 5:
            print(f"    Iteration {iteration_count[0]}: quality = {quality:.6f}, obj = {obj_val:.6f}")
        return obj_val
    
    # Use minimize to find the parameters that maximize minimum Jacobian
    # Try SLSQP first as it's more robust, then fall back to L-BFGS-B
    if verbose:
        print(f"  Starting optimization with SLSQP...")
    result = minimize(
        obj_func,
        x0=x0,
        method='SLSQP',  # Sequential Least Squares - more robust for bounded problems
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-6, 'disp': False}
    )
    
    # If SLSQP didn't iterate, try L-BFGS-B with looser tolerances
    if result.nit == 0:
        if verbose:
            print(f"  SLSQP didn't iterate (nit=0), trying L-BFGS-B with looser tolerances...")
        iteration_count[0] = 0  # Reset counter
        result = minimize(
            obj_func,
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-4, 'gtol': 1e-3}  # Looser tolerances
        )
    
    # If still no iterations, try a global optimization approach or different starting point
    if result.nit == 0:
        if verbose:
            print(f"  Still no iterations. Trying with perturbed initial guess...")
        # Add small random perturbation to break any symmetry
        x0_perturbed = x0 + np.random.normal(0, 0.01 * (offset_range[1] - offset_range[0]), n_control)
        x0_perturbed = np.clip(x0_perturbed, offset_range[0], offset_range[1])
        iteration_count[0] = 0
        result = minimize(
            obj_func,
            x0=x0_perturbed,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-5}
        )
    
    best_params = result.x
    best_quality, best_min_det, best_min, best_max, best_n_neg = evaluate_params(best_params)
    
    if verbose:
        print(f"\nOptimization complete:")
        print(f"  Best parameters (y-offsets): {best_params}")
        print(f"  Minimum Jacobian: {best_min_det:.6f}")
        print(f"  Jacobian range: [{best_min:.6f}, {best_max:.6f}]")
        print(f"  Negative Jacobians: {best_n_neg}")
        print(f"  Optimization success: {result.success}")
        print(f"  Number of iterations: {result.nit}")
        print(f"  Total function evaluations: {iteration_count[0]}")
        if result.nit == 0:
            print(f"  WARNING: Optimizer did not iterate!")
            print(f"    This may indicate:")
            print(f"    - The initial guess is already optimal")
            print(f"    - The objective function is flat (insensitive to parameters)")
            print(f"    - Numerical issues with gradient computation")
        else:
            # Re-evaluate initial quality for comparison
            initial_quality_final, _, _, _, _ = evaluate_params(np.zeros(n_control))
            improvement = best_quality - initial_quality_final
            print(f"  Quality improvement: {improvement:.6f} (from {initial_quality_final:.6f} to {best_quality:.6f})")
    
    results = {
        'best_params': best_params,
        'best_quality': best_quality,
        'min_jacobian': best_min_det,
        'jacobian_range': (best_min, best_max),
        'n_negative': best_n_neg,
        'optimization_result': result
    }
    
    return best_params, best_quality, results


# =========================================================
# Helper function to set top boundary rotation
# =========================================================
def set_top_boundary_rotation(angle_degrees: float, horizontal_shift: float = None, vertical_shift: float = None):
    """
    Set the rotation angle and shifts of the top boundary (outer curve).
    
    This is a convenience function to update the OUTER_ANGLE, OUTER_HORIZONTAL_SHIFT, and OUTER_VERTICAL_SHIFT parameters.
    You can also directly modify these parameters at the top of the file.
    
    Args:
        angle_degrees: Rotation angle in degrees
            - 0.0   = horizontal (parallel to x-axis)
            - > 0   = counterclockwise rotation
            - < 0   = clockwise rotation
            - ±90   = vertical
        horizontal_shift: Optional horizontal shift (fraction if < 1.0, absolute if >= 1.0)
            - Positive = shift right
            - Negative = shift left
            - If None, keeps current value
        vertical_shift: Optional vertical shift (fraction if < 1.0, absolute if >= 1.0)
            - Positive = shift up
            - Negative = shift down
            - If None, keeps current value
    
    Returns:
        Tuple of (angle, horizontal_shift, vertical_shift) that were set
    
    Example:
        >>> set_top_boundary_rotation(15.0, 0.1, 0.05)  # Rotate 15°, shift 10% right, 5% up
        >>> set_top_boundary_rotation(15.0)  # Only rotate, keep current shifts
    """
    global OUTER_ANGLE, OUTER_HORIZONTAL_SHIFT, OUTER_VERTICAL_SHIFT
    OUTER_ANGLE = float(angle_degrees)
    if horizontal_shift is not None:
        OUTER_HORIZONTAL_SHIFT = float(horizontal_shift)
    if vertical_shift is not None:
        OUTER_VERTICAL_SHIFT = float(vertical_shift)
    print(f"Top boundary rotation set to: {OUTER_ANGLE:.2f}°")
    if horizontal_shift is not None:
        print(f"Top boundary horizontal shift set to: {OUTER_HORIZONTAL_SHIFT:.4f}")
    if vertical_shift is not None:
        print(f"Top boundary vertical shift set to: {OUTER_VERTICAL_SHIFT:.4f}")
    return OUTER_ANGLE, OUTER_HORIZONTAL_SHIFT, OUTER_VERTICAL_SHIFT


# =========================================================
# Example usage
# =========================================================
if __name__ == "__main__":
    # =========================================================
    # Optional: Override top boundary rotation and shifts here
    # =========================================================
    # Uncomment and modify the lines below to change the top boundary:
    # set_top_boundary_rotation(15.0, 0.1, 0.05)  # Example: 15° rotation, 10% right shift, 5% up shift
    # set_top_boundary_rotation(15.0, 0.1)  # Example: Rotate and shift horizontally, keep vertical shift
    # set_top_boundary_rotation(15.0)  # Example: Only rotate, keep current shifts
    # Or use optimize_outer_angle() to automatically find the best angle
    # =========================================================
    # 0. Verify clustering is working
    # =========================================================
    if CLUSTERING_FACTOR > 0:
        print("=" * 60)
        print("Verifying clustering transform")
        print("=" * 60)
        test_etas = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        eta_clustered, deta_deta = eta_clustering_transform(test_etas, CLUSTERING_FACTOR)
        print(f"  Clustering factor: {CLUSTERING_FACTOR}")
        print(f"  Uniform eta:     {test_etas}")
        print(f"  Clustered eta:   {eta_clustered}")
        print(f"  Derivative:      {deta_deta}")
        print(f"  Compression at η=0: {deta_deta[0]:.4f}x")
        print(f"  Expansion at η=1:   {deta_deta[-1]:.4f}x")
        print(f"  Ratio (η=1/η=0):    {deta_deta[-1]/deta_deta[0]:.2f}x")
        print()
    
    # =========================================================
    # 1. Load coastline data
    # =========================================================
    print("=" * 60)
    print("Loading coastline data")
    print("=" * 60)
    
    # Load coastline from HDF5 file (already in scaled coordinates 0-1)
    coastline_data = load_hdf5_data(HDF5_FILE, 'coastline/coastline_smooth_scaled')
    coast_x = coastline_data[2:-2, 0]
    coast_y = coastline_data[2:-2, 1]
    
    print(f"  Loaded coastline from: {HDF5_FILE}")
    print(f"  Number of points: {len(coast_x)}")
    print(f"  X range: [{coast_x.min():.6f}, {coast_x.max():.6f}]")
    print(f"  Y range: [{coast_y.min():.6f}, {coast_y.max():.6f}]")
    
    # Check coastline direction and reverse if needed for right-handed coordinate system
    # For right-handed system: ξ should increase left-to-right (x increasing)
    dx_start_to_end = coast_x[-1] - coast_x[0]
    if dx_start_to_end < 0:
        print(f"  Coastline goes right-to-left (dx = {dx_start_to_end:.6f})")
        print(f"  Reversing parameterization to ensure ξ increases left-to-right...")
        coast_x = coast_x[::-1]
        coast_y = coast_y[::-1]
        print(f"  After reversal: X range: [{coast_x.min():.6f}, {coast_x.max():.6f}]")
        print(f"  After reversal: dx from start to end = {coast_x[-1] - coast_x[0]:.6f}")
    
    # =========================================================
    # 2. Generate mesh with transfinite interpolation
    # =========================================================
    print("\n" + "=" * 60)
    print("Generating mesh with transfinite interpolation")
    print(f"  Strategy: P(ξ, η) = (1-η) * C(ξ) + η * O(ξ)")
    print(f"  Outer curve angle: {OUTER_ANGLE:.2f}°")
    print(f"  Outer curve horizontal shift: {OUTER_HORIZONTAL_SHIFT:.4f}")
    print(f"  Outer curve vertical shift: {OUTER_VERTICAL_SHIFT:.4f}")
    print("=" * 60)
    
    # Optional: Automatically optimize the top boundary rotation angle
    # Uncomment the following lines to find the optimal angle:
    # print("\n" + "=" * 60)
    # print("Optimizing top boundary rotation angle...")
    # print("=" * 60)
    # best_angle, best_quality, opt_results = optimize_outer_angle(
    #     coast_x=coast_x,
    #     coast_y=coast_y,
    #     nx=NX,
    #     ny=NY,
    #     clustering_factor=CLUSTERING_FACTOR,
    #     L_normal=L_NORMAL,
    #     outer_offset=OUTER_OFFSET,
    #     spline_smoothing=SPLINE_SMOOTHING,
    #     spline_degree=SPLINE_DEGREE,
    #     angle_range=(-30.0, 30.0),  # Search range in degrees
    #     verbose=True
    # )
    # OUTER_ANGLE = best_angle  # Use the optimized angle
    # print(f"\nUsing optimized angle: {OUTER_ANGLE:.2f}°")
    
    # Generate mesh directly on given coordinates (already scaled 0-1)
    X, Y, XI, ETA, X_xi, X_eta, info = generate_coastline_mesh(
        coast_x=coast_x,
        coast_y=coast_y,
        nx=NX,
        ny=NY,
        clustering_factor=CLUSTERING_FACTOR,
        xi_clustering_factor=XI_CLUSTERING_FACTOR,
        xi_clustering_target_x=XI_CLUSTERING_TARGET_X,
        L_normal=L_NORMAL,
        outer_offset=OUTER_OFFSET,
        outer_angle=OUTER_ANGLE,  # Use specified angle
        outer_horizontal_shift=OUTER_HORIZONTAL_SHIFT,  # Use specified horizontal shift
        outer_vertical_shift=OUTER_VERTICAL_SHIFT,  # Use specified vertical shift
        outer_curve_params=None,  # Use straight line (not flexible curve)
        spline_smoothing=SPLINE_SMOOTHING,
        spline_degree=SPLINE_DEGREE
    )
    
    # Update info dictionary
    info['coast_x'] = coast_x
    info['coast_y'] = coast_y
    
    print(f"  Generated mesh")
    print(f"  X range: [{X.min():.6f}, {X.max():.6f}]")
    print(f"  Y range: [{Y.min():.6f}, {Y.max():.6f}]")
    
    # Calculate Jacobian determinant (in scaled coordinates)
    det_J = X_xi[...,0] * X_eta[...,1] - X_xi[...,1] * X_eta[...,0]
    
    n_negative = np.sum(det_J < 0)
    
    # Print Jacobian statistics (no warning)
    print(f"\nJacobian determinant statistics:")
    print(f"  Minimum: {det_J.min():.6f}")
    print(f"  Maximum: {det_J.max():.6f}")
    print(f"  Mean: {det_J.mean():.6f}")
    print(f"  Negative values: {n_negative} ({100.0 * n_negative / det_J.size:.2f}%)")
    if n_negative > 0:
        print(f"  ⚠️  Negative Jacobians indicate invalid mesh regions (see plot panel 3)")
        print(f"  Consider adjusting parameters:")
        print(f"    - Reduce OUTER_OFFSET (currently {OUTER_OFFSET})")
        print(f"    - Adjust OUTER_ANGLE (currently {OUTER_ANGLE}°)")
        print(f"    - Use find_safe_L_normal() or optimize_outer_angle() functions")
    else:
        print(f"  ✓ All Jacobian determinants are positive (mesh is valid)")
    
    # =========================================================
    # 3. Load and resample bathymetry onto mesh points
    # =========================================================
    bathy_resampled = None
    if PLOT_BATHYMETRY:
        print("\n" + "=" * 60)
        print("Loading and resampling bathymetry")
        print("=" * 60)
        
        # Load bathymetry data from HDF5 (already in scaled coordinates 0-1)
        bathy = load_hdf5_data(HDF5_FILE, 'bathymetry/bathy_interpolated')
        x_coords = load_hdf5_data(HDF5_FILE, 'coordinates/x_coords_scaled')
        y_coords = load_hdf5_data(HDF5_FILE, 'coordinates/y_coords_scaled')
        
        print(f"  Loaded bathymetry: shape {bathy.shape}")
        print(f"  Loaded X coordinates: shape {x_coords.shape}, range [{x_coords.min():.6f}, {x_coords.max():.6f}]")
        print(f"  Loaded Y coordinates: shape {y_coords.shape}, range [{y_coords.min():.6f}, {y_coords.max():.6f}]")
        
        # Use RegularGridInterpolator for much faster interpolation on regular grids
        # bathy is (ny, nx) where y_coords corresponds to rows and x_coords to columns
        ny_bathy, nx_bathy = bathy.shape
        
        print(f"  Using RegularGridInterpolator for fast regular grid interpolation...")
        print(f"  Bathymetry grid: {ny_bathy} x {nx_bathy} points")
        
        # Create interpolator: RegularGridInterpolator expects (y, x) ordering
        # bathy[i, j] corresponds to point (x_coords[j], y_coords[i])
        interpolator = RegularGridInterpolator(
            (y_coords, x_coords),  # Grid coordinates (y first, then x)
            bathy,                  # Values on grid
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        
        # Resample bathymetry onto mesh points
        print(f"  Resampling onto mesh points ({X.size} points)...")
        # Points need to be (N, 2) array with (y, x) or (x, y) ordering
        # RegularGridInterpolator expects points as (N, ndim) where ndim matches grid
        # Since we passed (y_coords, x_coords), we need points as (y, x)
        points_mesh = np.column_stack([Y.ravel(), X.ravel()])  # (y, x) ordering
        bathy_resampled_flat = interpolator(points_mesh)
        bathy_resampled = bathy_resampled_flat.reshape(X.shape)
        
        n_valid = np.sum(~np.isnan(bathy_resampled))
        print(f"  Resampled bathymetry: {n_valid}/{X.size} valid points")
        if n_valid > 0:
            print(f"  Depth range: [{np.nanmin(bathy_resampled):.2f}, {np.nanmax(bathy_resampled):.2f}] m")
    
    # =========================================================
    # 4. Plot results
    # =========================================================
    plot_coastline_mesh(
        X, Y, XI, ETA, X_xi, X_eta, info,
        clustering_factor=CLUSTERING_FACTOR,
        show_clustering_details=SHOW_CLUSTERING_DETAILS,
        save_filename=SAVE_FILENAME,
        bathy_resampled=bathy_resampled,
        bathy_vmin=BATHY_VMIN,
        bathy_vmax=BATHY_VMAX,
        hdf5_file=HDF5_FILE
    )

    # =========================================================
    # 5. Save mesh data
    # =========================================================
    print("\n" + "=" * 60)
    print("Saving mesh data")
    print("=" * 60)
    
    # Construct full Jacobian matrix J: (ξ,η) -> (x,y)
    # J = [∂x/∂ξ  ∂x/∂η]
    #     [∂y/∂ξ  ∂y/∂η]
    # Transform to original coordinates if scaling factors are available
    scaling_factors = None
    try:
        scaling_factors = get_hdf5_attrs(HDF5_FILE, 'scaling')
    except:
        pass
    
    if scaling_factors is not None:
        # Transform Jacobian components to original coordinates
        X_xi_original = X_xi.copy()
        X_xi_original[..., 0] *= scaling_factors['x_range']  # ∂x_original/∂ξ
        X_xi_original[..., 1] *= scaling_factors['y_range']  # ∂y_original/∂ξ
        
        X_eta_original = X_eta.copy()
        X_eta_original[..., 0] *= scaling_factors['x_range']  # ∂x_original/∂η
        X_eta_original[..., 1] *= scaling_factors['y_range']  # ∂x_original/∂η
        
        # Construct full Jacobian matrix in original coordinates
        ny, nx = X.shape
        J = np.zeros((ny, nx, 2, 2))
        J[:, :, 0, 0] = X_xi_original[:, :, 0]  # ∂x/∂ξ
        J[:, :, 0, 1] = X_eta_original[:, :, 0]  # ∂x/∂η
        J[:, :, 1, 0] = X_xi_original[:, :, 1]  # ∂y/∂ξ
        J[:, :, 1, 1] = X_eta_original[:, :, 1]  # ∂y/∂η
        
        # Use original coordinates for saving
        X_xi_save = X_xi_original
        X_eta_save = X_eta_original
    else:
        # Use scaled coordinates
        ny, nx = X.shape
        J = np.zeros((ny, nx, 2, 2))
        J[:, :, 0, 0] = X_xi[:, :, 0]  # ∂x/∂ξ
        J[:, :, 0, 1] = X_eta[:, :, 0]  # ∂x/∂η
        J[:, :, 1, 0] = X_xi[:, :, 1]  # ∂y/∂ξ
        J[:, :, 1, 1] = X_eta[:, :, 1]  # ∂y/∂η
        
        X_xi_save = X_xi
        X_eta_save = X_eta
    
    # Save mesh data to HDF5 file
    with h5py.File(HDF5_FILE, 'a') as f:  # 'a' mode to append/update existing file
        # Create mesh group if it doesn't exist
        if 'mesh' not in f:
            mesh_group = f.create_group('mesh')
        else:
            mesh_group = f['mesh']
        
        # Save coordinate arrays
        if 'X' in mesh_group:
            del mesh_group['X']
        if 'Y' in mesh_group:
            del mesh_group['Y']
        if 'XI' in mesh_group:
            del mesh_group['XI']
        if 'ETA' in mesh_group:
            del mesh_group['ETA']
        
        mesh_group.create_dataset('X', data=X)
        mesh_group.create_dataset('Y', data=Y)
        mesh_group.create_dataset('XI', data=XI)
        mesh_group.create_dataset('ETA', data=ETA)
        
        # Save Jacobian components
        jacobian_group = mesh_group.create_group('jacobian')
        
        # Save combined arrays (for compatibility)
        jacobian_group.create_dataset('X_xi', data=X_xi_save)  # (ny, nx, 2): [∂x/∂ξ, ∂y/∂ξ]
        jacobian_group.create_dataset('X_eta', data=X_eta_save)  # (ny, nx, 2): [∂x/∂η, ∂y/∂η]
        
        # Save individual components for clarity
        jacobian_group.create_dataset('x_xi', data=X_xi_save[:, :, 0])  # ∂x/∂ξ (ny, nx)
        jacobian_group.create_dataset('y_xi', data=X_xi_save[:, :, 1])  # ∂y/∂ξ (ny, nx)
        jacobian_group.create_dataset('x_eta', data=X_eta_save[:, :, 0])  # ∂x/∂η (ny, nx)
        jacobian_group.create_dataset('y_eta', data=X_eta_save[:, :, 1])  # ∂y/∂η (ny, nx)
        
        jacobian_group.create_dataset('J', data=J)  # Full Jacobian matrix (ny, nx, 2, 2)
        
        # Save Jacobian determinant
        det_J = X_xi_save[...,0] * X_eta_save[...,1] - X_xi_save[...,1] * X_eta_save[...,0]
        jacobian_group.create_dataset('det_J', data=det_J)
        
        # Add metadata
        jacobian_group.attrs['description'] = 'Jacobian data for mesh transformation (ξ,η) -> (x,y)'
        jacobian_group.attrs['X_xi_shape'] = X_xi_save.shape
        jacobian_group.attrs['X_eta_shape'] = X_eta_save.shape
        jacobian_group.attrs['J_shape'] = J.shape
        jacobian_group.attrs['X_xi_description'] = '∂X/∂ξ: derivative of physical coordinates w.r.t. ξ (ny, nx, 2) = [∂x/∂ξ, ∂y/∂ξ]'
        jacobian_group.attrs['X_eta_description'] = '∂X/∂η: derivative of physical coordinates w.r.t. η (ny, nx, 2) = [∂x/∂η, ∂y/∂η]'
        jacobian_group.attrs['x_xi_description'] = '∂x/∂ξ: derivative of x-coordinate w.r.t. ξ (ny, nx)'
        jacobian_group.attrs['y_xi_description'] = '∂y/∂ξ: derivative of y-coordinate w.r.t. ξ (ny, nx)'
        jacobian_group.attrs['x_eta_description'] = '∂x/∂η: derivative of x-coordinate w.r.t. η (ny, nx)'
        jacobian_group.attrs['y_eta_description'] = '∂y/∂η: derivative of y-coordinate w.r.t. η (ny, nx)'
        jacobian_group.attrs['J_description'] = 'Full Jacobian matrix J[i,j] = [[∂x/∂ξ, ∂x/∂η], [∂y/∂ξ, ∂y/∂η]] at grid point (i,j)'
        jacobian_group.attrs['det_J_description'] = 'Jacobian determinant det(J) = (∂x/∂ξ)(∂y/∂η) - (∂x/∂η)(∂y/∂ξ)'
        jacobian_group.attrs['det_J_min'] = float(np.nanmin(det_J))
        jacobian_group.attrs['det_J_max'] = float(np.nanmax(det_J))
        jacobian_group.attrs['det_J_mean'] = float(np.nanmean(det_J))
        jacobian_group.attrs['n_negative'] = int(np.sum(det_J < 0))
        
        mesh_group.attrs['nx'] = nx
        mesh_group.attrs['ny'] = ny
        mesh_group.attrs['description'] = 'Mesh coordinates and Jacobian data'
    
    print(f"  Saved mesh data to: {HDF5_FILE}")
    print(f"    - X, Y: physical coordinates (shape {X.shape})")
    print(f"    - XI, ETA: parameter space coordinates (shape {XI.shape})")
    print(f"    - X_xi: ∂X/∂ξ (shape {X_xi_save.shape}) = [∂x/∂ξ, ∂y/∂ξ]")
    print(f"    - X_eta: ∂X/∂η (shape {X_eta_save.shape}) = [∂x/∂η, ∂y/∂η]")
    print(f"    - x_xi: ∂x/∂ξ (shape {X_xi_save[:, :, 0].shape})")
    print(f"    - y_xi: ∂y/∂ξ (shape {X_xi_save[:, :, 1].shape})")
    print(f"    - x_eta: ∂x/∂η (shape {X_eta_save[:, :, 0].shape})")
    print(f"    - y_eta: ∂y/∂η (shape {X_eta_save[:, :, 1].shape})")
    print(f"    - J: full Jacobian matrix (shape {J.shape})")
    print(f"    - det_J: Jacobian determinant (shape {det_J.shape})")
    print(f"    - det_J range: [{np.nanmin(det_J):.6f}, {np.nanmax(det_J):.6f}]")
    print(f"    - Negative Jacobians: {np.sum(det_J < 0)}")

    
