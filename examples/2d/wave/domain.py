import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import os
import h5py

# =========================================================
# CONTROL PARAMETERS
# =========================================================
data_dir = "data"
flip_y_axis = True  # Set to True to flip y-axis, False to keep original orientation
edge_tolerance_factor = 0.02  # Tolerance for edge detection (fraction of dimension)

# B-spline parameters
spline_smoothing = 5.0  # Spline smoothing parameter (0 = interpolation, >0 = smoothing)
spline_degree = 3       # B-spline degree (cubic)
n_spline_points = 50   # Number of points for evaluating the smoothed B-spline

# Manual bay removal regions (polygons defined by list of vertices [x, y])
bay_regions = [
    [[-10, 0], [-10, 1000], [1000, 1000], [1000, -10]]  # Small bay polygon to remove
]

print(f"Loading interpolated data from '{data_dir}' folder...")

# Load interpolated bathymetry field
bathy = np.load(os.path.join(data_dir, "bathy_interpolated.npy"))
print(f"  Loaded interpolated bathymetry: shape {bathy.shape}, range {np.nanmin(bathy):.2f} to {np.nanmax(bathy):.2f} m")

# Load X and Y coordinate arrays
x_coords = np.load(os.path.join(data_dir, "x_coords.npy"))
y_coords = np.load(os.path.join(data_dir, "y_coords.npy"))
print(f"  Loaded X coordinates: shape {x_coords.shape}, range {np.nanmin(x_coords):.2f} to {np.nanmax(x_coords):.2f} m")
print(f"  Loaded Y coordinates: shape {y_coords.shape}, range {np.nanmin(y_coords):.2f} to {np.nanmax(y_coords):.2f} m")

# Create coordinate grids for coastline extraction
X_coords, Y_coords = np.meshgrid(x_coords, y_coords)

# =========================================================
# Function to test if point is inside polygon (ray casting algorithm)
# =========================================================
def point_in_polygon(point, polygon):
    """
    Test if a point is inside a polygon using ray casting algorithm.
    
    Args:
        point: [x, y] coordinates of the point
        polygon: List of [x, y] vertices defining the polygon
    
    Returns:
        True if point is inside polygon, False otherwise
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

# =========================================================
# Extract coastline (depth = 0)
# =========================================================
print(f"\nExtracting coastline (depth = 0)...")

fig_temp, ax_temp = plt.subplots(figsize=(1, 1))
contour = ax_temp.contour(X_coords, Y_coords, bathy, levels=[0.0], colors='none')
plt.close(fig_temp)

# Calculate data extent for edge detection
left_edge = np.nanmin(x_coords)
right_edge = np.nanmax(x_coords)
bottom_edge = np.nanmin(y_coords)
top_edge = np.nanmax(y_coords)
edge_tolerance = max((right_edge - left_edge) * edge_tolerance_factor, 
                    (top_edge - bottom_edge) * edge_tolerance_factor)

# Filter out inland water bodies and islands
# Only keep contours that touch the edges (coastline) and exclude closed contours (inland water/islands)
coastline_points = []
inland_water_count = 0
island_count = 0

if len(contour.allsegs) > 0 and len(contour.allsegs[0]) > 0:
    for segment in contour.allsegs[0]:
        if len(segment) > 0:
            segment = np.array(segment)
            
            # Check if segment touches any edge
            touches_left = np.any(np.abs(segment[:, 0] - left_edge) < edge_tolerance)
            touches_right = np.any(np.abs(segment[:, 0] - right_edge) < edge_tolerance)
            touches_bottom = np.any(np.abs(segment[:, 1] - bottom_edge) < edge_tolerance)
            touches_top = np.any(np.abs(segment[:, 1] - top_edge) < edge_tolerance)
            
            touches_edge = touches_left or touches_right or touches_bottom or touches_top
            
            # Check if segment is closed (start and end points are close) - indicates island or inland water
            is_closed = len(segment) > 2 and np.linalg.norm(segment[0] - segment[-1]) < edge_tolerance
            
            # Keep only if touches edge AND is not a closed loop (inland water/island)
            if touches_edge and not is_closed:
                coastline_points.extend(segment)
            else:
                if is_closed:
                    island_count += 1
                else:
                    inland_water_count += 1
    
    if len(coastline_points) > 0:
        coastline_points = np.array(coastline_points)
        print(f"  Coastline extracted: {len(coastline_points)} points found")
        print(f"  Filtered out {island_count} island(s) and {inland_water_count} inland water body contour(s)")
        
        # =========================================================
        # Manually remove small bays
        # =========================================================
        if len(bay_regions) > 0:
            print(f"\nManually removing small bays from specified regions...")
            coastline_before = len(coastline_points)
            keep_mask = np.ones(len(coastline_points), dtype=bool)
            
            for i, polygon in enumerate(bay_regions):
                # Test each point to see if it's inside the polygon
                in_region = np.array([point_in_polygon(point, polygon) 
                                     for point in coastline_points])
                
                n_removed = np.sum(in_region)
                keep_mask = keep_mask & ~in_region
                
                # Print polygon vertices for reference
                vertices_str = ", ".join([f"[{v[0]:.0f}, {v[1]:.0f}]" for v in polygon])
                print(f"  Region {i+1}: polygon with vertices {vertices_str} - removed {n_removed} points")
            
            coastline_points = coastline_points[keep_mask]
            coastline_after = len(coastline_points)
            total_removed = coastline_before - coastline_after
            print(f"  Total points removed: {total_removed} (from {coastline_before} to {coastline_after})")
        
        # Save raw coastline to file
        coastline_file = os.path.join(data_dir, "coastline_downsampled.npy")
        np.save(coastline_file, coastline_points)
        print(f"  Saved raw coastline to: {coastline_file}")
        
        # =========================================================
        # Create B-spline smoothed representation using BSpline class directly
        # =========================================================
        print(f"\nCreating B-spline smoothed coastline...")
        
        # Extract x and y coordinates
        coast_x = coastline_points[:, 0]
        coast_y = coastline_points[:, 1]
        n_points = len(coast_x)
        
        # Use fewer control points for smooth approximation (not interpolation)
        # This creates a smooth curve that approximates the coastline without following every zig-zag
        n_control = max(n_spline_points // 2, spline_degree + 1)  # Use fewer control points for smoothing
        n_control = min(n_control, n_points // 2)  # But don't use more than half the data points
        
        # Create parameterization: cumulative arc length for data points
        dx = np.diff(coast_x)
        dy = np.diff(coast_y)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0], np.cumsum(ds)])  # Cumulative arc length
        s_normalized = s / s[-1] if s[-1] > 0 else np.linspace(0, 1, n_points)  # Normalize to [0, 1]
        
        # Construct uniform knot vector for smooth approximation
        # For n_control control points and degree k, need n_control + k + 1 knots
        n_knots = n_control + spline_degree + 1
        n_interior = n_knots - 2 * (spline_degree + 1)
        
        if n_interior > 0:
            # Uniform interior knots
            interior_knots = np.linspace(0, 1, n_interior + 2)[1:-1]
        else:
            interior_knots = np.array([])
        
        # Construct clamped knot vector: [0, ..., 0, interior, 1, ..., 1]
        knots = np.concatenate([
            np.zeros(spline_degree + 1),  # Clamped at start
            interior_knots,  # Interior knots (uniform)
            np.ones(spline_degree + 1)   # Clamped at end
        ]).astype(np.float64)
        
        # Fit control points using least squares to approximate the coastline
        # Build basis matrix: B[i, j] = B-spline basis function j evaluated at parameter s_normalized[i]
        from scipy.linalg import lstsq
        
        # Create basis splines for each control point
        n_basis = n_control
        basis_matrix = np.zeros((n_points, n_basis))
        
        # Create temporary BSpline objects for each basis function
        # Each basis function has coefficient 1.0 at its control point, 0 elsewhere
        for j in range(n_basis):
            coeffs = np.zeros(n_control)
            coeffs[j] = 1.0
            basis_spline = BSpline(knots, coeffs, spline_degree)
            basis_matrix[:, j] = basis_spline(s_normalized)
        
        # Solve least squares: basis_matrix * control_points ≈ data_points
        # This finds control points that best approximate the coastline
        control_points_x, _, _, _ = lstsq(basis_matrix, coast_x)
        control_points_y, _, _, _ = lstsq(basis_matrix, coast_y)
        
        # Construct BSpline objects directly with fitted control points
        try:
            # Create BSpline objects with knots and fitted control points
            spline_x = BSpline(knots, control_points_x, spline_degree)
            spline_y = BSpline(knots, control_points_y, spline_degree)
            
            # Evaluate B-spline at n_spline_points points
            u_smooth = np.linspace(0, 1, n_spline_points)
            x_smooth = spline_x(u_smooth)
            y_smooth = spline_y(u_smooth)
            coastline_smooth = np.column_stack([x_smooth, y_smooth])
            
            print(f"  B-spline created: {len(coastline_smooth)} evaluation points")
            print(f"  Data points: {n_points}")
            print(f"  Control points: {n_control} (approximation, not interpolation)")
            print(f"  Spline degree: {spline_degree}")
            print(f"  Knot vector length: {len(knots)}")
            print(f"  Using BSpline class directly with least-squares fitting for smooth approximation")
            
            # Save smoothed coastline to file
            coastline_smooth_file = os.path.join(data_dir, "coastline_smooth.npy")
            np.save(coastline_smooth_file, coastline_smooth)
            print(f"  Saved smoothed coastline to: {coastline_smooth_file}")
            
        except Exception as e:
            print(f"  Error creating B-spline: {e}")
            import traceback
            traceback.print_exc()
            coastline_smooth = None
    else:
        coastline_points = np.array([])
        coastline_smooth = None
        print(f"  No coastline found (all {island_count + inland_water_count} contours appear to be islands or inland water bodies)")
else:
    coastline_points = np.array([])
    coastline_smooth = None
    print("  No coastline contour found (depth = 0)")

# Plot the data
plt.figure(figsize=(10, 10))
plt.contourf(x_coords, y_coords, bathy, cmap="terrain",levels=np.linspace(np.nanmin(bathy), 0, 100),vmax=0,vmin=np.nanmin(bathy))
plt.colorbar(label="Depth (m)")
# plt.xlim(10000-2000, 10000+2000)
# plt.ylim(10000-2000, 10000+2000)

# Plot wipe boxes (bay removal polygons)
if len(bay_regions) > 0:
    for i, polygon in enumerate(bay_regions):
        # Extract x and y coordinates of polygon vertices
        poly_x = [p[0] for p in polygon] + [polygon[0][0]]  # Close the polygon
        poly_y = [p[1] for p in polygon] + [polygon[0][1]]
        
        # Plot filled polygon (white out box)
        # plt.fill(poly_x, poly_y, color='white', alpha=0.7, 
        #         edgecolor='red', linewidth=2, linestyle='--',
        #         label='Wipe box' if i == 0 else '')
        
        # Also plot the outline
        # plt.plot(poly_x, poly_y, 'r--', linewidth=2, alpha=0.9)

# Overlay coastline if extracted
if len(coastline_points) > 0:
    # Plot raw coastline points
    plt.plot(coastline_points[:, 0], coastline_points[:, 1], 'r.-', 
            markersize=2, label='Raw coastline', alpha=0.5)
    
    # Plot smoothed B-spline if available
    if coastline_smooth is not None:
        plt.plot(coastline_smooth[:, 0], coastline_smooth[:, 1], 'b-', 
                linewidth=2, label='B-spline smoothed', alpha=0.9)
    
    plt.legend()

plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.title("Bathymetry - Nazaré")
plt.show()


# Scale and save the coastline points to HDF5 file
print(f"\nScaling coordinates and saving to HDF5...")

# Scale coordinates
coastline_smooth[:, 0] = (coastline_smooth[:, 0] - np.min(x_coords)) / (np.max(x_coords) - np.min(x_coords))
coastline_smooth[:, 1] = (coastline_smooth[:, 1] - np.min(y_coords)) / (np.max(y_coords) - np.min(y_coords))

x_coords_scaled = (x_coords - np.min(x_coords)) / (np.max(x_coords) - np.min(x_coords))
y_coords_scaled = (y_coords - np.min(y_coords)) / (np.max(y_coords) - np.min(y_coords))

# Calculate scaling factors
x_min = np.min(x_coords)
x_max = np.max(x_coords)
y_min = np.min(y_coords)
y_max = np.max(y_coords)
x_range = x_max - x_min
y_range = y_max - y_min

# Store all data in HDF5 format
hdf5_file = os.path.join(data_dir, "nazare_data.h5")
with h5py.File(hdf5_file, 'w') as f:
    # Coastline data
    coastline_group = f.create_group('coastline')
    coastline_group.create_dataset('coastline_smooth_scaled', data=coastline_smooth)
    coastline_group.attrs['description'] = 'Smoothed coastline in scaled coordinates (0-1)'
    coastline_group.attrs['n_points'] = len(coastline_smooth)
    
    # Coordinate grids
    coords_group = f.create_group('coordinates')
    coords_group.create_dataset('x_coords_scaled', data=x_coords_scaled)
    coords_group.create_dataset('y_coords_scaled', data=y_coords_scaled)
    coords_group.attrs['description'] = 'Scaled coordinate arrays (0-1 range)'
    coords_group.attrs['x_shape'] = x_coords_scaled.shape
    coords_group.attrs['y_shape'] = y_coords_scaled.shape
    
    # Bathymetry data
    bathy_group = f.create_group('bathymetry')
    bathy_group.create_dataset('bathy_interpolated', data=bathy)
    bathy_group.attrs['description'] = 'Interpolated bathymetry data'
    bathy_group.attrs['shape'] = bathy.shape
    bathy_group.attrs['depth_min'] = float(np.nanmin(bathy))
    bathy_group.attrs['depth_max'] = float(np.nanmax(bathy))
    bathy_group.attrs['units'] = 'meters'
    
    # Scaling metadata
    scaling_group = f.create_group('scaling')
    scaling_group.attrs['x_min'] = float(x_min)
    scaling_group.attrs['x_max'] = float(x_max)
    scaling_group.attrs['x_range'] = float(x_range)
    scaling_group.attrs['y_min'] = float(y_min)
    scaling_group.attrs['y_max'] = float(y_max)
    scaling_group.attrs['y_range'] = float(y_range)
    scaling_group.attrs['description'] = 'Original coordinate ranges for inverse transformation'
    scaling_group.attrs['units'] = 'meters'
    
    # File metadata
    f.attrs['created_by'] = 'plot_nazare.py'
    f.attrs['coordinate_system'] = 'scaled (0-1)'
    f.attrs['coastline_processing'] = 'B-spline smoothed with bay removal'

print(f"  Saved all data to: {hdf5_file}")
print(f"  - Coastline: {len(coastline_smooth)} points")
print(f"  - Coordinates: X shape {x_coords_scaled.shape}, Y shape {y_coords_scaled.shape}")
print(f"  - Bathymetry: shape {bathy.shape}, depth range [{np.nanmin(bathy):.2f}, {np.nanmax(bathy):.2f}] m")
print(f"  - Scaling: X range [{x_min:.2f}, {x_max:.2f}] m, Y range [{y_min:.2f}, {y_max:.2f}] m")