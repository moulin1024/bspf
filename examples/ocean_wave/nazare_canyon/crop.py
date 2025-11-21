import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.path import Path
from matplotlib.widgets import RectangleSelector
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.crs import CRS
from scipy.ndimage import rotate
from scipy.interpolate import griddata
import os

# =========================================================
# CONTROL PARAMETERS
# =========================================================

# Input/Output
tif_path = "data/bathy_nazare.tif"
output_dir = "data"

# Coordinate system
target_crs = CRS.from_epsg(32630)  # UTM Zone 30N (for meters)

# Data processing
resampling_method = Resampling.bilinear  # Resampling method for reprojection
trim_percent = 0.03  # Percentage to trim from all sides after initial crop

# # Zoom range (in meters, after reprojection to UTM)
# zoom_xmin = -10
# zoom_xmax = -9
# zoom_ymin = 39
# zoom_ymax = 40

# Rotation
rotation_angle_deg = -70.0  # Rotation angle in degrees (positive = counterclockwise)

# Coastline extraction
edge_tolerance_factor = 0.02  # Tolerance for edge detection (fraction of dimension)

# Point selection (in meters, after reprojection)
# Define two points A and B
point_A_x = -21500  # Point A X coordinate
point_A_y = 4.397e6  # Point A Y coordinate
point_B_x = -21500  # Point B X coordinate
point_B_y = 4.407e6   # Point B Y coordinate

length_line = np.sqrt((point_B_x - point_A_x)**2 + (point_B_y - point_A_y)**2)


# Number of points to create between A and B
n_points = 800  # Number of points along the line from A to B
n_normal = 800  # Number of points in the normal direction
normal_offset = 5000  # Total offset distance in normal direction (meters, positive = to the right of line direction)

dx = length_line / n_points
dy = np.abs(normal_offset) / n_normal
print(f"dx: {dx:.2f} m")
print(f"dy: {dy:.2f} m")
# Create new x coordinates by n_points * dx and n_normal * dy
x_coords_interpolated = np.linspace(0, n_points * dx, n_points)
y_coords_interpolated = np.linspace(0, n_normal * dy, n_normal)


# Plotting
plot_vmin = -500  # Minimum depth for colormap (meters)
plot_vmax = 0     # Maximum depth for colormap (meters)
plot_figsize = (10, 8)  # Figure size for plotting

# =========================================================
# MAIN PROCESSING
# =========================================================

print(f"Loading {tif_path}...")
with rasterio.open(tif_path) as src:
    # Read original data
    data_orig = src.read(1)
    
    print(f"Original CRS: {src.crs}")
    print(f"Data shape: {data_orig.shape}")
    print(f"Data range: {np.nanmin(data_orig):.2f} to {np.nanmax(data_orig):.2f} m")
    
    # =========================================================
    # Reproject to UTM (meters)
    # =========================================================
    print(f"\nReprojecting to UTM Zone 30N (meters)...")
    
    # Calculate transform for reprojection
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src.crs, target_crs, src.width, src.height,
        left=src.bounds.left, bottom=src.bounds.bottom,
        right=src.bounds.right, top=src.bounds.top
    )
    
    # Create destination array
    data = np.zeros((dst_height, dst_width), dtype=data_orig.dtype)
    
    # Reproject
    reproject(
        source=data_orig,
        destination=data,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=target_crs,
        resampling=resampling_method,
        src_nodata=src.nodata,
        dst_nodata=src.nodata
    )
    
    # Get extent in meters from transform
    # Transform format: [pixel_width, rotation, x_origin, rotation, pixel_height, y_origin]
    x_min = dst_transform[2]
    x_max = dst_transform[2] + dst_width * dst_transform[0]
    y_max = dst_transform[5]
    y_min = dst_transform[5] + dst_height * dst_transform[4]
    extent_meters = [x_min, x_max, y_min, y_max]
    
    print(f"  Reprojected shape: {data.shape}")
    print(f"  Extent (meters): X=[{extent_meters[0]:.2f}, {extent_meters[1]:.2f}], Y=[{extent_meters[2]:.2f}, {extent_meters[3]:.2f}]")
    
    # Create coordinate grids in meters for coastline extraction
    ny, nx = data.shape
    x_coords = np.linspace(extent_meters[0], extent_meters[1], nx)
    y_coords = np.linspace(extent_meters[3], extent_meters[2], ny)  # Reversed for origin='upper'
    X_coords, Y_coords = np.meshgrid(x_coords, y_coords)
    
    # =========================================================
    # Extract coastline (depth = 0)
    # =========================================================
    print(f"\nExtracting coastline (depth = 0)...")
    
    fig_temp, ax_temp = plt.subplots(figsize=(1, 1))
    contour = ax_temp.contour(X_coords, Y_coords, data, levels=[0.0], colors='none')
    plt.close(fig_temp)
    
    # Filter out inland water bodies
    # Only keep contours that touch the edges (coastline) and exclude closed contours (inland water)
    left_edge = extent_meters[0]
    right_edge = extent_meters[1]
    bottom_edge = extent_meters[2]
    top_edge = extent_meters[3]
    edge_tolerance = max((right_edge - left_edge) * edge_tolerance_factor, 
                        (top_edge - bottom_edge) * edge_tolerance_factor)
    
    coastline_points = []
    inland_water_count = 0
    
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
                
                # Additional check: if segment is closed (start and end points are close), it's likely inland water
                is_closed = len(segment) > 2 and np.linalg.norm(segment[0] - segment[-1]) < edge_tolerance
                
                # Keep only if touches edge AND is not a closed loop (inland water)
                if touches_edge and not is_closed:
                    coastline_points.extend(segment)
                else:
                    inland_water_count += 1
        
        if len(coastline_points) > 0:
            coastline_points = np.array(coastline_points)
            print(f"  Coastline extracted: {len(coastline_points)} points found")
            print(f"  Filtered out {inland_water_count} inland water body contour(s)")
        else:
            coastline_points = np.array([])
            print(f"  No coastline found (all {inland_water_count} contours appear to be inland water bodies)")
    else:
        coastline_points = np.array([])
        print("  No coastline contour found (depth = 0)")
    
    # Plot reprojected data with coastline (in meters)
    plt.figure(figsize=plot_figsize)
    plt.imshow(data, cmap="terrain", extent=extent_meters, origin='upper', 
               vmax=plot_vmax, vmin=plot_vmin)
    plt.axis('equal')
    plt.colorbar(label="Depth (m)")
    plt.title("Bathymetry - Nazaré (UTM Zone 30N, meters)")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    
    # # Apply zoom limits if specified (in meters)
    # if zoom_xmin is not None and zoom_xmax is not None:
    #     plt.xlim(zoom_xmin, zoom_xmax)
    # if zoom_ymin is not None and zoom_ymax is not None:
    #     plt.ylim(zoom_ymin, zoom_ymax)
    
    # Draw coastline if found
    if len(coastline_points) > 0:
        plt.plot(coastline_points[:, 0], coastline_points[:, 1], 'r.', 
                markersize=1, label='Coastline (raw)', alpha=0.7)
        plt.legend(loc='upper left')
    
    # =========================================================
    # Create 2D grid of points: N points along line A-B, M points in normal direction
    # =========================================================
    print(f"\nCreating {n_points}x{n_normal} grid of points...")
    print(f"  Point A: ({point_A_x:.2f}, {point_A_y:.2f})")
    print(f"  Point B: ({point_B_x:.2f}, {point_B_y:.2f})")
    print(f"  Normal offset: {normal_offset} m")
    
    # Calculate line direction vector
    dx = point_B_x - point_A_x
    dy = point_B_y - point_A_y
    line_length = np.sqrt(dx**2 + dy**2)
    
    if line_length < 1e-10:
        raise ValueError("Points A and B are too close together")
    
    # Normalize line direction vector
    line_dir = np.array([dx / line_length, dy / line_length])
    
    # Calculate normal vector (rotate 90 degrees counterclockwise: (-dy, dx))
    normal_dir = np.array([-line_dir[1], line_dir[0]])
    
    print(f"  Line length: {line_length:.2f} m")
    print(f"  Line direction: ({line_dir[0]:.4f}, {line_dir[1]:.4f})")
    print(f"  Normal direction: ({normal_dir[0]:.4f}, {normal_dir[1]:.4f})")
    
    # Create points along the line (parameter t from 0 to 1)
    t_values = np.linspace(0, 1, n_points)
    
    # Create offsets in normal direction (from 0 to normal_offset)
    offset_values = np.linspace(0, normal_offset, n_normal)
    
    # Create 2D grid of points
    points_x_grid = np.zeros((n_normal, n_points))
    points_y_grid = np.zeros((n_normal, n_points))
    
    for i, t in enumerate(t_values):
        # Point along the line
        base_x = point_A_x + t * (point_B_x - point_A_x)
        base_y = point_A_y + t * (point_B_y - point_A_y)
        
        for j, offset in enumerate(offset_values):
            # Offset in normal direction
            points_x_grid[j, i] = base_x + offset * normal_dir[0]
            points_y_grid[j, i] = base_y + offset * normal_dir[1]
    
    print(f"  Created {n_points}x{n_normal} = {n_points*n_normal} points")
    
    # Interpolate bathymetry data at grid points
    print(f"\nInterpolating bathymetry data at grid points...")
    
    # Prepare source data (all valid bathymetry points)
    points_source = np.column_stack([X_coords.ravel(), Y_coords.ravel()])
    values_source = data.ravel()
    
    # Remove NaN values from source
    valid_mask = ~np.isnan(values_source)
    points_source = points_source[valid_mask]
    values_source = values_source[valid_mask]
    
    print(f"  Using {len(values_source)} valid source data points")
    
    # Prepare target points (grid points)
    points_target = np.column_stack([points_x_grid.ravel(), points_y_grid.ravel()])
    
    # Interpolate bathymetry at grid points
    data_interpolated_flat = griddata(
        points_source, values_source,  # Source: all valid bathymetry data
        points_target,                 # Target: grid points
        method='linear',
        fill_value=np.nan
    )
    data_interpolated = data_interpolated_flat.reshape(points_x_grid.shape)
    
    n_valid = np.sum(~np.isnan(data_interpolated))
    print(f"  Interpolated data: {n_valid}/{n_points*n_normal} valid points")
    if n_valid > 0:
        print(f"  Depth range: [{np.nanmin(data_interpolated):.2f}, {np.nanmax(data_interpolated):.2f}] m")
    
    # Calculate grid spacing (dx and dy)
    # dx: spacing along line direction (between columns)
    if n_points > 1:
        dx = line_length / (n_points - 1)
    else:
        dx = line_length if line_length > 0 else 0.0
    
    # dy: spacing in normal direction (between rows)
    if n_normal > 1:
        dy = abs(normal_offset) / (n_normal - 1)
    else:
        dy = abs(normal_offset) if normal_offset != 0 else 0.0
    
    print(f"\n  Grid spacing:")
    print(f"    dx (along line): {dx:.2f} m")
    print(f"    dy (normal): {dy:.2f} m")
    
    # Save interpolated data and grid information
    os.makedirs(output_dir, exist_ok=True)
    
    # Save interpolated data array
    output_file = os.path.join(output_dir, "bathy_interpolated.npy")
    np.save(output_file, data_interpolated)
    print(f"\n  Saved interpolated data to: {output_file}")
    print(f"    Array shape: {data_interpolated.shape}")
    
    # Save coordinate arrays
    x_coords_file = os.path.join(output_dir, "x_coords.npy")
    y_coords_file = os.path.join(output_dir, "y_coords.npy")
    np.save(x_coords_file, x_coords_interpolated)
    np.save(y_coords_file, y_coords_interpolated)
    print(f"  Saved coordinate arrays:")
    print(f"    X coordinates: {x_coords_file}")
    print(f"    Y coordinates: {y_coords_file}")
    
    # Save grid metadata (dx, dy, and other info) as .npz
    grid_info_file = os.path.join(output_dir, "grid_info.npz")
    np.savez(grid_info_file,
             dx=dx,
             dy=dy,
             n_points=n_points,
             n_normal=n_normal,
             line_length=line_length,
             normal_offset=normal_offset,
             point_A_x=point_A_x,
             point_A_y=point_A_y,
             point_B_x=point_B_x,
             point_B_y=point_B_y)
    print(f"  Saved grid metadata to: {grid_info_file}")
    print(f"    Contains: dx, dy, n_points, n_normal, line_length, normal_offset, point_A/B coordinates")
    
    # Create two-panel plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # =========================================================
    # Left panel: Grid points on bathymetry
    # =========================================================
    im1 = ax1.imshow(data, cmap="terrain", extent=extent_meters, origin='upper', 
                     vmax=0, vmin=-4000)
    # cbar1 = plt.colorbar(im1, ax=ax1)
    # cbar1.set_label("Depth (m)")
    
    # Draw coastline if found
    # if len(coastline_points) > 0:
    # ax1.plot(coastline_points[:, 0], coastline_points[:, 1], 'r.', 
    #         markersize=1, label='Coastline', alpha=0.7)
    
    # Calculate bounding box of grid points and draw rectangle
    grid_x_min = np.min(points_x_grid)
    grid_x_max = np.max(points_x_grid)
    grid_y_min = np.min(points_y_grid)
    grid_y_max = np.max(points_y_grid)
    rect_width = grid_x_max - grid_x_min
    rect_height = grid_y_max - grid_y_min
    
    rect = Rectangle((grid_x_min, grid_y_min), rect_width, rect_height,
                    linewidth=1, edgecolor='r', facecolor='none', 
                    linestyle='--', alpha=0.8, 
                    label=f'Grid region ({n_points}x{n_normal})')
    ax1.add_patch(rect)
    
    ax1.set_aspect('equal')
    ax1.set_title(f"Grid Points on Bathymetry ({n_points}x{n_normal})")
    ax1.set_xlabel("Easting (m)")
    ax1.set_ylabel("Northing (m)")
    ax1.legend(loc='upper left')
    
    # =========================================================
    # Right panel: Grid data in row/column coordinates
    # =========================================================
    # Plot grid data using row and column indices as coordinates
    # Column index (j): 0 to n_points-1 (along line direction)
    # Row index (i): 0 to n_normal-1 (normal direction)
    extent_grid = [0, n_points, 0, n_normal]  # [left, right, bottom, top]
    
    im2 = ax2.contourf(x_coords_interpolated, y_coords_interpolated, data_interpolated, levels=np.linspace(-4000, 0, 100), cmap="terrain",vmin=-4000,vmax=0)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("Depth (m)")
    ax2.set_title(f"Interpolated Bathymetry (Grid Coordinates)")
    ax2.set_xlabel("Easting (m)")
    ax2.set_ylabel("Northing (m)")
    plt.tight_layout()
    plt.show()
