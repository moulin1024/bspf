import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from typing import Tuple, Optional


# =========================================================
# Core modular function for coastline mesh generation
# =========================================================
def generate_coastline_mesh(
    coast_x: np.ndarray,
    coast_y: np.ndarray,
    nx: int = 50,
    ny: int = 30,
    clustering_factor: float = 0.0,
    L_normal: float = 0.1,
    outer_offset: float = 0.8,
    spline_smoothing: float = 0.0,
    spline_degree: int = 3
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
        spline_smoothing: Spline smoothing parameter (0 = interpolation, >0 = smoothing)
        spline_degree: B-spline degree (default 3 for cubic)
    
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
    y_max = np.max(coast_y)
    
    def outer_curve(xi):
        """
        外层引导曲线 O(ξ)：
        x 与岸线相同，y 固定为 y_max + outer_offset
        """
        Cpos = C(xi)
        x = Cpos[...,0]
        y = np.full_like(x, y_max + outer_offset)
        return np.stack([x, y], axis=-1)
    
    # =========================================================
    # 3. Generate parameter space grid (uniform)
    # =========================================================
    xis = np.linspace(0.0, 1.0, nx)
    etas = np.linspace(0.0, 1.0, ny)
    XI, ETA = np.meshgrid(xis, etas)
    
    # =========================================================
    # 4. Generate physical mesh and Jacobian
    # =========================================================
    X, Y = _map_grid_internal(xis, etas, C, tangent_normal, outer_curve, 
                              L_normal, clustering_factor)
    X_xi, X_eta = _jacobian_internal(xis, etas, C, C1, tangent_normal, normal_derivative,
                                     outer_curve, L_normal, clustering_factor)
    
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
        'coast_x': coast_x,
        'coast_y': coast_y,
        'tck': tck,
        'y_max': y_max
    }
    
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


def _map_grid_internal(xis, etas, C, tangent_normal, outer_curve, L_normal, clustering_factor):
    """Internal mapping function"""
    XI, ETA = np.meshgrid(xis, etas)
    xi_flat = XI.ravel()
    eta_flat = ETA.ravel()
    
    eta_clustered, _ = eta_clustering_transform(eta_flat, clustering_factor)
    
    Cpos = C(xi_flat)
    t_hat, n_hat = tangent_normal(xi_flat)
    E = Cpos + L_normal * n_hat
    O = outer_curve(xi_flat)
    D = O - E
    
    X_flat = Cpos + (eta_clustered * L_normal)[:,None] * n_hat \
                    + (eta_clustered**2)[:,None] * D
    
    X = X_flat[:,0].reshape(XI.shape)
    Y = X_flat[:,1].reshape(XI.shape)
    return X, Y


def _jacobian_internal(xis, etas, C, C1, tangent_normal, normal_derivative, 
                       outer_curve, L_normal, clustering_factor):
    """Internal Jacobian computation"""
    XI, ETA = np.meshgrid(xis, etas)
    xi_flat = XI.ravel()
    eta_flat = ETA.ravel()
    
    eta_clustered, deta_clustered_deta = eta_clustering_transform(eta_flat, clustering_factor)
    
    Cpos = C(xi_flat)
    Cp = C1(xi_flat)
    n_hat = tangent_normal(xi_flat)[1]
    n_prime = normal_derivative(xi_flat)
    
    E = Cpos + L_normal * n_hat
    O = outer_curve(xi_flat)
    
    # O'(ξ) = (C'_x(ξ), 0)
    O_prime = np.stack([Cp[:,0], np.zeros_like(Cp[:,0])], axis=-1)
    
    D = O - E
    D_prime = O_prime - (Cp + L_normal * n_prime)
    
    dX_deta_clustered = L_normal * n_hat + 2 * eta_clustered[:,None] * D
    X_eta_flat = dX_deta_clustered * deta_clustered_deta[:,None]
    
    X_xi_flat = Cp + (eta_clustered * L_normal)[:,None] * n_prime \
                    + (eta_clustered**2)[:,None] * D_prime
    
    shape = XI.shape + (2,)
    return X_xi_flat.reshape(shape), X_eta_flat.reshape(shape)

# =========================================================
# Plotting function
# =========================================================
def plot_coastline_mesh(X, Y, XI, ETA, X_xi, X_eta, info, clustering_factor=0.0, 
                        show_clustering_details=True, save_filename=None):
    """
    Plot the coastline mesh visualization (same as before).
    
    Args:
        X, Y: Physical coordinates (ny, nx)
        XI, ETA: Parameter space coordinates (ny, nx)
        X_xi, X_eta: Jacobian components (ny, nx, 2)
        info: Dictionary with additional information
        clustering_factor: Clustering factor used
        show_clustering_details: Whether to show clustering visualization
        save_filename: Optional filename to save the plot
    """
    ny, nx = X.shape
    det_J = X_xi[...,0] * X_eta[...,1] - X_xi[...,1] * X_eta[...,0]
    
    # Create three-panel figure
    fig = plt.figure(figsize=(18, 6))
    
    # Left panel: Physical mesh
    ax1 = plt.subplot(1, 3, 1)
    for j in range(ny):
        ax1.plot(X[j,:], Y[j,:], "b-", lw=0.6, alpha=0.7)
    for i in range(nx):
        ax1.plot(X[:,i], Y[:,i], "b-", lw=0.6, alpha=0.7)
    
    # Original coastline points
    ax1.plot(info['coast_x'], info['coast_y'], "ko", ms=4, label="input points")
    # Spline coastline
    C_plot = info['C'](np.linspace(0,1,200))
    ax1.plot(C_plot[:,0], C_plot[:,1], "r-", lw=2, label="spline coast")
    ax1.axis("equal")
    ax1.legend()
    ax1.set_title("Physical Mesh", fontsize=14, fontweight='bold')
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.grid(True, alpha=0.3)
    
    # Middle panel: Parameter space mesh
    ax2 = plt.subplot(1, 3, 2)
    for j in range(ny):
        ax2.plot(XI[j,:], ETA[j,:], "g-", lw=0.6, alpha=0.7)
    for i in range(nx):
        ax2.plot(XI[:,i], ETA[:,i], "g-", lw=0.6, alpha=0.7)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.set_title("Parameter Space (ξ, η)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("ξ")
    ax2.set_ylabel("η")
    ax2.grid(True, alpha=0.3)
    
    # Right panel: Jacobian determinant
    ax3 = plt.subplot(1, 3, 3)
    im = ax3.pcolormesh(XI, ETA, det_J, shading='gouraud', cmap='viridis')
    plt.colorbar(im, ax=ax3, label='det(J) = Area Scaling Factor')
    for j in range(0, ny, 5):
        ax3.plot(XI[j,:], ETA[j,:], "w-", lw=0.3, alpha=0.5)
    for i in range(0, nx, 5):
        ax3.plot(XI[:,i], ETA[:,i], "w-", lw=0.3, alpha=0.5)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_aspect('equal')
    ax3.set_title("Jacobian Determinant\n(Deformation Measure)", fontsize=14, fontweight='bold')
    ax3.set_xlabel("ξ")
    ax3.set_ylabel("η")
    
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
    
    # Clustering information
    if clustering_factor > 0 and show_clustering_details:
        etas = ETA[:, 0]  # Get eta values from parameter space
        print(f"\nClustering information:")
        print(f"  Factor: {clustering_factor}")
        print(f"  Parameter space: UNIFORM (eta = {etas[:3]} ... {etas[-3:]})")
        physical_spacing_coast = np.mean([np.linalg.norm([X[1,i]-X[0,i], Y[1,i]-Y[0,i]]) for i in range(nx)])
        physical_spacing_outer = np.mean([np.linalg.norm([X[-1,i]-X[-2,i], Y[-1,i]-Y[-2,i]]) for i in range(nx)])
        print(f"  Physical spacing near coast: {physical_spacing_coast:.6f}")
        print(f"  Physical spacing near outer: {physical_spacing_outer:.6f}")
        print(f"  Ratio (outer/coast physical spacing): {physical_spacing_outer / physical_spacing_coast:.2f}x")
        
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
# Example usage
# =========================================================
if __name__ == "__main__":
    # =========================================================
    # 1. Define coastline points
    # =========================================================
    # Example: create a coastline with some variation
    s_samples = np.linspace(0, 1, 15)
    coast_x = s_samples
    coast_y = 0.2 * np.sin(2 * np.pi * s_samples) - 0.15 * np.exp(-((s_samples - 0.5)/0.18)**2)
    
    # You can also load your own coastline data here:
    # coast_x = np.array([...])  # Your coastline x coordinates
    # coast_y = np.array([...])  # Your coastline y coordinates
    
    # =========================================================
    # 2. Set mesh parameters
    # =========================================================
    nx = 50              # Grid points along coast (ξ direction)
    ny = 100             # Grid points normal to coast (η direction)
    clustering_factor = 1.0  # Clustering towards coast (0 = uniform, >0 = clustered)
    L_normal = 0.1       # Normal extension depth
    outer_offset = 0.8   # Vertical offset for outer boundary
    
    # =========================================================
    # 3. Generate mesh
    # =========================================================
    X, Y, XI, ETA, X_xi, X_eta, info = generate_coastline_mesh(
        coast_x=coast_x,
        coast_y=coast_y,
        nx=nx,
        ny=ny,
        clustering_factor=clustering_factor,
        L_normal=L_normal,
        outer_offset=outer_offset,
        spline_smoothing=0.0,  # 0 = interpolation, >0 = smoothing
        spline_degree=3        # Cubic B-spline
    )
    
    # =========================================================
    # 4. Plot results
    # =========================================================
    plot_coastline_mesh(
        X, Y, XI, ETA, X_xi, X_eta, info,
        clustering_factor=clustering_factor,
        show_clustering_details=True,
        save_filename='coastline_mesh.png'
    )
