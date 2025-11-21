"""
Test program to verify zero-flux boundary condition enforcement in physical space
by transforming to parameter space using the Jacobian.

The key transformation:
- Physical space: ∇u_phys · n_phys = 0
- Parameter space: We need to compute what flux(U_ξ, U_η) gives zero physical flux
- Using Jacobian: ∇u_phys = J^{-T} [U_ξ, U_η]^T
- Constraint: n_phys^T J^{-T} [U_ξ, U_η]^T = 0
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from coon_mesh import map_grid, jacobian

from bspf import bspf2d


def compute_physical_normal_on_boundary(xis, etas, X_xi, X_eta, boundary='bottom'):
    """
    Compute outward normal vector in physical space on a boundary.
    """
    if boundary == 'bottom':  # η=0
        tangent = X_xi[0, :, :]  # (nx, 2)
        n_phys = np.stack([tangent[:, 1], -tangent[:, 0]], axis=-1)  # (nx, 2)
        norm = np.linalg.norm(n_phys, axis=-1, keepdims=True)
        n_phys = n_phys / (norm + 1e-12)
    elif boundary == 'top':  # η=1
        tangent = X_xi[-1, :, :]
        n_phys = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
        norm = np.linalg.norm(n_phys, axis=-1, keepdims=True)
        n_phys = n_phys / (norm + 1e-12)
    elif boundary == 'left':  # ξ=0
        tangent = X_eta[:, 0, :]
        n_phys = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
        norm = np.linalg.norm(n_phys, axis=-1, keepdims=True)
        n_phys = n_phys / (norm + 1e-12)
    elif boundary == 'right':  # ξ=1
        tangent = X_eta[:, -1, :]
        n_phys = np.stack([tangent[:, 1], -tangent[:, 0]], axis=-1)
        norm = np.linalg.norm(n_phys, axis=-1, keepdims=True)
        n_phys = n_phys / (norm + 1e-12)
    else:
        raise ValueError(f"Unknown boundary: {boundary}")
    
    return n_phys


def compute_parameter_space_flux_for_zero_physical_flux(U, plan_xi, plan_eta, 
                                                         xis, etas, X_xi, X_eta, 
                                                         boundary='bottom'):
    """
    Compute the parameter space flux that enforces zero flux in physical space.
    
    The constraint: n_phys · ∇u_phys = 0
    
    Using the gradient transformation:
    ∇u_phys = (1/det(J)) * [X_η_y  -X_ξ_y] [U_ξ]
               [-X_η_x   X_ξ_x] [U_η]
    
    So: n_phys · ∇u_phys = (1/det(J)) * [n_x, n_y] · [X_η_y*U_ξ - X_ξ_y*U_η, 
                                                      -X_η_x*U_ξ + X_ξ_x*U_η]
        = (1/det(J)) * (n_x*(X_η_y*U_ξ - X_ξ_y*U_η) + n_y*(-X_η_x*U_ξ + X_ξ_x*U_η))
        = (1/det(J)) * (U_ξ*(n_x*X_η_y - n_y*X_η_x) + U_η*(-n_x*X_ξ_y + n_y*X_ξ_x))
    
    For zero flux: U_ξ*coeff_xi + U_η*coeff_eta = 0
    
    On boundaries:
    - Bottom/Top (η=const): We control U_η, so U_η = -U_ξ * coeff_xi / coeff_eta
    - Left/Right (ξ=const): We control U_ξ, so U_ξ = -U_η * coeff_eta / coeff_xi
    
    Returns:
        flux_xi, flux_eta: flux values in parameter space (nx,) or (ny,)
    """
    # Get physical normal
    n_phys = compute_physical_normal_on_boundary(xis, etas, X_xi, X_eta, boundary)
    
    # First, compute U_ξ and U_η with zero parameter space flux to get interior values
    U_xi_interior = plan_xi.apply(U, flux=(0.0, 0.0))
    U_eta_interior = plan_eta.apply(U, flux=(0.0, 0.0))
    
    if boundary == 'bottom':  # η=0, we control U_η
        X_xi_bdry = X_xi[0, :, :]  # (nx, 2)
        X_eta_bdry = X_eta[0, :, :]  # (nx, 2)
        det_J_bdry = X_xi_bdry[:, 0] * X_eta_bdry[:, 1] - X_xi_bdry[:, 1] * X_eta_bdry[:, 0]
        
        # Coefficients from the constraint equation
        coeff_xi = n_phys[:, 0] * X_eta_bdry[:, 1] - n_phys[:, 1] * X_eta_bdry[:, 0]
        coeff_eta = -n_phys[:, 0] * X_xi_bdry[:, 1] + n_phys[:, 1] * X_xi_bdry[:, 0]
        
        # Get U_ξ at the boundary (from interior computation)
        U_xi_bdry = U_xi_interior[0, :]  # (nx,)
        
        # Solve for U_η: U_η = -U_ξ * coeff_xi / coeff_eta
        # This is the flux we need to enforce
        mask = np.abs(coeff_eta) > 1e-12
        flux_eta = np.zeros(len(xis))
        flux_eta[mask] = -U_xi_bdry[mask] * coeff_xi[mask] / coeff_eta[mask]
        flux_eta[~mask] = 0.0  # If coeff_eta is zero, set flux to zero
        
        return np.zeros(len(xis)), flux_eta
        
    elif boundary == 'top':  # η=1
        X_xi_bdry = X_xi[-1, :, :]
        X_eta_bdry = X_eta[-1, :, :]
        coeff_xi = n_phys[:, 0] * X_eta_bdry[:, 1] - n_phys[:, 1] * X_eta_bdry[:, 0]
        coeff_eta = -n_phys[:, 0] * X_xi_bdry[:, 1] + n_phys[:, 1] * X_xi_bdry[:, 0]
        U_xi_bdry = U_xi_interior[-1, :]
        mask = np.abs(coeff_eta) > 1e-12
        flux_eta = np.zeros(len(xis))
        flux_eta[mask] = -U_xi_bdry[mask] * coeff_xi[mask] / coeff_eta[mask]
        flux_eta[~mask] = 0.0
        return np.zeros(len(xis)), flux_eta
        
    elif boundary == 'left':  # ξ=0, we control U_ξ
        X_xi_bdry = X_xi[:, 0, :]  # (ny, 2)
        X_eta_bdry = X_eta[:, 0, :]  # (ny, 2)
        coeff_xi = n_phys[:, 0] * X_eta_bdry[:, 1] - n_phys[:, 1] * X_eta_bdry[:, 0]
        coeff_eta = -n_phys[:, 0] * X_xi_bdry[:, 1] + n_phys[:, 1] * X_xi_bdry[:, 0]
        U_eta_bdry = U_eta_interior[:, 0]  # (ny,)
        mask = np.abs(coeff_xi) > 1e-12
        flux_xi = np.zeros(len(etas))
        flux_xi[mask] = -U_eta_bdry[mask] * coeff_eta[mask] / coeff_xi[mask]
        flux_xi[~mask] = 0.0
        return flux_xi, np.zeros(len(etas))
        
    elif boundary == 'right':  # ξ=1
        X_xi_bdry = X_xi[:, -1, :]
        X_eta_bdry = X_eta[:, -1, :]
        coeff_xi = n_phys[:, 0] * X_eta_bdry[:, 1] - n_phys[:, 1] * X_eta_bdry[:, 0]
        coeff_eta = -n_phys[:, 0] * X_xi_bdry[:, 1] + n_phys[:, 1] * X_xi_bdry[:, 0]
        U_eta_bdry = U_eta_interior[:, -1]
        mask = np.abs(coeff_xi) > 1e-12
        flux_xi = np.zeros(len(etas))
        flux_xi[mask] = -U_eta_bdry[mask] * coeff_eta[mask] / coeff_xi[mask]
        flux_xi[~mask] = 0.0
        return flux_xi, np.zeros(len(etas))
    
    else:
        raise ValueError(f"Unknown boundary: {boundary}")


def compute_physical_flux_from_parameter_space(U_xi, U_eta, X_xi, X_eta, n_phys):
    """
    Compute physical space flux from parameter space derivatives.
    
    Returns:
        flux: n_phys · ∇u_phys
    """
    det_J = X_xi[..., 0] * X_eta[..., 1] - X_xi[..., 1] * X_eta[..., 0]
    
    # Transform to physical gradient
    U_x = (X_eta[..., 1] * U_xi - X_xi[..., 1] * U_eta) / (det_J + 1e-12)
    U_y = (-X_eta[..., 0] * U_xi + X_xi[..., 0] * U_eta) / (det_J + 1e-12)
    
    # Compute flux: n · ∇u
    grad = np.stack([U_x, U_y], axis=-1)  # (..., 2)
    flux = np.sum(grad * n_phys, axis=-1)  # (...)
    
    return flux


def test_boundary_flux_enforcement():
    """
    Test that zero flux in physical space is correctly enforced.
    """
    print("=" * 60)
    print("Testing Zero-Flux Boundary Condition Enforcement")
    print("=" * 60)
    
    # Setup grid
    nx, ny = 32, 32
    xis = np.linspace(0.0, 1.0, nx)
    etas = np.linspace(0.0, 1.0, ny)
    
    # Compute mesh and Jacobian
    X, Y = map_grid(xis, etas)
    X_xi, X_eta = jacobian(xis, etas)
    
    # Create a test function: u(x,y) = x^2 + y^2 (or in parameter space)
    # We'll use a simple function that has non-zero gradients
    XI, ETA = np.meshgrid(xis, etas)
    U = 0.5 * (XI**2 + ETA**2) + 1.0
    
    # Setup bspf operators
    op = bspf2d.from_grids(x=xis, y=etas, degree_x=8, degree_y=8, correction="spectral")
    plan_xi = op.make_plan_dx(order=1, lam=0.0, neumann=True)
    plan_eta = op.make_plan_dy(order=1, lam=0.0, neumann=True)
    
    print("\n1. Testing with zero parameter space flux (should have non-zero physical flux)")
    print("-" * 60)
    
    # Compute derivatives with zero parameter space flux
    U_xi_zero = plan_xi.apply(U, flux=(0.0, 0.0))
    U_eta_zero = plan_eta.apply(U, flux=(0.0, 0.0))
    
    # Check physical flux on each boundary
    boundaries = ['bottom', 'top', 'left', 'right']
    physical_fluxes_zero = {}
    
    for boundary in boundaries:
        n_phys = compute_physical_normal_on_boundary(xis, etas, X_xi, X_eta, boundary)
        
        if boundary in ['bottom', 'top']:
            if boundary == 'bottom':
                idx = 0
                U_xi_bdry = U_xi_zero[0:1, :]
                U_eta_bdry = U_eta_zero[0:1, :]
                X_xi_bdry = X_xi[0:1, :, :]
                X_eta_bdry = X_eta[0:1, :, :]
                n_phys_bdry = n_phys[None, :, :]
            else:  # top
                idx = -1
                U_xi_bdry = U_xi_zero[-1:, :]
                U_eta_bdry = U_eta_zero[-1:, :]
                X_xi_bdry = X_xi[-1:, :, :]
                X_eta_bdry = X_eta[-1:, :, :]
                n_phys_bdry = n_phys[None, :, :]
            
            flux = compute_physical_flux_from_parameter_space(
                U_xi_bdry, U_eta_bdry, X_xi_bdry, X_eta_bdry, n_phys_bdry
            )
            physical_fluxes_zero[boundary] = flux[0, :]
        else:
            if boundary == 'left':
                idx = 0
                U_xi_bdry = U_xi_zero[:, 0:1]
                U_eta_bdry = U_eta_zero[:, 0:1]
                X_xi_bdry = X_xi[:, 0:1, :]
                X_eta_bdry = X_eta[:, 0:1, :]
                n_phys_bdry = n_phys[:, None, :]
            else:  # right
                idx = -1
                U_xi_bdry = U_xi_zero[:, -1:]
                U_eta_bdry = U_eta_zero[:, -1:]
                X_xi_bdry = X_xi[:, -1:, :]
                X_eta_bdry = X_eta[:, -1:, :]
                n_phys_bdry = n_phys[:, None, :]
            
            flux = compute_physical_flux_from_parameter_space(
                U_xi_bdry, U_eta_bdry, X_xi_bdry, X_eta_bdry, n_phys_bdry
            )
            physical_fluxes_zero[boundary] = flux[:, 0]
        
        max_flux = np.max(np.abs(physical_fluxes_zero[boundary]))
        print(f"  {boundary:8s}: max |∇u·n| = {max_flux:.6e}")
    
    print("\n2. Computing required parameter space flux for zero physical flux")
    print("-" * 60)
    
    # Compute required parameter space fluxes
    flux_xi_bottom, flux_eta_bottom = compute_parameter_space_flux_for_zero_physical_flux(
        U, plan_xi, plan_eta, xis, etas, X_xi, X_eta, 'bottom'
    )
    flux_xi_top, flux_eta_top = compute_parameter_space_flux_for_zero_physical_flux(
        U, plan_xi, plan_eta, xis, etas, X_xi, X_eta, 'top'
    )
    flux_xi_left, flux_eta_left = compute_parameter_space_flux_for_zero_physical_flux(
        U, plan_xi, plan_eta, xis, etas, X_xi, X_eta, 'left'
    )
    flux_xi_right, flux_eta_right = compute_parameter_space_flux_for_zero_physical_flux(
        U, plan_xi, plan_eta, xis, etas, X_xi, X_eta, 'right'
    )
    
    print(f"  Bottom flux_eta range: [{flux_eta_bottom.min():.6e}, {flux_eta_bottom.max():.6e}]")
    print(f"  Top flux_eta range:    [{flux_eta_top.min():.6e}, {flux_eta_top.max():.6e}]")
    print(f"  Left flux_xi range:    [{flux_xi_left.min():.6e}, {flux_xi_left.max():.6e}]")
    print(f"  Right flux_xi range:   [{flux_xi_right.min():.6e}, {flux_xi_right.max():.6e}]")
    
    print("\n3. Testing with computed parameter space flux (should have zero physical flux)")
    print("-" * 60)
    
    # Compute derivatives with the computed fluxes
    U_xi_corrected = plan_xi.apply(U, flux=(flux_xi_left, flux_xi_right))
    U_eta_corrected = plan_eta.apply(U, flux=(flux_eta_bottom, flux_eta_top))
    
    # Check physical flux on each boundary
    physical_fluxes_corrected = {}
    
    for boundary in boundaries:
        n_phys = compute_physical_normal_on_boundary(xis, etas, X_xi, X_eta, boundary)
        
        if boundary in ['bottom', 'top']:
            if boundary == 'bottom':
                U_xi_bdry = U_xi_corrected[0:1, :]
                U_eta_bdry = U_eta_corrected[0:1, :]
                X_xi_bdry = X_xi[0:1, :, :]
                X_eta_bdry = X_eta[0:1, :, :]
                n_phys_bdry = n_phys[None, :, :]
            else:  # top
                U_xi_bdry = U_xi_corrected[-1:, :]
                U_eta_bdry = U_eta_corrected[-1:, :]
                X_xi_bdry = X_xi[-1:, :, :]
                X_eta_bdry = X_eta[-1:, :, :]
                n_phys_bdry = n_phys[None, :, :]
            
            flux = compute_physical_flux_from_parameter_space(
                U_xi_bdry, U_eta_bdry, X_xi_bdry, X_eta_bdry, n_phys_bdry
            )
            physical_fluxes_corrected[boundary] = flux[0, :]
        else:
            if boundary == 'left':
                U_xi_bdry = U_xi_corrected[:, 0:1]
                U_eta_bdry = U_eta_corrected[:, 0:1]
                X_xi_bdry = X_xi[:, 0:1, :]
                X_eta_bdry = X_eta[:, 0:1, :]
                n_phys_bdry = n_phys[:, None, :]
            else:  # right
                U_xi_bdry = U_xi_corrected[:, -1:]
                U_eta_bdry = U_eta_corrected[:, -1:]
                X_xi_bdry = X_xi[:, -1:, :]
                X_eta_bdry = X_eta[:, -1:, :]
                n_phys_bdry = n_phys[:, None, :]
            
            flux = compute_physical_flux_from_parameter_space(
                U_xi_bdry, U_eta_bdry, X_xi_bdry, X_eta_bdry, n_phys_bdry
            )
            physical_fluxes_corrected[boundary] = flux[:, 0]
        
        max_flux = np.max(np.abs(physical_fluxes_corrected[boundary]))
        print(f"  {boundary:8s}: max |∇u·n| = {max_flux:.6e}")
    
    print("\n4. Summary")
    print("-" * 60)
    print("Improvement factors (reduction in physical flux):")
    for boundary in boundaries:
        max_before = np.max(np.abs(physical_fluxes_zero[boundary]))
        max_after = np.max(np.abs(physical_fluxes_corrected[boundary]))
        if max_before > 1e-12:
            improvement = max_before / max_after
            print(f"  {boundary:8s}: {improvement:.2e}x reduction")
        else:
            print(f"  {boundary:8s}: already zero")
    
    # Compute Jacobian determinant for visualization
    det_J = X_xi[..., 0] * X_eta[..., 1] - X_xi[..., 1] * X_eta[..., 0]
    
    # Visualization 1: Physical mesh, parameter space, and Jacobian
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Physical mesh
    ax = axes1[0]
    # Draw mesh lines
    for j in range(0, ny, 2):
        ax.plot(X[j, :], Y[j, :], 'b-', lw=0.6, alpha=0.7)
    for i in range(0, nx, 2):
        ax.plot(X[:, i], Y[:, i], 'b-', lw=0.6, alpha=0.7)
    # Mark boundaries
    ax.plot(X[0, :], Y[0, :], 'r-', lw=2, label='Bottom (η=0)')
    ax.plot(X[-1, :], Y[-1, :], 'g-', lw=2, label='Top (η=1)')
    ax.plot(X[:, 0], Y[:, 0], 'm-', lw=2, label='Left (ξ=0)')
    ax.plot(X[:, -1], Y[:, -1], 'c-', lw=2, label='Right (ξ=1)')
    ax.set_aspect('equal')
    ax.set_title('Physical Mesh (x, y)', fontsize=14, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Parameter space mesh
    ax = axes1[1]
    # Draw mesh lines
    for j in range(0, ny, 2):
        ax.plot(XI[j, :], ETA[j, :], 'b-', lw=0.6, alpha=0.7)
    for i in range(0, nx, 2):
        ax.plot(XI[:, i], ETA[:, i], 'b-', lw=0.6, alpha=0.7)
    # Mark boundaries
    ax.plot(XI[0, :], ETA[0, :], 'r-', lw=2, label='Bottom (η=0)')
    ax.plot(XI[-1, :], ETA[-1, :], 'g-', lw=2, label='Top (η=1)')
    ax.plot(XI[:, 0], ETA[:, 0], 'm-', lw=2, label='Left (ξ=0)')
    ax.plot(XI[:, -1], ETA[:, -1], 'c-', lw=2, label='Right (ξ=1)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title('Parameter Space (ξ, η)', fontsize=14, fontweight='bold')
    ax.set_xlabel('ξ')
    ax.set_ylabel('η')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Jacobian determinant
    ax = axes1[2]
    im = ax.pcolormesh(XI, ETA, det_J, shading='gouraud', cmap='viridis')
    cbar = plt.colorbar(im, ax=ax, label='det(J)')
    # Overlay some mesh lines
    for j in range(0, ny, 4):
        ax.plot(XI[j, :], ETA[j, :], 'w-', lw=0.3, alpha=0.5)
    for i in range(0, nx, 4):
        ax.plot(XI[:, i], ETA[:, i], 'w-', lw=0.3, alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title('Jacobian Determinant\n(Mesh Quality)', fontsize=14, fontweight='bold')
    ax.set_xlabel('ξ')
    ax.set_ylabel('η')
    
    plt.tight_layout()
    plt.savefig('test_physical_flux_mesh_visualization.png', dpi=150)
    print("\nMesh visualization saved to 'test_physical_flux_mesh_visualization.png'")
    
    # Visualization 2: Flux comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Physical flux before correction (bottom boundary)
    ax = axes[0, 0]
    ax.plot(xis, physical_fluxes_zero['bottom'], 'r-', label='Before correction', lw=2)
    ax.plot(xis, physical_fluxes_corrected['bottom'], 'b--', label='After correction', lw=2)
    ax.set_xlabel('ξ')
    ax.set_ylabel('Physical flux ∇u·n')
    ax.set_title('Bottom Boundary (η=0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    
    # Plot 2: Physical flux before correction (top boundary)
    ax = axes[0, 1]
    ax.plot(xis, physical_fluxes_zero['top'], 'r-', label='Before correction', lw=2)
    ax.plot(xis, physical_fluxes_corrected['top'], 'b--', label='After correction', lw=2)
    ax.set_xlabel('ξ')
    ax.set_ylabel('Physical flux ∇u·n')
    ax.set_title('Top Boundary (η=1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    
    # Plot 3: Physical flux before correction (left boundary)
    ax = axes[1, 0]
    ax.plot(etas, physical_fluxes_zero['left'], 'r-', label='Before correction', lw=2)
    ax.plot(etas, physical_fluxes_corrected['left'], 'b--', label='After correction', lw=2)
    ax.set_xlabel('η')
    ax.set_ylabel('Physical flux ∇u·n')
    ax.set_title('Left Boundary (ξ=0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    
    # Plot 4: Physical flux before correction (right boundary)
    ax = axes[1, 1]
    ax.plot(etas, physical_fluxes_zero['right'], 'r-', label='Before correction', lw=2)
    ax.plot(etas, physical_fluxes_corrected['right'], 'b--', label='After correction', lw=2)
    ax.set_xlabel('η')
    ax.set_ylabel('Physical flux ∇u·n')
    ax.set_title('Right Boundary (ξ=1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('test_physical_flux_bc.png', dpi=150)
    print("Flux comparison plots saved to 'test_physical_flux_bc.png'")
    
    # Print mesh quality info
    print(f"\nMesh Quality Information:")
    print(f"  det(J) range: [{det_J.min():.6e}, {det_J.max():.6e}]")
    print(f"  min det(J) > 0: {np.all(det_J > 0)}")
    print(f"  mean |det(J)|: {np.abs(det_J).mean():.6e}")
    
    plt.show()


if __name__ == "__main__":
    test_boundary_flux_enforcement()

