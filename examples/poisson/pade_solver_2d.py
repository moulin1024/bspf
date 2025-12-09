"""
2D Poisson Equation Solver using Padé 4th Order Compact Finite Difference.

This script solves the 2D Poisson equation:
    -∇²u(x,y) = f(x,y),  (x,y) in [a, b] × [c, d]
    with Dirichlet boundary conditions

using the Method of Manufactured Solutions (MMS) to construct a test problem,
and Padé 4th order compact finite difference scheme.

Run from repository root:
    python examples/poisson/pade_solver_2d.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, bicgstab
from scipy.sparse.linalg import LinearOperator

from bspf.utils.padefd import padefd2


# ============================================================================
# Parameters
# ============================================================================

# Domain
a, b = 0.0, 1.0  # x-domain
c, d = 0.0, 1.0  # y-domain
nx = 51  # Number of grid points in x (including endpoints)
ny = 51  # Number of grid points in y (including endpoints)

# Padé scheme order
PADE_ORDER = 4  # 4th order Padé scheme


# ============================================================================
# Method of Manufactured Solutions (MMS)
# ============================================================================

def u_exact(x, y):
    """
    Exact solution for MMS test case.
    
    We choose: u(x,y) = sin(2πx) * sin(2πy) + x² + y²
    This gives smooth, non-trivial behavior.
    """
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) + x**2 + y**2


def f_source(x, y):
    """
    Source term f(x,y) = -∇²u(x,y) for the MMS test case.
    
    For u(x,y) = sin(2πx) * sin(2πy) + x² + y²:
        ∂²u/∂x² = -4π² sin(2πx) sin(2πy) + 2
        ∂²u/∂y² = -4π² sin(2πx) sin(2πy) + 2
        ∇²u = -8π² sin(2πx) sin(2πy) + 4
        f = -∇²u = 8π² sin(2πx) sin(2πy) - 4
    """
    return 8 * np.pi**2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) - 4


# ============================================================================
# 2D Matrix Assembly
# ============================================================================

def build_2d_laplacian_matrix(nx, ny, hx, hy, order=PADE_ORDER):
    """
    Build 2D Laplacian matrix using Kronecker product.
    
    For 2D: ∇²u = ∂²u/∂x² + ∂²u/∂y²
    
    Using Kronecker product:
        ∇²u ≈ (I_y ⊗ D2_x + D2_y ⊗ I_x) @ u
    
    where:
        - D2_x: second derivative matrix in x-direction (nx × nx)
        - D2_y: second derivative matrix in y-direction (ny × ny)
        - I_x: identity matrix (nx × nx)
        - I_y: identity matrix (ny × ny)
        - u: solution vector of size (nx * ny,)
    
    Parameters
    ----------
    nx, ny : int
        Number of grid points in x and y directions
    hx, hy : float
        Grid spacing in x and y directions
    order : int
        Padé scheme order
    
    Returns
    -------
    L2d : sparse matrix (CSR format)
        2D Laplacian matrix, shape (nx*ny, nx*ny)
    """
    # Build 1D second derivative matrices using padefd2
    d2op_x = padefd2(N=nx, h=hx, order=order)
    d2op_y = padefd2(N=ny, h=hy, order=order)
    
    # Build matrices by applying operators to unit vectors
    def build_matrix_from_operator(op, n):
        """Build matrix from operator by applying to unit vectors."""
        D_data = []
        D_row = []
        D_col = []
        I = np.eye(n)
        for j in range(n):
            result = op(I[:, j])
            for i in range(n):
                if abs(result[i]) > 1e-12:
                    D_row.append(i)
                    D_col.append(j)
                    D_data.append(result[i])
        return sp.csr_matrix((D_data, (D_row, D_col)), shape=(n, n))
    
    D2_x = build_matrix_from_operator(d2op_x, nx)
    D2_y = build_matrix_from_operator(d2op_y, ny)
    
    # Build identity matrices
    I_x = sp.eye(nx, format='csr')
    I_y = sp.eye(ny, format='csr')
    
    # Construct 2D Laplacian using Kronecker product
    # ∇²u = (I_y ⊗ D2_x + D2_y ⊗ I_x) @ u
    L2d = sp.kron(I_y, D2_x) + sp.kron(D2_y, I_x)
    
    return L2d


def apply_dirichlet_bc_2d(L2d, nx, ny, u_bc_func):
    """
    Apply Dirichlet boundary conditions to 2D Laplacian matrix.
    
    Parameters
    ----------
    L2d : sparse matrix
        2D Laplacian matrix
    nx, ny : int
        Grid dimensions
    u_bc_func : callable
        Function u_bc(x, y) that returns boundary values
    
    Returns
    -------
    A : sparse matrix
        Modified matrix with BCs
    """
    A = L2d.tolil()
    
    # Create grid
    x = np.linspace(a, b, nx)
    y = np.linspace(c, d, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Find boundary indices
    # Left boundary: i = 0
    # Right boundary: i = nx - 1
    # Bottom boundary: j = 0
    # Top boundary: j = ny - 1
    
    boundary_indices = []
    
    # Left and right boundaries
    for j in range(ny):
        # Left: (0, j) -> index = 0 * ny + j = j
        idx_left = j
        boundary_indices.append(idx_left)
        
        # Right: (nx-1, j) -> index = (nx-1) * ny + j
        idx_right = (nx - 1) * ny + j
        boundary_indices.append(idx_right)
    
    # Bottom and top boundaries
    for i in range(1, nx - 1):  # Exclude corners (already handled)
        # Bottom: (i, 0) -> index = i * ny + 0
        idx_bottom = i * ny
        boundary_indices.append(idx_bottom)
        
        # Top: (i, ny-1) -> index = i * ny + (ny-1)
        idx_top = i * ny + (ny - 1)
        boundary_indices.append(idx_top)
    
    # Remove duplicates and sort
    boundary_indices = sorted(set(boundary_indices))
    
    # Apply Dirichlet BCs: set boundary rows to identity
    for idx in boundary_indices:
        # Convert linear index to (i, j)
        i = idx // ny
        j = idx % ny
        x_val = x[i]
        y_val = y[j]
        
        # Set entire row to zero, then diagonal to 1
        A[idx, :] = 0.0
        A[idx, idx] = 1.0
    
    return A.tocsr(), boundary_indices


# ============================================================================
# Main Solver
# ============================================================================

def solve_poisson_2d(nx, ny, a, b, c, d, order=PADE_ORDER, use_iterative=False):
    """
    Solve 2D Poisson equation using Padé compact finite difference.
    
    Parameters
    ----------
    nx, ny : int
        Number of grid points in x and y
    a, b : float
        x-domain boundaries
    c, d : float
        y-domain boundaries
    order : int
        Padé scheme order
    use_iterative : bool
        If True, use BICGSTAB. If False, use direct solver.
    
    Returns
    -------
    x, y : arrays
        Grid points
    u_numerical : array
        Numerical solution (2D)
    u_exact : array
        Exact solution (2D)
    A : sparse matrix
        System matrix
    info : dict
        Solver information
    """
    # Create grid
    x = np.linspace(a, b, nx, endpoint=True)
    y = np.linspace(c, d, ny, endpoint=True)
    hx = x[1] - x[0]
    hy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Build 2D Laplacian matrix
    print("Building 2D Laplacian matrix...")
    L2d = build_2d_laplacian_matrix(nx, ny, hx, hy, order=order)
    
    # Apply boundary conditions
    print("Applying boundary conditions...")
    A, boundary_indices = apply_dirichlet_bc_2d(L2d, nx, ny, u_exact)
    
    # Build RHS
    f = f_source(X, Y)
    u_exact_2d = u_exact(X, Y)
    
    # Flatten for matrix-vector operations
    rhs = -f.flatten()  # -∇²u = f, so ∇²u = -f
    
    # Apply boundary conditions to RHS
    for idx in boundary_indices:
        i = idx // ny
        j = idx % ny
        rhs[idx] = u_exact_2d[i, j]
    
    # Solve
    info = {}
    if use_iterative:
        print("Using BICGSTAB iterative solver with Jacobi preconditioner...")
        # Create Jacobi preconditioner
        diag = A.diagonal()
        diag = np.where(np.abs(diag) > 1e-12, diag, 1.0)
        inv_diag = 1.0 / diag
        
        def apply_preconditioner(x):
            return inv_diag * x
        
        M = LinearOperator(shape=A.shape, matvec=apply_preconditioner, dtype=A.dtype)
        
        # Set tolerance and max iterations
        # For 2D problems, we need more iterations and slightly relaxed tolerance
        maxiter = nx * ny * 10  # Allow more iterations for larger systems
        rtol = 1e-8  # Slightly relaxed for iterative solver
        atol = 1e-10
        
        u_flat, exit_code = bicgstab(
            A, rhs, 
            M=M,  # Preconditioner
            rtol=rtol, 
            atol=atol, 
            maxiter=maxiter
        )
        
        info['method'] = 'BICGSTAB'
        info['exit_code'] = exit_code
        info['preconditioner'] = 'Jacobi'
        info['rtol'] = rtol
        info['maxiter'] = maxiter
        
        if exit_code == 0:
            print(f"  Converged successfully (rtol: {rtol})")
        elif exit_code > 0:
            print(f"  Warning: Did not converge after {maxiter} iterations (exit code: {exit_code})")
            print(f"  Consider increasing maxiter or relaxing tolerance")
        else:
            print(f"  Error: Illegal input or breakdown (exit code: {exit_code})")
    else:
        print("Using direct solver (spsolve)...")
        u_flat = spsolve(A, rhs)
        info['method'] = 'Direct (spsolve)'
        info['exit_code'] = 0
    
    # Reshape solution
    u_numerical = u_flat.reshape(nx, ny)
    
    return x, y, u_numerical, u_exact_2d, A, info


# ============================================================================
# Visualization
# ============================================================================

def plot_solution_2d(x, y, u_num, u_exact, error):
    """Plot 2D solution comparison and error."""
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    fig = plt.figure(figsize=(16, 5))
    
    # Plot 1: Numerical solution
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, u_num, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('$x$', fontsize=12)
    ax1.set_ylabel('$y$', fontsize=12)
    ax1.set_zlabel('$u(x,y)$', fontsize=12)
    ax1.set_title('Numerical Solution', fontsize=14, fontweight='bold')
    plt.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # Plot 2: Exact solution
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, u_exact, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('$x$', fontsize=12)
    ax2.set_ylabel('$y$', fontsize=12)
    ax2.set_zlabel('$u(x,y)$', fontsize=12)
    ax2.set_title('Exact Solution', fontsize=14, fontweight='bold')
    plt.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # Plot 3: Error
    ax3 = fig.add_subplot(1, 3, 3)
    im = ax3.contourf(X, Y, error, levels=20, cmap='Reds')
    ax3.set_xlabel('$x$', fontsize=12)
    ax3.set_ylabel('$y$', fontsize=12)
    ax3.set_title('Error', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('poisson_pade_2d.png', dpi=150, bbox_inches='tight')
    print("Plot saved to poisson_pade_2d.png")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("2D Poisson Equation Solver using Padé 4th Order Scheme")
    print("=" * 60)
    print(f"Domain: [{a}, {b}] × [{c}, {d}]")
    print(f"Grid points: {nx} × {ny}")
    print(f"Grid spacing: hx = {(b-a)/(nx-1):.6e}, hy = {(d-c)/(ny-1):.6e}")
    print(f"Padé order: {PADE_ORDER}")
    print()
    
    # Solve using BICGSTAB iterative solver
    x, y, u_num, u_exact_2d, A, info = solve_poisson_2d(
        nx, ny, a, b, c, d,
        order=PADE_ORDER,
        use_iterative=True
    )
    
    # Compute error
    error = np.abs(u_num - u_exact_2d)
    l2_error = np.sqrt(np.trapz(np.trapz(error**2, y, axis=1), x))
    max_error = np.max(error)
    
    print("\nSolution Statistics:")
    print(f"  L2 error:  {l2_error:.6e}")
    print(f"  Max error: {max_error:.6e}")
    print()
    
    # Solver information
    print("Solver Information:")
    print(f"  Method: {info.get('method', 'Unknown')}")
    if 'preconditioner' in info:
        print(f"  Preconditioner: {info['preconditioner']}")
    if 'exit_code' in info:
        exit_code = info['exit_code']
        if exit_code == 0:
            print(f"  Status: Converged")
        elif exit_code > 0:
            print(f"  Status: Not converged (exit code: {exit_code})")
        else:
            print(f"  Status: Error (exit code: {exit_code})")
    if 'rtol' in info:
        print(f"  Relative tolerance: {info['rtol']}")
    if 'maxiter' in info:
        print(f"  Max iterations: {info['maxiter']}")
    print()
    
    # Matrix statistics
    print("Matrix Statistics:")
    print(f"  Size: {A.shape[0]} × {A.shape[1]}")
    print(f"  Non-zero entries: {A.nnz}")
    print(f"  Sparsity: {(1 - A.nnz / (A.shape[0] * A.shape[1])) * 100:.2f}%")
    
    # Analyze bandwidth
    A_coo = A.tocoo()
    if len(A_coo.row) > 0:
        bandwidth = np.max(np.abs(A_coo.row - A_coo.col))
        print(f"  Estimated bandwidth: {bandwidth}")
    print()
    
    # Visualization
    print("Generating plots...")
    plot_solution_2d(x, y, u_num, u_exact_2d, error)
    
    print("\nDone!")

