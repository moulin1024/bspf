"""
Poisson Equation Solver using Padé 4th Order Compact Finite Difference.

This script solves the 1D Poisson equation:
    -u''(x) = f(x),  x in [a, b]
    u(a) = u_a,  u(b) = u_b  (non-homogeneous Dirichlet BCs)

using the Method of Manufactured Solutions (MMS) to construct a test problem,
and Padé 4th order compact finite difference scheme to assemble the matrix.

Run from repository root:
    python examples/poisson/pade_solver.py
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, bicgstab
from scipy.sparse.linalg import LinearOperator

try:
    from findiff import FinDiff
except ImportError:
    raise ImportError("This script requires findiff. Install via: pip install findiff")

from bspf.utils import padefd
from bspf.utils.padefd import padefd2


# ============================================================================
# Parameters
# ============================================================================

# Domain
a = 0.0
b = 1.0
nx = 101  # Number of grid points (including endpoints)

# Padé scheme order
PADE_ORDER = 4  # 4th order Padé scheme

# ============================================================================
# Method of Manufactured Solutions (MMS)
# ============================================================================

def u_exact(x):
    """
    Exact solution for MMS test case.
    
    We choose: u(x) = sin(2*pi*x) + x^2
    This gives smooth, non-trivial behavior.
    """
    return np.sin(2 * np.pi * x) + x**2


def f_source(x):
    """
    Source term f(x) = -u''(x) for the MMS test case.
    
    For u(x) = sin(2*pi*x) + x^2:
        u'(x) = 2*pi*cos(2*pi*x) + 2*x
        u''(x) = -4*pi^2*sin(2*pi*x) + 2
        f(x) = -u''(x) = 4*pi^2*sin(2*pi*x) - 2
    """
    return 4 * np.pi**2 * np.sin(2 * np.pi * x) - 2


def u_boundary_left():
    """Left boundary value u(a)."""
    return u_exact(a)


def u_boundary_right():
    """Right boundary value u(b)."""
    return u_exact(b)


# ============================================================================
# Padé 4th Order Matrix Assembly
# ============================================================================

def build_pade_first_derivative_matrix(nx, h, order=PADE_ORDER):
    """
    Build the first derivative matrix using Padé compact finite difference.
    
    This constructs the matrix representation of the Padé operator by applying
    it to unit vectors (columns of identity matrix).
    
    Parameters
    ----------
    nx : int
        Number of grid points
    h : float
        Grid spacing
    order : int
        Padé scheme order (4, 6, 8, 10, 12)
    
    Returns
    -------
    D1 : sparse matrix (CSR format)
        First derivative matrix, shape (nx, nx)
    """
    # Create Padé operator
    pade_op = padefd(N=nx, h=h, order=order)
    
    # Build matrix by applying operator to unit vectors
    D1_data = []
    D1_row = []
    D1_col = []
    
    # Apply operator to each unit vector (column of identity)
    I = np.eye(nx)
    for j in range(nx):
        result = pade_op(I[:, j])
        # Find non-zero entries
        for i in range(nx):
            if abs(result[i]) > 1e-12:
                D1_row.append(i)
                D1_col.append(j)
                D1_data.append(result[i])
    
    # Build sparse matrix
    D1 = sp.csr_matrix((D1_data, (D1_row, D1_col)), shape=(nx, nx))
    
    return D1


def derive_pade_second_derivative_coeffs(K, as_float=True):
    """
    Derive coefficients for compact Padé second derivative scheme.
    
    The compact Padé format for second derivative is:
        α * u''_{i-1} + u''_i + α * u''_{i+1} 
        = (1/h²) * Σ_{k=1}^K b_k * (u_{i+k} - 2*u_i + u_{i-k})
    
    Known coefficients:
    - K=1 (4th order): α = 1/10, b_1 = 12/10 = 6/5
    - K=2 (6th order): α = 2/11, b_1 = 12/11, b_2 = 3/11
    
    Parameters
    ----------
    K : int
        Half-width of the RHS stencil (K = 1, 2, 3, ...)
    as_float : bool
        If True, return floats; otherwise return exact SymPy rationals
    
    Returns
    -------
    alpha : float
        LHS coefficient α
    b_dict : dict
        Mapping k -> b_k for k=1..K
    order : int
        Formal order of accuracy (typically 2K+2)
    """
    # Use known coefficients for common cases
    known_coeffs = {
        1: (1/10, {1: 12/10}),  # 4th order
        2: (2/11, {1: 12/11, 2: 3/11}),  # 6th order
    }
    
    if K in known_coeffs:
        alpha, b_dict = known_coeffs[K]
        if as_float:
            return float(alpha), {k: float(v) for k, v in b_dict.items()}, 2*K + 2
        else:
            return alpha, b_dict, 2*K + 2
    
    # For other K, derive symbolically
    try:
        import sympy as sp
    except ImportError as e:
        raise ImportError("This function requires sympy for K > 2. Install via: pip install sympy") from e
    
    # Unknowns: alpha and b1..bK
    alpha = sp.symbols('alpha')
    b_syms = sp.symbols(' '.join([f'b{k}' for k in range(1, K+1)]))
    if K == 1:
        b_syms = (b_syms,)
    
    # Build equations by matching Taylor series coefficients
    # LHS: α * (u''_{i-1} + u''_{i+1}) + u''_i
    # RHS: (1/h²) * Σ b_k * (u_{i+k} - 2*u_i + u_{i-k})
    
    eqs = []
    for q in range(0, K+1):
        # Match coefficient of h^{2q} * u^{(2q+2)}
        if q == 0:
            # Coefficient of u''_i
            lhs = 1 + 2*alpha
            rhs = sum(bk * (k**2) for k, bk in enumerate(b_syms, start=1))
        else:
            # Higher order terms
            lhs = 2*alpha / sp.factorial(2*q)
            rhs = sum(bk * (k**(2*q + 2)) / sp.factorial(2*q + 2) 
                     for k, bk in enumerate(b_syms, start=1))
        
        eqs.append(sp.Eq(lhs, rhs))
    
    # Solve
    sol = sp.solve(eqs, (alpha, *b_syms), dict=True)
    if not sol:
        raise RuntimeError(f"No solution found for K={K}")
    sol = sol[0]
    
    alpha_val = sp.nsimplify(sol[alpha])
    b_vals = {k: sp.nsimplify(sol[b_syms[k-1]]) for k in range(1, K+1)}
    
    if as_float:
        alpha_val = float(alpha_val)
        b_vals = {k: float(v) for k, v in b_vals.items()}
    
    order = 2*K + 2
    return alpha_val, b_vals, order


def build_pade_second_derivative_matrix_direct(nx, h, order=PADE_ORDER):
    """
    Build compact Padé second derivative matrix directly (not via D1 @ D1).
    
    This constructs a truly compact second derivative matrix using the Padé format:
        α * u''_{i-1} + u''_i + α * u''_{i+1} = (1/h²) * Σ b_k * (u_{i+k} - 2*u_i + u_{i-k})
    
    This is much more compact than D1 @ D1, with bandwidth ~(2K+1) instead of much wider.
    
    Parameters
    ----------
    nx : int
        Number of grid points
    h : float
        Grid spacing
    order : int
        Padé scheme order (4, 6, 8, 10, 12)
    
    Returns
    -------
    D2 : sparse matrix (CSR format)
        Compact second derivative matrix, shape (nx, nx)
    """
    # Map order to K (half-stencil width)
    # For second derivative Padé: order = 2K + 2
    K = (order - 2) // 2
    if K < 1:
        raise ValueError(f"Order {order} too low for second derivative Padé (minimum K=1, order=4)")
    
    # Derive coefficients
    alpha, b_dict, _ = derive_pade_second_derivative_coeffs(K, as_float=True)
    
    # For known schemes, we can also use pre-computed values
    # 4th order (K=1): α = 1/10, b_1 = 12/10 = 6/5
    # 6th order (K=2): α = 2/11, b_1 = 12/11, b_2 = 3/11
    # Let's use derived values for now
    
    # Build the matrix
    # The system is: L @ u'' = (1/h²) * R @ u
    # where L is tridiagonal with (α, 1, α) on each row
    # and R has the RHS stencil
    
    # We need to solve: u'' = L^{-1} @ ((1/h²) * R @ u) = D2 @ u
    # So D2 = (1/h²) * L^{-1} @ R
    
    # For now, let's construct it by applying to unit vectors
    # This is simpler and more robust
    
    # Create LHS matrix (tridiagonal: α, 1, α)
    L_data = []
    L_row = []
    L_col = []
    
    for i in range(nx):
        if i > 0:
            L_row.append(i)
            L_col.append(i-1)
            L_data.append(alpha)
        L_row.append(i)
        L_col.append(i)
        L_data.append(1.0)
        if i < nx - 1:
            L_row.append(i)
            L_col.append(i+1)
            L_data.append(alpha)
    
    L = sp.csr_matrix((L_data, (L_row, L_col)), shape=(nx, nx))
    
    # Create RHS matrix (RHS stencil)
    R_data = []
    R_row = []
    R_col = []
    
    for i in range(nx):
        for k, bk in b_dict.items():
            # Term: b_k * (u_{i+k} - 2*u_i + u_{i-k})
            if i + k < nx:
                R_row.append(i)
                R_col.append(i + k)
                R_data.append(bk)
            if i - k >= 0:
                R_row.append(i)
                R_col.append(i - k)
                R_data.append(bk)
            # -2*b_k * u_i
            R_row.append(i)
            R_col.append(i)
            R_data.append(-2 * bk)
    
    R = sp.csr_matrix((R_data, (R_row, R_col)), shape=(nx, nx))
    
    # Boundary treatment: use one-sided stencils from findiff
    # Replace boundary rows in L and R
    acc = min(order, 10)
    D_boundary = FinDiff(0, h, 2, acc=acc).matrix((nx,)).tocsr()
    
    # Replace first K and last K rows
    K_boundary = min(K, nx // 4)  # Use at least K boundary points
    L = L.tolil()
    R = R.tolil()
    
    for i in range(K_boundary):
        # Left boundary
        L[i, :] = 0.0
        L[i, i] = 1.0
        R[i, :] = D_boundary[i, :] * (h**2)  # Scale to match our format
        
        # Right boundary
        j = nx - 1 - i
        L[j, :] = 0.0
        L[j, j] = 1.0
        R[j, :] = D_boundary[j, :] * (h**2)
    
    L = L.tocsr()
    R = R.tocsr()
    
    # D2 = (1/h²) * L^{-1} @ R
    # Solve L @ D2 = (1/h²) * R for each column
    D2_data = []
    D2_row = []
    D2_col = []
    
    # Solve for each column of D2
    from scipy.sparse.linalg import spsolve
    for j in range(nx):
        rhs_col = (1.0 / (h**2)) * R[:, j].toarray().flatten()
        d2_col = spsolve(L, rhs_col)
        # Store non-zero entries
        for i in range(nx):
            if abs(d2_col[i]) > 1e-12:
                D2_row.append(i)
                D2_col.append(j)
                D2_data.append(d2_col[i])
    
    D2 = sp.csr_matrix((D2_data, (D2_row, D2_col)), shape=(nx, nx))
    
    return D2


def build_pade_laplacian_matrix(nx, h, order=PADE_ORDER, use_padefd2=True):
    """
    Build the Laplacian (second derivative) matrix using Padé compact finite difference.
    
    For Poisson equation: -u'' = f, we need the second derivative operator.
    
    Parameters
    ----------
    nx : int
        Number of grid points
    h : float
        Grid spacing
    order : int
        Padé scheme order (4, 6, 8, 10, 12)
    use_padefd2 : bool
        If True, use padefd2 class (most compact, recommended)
        If False, use direct construction or D1 @ D1
    
    Returns
    -------
    L : sparse matrix (CSR format)
        Second derivative matrix (Laplacian), shape (nx, nx)
    """
    if use_padefd2 and order == 4:
        # Use padefd2 class - most elegant and compact
        d2op = padefd2(N=nx, h=h, order=order)
        
        # Build matrix by applying operator to unit vectors
        D2_data = []
        D2_row = []
        D2_col = []
        
        I = np.eye(nx)
        for j in range(nx):
            result = d2op(I[:, j])
            # Find non-zero entries
            for i in range(nx):
                if abs(result[i]) > 1e-12:
                    D2_row.append(i)
                    D2_col.append(j)
                    D2_data.append(result[i])
        
        D2 = sp.csr_matrix((D2_data, (D2_row, D2_col)), shape=(nx, nx))
        return D2
    elif use_padefd2 is False:
        # Fall back to direct construction
        return build_pade_second_derivative_matrix_direct(nx, h, order=order)
    else:
        # For orders other than 4, use direct construction
        return build_pade_second_derivative_matrix_direct(nx, h, order=order)


def build_poisson_matrix_with_bc(nx, h, order=PADE_ORDER):
    """
    Build the Poisson equation matrix with Dirichlet boundary conditions.
    
    The system is: -u'' = f with u(a) = u_a, u(b) = u_b
    
    We enforce Dirichlet BCs by replacing boundary rows with identity.
    This is the standard approach and ensures exact boundary enforcement.
    
    Parameters
    ----------
    nx : int
        Number of grid points
    h : float
        Grid spacing
    order : int
        Padé scheme order
    
    Returns
    -------
    A : sparse matrix (CSR format)
        Modified Poisson matrix with BCs, shape (nx, nx)
    """
    # Build Laplacian matrix (use padefd2 class for order=4, most compact)
    L = build_pade_laplacian_matrix(nx, h, order=order, use_padefd2=True)
    
    # For Poisson: -u'' = f, so we need -L
    # Convert to LIL format for efficient row modification
    A = -L.tolil()
    
    # Apply Dirichlet boundary conditions
    # Left boundary: u[0] = u_a
    # Set entire row to zero, then set diagonal to 1
    A[0, :] = 0.0
    A[0, 0] = 1.0
    
    # Right boundary: u[-1] = u_b
    # Set entire row to zero, then set diagonal to 1
    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    
    return A.tocsr()


# ============================================================================
# Main Solver
# ============================================================================

def create_jacobi_preconditioner(A):
    """
    Create Jacobi (diagonal) preconditioner for matrix A.
    
    The Jacobi preconditioner is M = diag(A)^{-1}, i.e., the inverse
    of the diagonal of A.
    
    Parameters
    ----------
    A : sparse matrix
        System matrix
    
    Returns
    -------
    M : LinearOperator
        Preconditioner operator that applies M @ x
    """
    # Extract diagonal
    diag = A.diagonal()
    
    # Avoid division by zero (shouldn't happen for well-posed problems)
    diag = np.where(np.abs(diag) > 1e-12, diag, 1.0)
    
    # Preconditioner: M @ x = diag^{-1} @ x
    inv_diag = 1.0 / diag
    
    def apply_preconditioner(x):
        """Apply Jacobi preconditioner: M @ x = diag^{-1} * x (element-wise)"""
        return inv_diag * x
    
    # Create LinearOperator
    M = LinearOperator(
        shape=A.shape,
        matvec=apply_preconditioner,
        dtype=A.dtype
    )
    
    return M


def solve_poisson_pade(nx, a, b, order=PADE_ORDER, use_iterative=True, tol=1e-10, maxiter=None):
    """
    Solve Poisson equation using Padé compact finite difference.
    
    Parameters
    ----------
    nx : int
        Number of grid points
    a, b : float
        Domain boundaries
    order : int
        Padé scheme order
    use_iterative : bool
        If True, use BICGSTAB with Jacobi preconditioner.
        If False, use direct solver (spsolve).
    tol : float
        Tolerance for iterative solver
    maxiter : int, optional
        Maximum iterations for iterative solver. If None, use default (nx).
    
    Returns
    -------
    x : array
        Grid points
    u_numerical : array
        Numerical solution
    u_exact : array
        Exact solution
    A : sparse matrix
        System matrix (for sparsity visualization)
    info : dict
        Solver information (iterations, convergence, etc.)
    """
    # Create grid
    x = np.linspace(a, b, nx, endpoint=True)
    h = x[1] - x[0]
    
    # Build matrix
    A = build_poisson_matrix_with_bc(nx, h, order=order)
    
    # Build RHS: f(x) with boundary conditions
    f = f_source(x)
    rhs = f.copy()
    
    # Apply Dirichlet boundary conditions to RHS
    # Since we've set A[0,0]=1 and A[-1,-1]=1 with zeros elsewhere in those rows,
    # setting rhs[0] and rhs[-1] to the boundary values enforces u[0]=u_a, u[-1]=u_b
    rhs[0] = u_boundary_left()   # u(a) = u_a
    rhs[-1] = u_boundary_right()  # u(b) = u_b
    
    # Solve linear system
    info = {}
    
    if use_iterative:
        # Use BICGSTAB with Jacobi preconditioner
        print("Using BICGSTAB iterative solver with Jacobi preconditioner...")
        
        # Create preconditioner
        M = create_jacobi_preconditioner(A)
        
        # Set default maxiter if not provided
        if maxiter is None:
            maxiter = nx
        
        # Solve: A @ u = rhs
        u_numerical, exit_code = bicgstab(
            A, rhs,
            M=M,  # Preconditioner
            rtol=tol,  # Relative tolerance
            atol=1e-12,  # Absolute tolerance
            maxiter=maxiter
        )
        
        info['method'] = 'BICGSTAB'
        info['exit_code'] = exit_code
        info['preconditioner'] = 'Jacobi'
        
        if exit_code == 0:
            print(f"  Converged successfully (rtol: {tol})")
        elif exit_code > 0:
            print(f"  Warning: Did not converge after {maxiter} iterations (exit code: {exit_code})")
            print(f"  Consider increasing maxiter or relaxing tolerance")
        else:
            print(f"  Error: Illegal input or breakdown (exit code: {exit_code})")
    else:
        # Use direct solver
        print("Using direct solver (spsolve)...")
        u_numerical = spsolve(A, rhs)
        info['method'] = 'Direct (spsolve)'
        info['exit_code'] = 0
    
    # Exact solution
    u_exact_vals = u_exact(x)
    
    return x, u_numerical, u_exact_vals, A, info


# ============================================================================
# Visualization
# ============================================================================

def plot_sparsity_pattern(A, title="Sparsity Pattern"):
    """
    Plot the sparsity pattern of matrix A.
    
    Parameters
    ----------
    A : sparse matrix
        Matrix to visualize
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 10))
    plt.spy(A, markersize=2, precision=1e-10)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Column index', fontsize=12)
    plt.ylabel('Row index', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Poisson Equation Solver using Padé 4th Order Scheme")
    print("=" * 60)
    print(f"Domain: [{a}, {b}]")
    print(f"Grid points: {nx}")
    print(f"Grid spacing: h = {(b-a)/(nx-1):.6e}")
    print(f"Padé order: {PADE_ORDER}")
    print(f"Boundary conditions: u({a}) = {u_boundary_left():.6f}, u({b}) = {u_boundary_right():.6f}")
    print()
    
    # Solve using direct solver for accuracy (iterative solver may have convergence issues)
    use_iterative = False  # Use direct solver for better accuracy
    tol = 1e-10  # Relative tolerance for iterative solver (if used)
    maxiter = 2 * nx  # Maximum iterations (allow more iterations for convergence)
    
    x, u_num, u_exact_vals, A, info = solve_poisson_pade(
        nx, a, b, 
        order=PADE_ORDER,
        use_iterative=use_iterative,
        tol=tol,
        maxiter=maxiter
    )
    
    # Compute error
    error = np.abs(u_num - u_exact_vals)
    l2_error = np.sqrt(np.trapz(error**2, x))
    max_error = np.max(error)
    
    print("Solution Statistics:")
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
    print()
    
    # Matrix statistics
    print("Matrix Statistics:")
    print(f"  Size: {A.shape[0]} x {A.shape[1]}")
    print(f"  Non-zero entries: {A.nnz}")
    print(f"  Sparsity: {(1 - A.nnz / (A.shape[0] * A.shape[1])) * 100:.2f}%")
    
    # Analyze bandwidth
    # For a sparse matrix, we can estimate bandwidth by finding max distance from diagonal
    A_coo = A.tocoo()
    if len(A_coo.row) > 0:
        bandwidth = np.max(np.abs(A_coo.row - A_coo.col))
        print(f"  Estimated bandwidth: {bandwidth} (half-bandwidth: {bandwidth//2})")
        print(f"  Note: Using direct compact Padé second derivative construction")
        print(f"        (more compact than D1 @ D1, bandwidth ~{bandwidth} vs ~50 for D1@D1)")
    print()
    
    # Plotting
    fig = plt.figure(figsize=(16, 6))
    
    # Plot 1: Solution comparison
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(x, u_exact_vals, 'r-', label='Exact', linewidth=2, alpha=0.8)
    ax1.plot(x, u_num, 'b--', label='Numerical', linewidth=2, alpha=0.8, markersize=4)
    ax1.set_xlabel('$x$', fontsize=12)
    ax1.set_ylabel('$u(x)$', fontsize=12)
    ax1.set_title('Solution Comparison', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error (with emphasis on boundary regions)
    ax2 = plt.subplot(1, 3, 2)
    ax2.semilogy(x, error, 'r-', linewidth=2, alpha=0.8, label='Error')
    # Highlight boundary regions
    boundary_region = int(0.1 * nx)  # First and last 10% of points
    ax2.axvspan(x[0], x[boundary_region], alpha=0.1, color='blue', label='Boundary region')
    ax2.axvspan(x[-boundary_region], x[-1], alpha=0.1, color='blue')
    ax2.set_xlabel('$x$', fontsize=12)
    ax2.set_ylabel('$|Error|$', fontsize=12)
    ax2.set_title('Pointwise Error (Log Scale)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sparsity pattern
    ax3 = plt.subplot(1, 3, 3)
    ax3.spy(A, markersize=1.5, precision=1e-10)
    ax3.set_xlabel('Column index', fontsize=12)
    ax3.set_ylabel('Row index', fontsize=12)
    ax3.set_title(f'Sparsity Pattern (Padé {PADE_ORDER}th Order)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('poisson_pade_solver.png', dpi=150, bbox_inches='tight')
    print("Plot saved to poisson_pade_solver.png")
    
    # Also create a separate large sparsity pattern plot
    fig_sparse = plot_sparsity_pattern(A, title=f'Poisson Matrix Sparsity Pattern (Padé {PADE_ORDER}th Order)')
    plt.savefig('poisson_pade_sparsity.png', dpi=150, bbox_inches='tight')
    print("Sparsity pattern saved to poisson_pade_sparsity.png")
    
    plt.show()

