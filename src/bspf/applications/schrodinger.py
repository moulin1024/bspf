"""
Schrödinger equation solver utilities.

Provides reusable functions for solving the Schrödinger equation:
    i*ℏ*∂ψ/∂t = -ℏ²/(2m)*∂²ψ/∂x² + V(x)*ψ + g*|ψ|²*ψ

In non-dimensional form (ℏ = m = 1):
    i*∂ψ/∂t = -(1/2)*∂²ψ/∂x² + V(x)*ψ + g*|ψ|²*ψ

Supports both linear (g=0) and nonlinear (g≠0) cases.
Supports both CPU (NumPy) and GPU (CuPy) computing.
"""

import numpy as np
from typing import Callable, Optional, Union
from ..bspf1d import bspf1d
from ..time_steppers import TimeStepperState, time_step

# Optional GPU backend
_HAS_CUPY = False
try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None


def create_schrodinger_rhs(bspf_op: bspf1d, V: np.ndarray, g: float = 0.0,
                           enforce_bc: Optional[Callable] = None,
                           neumann_bc: Optional[tuple] = None) -> Callable:
    """
    Create RHS function for Schrödinger equation.
    
    The equation in non-dimensional form (ℏ = m = 1):
        i*∂ψ/∂t = -(1/2)*∂²ψ/∂x² + V(x)*ψ + g*|ψ|²*ψ
    
    Parameters:
    -----------
    bspf_op : bspf1d
        BSPF operator for spatial differentiation
    V : array
        Potential V(x) (dimensionless, can be NumPy or CuPy array)
    g : float, optional
        Nonlinearity parameter. g=0 for linear case, g≠0 for nonlinear case.
        Default: 0.0 (linear Schrödinger equation)
    enforce_bc : callable, optional
        Function to enforce boundary conditions: enforce_bc(psi) -> psi_bc
        Used for Dirichlet BCs. If None, no BC enforcement is applied.
        Default: None
    neumann_bc : tuple, optional
        Neumann boundary conditions: (left_flux, right_flux) where
        left_flux = ∂ψ/∂x at left boundary, right_flux = ∂ψ/∂x at right boundary.
        If None, no Neumann BCs are applied.
        Default: None
    
    Returns:
    --------
    rhs_func : callable
        RHS function with signature: rhs_func(psi) -> dpsi_dt
    """
    # Detect backend from V array (GPU-aware)
    if _HAS_CUPY and isinstance(V, cp.ndarray):
        xp = cp
    else:
        xp = np
    
    def schrodinger_rhs(psi: np.ndarray) -> np.ndarray:
        """
        RHS for Schrödinger equation.
        GPU-aware: automatically works with CuPy arrays if V is a GPU array.
        
        Parameters:
        -----------
        psi : complex array
            Complex wavefunction (dimensionless, NumPy or CuPy array)
        
        Returns:
        --------
        dpsi_dt : complex array
            Time derivative of wavefunction (dimensionless)
        """
        # Enforce Dirichlet boundary conditions if provided
        if enforce_bc is not None:
            psi_bc = enforce_bc(psi)
        else:
            psi_bc = psi.copy()
        
        # Compute second derivative using bspf (handles complex arrays and GPU)
        # Pass neumann_bc to differentiate() to enforce Neumann BCs during differentiation
        if neumann_bc is not None:
            d2psi_dx2, _ = bspf_op.differentiate(psi_bc, k=2, neumann_bc=neumann_bc)
        else:
            d2psi_dx2, _ = bspf_op.differentiate(psi_bc, k=2)
        
        # Enforce Dirichlet BCs on derivative if BC function provided
        if enforce_bc is not None:
            d2psi_dx2 = enforce_bc(d2psi_dx2)
        
        # Linear terms: kinetic energy + potential
        # In non-dimensional units (ℏ = m = 1):
        # ψ_t = (i/2)*ψ_xx - i*V(x)*ψ
        linear_term = 0.5j * d2psi_dx2 - 1j * V * psi_bc
        
        # Nonlinear term: g*|ψ|²*ψ (only if g ≠ 0)
        if g != 0.0:
            # In non-dimensional units: ψ_t += -i*g*|ψ|²*ψ
            # Use xp.abs for GPU-aware absolute value
            nonlinear_term = -1j * g * xp.abs(psi_bc)**2 * psi_bc
            dpsi_dt = linear_term + nonlinear_term
        else:
            dpsi_dt = linear_term
        
        # Enforce Dirichlet boundary conditions on RHS if provided
        # For Dirichlet BCs, dψ/dt = 0 at boundaries (since ψ is fixed)
        if enforce_bc is not None:
            dpsi_dt[0] = 0.0   # Left boundary: dψ/dt = 0
            dpsi_dt[-1] = 0.0  # Right boundary: dψ/dt = 0
        
        return dpsi_dt
    
    return schrodinger_rhs


def create_dirichlet_bc_enforcer(n_grid: int) -> Callable:
    """
    Create a boundary condition enforcer for Dirichlet BCs: ψ = 0 at boundaries.
    
    Parameters:
    -----------
    n_grid : int
        Number of grid points
    
    Returns:
    --------
    enforce_bc : callable
        Function that enforces Dirichlet BCs: enforce_bc(psi) -> psi_bc
    """
    def enforce_bc(psi: np.ndarray) -> np.ndarray:
        """Enforce Dirichlet BCs: ψ = 0 at boundaries."""
        psi_bc = psi.copy()
        psi_bc[0] = 0.0
        psi_bc[-1] = 0.0
        return psi_bc
    
    return enforce_bc


def create_neumann_bc_enforcer(n_grid: int, left_flux: Optional[float] = None,
                                right_flux: Optional[float] = None) -> tuple:
    """
    Create Neumann BC parameters for use with bspf_op.differentiate().
    
    Note: Neumann BCs are enforced during differentiation, not by modifying
    the solution array. This function returns the BC tuple to pass to
    bspf_op.differentiate(..., neumann_bc=...).
    
    Parameters:
    -----------
    n_grid : int
        Number of grid points (not used, kept for API consistency)
    left_flux : float, optional
        Flux at left boundary: ∂ψ/∂x(0) = left_flux. If None, uses zero flux.
    right_flux : float, optional
        Flux at right boundary: ∂ψ/∂x(L) = right_flux. If None, uses zero flux.
    
    Returns:
    --------
    neumann_bc : tuple
        Tuple (left_flux, right_flux) to pass to bspf_op.differentiate()
    """
    left_flux = left_flux if left_flux is not None else 0.0
    right_flux = right_flux if right_flux is not None else 0.0
    return (left_flux, right_flux)


def solve_schrodinger(psi_init: np.ndarray, x: np.ndarray, V: np.ndarray,
                     dt: float, n_steps: int, 
                     bspf_op: Optional[bspf1d] = None,
                     degree: int = 5,
                     g: float = 0.0,
                     time_method: str = 'rk45',
                     bc_type: str = 'dirichlet',
                     bc_params: Optional[dict] = None,
                     save_callback: Optional[Callable] = None,
                     save_interval: int = 1,
                     use_gpu: bool = False) -> tuple:
    """
    Solve the Schrödinger equation.
    
    Parameters:
    -----------
    psi_init : complex array
        Initial wavefunction
    x : array
        Spatial grid (dimensionless)
    V : array
        Potential V(x) (dimensionless)
    dt : float
        Time step (dimensionless)
    n_steps : int
        Number of time steps
    bspf_op : bspf1d, optional
        BSPF operator. If None, creates one from the grid.
    degree : int, optional
        BSPF spline degree (used if bspf_op is None). Default: 5
    g : float, optional
        Nonlinearity parameter. g=0 for linear, g≠0 for nonlinear. Default: 0.0
    time_method : str, optional
        Time stepping method: 'rk4', 'rk45', 'rk23', or 'bdf2'. Default: 'rk45'
    bc_type : str, optional
        Boundary condition type: 'dirichlet' or 'neumann'. Default: 'dirichlet'
    bc_params : dict, optional
        Additional BC parameters (e.g., {'left_flux': 0.0, 'right_flux': 0.0} for Neumann)
    save_callback : callable, optional
        Callback function called every save_interval steps: save_callback(n, t, psi)
    save_interval : int, optional
        Interval for calling save_callback. Default: 1
    use_gpu : bool, optional
        If True, use GPU acceleration with CuPy. Default: False
    
    Returns:
    --------
    psi_final : array
        Final wavefunction (converted back to NumPy if GPU was used)
    times : array
        Time points at which solution was saved (if save_callback provided)
    psi_history : list, optional
        History of solutions (only if save_callback saves them)
    """
    # Check GPU availability
    if use_gpu and not _HAS_CUPY:
        raise RuntimeError(
            "use_gpu=True but CuPy is not available. "
            "Install cupy (e.g., `pip install cupy-cuda12x`) or set use_gpu=False."
        )
    
    n_grid = len(x)
    
    # Convert arrays to GPU if needed
    if use_gpu and _HAS_CUPY:
        xp = cp
        psi_init = cp.asarray(psi_init)
        x = cp.asarray(x)
        V = cp.asarray(V)
    else:
        xp = np
        # Ensure arrays are NumPy arrays
        psi_init = np.asarray(psi_init)
        x = np.asarray(x)
        V = np.asarray(V)
    
    # Create BSPF operator if not provided
    if bspf_op is None:
        bspf_op = bspf1d.from_grid(
            degree=degree,
            x=x,
            order=degree,
            use_clustering=True,
            clustering_factor=2.0,
            correction='spectral',
            use_gpu=use_gpu
        )
    
    # Set up boundary conditions
    enforce_bc = None
    neumann_bc = None
    
    if bc_type.lower() == 'dirichlet':
        enforce_bc = create_dirichlet_bc_enforcer(n_grid)
    elif bc_type.lower() == 'neumann':
        bc_params = bc_params or {}
        neumann_bc = create_neumann_bc_enforcer(
            n_grid,
            left_flux=bc_params.get('left_flux', 0.0),
            right_flux=bc_params.get('right_flux', 0.0)
        )
    # else: both enforce_bc and neumann_bc remain None (no BCs)
    
    # Create RHS function
    rhs_func = create_schrodinger_rhs(bspf_op, V, g=g, enforce_bc=enforce_bc, neumann_bc=neumann_bc)
    
    # Initialize time stepper
    T_final = n_steps * dt
    psi = psi_init.copy()
    
    # Apply initial BCs (Dirichlet only - Neumann BCs are enforced during differentiation)
    if enforce_bc is not None:
        psi = enforce_bc(psi)
    
    times = []
    psi_history = []
    
    with TimeStepperState(psi.copy(), t_init=0.0, dt=dt, method=time_method,
                         t_final=T_final, show_progress=False) as state:
        for nn in range(n_steps):
            # Time step
            psi_next = time_step(state, dt, rhs_func, method=time_method)
            psi = state.get_current()
            
            # Enforce Dirichlet BCs (safety check - Neumann BCs are enforced during differentiation)
            if enforce_bc is not None:
                psi = enforce_bc(psi)
                state.psi_now = psi.copy()
            
            # Save callback
            if save_callback is not None and nn % save_interval == 0:
                t_current = state.get_current_time()
                # Convert to NumPy for callback if on GPU
                psi_callback = cp.asnumpy(psi) if use_gpu and _HAS_CUPY else psi
                save_callback(nn, t_current, psi_callback)
                times.append(t_current)
                # Store copy (convert to NumPy if on GPU)
                if use_gpu and _HAS_CUPY:
                    psi_history.append(cp.asnumpy(psi.copy()))
                else:
                    psi_history.append(psi.copy())
    
    # Convert final result back to NumPy if GPU was used
    if use_gpu and _HAS_CUPY:
        psi = cp.asnumpy(psi)
    
    return psi, np.array(times) if times else None, psi_history if psi_history else None

