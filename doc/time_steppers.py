"""
Time stepping methods for NLSE solvers.
Shared module for RK4, RK45, RK23, and BDF time integrators with uniform interface.
BDF method uses scipy.integrate.solve_ivp.
Supports both CPU (NumPy) and GPU (CuPy) computing (BDF is CPU only).
"""

import numpy as np
from numpy.linalg import solve, norm
from typing import Callable, Optional, Dict, Any

try:
    from scipy.integrate import solve_ivp
    HAS_SCIPY_INTEGRATE = True
except ImportError:
    HAS_SCIPY_INTEGRATE = False
    solve_ivp = None

# Optional GPU backend
_HAS_CUPY = False
try:
    import cupy as cp
    import cupyx.scipy.linalg as cpla
    _HAS_CUPY = True
except Exception:
    cp = None
    cpla = None

try:
    from scipy.sparse.linalg import LinearOperator, gmres
    HAS_SCIPY_SPARSE = True
except ImportError:
    HAS_SCIPY_SPARSE = False
    LinearOperator = None

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Dummy tqdm class if not available
    class tqdm:
        def __init__(self, *args, **kwargs):
            pass
        def update(self, *args, **kwargs):
            pass
        def close(self):
            pass
        def refresh(self):
            pass


# ============================================================
#  GPU/CPU detection and backend selection
# ============================================================
def _get_backend(psi):
    """
    Detect if array is CuPy or NumPy and return appropriate backend.
    
    Parameters:
    -----------
    psi : array
        Array to check (can be NumPy or CuPy)
    
    Returns:
    --------
    xp : module
        Array module (numpy or cupy)
    la : module
        Linear algebra module (scipy.linalg or cupyx.scipy.linalg)
    is_gpu : bool
        True if GPU array detected
    """
    if _HAS_CUPY and isinstance(psi, cp.ndarray):
        return cp, cpla, True
    else:
        return np, None, False  # Use numpy.linalg for CPU


# ============================================================
#  RK45 time stepping (Dormand-Prince 4(5) embedded pair)
# ============================================================
def rk45_step(psi, dt, rhs_func, *args):
    """
    Fifth-order Runge-Kutta time stepping (Dormand-Prince method).
    This is an embedded 4(5) pair, but we use the 5th order solution for fixed-step.
    GPU-aware: automatically detects and uses CuPy if input is a GPU array.
    
    Parameters:
    -----------
    psi : array
        Current solution state (NumPy or CuPy array)
    dt : float
        Time step
    rhs_func : callable
        Function that computes RHS: rhs_func(psi, *args) -> dpsi/dt
    *args : additional arguments
        Additional arguments to pass to rhs_func
        
    Returns:
    --------
    array : Updated solution at next time step (5th order accurate)
    """
    xp, _, _ = _get_backend(psi)
    
    # Dormand-Prince 4(5) stage coefficients
    a21 = 1/5
    a31 = 3/40
    a32 = 9/40
    a41 = 44/45
    a42 = -56/15
    a43 = 32/9
    a51 = 19372/6561
    a52 = -25360/2187
    a53 = 64448/6561
    a54 = -212/729
    a61 = 9017/3168
    a62 = -355/33
    a63 = 46732/5247
    a64 = 49/176
    a65 = -5103/18656
    
    # 5th order solution coefficients
    b1 = 35/384
    b2 = 0
    b3 = 500/1113
    b4 = 125/192
    b5 = -2187/6784
    b6 = 11/84
    
    k1 = rhs_func(psi, *args)
    k2 = rhs_func(psi + dt*a21*k1, *args)
    k3 = rhs_func(psi + dt*(a31*k1 + a32*k2), *args)
    k4 = rhs_func(psi + dt*(a41*k1 + a42*k2 + a43*k3), *args)
    k5 = rhs_func(psi + dt*(a51*k1 + a52*k2 + a53*k3 + a54*k4), *args)
    k6 = rhs_func(psi + dt*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5), *args)
    
    # Compute 5th order solution
    psi_next = psi + dt*(b1*k1 + b2*k2 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
    
    # k7 would be computed from psi_next for FSAL property, but we don't need it for fixed-step
    return psi_next


# ============================================================
#  RK4 time stepping (explicit) - kept for backward compatibility
# ============================================================
def rk4_step(psi, dt, rhs_func, *args):
    """
    Fourth-order Runge-Kutta time stepping (explicit).
    GPU-aware: automatically detects and uses CuPy if input is a GPU array.
    
    Parameters:
    -----------
    psi : array
        Current solution state (NumPy or CuPy array)
    dt : float
        Time step
    rhs_func : callable
        Function that computes RHS: rhs_func(psi, *args) -> dpsi/dt
    *args : additional arguments
        Additional arguments to pass to rhs_func
        
    Returns:
    --------
    array : Updated solution at next time step
    """
    k1 = rhs_func(psi, *args)
    k2 = rhs_func(psi + 0.5*dt*k1, *args)
    k3 = rhs_func(psi + 0.5*dt*k2, *args)
    k4 = rhs_func(psi + dt*k3, *args)
    
    return psi + dt*(k1 + 2*k2 + 2*k3 + k4)/6


# ============================================================
#  RK23 time stepping (Bogacki-Shampine, 2(3) embedded pair)
# ============================================================
def rk23_step(psi, dt, rhs_func, *args):
    """
    Third-order Runge-Kutta time stepping (Bogacki-Shampine method).
    This is an embedded 2(3) pair, but we use the 3rd order solution for fixed-step.
    GPU-aware: automatically detects and uses CuPy if input is a GPU array.
    
    Parameters:
    -----------
    psi : array
        Current solution state (NumPy or CuPy array)
    dt : float
        Time step
    rhs_func : callable
        Function that computes RHS: rhs_func(psi, *args) -> dpsi/dt
    *args : additional arguments
        Additional arguments to pass to rhs_func
        
    Returns:
    --------
    array : Updated solution at next time step (3rd order accurate)
    """
    k1 = rhs_func(psi, *args)
    k2 = rhs_func(psi + 0.5*dt*k1, *args)
    k3 = rhs_func(psi + 0.75*dt*k2, *args)
    
    # Use 3rd order solution for fixed-step integration
    # Note: k4 would be needed for the 2nd order solution, but we skip it here
    return psi + dt*(2*k1/9 + k2/3 + 4*k3/9)


# ============================================================
#  Leapfrog time stepping (2nd order, explicit, requires 2 previous states)
# ============================================================
def leapfrog_step(psi_prev, psi_now, dt, rhs_func, *args):
    """
    Leapfrog time stepping (2nd order explicit method):
        u^{n+1} = u^{n-1} + 2*dt*F(u^n)
    
    This is a second-order explicit method that requires two previous states.
    GPU-aware: automatically detects and uses CuPy if input is a GPU array.
    
    Parameters:
    -----------
    psi_prev : array
        Solution at time step n-1 (NumPy or CuPy array)
    psi_now : array
        Solution at time step n (NumPy or CuPy array)
    dt : float
        Time step
    rhs_func : callable
        Function that computes RHS: rhs_func(psi, *args) -> dpsi/dt
    *args : additional arguments
        Additional arguments to pass to rhs_func
        
    Returns:
    --------
    array : Updated solution at time step n+1 (2nd order accurate)
    """
    # Compute RHS at current time step
    rhs_now = rhs_func(psi_now, *args)
    
    # Leapfrog formula: u^{n+1} = u^{n-1} + 2*dt*F(u^n)
    psi_next = psi_prev + 2.0 * dt * rhs_now
    
    return psi_next


# ============================================================
#  BDF method using scipy.integrate.solve_ivp
# ============================================================
def bdf_step(psi_now, t_now, dt, rhs_func, jacobian_func=None, *args, 
             rtol=1e-6, atol=1e-8):
    """
    BDF method using scipy.integrate.solve_ivp.
    This uses scipy's adaptive BDF solver (order 1-5) for a single time step.
    
    Parameters:
    -----------
    psi_now : array
        Solution at time step n (NumPy array, GPU not supported by solve_ivp)
    t_now : float
        Current time
    dt : float
        Time step
    rhs_func : callable
        Function that computes RHS: rhs_func(psi, *args) -> dpsi/dt
        Note: solve_ivp expects fun(t, y), so we wrap rhs_func
    jacobian_func : callable, optional
        Function that computes analytical Jacobian: jacobian_func(t, y, *args) -> J
        For solve_ivp, this should be jac(t, y) -> J
    *args : additional arguments
        Additional arguments to pass to rhs_func and jacobian_func
    rtol : float
        Relative tolerance for solve_ivp
    atol : float
        Absolute tolerance for solve_ivp
        
    Returns:
    --------
    array : Updated solution at time step n+1
    """
    if not HAS_SCIPY_INTEGRATE:
        raise ImportError("scipy.integrate.solve_ivp is required for BDF method")
    
    # Convert to NumPy if needed (solve_ivp doesn't support GPU arrays)
    if hasattr(psi_now, 'get'):  # CuPy array
        psi_now = psi_now.get()
    
    # Wrap rhs_func to match solve_ivp's interface: fun(t, y)
    def fun(t, y):
        # args are captured from closure
        return rhs_func(y, *args)
    
    # Wrap jacobian_func if provided
    jac = None
    if jacobian_func is not None:
        def jac_wrapper(t, y):
            # jacobian_func might expect (y, *args) or (t, y, *args)
            try:
                return jacobian_func(t, y, *args)
            except TypeError:
                # Fallback: try without t
                return jacobian_func(y, *args)
        jac = jac_wrapper
    
    # Solve from t_now to t_now + dt
    t_span = (t_now, t_now + dt)
    result = solve_ivp(fun, t_span, psi_now, method='BDF', 
                       jac=jac, rtol=rtol, atol=atol,
                       dense_output=False)
    
    if not result.success:
        raise RuntimeError(f"solve_ivp failed: {result.message}")
    
    # Return the solution at the final time
    return result.y[:, -1]


# ============================================================
#  Time Stepper State (for multi-step methods)
# ============================================================
class TimeStepperState:
    """
    State container for time steppers.
    Tracks previous states needed for multi-step methods like BDF2 and leapfrog.
    Also tracks current time for progress monitoring.
    Caches precomputed coefficients for efficiency.
    Supports automatic progress tracking with tqdm.
    """
    def __init__(self, psi_init, t_init=0.0, dt=None, method='rk4', 
                 t_final=None, show_progress=True):
        """
        Initialize state with initial condition.
        
        Parameters:
        -----------
        psi_init : array
            Initial solution state
        t_init : float
            Initial time (default: 0.0)
        dt : float, optional
            Time step (used to precompute BDF2 coefficients)
        method : str
            Time stepping method: 'rk4', 'rk45', 'rk23', 'leapfrog', 'bdf2', or 'bdf'
        t_final : float, optional
            Final time for progress tracking. If provided and show_progress=True,
            automatically creates and manages a progress bar.
        show_progress : bool
            If True and t_final is provided, automatically show progress bar
        """
        self.method = method.lower()
        
        # For BDF2 and leapfrog: need 1 previous state
        self.psi_prev = None
        self.psi_now = psi_init.copy()
        self.initialized = False
        self.t_current = t_init
        self.t_prev = t_init
        
        # BDF2 coefficients no longer needed (using solve_ivp)
        # Keep for backward compatibility but not used
        self.bdf2_coeffs = None
        
        # Progress tracking
        self.t_final = t_final
        self.progress_bar = None
        if show_progress and t_final is not None and HAS_TQDM:
            self.progress_bar = tqdm(total=t_final, desc=f"Integrating ({method.upper()})", 
                                    unit="time",
                                    bar_format='{l_bar}{bar}| {n:.3f}/{total:.3f} [{elapsed}<{remaining}]')
            self.progress_bar.n = t_init  # Initialize to starting time
            self.progress_bar.refresh()
    
    def close_progress(self):
        """Close the progress bar if it exists."""
        if self.progress_bar is not None:
            # Ensure progress bar shows 100% completion
            if self.t_final is not None:
                self.progress_bar.n = self.t_final
                self.progress_bar.refresh()
            self.progress_bar.close()
            self.progress_bar = None
    
    def __enter__(self):
        """Context manager entry - returns self."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes progress bar."""
        self.close_progress()
        return False  # Don't suppress exceptions
    
    def update(self, psi_next, t_next=None):
        """
        Update state after a time step.
        
        Parameters:
        -----------
        psi_next : array
            Solution at next time step
        t_next : float, optional
            Time at next step. If None, increments by dt from last update.
        """
        self.psi_prev = self.psi_now.copy()
        self.psi_now = psi_next.copy()
        self.t_prev = self.t_current
        if t_next is not None:
            self.t_current = t_next
        self.initialized = True
    
    def get_current(self):
        """Get current solution state."""
        return self.psi_now.copy()
    
    def get_previous(self):
        """Get previous solution state (for BDF2)."""
        if not self.initialized:
            return self.psi_now.copy()  # Use current if not yet initialized
        return self.psi_prev.copy()
    
    def get_current_time(self):
        """Get current time."""
        return self.t_current
    
    def increment_time(self, dt):
        """Increment current time by dt."""
        self.t_current += dt


# ============================================================
#  Uniform Time Stepper Interface
# ============================================================
def time_step(state: TimeStepperState, dt: float, rhs_func: Callable, 
              method: str = 'rk4', jacobian_func: Optional[Callable] = None,
              progress_bar: Optional[Any] = None, t_final: Optional[float] = None,
              *args, **kwargs) -> np.ndarray:
    """
    Uniform interface for time stepping methods.
    
    Parameters:
    -----------
    state : TimeStepperState
        State object containing current and previous solution states
    dt : float
        Time step
    rhs_func : callable
        Function that computes RHS: rhs_func(psi, *args) -> dpsi/dt
    method : str
        Time stepping method: 'rk4', 'rk45', 'rk23', 'leapfrog', or 'bdf2'
    jacobian_func : callable, optional
        Function that computes analytical Jacobian (required for BDF2)
        jacobian_func(psi, *args) -> J
    progress_bar : tqdm progress bar, optional
        Progress bar to update (tracks time progress).
        If None, uses the progress bar from state if available (when state was
        created with t_final and show_progress=True).
    t_final : float, optional
        Final time (needed for progress bar).
        If None, uses t_final from state if available.
    *args : additional arguments
        Additional arguments to pass to rhs_func and jacobian_func
    **kwargs : additional keyword arguments
        Additional options (e.g., max_iter, tol for BDF2)
        
    Returns:
    --------
    array : Updated solution at next time step
    """
    method = method.lower()
    
    # Use state's progress bar if available, otherwise use provided one
    effective_progress_bar = state.progress_bar if state.progress_bar is not None else progress_bar
    effective_t_final = state.t_final if state.t_final is not None else t_final
    
    if method == 'rk4':
        psi_next = rk4_step(state.get_current(), dt, rhs_func, *args)
        t_next = state.get_current_time() + dt
        state.update(psi_next, t_next)
        
        # Update progress bar if available
        if effective_progress_bar is not None and effective_t_final is not None:
            current_time = state.get_current_time()
            # Update to current time (clamp to t_final)
            effective_progress_bar.n = min(current_time, effective_t_final)
            effective_progress_bar.refresh()
        
        return psi_next
    
    elif method == 'rk45':
        psi_next = rk45_step(state.get_current(), dt, rhs_func, *args)
        t_next = state.get_current_time() + dt
        state.update(psi_next, t_next)
        
        # Update progress bar if available
        if effective_progress_bar is not None and effective_t_final is not None:
            current_time = state.get_current_time()
            effective_progress_bar.n = min(current_time, effective_t_final)
            effective_progress_bar.refresh()
        
        return psi_next
    
    elif method == 'rk23':
        psi_next = rk23_step(state.get_current(), dt, rhs_func, *args)
        t_next = state.get_current_time() + dt
        state.update(psi_next, t_next)
        
        # Update progress bar if available
        if effective_progress_bar is not None and effective_t_final is not None:
            current_time = state.get_current_time()
            effective_progress_bar.n = min(current_time, effective_t_final)
            effective_progress_bar.refresh()
        
        return psi_next
    
    elif method == 'leapfrog':
        if not state.initialized:
            # First step: use RK4 to initialize (need two states for leapfrog)
            psi_next = rk4_step(state.get_current(), dt, rhs_func, *args)
            t_next = state.get_current_time() + dt
            state.update(psi_next, t_next)
            
            # Update progress bar if available
            if effective_progress_bar is not None and effective_t_final is not None:
                current_time = state.get_current_time()
                effective_progress_bar.n = min(current_time, effective_t_final)
                effective_progress_bar.refresh()
            
            return psi_next
        else:
            # Subsequent steps: use leapfrog
            psi_prev = state.get_previous()
            psi_now = state.get_current()
            psi_next = leapfrog_step(psi_prev, psi_now, dt, rhs_func, *args)
            t_next = state.get_current_time() + dt
            state.update(psi_next, t_next)
            
            # Update progress bar if available
            if effective_progress_bar is not None and effective_t_final is not None:
                current_time = state.get_current_time()
                effective_progress_bar.n = min(current_time, effective_t_final)
                effective_progress_bar.refresh()
            
            return psi_next
    
    elif method == 'bdf2' or method == 'bdf':
        # Use scipy.integrate.solve_ivp's BDF method
        if not HAS_SCIPY_INTEGRATE:
            raise ImportError("scipy.integrate.solve_ivp is required for BDF method. "
                            "Install scipy or use a different method.")
        
        psi_now = state.get_current()
        t_now = state.get_current_time()
        
        # Get tolerance parameters
        rtol = kwargs.get('rtol', 1e-6)
        atol = kwargs.get('atol', 1e-8)
        
        # Use BDF step with solve_ivp
        psi_next = bdf_step(psi_now, t_now, dt, rhs_func, 
                           jacobian_func, *args,
                           rtol=rtol, atol=atol)
        t_next = t_now + dt
        state.update(psi_next, t_next)
        
        # Update progress bar if available
        if effective_progress_bar is not None and effective_t_final is not None:
            current_time = state.get_current_time()
            effective_progress_bar.n = min(current_time, effective_t_final)
            effective_progress_bar.refresh()
        
        return psi_next
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'rk4', 'rk45', 'rk23', 'leapfrog', 'bdf2', or 'bdf'.")


# Convenience function for backward compatibility
def create_time_stepper(psi_init, method='rk4'):
    """
    Create a time stepper state and return a convenient stepping function.
    
    Parameters:
    -----------
    psi_init : array
        Initial solution state
    method : str
        Time stepping method: 'rk4', 'rk45', 'rk23', 'leapfrog', or 'bdf2'
        
    Returns:
    --------
    stepper_func : callable
        Function with signature: stepper_func(dt, rhs_func, jacobian_func=None, *args, **kwargs)
    """
    state = TimeStepperState(psi_init)
    
    def stepper(dt, rhs_func, jacobian_func=None, *args, **kwargs):
        return time_step(state, dt, rhs_func, method, jacobian_func, *args, **kwargs)
    
    return stepper, state
