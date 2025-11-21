"""
Unit test for heat equation time-stepping accuracy.

Tests that RK45, RK23, and BDF2 achieve required L2 accuracy with the configuration
from examples/heat_eq/zero_flux_1d.py.
"""

import numpy as np
import numpy.typing as npt
import pytest

from bspf.utils import TimeStepperState, time_step
from bspf import bspf1d
from bspf.solver import (
    create_heat_rhs_1d,
    build_laplacian_matrix_1d,
    create_heat_jacobian_1d
)

Array = npt.NDArray[np.float64]


def heat_neumann_cosx(x, t, kappa=1.0):
    """
    Solution of u_t = kappa*u_xx on [0, 2π] with Neumann BCs
    and initial condition u(x,0) = cos(x).
    """
    return np.cos(x) * np.exp(-kappa * t)


def test_heat_eq_rk45_accuracy():
    """
    Test that RK45 achieves L2 error below 1e-10.
    
    Uses the same configuration as examples/heat_eq/zero_flux_1d.py:
    - nu = 1e-1 (diffusivity)
    - L_domain = 3*pi
    - nx = 101
    - T = 10
    - dt = 0.01
    - degree = 9
    - Initial condition: u(x,0) = cos(x)
    """
    # Parameters from zero_flux_1d.py
    nu = 1e-1
    L_domain = 3 * np.pi
    nx = 101
    T = 10
    dt = 0.01
    nt = int(T / dt) + 1
    
    # Grid and operator
    x = np.linspace(0.0, L_domain, nx)
    bf = bspf1d.from_grid(degree=9, x=x)
    
    # Initial condition
    u0 = np.cos(x)
    
    # Create RHS function
    rhs_func = create_heat_rhs_1d(bf, nu, neumann_bc=(0.0, 0.0))
    
    # Time integration with RK45
    state_rk45 = TimeStepperState(u0.copy(), t_init=0.0, dt=dt, method='rk45')
    U_rk45 = np.empty((nt, nx), dtype=np.float64)
    U_rk45[0] = u0.copy()
    times_rk45 = np.zeros(nt)
    times_rk45[0] = 0.0
    
    for step in range(1, nt):
        u_next = time_step(state_rk45, dt, rhs_func, method='rk45')
        U_rk45[step] = state_rk45.get_current()
        times_rk45[step] = state_rk45.get_current_time()
    
    # Compute exact solution and error
    u_exact_rk45 = heat_neumann_cosx(x[None, :], times_rk45[:, None], kappa=nu)
    error_rk45 = np.abs(U_rk45 - u_exact_rk45)
    l2_error_rk45 = np.sqrt(np.mean(error_rk45**2))
    
    # Assert accuracy requirement
    assert l2_error_rk45 < 1e-10, (
        f"RK45 L2 error {l2_error_rk45:.2e} does not meet requirement < 1e-10 "
        f"at T={T}, dt={dt}, nx={nx}, nu={nu}"
    )


def test_heat_eq_rk23_accuracy():
    """
    Test that RK23 achieves L2 error below 1e-10.
    
    Same configuration as test_heat_eq_rk45_accuracy.
    """
    # Parameters from zero_flux_1d.py
    nu = 1e-1
    L_domain = 3 * np.pi
    nx = 101
    T = 10
    dt = 0.01
    nt = int(T / dt) + 1
    
    # Grid and operator
    x = np.linspace(0.0, L_domain, nx)
    bf = bspf1d.from_grid(degree=9, x=x)
    
    # Initial condition
    u0 = np.cos(x)
    
    # Create RHS function
    rhs_func = create_heat_rhs_1d(bf, nu, neumann_bc=(0.0, 0.0))
    
    # Time integration with RK23
    state_rk23 = TimeStepperState(u0.copy(), t_init=0.0, dt=dt, method='rk23')
    U_rk23 = np.empty((nt, nx), dtype=np.float64)
    U_rk23[0] = u0.copy()
    times_rk23 = np.zeros(nt)
    times_rk23[0] = 0.0
    
    for step in range(1, nt):
        u_next = time_step(state_rk23, dt, rhs_func, method='rk23')
        U_rk23[step] = state_rk23.get_current()
        times_rk23[step] = state_rk23.get_current_time()
    
    # Compute exact solution and error
    u_exact_rk23 = heat_neumann_cosx(x[None, :], times_rk23[:, None], kappa=nu)
    error_rk23 = np.abs(U_rk23 - u_exact_rk23)
    l2_error_rk23 = np.sqrt(np.mean(error_rk23**2))
    
    # Assert accuracy requirement
    assert l2_error_rk23 < 1e-10, (
        f"RK23 L2 error {l2_error_rk23:.2e} does not meet requirement < 1e-10 "
        f"at T={T}, dt={dt}, nx={nx}, nu={nu}"
    )


def test_heat_eq_bdf2_accuracy():
    """
    Test that BDF2 achieves L2 error below 1e-7.
    
    Same configuration as test_heat_eq_rk45_accuracy.
    """
    # Parameters from zero_flux_1d.py
    nu = 1e-1
    L_domain = 3 * np.pi
    nx = 101
    T = 10
    dt = 0.01
    nt = int(T / dt) + 1
    
    # Grid and operator
    x = np.linspace(0.0, L_domain, nx)
    bf = bspf1d.from_grid(degree=9, x=x)
    
    # Initial condition
    u0 = np.cos(x)
    
    # Build Laplacian matrix for BDF2
    L = build_laplacian_matrix_1d(bf, neumann_bc=(0.0, 0.0))
    
    # Create RHS and Jacobian functions
    rhs_func = create_heat_rhs_1d(bf, nu, neumann_bc=(0.0, 0.0))
    jacobian_func = create_heat_jacobian_1d(L, nu)
    
    # Time integration with BDF2
    state_bdf2 = TimeStepperState(u0.copy(), t_init=0.0, dt=dt, method='bdf2')
    U_bdf2 = np.empty((nt, nx), dtype=np.float64)
    U_bdf2[0] = u0.copy()
    times_bdf2 = np.zeros(nt)
    times_bdf2[0] = 0.0
    
    for step in range(1, nt):
        u_next = time_step(state_bdf2, dt, rhs_func, method='bdf2', 
                           jacobian_func=jacobian_func)
        U_bdf2[step] = state_bdf2.get_current()
        times_bdf2[step] = state_bdf2.get_current_time()
    
    # Compute exact solution and error
    u_exact_bdf2 = heat_neumann_cosx(x[None, :], times_bdf2[:, None], kappa=nu)
    error_bdf2 = np.abs(U_bdf2 - u_exact_bdf2)
    l2_error_bdf2 = np.sqrt(np.mean(error_bdf2**2))
    
    # Assert accuracy requirement
    assert l2_error_bdf2 < 1e-7, (
        f"BDF2 L2 error {l2_error_bdf2:.2e} does not meet requirement < 1e-7 "
        f"at T={T}, dt={dt}, nx={nx}, nu={nu}"
    )

