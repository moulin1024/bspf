"""
Unit test for BSPF1D accuracy.

Tests that BSPF1D achieves accuracy below 5e-09 at N=4300 with the configuration
from examples/basic/diff_1d.py.
"""

import numpy as np
import sympy as sp
import pytest

from bspf import bspf1d


def define_test_function():
    """
    Define the test function and its analytical derivative using SymPy.
    
    Returns
    -------
    func : callable
        Function f(x) as a NumPy-compatible function
    func_deriv : callable
        Derivative f'(x) as a NumPy-compatible function
    """
    t = sp.Symbol('t')
    
    # Test function: sin(t / (1.01 + cos(t)))
    # This is a smooth, non-periodic function with varying frequency
    f_sym = sp.sin(t / (1.01 + sp.cos(t)))
    df_sym = sp.diff(f_sym, t)
    
    # Convert to NumPy functions
    func = sp.lambdify(t, f_sym, modules='numpy')
    func_deriv = sp.lambdify(t, df_sym, modules='numpy')
    
    return func, func_deriv


def test_bspf1d_accuracy_at_n4300():
    """
    Test that BSPF1D achieves accuracy below 5e-09 at N=4300.
    
    Uses the same configuration as examples/basic/diff_1d.py:
    - DEGREE = 5
    - NUM_BOUNDARY_POINTS = DEGREE
    - N_BASIS = 2 * DEGREE
    - REG_PARAM = 1e-3
    - DOMAIN = [0, 2*pi]
    - NUM_POINTS = 4300
    - CLUSTERING_FACTOR = 2.0
    - USE_CLUSTERING = True
    """
    # Parameters from diff_1d.py
    DEGREE = 5
    NUM_BOUNDARY_POINTS = DEGREE
    N_BASIS = 2 * DEGREE
    REG_PARAM = 1e-3
    DOMAIN = [0, 2*np.pi]
    NUM_POINTS = 4300
    CLUSTERING_FACTOR = 2.0
    USE_CLUSTERING = True
    
    # Define test function
    test_func, test_func_deriv = define_test_function()
    
    # Create grid
    x = np.linspace(DOMAIN[0], DOMAIN[1], NUM_POINTS, endpoint=True)
    
    # Initialize BSPF model
    model = bspf1d.from_grid(
        degree=DEGREE,
        x=x,
        domain=tuple(DOMAIN),
        order=DEGREE,
        n_basis=N_BASIS,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        use_clustering=USE_CLUSTERING,
        clustering_factor=CLUSTERING_FACTOR
    )
    
    # Compute function values and exact derivative
    y = test_func(x)
    y_deriv_exact = test_func_deriv(x)
    
    # Compute derivative using BSPF
    y_deriv_bspf, _ = model.differentiate(y, k=1, lam=REG_PARAM)
    
    # Compute maximum error
    max_error = np.max(np.abs(y_deriv_bspf - y_deriv_exact))
    
    # Assert accuracy requirement
    assert max_error < 5e-09, (
        f"BSPF1D accuracy {max_error:.2e} does not meet requirement < 5e-09 "
        f"at N={NUM_POINTS} with degree={DEGREE}, reg_param={REG_PARAM}"
    )