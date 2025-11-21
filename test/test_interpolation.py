"""
Unit test for BSPF1D interpolation functionality.

Tests the high-order interpolation method that doubles resolution by inserting
midpoints between existing grid points.
"""

import numpy as np
import pytest

from bspf import bspf1d


def test_interpolation_size():
    """
    Test that interpolation correctly doubles the resolution.
    
    Verifies that N_new = 2*N_old - 1 as specified.
    """
    N = 64
    degree = 5
    
    # Create grid and model
    x = np.linspace(0, 2*np.pi, N)
    model = bspf1d.from_grid(degree=degree, x=x)
    
    # Test function
    f = np.sin(x)
    
    # Test interpolation
    x_new, f_new = model.interpolate(f)
    
    # Verify sizes
    assert len(f_new) == 2*N - 1, (
        f"Interpolated size {len(f_new)} does not match expected {2*N - 1}"
    )
    assert len(x_new) == 2*N - 1, (
        f"New grid size {len(x_new)} does not match expected {2*N - 1}"
    )


def test_interpolation_preserves_original_points():
    """
    Test that original grid points are preserved in the interpolated result.
    
    The original points should appear at even indices in the new grid,
    and their values should match the input (within numerical precision).
    """
    N = 64
    degree = 5
    
    # Create grid and model
    x = np.linspace(0, 2*np.pi, N)
    model = bspf1d.from_grid(degree=degree, x=x)
    
    # Test function
    f = np.sin(x)
    
    # Test interpolation
    x_new, f_new = model.interpolate(f)
    
    # Verify original points are at even indices
    x_original = x_new[::2]
    f_original = f_new[::2]
    
    # Check grid points match
    assert np.allclose(x_original, x, rtol=1e-13, atol=1e-13), (
        "Original grid points not preserved correctly"
    )
    
    # Check function values at original points
    # Allow some tolerance due to spline fitting
    error_at_original = np.max(np.abs(f_original - f))
    assert error_at_original < 1e-5, (
        f"Function values at original points not preserved. "
        f"Max error: {error_at_original:.2e}"
    )


def test_interpolation_midpoint_accuracy():
    """
    Test that interpolation achieves high-order accuracy at midpoints.
    
    For smooth functions, the interpolated values at midpoints should
    be very close to the true function values.
    """
    N = 64
    degree = 5
    
    # Create grid and model
    x = np.linspace(0, 2*np.pi, N)
    model = bspf1d.from_grid(degree=degree, x=x)
    
    # Test function (smooth)
    f = np.sin(x)
    
    # Test interpolation
    x_new, f_new = model.interpolate(f)
    
    # Get midpoints (odd indices)
    x_midpoints = x_new[1::2]
    f_interp_midpoints = f_new[1::2]
    
    # True function values at midpoints
    f_true_midpoints = np.sin(x_midpoints)
    
    # Check interpolation accuracy
    error_at_midpoints = np.max(np.abs(f_interp_midpoints - f_true_midpoints))
    assert error_at_midpoints < 1e-4, (
        f"Interpolation accuracy at midpoints insufficient. "
        f"Max error: {error_at_midpoints:.2e}"
    )


def test_interpolation_grid_spacing():
    """
    Test that the new grid has correct spacing.
    
    The new grid spacing should be half of the original spacing.
    """
    N = 64
    degree = 5
    
    # Create grid and model
    x = np.linspace(0, 2*np.pi, N)
    model = bspf1d.from_grid(degree=degree, x=x)
    
    # Test function
    f = np.sin(x)
    
    # Test interpolation
    x_new, f_new = model.interpolate(f)
    
    # Check grid spacing
    dx_old = x[1] - x[0]
    dx_new = x_new[1] - x_new[0]
    
    # New spacing should be half of old spacing
    assert np.isclose(dx_new, dx_old / 2, rtol=1e-13, atol=1e-13), (
        f"Grid spacing not halved correctly. "
        f"Old: {dx_old:.6f}, New: {dx_new:.6f}, Expected: {dx_old/2:.6f}"
    )
    
    # Verify uniform spacing in new grid
    dx_new_all = np.diff(x_new)
    assert np.allclose(dx_new_all, dx_new, rtol=1e-13, atol=1e-13), (
        "New grid is not uniformly spaced"
    )


def test_interpolation_different_degrees():
    """
    Test interpolation with different B-spline degrees.
    """
    N = 32
    degrees = [3, 5, 7]
    
    for degree in degrees:
        # Create grid and model
        x = np.linspace(0, 2*np.pi, N)
        model = bspf1d.from_grid(degree=degree, x=x)
        
        # Test function
        f = np.sin(x)
        
        # Test interpolation
        x_new, f_new = model.interpolate(f)
        
        # Verify size
        assert len(f_new) == 2*N - 1, (
            f"Size mismatch for degree {degree}"
        )
        
        # Verify original points preserved
        f_original = f_new[::2]
        error = np.max(np.abs(f_original - f))
        # Lower degrees may have slightly larger errors due to fewer basis functions
        tolerance = 1e-3 if degree <= 3 else 1e-4
        assert error < tolerance, (
            f"Original points not preserved for degree {degree}. "
            f"Error: {error:.2e}"
        )


def test_interpolation_different_functions():
    """
    Test interpolation with different smooth functions.
    """
    N = 64
    degree = 5
    
    # Create grid and model
    x = np.linspace(0, 2*np.pi, N)
    model = bspf1d.from_grid(degree=degree, x=x)
    
    # Test functions
    test_functions = [
        ("sin", lambda x: np.sin(x)),
        ("cos", lambda x: np.cos(x)),
        ("exp", lambda x: np.exp(-x/10)),
        ("polynomial", lambda x: x**3 - 2*x**2 + x),
    ]
    
    for name, func in test_functions:
        f = func(x)
        
        # Test interpolation
        x_new, f_new = model.interpolate(f)
        
        # Verify size
        assert len(f_new) == 2*N - 1, (
            f"Size mismatch for function {name}"
        )
        
        # Verify original points preserved
        f_original = f_new[::2]
        error = np.max(np.abs(f_original - f))
        assert error < 1e-4, (
            f"Original points not preserved for function {name}. "
            f"Error: {error:.2e}"
        )
        
        # Verify midpoint accuracy
        x_midpoints = x_new[1::2]
        f_interp_midpoints = f_new[1::2]
        f_true_midpoints = func(x_midpoints)
        error_midpoints = np.max(np.abs(f_interp_midpoints - f_true_midpoints))
        assert error_midpoints < 1e-3, (
            f"Midpoint accuracy insufficient for function {name}. "
            f"Error: {error_midpoints:.2e}"
        )


def test_interpolation_regularization():
    """
    Test that interpolation works with regularization parameter.
    
    Uses lambda = 0 (default) which is the most common use case.
    """
    N = 64
    degree = 5
    
    # Create grid and model
    x = np.linspace(0, 2*np.pi, N)
    model = bspf1d.from_grid(degree=degree, x=x)
    
    # Test function
    f = np.sin(x)
    
    # Test with lambda = 0 (default)
    lam = 0.0
    x_new, f_new = model.interpolate(f, lam=lam)
    
    # Verify size
    assert len(f_new) == 2*N - 1, (
        f"Size mismatch for lambda={lam}"
    )
    
    # Verify original points preserved
    f_original = f_new[::2]
    error = np.max(np.abs(f_original - f))
    assert error < 1e-4, (
        f"Original points not preserved for lambda={lam}. "
        f"Error: {error:.2e}"
    )


def test_interpolation_input_validation():
    """
    Test that interpolation validates input correctly.
    """
    N = 64
    degree = 5
    
    # Create grid and model
    x = np.linspace(0, 2*np.pi, N)
    model = bspf1d.from_grid(degree=degree, x=x)
    
    # Test with wrong size input
    f_wrong = np.sin(np.linspace(0, 2*np.pi, N+10))
    
    with pytest.raises(ValueError, match="must match grid size"):
        model.interpolate(f_wrong)


def test_interpolation_small_grid():
    """
    Test interpolation with small grid sizes.
    """
    degree = 5
    
    for N in [8, 16, 32]:
        # Create grid and model
        x = np.linspace(0, 2*np.pi, N)
        model = bspf1d.from_grid(degree=degree, x=x)
        
        # Test function
        f = np.sin(x)
        
        # Test interpolation
        x_new, f_new = model.interpolate(f)
        
        # Verify size
        assert len(f_new) == 2*N - 1, (
            f"Size mismatch for N={N}"
        )
        
        # Verify original points preserved
        f_original = f_new[::2]
        error = np.max(np.abs(f_original - f))
        assert error < 1e-3, (
            f"Original points not preserved for N={N}. "
            f"Error: {error:.2e}"
        )

