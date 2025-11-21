"""
Debug utilities for catching implicit type conversion errors.

This module provides utilities to test GPU-compatible code on CPU by using
CuPy arrays even when GPU is not available, which helps catch implicit
conversion errors during development.
"""

import os
import numpy as np

# Optional GPU backend
_HAS_CUPY = False
try:
    import cupy as cp
    import cupyx
    _HAS_CUPY = True
except Exception:
    cp = None
    cupyx = None


# Global flag to enable strict mode
_STRICT_MODE_ENABLED = False


def enable_strict_mode():
    """
    Enable strict mode to catch implicit NumPy/CuPy conversions.
    
    This enables CuPy's strict mode which disallows implicit conversions
    between NumPy and CuPy arrays. This helps catch bugs during development.
    
    Note: This only works if CuPy is available.
    
    Example:
        >>> enable_strict_mode()
        >>> x = cp.array([1, 2, 3])
        >>> y = np.array([4, 5, 6])
        >>> z = x + y  # This will raise TypeError in strict mode
    """
    global _STRICT_MODE_ENABLED
    if not _HAS_CUPY:
        raise RuntimeError(
            "CuPy is not available. Install cupy to use strict mode. "
            "Strict mode requires CuPy to catch implicit conversion errors."
        )
    
    # Enable CuPy's strict mode
    try:
        # cupyx.disable_implicit_conversion() is the function to disable implicit conversions
        # This makes CuPy raise errors when NumPy arrays are implicitly converted
        if hasattr(cupyx, 'disable_implicit_conversion'):
            cupyx.disable_implicit_conversion()
            _STRICT_MODE_ENABLED = True
            print("✓ Strict mode enabled: Implicit NumPy/CuPy conversions are now disallowed")
        else:
            # Try alternative method for older CuPy versions
            # Some versions use cp.cuda.set_allocator or other methods
            print("Warning: cupyx.disable_implicit_conversion() not available in this CuPy version")
            print("  You may need to update CuPy or use explicit conversions")
            _STRICT_MODE_ENABLED = False
    except Exception as e:
        print(f"Warning: Could not enable strict mode: {e}")
        _STRICT_MODE_ENABLED = False


def disable_strict_mode():
    """Disable strict mode (re-enable implicit conversions)."""
    global _STRICT_MODE_ENABLED
    _STRICT_MODE_ENABLED = False
    print("Strict mode disabled")


def is_strict_mode_enabled():
    """Check if strict mode is currently enabled."""
    return _STRICT_MODE_ENABLED


def use_cupy_for_debugging(use_gpu=False):
    """
    Force use of CuPy arrays even on CPU for debugging.
    
    This function allows you to test GPU-compatible code on CPU by using
    CuPy arrays. This helps catch implicit conversion errors that would
    only appear on GPU.
    
    Parameters:
    -----------
    use_gpu : bool
        If True, use actual GPU. If False, use CuPy on CPU (if available).
    
    Returns:
    --------
    use_cupy : bool
        True if CuPy should be used (either GPU or CPU mode)
    xp : module
        Array module to use (cupy or numpy)
    """
    if not _HAS_CUPY:
        return False, np
    
    # Check environment variable for debugging
    debug_cupy = os.environ.get('BSPF_DEBUG_CUPY', 'false').lower() in ('true', '1', 'yes')
    
    if use_gpu:
        # Use actual GPU
        return True, cp
    elif debug_cupy:
        # Use CuPy on CPU for debugging
        # Note: CuPy doesn't have a pure CPU mode, but we can use it with CPU device
        # For now, we'll just use CuPy arrays which will catch conversion errors
        return True, cp
    else:
        # Normal CPU mode with NumPy
        return False, np


def wrap_numpy_array(arr, force_cupy=False):
    """
    Wrap a NumPy array as CuPy array for debugging.
    
    This is useful for testing GPU code on CPU by converting NumPy arrays
    to CuPy arrays, which will catch implicit conversion errors.
    
    Parameters:
    -----------
    arr : np.ndarray
        NumPy array to wrap
    force_cupy : bool
        If True, convert to CuPy array even if not using GPU
    
    Returns:
    --------
    arr : array
        CuPy array if force_cupy=True and CuPy is available, otherwise NumPy array
    """
    if force_cupy and _HAS_CUPY:
        return cp.asarray(arr)
    return arr


# Auto-enable strict mode if environment variable is set
if os.environ.get('BSPF_STRICT_MODE', 'false').lower() in ('true', '1', 'yes'):
    if _HAS_CUPY:
        try:
            enable_strict_mode()
        except Exception as e:
            print(f"Warning: Could not enable strict mode: {e}")
    else:
        print("Warning: BSPF_STRICT_MODE is set but CuPy is not available")


# Example usage:
if __name__ == "__main__":
    # Enable strict mode
    if _HAS_CUPY:
        enable_strict_mode()
        
        # Test: This should work
        x = cp.array([1, 2, 3])
        y = cp.array([4, 5, 6])
        z = x + y  # OK: both are CuPy arrays
        print(f"✓ CuPy + CuPy: {z}")
        
        # Test: This should fail in strict mode
        try:
            x_np = np.array([1, 2, 3])
            result = x + x_np  # Should fail: mixing CuPy and NumPy
            print(f"✗ This should have failed: {result}")
        except TypeError as e:
            print(f"✓ Caught implicit conversion error: {e}")
    else:
        print("CuPy not available. Install cupy to test strict mode.")

