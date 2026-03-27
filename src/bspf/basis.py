"""! @file basis.py
@brief B-spline basis helpers.

This module currently forwards to the legacy implementation while the package
is being split into focused modules.
"""

# Re-export the legacy class to keep the public API stable during migration.
from bspf1d import BSplineBasis1D

__all__ = ["BSplineBasis1D"]
