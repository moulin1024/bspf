"""! @file correction.py
@brief Residual correction strategies.

This module currently forwards to the legacy implementation while the package
is being split into focused modules.
"""

# Re-export the legacy class to keep the public API stable during migration.
from bspf1d import ResidualCorrection

__all__ = ["ResidualCorrection"]
