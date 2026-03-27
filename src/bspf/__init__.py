"""Public package API for BSPF.

The numerical implementation is still delegated to the legacy root-level
``bspf1d.py`` module while the package is migrated into smaller modules.
"""

from .grid import Grid1D
from .operators import BSPF1D, PiecewiseBSPF1D, bspf1d

__all__ = ["BSPF1D", "Grid1D", "PiecewiseBSPF1D", "bspf1d"]
