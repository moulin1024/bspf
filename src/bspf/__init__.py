"""! @file __init__.py
@brief Public package exports for the BSPF package.

This module exposes the stable user-facing API while the internal codebase is
being migrated away from the legacy monolithic implementation.
"""

# Re-export the canonical grid type and the current operator wrappers so users
# can import from ``bspf`` directly instead of depending on file layout.
from .grid import Grid1D
from .operators import BSPF1D, PiecewiseBSPF1D, bspf1d

__all__ = ["BSPF1D", "Grid1D", "PiecewiseBSPF1D", "bspf1d"]
