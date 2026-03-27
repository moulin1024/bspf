"""Backend selection and explicit host/device conversion helpers.

This module currently re-exports the legacy implementation while the package
is being split into focused modules.
"""

from bspf1d import _Backend

__all__ = ["_Backend"]
