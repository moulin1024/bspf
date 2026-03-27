"""Primary 1D BSPF operator.

For now this wraps the legacy implementation so the package can present a
stable public API before the internals are fully migrated.
"""

from bspf1d import bspf1d as _LegacyBSPF1D


class BSPF1D(_LegacyBSPF1D):
    """Compatibility wrapper around the legacy 1D BSPF implementation."""


bspf1d = BSPF1D

__all__ = ["BSPF1D", "bspf1d"]
