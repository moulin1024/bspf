"""Piecewise operator wrapper for discontinuous signals."""

from bspf1d import PiecewiseBSPF1D as _LegacyPiecewiseBSPF1D


class PiecewiseBSPF1D(_LegacyPiecewiseBSPF1D):
    """Compatibility wrapper around the legacy piecewise implementation."""


__all__ = ["PiecewiseBSPF1D"]
