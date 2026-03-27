"""! @file ops/interpolation.py
@brief Package-owned interpolation and spline-fit workflows for BSPF1D.
"""

from __future__ import annotations

from typing import Optional, Tuple

from bspf1d import bspf1d as _LegacyBSPF1D

from ..types import Array


def enforced_zero_flux(self, f: Array) -> Tuple[float, float]:
    """! @brief Repair endpoint values to satisfy a zero-flux condition.

    @param self ``BSPF1D`` instance.
    @param f Input samples on the operator grid.
    @return Tuple ``(f_left_corrected, f_right_corrected)``.
    """
    return _LegacyBSPF1D.enforced_zero_flux(self, f)


def interpolate(self, f: Array, lam: float = 0.0, use_fft: bool = False):
    """! @brief Interpolate the signal onto a grid with inserted midpoints.

    @param self ``BSPF1D`` instance.
    @param f Input samples on the operator grid.
    @param lam Tikhonov regularization parameter.
    @param use_fft Whether to use the FFT-only interpolation path.
    @return Tuple ``(x_new, f_new)``.
    """
    return _LegacyBSPF1D.interpolate(self, f, lam=lam, use_fft=use_fft)


def fit_spline(
    self,
    f: Array,
    lam: float = 0.0,
    neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None,
):
    """! @brief Fit spline coefficients and return the fitted spline and residual.

    @param self ``BSPF1D`` instance.
    @param f Input samples on the operator grid.
    @param lam Tikhonov regularization parameter.
    @param neumann_bc Optional boundary flux values.
    @return Tuple ``(P, f_spline, residual)``.
    """
    return _LegacyBSPF1D.fit_spline(self, f, lam=lam, neumann_bc=neumann_bc)


def interpolate_split_mesh(
    self,
    f: Array,
    refine_factor: int,
    lam: float = 0.0,
    neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None,
):
    """! @brief Interpolate onto an arbitrarily refined mesh with spline/residual split.

    @param self ``BSPF1D`` instance.
    @param f Input samples on the operator grid.
    @param refine_factor Fine-to-coarse refinement ratio.
    @param lam Tikhonov regularization parameter.
    @param neumann_bc Optional boundary flux values.
    @return Tuple ``(x_fine, f_fine, f_spline_fine, r_fine)``.
    """
    return _LegacyBSPF1D.interpolate_split_mesh(
        self,
        f,
        refine_factor=refine_factor,
        lam=lam,
        neumann_bc=neumann_bc,
    )


__all__ = [
    "enforced_zero_flux",
    "fit_spline",
    "interpolate",
    "interpolate_split_mesh",
]
