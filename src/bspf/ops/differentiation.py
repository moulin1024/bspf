"""! @file ops/differentiation.py
@brief Package-owned differentiation workflows for BSPF1D.

The numerical implementation is delegated to the legacy method bodies for now,
but the package owns the callable surface and module boundaries starting in
Phase 5.
"""

from __future__ import annotations

from typing import Optional, Tuple

from bspf1d import bspf1d as _LegacyBSPF1D

from ..types import Array


def differentiate(
    self,
    f: Array,
    k: int = 1,
    lam: float = 0.0,
    *,
    neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None,
):
    """! @brief Differentiate a sampled signal.

    @param self ``BSPF1D`` instance.
    @param f Input samples on the operator grid.
    @param k Derivative order.
    @param lam Tikhonov regularization parameter.
    @param neumann_bc Optional boundary flux values.
    @return Tuple ``(df, f_spline)``.
    """
    # Delegate to the legacy implementation while the package transitions the
    # numerical kernels into standalone operation modules.
    return _LegacyBSPF1D.differentiate(self, f, k=k, lam=lam, neumann_bc=neumann_bc)


def differentiate_1_2(
    self,
    f: Array,
    lam: float = 0.0,
    *,
    neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None,
):
    """! @brief Compute the first and second derivatives together.

    @param self ``BSPF1D`` instance.
    @param f Input samples on the operator grid.
    @param lam Tikhonov regularization parameter.
    @param neumann_bc Optional boundary flux values.
    @return Tuple ``(df1, df2, f_spline)``.
    """
    return _LegacyBSPF1D.differentiate_1_2(self, f, lam=lam, neumann_bc=neumann_bc)


def differentiate_1_2_3(
    self,
    f: Array,
    lam: float = 0.0,
    *,
    neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None,
):
    """! @brief Compute the first, second, and third derivatives together.

    @param self ``BSPF1D`` instance.
    @param f Input samples on the operator grid.
    @param lam Tikhonov regularization parameter.
    @param neumann_bc Optional boundary flux values.
    @return Tuple ``(df1, df2, df3, f_spline)``.
    """
    return _LegacyBSPF1D.differentiate_1_2_3(self, f, lam=lam, neumann_bc=neumann_bc)


def differentiate_1_2_batched(
    self,
    f: Array,
    lam: float = 0.0,
    *,
    neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None,
):
    """! @brief Batched first and second derivatives for multiple signals.

    @param self ``BSPF1D`` instance.
    @param f Input samples with shape ``(n, batch)``.
    @param lam Tikhonov regularization parameter.
    @param neumann_bc Optional boundary flux values.
    @return Tuple ``(df1, df2, f_spline)`` for the batch.
    """
    return _LegacyBSPF1D.differentiate_1_2_batched(self, f, lam=lam, neumann_bc=neumann_bc)


__all__ = [
    "differentiate",
    "differentiate_1_2",
    "differentiate_1_2_3",
    "differentiate_1_2_batched",
]
