"""! @file ops/integration.py
@brief Package-owned integration workflows for BSPF1D.
"""

from __future__ import annotations

from typing import Optional

from bspf1d import bspf1d as _LegacyBSPF1D

from ..types import Array


def definite_integral(
    self,
    f: Array,
    a: Optional[float] = None,
    b: Optional[float] = None,
    lam: float = 0.0,
) -> float:
    """! @brief Compute a definite integral of the sampled signal.

    @param self ``BSPF1D`` instance.
    @param f Input samples on the operator grid.
    @param a Optional left integration bound.
    @param b Optional right integration bound.
    @param lam Tikhonov regularization parameter.
    @return Scalar integral estimate.
    """
    return _LegacyBSPF1D.definite_integral(self, f, a=a, b=b, lam=lam)


def antiderivative(
    self,
    f: Array,
    order: int = 1,
    *,
    left_value: float = 0.0,
    match_right: Optional[float] = None,
    lam: float = 0.0,
):
    """! @brief Compute a first or second antiderivative of the sampled signal.

    @param self ``BSPF1D`` instance.
    @param f Input samples on the operator grid.
    @param order Antiderivative order.
    @param left_value Value enforced at the left endpoint.
    @param match_right Optional value matched at the right endpoint.
    @param lam Tikhonov regularization parameter.
    @return Tuple ``(F, f_spline)``.
    """
    return _LegacyBSPF1D.antiderivative(
        self,
        f,
        order=order,
        left_value=left_value,
        match_right=match_right,
        lam=lam,
    )


__all__ = ["antiderivative", "definite_integral"]
