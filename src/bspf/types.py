"""Shared typing helpers for the package."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

__all__ = ["Array"]
