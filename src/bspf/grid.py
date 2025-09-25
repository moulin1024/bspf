"""Grid operations for BSPF."""
from __future__ import annotations
import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


class Grid1D:
    """Uniform 1D grid with rFFT frequencies and trapezoid weights."""
    
    def __init__(self, x: Array, *, atol: float = 1e-13):
        x = np.asarray(x, dtype=np.float64)
        if x.size < 2:
            raise ValueError("x must have at least 2 points.")
        dx = float(x[1] - x[0])
        if not np.allclose(np.diff(x), dx, rtol=0, atol=atol):
            raise ValueError("x must be uniformly spaced.")
        self.x: Array = x
        self.dx: float = dx
        self.omega: Array = 2.0 * np.pi * np.fft.rfftfreq(x.size, d=dx)
        w = np.full(x.size, dx, dtype=np.float64)
        w[0] = w[-1] = dx / 2.0
        self.trap: Array = w

    @property
    def a(self) -> float: 
        return float(self.x[0])

    @property
    def b(self) -> float: 
        return float(self.x[-1])

    @property
    def n(self) -> int: 
        return self.x.size
