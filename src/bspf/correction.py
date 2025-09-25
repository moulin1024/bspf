"""Residual correction strategies for BSPF."""
from __future__ import annotations
from typing import Optional
import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


class ResidualCorrection:
    """Pluggable residual correction strategies."""
    
    @staticmethod
    def none(
        residual: Array, 
        omega: Array, 
        *, 
        kind: str, 
        order: int, 
        n: int, 
        x: Optional[Array] = None
    ) -> Array:
        """No correction - return zeros.
        
        Parameters
        ----------
        residual : Array
            Residual values
        omega : Array
            Frequency array
        kind : str
            Operation type ('diff' or 'int')
        order : int
            Derivative/integral order
        n : int
            Output length
        x : Optional[Array]
            Grid points (unused for this correction)
            
        Returns
        -------
        Array
            Zero correction
        """
        return np.zeros(n, dtype=np.float64)

    @staticmethod
    def spectral(
        residual: Array, 
        omega: Array, 
        *, 
        kind: str, 
        order: int, 
        n: int, 
        x: Optional[Array] = None
    ) -> Array:
        """Spectral correction using FFT.
        
        Parameters
        ----------
        residual : Array
            Residual values to correct
        omega : Array
            Frequency array from grid
        kind : str
            Operation type - 'diff' for differentiation, 'int' for integration
        order : int
            Order of derivative/integral
        n : int
            Output length
        x : Optional[Array]
            Grid points (needed for integration nullspace handling)
            
        Returns
        -------
        Array
            Correction values
            
        Raises
        ------
        ValueError
            If kind is not 'diff' or 'int', or if unsupported integration order
        """
        R = np.fft.rfft(residual)

        if kind == "diff":
            return np.fft.irfft(R * (1j * omega) ** order, n=n).astype(np.float64)

        if kind == "int":
            out_hat = np.zeros_like(R, dtype=np.complex128)
            nz = omega != 0.0
            out_hat[nz] = R[nz] / ((1j * omega[nz]) ** order)
            out = np.fft.irfft(out_hat, n=n).astype(np.float64)

            # Need x for correct nullspace handling
            if x is None:
                x0, x1 = 0.0, 1.0
                xx = np.linspace(x0, x1, n)
            else:
                xx = x
                x0 = float(xx[0])
                x1 = float(xx[-1])

            if order == 1:
                # Restore dropped DC in derivative: add mean(residual)*(x-x0),
                # then anchor the left value.
                mean_r = float(np.mean(residual))
                out = out + mean_r * (xx - x0)
                out -= out[0]
                return out

            if order == 2:
                # Restore DC (constant curvature) as quadratic with zero endpoints
                mean_r = float(np.mean(residual))
                q = 0.5 * mean_r * (xx - x0) * (xx - x1)  # q'' = mean_r, q(x0)=q(x1)=0
                return out + q

            raise ValueError("Only int orders 1 and 2 are supported.")

        raise ValueError("kind must be 'diff' or 'int'.")
