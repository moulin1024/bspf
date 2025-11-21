from __future__ import annotations

from typing import Optional, Tuple, List
import numpy as np
import numpy.typing as npt

from .bspf1d import bspf1d

Array = npt.NDArray[np.float64]


# =============================================================================
# Piecewise BSPF for functions with discontinuities
# =============================================================================
class PiecewiseBSPF1D:
    """
    Piecewise BSPF operator for functions with known discontinuities.
    
    Segments the domain at breakpoints and applies bspf1d to each segment
    independently. This improves accuracy for functions with jumps or
    discontinuities.
    
    Breakpoints are interpreted as physical coordinates that can fall between
    grid cells. Each breakpoint is interpreted as: discontinuity lies between
    x[idx-1] and x[idx]. Left segment uses indices 0..idx-1, right segment
    uses idx..N-1.
    
    Parameters
    ----------
    degree : int
        B-spline degree for each segment
    x : Array
        Full grid points (must be uniformly spaced)
    breakpoints : List[float], optional
        List of x-coordinates where discontinuities occur. Default: []
    min_points_per_seg : int, default 16
        Minimum number of points required per segment. Segments with fewer
        points are skipped.
    **bspf_kwargs
        Additional arguments passed to bspf1d.from_grid for each segment
        (e.g., order, correction, use_gpu)
    
    Example
    -------
    >>> x = np.linspace(0, 2*np.pi, 512)
    >>> pw = PiecewiseBSPF1D(degree=5, x=x, breakpoints=[np.pi/2, 3*np.pi/2])
    >>> df1, df2, f_spline = pw.differentiate_1_2(f)
    """
    
    def __init__(self, degree: int, x: Array, breakpoints: Optional[List[float]] = None,
                 min_points_per_seg: int = 16, **bspf_kwargs):
        self.degree = int(degree)
        self.x = np.asarray(x, dtype=np.float64)
        self.breakpoints = sorted(breakpoints or [])
        self.min_points_per_seg = int(min_points_per_seg)
        
        N = self.x.size
        
        # 1. Convert physical coordinates to cell interface indices
        cut_indices = []
        for bp in self.breakpoints:
            idx = int(np.searchsorted(self.x, bp))  # x[idx-1] < bp <= x[idx]
            if 1 <= idx <= N - 1:
                cut_indices.append(idx)
        cut_indices = sorted(set(cut_indices))
        
        self.segments = []  # Each segment: {i0, i1, op}
        
        i_start = 0
        for idx in cut_indices:
            i_end = idx - 1  # Left segment goes to idx-1
            if i_end - i_start + 1 >= self.min_points_per_seg:
                x_seg = self.x[i_start:i_end + 1]
                op = bspf1d.from_grid(degree=self.degree, x=x_seg, **bspf_kwargs)
                self.segments.append(dict(i0=i_start, i1=i_end, op=op))
            # Otherwise segment is too short, skip it
            i_start = idx  # Right segment starts from idx
        
        # Last segment
        if N - 1 - i_start + 1 >= self.min_points_per_seg:
            x_seg = self.x[i_start:]
            op = bspf1d.from_grid(degree=self.degree, x=x_seg, **bspf_kwargs)
            self.segments.append(dict(i0=i_start, i1=N - 1, op=op))
    
    def differentiate(self, f: Array, k: int = 1, lam: float = 0.0,
                     neumann_bc_global: Optional[Tuple[Optional[float], Optional[float]]] = None
                     ) -> Tuple[Array, Array]:
        """
        Compute derivative of order k using piecewise BSPF.
        
        Calls bspf1d.differentiate on each segment, then concatenates results.
        
        Parameters
        ----------
        f : Array
            Function values on the full grid
        k : int, default 1
            Derivative order (1, 2, or 3)
        lam : float, default 0.0
            Tikhonov regularization parameter
        neumann_bc_global : tuple, optional
            Neumann boundary conditions (left_flux, right_flux) for the global domain.
            Interior interfaces do not apply Neumann BCs, determined by physics.
        
        Returns
        -------
        df : Array
            Derivative of order k on full grid
        f_spline : Array
            Spline approximation on full grid
        """
        f = np.asarray(f, dtype=np.float64)
        if f.shape[0] != self.x.size:
            raise ValueError(f"f length {f.shape[0]} must match x length {self.x.size}")
        
        df_full = np.zeros_like(f, dtype=np.float64)
        fs_full = np.zeros_like(f, dtype=np.float64)
        
        if neumann_bc_global is not None:
            left_flux_global, right_flux_global = neumann_bc_global
        else:
            left_flux_global = right_flux_global = None
        
        n_seg = len(self.segments)
        for seg_idx, seg in enumerate(self.segments):
            i0, i1, op = seg["i0"], seg["i1"], seg["op"]
            f_seg = f[i0:i1 + 1]
            
            # Only apply global Neumann BC at the two ends of the entire domain
            if seg_idx == 0:
                bc_left = left_flux_global
            else:
                bc_left = None
            if seg_idx == n_seg - 1:
                bc_right = right_flux_global
            else:
                bc_right = None
            neumann_bc_seg = (bc_left, bc_right)
            
            df_seg, fs_seg = op.differentiate(
                f_seg, k=k, lam=lam, neumann_bc=neumann_bc_seg
            )
            
            df_full[i0:i1 + 1] = df_seg
            fs_full[i0:i1 + 1] = fs_seg
        
        return df_full, fs_full
    
    def differentiate_1_2(self, f: Array, lam: float = 0.0,
                          neumann_bc_global: Optional[Tuple[Optional[float], Optional[float]]] = None
                          ) -> Tuple[Array, Array, Array]:
        """
        Compute first and second derivatives using piecewise BSPF.
        
        Calls bspf1d.differentiate_1_2 on each segment, then concatenates results.
        
        Parameters
        ----------
        f : Array
            Function values on the full grid
        lam : float, default 0.0
            Tikhonov regularization parameter
        neumann_bc_global : tuple, optional
            Neumann boundary conditions (left_flux, right_flux) for the global domain.
            Interior interfaces do not apply Neumann BCs, determined by physics.
        
        Returns
        -------
        df1 : Array
            First derivative on full grid
        df2 : Array
            Second derivative on full grid
        f_spline : Array
            Spline approximation on full grid
        """
        f = np.asarray(f, dtype=np.float64)
        if f.shape[0] != self.x.size:
            raise ValueError(f"f length {f.shape[0]} must match x length {self.x.size}")
        
        df1_full = np.zeros_like(f, dtype=np.float64)
        df2_full = np.zeros_like(f, dtype=np.float64)
        fs_full = np.zeros_like(f, dtype=np.float64)
        
        if neumann_bc_global is not None:
            left_flux_global, right_flux_global = neumann_bc_global
        else:
            left_flux_global = right_flux_global = None
        
        n_seg = len(self.segments)
        for k, seg in enumerate(self.segments):
            i0, i1, op = seg["i0"], seg["i1"], seg["op"]
            f_seg = f[i0:i1 + 1]
            
            # Only apply global Neumann BC at the two ends of the entire domain
            if k == 0:
                bc_left = left_flux_global
            else:
                bc_left = None
            if k == n_seg - 1:
                bc_right = right_flux_global
            else:
                bc_right = None
            neumann_bc_seg = (bc_left, bc_right)
            
            d1_seg, d2_seg, fs_seg = op.differentiate_1_2(
                f_seg, lam=lam, neumann_bc=neumann_bc_seg
            )
            
            df1_full[i0:i1 + 1] = d1_seg
            df2_full[i0:i1 + 1] = d2_seg
            fs_full[i0:i1 + 1] = fs_seg
        
        return df1_full, df2_full, fs_full

