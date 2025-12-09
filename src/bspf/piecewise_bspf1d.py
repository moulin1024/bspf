from __future__ import annotations

from typing import Optional, Tuple, List, Union
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
        df_segments = []
        fs_segments = []
        
        for seg_idx, seg in enumerate(self.segments):
            i0, i1, op = seg["i0"], seg["i1"], seg["op"]
            
            # Get function values on segment grid (interpolate if needed)
            if "x_seg" in seg:
                x_seg = seg["x_seg"]
                f_seg = np.interp(x_seg, self.x, f)
            else:
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
            
            df_segments.append(df_seg)
            fs_segments.append(fs_seg)
        
        # Interpolate results back to full grid
        df_full = self._interpolate_from_segments(df_segments)
        fs_full = self._interpolate_from_segments(fs_segments)
        
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
        d1_segments = []
        d2_segments = []
        fs_segments = []
        
        for k, seg in enumerate(self.segments):
            i0, i1, op = seg["i0"], seg["i1"], seg["op"]
            
            # Get function values on segment grid (interpolate if needed)
            if "x_seg" in seg:
                x_seg = seg["x_seg"]
                f_seg = np.interp(x_seg, self.x, f)
            else:
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
            
            d1_segments.append(d1_seg)
            d2_segments.append(d2_seg)
            fs_segments.append(fs_seg)
        
        # Interpolate results back to full grid
        df1_full = self._interpolate_from_segments(d1_segments)
        df2_full = self._interpolate_from_segments(d2_segments)
        fs_full = self._interpolate_from_segments(fs_segments)
        
        return df1_full, df2_full, fs_full
    
    @classmethod
    def from_domains(
        cls,
        degree: int,
        domain_boundaries: List[float],
        n_points_per_segment: Union[int, List[int]],
        **bspf_kwargs
    ) -> "PiecewiseBSPF1D":
        """
        Create PiecewiseBSPF1D with independent mesh resolution for each subdomain.
        
        This method allows you to specify different grid resolutions for each
        subdomain, enabling local refinement strategies.
        
        Parameters
        ----------
        degree : int
            B-spline degree for each segment
        domain_boundaries : List[float]
            List of domain boundaries [a, b1, b2, ..., bN, c] defining N+1 subdomains:
            [a, b1), [b1, b2), ..., [bN, c]
        n_points_per_segment : int or List[int]
            If int: same number of points for all segments
            If List[int]: number of points for each segment (length must be N+1)
        **bspf_kwargs
            Additional arguments passed to bspf1d.from_grid for each segment
        
        Returns
        -------
        PiecewiseBSPF1D
            Instance with independent grids for each subdomain
        
        Example
        -------
        >>> # Create 3 subdomains with different resolutions
        >>> pw = PiecewiseBSPF1D.from_domains(
        ...     degree=5,
        ...     domain_boundaries=[0, np.pi/2, 3*np.pi/2, 2*np.pi],
        ...     n_points_per_segment=[100, 500, 100]  # Fine resolution in middle
        ... )
        """
        domain_boundaries = sorted(domain_boundaries)
        if len(domain_boundaries) < 2:
            raise ValueError("domain_boundaries must have at least 2 elements")
        
        n_segments = len(domain_boundaries) - 1
        
        # Handle n_points_per_segment
        if isinstance(n_points_per_segment, int):
            n_points_list = [n_points_per_segment] * n_segments
        elif isinstance(n_points_per_segment, list):
            if len(n_points_per_segment) != n_segments:
                raise ValueError(
                    f"n_points_per_segment length {len(n_points_per_segment)} "
                    f"must match number of segments {n_segments}"
                )
            n_points_list = n_points_per_segment
        else:
            raise TypeError("n_points_per_segment must be int or List[int]")
        
        # Create grids for each segment
        segment_grids = []
        segment_domains = []
        for i in range(n_segments):
            a = domain_boundaries[i]
            b = domain_boundaries[i + 1]
            n_pts = n_points_list[i]
            x_seg = np.linspace(a, b, n_pts, endpoint=True)
            segment_grids.append(x_seg)
            segment_domains.append((a, b))
        
        # Build full grid by concatenating segment grids, handling boundaries carefully
        # For interior boundaries, include the point only once (from the right segment)
        x_full_list = []
        for i, x_seg in enumerate(segment_grids):
            if i == 0:
                # First segment: include all points
                x_full_list.append(x_seg)
            else:
                # Subsequent segments: skip first point (it's the boundary from previous segment)
                x_full_list.append(x_seg[1:])
        
        x_full = np.concatenate(x_full_list)
        
        # Create instance with dummy initialization
        instance = cls.__new__(cls)
        instance.degree = int(degree)
        instance.x = x_full
        instance.breakpoints = domain_boundaries[1:-1]  # Interior boundaries only
        instance.min_points_per_seg = 16
        
        # Create segments with independent grids
        instance.segments = []
        
        for i, (x_seg, domain_seg) in enumerate(zip(segment_grids, segment_domains)):
            # Find indices in full grid corresponding to this segment
            # For first segment: [a, b]
            # For subsequent segments: (prev_b, b] - need to find where prev_b ends
            if i == 0:
                # First segment: find all points in [a, b]
                mask = (x_full >= domain_seg[0] - 1e-12) & (x_full <= domain_seg[1] + 1e-12)
            else:
                # Subsequent segments: find points in (prev_b, b]
                # Use > instead of >= to exclude the boundary point
                prev_b = domain_boundaries[i]
                mask = (x_full > prev_b - 1e-12) & (x_full <= domain_seg[1] + 1e-12)
            
            indices = np.where(mask)[0]
            
            if len(indices) < instance.min_points_per_seg:
                continue
            
            i0 = int(indices[0])
            i1 = int(indices[-1])
            
            # Create bspf1d operator for this segment using its own grid
            op = bspf1d.from_grid(degree=instance.degree, x=x_seg, domain=domain_seg, **bspf_kwargs)
            
            instance.segments.append(dict(
                i0=i0, 
                i1=i1, 
                op=op,
                x_seg=x_seg,  # Store segment grid
                domain=domain_seg  # Store domain
            ))
        
        return instance
    
    def _interpolate_from_segments(self, results_segments: List[Array]) -> Array:
        """
        Interpolate results from segment grids back to full grid.
        
        Parameters
        ----------
        results_segments : List[Array]
            List of result arrays, one per segment (on segment grids)
        
        Returns
        -------
        Array
            Results interpolated to full grid
        """
        result_full = np.zeros_like(self.x, dtype=np.float64)
        
        for seg, result_seg in zip(self.segments, results_segments):
            i0, i1 = seg["i0"], seg["i1"]
            
            if "x_seg" in seg:
                # Interpolate from segment grid to full grid
                x_seg = seg["x_seg"]
                x_full_seg = self.x[i0:i1 + 1]
                result_interp = np.interp(x_full_seg, x_seg, result_seg)
                result_full[i0:i1 + 1] = result_interp
            else:
                # Direct assignment (same grid)
                result_full[i0:i1 + 1] = result_seg
        
        return result_full

