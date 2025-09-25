"""Core BSPF implementation."""
from __future__ import annotations
from typing import Dict, Optional, Tuple, Callable
import numpy as np
import numpy.typing as npt
from scipy import linalg as sla

from .grid import Grid1D
from .knots import KnotGenerator
from .basis import BSplineBasis1D
from .endpoints import EndpointOps1D
from .correction import ResidualCorrection

Array = npt.NDArray[np.float64]


class bspf1d:
    """Facade for 1D BFPSM: derivatives, definite integrals, antiderivatives."""
    
    def __init__(
        self,
        *,
        grid: Grid1D,
        degree: int,
        knots: Array,
        order: Optional[int] = None,
        num_boundary_points: Optional[int] = None,
        correction: str = "spectral",
    ):
        """Initialize BSPF solver.
        
        Parameters
        ----------
        grid : Grid1D
            Uniform grid object
        degree : int
            B-spline degree
        knots : Array
            Knot vector
        order : Optional[int]
            Maximum derivative order for constraints (default: degree+1)
        num_boundary_points : Optional[int] 
            Number of boundary points for finite differences (default: degree+1)
        correction : str
            Residual correction method ('spectral' or 'none')
        """
        self.grid = grid
        self.degree = int(degree)
        self.order = self.degree + 1 if order is None else int(order)
        self.num_bd = self.degree + 1 if num_boundary_points is None else int(num_boundary_points)
        self.knots = np.asarray(knots, dtype=np.float64)

        self.basis = BSplineBasis1D(self.degree, self.knots, self.grid)
        self.BW = self.basis.B0 * self.grid.trap
        self.Q = self.BW @ self.basis.B0.T

        self.end = EndpointOps1D(self.basis, order=self.order, num_bd=self.num_bd)

        if correction == "spectral":
            self._correct = lambda residual, omega, kind, order, n: ResidualCorrection.spectral(
                residual, omega, kind=kind, order=order, n=n, x=self.grid.x
            )
        else:
            self._correct = ResidualCorrection.none

        self._kkt_cache: Dict[float, Tuple[Array, Array]] = {}
        self._cached_arrays: Dict[str, Array] = {}

    @classmethod
    def from_grid(
        cls,
        degree: int,
        x: Array,
        *,
        knots: Optional[Array] = None,
        n_basis: Optional[int] = None,
        domain: Optional[Tuple[float, float]] = None,
        use_clustering: bool = False,
        clustering_factor: float = 2.0,
        order: Optional[int] = None,
        num_boundary_points: Optional[int] = None,
        correction: str = "spectral",
    ) -> "bspf1d":
        """Create BSPF solver from grid points.
        
        Parameters
        ----------
        degree : int
            B-spline degree
        x : Array
            Grid points (must be uniformly spaced)
        knots : Optional[Array]
            Explicit knot vector (if None, will be generated)
        n_basis : Optional[int]
            Number of basis functions (default: 2*(degree+1)*2)
        domain : Optional[Tuple[float, float]]
            Domain for knot generation (default: [x[0], x[-1]])
        use_clustering : bool
            Use clustering near boundaries for knot generation
        clustering_factor : float
            Clustering parameter (higher = more clustering)
        order : Optional[int]
            Maximum derivative order for constraints
        num_boundary_points : Optional[int]
            Number of boundary points for finite differences
        correction : str
            Residual correction method
            
        Returns
        -------
        bspf1d
            Initialized solver
        """
        grid = Grid1D(x)
        k = KnotGenerator.resolve(
            degree=degree, grid=grid, knots=knots, n_basis=n_basis, domain=domain,
            use_clustering=use_clustering, clustering_factor=clustering_factor
        )
        return cls(
            grid=grid, degree=degree, knots=k,
            order=order, num_boundary_points=num_boundary_points, correction=correction
        )

    # ---------- private solvers ----------
    def _kkt_lu(self, lam: float) -> Tuple[Array, Array]:
        """Get cached LU factorization of KKT system."""
        lam = float(lam)
        if lam in self._kkt_cache:
            return self._kkt_cache[lam]
        n_b = self.basis.B0.shape[0]
        m = 2 * self.order
        KKT = np.zeros((n_b + m, n_b + m), dtype=np.float64)
        KKT[:n_b, :n_b] = 2.0 * (self.Q + lam * np.eye(n_b))
        KKT[:n_b, n_b:] = -self.end.C.T
        KKT[n_b:, :n_b] = self.end.C
        lu, piv = sla.lu_factor(KKT)
        self._kkt_cache[lam] = (lu, piv)
        return lu, piv

    def _get_or_compute_array(self, key: str, compute_func: Callable[[], Array]) -> Array:
        """Get cached array or compute if not cached."""
        if key not in self._cached_arrays:
            self._cached_arrays[key] = compute_func()
        return self._cached_arrays[key]

    # ---------- public operations ----------
    def differentiate_with_spline(
        self, 
        f: Array, 
        k: int = 1, 
        lam: float = 0.0, 
        *, 
        neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None
    ) -> Tuple[Array, Array, Array]:
        """Compute k-th derivative with spline approximation returned.
        
        Parameters
        ----------
        f : Array
            Input function values
        k : int
            Derivative order (1, 2, or 3)
        lam : float
            Tikhonov regularization parameter
        neumann_bc : Optional[Tuple[Optional[float], Optional[float]]]
            Neumann boundary conditions (left_flux, right_flux)
            None means natural BC, float enforces specific flux
            
        Returns
        -------
        df : Array
            k-th derivative
        f_spline : Array
            Spline approximation of input
        df_spline : Array
            Spline derivative (before correction)
        """
        if k not in (1, 2, 3):
            raise ValueError("Only 1st/2nd/3rd derivatives are supported.")
        f = np.asarray(f, dtype=np.float64)
        if f.shape[0] != self.grid.n:
            raise ValueError("Length of f must match grid size.")

        rhs_2bw = self._get_or_compute_array('2bw', lambda: 2.0 * (self.BW @ f))
        dY = self.end.BND @ f
        
        # Enforce Neumann BC if specified
        if neumann_bc is not None:
            if self.order < 1:
                raise ValueError("Neumann BC requires self.order ≥ 1.")
            left_flux, right_flux = neumann_bc
            if left_flux is not None:
                dY[1] = float(left_flux)               # left boundary ∂f/∂x
            if right_flux is not None:
                dY[self.order] = float(right_flux)     # right boundary ∂f/∂x
        
        rhs = np.concatenate((rhs_2bw, dY))

        lu, piv = self._kkt_lu(lam)
        sol = sla.lu_solve((lu, piv), rhs, overwrite_b=False)
        P = sol[: self.basis.B0.shape[0]]

        f_spline = self.basis.BT0 @ P
        df_spline = self.basis.BkT(k) @ P

        residual = f - f_spline
        corr = self._correct(residual, self.grid.omega, kind="diff", order=k, n=self.grid.n)
        return (df_spline + corr).astype(np.float64), f_spline.astype(np.float64), df_spline

    def differentiate(
        self, 
        f: Array, 
        k: int = 1, 
        lam: float = 0.0, 
        *, 
        neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None
    ) -> Array:
        """Compute k-th derivative.
        
        Parameters
        ----------
        f : Array
            Input function values
        k : int
            Derivative order (1, 2, or 3)
        lam : float
            Tikhonov regularization parameter
        neumann_bc : Optional[Tuple[Optional[float], Optional[float]]]
            Neumann boundary conditions (left_flux, right_flux)
            
        Returns
        -------
        Array
            k-th derivative
        """
        df, _, _ = self.differentiate_with_spline(f, k=k, lam=lam, neumann_bc=neumann_bc)
        return df

    def differentiate_1_2(
        self, 
        f: Array, 
        lam: float = 0.0,
        *, 
        neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None
    ) -> Tuple[Array, Array, Array]:        
        """Compute both first and second derivatives efficiently.
        
        This reduces FFT operations by computing both derivatives from the same spline fit.

        Parameters
        ----------
        f : Array
            Input function values
        lam : float
            Tikhonov regularization parameter
        neumann_bc : Optional[Tuple[Optional[float], Optional[float]]]
            Neumann boundary conditions

        Returns
        -------
        df1 : Array
            First derivative
        df2 : Array
            Second derivative
        f_spline : Array
            Spline approximation of input function
        """
        f = np.asarray(f, dtype=np.float64)
        if f.shape[0] != self.grid.n:
            raise ValueError("Length of f must match grid size.")

        # Compute spline coefficients (reused for both derivatives)
        rhs_2bw = self._get_or_compute_array('2bw', lambda: 2.0 * (self.BW @ f))
        dY = self.end.BND @ f

        # Enforce Neumann BC if specified
        if neumann_bc is not None:
            if self.order < 1:
                raise ValueError("Neumann BC requires self.order ≥ 1.")
            left_flux, right_flux = neumann_bc
            if left_flux is not None:
                dY[1] = float(left_flux)               # left boundary ∂f/∂x
            if right_flux is not None:
                dY[self.order] = float(right_flux)     # right boundary ∂f/∂x
        
        rhs = np.concatenate((rhs_2bw, dY))

        lu, piv = self._kkt_lu(lam)
        sol = sla.lu_solve((lu, piv), rhs, overwrite_b=False)
        P = sol[: self.basis.B0.shape[0]]

        # Compute spline approximation (used for residual)
        f_spline = self.basis.BT0 @ P

        # Compute spline derivatives
        df1_spline = self.basis.BkT(1) @ P
        df2_spline = self.basis.BkT(2) @ P

        # Compute residual once
        residual = f - f_spline
        R = np.fft.rfft(residual)  # Single FFT for residual
        omega = self.grid.omega

        # Compute spectral corrections for both derivatives
        corr1 = np.fft.irfft(R * (1j * omega), n=self.grid.n)
        corr2 = np.fft.irfft(R * (1j * omega) ** 2, n=self.grid.n)

        # Combine spline and correction terms
        df1 = (df1_spline + corr1).astype(np.float64)
        df2 = (df2_spline + corr2).astype(np.float64)

        return df1, df2, f_spline.astype(np.float64)

    def definite_integral(
        self, 
        f: Array, 
        a: Optional[float] = None, 
        b: Optional[float] = None, 
        lam: float = 0.0
    ) -> float:
        """Compute definite integral ∫[a,b] f(x) dx.
        
        Parameters
        ----------
        f : Array
            Function values
        a : Optional[float]
            Lower bound (default: grid.a)
        b : Optional[float] 
            Upper bound (default: grid.b)
        lam : float
            Tikhonov regularization parameter
            
        Returns
        -------
        float
            Definite integral value
        """
        f = np.asarray(f, dtype=np.float64)
        if f.shape[0] != self.grid.n:
            raise ValueError("Length of f must match grid size.")
        a = self.grid.a if a is None else float(a)
        b = self.grid.b if b is None else float(b)

        rhs_2bw = 2.0 * (self.BW @ f)
        dY = self.end.BND @ f
        rhs = np.concatenate((rhs_2bw, dY))

        lu, piv = self._kkt_lu(lam)
        sol = sla.lu_solve((lu, piv), rhs, overwrite_b=False)
        P = sol[: self.basis.B0.shape[0]]

        basis_integrals = self.basis.integrate_basis(a, b)
        spline_integral = basis_integrals @ P

        residual = f - (self.basis.BT0 @ P)
        residual_integral = np.sum(residual * self.grid.trap)
        return float(spline_integral + residual_integral)

    def antiderivative(
        self,
        f: Array,
        order: int = 1,
        *,
        left_value: float = 0.0,
        match_right: Optional[float] = None,
        lam: float = 0.0,
    ) -> Tuple[Array, Array, Array]:
        """Compute antiderivative of f.
        
        Parameters
        ----------
        f : Array
            Function values to integrate
        order : int
            Integration order (1 or 2)
        left_value : float
            Value at left boundary
        match_right : Optional[float]
            If given, adjust nullspace to match this value at right boundary
        lam : float
            Tikhonov regularization parameter
            
        Returns
        -------
        F : Array
            Antiderivative values
        f_spline : Array
            Spline approximation of input
        F_spline : Array
            Spline antiderivative (before nullspace adjustment)
        """
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2.")

        f = np.asarray(f, dtype=np.float64)
        if f.shape[0] != self.grid.n:
            raise ValueError("Length of f must match grid size.")

        rhs_2bw = 2.0 * (self.BW @ f)
        dY = self.end.BND @ f
        rhs = np.concatenate((rhs_2bw, dY))
        lu, piv = self._kkt_lu(lam)
        sol = sla.lu_solve((lu, piv), rhs, overwrite_b=False)
        P = sol[: self.basis.B0.shape[0]]

        x = self.grid.x
        f_spline = self.basis.BT0 @ P

        # Antiderivative of the spline part
        F_spline = np.zeros_like(x)
        for i, s in enumerate(self.basis._splines):
            s_int = s.antiderivative(order)
            F_spline += P[i] * s_int(x)

        # Residual correction via spectral integration (with DC handling)
        residual = f - f_spline
        F_corr = self._correct(residual, self.grid.omega, kind="int", order=order, n=self.grid.n)

        # Combine and enforce boundary constraints using correct nullspace
        F = F_spline + F_corr
        x0, x1 = float(x[0]), float(x[-1])

        # Left value
        F = F - (F[0] - float(left_value))

        # Optional right value using the correct nullspace
        if match_right is not None:
            if order == 1:
                F = F + (float(match_right) - F[-1])  # constant shift
            else:
                F = F + (float(match_right) - F[-1]) * (x - x0) / (x1 - x0)  # linear term

        return F, f_spline, F_spline
