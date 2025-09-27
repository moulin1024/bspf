from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Callable

import numpy as np
import numpy.typing as npt
from scipy import linalg as sla
from scipy.interpolate import BSpline

Array = npt.NDArray[np.float64]


# =============================================================================
# Low-level building blocks
# =============================================================================
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
    def a(self) -> float: return float(self.x[0])

    @property
    def b(self) -> float: return float(self.x[-1])

    @property
    def n(self) -> int: return self.x.size


class _Knot:
    @staticmethod
    def _generate(
        *, degree: int, domain: Tuple[float, float], n_basis: int,
        use_clustering: bool, clustering_factor: float
    ) -> Array:
        if n_basis <= degree:
            raise ValueError("n_basis must exceed degree.")
        n_knots = n_basis + degree + 1
        n_interior = n_knots - 2 * (degree + 1)

        if n_interior > 0:
            u = np.linspace(-1.0, 1.0, n_interior + 2)
            if use_clustering:
                u = np.tanh(clustering_factor * u) / np.tanh(clustering_factor)
            uniq = degree * (u + 1.0) / 2.0  # in [0, degree]
            ks = [float(uniq[0])] * (degree + 1) + list(map(float, uniq[1:-1])) + [float(uniq[-1])] * (degree + 1)
            k = np.array(ks, dtype=np.float64)
        else:
            k = np.concatenate([np.zeros(degree + 1), np.full(degree + 1, degree)], dtype=np.float64)

        a, b = domain
        return (k / degree) * (b - a) + a

    @staticmethod
    def resolve(
        *, degree: int, grid: Grid1D,
        knots: Optional[Array],
        n_basis: Optional[int],
        domain: Optional[Tuple[float, float]],
        use_clustering: bool,
        clustering_factor: float,
    ) -> Array:
        if knots is not None:
            k = np.asarray(knots, dtype=np.float64)
            if k.ndim != 1:
                raise ValueError("knots must be a 1D array.")
            return k
        if n_basis is None:
            n_basis = 2 * (degree + 1) * 2
        if domain is None:
            domain = (grid.a, grid.b)
        return _Knot._generate(
            degree=degree, domain=domain, n_basis=n_basis,
            use_clustering=use_clustering, clustering_factor=clustering_factor
        )


class BSplineBasis1D:
    """B-spline basis on a uniform grid with lazy derivative matrices."""
    def __init__(self, degree: int, knots: Array, grid: Grid1D):
        self.degree = int(degree)
        self.knots: Array = np.asarray(knots, dtype=np.float64)
        self.grid = grid

        self._splines = self._mk_splines()
        n_basis = len(self._splines)
        B0 = np.empty((n_basis, grid.n), dtype=np.float64)
        for i, s in enumerate(self._splines):
            B0[i, :] = s(grid.x)
        self._B0: Array = B0
        self._BT0: Array = B0.T.copy()

        self._BkT: Dict[int, Array] = {}
        self._eval_cache: Dict[Tuple[float, int], Array] = {}

    def _mk_splines(self) -> list[BSpline]:
        n_basis = len(self.knots) - self.degree - 1
        coeffs = np.eye(n_basis, dtype=np.float64)
        return [BSpline(self.knots, c, self.degree) for c in coeffs]

    def _evaluate_splines_vectorized(self, x: Array, deriv_order: int = 0) -> Array:
        cache_key = (float(x[0]), deriv_order)  # uniform-grid key
        if cache_key in self._eval_cache:
            return self._eval_cache[cache_key]
        n_basis = len(self._splines)
        result = np.empty((n_basis, len(x)), dtype=np.float64)
        for i, s in enumerate(self._splines):
            result[i, :] = (s.derivative(deriv_order) if deriv_order else s)(x)
        self._eval_cache[cache_key] = result
        return result

    @property
    def B0(self) -> Array: return self._B0
    @property
    def BT0(self) -> Array: return self._BT0

    def BkT(self, k: int) -> Array:
        if k == 0:
            return self._BT0
        if k not in self._BkT:
            Bk = self._evaluate_splines_vectorized(self.grid.x, deriv_order=k)
            self._BkT[k] = Bk.T.copy()
        return self._BkT[k]

    def integrate_basis(self, a: float, b: float) -> Array:
        return np.array([s.integrate(a, b) for s in self._splines])


class EndpointOps1D:
    """Endpoint constraints and sample-to-endpoint operator for endpoint derivatives."""
    def __init__(self, basis: BSplineBasis1D, *, order: int, num_bd: int):
        self.order = int(order)
        self.num_bd = int(num_bd)
        self.grid = basis.grid
        B0 = basis.B0
        Bk = {0: B0}
        for k in range(1, order + 1):
            Bk[k] = basis.BkT(k).T

        n_basis, n_points = B0.shape

        C = np.zeros((2 * order, n_basis), dtype=np.float64)
        for p in range(order):
            C[p, :] = Bk[p][:, 0]
            C[order + p, :] = Bk[p][:, -1]

        i, j = np.meshgrid(np.arange(num_bd), np.arange(num_bd), indexing="ij")
        fact = np.array([math.factorial(k) for k in range(num_bd)], dtype=np.float64)
        A_left = (j**i) / fact[:, None]
        A_right = np.flip(A_left * ((-1.0) ** i), axis=(0, 1))

        E_left = np.eye(num_bd, dtype=np.float64)[:order, :].T
        idx = np.arange(num_bd - 1, num_bd - order - 1, -1)
        E_right = np.eye(num_bd, dtype=np.float64)[idx, :].T

        X_left = sla.solve(A_left, E_left).T
        X_right = sla.solve(A_right, E_right).T

        dx_pows = self.grid.dx ** np.arange(order, dtype=np.float64)
        BND = np.zeros((2 * order, n_points), dtype=np.float64)
        BND[:order, :num_bd] = X_left / dx_pows[:, None]
        BND[order:, n_points - num_bd:] = X_right / dx_pows[:, None]

        self.C: Array = C.astype(np.float64)
        self.BND: Array = BND.astype(np.float64)
        self.X_left: Array = X_left.astype(np.float64)
        self.X_right: Array = X_right.astype(np.float64)


# =============================================================================
# Residual correction strategies
# =============================================================================
class ResidualCorrection:
    """Pluggable residual correction."""
    @staticmethod
    def none(residual: Array, omega: Array, *, kind: str, order: int, n: int, x: Optional[Array] = None) -> Array:
        return np.zeros(n, dtype=np.float64)

    @staticmethod
    def spectral(residual: Array, omega: Array, *, kind: str, order: int, n: int, x: Optional[Array] = None) -> Array:
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
                x0 = float(xx[0]); x1 = float(xx[-1])

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


# =============================================================================
# Facade
# =============================================================================
class bspf1d:
    """Facade for 1D bspf: derivatives, definite integrals, antiderivatives."""
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
        self.grid = grid
        self.degree = int(degree)
        self.order = self.degree - 1 if order is None else int(order)
        self.num_bd = self.degree if num_boundary_points is None else int(num_boundary_points)
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
        grid = Grid1D(x)
        k = _Knot.resolve(
            degree=degree, grid=grid, knots=knots, n_basis=n_basis, domain=domain,
            use_clustering=use_clustering, clustering_factor=clustering_factor
        )
        return cls(
            grid=grid, degree=degree, knots=k,
            order=order, num_boundary_points=num_boundary_points, correction=correction
        )

    # ---------- private solvers ----------
    def _kkt_lu(self, lam: float) -> Tuple[Array, Array]:
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
        if key not in self._cached_arrays:
            self._cached_arrays[key] = compute_func()
        return self._cached_arrays[key]

    # ---------- public operations ----------
    def differentiate(self, f: Array, k: int = 1, lam: float = 0.0, *, 
        neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None) -> Tuple[Array, Array]:
        if k not in (1, 2, 3):
            raise ValueError("Only 1st/2nd/3rd derivatives are supported.")
        f = np.asarray(f, dtype=np.float64)
        if f.shape[0] != self.grid.n:
            raise ValueError("Length of f must match grid size.")

        rhs_2bw = self._get_or_compute_array('2bw', lambda: 2.0 * (self.BW @ f))
        dY = self.end.BND @ f
        # ---------- NEW: enforce Neumann BC (first-derivative / flux) ----------
        if neumann_bc is not None:
            if self.order < 1:
                raise ValueError("Neumann BC requires self.order ≥ 1.")
            left_flux, right_flux = neumann_bc   # (None means “don’t care”)
            if left_flux is not None:
                dY[1] = float(left_flux)               # left boundary ∂f/∂x
            if right_flux is not None:
                dY[self.order + 1] = float(right_flux) # right boundary ∂f/∂x
        
        rhs = np.concatenate((rhs_2bw, dY))

        lu, piv = self._kkt_lu(lam)
        sol = sla.lu_solve((lu, piv), rhs, overwrite_b=False)
        P = sol[: self.basis.B0.shape[0]]

        f_spline = self.basis.BT0 @ P
        df = self.basis.BkT(k) @ P

        residual = f - f_spline
        corr = self._correct(residual, self.grid.omega, kind="diff", order=k, n=self.grid.n)
        return (df + corr).astype(np.float64), f_spline.astype(np.float64)

    def differentiate_1_2(self, f: Array, lam: float = 0.0,*, 
                          neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None) -> Tuple[Array, Array, Array]:        
        """
        Compute both first and second derivatives together efficiently.
        This reduces FFT operations by computing both derivatives from the same spline fit.

        Parameters
        ----------
        f : Array
            Input function values
        lam : float, default 0.0
            Tikhonov regularization parameter

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

        # ---------- NEW: enforce Neumann BC (first-derivative / flux) ----------
        if neumann_bc is not None:
            if self.order < 1:
                raise ValueError("Neumann BC requires self.order ≥ 1.")
            left_flux, right_flux = neumann_bc   # (None means “don’t care”)
            if left_flux is not None:
                dY[1] = float(left_flux)               # left boundary ∂f/∂x
            if right_flux is not None:
                dY[self.order + 1] = float(right_flux) # right boundary ∂f/∂x
        # ----------------------------------------------------------------------
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

    def definite_integral(self, f: Array, a: Optional[float] = None, b: Optional[float] = None, lam: float = 0.0) -> float:
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

    # ---------- Antiderivative (order 1 or 2) ----------
    def antiderivative(
        self,
        f: Array,
        order: int = 1,
        *,
        left_value: float = 0.0,
        match_right: Optional[float] = None,
        lam: float = 0.0,
    ) -> Array:
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

        return F, f_spline
