from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Tuple as TupType
import numpy as np
import numpy.typing as npt
from scipy import linalg as sla

from .bspf1d import bspf1d  # 1D facade with cached LU and basis blocks

Array = npt.NDArray[np.float64]


# =====================================================================
# Precomputed derivative plans (fast path for repeated evaluations)
# =====================================================================
@dataclass
class _AxisPlan:
    """Precomputed derivative plan along one axis for repeated calls."""
    model: bspf1d
    axis: int                  # 1 for x (columns), 0 for y (rows)
    order: int                 # derivative order k
    lam: float
    neumann: bool = False      # enforce first-derivative flux at the ends?

    # cached handles / constants (filled in __post_init__)
    BW: Array = field(init=False)
    BND: Array = field(init=False)
    BT: Array = field(init=False)
    BkT: Array = field(init=False)
    omega: Array = field(init=False)
    n_b: int = field(init=False)
    m: int = field(init=False)
    n: int = field(init=False)
    left_row: int = field(init=False)
    right_row: int = field(init=False)
    lu: Array = field(init=False)
    piv: Array = field(init=False)

    def __post_init__(self):
        self.BW  = self.model.BW
        self.BND = self.model.end.BND
        self.BT  = self.model.basis.BT0
        self.BkT = self.model.basis.BkT(self.order)   # cached inside basis
        self.omega = self.model.grid.omega
        self.n_b = self.BW.shape[0]
        self.m   = self.BND.shape[0]
        self.n   = self.model.grid.n

        # Which constraint rows correspond to first derivative at the ends?
        ord_ = self.model.end.order
        if self.neumann:
            if ord_ < 2:
                raise ValueError("Model 'order' must be >= 2 to enforce Neumann flux.")
            self.left_row  = 1            # first-derivative @ left
            self.right_row = ord_ + 1     # first-derivative @ right
        else:
            self.left_row = self.right_row = -1  # unused

        # Precompute LU of KKT (bspf1d caches per lam internally; we hold a handle)
        self.lu, self.piv = self.model._kkt_lu(self.lam)

    @staticmethod
    def _broadcast_flux(val, batch: int) -> np.ndarray:
        v = np.asarray(val, dtype=np.float64)
        if v.ndim == 0:
            return np.full(batch, float(v))
        if v.shape == (batch,):
            return v
        raise ValueError(f"Flux must be scalar or shape=({batch},), got {v.shape}.")

    def apply(self, F: Array, *, flux: Tuple[float | Array, float | Array] = (0.0, 0.0),
              return_spline: bool = False):
        """Compute ∂^order(F)/∂axis^order with optional Neumann flux enforcement."""
        # Move selected axis to front → (n, batch)
        FT = np.moveaxis(F, self.axis, 0)
        n, batch = FT.shape

        # Build RHS = [ 2*BW@FT ; BND@FT ] and optionally override flux rows
        rhs_top = 2.0 * (self.BW @ FT)           # (n_b, batch)
        dY = self.BND @ FT                       # (m, batch)
        if self.neumann:
            lf = self._broadcast_flux(flux[0], batch)
            rf = self._broadcast_flux(flux[1], batch)
            dY[self.left_row,  :] = lf
            dY[self.right_row, :] = rf

        RHS = np.vstack([rhs_top, dY])           # (n_b+m, batch)

        # Solve KKT with precomputed LU
        SOL = sla.lu_solve((self.lu, self.piv), RHS)     # (n_b+m, batch)
        P   = SOL[: self.n_b, :]                         # spline coeffs

        # Spline piece + residual spectral correction
        spline = self.BT @ P                              # (n, batch)
        deriv  = self.BkT @ P                             # (n, batch)

        resid = FT - spline
        R     = np.fft.rfft(resid, axis=0)
        corr  = np.fft.irfft(R * (1j * self.omega)[:, None]**self.order, n=n, axis=0)

        D = deriv + corr
        if self.neumann:
            # For k==1, pin endpoints exactly to requested flux
            if self.order == 1:
                D[0,  :] = dY[self.left_row,  :]
                D[-1, :] = dY[self.right_row, :]

        if return_spline:
            return np.moveaxis(D, 0, self.axis), np.moveaxis(spline, 0, self.axis)
        return np.moveaxis(D, 0, self.axis)


@dataclass
class DiffPlan2D:
    """Two-axis plan for repeated derivatives with fixed (order, lam, BCs)."""
    x_plan: _AxisPlan
    y_plan: _AxisPlan

    def dx(self, F: Array, *, flux=(0.0, 0.0), return_spline=False):
        return self.x_plan.apply(F, flux=flux, return_spline=return_spline)

    def dy(self, F: Array, *, flux=(0.0, 0.0), return_spline=False):
        return self.y_plan.apply(F, flux=flux, return_spline=return_spline)


# =====================================================================
# 2D facade
# =====================================================================
@dataclass
class bspf2d:
    """
    Vectorized 2D facade composed from two bspf1d models.

    Data layout
    -----------
    F has shape (ny, nx) with:
      - axis 0 ≡ y (rows)
      - axis 1 ≡ x (columns)

    Axis mapping
    ------------
    - partial_dx acts along axis=1 (x).
    - partial_dy acts along axis=0 (y).

    Also provides:
    - partial_dxx, partial_dyy (2nd order)
    - partial_dxy (mixed) with optional symmetrization
    - *_neumann variants
    - hessian() and laplacian() helpers
    """
    x: Array           # (nx,)
    y: Array           # (ny,)
    x_model: bspf1d   # acts along axis=1 (x)
    y_model: bspf1d   # acts along axis=0 (y)

    # ---------- construction ----------
    @classmethod
    def from_grids(
        cls,
        *,
        x: Array,
        y: Array,
        degree_x: int = 10,
        degree_y: Optional[int] = None,
        knots_x: Optional[Array] = None, knots_y: Optional[Array] = None,
        n_basis_x: Optional[int] = None, n_basis_y: Optional[int] = None,
        domain_x: Optional[Tuple[float, float]] = None, domain_y: Optional[Tuple[float, float]] = None,
        use_clustering_x: bool = False, use_clustering_y: bool = False,
        order_x: Optional[int] = None, order_y: Optional[int] = None,
        num_boundary_points_x: Optional[int] = None, num_boundary_points_y: Optional[int] = None,
        correction: str = "spectral",
    ) -> "bspf2d":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if degree_y is None:
            degree_y = degree_x
        xm = bspf1d.from_grid(
            degree=degree_x, x=x, knots=knots_x, n_basis=n_basis_x, domain=domain_x,
            use_clustering=use_clustering_x, order=order_x, num_boundary_points=num_boundary_points_x,
            correction=correction,
        )
        ym = bspf1d.from_grid(
            degree=degree_y, x=y, knots=knots_y, n_basis=n_basis_y, domain=domain_y,
            use_clustering=use_clustering_y, order=order_y, num_boundary_points=num_boundary_points_y,
            correction=correction,
        )
        return cls(x=x, y=y, x_model=xm, y_model=ym)

    # ---------- init cache ----------
    def __post_init__(self):
        # cache for precomputed plans: key -> _AxisPlan
        self._plan_cache: Dict[Tuple[str, int, float, bool], _AxisPlan] = {}

    # ---------- shape guard ----------
    def _check_shape(self, F: Array) -> Tuple[int, int]:
        if F.ndim != 2:
            raise ValueError("F must be 2D (ny, nx).")
        ny, nx = F.shape
        if ny != self.y.size or nx != self.x.size:
            raise ValueError(f"F shape {F.shape} must match (len(y), len(x))=({self.y.size},{self.x.size}).")
        return ny, nx

    # ---------- axis-generic kernels (on-the-fly path) ----------
    @staticmethod
    def _diff_axis(F: Array, model: bspf1d, *, lam: float, k: int, axis: int, return_spline: bool):
        """
        Vectorized derivative of order k along `axis` using a single bspf1d model.
        """
        # Move working axis to front -> (n, batch)
        FT = np.moveaxis(F, axis, 0)
        n, batch = FT.shape

        BW, BND, BT = model.BW, model.end.BND, model.basis.BT0     # (n_b,n), (m,n), (n,n_b)
        BkT = model.basis.BkT(k)                                   # (n,n_b)
        om  = model.grid.omega                                     # (n//2+1,)
        n_b = BW.shape[0]

        # Multi-RHS KKT solve (LU is cached inside the model)
        RHS = np.vstack([2.0 * (BW @ FT), BND @ FT])               # (n_b+m, batch)
        lu, piv = model._kkt_lu(lam)
        SOL = sla.lu_solve((lu, piv), RHS)                         # (n_b+m, batch)
        P   = SOL[:n_b, :]                                         # (n_b, batch)

        # Spline part + spectral correction along axis 0
        spline = BT @ P                                            # (n, batch)
        deriv  = BkT @ P                                           # (n, batch)
        resid  = FT - spline
        R      = np.fft.rfft(resid, axis=0)                        # (n//2+1, batch)
        corr   = np.fft.irfft(R * (1j * om)[:, None]**k, n=n, axis=0)

        D = deriv + corr
        if return_spline:
            return np.moveaxis(D, 0, axis), np.moveaxis(spline, 0, axis)
        return np.moveaxis(D, 0, axis)

    @staticmethod
    def _broadcast_flux(val, batch: int) -> np.ndarray:
        v = np.asarray(val, dtype=np.float64)
        if v.ndim == 0:
            return np.full(batch, float(v))
        if v.shape == (batch,):
            return v
        raise ValueError(f"Flux must be scalar or shape=({batch},), got {v.shape}.")

    @staticmethod
    def _diff_axis_neumann(
        F: Array,
        model: bspf1d,
        *,
        lam: float,
        k: int,
        axis: int,
        flux: Tuple[float | Array, float | Array],   # (left_flux, right_flux)
        return_spline: bool,
    ):
        """
        Vectorized derivative of order k along `axis` while enforcing
        Neumann (first-derivative) boundary flux on the underlying spline fit.
        """
        # Move working axis to front -> (n, batch)
        FT = np.moveaxis(F, axis, 0)
        n, batch = FT.shape

        BW, BND, BT = model.BW, model.end.BND, model.basis.BT0     # (n_b,n), (m,n), (n,n_b)
        BkT = model.basis.BkT(k)                                   # (n,n_b)
        om  = model.grid.omega
        n_b = BW.shape[0]
        m   = BND.shape[0]
        ord_ = model.end.order
        if ord_ < 2:
            raise ValueError("Model 'order' must be >=2 to enforce first-derivative Neumann flux.")

        # Build constraint RHS and override d/dx rows with flux
        dY = BND @ FT                                              # (m, batch)
        left_row  = 1              # first derivative @ left
        right_row = ord_ + 1       # first derivative @ right
        left_flux  = bspf2d._broadcast_flux(flux[0], batch)
        right_flux = bspf2d._broadcast_flux(flux[1], batch)
        dY[left_row,  :] = left_flux
        dY[right_row, :] = right_flux

        RHS = np.vstack([2.0 * (BW @ FT), dY])                    # (n_b+m, batch)
        lu, piv = model._kkt_lu(lam)
        SOL = sla.lu_solve((lu, piv), RHS)                        # (n_b+m, batch)
        P   = SOL[:n_b, :]

        # Spline part + spectral correction
        spline = BT @ P                                           # (n, batch)
        deriv  = BkT @ P
        resid  = FT - spline
        R      = np.fft.rfft(resid, axis=0)
        corr   = np.fft.irfft(R * (1j * om)[:, None]**k, n=n, axis=0)

        D = deriv + corr

        # For k==1, pin endpoint derivative samples to requested flux
        if k == 1:
            D[0,  :] = left_flux
            D[-1, :] = right_flux

        if return_spline:
            return np.moveaxis(D, 0, axis), np.moveaxis(spline, 0, axis)
        return np.moveaxis(D, 0, axis)

    # ---------- precomputed plan builders ----------
    def make_plan_dx(self, *, order: int = 1, lam: float = 0.0, neumann: bool = False) -> _AxisPlan:
        """Precompute a derivative plan along x (axis=1)."""
        key = ('x', order, float(lam), bool(neumann))
        plan = self._plan_cache.get(key)
        if plan is None:
            plan = _AxisPlan(model=self.x_model, axis=1, order=order, lam=lam, neumann=neumann)
            self._plan_cache[key] = plan
        return plan

    def make_plan_dy(self, *, order: int = 1, lam: float = 0.0, neumann: bool = False) -> _AxisPlan:
        """Precompute a derivative plan along y (axis=0)."""
        key = ('y', order, float(lam), bool(neumann))
        plan = self._plan_cache.get(key)
        if plan is None:
            plan = _AxisPlan(model=self.y_model, axis=0, order=order, lam=lam, neumann=neumann)
            self._plan_cache[key] = plan
        return plan

    def make_plan_pair(
        self,
        *,
        order_x: int = 1, lam_x: float = 0.0, neumann_x: bool = False,
        order_y: int = 1, lam_y: float = 0.0, neumann_y: bool = False,
    ) -> DiffPlan2D:
        """Build a two-axis plan in one go."""
        return DiffPlan2D(
            x_plan=self.make_plan_dx(order=order_x, lam=lam_x, neumann=neumann_x),
            y_plan=self.make_plan_dy(order=order_y, lam=lam_y, neumann=neumann_y),
        )

    # ---------- public API (on-the-fly path) ----------
    def partial_dx(self, F: Array, *, order: int = 1, lam: float = 0.0, return_spline: bool = False):
        """∂^order F / ∂x^order  (x ≡ axis=1)"""
        self._check_shape(F)
        return self._diff_axis(F, self.x_model, lam=lam, k=order, axis=1, return_spline=return_spline)

    def partial_dy(self, F: Array, *, order: int = 1, lam: float = 0.0, return_spline: bool = False):
        """∂^order F / ∂y^order  (y ≡ axis=0)"""
        self._check_shape(F)
        return self._diff_axis(F, self.y_model, lam=lam, k=order, axis=0, return_spline=return_spline)

    # ---- second-order partials ----
    def partial_dxx(self, F: Array, *, lam: float = 0.0) -> Array:
        """Fxx = ∂²F/∂x²"""
        return self.partial_dx(F, order=2, lam=lam)

    def partial_dyy(self, F: Array, *, lam: float = 0.0) -> Array:
        """Fyy = ∂²F/∂y²"""
        return self.partial_dy(F, order=2, lam=lam)

    # ---- mixed partial ----
    def partial_dxy(self, F: Array, *, lam_x: float = 0.0, lam_y: float = 0.0, symmetrize: bool = True) -> Array:
        """
        Fxy ≈ ∂y(∂x F). If symmetrize=True, returns 0.5*(∂y∂x F + ∂x∂y F)
        to reduce small asymmetries from separate 1D residual corrections.
        """
        self._check_shape(F)

        # ∂y(∂x F)
        dFx = self._diff_axis(F, self.x_model, lam=lam_x, k=1, axis=1, return_spline=False)
        dxy = self._diff_axis(dFx, self.y_model, lam=lam_y, k=1, axis=0, return_spline=False)

        if not symmetrize:
            return dxy.astype(np.float64)

        # ∂x(∂y F)
        dFy = self._diff_axis(F, self.y_model, lam=lam_y, k=1, axis=0, return_spline=False)
        dyx = self._diff_axis(dFy, self.x_model, lam=lam_x, k=1, axis=1, return_spline=False)

        return (0.5 * (dxy + dyx)).astype(np.float64)

    # ---- Neumann-enforced variants ----
    def partial_dx_neumann(
        self,
        F: Array,
        *,
        order: int = 1,
        lam: float = 0.0,
        flux: Tuple[float | Array, float | Array] = (0.0, 0.0),  # (left, right) per row or scalar
        return_spline: bool = False,
    ):
        """∂^order F/∂x^order with Neumann flux enforced at x-ends (x ≡ axis=1).

        flux can be a pair of scalars, or two arrays of shape (ny,) to vary along y.
        """
        self._check_shape(F)
        return self._diff_axis_neumann(
            F, self.x_model, lam=lam, k=order, axis=1, flux=flux, return_spline=return_spline
        )

    def partial_dy_neumann(
        self,
        F: Array,
        *,
        order: int = 1,
        lam: float = 0.0,
        flux: Tuple[float | Array, float | Array] = (0.0, 0.0),  # (bottom, top) per column or scalar
        return_spline: bool = False,
    ):
        """∂^order F/∂y^order with Neumann flux enforced at y-ends (y ≡ axis=0).

        flux can be a pair of scalars, or two arrays of shape (nx,) to vary along x.
        """
        self._check_shape(F)
        return self._diff_axis_neumann(
            F, self.y_model, lam=lam, k=order, axis=0, flux=flux, return_spline=return_spline
        )

    def partial_dxx_neumann(
        self,
        F: Array,
        *,
        lam: float = 0.0,
        flux: Tuple[float | Array, float | Array] = (0.0, 0.0),
        return_spline: bool = False,
    ):
        """Fxx = ∂²F/∂x² with Neumann (first-derivative) flux enforced at x-ends.

        `flux` can be two scalars or two arrays of shape (ny,) to vary along y.
        """
        return self.partial_dx_neumann(F, order=2, lam=lam, flux=flux, return_spline=return_spline)

    def partial_dyy_neumann(
        self,
        F: Array,
        *,
        lam: float = 0.0,
        flux: Tuple[float | Array, float | Array] = (0.0, 0.0),
        return_spline: bool = False,
    ):
        """Fyy = ∂²F/∂y² with Neumann (first-derivative) flux enforced at y-ends.

        `flux` can be two scalars or two arrays of shape (nx,) to vary along x.
        """
        return self.partial_dy_neumann(F, order=2, lam=lam, flux=flux, return_spline=return_spline)

    def laplacian_neumann(
        self,
        F: Array,
        *,
        lam_x: float = 0.0,
        lam_y: float = 0.0,
        flux_x: Tuple[float | Array, float | Array] = (0.0, 0.0),
        flux_y: Tuple[float | Array, float | Array] = (0.0, 0.0),
    ) -> Array:
        """Return ∆F using Neumann-enforced second derivatives along each axis."""
        Fxx = self.partial_dxx_neumann(F, lam=lam_x, flux=flux_x)
        Fyy = self.partial_dyy_neumann(F, lam=lam_y, flux=flux_y)
        return (Fxx + Fyy).astype(np.float64)

    # ---- convenience: Hessian & Laplacian ----
    def hessian(self, F: Array, *, lam_x: float = 0.0, lam_y: float = 0.0, symmetrize_xy: bool = True):
        """Return (Fxx, Fxy, Fyy)."""
        Fxx = self.partial_dxx(F, lam=lam_x)
        Fyy = self.partial_dyy(F, lam=lam_y)
        Fxy = self.partial_dxy(F, lam_x=lam_x, lam_y=lam_y, symmetrize=symmetrize_xy)
        return Fxx, Fxy, Fyy

    def laplacian(self, F: Array, *, lam_x: float = 0.0, lam_y: float = 0.0) -> Array:
        """Return ∆F = Fxx + Fyy."""
        return (self.partial_dxx(F, lam=lam_x) + self.partial_dyy(F, lam=lam_y)).astype(np.float64)
