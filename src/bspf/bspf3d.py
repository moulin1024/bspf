# =====================================================================
# 3D facade (differentiation only)
# Data layout: F has shape (nz, ny, nx)
# Axis mapping:
#   - x ≡ axis=2 (columns)
#   - y ≡ axis=1 (rows)
#   - z ≡ axis=0 (depth)
# =====================================================================
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

import numpy as np
from scipy import linalg as sla  # CPU fallback
from bspf import bspf1d

# Optional GPU backend
_HAS_CUPY = False
try:
    import cupy as cp
    import cupyx.scipy.linalg as cpla
    _HAS_CUPY = True
except Exception:
    cp = None
    cpla = None

Array = np.ndarray


# -------------------------- backend helpers --------------------------
class _Backend:
    """Tiny dispatch wrapper to switch between NumPy/SciPy and CuPy/CuPyX."""
    __slots__ = ("xp", "la", "fft", "is_gpu")

    def __init__(self, use_gpu: bool):
        if use_gpu:
            if not _HAS_CUPY:
                raise RuntimeError(
                    "use_gpu=True but CuPy is not available. "
                    "Install cupy (e.g., `pip install cupy-cuda12x`) or set use_gpu=False."
                )
            self.xp = cp
            self.la = cpla
            self.fft = cp.fft
            self.is_gpu = True
        else:
            self.xp = np
            self.la = sla
            self.fft = np.fft
            self.is_gpu = False

    def to_device(self, a):
        return self.xp.asarray(a)

    def to_host(self, a):
        if self.is_gpu:
            return cp.asnumpy(a)
        return a

    def ensure_like_input(self, out, input_was_numpy: bool):
        if self.is_gpu and input_was_numpy:
            return self.to_host(out)
        return out


# -------------------------- 3D Axis Plan --------------------------
@dataclass
class _AxisPlan3D:
    """Precomputed derivative plan along one axis for repeated 3D calls."""
    model: bspf1d
    axis: int                  # 2 for x, 1 for y, 0 for z
    order: int                 # derivative order k
    lam: float
    neumann: bool = False      # enforce first-derivative flux at the ends?

    # GPU/CPU selection
    use_gpu: bool = False

    # Uniform BC support: use one boundary RHS vector, broadcast across bundle
    uniform_bc: bool = False
    bc: float | Array | None = None  # scalar or (m,)

    # cached handles / constants (filled in __post_init__)
    BW: any = field(init=False)
    BND: any = field(init=False)
    BT: any = field(init=False)
    BkT: any = field(init=False)
    omega: any = field(init=False)
    n_b: int = field(init=False)
    m: int = field(init=False)
    n: int = field(init=False)
    left_row: int = field(init=False)
    right_row: int = field(init=False)
    lu: any = field(init=False)
    piv: any = field(init=False)
    _bc_vec: Optional[any] = field(init=False, default=None)

    # backend
    _bk: _Backend = field(init=False, repr=False)

    def __post_init__(self):
        self._bk = _Backend(self.use_gpu)
        xp = self._bk.xp

        # Move static mats to device/host backend
        self.BW  = xp.asarray(self.model.BW)
        self.BND = xp.asarray(self.model.end.BND)
        self.BT  = xp.asarray(self.model.basis.BT0)
        self.BkT = xp.asarray(self.model.basis.BkT(self.order))
        self.omega = xp.asarray(self.model.grid.omega)

        self.n_b = int(self.BW.shape[0])
        self.m   = int(self.BND.shape[0])
        self.n   = int(self.model.grid.n)

        ord_ = self.model.end.order
        if self.neumann:
            if ord_ < 2:
                raise ValueError("Model 'order' must be >= 2 to enforce Neumann flux.")
            self.left_row  = 1
            self.right_row = ord_ + 1
        else:
            self.left_row = self.right_row = -1

        # Precompute LU on CPU, copy to device if needed
        lu_cpu, piv_cpu = self.model._kkt_lu(self.lam)
        if self._bk.is_gpu:
            self.lu  = xp.asarray(lu_cpu)
            self.piv = xp.asarray(piv_cpu)
        else:
            self.lu, self.piv = lu_cpu, piv_cpu

        # Prepare uniform BC vector if requested
        if self.uniform_bc:
            if self.bc is None:
                self._bc_vec = xp.zeros(self.m, dtype=xp.float64)
            else:
                v = xp.asarray(self.bc, dtype=xp.float64)
                if v.ndim == 0:
                    self._bc_vec = xp.full(self.m, float(v))
                elif v.shape == (self.m,):
                    self._bc_vec = v
                else:
                    raise ValueError(f"'bc' must be scalar or shape=({self.m},), got {tuple(v.shape)}.")
        else:
            self._bc_vec = None

    def _broadcast_flux(self, val, other_shape: Tuple[int, int]) -> any:
        """Flux can be scalar or array shaped like the two non-working dims."""
        xp = self._bk.xp
        batch = other_shape[0] * other_shape[1]
        v = xp.asarray(val, dtype=xp.float64)
        if v.ndim == 0:
            return xp.full(batch, float(v))
        if v.size == batch:
            return v.reshape(batch)
        if v.shape == other_shape:
            return v.reshape(batch)
        raise ValueError(
            f"Flux must be scalar or have size matching product of other dims "
            f"{other_shape} (={batch}), got {tuple(v.shape)}."
        )

    def apply(self, F: Array, *, flux: Tuple[float | Array, float | Array] = (0.0, 0.0),
              return_spline: bool = False):
        """
        Compute ∂^order(F)/∂axis^order for 3D field F (nz, ny, nx).
        Accepts Neumann flux overrides; flux may be scalar or shaped like the two non-working dims.
        """
        xp, la, fft = self._bk.xp, self._bk.la, self._bk.fft
        # Bring working axis to front -> (n, d1, d2)
        FT = xp.moveaxis(xp.asarray(F, dtype=xp.float64), self.axis, 0)
        n, d1, d2 = FT.shape
        batch = d1 * d2
        FT2 = FT.reshape(n, batch)

        # Build RHS
        rhs_top = 2.0 * (self.BW @ FT2)  # (n_b, batch)
        if self.uniform_bc:
            dY = xp.repeat(self._bc_vec[:, None], batch, axis=1)
        else:
            dY = self.BND @ FT2

        if self.neumann:
            lf = self._broadcast_flux(flux[0], (d1, d2))
            rf = self._broadcast_flux(flux[1], (d1, d2))
            dY[self.left_row,  :] = lf
            dY[self.right_row, :] = rf

        RHS = xp.vstack([rhs_top, dY])  # (n_b+m, batch)

        # Solve (device or host depending on backend)
        SOL = la.lu_solve((self.lu, self.piv), RHS)
        P   = SOL[: self.n_b, :]

        spline = self.BT @ P                    # (n, batch)
        deriv  = self.BkT @ P

        resid = FT2 - spline
        R     = fft.rfft(resid, axis=0)
        corr  = fft.irfft(R * (1j * self.omega)[:, None]**self.order, n=n, axis=0)

        D2 = deriv + corr

        if self.neumann and self.order == 1:
            lf = dY[self.left_row,  :]
            rf = dY[self.right_row, :]
            D2[0,  :] = lf
            D2[-1, :] = rf

        # reshape back to (n, d1, d2) -> move axis back
        D = xp.moveaxis(D2.reshape(n, d1, d2), 0, self.axis)
        S = xp.moveaxis(spline.reshape(n, d1, d2), 0, self.axis)

        if return_spline:
            return D, S
        return D


# -------------------------- 3D Plan wrapper --------------------------
@dataclass
class DiffPlan3D:
    """Three-axis plan for repeated derivatives with fixed (order, lam, BCs)."""
    x_plan: _AxisPlan3D
    y_plan: _AxisPlan3D
    z_plan: _AxisPlan3D

    def dx(self, F: Array, *, flux=(0.0, 0.0), return_spline=False):
        return self.x_plan.apply(F, flux=flux, return_spline=return_spline)

    def dy(self, F: Array, *, flux=(0.0, 0.0), return_spline=False):
        return self.y_plan.apply(F, flux=flux, return_spline=return_spline)

    def dz(self, F: Array, *, flux=(0.0, 0.0), return_spline=False):
        return self.z_plan.apply(F, flux=flux, return_spline=return_spline)


# -------------------------- 3D facade --------------------------
@dataclass
class bspf3d:
    """
    3D facade composed from three bspf1d models.

    Data layout
    -----------
    F has shape (nz, ny, nx) with:
      - axis 2 ≡ x (columns)
      - axis 1 ≡ y (rows)
      - axis 0 ≡ z (depth)

    Only differentiation is provided (first/second order) with optional Neumann BC
    and uniform boundary RHS acceleration. GPU is supported via `use_gpu=True`.
    """
    x: Array           # (nx,)
    y: Array           # (ny,)
    z: Array           # (nz,)
    x_model: bspf1d    # acts along axis=2 (x)
    y_model: bspf1d    # acts along axis=1 (y)
    z_model: bspf1d    # acts along axis=0 (z)
    use_gpu: bool = False

    # ---------- construction ----------
    @classmethod
    def from_grids(
        cls,
        *,
        x: Array,
        y: Array,
        z: Array,
        degree_x: int = 10,
        degree_y: Optional[int] = None,
        degree_z: Optional[int] = None,
        knots_x: Optional[Array] = None, knots_y: Optional[Array] = None, knots_z: Optional[Array] = None,
        n_basis_x: Optional[int] = None, n_basis_y: Optional[int] = None, n_basis_z: Optional[int] = None,
        domain_x: Optional[Tuple[float, float]] = None, domain_y: Optional[Tuple[float, float]] = None, domain_z: Optional[Tuple[float, float]] = None,
        use_clustering_x: bool = False, use_clustering_y: bool = False, use_clustering_z: bool = False,
        order_x: Optional[int] = None, order_y: Optional[int] = None, order_z: Optional[int] = None,
        num_boundary_points_x: Optional[int] = None, num_boundary_points_y: Optional[int] = None, num_boundary_points_z: Optional[int] = None,
        correction: str = "spectral",
        use_gpu: bool = False,
    ) -> "bspf3d":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        if degree_y is None: degree_y = degree_x
        if degree_z is None: degree_z = degree_x
        xm = bspf1d.from_grid(
            degree=degree_x, x=x, knots=knots_x, n_basis=n_basis_x, domain=domain_x,
            use_clustering=use_clustering_x, order=order_x, num_boundary_points=num_boundary_points_x,
            correction=correction, use_gpu=use_gpu,
        )
        ym = bspf1d.from_grid(
            degree=degree_y, x=y, knots=knots_y, n_basis=n_basis_y, domain=domain_y,
            use_clustering=use_clustering_y, order=order_y, num_boundary_points=num_boundary_points_y,
            correction=correction, use_gpu=use_gpu,
        )
        zm = bspf1d.from_grid(
            degree=degree_z, x=z, knots=knots_z, n_basis=n_basis_z, domain=domain_z,
            use_clustering=use_clustering_z, order=order_z, num_boundary_points=num_boundary_points_z,
            correction=correction, use_gpu=use_gpu,
        )
        return cls(x=x, y=y, z=z, x_model=xm, y_model=ym, z_model=zm, use_gpu=use_gpu)

    # ---------- init cache ----------
    def __post_init__(self):
        # cache for precomputed 3D plans: key -> _AxisPlan3D
        self._plan_cache: Dict[
            Tuple[str, int, float, bool, bool, Tuple[float, ...] | None, bool], _AxisPlan3D
        ] = {}

    # ---------- shape guard ----------
    def _check_shape(self, F: Array) -> Tuple[int, int, int]:
        if F.ndim != 3:
            raise ValueError("F must be 3D (nz, ny, nx).")
        nz, ny, nx = F.shape
        if (nz != self.z.size) or (ny != self.y.size) or (nx != self.x.size):
            raise ValueError(
                f"F shape {F.shape} must match (len(z), len(y), len(x))=({self.z.size},{self.y.size},{self.x.size})."
            )
        return nz, ny, nx

    # ---------- helpers: uniform BC key ----------
    @staticmethod
    def _prepare_bc_vector(model: bspf1d, bc: float | Array | None) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, ...]]]:
        if bc is None:
            return None, None
        m = model.end.BND.shape[0]
        v = np.asarray(bc, dtype=np.float64)
        if v.ndim == 0:
            vec = np.full(m, float(v))
            return vec, tuple(vec.tolist())
        if v.shape == (m,):
            return v, tuple(v.tolist())
        raise ValueError(f"'bc' must be scalar or shape=({m},), got {v.shape}.")

    # ---------- on-the-fly kernels (matrices moved each call) ----------
    @staticmethod
    def _diff_axis_3d(
        F: Array,
        model: bspf1d,
        *,
        lam: float,
        k: int,
        axis: int,
        return_spline: bool,
        uniform_bc: bool = False,
        bc: float | Array | None = None,
        use_gpu: bool = False,
    ):
        bk = _Backend(use_gpu)
        xp, la, fft = bk.xp, bk.la, bk.fft

        FT = xp.moveaxis(xp.asarray(F, dtype=xp.float64), axis, 0)  # (n, d1, d2)
        n, d1, d2 = FT.shape
        batch = d1 * d2
        FT2 = FT.reshape(n, batch)

        BW  = xp.asarray(model.BW)
        BND = xp.asarray(model.end.BND)
        BT  = xp.asarray(model.basis.BT0)
        BkT = xp.asarray(model.basis.BkT(k))
        om  = xp.asarray(model.grid.omega)
        m   = BND.shape[0]

        if uniform_bc:
            if bc is None:
                dY = xp.zeros((m, batch), dtype=xp.float64)
            else:
                v = xp.asarray(bc, dtype=xp.float64)
                if v.ndim == 0:
                    v = xp.full(m, float(v))
                elif v.shape != (m,):
                    raise ValueError(f"'bc' must be scalar or shape=({m},), got {tuple(v.shape)}.")
                dY = xp.repeat(v[:, None], batch, axis=1)
        else:
            dY = BND @ FT2

        RHS = xp.vstack([2.0 * (BW @ FT2), dY])

        lu_cpu, piv_cpu = model._kkt_lu(lam)
        if bk.is_gpu:
            SOL = la.lu_solve((xp.asarray(lu_cpu), xp.asarray(piv_cpu)), RHS)
        else:
            SOL = la.lu_solve((lu_cpu, piv_cpu), bk.to_host(RHS))
            SOL = xp.asarray(SOL)

        n_b = BW.shape[0]
        P   = SOL[:n_b, :]

        spline = BT @ P
        deriv  = BkT @ P
        resid  = FT2 - spline
        R      = fft.rfft(resid, axis=0)
        corr   = fft.irfft(R * (1j * om)[:, None]**k, n=n, axis=0)

        D2 = deriv + corr
        # back to 3D & original axis
        D = xp.moveaxis(D2.reshape(n, d1, d2), 0, axis)
        S = xp.moveaxis(spline.reshape(n, d1, d2), 0, axis)
        if return_spline:
            return D, S
        return D

    @staticmethod
    def _broadcast_flux_any(bk: _Backend, val, other_shape: Tuple[int, int]):
        xp = bk.xp
        batch = other_shape[0] * other_shape[1]
        v = xp.asarray(val, dtype=xp.float64)
        if v.ndim == 0:
            return xp.full(batch, float(v))
        if v.size == batch:
            return v.reshape(batch)
        if v.shape == other_shape:
            return v.reshape(batch)
        raise ValueError(
            f"Flux must be scalar or have size matching product of other dims "
            f"{other_shape} (={batch}), got {tuple(v.shape)}."
        )

    @staticmethod
    def _diff_axis_neumann_3d(
        F: Array,
        model: bspf1d,
        *,
        lam: float,
        k: int,
        axis: int,
        flux: Tuple[float | Array, float | Array],
        return_spline: bool,
        uniform_bc: bool = False,
        bc: float | Array | None = None,
        use_gpu: bool = False,
    ):
        bk = _Backend(use_gpu)
        xp, la, fft = bk.xp, bk.la, bk.fft

        FT = xp.moveaxis(xp.asarray(F, dtype=xp.float64), axis, 0)  # (n, d1, d2)
        n, d1, d2 = FT.shape
        batch = d1 * d2
        FT2 = FT.reshape(n, batch)

        BW  = xp.asarray(model.BW)
        BND = xp.asarray(model.end.BND)
        BT  = xp.asarray(model.basis.BT0)
        BkT = xp.asarray(model.basis.BkT(k))
        om  = xp.asarray(model.grid.omega)
        m   = BND.shape[0]
        ord_ = model.end.order
        if ord_ < 2:
            raise ValueError("Model 'order' must be >=2 to enforce first-derivative Neumann flux.")

        if uniform_bc:
            if bc is None:
                dY = xp.zeros((m, batch), dtype=xp.float64)
            else:
                v = xp.asarray(bc, dtype=xp.float64)
                if v.ndim == 0:
                    v = xp.full(m, float(v))
                elif v.shape != (m,):
                    raise ValueError(f"'bc' must be scalar or shape=({m},), got {tuple(v.shape)}.")
                dY = xp.repeat(v[:, None], batch, axis=1)
        else:
            dY = BND @ FT2

        left_row  = 1
        right_row = ord_ + 1
        left_flux  = bspf3d._broadcast_flux_any(bk, flux[0], (d1, d2))
        right_flux = bspf3d._broadcast_flux_any(bk, flux[1], (d1, d2))
        dY[left_row,  :] = left_flux
        dY[right_row, :] = right_flux

        RHS = xp.vstack([2.0 * (BW @ FT2), dY])
        lu_cpu, piv_cpu = model._kkt_lu(lam)
        if bk.is_gpu:
            SOL = la.lu_solve((xp.asarray(lu_cpu), xp.asarray(piv_cpu)), RHS)
        else:
            SOL = la.lu_solve((lu_cpu, piv_cpu), bk.to_host(RHS))
            SOL = xp.asarray(SOL)

        n_b = BW.shape[0]
        P   = SOL[:n_b, :]

        spline = BT @ P
        deriv  = BkT @ P
        resid  = FT2 - spline
        R      = fft.rfft(resid, axis=0)
        corr   = fft.irfft(R * (1j * om)[:, None]**k, n=n, axis=0)

        D2 = deriv + corr
        if k == 1:
            D2[0,  :] = left_flux
            D2[-1, :] = right_flux

        D = xp.moveaxis(D2.reshape(n, d1, d2), 0, axis)
        S = xp.moveaxis(spline.reshape(n, d1, d2), 0, axis)
        if return_spline:
            return D, S
        return D

    # ---------- plan builders ----------
    def make_plan_dx(
        self,
        *,
        order: int = 1,
        lam: float = 0.0,
        neumann: bool = False,
        uniform_bc: bool = False,
        bc: float | Array | None = None,
    ) -> _AxisPlan3D:
        bc_vec, bc_key = (None, None)
        if uniform_bc:
            bc_vec, bc_key = self._prepare_bc_vector(self.x_model, bc)
        key = ('x', order, float(lam), bool(neumann), bool(uniform_bc), bc_key, bool(self.use_gpu))
        plan = self._plan_cache.get(key)
        if plan is None:
            plan = _AxisPlan3D(
                model=self.x_model, axis=2, order=order, lam=lam,
                neumann=neumann, use_gpu=self.use_gpu, uniform_bc=uniform_bc, bc=bc_vec
            )
            self._plan_cache[key] = plan
        return plan

    def make_plan_dy(
        self,
        *,
        order: int = 1,
        lam: float = 0.0,
        neumann: bool = False,
        uniform_bc: bool = False,
        bc: float | Array | None = None,
    ) -> _AxisPlan3D:
        bc_vec, bc_key = (None, None)
        if uniform_bc:
            bc_vec, bc_key = self._prepare_bc_vector(self.y_model, bc)
        key = ('y', order, float(lam), bool(neumann), bool(uniform_bc), bc_key, bool(self.use_gpu))
        plan = self._plan_cache.get(key)
        if plan is None:
            plan = _AxisPlan3D(
                model=self.y_model, axis=1, order=order, lam=lam,
                neumann=neumann, use_gpu=self.use_gpu, uniform_bc=uniform_bc, bc=bc_vec
            )
            self._plan_cache[key] = plan
        return plan

    def make_plan_dz(
        self,
        *,
        order: int = 1,
        lam: float = 0.0,
        neumann: bool = False,
        uniform_bc: bool = False,
        bc: float | Array | None = None,
    ) -> _AxisPlan3D:
        bc_vec, bc_key = (None, None)
        if uniform_bc:
            bc_vec, bc_key = self._prepare_bc_vector(self.z_model, bc)
        key = ('z', order, float(lam), bool(neumann), bool(uniform_bc), bc_key, bool(self.use_gpu))
        plan = self._plan_cache.get(key)
        if plan is None:
            plan = _AxisPlan3D(
                model=self.z_model, axis=0, order=order, lam=lam,
                neumann=neumann, use_gpu=self.use_gpu, uniform_bc=uniform_bc, bc=bc_vec
            )
            self._plan_cache[key] = plan
        return plan

    def make_plan_triple(
        self,
        *,
        order_x: int = 1, lam_x: float = 0.0, neumann_x: bool = False, uniform_bc_x: bool = False, bc_x: float | Array | None = None,
        order_y: int = 1, lam_y: float = 0.0, neumann_y: bool = False, uniform_bc_y: bool = False, bc_y: float | Array | None = None,
        order_z: int = 1, lam_z: float = 0.0, neumann_z: bool = False, uniform_bc_z: bool = False, bc_z: float | Array | None = None,
    ) -> DiffPlan3D:
        return DiffPlan3D(
            x_plan=self.make_plan_dx(order=order_x, lam=lam_x, neumann=neumann_x, uniform_bc=uniform_bc_x, bc=bc_x),
            y_plan=self.make_plan_dy(order=order_y, lam=lam_y, neumann=neumann_y, uniform_bc=uniform_bc_y, bc=bc_y),
            z_plan=self.make_plan_dz(order=order_z, lam=lam_z, neumann=neumann_z, uniform_bc=uniform_bc_z, bc=bc_z),
        )

    # ---------- public API (on-the-fly path) ----------
    def partial_dx(
        self,
        F: Array,
        *,
        order: int = 1,
        lam: float = 0.0,
        return_spline: bool = False,
        uniform_bc: bool = False,
        bc: float | Array | None = None,
    ):
        self._check_shape(F)
        return self._diff_axis_3d(
            F, self.x_model, lam=lam, k=order, axis=2, return_spline=return_spline,
            uniform_bc=uniform_bc, bc=bc, use_gpu=self.use_gpu
        )

    def partial_dy(
        self,
        F: Array,
        *,
        order: int = 1,
        lam: float = 0.0,
        return_spline: bool = False,
        uniform_bc: bool = False,
        bc: float | Array | None = None,
    ):
        self._check_shape(F)
        return self._diff_axis_3d(
            F, self.y_model, lam=lam, k=order, axis=1, return_spline=return_spline,
            uniform_bc=uniform_bc, bc=bc, use_gpu=self.use_gpu
        )

    def partial_dz(
        self,
        F: Array,
        *,
        order: int = 1,
        lam: float = 0.0,
        return_spline: bool = False,
        uniform_bc: bool = False,
        bc: float | Array | None = None,
    ):
        self._check_shape(F)
        return self._diff_axis_3d(
            F, self.z_model, lam=lam, k=order, axis=0, return_spline=return_spline,
            uniform_bc=uniform_bc, bc=bc, use_gpu=self.use_gpu
        )

    # ---- second-order convenience ----
    def partial_dxx(self, F: Array, *, lam: float = 0.0, uniform_bc: bool = False, bc: float | Array | None = None):
        return self.partial_dx(F, order=2, lam=lam, uniform_bc=uniform_bc, bc=bc)

    def partial_dyy(self, F: Array, *, lam: float = 0.0, uniform_bc: bool = False, bc: float | Array | None = None):
        return self.partial_dy(F, order=2, lam=lam, uniform_bc=uniform_bc, bc=bc)

    def partial_dzz(self, F: Array, *, lam: float = 0.0, uniform_bc: bool = False, bc: float | Array | None = None):
        return self.partial_dz(F, order=2, lam=lam, uniform_bc=uniform_bc, bc=bc)

    # ---- compute first and second derivatives together (efficient) ----
    def differentiate_1_2(
        self,
        F: Array,
        *,
        lam_x: float = 0.0,
        lam_y: float = 0.0,
        lam_z: float = 0.0,
        uniform_bc_x: bool = False,
        uniform_bc_y: bool = False,
        uniform_bc_z: bool = False,
        bc_x: float | Array | None = None,
        bc_y: float | Array | None = None,
        bc_z: float | Array | None = None,
        use_loop: bool = False,
    ) -> Tuple[Array, Array, Array, Array, Array, Array]:
        """
        Compute first and second derivatives in x, y, and z directions together.
        More efficient than calling partial_dx, partial_dy, partial_dz, partial_dxx, partial_dyy, partial_dzz separately
        because it reuses intermediate computations.
        
        Parameters
        ----------
        F : Array
            3D array of shape (nz, ny, nx)
        lam_x : float, default 0.0
            Tikhonov regularization parameter for x-direction
        lam_y : float, default 0.0
            Tikhonov regularization parameter for y-direction
        lam_z : float, default 0.0
            Tikhonov regularization parameter for z-direction
        uniform_bc_x : bool, default False
            Whether to use uniform boundary conditions in x-direction
        uniform_bc_y : bool, default False
            Whether to use uniform boundary conditions in y-direction
        uniform_bc_z : bool, default False
            Whether to use uniform boundary conditions in z-direction
        bc_x : float | Array | None, default None
            Boundary condition values for x-direction
        bc_y : float | Array | None, default None
            Boundary condition values for y-direction
        bc_z : float | Array | None, default None
            Boundary condition values for z-direction
        use_loop : bool, default False
            If True, use loop-based implementation (faster for CPU on smaller grids).
            If False, use batched operations (better for GPU or very large grids).
        
        Returns
        -------
        dF_dx : Array
            First derivative in x-direction: ∂F/∂x, shape (nz, ny, nx)
        dF_dy : Array
            First derivative in y-direction: ∂F/∂y, shape (nz, ny, nx)
        dF_dz : Array
            First derivative in z-direction: ∂F/∂z, shape (nz, ny, nx)
        d2F_dx2 : Array
            Second derivative in x-direction: ∂²F/∂x², shape (nz, ny, nx)
        d2F_dy2 : Array
            Second derivative in y-direction: ∂²F/∂y², shape (nz, ny, nx)
        d2F_dz2 : Array
            Second derivative in z-direction: ∂²F/∂z², shape (nz, ny, nx)
        """
        self._check_shape(F)
        
        if use_loop:
            # Loop-based implementation (better cache behavior for CPU)
            F_np = np.asarray(F, dtype=np.float64)
            nz, ny, nx = F_np.shape
            
            # For x-direction: differentiate along axis=2 (columns)
            dF_dx_list = []
            d2F_dx2_list = []
            for i in range(nz):
                for j in range(ny):
                    f_col = F_np[i, j, :]  # (nx,)
                    df1_x, df2_x, _ = self.x_model.differentiate_1_2(f_col, lam=lam_x)
                    dF_dx_list.append(df1_x)
                    d2F_dx2_list.append(df2_x)
            dF_dx = np.array(dF_dx_list).reshape(nz, ny, nx)
            d2F_dx2 = np.array(d2F_dx2_list).reshape(nz, ny, nx)
            
            # For y-direction: differentiate along axis=1 (rows)
            dF_dy_list = []
            d2F_dy2_list = []
            for i in range(nz):
                for j in range(nx):
                    f_row = F_np[i, :, j]  # (ny,)
                    df1_y, df2_y, _ = self.y_model.differentiate_1_2(f_row, lam=lam_y)
                    dF_dy_list.append(df1_y)
                    d2F_dy2_list.append(df2_y)
            # Reshape: list order is (z=0,x=0), (z=0,x=1), ..., (z=0,x=nx-1), (z=1,x=0), ...
            # So reshape to (nz, nx, ny) then transpose to (nz, ny, nx)
            dF_dy = np.array(dF_dy_list).reshape(nz, nx, ny).transpose(0, 2, 1)
            d2F_dy2 = np.array(d2F_dy2_list).reshape(nz, nx, ny).transpose(0, 2, 1)
            
            # For z-direction: differentiate along axis=0 (depth)
            dF_dz_list = []
            d2F_dz2_list = []
            for j in range(ny):
                for k in range(nx):
                    f_depth = F_np[:, j, k]  # (nz,)
                    df1_z, df2_z, _ = self.z_model.differentiate_1_2(f_depth, lam=lam_z)
                    dF_dz_list.append(df1_z)
                    d2F_dz2_list.append(df2_z)
            # Reshape: list order is (y=0,x=0), (y=0,x=1), ..., (y=0,x=nx-1), (y=1,x=0), ...
            # So reshape to (ny, nx, nz) then transpose to (nz, ny, nx)
            dF_dz = np.array(dF_dz_list).reshape(ny, nx, nz).transpose(2, 0, 1)
            d2F_dz2 = np.array(d2F_dz2_list).reshape(ny, nx, nz).transpose(2, 0, 1)
        else:
            # Batched implementation (better for GPU or very large grids)
            # Check if differentiate_1_2_batched is available (temporarily in profiling version)
            has_batched = hasattr(self.x_model, 'differentiate_1_2_batched')
            
            if not has_batched:
                # Fall back to loop version if batched method is not available
                # This will happen when using the regular bspf1d module (not profiling version)
                F_np = np.asarray(F, dtype=np.float64)
                nz, ny, nx = F_np.shape
                
                # For x-direction: differentiate along axis=2 (columns)
                dF_dx_list = []
                d2F_dx2_list = []
                for i in range(nz):
                    for j in range(ny):
                        f_col = F_np[i, j, :]  # (nx,)
                        df1_x, df2_x, _ = self.x_model.differentiate_1_2(f_col, lam=lam_x)
                        dF_dx_list.append(df1_x)
                        d2F_dx2_list.append(df2_x)
                dF_dx = np.array(dF_dx_list).reshape(nz, ny, nx)
                d2F_dx2 = np.array(d2F_dx2_list).reshape(nz, ny, nx)
                
                # For y-direction: differentiate along axis=1 (rows)
                dF_dy_list = []
                d2F_dy2_list = []
                for i in range(nz):
                    for j in range(nx):
                        f_row = F_np[i, :, j]  # (ny,)
                        df1_y, df2_y, _ = self.y_model.differentiate_1_2(f_row, lam=lam_y)
                        dF_dy_list.append(df1_y)
                        d2F_dy2_list.append(df2_y)
                # Reshape: list order is (z=0,x=0), (z=0,x=1), ..., (z=0,x=nx-1), (z=1,x=0), ...
                # So reshape to (nz, nx, ny) then transpose to (nz, ny, nx)
                dF_dy = np.array(dF_dy_list).reshape(nz, nx, ny).transpose(0, 2, 1)
                d2F_dy2 = np.array(d2F_dy2_list).reshape(nz, nx, ny).transpose(0, 2, 1)
                
                # For z-direction: differentiate along axis=0 (depth)
                dF_dz_list = []
                d2F_dz2_list = []
                for j in range(ny):
                    for k in range(nx):
                        f_depth = F_np[:, j, k]  # (nz,)
                        df1_z, df2_z, _ = self.z_model.differentiate_1_2(f_depth, lam=lam_z)
                        dF_dz_list.append(df1_z)
                        d2F_dz2_list.append(df2_z)
                # Reshape: list order is (y=0,x=0), (y=0,x=1), ..., (y=0,x=nx-1), (y=1,x=0), ...
                # So reshape to (ny, nx, nz) then transpose to (nz, ny, nx)
                dF_dz = np.array(dF_dz_list).reshape(ny, nx, nz).transpose(2, 0, 1)
                d2F_dz2 = np.array(d2F_dz2_list).reshape(ny, nx, nz).transpose(2, 0, 1)
            else:
                # Use batched implementation when available
                F_np = np.asarray(F, dtype=np.float64)
                nz, ny, nx = F_np.shape
                
                # For x-direction: differentiate along axis=2 (columns)
                # Reshape to (nx, nz*ny) for batched differentiation
                F_reshaped_x = F_np.reshape(nz * ny, nx).T  # (nx, nz*ny)
                dF_dx_T, d2F_dx2_T, _ = self.x_model.differentiate_1_2_batched(F_reshaped_x, lam=lam_x)
                dF_dx = dF_dx_T.T.reshape(nz, ny, nx)
                d2F_dx2 = d2F_dx2_T.T.reshape(nz, ny, nx)
                
                # For y-direction: differentiate along axis=1 (rows)
                # Reshape to (ny, nz*nx) for batched differentiation
                F_reshaped_y = F_np.transpose(0, 2, 1).reshape(nz * nx, ny).T  # (ny, nz*nx)
                dF_dy_T, d2F_dy2_T, _ = self.y_model.differentiate_1_2_batched(F_reshaped_y, lam=lam_y)
                dF_dy_T_reshaped = dF_dy_T.T.reshape(nz, nx, ny)
                dF_dy = dF_dy_T_reshaped.transpose(0, 2, 1)
                d2F_dy2_T_reshaped = d2F_dy2_T.T.reshape(nz, nx, ny)
                d2F_dy2 = d2F_dy2_T_reshaped.transpose(0, 2, 1)
                
                # For z-direction: differentiate along axis=0 (depth)
                # Reshape to (nz, ny*nx) for batched differentiation
                F_reshaped_z = F_np.reshape(nz, ny * nx)  # (nz, ny*nx)
                dF_dz, d2F_dz2, _ = self.z_model.differentiate_1_2_batched(F_reshaped_z, lam=lam_z)
                dF_dz = dF_dz.reshape(nz, ny, nx)
                d2F_dz2 = d2F_dz2.reshape(nz, ny, nx)
        
        return (
            dF_dx.astype(np.float64),
            dF_dy.astype(np.float64),
            dF_dz.astype(np.float64),
            d2F_dx2.astype(np.float64),
            d2F_dy2.astype(np.float64),
            d2F_dz2.astype(np.float64),
        )

    # ---- Neumann-enforced variants ----
    def partial_dx_neumann(
        self,
        F: Array,
        *,
        order: int = 1,
        lam: float = 0.0,
        flux: Tuple[float | Array, float | Array] = (0.0, 0.0),  # (left, right), scalar or shaped like (nz, ny)
        return_spline: bool = False,
        uniform_bc: bool = False,
        bc: float | Array | None = None,
    ):
        self._check_shape(F)
        return self._diff_axis_neumann_3d(
            F, self.x_model, lam=lam, k=order, axis=2, flux=flux, return_spline=return_spline,
            uniform_bc=uniform_bc, bc=bc, use_gpu=self.use_gpu
        )

    def partial_dy_neumann(
        self,
        F: Array,
        *,
        order: int = 1,
        lam: float = 0.0,
        flux: Tuple[float | Array, float | Array] = (0.0, 0.0),  # (bottom, top), scalar or shaped like (nz, nx)
        return_spline: bool = False,
        uniform_bc: bool = False,
        bc: float | Array | None = None,
    ):
        self._check_shape(F)
        return self._diff_axis_neumann_3d(
            F, self.y_model, lam=lam, k=order, axis=1, flux=flux, return_spline=return_spline,
            uniform_bc=uniform_bc, bc=bc, use_gpu=self.use_gpu
        )

    def partial_dz_neumann(
        self,
        F: Array,
        *,
        order: int = 1,
        lam: float = 0.0,
        flux: Tuple[float | Array, float | Array] = (0.0, 0.0),  # (back, front), scalar or shaped like (ny, nx)
        return_spline: bool = False,
        uniform_bc: bool = False,
        bc: float | Array | None = None,
    ):
        self._check_shape(F)
        return self._diff_axis_neumann_3d(
            F, self.z_model, lam=lam, k=order, axis=0, flux=flux, return_spline=return_spline,
            uniform_bc=uniform_bc, bc=bc, use_gpu=self.use_gpu
        )
