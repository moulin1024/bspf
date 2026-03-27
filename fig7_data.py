#!/usr/bin/env python3
"""
2-D shallow-water (BSLF spatial ops) + DCT/DST exponential dealias filters

BCs:
  • η : zero-flux (Neumann)  -> filtered with DCT-II (even extension)
  • M,N : Dirichlet(0)       -> filtered with DST-II (odd  extension)

We precompute bfpsm2d derivative plans once, then time-step explicitly.
"""

from __future__ import annotations
from bspf2d import bspf2d
from matplotlib import pyplot
import numpy as np
from scipy.fft import dct, idct, dst, idst

# Optional GPU backend
_HAS_CUPY = False
try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None


# ───────────────────────────── helpers: BCs ─────────────────────────────

def _enforce_dirichlet_zero(F: np.ndarray) -> None:
    F[0, :]  = 0.0
    F[-1, :] = 0.0
    F[:, 0]  = 0.0
    F[:, -1] = 0.0

def _enforce_neumann_zero_eta_in_field(eta: np.ndarray) -> None:
    """Copy interior to boundary so stored field respects ∂η/∂n≈0."""
    eta[0, :]  = eta[1, :]
    eta[-1, :] = eta[-2, :]
    eta[:, 0]  = eta[:, 1]
    eta[:, -1] = eta[:, -2]


# ─────────────────────── BC-compatible spectral filters ──────────────────────
# We use type-II transforms with 'ortho' norm so idct/dst(type=2) invert dct/dst(type=2)

def _dct2(a: np.ndarray) -> np.ndarray:
    return dct(dct(a, type=2, axis=0, norm='ortho'), type=2, axis=1, norm='ortho')

def _idct2(A: np.ndarray) -> np.ndarray:
    return idct(idct(A, type=2, axis=0, norm='ortho'), type=2, axis=1, norm='ortho')

def _dst2(a: np.ndarray) -> np.ndarray:
    return dst(dst(a, type=2, axis=0, norm='ortho'), type=2, axis=1, norm='ortho')

def _idst2(A: np.ndarray) -> np.ndarray:
    return idst(idst(A, type=2, axis=0, norm='ortho'), type=2, axis=1, norm='ortho')


def dct_filter_neumann(F: np.ndarray, alpha: float = 36.0, order: int = 8) -> np.ndarray:
    """
    Exponential filter in DCT-II space (even extension → compatible with Neumann).
    sigma(k) = exp( -alpha * (|k|/k_max)^order )
    """
    ny, nx = F.shape
    ky = np.arange(ny)           # DCT-II mode index 0..N-1
    kx = np.arange(nx)
    ky_max = max(ky[-1], 1)
    kx_max = max(kx[-1], 1)

    KY = ky[:, None] / ky_max
    KX = kx[None, :] / kx_max
    rr = np.sqrt(KY**2 + KX**2)
    sigma = np.exp(-alpha * (rr**order))

    Fhat = _dct2(F)
    Fhat *= sigma
    return _idct2(Fhat)


def dst_filter_dirichlet(F: np.ndarray, alpha: float = 36.0, order: int = 8) -> np.ndarray:
    """
    Exponential filter in DST-II space (odd extension → compatible with Dirichlet 0).
    sigma(k) = exp( -alpha * (|k|/k_max)^order )
    """
    ny, nx = F.shape
    ky = np.arange(1, ny + 1)    # DST-II modes 1..N
    kx = np.arange(1, nx + 1)
    ky_max = max(ky[-1], 1)
    kx_max = max(kx[-1], 1)

    KY = ky[:, None] / ky_max
    KX = kx[None, :] / kx_max
    rr = np.sqrt(KY**2 + KX**2)
    sigma = np.exp(-alpha * (rr**order))

    Fhat = _dst2(F)
    Fhat *= sigma
    return _idst2(Fhat)


# ───────────────────────── BSLF update kernels ──────────────────────────

def update_eta_2D_bfpsm(eta, M, N, dt, plan_dx, plan_dy, M_bc_work=None, N_bc_work=None, 
                       w_sponge=None, sponge_strength=0.0, sponge_diffusion=0.0, 
                       plan_dxx=None, plan_dyy=None, xp=None):
    """
    Continuity: η_t = -(M_x + N_y).
    We impose Dirichlet(0) on M,N before taking derivatives.
    
    Optimized: uses pre-allocated work arrays to avoid unnecessary copies.
    
    Parameters
    ----------
    w_sponge : array, optional
        Sponge layer weight (0 = interior, 1 = boundary). If None, no sponge applied.
    sponge_strength : float
        Sponge layer damping strength (default: 0.0, no damping).
    sponge_diffusion : float
        Sponge layer diffusion coefficient (default: 0.0, no diffusion).
    plan_dxx, plan_dyy : plans, optional
        Second derivative plans for Laplacian computation in sponge layer.
    xp : module, optional
        Backend module (numpy or cupy). Auto-detected if None.
    """
    if xp is None:
        if _HAS_CUPY and isinstance(eta, cp.ndarray):
            xp = cp
        else:
            xp = np
    
    # Use pre-allocated work arrays if provided, otherwise create new ones
    if M_bc_work is None:
        M_bc_work = M.copy()
    else:
        M_bc_work[:] = M
    _enforce_dirichlet_zero(M_bc_work)
    
    if N_bc_work is None:
        N_bc_work = N.copy()
    else:
        N_bc_work[:] = N
    _enforce_dirichlet_zero(N_bc_work)

    dMdx = plan_dx.apply(M_bc_work)   # ∂M/∂x
    dNdy = plan_dy.apply(N_bc_work)   # ∂N/∂y
    eta_new = eta - dt * (dMdx + dNdy)
    
    # Apply sponge layer damping (drive eta toward 0 in sponge region)
    if w_sponge is not None and sponge_strength > 0:
        eta_new = eta_new - dt * sponge_strength * w_sponge * eta
    
    # Apply sponge layer diffusion (strong diffusion in sponge region)
    if w_sponge is not None and sponge_diffusion > 0 and plan_dxx is not None and plan_dyy is not None:
        # Compute Laplacian: ∇²η = ∂²η/∂x² + ∂²η/∂y²
        d2eta_dx2 = plan_dxx.apply(eta)
        d2eta_dy2 = plan_dyy.apply(eta)
        laplacian_eta = d2eta_dx2 + d2eta_dy2
        # Add diffusion term weighted by sponge layer
        eta_new = eta_new + dt * sponge_diffusion * w_sponge * laplacian_eta
    
    return eta_new


def update_M_2D_bfpsm(eta, M, N, D, g, h, alpha, dt, plan_dx, plan_dy, plan_dx_eta_neu, 
                     xp=None, w_sponge=None, sponge_strength=0.0, sponge_diffusion=0.0,
                     plan_dxx=None, plan_dyy=None):
    """
    M_t = -(∂x(M^2/D) + ∂y(MN/D) + g D ∂x η) - friction
    Use Neumann(0) for ∂x η at boundaries; standard plans for the rest.
    
    Optimized: uses xp (numpy or cupy) for backend-agnostic operations.
    
    Parameters
    ----------
    w_sponge : array, optional
        Sponge layer weight (0 = interior, 1 = boundary). If None, no sponge applied.
    sponge_strength : float
        Sponge layer damping strength (default: 0.0, no damping).
    sponge_diffusion : float
        Sponge layer diffusion coefficient (default: 0.0, no diffusion).
    plan_dxx, plan_dyy : plans, optional
        Second derivative plans for Laplacian computation in sponge layer.
    """
    if xp is None:
        # Auto-detect backend from array type
        if _HAS_CUPY and isinstance(M, cp.ndarray):
            xp = cp
        else:
            xp = np
    
    detadx  = plan_dx_eta_neu.apply(eta, flux=(0.0, 0.0))
    darg1dx = plan_dx.apply((M**2) / D)
    darg2dy = plan_dy.apply((M * N) / D)

    fric = g * alpha**2 * M * xp.sqrt(M**2 + N**2) / (D ** (7.0 / 3.0))
    M_new = M - dt * (darg1dx + darg2dy + g * D * detadx + fric)
    
    # Apply sponge layer damping (drive M toward 0 in sponge region)
    if w_sponge is not None and sponge_strength > 0:
        M_new = M_new - dt * sponge_strength * w_sponge * M
    
    # Apply sponge layer diffusion (strong diffusion in sponge region)
    if w_sponge is not None and sponge_diffusion > 0 and plan_dxx is not None and plan_dyy is not None:
        # Compute Laplacian: ∇²M = ∂²M/∂x² + ∂²M/∂y²
        d2M_dx2 = plan_dxx.apply(M)
        d2M_dy2 = plan_dyy.apply(M)
        laplacian_M = d2M_dx2 + d2M_dy2
        # Add diffusion term weighted by sponge layer
        M_new = M_new + dt * sponge_diffusion * w_sponge * laplacian_M

    _enforce_dirichlet_zero(M_new)
    return M_new


def update_N_2D_bfpsm(eta, M, N, D, g, h, alpha, dt, plan_dx, plan_dy, plan_dy_eta_neu, 
                     xp=None, w_sponge=None, sponge_strength=0.0, sponge_diffusion=0.0,
                     plan_dxx=None, plan_dyy=None):
    """
    N_t = -(∂x(MN/D) + ∂y(N^2/D) + g D ∂y η) - friction
    Use Neumann(0) for ∂y η at boundaries; standard plans for the rest.
    
    Optimized: uses xp (numpy or cupy) for backend-agnostic operations.
    
    Parameters
    ----------
    w_sponge : array, optional
        Sponge layer weight (0 = interior, 1 = boundary). If None, no sponge applied.
    sponge_strength : float
        Sponge layer damping strength (default: 0.0, no damping).
    sponge_diffusion : float
        Sponge layer diffusion coefficient (default: 0.0, no diffusion).
    plan_dxx, plan_dyy : plans, optional
        Second derivative plans for Laplacian computation in sponge layer.
    """
    if xp is None:
        # Auto-detect backend from array type
        if _HAS_CUPY and isinstance(N, cp.ndarray):
            xp = cp
        else:
            xp = np
    
    detady  = plan_dy_eta_neu.apply(eta, flux=(0.0, 0.0))
    darg1dx = plan_dx.apply((M * N) / D)
    darg2dy = plan_dy.apply((N**2) / D)

    fric = g * alpha**2 * N * xp.sqrt(M**2 + N**2) / (D ** (7.0 / 3.0))
    N_new = N - dt * (darg1dx + darg2dy + g * D * detady + fric)
    
    # Apply sponge layer damping (drive N toward 0 in sponge region)
    if w_sponge is not None and sponge_strength > 0:
        N_new = N_new - dt * sponge_strength * w_sponge * N
    
    # Apply sponge layer diffusion (strong diffusion in sponge region)
    if w_sponge is not None and sponge_diffusion > 0 and plan_dxx is not None and plan_dyy is not None:
        # Compute Laplacian: ∇²N = ∂²N/∂x² + ∂²N/∂y²
        d2N_dx2 = plan_dxx.apply(N)
        d2N_dy2 = plan_dyy.apply(N)
        laplacian_N = d2N_dx2 + d2N_dy2
        # Add diffusion term weighted by sponge layer
        N_new = N_new + dt * sponge_diffusion * w_sponge * laplacian_N

    _enforce_dirichlet_zero(N_new)
    return N_new


# ───────────────────────────── main driver ──────────────────────────────

def Shallow_water_2D_bfpsm(
    eta0, M0, N0, h, g, alpha, nt, dt, x, y, degree=10,
    use_filter=True, filter_alpha=36.0, filter_order=8, filter_stride=1,
    filter_eta=True,  # toggle filtering of η (you can turn this off)
    use_gpu=True,
    sponge_width=20.0,  # Sponge layer width (in grid units or physical units)
    sponge_strength=1.0,  # Sponge layer damping strength
    sponge_diffusion=1.0,  # Sponge layer diffusion coefficient
):
    """
    Uses bfpsm2d spatial derivatives with precomputed plans and applies
    BC-compatible DCT/DST exponential filters for dealiasing.
    
    Optimized: converts arrays to GPU (CuPy) when use_gpu=True to avoid
    CPU↔GPU transfers during computation.
    """
    # Convert arrays to GPU if needed
    if use_gpu:
        if not _HAS_CUPY:
            raise RuntimeError("use_gpu=True requires CuPy. Install with: pip install cupy-cuda12x")
        xp = cp
        # Convert input arrays to CuPy arrays
        x = cp.asarray(x, dtype=cp.float64)
        y = cp.asarray(y, dtype=cp.float64)
        eta = cp.asarray(eta0, dtype=cp.float64).copy()
        M = cp.asarray(M0, dtype=cp.float64).copy()
        N = cp.asarray(N0, dtype=cp.float64).copy()
        h = cp.asarray(h, dtype=cp.float64)
    else:
        xp = np
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        eta = np.asarray(eta0, dtype=np.float64).copy()
        M = np.asarray(M0, dtype=np.float64).copy()
        N = np.asarray(N0, dtype=np.float64).copy()
        h = np.asarray(h, dtype=np.float64)
    
    D = eta + h

    # Build BSLF2D operator once (arrays must match use_gpu backend)
    op = bspf2d.from_grids(x=x, 
                            y=y, 
                            degree_x=degree,
                            degree_y=degree,
                            use_clustering_x=True,
                            use_clustering_y=True,
                            correction="spectral",
                            use_gpu=use_gpu)

    # Precompute derivative plans (fast path)
    plan_dx = op.make_plan_dx(order=1, lam=0.0, neumann=False)
    plan_dy = op.make_plan_dy(order=1, lam=0.0, neumann=False)
    plan_dx_eta_neu = op.make_plan_dx(order=1, lam=0.0, neumann=True)
    plan_dy_eta_neu = op.make_plan_dy(order=1, lam=0.0, neumann=True)
    
    # Precompute second derivative plans for diffusion in sponge layer
    plan_dxx = op.make_plan_dx(order=2, lam=0.0, neumann=False)
    plan_dyy = op.make_plan_dy(order=2, lam=0.0, neumann=False)

    # Pre-allocate work arrays to avoid repeated allocations
    M_bc_work = xp.empty_like(M)
    N_bc_work = xp.empty_like(N)
    
    # Build sponge layer weight function for left, top, and bottom boundaries
    # Right boundary (x = Lx) remains as solid wall (no sponge)
    ny, nx = eta.shape
    w_sponge = xp.zeros_like(eta)
    
    # Create coordinate arrays (ensure they're 1D)
    if use_gpu:
        x_grid = xp.asarray(x).flatten()
        y_grid = xp.asarray(y).flatten()
    else:
        x_grid = np.asarray(x).flatten()
        y_grid = np.asarray(y).flatten()
    
    # Create meshgrid for sponge weight calculation (use 'xy' indexing to match eta shape)
    # eta has shape (ny, nx), so meshgrid with default 'xy' gives (ny, nx) arrays
    X_sponge, Y_sponge = xp.meshgrid(x_grid, y_grid, indexing='xy')
    
    # Left boundary (x = 0): sponge layer
    # Weight increases from 0 (at x = sponge_width) to 1 (at x = 0)
    mask_left = X_sponge < sponge_width
    if xp.any(mask_left):
        w_left = xp.where(mask_left, 
                         (1.0 - X_sponge / sponge_width)**4, 
                         0.0)
        w_sponge = xp.maximum(w_sponge, w_left)
    
    # Bottom boundary (y = 0): sponge layer
    # Weight increases from 0 (at y = sponge_width) to 1 (at y = 0)
    mask_bottom = Y_sponge < sponge_width
    if xp.any(mask_bottom):
        w_bottom = xp.where(mask_bottom,
                           (1.0 - Y_sponge / sponge_width)**4,
                           0.0)
        w_sponge = xp.maximum(w_sponge, w_bottom)
    
    # Top boundary (y = Ly): sponge layer
    # Weight increases from 0 (at y = Ly - sponge_width) to 1 (at y = Ly)
    y_max = y_grid.max()
    mask_top = Y_sponge > (y_max - sponge_width)
    if xp.any(mask_top):
        w_top = xp.where(mask_top,
                        ((Y_sponge - (y_max - sponge_width)) / sponge_width)**4,
                        0.0)
        w_sponge = xp.maximum(w_sponge, w_top)
    
    # Ensure w_sponge is in [0, 1]
    w_sponge = xp.clip(w_sponge, 0.0, 1.0)

    # plotting (convert to NumPy for matplotlib)
    fig = pyplot.figure(figsize=(10., 6.))
    cmap = 'Blues_r'
    pyplot.tight_layout()
    # Convert to NumPy for extent calculation
    if use_gpu:
        x_np = cp.asnumpy(x)
        y_np = cp.asnumpy(y)
    else:
        x_np = x
        y_np = y
    extent = [x_np.min(), x_np.max(), y_np.min(), y_np.max()]

    # Convert eta to NumPy for initial plot
    if use_gpu:
        eta_plot = cp.asnumpy(eta)
    else:
        eta_plot = eta
    # topo = pyplot.imshow(np.flipud(-h), cmap=pyplot.cm.gray, interpolation='nearest', extent=extent)
    im = pyplot.imshow(np.flipud(eta_plot), extent=extent, interpolation='spline36',
                       cmap=cmap, alpha=1, vmin=-1, vmax=1)
    pyplot.xlabel('x [m]'); pyplot.ylabel('y [m]')
    cbar = pyplot.colorbar(im)
    pyplot.gca().invert_yaxis()
    cbar.set_label(r'$\eta$ [m]')
    pyplot.ion()

    nsnap = 100; snap_count = 0

    # Pre-allocate snapshot array (on CPU for saving)
    data_snap = np.zeros((x_np.shape[0], y_np.shape[0], 100))
    for n in range(nt+1):

        # 1) continuity (using pre-allocated work arrays)
        eta = update_eta_2D_bfpsm(eta, M, N, dt, plan_dx, plan_dy, M_bc_work, N_bc_work,
                                 w_sponge=w_sponge, sponge_strength=sponge_strength, 
                                 sponge_diffusion=sponge_diffusion, plan_dxx=plan_dxx, 
                                 plan_dyy=plan_dyy, xp=xp)

        # 2) momentum x (pass xp for backend-agnostic operations)
        M = update_M_2D_bfpsm(eta, M, N, D, g, h, alpha, dt, plan_dx, plan_dy, plan_dx_eta_neu, 
                             xp=xp, w_sponge=w_sponge, sponge_strength=sponge_strength,
                             sponge_diffusion=sponge_diffusion, plan_dxx=plan_dxx, plan_dyy=plan_dyy)

        # 3) momentum y (pass xp for backend-agnostic operations)
        N = update_N_2D_bfpsm(eta, M, N, D, g, h, alpha, dt, plan_dx, plan_dy, plan_dy_eta_neu, 
                             xp=xp, w_sponge=w_sponge, sponge_strength=sponge_strength,
                             sponge_diffusion=sponge_diffusion, plan_dxx=plan_dxx, plan_dyy=plan_dyy)

        # 4) update column height
        D = eta + h

        # 5) BC-compatible dealiasing filters (every filter_stride steps)
        if use_filter and (n % filter_stride == 0):
            # Dirichlet(0) fluxes → DST filter
            # Note: DCT/DST filters expect NumPy arrays, so convert if on GPU
            if use_gpu:
                M_np = cp.asnumpy(M)
                N_np = cp.asnumpy(N)
                M_np = dst_filter_dirichlet(M_np, alpha=filter_alpha, order=filter_order)
                N_np = dst_filter_dirichlet(N_np, alpha=filter_alpha, order=filter_order)
                _enforce_dirichlet_zero(M_np)
                _enforce_dirichlet_zero(N_np)
                M = cp.asarray(M_np)
                N = cp.asarray(N_np)
            else:
                M = dst_filter_dirichlet(M, alpha=filter_alpha, order=filter_order)
                N = dst_filter_dirichlet(N, alpha=filter_alpha, order=filter_order)
                _enforce_dirichlet_zero(M)
                _enforce_dirichlet_zero(N)

            # Neumann(0) surface → DCT filter (optional)
            if filter_eta:
                if use_gpu:
                    eta_np = cp.asnumpy(eta)
                    eta_np = dct_filter_neumann(eta_np, alpha=filter_alpha, order=filter_order)
                    _enforce_neumann_zero_eta_in_field(eta_np)
                    eta = cp.asarray(eta_np)
                else:
                    eta = dct_filter_neumann(eta, alpha=filter_alpha, order=filter_order)
                    _enforce_neumann_zero_eta_in_field(eta)

        if (n % nsnap) == 0:
            print(f"Time step {n} of {nt}")
            # Convert to NumPy for plotting
            if use_gpu:
                eta_plot = cp.asnumpy(eta)
            else:
                eta_plot = eta
            im.set_data(eta_plot)
            fig.canvas.draw()
            fname = f"image_out/Shallow_water_2D_bfpsm_filtered_{snap_count:04d}.png"
            pyplot.savefig(fname, format='png', bbox_inches='tight', dpi=125)
            data_snap[:,:,snap_count] = eta_plot
            snap_count += 1

    # Convert final results to NumPy before returning
    if use_gpu:
        eta = cp.asnumpy(eta)
        M = cp.asnumpy(M)
        N = cp.asnumpy(N)
    
    return eta, M, N, data_snap


if __name__ == "__main__":
    # Domain
    Lx = 100.0
    Ly = 100.0
    nx = 501
    ny = 501

    x = np.linspace(0.0, Lx, num=nx)
    y = np.linspace(0.0, Ly, num=ny)
    X, Y = np.meshgrid(x, y)

    # Depth (example: mild shelf)
    h = 10 - 7.5 * np.tanh((X - 75.) / 10.)

    # Initial surface (strong Gaussian pulse)
    eta0 = 1 * np.exp(-((X - 50)**2 / 10) - ((Y - 50)**2 / 10))

    # Initial fluxes
    M0 = 100.0 * eta0
    N0 = 0.0 * M0

    g = 9.81
    alpha = 0.025

    Tmax = 10.0
    # Estimate dt from CFL condition
    # c = np.sqrt(g * np.max(h))
    # dx = x[1] - x[0]
    # dy = y[1] - y[0]
    # dt = 0.1 * np.min(dx / c)
    dt = 1 / 1000
    print(f"dt = {dt}")
    nt = int(Tmax / dt)

    # Run
    eta, M, N, data_snap = Shallow_water_2D_bfpsm(
        eta0, M0, N0, h, g, alpha, nt, dt, x, y, degree=8,
        use_filter=True, filter_alpha=36.0, filter_order=36, filter_stride=1,
        filter_eta=True,
    )

    # # np.savez(f"tsunami_bfpsm_dctdst_{nx}.npz", eta=eta, M=M, N=N)
    # np.savez(f"tsunami_bfpsm_dctdst_{nx}_snaps.npz", data_snap=data_snap)
