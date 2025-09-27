#!/usr/bin/env python3
"""
2-D shallow-water (BSLF spatial ops) + DCT/DST exponential dealias filters

BCs:
  • η : zero-flux (Neumann)  -> filtered with DCT-II (even extension)
  • M,N : Dirichlet(0)       -> filtered with DST-II (odd  extension)

We precompute bspf2d derivative plans once, then time-step explicitly.
"""

from __future__ import annotations
from bspf import bspf2d
from matplotlib import pyplot
import numpy as np
from scipy.fft import dct, idct, dst, idst


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

def update_eta_2D_bspf(eta, M, N, dt, plan_dx, plan_dy):
    """
    Continuity: η_t = -(M_x + N_y).
    We impose Dirichlet(0) on M,N before taking derivatives.
    """
    M_bc = M.copy(); _enforce_dirichlet_zero(M_bc)
    N_bc = N.copy(); _enforce_dirichlet_zero(N_bc)

    dMdx = plan_dx.apply(M_bc)   # ∂M/∂x
    dNdy = plan_dy.apply(N_bc)   # ∂N/∂y
    eta_new = eta - dt * (dMdx + dNdy)
    return eta_new


def update_M_2D_bspf(eta, M, N, D, g, h, alpha, dt, plan_dx, plan_dy, plan_dx_eta_neu):
    """
    M_t = -(∂x(M^2/D) + ∂y(MN/D) + g D ∂x η) - friction
    Use Neumann(0) for ∂x η at boundaries; standard plans for the rest.
    """
    detadx  = plan_dx_eta_neu.apply(eta, flux=(0.0, 0.0))
    darg1dx = plan_dx.apply((M**2) / D)
    darg2dy = plan_dy.apply((M * N) / D)

    fric = g * alpha**2 * M * np.sqrt(M**2 + N**2) / (D ** (7.0 / 3.0))
    M_new = M - dt * (darg1dx + darg2dy + g * D * detadx + fric)

    _enforce_dirichlet_zero(M_new)
    return M_new


def update_N_2D_bspf(eta, M, N, D, g, h, alpha, dt, plan_dx, plan_dy, plan_dy_eta_neu):
    """
    N_t = -(∂x(MN/D) + ∂y(N^2/D) + g D ∂y η) - friction
    Use Neumann(0) for ∂y η at boundaries; standard plans for the rest.
    """
    detady  = plan_dy_eta_neu.apply(eta, flux=(0.0, 0.0))
    darg1dx = plan_dx.apply((M * N) / D)
    darg2dy = plan_dy.apply((N**2) / D)

    fric = g * alpha**2 * N * np.sqrt(M**2 + N**2) / (D ** (7.0 / 3.0))
    N_new = N - dt * (darg1dx + darg2dy + g * D * detady + fric)

    _enforce_dirichlet_zero(N_new)
    return N_new


# ───────────────────────────── main driver ──────────────────────────────

def Shallow_water_2D_bspf(
    eta0, M0, N0, h, g, alpha, nt, dt, x, y, degree=10,
    use_filter=True, filter_alpha=36.0, filter_order=8, filter_stride=1,
    filter_eta=True,  # toggle filtering of η (you can turn this off)
):
    """
    Uses bspf2d spatial derivatives with precomputed plans and applies
    BC-compatible DCT/DST exponential filters for dealiasing.
    """
    eta = eta0.copy(); M = M0.copy(); N = N0.copy()
    D = eta + h

    # Build BSLF2D operator once
    op = bspf2d.from_grids(x=x, 
                            y=y, 
                            degree_x=degree, 
                            degree_y=degree, 
                            order_x= degree, 
                            order_y=degree, 
                            num_boundary_points_x=degree+1,
                            num_boundary_points_y=degree+1,
                            use_clustering_x=True,
                            use_clustering_y=True,
                            correction="spectral")

    # Precompute derivative plans (fast path)
    plan_dx = op.make_plan_dx(order=1, lam=0.0, neumann=False)
    plan_dy = op.make_plan_dy(order=1, lam=0.0, neumann=False)
    plan_dx_eta_neu = op.make_plan_dx(order=1, lam=0.0, neumann=True)
    plan_dy_eta_neu = op.make_plan_dy(order=1, lam=0.0, neumann=True)

    # plotting (unchanged)
    fig = pyplot.figure(figsize=(10., 6.))
    cmap = 'Blues_r'
    pyplot.tight_layout()
    extent = [x.min(), x.max(), y.min(), y.max()]

    # topo = pyplot.imshow(np.flipud(-h), cmap=pyplot.cm.gray, interpolation='nearest', extent=extent)
    im = pyplot.imshow(np.flipud(eta), extent=extent, interpolation='spline36',
                       cmap=cmap, alpha=1, vmin=-1.5, vmax=1.5)
    pyplot.xlabel('x [m]'); pyplot.ylabel('y [m]')
    cbar = pyplot.colorbar(im)
    pyplot.gca().invert_yaxis()
    cbar.set_label(r'$\eta$ [m]')
    pyplot.ion()

    nsnap = 100; snap_count = 0

    data_snap = np.zeros((x.shape[0],y.shape[0],100))
    for n in range(nt+1):

        # 1) continuity
        eta = update_eta_2D_bspf(eta, M, N, dt, plan_dx, plan_dy)

        # 2) momentum x
        M = update_M_2D_bspf(eta, M, N, D, g, h, alpha, dt, plan_dx, plan_dy, plan_dx_eta_neu)

        # 3) momentum y
        N = update_N_2D_bspf(eta, M, N, D, g, h, alpha, dt, plan_dx, plan_dy, plan_dy_eta_neu)

        # 4) update column height
        D = eta + h

        # 5) BC-compatible dealiasing filters (every filter_stride steps)
        if use_filter and (n % filter_stride == 0):
            # Dirichlet(0) fluxes → DST filter
            M = dst_filter_dirichlet(M, alpha=filter_alpha, order=filter_order)
            N = dst_filter_dirichlet(N, alpha=filter_alpha, order=filter_order)
            _enforce_dirichlet_zero(M)
            _enforce_dirichlet_zero(N)

            # Neumann(0) surface → DCT filter (optional)
            if filter_eta:
                eta = dct_filter_neumann(eta, alpha=filter_alpha, order=filter_order)
                _enforce_neumann_zero_eta_in_field(eta)

        if (n % nsnap) == 0:
            print(f"Time step {n} of {nt}")
            im.set_data(eta)
            fig.canvas.draw()
            fname = f"tsunami_{snap_count:04d}.png"
            pyplot.savefig(fname, format='png', bbox_inches='tight', dpi=125)
            data_snap[:,:,snap_count] = eta
            snap_count += 1

    return eta, M, N, data_snap


if __name__ == "__main__":
    # Domain
    Lx = 100.0
    Ly = 100.0
    nx = 256
    ny = 256

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

    Tmax = 1.0
    # Estimate dt from CFL condition
    # c = np.sqrt(g * np.max(h))
    # dx = x[1] - x[0]
    # dy = y[1] - y[0]
    # dt = 0.1 * np.min(dx / c)
    dt = 1 / 1000
    print(f"dt = {dt}")
    nt = int(Tmax / dt)

    # Run
    eta, M, N, data_snap = Shallow_water_2D_bspf(
        eta0, M0, N0, h, g, alpha, nt, dt, x, y, degree=9,
        use_filter=True, filter_alpha=36.0, filter_order=36, filter_stride=1,
        filter_eta=True,
    )
