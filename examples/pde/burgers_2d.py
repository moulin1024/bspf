# SPDX-License-Identifier: GPL-3.0-or-later
"""2D viscous Burgers equations against an exact traveling-wave solution.

    u_t + u u_x + v u_y = nu (u_xx + u_yy)
    v_t + u v_x + v v_y = nu (v_xx + v_yy),      Re = 1 / nu,

on the unit square with Dirichlet walls. The exact solution (Fletcher 1983;
Bahadir 2003) is a front moving across the box:
    u = 3/4 - s/4,  v = 3/4 + s/4,  s = 1 / (1 + exp(theta)),
    theta = Re (-4x + 4y - t) / 32,
so the wall data are non-periodic and time dependent. The front is centered on
y - x = t/4 and is about 8/Re wide in y - x.

Method of lines with classical RK4 in time:
* advection u . grad(u, v) with BSPF first derivatives (Grid.partial, with u and
  v batched on a trailing axis);
* diffusion with the energy-stable Galerkin Laplacian Grid.weak_laplacian. The
  exact Laplacian of the wall data enters as the result's wall values. Use
  `--laplacian strong` for the collocation Grid.laplacian instead.
Wall samples follow the exact data: their right-hand side is g_t, and they are
reset to g after every step.

Endpoint jets (--jets). The collocation first derivative with Chebyshev jets,
clamped at the walls, has boundary modes that grow (eigenvalues of -D1 with real
part up to +30 on 65 points). Viscosity damps them at moderate Re but not in
advection-dominated runs. Finite-difference jets (degree 6) keep the advective
spectrum on the imaginary axis but are less accurate. 'auto' takes Chebyshev
(q = 6) for Re <= 200 and FD degree 6 above. Measured relative max errors at
T = 1:
    Re = 100, Chebyshev:  5.4e-6 (33^2), 1.0e-7 (49^2), 4.4e-9 (65^2), 4.1e-11 (97^2)
    Re = 100, FD:         4.6e-6 (33^2), 4.6e-8 (65^2), 3.9e-9 (97^2)
    Re = 1000, FD:        5.4e-3 (97^2), 1.1e-3 (129^2), 1.1e-4 (193^2)
                          (Chebyshev jets blow up at Re = 1000 on 65^2 and 97^2)
With the collocation Laplacian (--laplacian strong) the Re = 100 Chebyshev
errors are about 18x larger (7.9e-8 on 65^2).

Time step: dt = cfl / (A / 2.83 + D / 2.78). A = max|u| (pi/dx + pi/dy) bounds
the advective frequencies (RK4 is stable on the imaginary axis up to 2.83).
D = nu (rho_x + rho_y) bounds the diffusive ones (real axis up to 2.78), with
rho from a power iteration on the weak operators.

    python examples/pde/burgers_2d.py [--re 100] [--n 65] [--jets auto|chebyshev|fd]
        [--laplacian weak|strong] [--sizes 33 49 65 97] [--plot out.png]
"""
import argparse
import time

import numpy as np

import bspf_cpp as bspf

T_FINAL = 1.0
CFL = 0.5
OPTIONS = dict(degree=9, n_basis=18, lam=1e-6, endpoint_method="chebyshev", boundary_points=16,
               chebyshev_modes=12, chebyshev_alpha=1e-12, constraint_order=6)
FD_OPTIONS = dict(degree=6)  # finite-difference jets: advective spectrum on the imaginary axis


def jet_options(jets, re):
    if jets == "auto":
        jets = "chebyshev" if re <= 200 else "fd"
    return jets, (OPTIONS if jets == "chebyshev" else FD_OPTIONS)


class TravelingWave:
    """The exact solution and the derivatives the scheme needs on the walls."""

    def __init__(self, re):
        self.re = re
        self.theta_x, self.theta_y, self.theta_t = -re / 8, re / 8, -re / 32

    def s(self, X, Y, t):
        return 1 / (1 + np.exp(self.re * (-4 * X + 4 * Y - t) / 32))

    def velocity(self, X, Y, t):  # (..., 2): (u, v)
        s = self.s(X, Y, t)
        return np.stack((0.75 - s / 4, 0.75 + s / 4), axis=-1)

    def rate(self, X, Y, t):  # (u_t, v_t); ds/dtheta = -s (1 - s)
        s = self.s(X, Y, t)
        ds = -s * (1 - s) * self.theta_t
        return np.stack((-ds / 4, ds / 4), axis=-1)

    def laplacian(self, X, Y, t):  # (Lap u, Lap v); d2s/dtheta2 = s (1 - s) (1 - 2 s)
        s = self.s(X, Y, t)
        lap = s * (1 - s) * (1 - 2 * s) * (self.theta_x**2 + self.theta_y**2)
        return np.stack((-lap / 4, lap / 4), axis=-1)


def boundary_mask(shape):
    mask = np.zeros(shape, dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
    return mask


def spectral_radius(galerkin, iterations=200):
    """Largest |lambda| of the interior weak operator by power iteration (matrix-free)."""
    n = galerkin.size
    v = np.random.default_rng(0).standard_normal(n - 2)
    radius = 0.0
    for _ in range(iterations):
        full = np.zeros((n, 1))
        full[1:-1, 0] = v / np.linalg.norm(v)
        v = galerkin.second_derivative(full)[:, 0]
        radius = np.linalg.norm(v)
    return radius


def solve(n, re, laplacian="weak", cfl=CFL, history=None, options=None):
    """Integrate to T_FINAL on an n x n grid; returns (x, y, w, steps) with w = (u, v) on the last axis."""
    options = OPTIONS if options is None else options
    exact = TravelingWave(re)
    nu = 1.0 / re
    x = y = np.linspace(0.0, 1.0, n)
    grid = bspf.Grid.uniform((x, y), **options)
    X, Y = np.meshgrid(x, y, indexing="ij")
    on_wall = boundary_mask((n, n))
    Xw, Yw = X[on_wall], Y[on_wall]
    walls = np.zeros((n, n, 2))

    rho = sum(spectral_radius(grid.galerkin(i)) for i in range(2))
    advective = 1.0 * 2 * np.pi / (x[1] - x[0])  # max |u| = 1
    steps = int(np.ceil(T_FINAL / (cfl / (advective / 2.83 + nu * rho / 2.78))))
    dt = T_FINAL / steps

    def rhs(w, t):
        wx, wy = grid.partial(w, (1, 0)), grid.partial(w, (0, 1))
        if laplacian == "weak":
            walls[on_wall] = exact.laplacian(Xw, Yw, t)
            lap = grid.weak_laplacian(w, walls)
        else:
            lap = grid.laplacian(w)
        dw = nu * lap - (w[..., :1] * wx + w[..., 1:] * wy)
        dw[on_wall] = exact.rate(Xw, Yw, t)
        return dw

    w = exact.velocity(X, Y, 0.0)
    for step in range(steps):
        t = step * dt
        k1 = rhs(w, t)
        k2 = rhs(w + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = rhs(w + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = rhs(w + dt * k3, t + dt)
        w = w + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        w[on_wall] = exact.velocity(Xw, Yw, t + dt)
        if history is not None and (step + 1) % max(1, steps // 40) == 0:
            history.append((t + dt, relative_error(w, exact.velocity(X, Y, t + dt))))
    return x, y, w, steps


def relative_error(w, reference):
    return float(np.max(np.abs(w - reference)) / np.max(np.abs(reference)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--re", type=float, default=100.0, help="Reynolds number 1/nu")
    parser.add_argument("--n", type=int, default=65, help="points per axis")
    parser.add_argument("--jets", choices=("auto", "chebyshev", "fd"), default="auto")
    parser.add_argument("--laplacian", choices=("weak", "strong"), default="weak")
    parser.add_argument("--sizes", type=int, nargs="+", default=[33, 49, 65], help="refinement grids")
    parser.add_argument("--plot", metavar="PNG", help="save a figure")
    args = parser.parse_args()

    jets, options = jet_options(args.jets, args.re)
    exact = TravelingWave(args.re)
    history = []
    t0 = time.perf_counter()
    x, y, w, steps = solve(args.n, args.re, args.laplacian, history=history, options=options)
    elapsed = time.perf_counter() - t0
    X, Y = np.meshgrid(x, y, indexing="ij")
    reference = exact.velocity(X, Y, T_FINAL)
    error = relative_error(w, reference)
    print(f"Re = {args.re:g}, grid {args.n} x {args.n}, T = {T_FINAL}, {args.laplacian} Laplacian, {jets} jets")
    print(f"  front width ~ {8 / args.re:.3f} in y - x; {steps} RK4 steps in {elapsed:.2f} s")
    print(f"  relative max error at T (u and v): {error:.3e}")

    print("refinement (relative max error at T):")
    sizes, errors = sorted(set(args.sizes) | {args.n}), []
    for n in sizes:
        if n == args.n:
            errors.append(error)
        else:
            xs, ys, ws, _ = solve(n, args.re, args.laplacian, options=options)
            Xs, Ys = np.meshgrid(xs, ys, indexing="ij")
            errors.append(relative_error(ws, exact.velocity(Xs, Ys, T_FINAL)))
        print(f"  {n:3d} x {n:3d}: {errors[-1]:.3e}")
    assert errors[-1] < errors[0]  # converging

    if args.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), constrained_layout=True)
        for ax, values, title in zip(axes, (w[..., 0], np.abs(w - reference).max(axis=-1)),
                                     (f"u at t = {T_FINAL}", "absolute error (max over u, v)")):
            image = ax.pcolormesh(X, Y, values, shading="auto")
            ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
            fig.colorbar(image, ax=ax)
        times, errs = zip(*history)
        axes[2].semilogy(times, errs, "o-")
        axes[2].set(title="error in time", xlabel="t", ylabel="relative max error")
        axes[3].loglog(sizes, errors, "o-")
        axes[3].set(title="refinement at T", xlabel="points per axis", ylabel="relative max error")
        axes[3].set_xticks(sizes, [str(n) for n in sizes])
        axes[3].minorticks_off()
        fig.savefig(args.plot, dpi=120)
        print("saved", args.plot)


if __name__ == "__main__":
    main()
