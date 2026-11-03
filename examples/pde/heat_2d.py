# SPDX-License-Identifier: GPL-3.0-or-later
"""2D heat equation u_t = kappa (u_xx + u_yy) on a rectangle with Dirichlet walls.

Method of lines with the energy-stable (Galerkin) BSPF Laplacian
Grid.weak_laplacian in space and classical RK4 in time. The exact solution is a
decaying separable mode whose wavenumbers do not fit the box,
    u = exp(-kappa (kx^2 + ky^2) t) cos(kx x + px) cos(ky y + py),
so the boundary data are nonzero, time dependent and not periodic. Wall
samples follow the Dirichlet data g. The right-hand side is
kappa weak_laplacian(u, g_t / kappa): the walls' rates enter the interior
through the Galerkin mass. Walls are reset to g after every step.

Weak form: M u_t = -kappa K u on the BSPF trial space, with M = int phi_i phi_j
and K = int phi_i' phi_j'. The interior operator -M_II^{-1} K_II is similar to
a symmetric negative-definite matrix, so the discrete energy u^T M u decays
monotonically for any endpoint jets. Endpoint jets are local Chebyshev fits with
constraint order 6. Higher orders amplify the setup's rounding (see
GalerkinLine.fit_amplification), lower ones cost accuracy.

The step is dt = cfl * 2.78 / (kappa (rho_x + rho_y)), with rho the spectral
radius of each axis's weak operator (matrix-free power iteration). RK4's
stability limit on the negative real axis is 2.78; the weak operators' spectra
are real and at most -pi^2 / L^2.

    python examples/pde/heat_2d.py [--n 65] [--plot out.png]
"""
import argparse
import time

import numpy as np

import bspf_cpp as bspf

KAPPA = 0.05
KX, KY = 3.1, 4.3        # not multiples of pi / L: non-periodic, nonzero walls
PX, PY = 0.4, -0.7
LX, LY = 1.0, 1.5
T_FINAL = 1.0
OPTIONS = dict(degree=9, n_basis=18, lam=1e-6, endpoint_method="chebyshev", boundary_points=16,
               chebyshev_modes=12, chebyshev_alpha=1e-12, constraint_order=6)


# The exact solution at points (X, Y): full meshes or just the wall points.
def value(X, Y, t):
    return np.exp(-KAPPA * (KX**2 + KY**2) * t) * np.cos(KX * X + PX) * np.cos(KY * Y + PY)


def rate(X, Y, t):  # u_t
    return -KAPPA * (KX**2 + KY**2) * value(X, Y, t)


def exact(x, y, t):
    return value(*np.meshgrid(x, y, indexing="ij"), t)


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


def solve(nx, ny, cfl=0.7, history=None, options=None):
    """Integrate to T_FINAL; returns (x, y, u, steps). Appends (t, error) to history."""
    options = OPTIONS if options is None else options
    x, y = np.linspace(0, LX, nx), np.linspace(0, LY, ny)
    grid = bspf.Grid.uniform((x, y), **options)
    on_wall = boundary_mask((nx, ny))
    X, Y = np.meshgrid(x, y, indexing="ij")
    Xw, Yw = X[on_wall], Y[on_wall]  # wall data is only ever needed there
    walls = np.zeros((nx, ny))
    rho = spectral_radius(grid.galerkin(0)) + spectral_radius(grid.galerkin(1))
    steps = int(np.ceil(T_FINAL / (cfl * 2.78 / (KAPPA * rho))))
    dt = T_FINAL / steps

    def rhs(u, t):
        walls[on_wall] = rate(Xw, Yw, t) / KAPPA
        return KAPPA * grid.weak_laplacian(u, walls)  # the result's walls are g_t

    u = value(X, Y, 0.0)
    for step in range(steps):
        t = step * dt
        k1 = rhs(u, t)
        k2 = rhs(u + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = rhs(u + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = rhs(u + dt * k3, t + dt)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        u[on_wall] = value(Xw, Yw, t + dt)
        if history is not None and (step + 1) % max(1, steps // 20) == 0:
            reference = exact(x, y, t + dt)
            history.append((t + dt, np.max(np.abs(u - reference)) / np.max(np.abs(reference))))
    return x, y, u, steps


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=65, help="points along x (y scales with the aspect ratio)")
    parser.add_argument("--plot", metavar="PNG", help="save a figure")
    args = parser.parse_args()

    ny = int(round((args.n - 1) * LY / LX)) + 1
    history = []
    t0 = time.perf_counter()
    x, y, u, steps = solve(args.n, ny, history=history)
    elapsed = time.perf_counter() - t0
    reference = exact(x, y, T_FINAL)
    error = np.max(np.abs(u - reference)) / np.max(np.abs(reference))
    print(f"grid {args.n} x {ny} on [0, {LX}] x [0, {LY}], kappa = {KAPPA}, T = {T_FINAL}")
    print(f"  {steps} RK4 steps in {elapsed:.2f} s ({4 * steps / elapsed:.0f} weak Laplacians/s)")
    print(f"  amplitude decayed to {np.exp(-KAPPA * (KX**2 + KY**2) * T_FINAL):.3f}")
    print(f"  relative max error at T: {error:.3e}")

    print("refinement (relative max error at T):")
    sizes, errors = sorted({17, 25, 33, 49, args.n}), []
    for n in sizes:
        m = int(round((n - 1) * LY / LX)) + 1
        if n == args.n:
            errors.append(error)  # already computed above
        else:
            xs, ys, us, _ = solve(n, m)
            ref = exact(xs, ys, T_FINAL)
            errors.append(np.max(np.abs(us - ref)) / np.max(np.abs(ref)))
        print(f"  {n:3d} x {m:3d}: {errors[-1]:.3e}")
    assert errors[-1] < errors[0] / 100

    if args.plot:
        import matplotlib.pyplot as plt
        X, Y = np.meshgrid(x, y, indexing="ij")
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.6), constrained_layout=True)
        for ax, values, title in zip(axes, (u, np.abs(u - reference)), (f"u at t = {T_FINAL}", "absolute error")):
            image = ax.pcolormesh(X, Y, values, shading="auto")
            ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
            fig.colorbar(image, ax=ax)
        times, errs = zip(*history)
        axes[2].semilogy(times, errs, "o-")
        axes[2].set(title="error in time", xlabel="t", ylabel="relative max error")
        axes[3].semilogy(sizes, errors, "o-")
        axes[3].set(title="refinement at T", xlabel="points along x", ylabel="relative max error")
        axes[3].set_xticks(sizes, [str(n) for n in sizes])
        axes[3].minorticks_off()
        fig.savefig(args.plot, dpi=120)
        print("saved", args.plot)


if __name__ == "__main__":
    main()
