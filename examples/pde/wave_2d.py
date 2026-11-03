# SPDX-License-Identifier: GPL-3.0-or-later
"""2D linear wave equation u_tt = c^2 (u_xx + u_yy) on a rectangle with Dirichlet walls.

First-order form u_t = v, v_t = c^2 Lap(u), with the energy-stable (Galerkin)
BSPF Laplacian Grid.weak_laplacian and Chebyshev endpoint jets in space, and
classical RK4 in time. The exact solution superposes two plane waves crossing
the box in different directions,
    u = sin(k1 . x - w1 t + p1) + 0.5 cos(k2 . x - w2 t + p2),   w = c |k|,
so the wall data are nonzero, time dependent and not periodic. Wall samples of
u and v follow g and g_t. v's right-hand side is c^2 weak_laplacian(u, g_tt /
c^2): the walls' accelerations enter the interior through the Galerkin mass.

Why the weak form. The waves' eigenvalues are +-c sqrt(lambda) for eigenvalues
lambda of the wall-clamped Laplacian, so they stay on the imaginary axis only if
every lambda is real. The collocation operator (Grid.laplacian) with Chebyshev
jets has complex lambda: its wave modes grow by ~1e13 over this run already on
a 33-point grid (the script prints the rate), and faster on finer grids. The weak
form uses the same BSPF trial space with M = int phi_i phi_j, K = int phi_i'
phi_j'. Its interior operator -M_II^{-1} K_II is similar to a symmetric matrix,
so its spectrum is real for any jets.

Time step. dt = cfl * 2.83 / w_max, with w_max = c sqrt(rho_x + rho_y) and rho
the spectral radius of each axis's weak operator (matrix-free power iteration).
RK4 is stable on the imaginary axis up to 2.83. Its phase error, about
(w dt)^4 per unit time, stays below the spatial error at cfl = 0.1.

    python examples/pde/wave_2d.py [--n 65] [--plot out.png]
"""
import argparse
import time

import numpy as np

import bspf_cpp as bspf

C = 1.0
K1, K2 = np.array([6.0, 4.0]), np.array([-3.0, 7.0])
P1, P2 = 0.3, -1.1
W1, W2 = C * np.linalg.norm(K1), C * np.linalg.norm(K2)
LX, LY = 1.0, 1.5
T_FINAL = 2.0
CFL = 0.1
# Chebyshev endpoint jets of order <= 3 (constraint_order 4), as in the 3D benchmark.
OPTIONS = dict(degree=9, n_basis=18, lam=1e-6, endpoint_method="chebyshev", boundary_points=16,
               chebyshev_modes=12, chebyshev_alpha=1e-12, constraint_order=4)


# The exact solution at points (X, Y): full meshes or just the wall points.
def phases(X, Y, t):
    return K1[0] * X + K1[1] * Y - W1 * t + P1, K2[0] * X + K2[1] * Y - W2 * t + P2


def value(X, Y, t):
    a, b = phases(X, Y, t)
    return np.sin(a) + 0.5 * np.cos(b)


def rate(X, Y, t):  # u_t
    a, b = phases(X, Y, t)
    return -W1 * np.cos(a) + 0.5 * W2 * np.sin(b)


def acceleration(X, Y, t):  # u_tt
    a, b = phases(X, Y, t)
    return -W1**2 * np.sin(a) - 0.5 * W2**2 * np.cos(b)


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


def strong_growth(options, n=33):
    """Largest growth rate of wave modes with the collocation Laplacian (small dense check)."""
    line = bspf.Line(np.linspace(0, LX, n), **options)
    eigenvalues = np.linalg.eigvals(line.derivative(2).matrix[1:-1, 1:-1]).astype(complex)
    return C * np.abs(np.sqrt(-eigenvalues).imag).max()


def solve(nx, ny, cfl=CFL, history=None, options=None):
    """Integrate to T_FINAL; returns (x, y, u, steps). Appends (t, error) to history."""
    options = OPTIONS if options is None else options
    x, y = np.linspace(0, LX, nx), np.linspace(0, LY, ny)
    grid = bspf.Grid.uniform((x, y), **options)
    frequency = C * np.sqrt(sum(spectral_radius(grid.galerkin(i)) for i in range(2)))
    steps = int(np.ceil(T_FINAL / (cfl * 2.83 / frequency)))
    dt = T_FINAL / steps
    on_wall = boundary_mask((nx, ny))
    X, Y = np.meshgrid(x, y, indexing="ij")
    Xw, Yw = X[on_wall], Y[on_wall]  # wall data is only ever needed there
    walls = np.zeros((nx, ny))

    def rhs(u, v, t):
        du = v.copy()
        du[on_wall] = rate(Xw, Yw, t)
        walls[on_wall] = acceleration(Xw, Yw, t) / C**2
        dv = C**2 * grid.weak_laplacian(u, walls)  # the result's walls are g_tt
        return du, dv

    u, v = value(X, Y, 0.0), rate(X, Y, 0.0)
    for step in range(steps):
        t = step * dt
        k1 = rhs(u, v, t)
        k2 = rhs(u + 0.5 * dt * k1[0], v + 0.5 * dt * k1[1], t + 0.5 * dt)
        k3 = rhs(u + 0.5 * dt * k2[0], v + 0.5 * dt * k2[1], t + 0.5 * dt)
        k4 = rhs(u + dt * k3[0], v + dt * k3[1], t + dt)
        u = u + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v = v + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        u[on_wall] = value(Xw, Yw, t + dt)
        v[on_wall] = rate(Xw, Yw, t + dt)
        if history is not None and (step + 1) % max(1, steps // 40) == 0:
            reference = exact(x, y, t + dt)
            history.append((t + dt, np.max(np.abs(u - reference)) / np.max(np.abs(reference))))
    return x, y, u, steps


def ny_for(nx):
    return int(round((nx - 1) * LY / LX)) + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=65, help="points along x (y scales with the aspect ratio)")
    parser.add_argument("--plot", metavar="PNG", help="save a figure")
    args = parser.parse_args()

    growth = strong_growth(OPTIONS)
    print(f"collocation Laplacian with these jets: wave modes grow at rate {growth:.0f} "
          f"(x{np.exp(growth * T_FINAL):.0e} over the run) -> use the weak form")

    history = []
    t0 = time.perf_counter()
    x, y, u, steps = solve(args.n, ny_for(args.n), history=history)
    elapsed = time.perf_counter() - t0
    reference = exact(x, y, T_FINAL)
    error = np.max(np.abs(u - reference)) / np.max(np.abs(reference))
    wavelengths = 2 * np.pi / np.array([np.linalg.norm(K1), np.linalg.norm(K2)])
    print(f"grid {args.n} x {ny_for(args.n)} on [0, {LX}] x [0, {LY}], c = {C}, T = {T_FINAL}")
    print(f"  wavelengths {np.round(wavelengths, 3)}; waves travel {C * T_FINAL:.1f} (box diagonal {np.hypot(LX, LY):.2f})")
    print(f"  {steps} RK4 steps in {elapsed:.2f} s ({4 * steps / elapsed:.0f} weak Laplacians/s)")
    print(f"  relative max error at T: {error:.3e}")

    print("refinement (relative max error at T):")
    sizes, errors = sorted({33, 49, 65, 97, args.n}), []
    for n in sizes:
        if n == args.n:
            errors.append(error)
        else:
            xs, ys, us, _ = solve(n, ny_for(n))
            ref = exact(xs, ys, T_FINAL)
            errors.append(np.max(np.abs(us - ref)) / np.max(np.abs(ref)))
        print(f"  {n:3d} x {ny_for(n):3d}: {errors[-1]:.3e}")
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
        axes[3].loglog(sizes, errors, "o-")
        axes[3].set(title="refinement at T", xlabel="points along x", ylabel="relative max error")
        axes[3].set_xticks(sizes, [str(n) for n in sizes])
        axes[3].minorticks_off()
        fig.savefig(args.plot, dpi=120)
        print("saved", args.plot)


if __name__ == "__main__":
    main()
