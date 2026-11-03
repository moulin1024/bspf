# SPDX-License-Identifier: GPL-3.0-or-later
"""2D linear wave equation u_tt = c^2 (u_xx + u_yy) on a rectangle with Dirichlet walls.

First-order form u_t = v, v_t = c^2 Lap(u): the BSPF Laplacian (Grid.laplacian)
in space, classical RK4 in time. The exact solution superposes two plane waves
crossing the box in different directions,
    u = sin(k1 . x - w1 t + p1) + 0.5 cos(k2 . x - w2 t + p2),   w = c |k|,
so the wall data are nonzero, time dependent and not periodic. Wall samples of
u and v follow g and g_t: their RK4 right-hand sides are g_t and g_tt, and they
are reset to the exact values after every step.

Stability. The wave system's eigenvalues are +-c sqrt(lambda) for eigenvalues
lambda of the wall-clamped Laplacian. They are purely imaginary only if every
lambda is real and negative. That holds for finite-difference endpoint jets with
degree <= 6 (default q and boundary points); higher degrees, wider stencils and
Chebyshev jets give complex lambda and growing modes, which RK4 does not damp.
The script checks the spectrum and refuses unstable settings.

Time step. RK4 is stable on the imaginary axis up to |w dt| = 2.83, but its
phase error, about (w dt)^4 per unit time, exceeds the spatial error (about
N^-6 for these jets) unless cfl <= 0.3. A sixth-order RK (Butcher, 7 stages)
was tried and dropped: it is not stable on the imaginary axis, so its usable
step is 6.6x smaller, and it needed 3.6x more Laplacians for the same error.

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
OPTIONS = dict(degree=6)  # finite-difference jets, q = 5, 6 boundary points

def phases(x, y, t):
    X, Y = np.meshgrid(x, y, indexing="ij")
    return K1[0] * X + K1[1] * Y - W1 * t + P1, K2[0] * X + K2[1] * Y - W2 * t + P2


def exact(x, y, t):
    a, b = phases(x, y, t)
    return np.sin(a) + 0.5 * np.cos(b)


def exact_rate(x, y, t):  # u_t
    a, b = phases(x, y, t)
    return -W1 * np.cos(a) + 0.5 * W2 * np.sin(b)


def exact_acceleration(x, y, t):  # u_tt
    a, b = phases(x, y, t)
    return -W1**2 * np.sin(a) - 0.5 * W2**2 * np.cos(b)


def boundary_mask(shape):
    mask = np.zeros(shape, dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
    return mask


def clamped_spectrum(line):
    """Eigenvalues of the interior (Dirichlet-clamped) block of d^2/dx^2."""
    return np.linalg.eigvals(line.derivative(2).matrix[1:-1, 1:-1])


def solve(nx, ny, cfl=0.3, history=None, options=None):
    """Integrate to T_FINAL; returns (x, y, u, steps). Appends (t, error) to history."""
    options = OPTIONS if options is None else options
    x, y = np.linspace(0, LX, nx), np.linspace(0, LY, ny)
    grid = bspf.Grid.uniform((x, y), **options)
    spectra = [clamped_spectrum(grid.line(i)) for i in range(2)]
    if any(np.abs(s.imag).max() > 1e-8 * np.abs(s).max() or s.real.max() >= 0 for s in spectra):
        raise ValueError("these options give a complex or non-negative clamped spectrum: the wave scheme would grow")
    # Highest wave frequency c sqrt(rho_x + rho_y); RK4 is stable on the imaginary axis up to 2.83.
    frequency = C * np.sqrt(sum(np.abs(s).max() for s in spectra))
    steps = int(np.ceil(T_FINAL / (cfl * 2.83 / frequency)))
    dt = T_FINAL / steps
    on_wall = boundary_mask((nx, ny))

    def rhs(u, v, t):
        du, dv = v.copy(), C**2 * grid.laplacian(u)
        du[on_wall] = exact_rate(x, y, t)[on_wall]
        dv[on_wall] = exact_acceleration(x, y, t)[on_wall]
        return du, dv

    u, v = exact(x, y, 0.0), exact_rate(x, y, 0.0)
    for step in range(steps):
        t = step * dt
        k1 = rhs(u, v, t)
        k2 = rhs(u + 0.5 * dt * k1[0], v + 0.5 * dt * k1[1], t + 0.5 * dt)
        k3 = rhs(u + 0.5 * dt * k2[0], v + 0.5 * dt * k2[1], t + 0.5 * dt)
        k4 = rhs(u + dt * k3[0], v + dt * k3[1], t + dt)
        u = u + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v = v + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        u[on_wall] = exact(x, y, t + dt)[on_wall]
        v[on_wall] = exact_rate(x, y, t + dt)[on_wall]
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

    history = []
    t0 = time.perf_counter()
    x, y, u, steps = solve(args.n, ny_for(args.n), history=history)
    elapsed = time.perf_counter() - t0
    reference = exact(x, y, T_FINAL)
    error = np.max(np.abs(u - reference)) / np.max(np.abs(reference))
    wavelengths = 2 * np.pi / np.array([np.linalg.norm(K1), np.linalg.norm(K2)])
    print(f"grid {args.n} x {ny_for(args.n)} on [0, {LX}] x [0, {LY}], c = {C}, T = {T_FINAL}")
    print(f"  wavelengths {np.round(wavelengths, 3)}; waves travel {C * T_FINAL:.1f} (box diagonal {np.hypot(LX, LY):.2f})")
    print(f"  {steps} RK4 steps in {elapsed:.2f} s ({4 * steps / elapsed:.0f} Laplacians/s)")
    print(f"  relative max error at T: {error:.3e}")

    try:
        solve(17, ny_for(17), options=dict(degree=9))
    except ValueError as reason:
        print(f"  degree 9 refused: {reason}")

    print("refinement (relative max error at T):")
    sizes, errors = sorted({33, 49, 65, 97, args.n}), []  # spatial error ~ N^-6
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
