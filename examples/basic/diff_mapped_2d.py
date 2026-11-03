# SPDX-License-Identifier: GPL-3.0-or-later
"""2D differentiation on a deformed region bounded by four spline curves.

The sides are B-spline curves (SplineCurve) oriented bottom: P00 -> P10,
right: P10 -> P11, top: P01 -> P11, left: P00 -> P01. MappedGrid2D maps the unit
square onto the region by transfinite (Coons) interpolation. The grid stays
structured: nodes are the images of uniform BSPF lines in (xi, eta). Physical
derivatives come from the chain rule with exact metrics: direct BSPF reference
partials (xi, eta, xi-xi, xi-eta, eta-eta) times inverse metrics and their
derivatives, all evaluated from the curves' first and second derivatives.

The field is f(x, y) = sin(2x + 1/2) cos(3y - 1/5) + exp(xy) / 2, sampled at the
physical nodes, against its exact first and second derivatives.

Accuracy follows the smoothness of the map. Single-segment (Bezier) sides, as
here, are polynomials, so f(X(xi, eta)) is analytic and BSPF converges
spectrally. Multi-segment splines are only C^(d-1) at their knots, which limits
convergence to an algebraic rate.

    python examples/basic/diff_mapped_2d.py [--plot out.png]
"""
import argparse

import numpy as np

import bspf_cpp as bspf

OPTIONS = dict(degree=9, n_basis=18, lam=1e-6, endpoint_method="chebyshev", boundary_points=16,
               chebyshev_modes=12, chebyshev_alpha=1e-12, constraint_order=6)


def bulged_side(p0, p1, bump, normal, degree=5):
    """A degree-d Bezier curve from p0 to p1, pushed by `bump` along `normal`."""
    t = np.linspace(0, 1, degree + 1)[:, None]
    points = (1 - t) * np.asarray(p0, float) + t * np.asarray(p1, float)
    points[1:-1] += bump * np.sin(np.pi * t[1:-1]) * np.asarray(normal, float)
    return bspf.SplineCurve(points, degree)


def sides():
    P00, P10, P01, P11 = (0.0, 0.0), (1.0, 0.1), (-0.1, 1.0), (1.1, 1.2)
    return (bulged_side(P00, P10, 0.12, (0, 1)),    # bottom
            bulged_side(P10, P11, 0.10, (1, 0)),    # right
            bulged_side(P01, P11, 0.08, (0, 1)),    # top
            bulged_side(P00, P01, -0.10, (1, 0)))   # left


def f(x, y):
    return np.sin(2 * x + 0.5) * np.cos(3 * y - 0.2) + 0.5 * np.exp(x * y)


EXACT = {
    (1, 0): lambda x, y: 2 * np.cos(2 * x + 0.5) * np.cos(3 * y - 0.2) + 0.5 * y * np.exp(x * y),
    (0, 1): lambda x, y: -3 * np.sin(2 * x + 0.5) * np.sin(3 * y - 0.2) + 0.5 * x * np.exp(x * y),
    (2, 0): lambda x, y: -4 * np.sin(2 * x + 0.5) * np.cos(3 * y - 0.2) + 0.5 * y * y * np.exp(x * y),
    (0, 2): lambda x, y: -9 * np.sin(2 * x + 0.5) * np.cos(3 * y - 0.2) + 0.5 * x * x * np.exp(x * y),
    (1, 1): lambda x, y: -6 * np.cos(2 * x + 0.5) * np.sin(3 * y - 0.2) + 0.5 * (1 + x * y) * np.exp(x * y),
}
NAMES = {(1, 0): "f_x", (0, 1): "f_y", (2, 0): "f_xx", (0, 2): "f_yy", (1, 1): "f_xy"}


def relative_max(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=65, help="points per reference axis")
    parser.add_argument("--plot", metavar="PNG", help="save a figure")
    args = parser.parse_args()

    grid = bspf.MappedGrid2D(*sides(), (args.n, args.n), **OPTIONS)
    X, Y = grid.x, grid.y
    u = f(X, Y)
    print(f"{args.n} x {args.n} nodes; Jacobian in [{grid.jacobian.min():.3f}, {grid.jacobian.max():.3f}]")
    for orders, exact in EXACT.items():
        print(f"  {NAMES[orders]:5s} relative max error {relative_max(grid.partial(u, orders), exact(X, Y)):.3e}")

    print("refinement (relative max error):")
    sizes, table = (17, 33, 65, 129), {o: [] for o in EXACT}
    for n in sizes:
        g = bspf.MappedGrid2D(*sides(), (n, n), **OPTIONS)
        un = f(g.x, g.y)
        for orders, exact in EXACT.items():
            table[orders].append(relative_max(g.partial(un, orders), exact(g.x, g.y)))
        print(f"  N = {n:3d}: " + "  ".join(f"{NAMES[o]} {table[o][-1]:.1e}" for o in EXACT))
    assert all(errors[-1] < errors[0] / 1000 for errors in table.values())

    if args.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), constrained_layout=True)
        step = max(1, args.n // 16)
        for i in range(0, args.n, step):
            axes[0].plot(X[i, :], Y[i, :], "k-", lw=0.5)
            axes[0].plot(X[:, i], Y[:, i], "k-", lw=0.5)
        axes[0].set(title="mapped grid (every %d-th line)" % step, xlabel="x", ylabel="y", aspect="equal")
        for ax, values, title in zip(axes[1:3], (u, np.abs(grid.partial(u, (1, 0)) - EXACT[(1, 0)](X, Y))),
                                     ("f(x, y)", "|f_x error|")):
            image = ax.pcolormesh(X, Y, values, shading="gouraud")  # curvilinear nodes
            ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
            fig.colorbar(image, ax=ax)
        for orders, errors in table.items():
            axes[3].semilogy(sizes, errors, "o-", label=NAMES[orders])
        axes[3].set(title="refinement", xlabel="points per axis", ylabel="relative max error")
        axes[3].set_xticks(sizes, [str(n) for n in sizes])
        axes[3].minorticks_off()
        axes[3].legend()
        fig.savefig(args.plot, dpi=120)
        print("saved", args.plot)


if __name__ == "__main__":
    main()
