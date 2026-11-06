# SPDX-License-Identifier: GPL-3.0-or-later
"""Multi-patch 2D differentiation on a doubly connected region (a ring around a hole).

The ring lies between a circle of radius 1/2 and a wavy outer boundary
r = 1.2 + 0.15 cos(3 theta). It is cut into four deformed patches (MappedGrid2D),
each bounded by two radial cuts and two arcs. The arcs are degree-9 Bezier fits
with exact endpoints. In every patch xi runs radially outward and eta along the
angle, so the Jacobian is positive. MultiPatch2D joins the patches at their
radial cuts, (k, 1, 1) <-> (k + 1, 1, 0); the fourth interface closes the ring.
This is the topology of bspf-fluid's PatchMesh2D: faces (patch, axis, side) and
interfaces with an optional reverse flag.

Each patch differentiates with its own BSPF lines and exact metrics. With
overlap = 0 the lines end at the cuts, where the one-sided endpoint jets put the
largest error. With overlap = 8 (the default) each line crossing a cut is
extended by up to 8 ghost points into the neighbour (valued by local Lagrange
interpolation of the neighbour's samples), so the cut is an interior point of
the line. Nodes shared by several patches then take the average of the patches'
values. Before averaging the patches disagree there by the discretization error,
printed as the interface jump. The remaining largest errors sit at the physical
walls (inner circle, outer wavy boundary), which stay one-sided.

    python examples/basic/diff_multipatch_2d.py [--n 65] [--plot out.png]
"""
import argparse
from math import comb

import numpy as np

import bspf_cpp as bspf

OPTIONS = dict(degree=9, n_basis=18, lam=1e-6, endpoint_method="chebyshev", boundary_points=12,
               chebyshev_modes=12, chebyshev_alpha=1e-12, constraint_order=8)
PATCHES = 4


def inner(t):
    return 0.5 + 0 * t


def outer(t):
    return 1.2 + 0.15 * np.cos(3 * t)


def bezier_fit(points, degree):
    """Least-squares Bezier control points for sampled points, endpoints exact."""
    s = np.linspace(0, 1, len(points))
    B = np.stack([comb(degree, k) * s**k * (1 - s) ** (degree - k) for k in range(degree + 1)], axis=1)
    control = np.zeros((degree + 1, 2))
    control[0], control[-1] = points[0], points[-1]
    rhs = points - B[:, [0]] * control[0] - B[:, [-1]] * control[-1]
    control[1:-1] = np.linalg.lstsq(B[:, 1:-1], rhs, rcond=None)[0]
    return control


def arc(radius, t0, t1, degree=9):
    t = np.linspace(t0, t1, 200)
    r = radius(t)
    return bspf.SplineCurve(bezier_fit(np.column_stack((r * np.cos(t), r * np.sin(t))), degree), degree)


def ray(t):
    direction = np.array([np.cos(t), np.sin(t)])
    return bspf.SplineCurve(np.array([inner(t) * direction, outer(t) * direction]), 1)


def ring(n, overlap=8):
    """Four patches of n x 2n - 1 nodes (radial x angular), patch k spanning the
    angles [k pi/2, (k+1) pi/2], so the cells are roughly square."""
    grids = []
    for k in range(PATCHES):
        t0, t1 = 2 * np.pi * k / PATCHES, 2 * np.pi * (k + 1) / PATCHES
        # bottom (eta = 0), right (xi = 1), top (eta = 1), left (xi = 0)
        grids.append(bspf.MappedGrid2D(ray(t0), arc(outer, t0, t1), ray(t1), arc(inner, t0, t1), (n, 2 * n - 1),
                                       **OPTIONS))
    interfaces = [((k, 1, 1), ((k + 1) % PATCHES, 1, 0), False) for k in range(PATCHES)]
    return bspf.MultiPatch2D(grids, interfaces, overlap=overlap)


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


def errors(mp):
    """Relative max error per derivative, and the interface jump before averaging."""
    nodes = [(mp.patch(p).x, mp.patch(p).y) for p in range(mp.patch_count)]
    u = [f(x, y) for x, y in nodes]
    out = {}
    for orders, exact in EXACT.items():
        reference = [exact(x, y) for x, y in nodes]
        scale = max(np.abs(r).max() for r in reference)
        values = mp.partial(u, orders)
        error = max(np.abs(v - r).max() for v, r in zip(values, reference)) / scale
        jump = mp.shared_jump(mp.partial(u, orders, average_shared=False)) / scale
        out[orders] = (error, jump)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=33, help="radial nodes per patch (angular: 2n - 1)")
    parser.add_argument("--plot", metavar="PNG", help="save a figure")
    args = parser.parse_args()

    mp = ring(args.n)
    print(f"{mp.patch_count} patches of {args.n} x {2 * args.n - 1}, {mp.node_count} distinct nodes, "
          f"{len(mp.boundary_faces)} boundary faces (inner and outer rings)")
    one_sided, overlapped = errors(ring(args.n, overlap=0)), errors(mp)
    print("relative max error (interface jump before averaging): one-sided -> overlap 8")
    for orders in EXACT:
        (e0, j0), (e8, j8) = one_sided[orders], overlapped[orders]
        print(f"  {NAMES[orders]:5s} {e0:.1e} ({j0:.1e})  ->  {e8:.1e} ({j8:.1e})")

    print("refinement with overlap (relative max error):")
    sizes, table = (17, 25, 33, 49), {o: [] for o in EXACT}
    for n in sizes:
        result = errors(ring(n))
        for orders in EXACT:
            table[orders].append(result[orders][0])
        print(f"  N = {n:3d}: " + "  ".join(f"{NAMES[o]} {table[o][-1]:.1e}" for o in EXACT))
    assert all(e[-1] < min(e[0] / 10, 2e-9) for e in table.values())  # second derivatives level off near 1e-9 (round-off)
    if args.n >= 17:
        assert all(overlapped[o][1] < one_sided[o][1] for o in EXACT)

    if args.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)
        step = max(1, args.n // 12)
        fx = mp.partial([f(mp.patch(p).x, mp.patch(p).y) for p in range(mp.patch_count)], (1, 0))
        for p in range(mp.patch_count):
            g = mp.patch(p)
            colour = f"C{p}"
            for i in range(0, g.x.shape[0], step):
                axes[0].plot(g.x[i, :], g.y[i, :], "-", color=colour, lw=0.6)
            for j in range(0, g.x.shape[1], step):
                axes[0].plot(g.x[:, j], g.y[:, j], "-", color=colour, lw=0.6)
            axes[1].pcolormesh(g.x, g.y, f(g.x, g.y), shading="gouraud", vmin=-0.6, vmax=1.6)
            image = axes[2].pcolormesh(g.x, g.y, np.abs(fx[p] - EXACT[(1, 0)](g.x, g.y)), shading="gouraud",
                                       vmin=0, vmax=max(np.abs(fx[q] - EXACT[(1, 0)](mp.patch(q).x, mp.patch(q).y)).max()
                                                         for q in range(mp.patch_count)))
        fig.colorbar(image, ax=axes[2])
        axes[0].set(title=f"{mp.patch_count} patches (every {step}-th line)")
        axes[1].set(title="f(x, y)")
        axes[2].set(title="|f_x error|")
        for ax in axes[:3]:
            ax.set(xlabel="x", ylabel="y", aspect="equal")
        for orders, values in table.items():
            axes[3].semilogy(sizes, values, "o-", label=NAMES[orders])
        axes[3].set(title="refinement (overlap 8)", xlabel="radial nodes per patch", ylabel="relative max error")
        axes[3].set_xticks(sizes, [str(n) for n in sizes])
        axes[3].minorticks_off()
        axes[3].legend()
        fig.savefig(args.plot, dpi=120)
        print("saved", args.plot)


if __name__ == "__main__":
    main()
