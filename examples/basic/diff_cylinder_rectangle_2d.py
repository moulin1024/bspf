# SPDX-License-Identifier: GPL-3.0-or-later
"""MMS differentiation on [-2, 2] x [-1.5, 1.5] outside a radius-0.5 cylinder.

Four conforming transfinite patches join circle arcs to straight rectangle
sides. The polynomial circle approximation is checked independently. This
verifies differential operators on sampled exact data, not a PDE solve.
The manufactured field has a boundary layer of thickness delta at the circle:
u = (1 - exp(-(r - 0.5)/delta)) * u0. The mesh is unchanged.
For -Delta u = s, s = -lap_exact; Dirichlet data are u on both walls.

PYTHONPATH=python python examples/basic/diff_cylinder_rectangle_2d.py --plot cylinder_mms.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import bspf_cpp as bspf

from diff_multipatch_2d import bezier_fit, f as base_field, EXACT as BASE_EXACT, NAMES


def manufactured(x, y, thickness=0.1, shock_amplitude=0., shock_radius=1., shock_width=0.02):
    """Field and Cartesian derivatives, using the analytic radial Hessian.

    B = 1-exp(-(r-R)/delta), grad B = exp(-(r-R)/delta) n/delta,
    Hess B = exp(-(r-R)/delta) [(I-n n^T)/(delta r)-n n^T/delta^2].
    Optional shock: u = B*(u0+S), S=A/2*(1-tanh((r-rs)/w)).
    This is a smooth stationary shock-like MMS, not an Euler shock solution.
    """
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError("layer thickness must be finite and positive")
    if not np.isfinite(shock_amplitude) or shock_amplitude < 0:
        raise ValueError("shock amplitude must be finite and nonnegative")
    if not np.isfinite(shock_radius) or not 0.5 < shock_radius < 1.5:
        raise ValueError("shock radius must be between 0.5 and 1.5")
    if not np.isfinite(shock_width) or shock_width <= 0:
        raise ValueError("shock width must be finite and positive")
    r = np.hypot(x, y)
    nx, ny = x/r, y/r
    e = np.exp(-(r-0.5)/thickness)
    b = -np.expm1(-(r-0.5)/thickness)
    bx, by = e*nx/thickness, e*ny/thickness
    bxx = e*((1-nx*nx)/(thickness*r) - nx*nx/thickness**2)
    byy = e*((1-ny*ny)/(thickness*r) - ny*ny/thickness**2)
    bxy = -e*nx*ny*(1/(thickness*r) + 1/thickness**2)
    u = base_field(x, y)
    d = {o: fun(x, y) for o, fun in BASE_EXACT.items()}
    if shock_amplitude:
        z = (r-shock_radius)/shock_width
        t = np.tanh(z)
        # Stable sech^2, including the far tails where 1-tanh(z)^2 cancels.
        tail = np.exp(-2*np.abs(z))
        sech2 = 4*tail/(1+tail)**2
        sr = -shock_amplitude*sech2/(2*shock_width)
        srr = shock_amplitude*sech2*t/shock_width**2
        u = u + 0.5*shock_amplitude*(1-t)
        d[1, 0] = d[1, 0] + sr*nx
        d[0, 1] = d[0, 1] + sr*ny
        d[2, 0] = d[2, 0] + srr*nx*nx + sr/r*(1-nx*nx)
        d[0, 2] = d[0, 2] + srr*ny*ny + sr/r*(1-ny*ny)
        d[1, 1] = d[1, 1] + (srr-sr/r)*nx*ny
    result = {
        "u": b*u,
        "f_x": bx*u + b*d[1, 0],
        "f_y": by*u + b*d[0, 1],
        "f_xx": bxx*u + 2*bx*d[1, 0] + b*d[2, 0],
        "f_yy": byy*u + 2*by*d[0, 1] + b*d[0, 2],
        "f_xy": bxy*u + bx*d[0, 1] + by*d[1, 0] + b*d[1, 1],
    }
    # Independent radial product-rule expression for the Laplacian.
    result["lap"] = (b*(d[2, 0] + d[0, 2])
                     + 2*e/thickness*(nx*d[1, 0] + ny*d[0, 1])
                     + e*(1/(thickness*r) - 1/thickness**2)*u)
    return result


def lap_exact(x, y, thickness=0.1):
    return manufactured(x, y, thickness)["lap"]


def cylinder_boundaries():
    """Shared polynomial arcs and rectangle corners for either nodal scheme."""
    corners = np.array([[2., -1.5], [2., 1.5], [-2., 1.5], [-2., -1.5]])
    theta = np.unwrap(np.arctan2(corners[:, 1], corners[:, 0]))
    theta = np.append(theta, theta[0] + 2*np.pi)
    inner = 0.5 * corners / np.linalg.norm(corners, axis=1)[:, None]
    boundaries = []
    for k in range(4):
        j = (k + 1) % 4
        t = np.linspace(theta[k], theta[k+1], 300)
        points = 0.5 * np.column_stack((np.cos(t), np.sin(t)))
        points[0], points[-1] = inner[k], inner[j]
        arc = bspf.SplineCurve(bezier_fit(points, 15), 15)
        boundaries.append((inner[k], corners[k], inner[j], corners[j], arc))
    return boundaries


def cylinder_rectangle(n, overlap=8):
    if n < 17:
        raise ValueError("at least 17 nodes per direction are required")
    options = dict(degree=9, n_basis=18, lam=1e-6,
                   endpoint_method="chebyshev", boundary_points=min(n, 24),
                   chebyshev_modes=12, chebyshev_alpha=1e-12, constraint_order=8)
    def line(a, b):
        return bspf.SplineCurve(np.array([a, b]), 1)
    patches, circle_error = [], 0.
    for a, b, c, d, arc in cylinder_boundaries():
        sampled = arc.evaluate(np.linspace(0, 1, 2001))
        circle_error = max(circle_error, float(np.max(np.abs(np.linalg.norm(sampled, axis=1) - 0.5))))
        grid = bspf.MappedGrid2D(line(a, b), line(b, d), line(c, d), arc, (n, n), **options)
        if not np.all(grid.jacobian > 0):
            raise ValueError("folded geometry")
        patches.append(grid)
    interfaces = [((k, 1, 1), ((k+1) % 4, 1, 0), False) for k in range(4)]
    return bspf.MultiPatch2D(patches, interfaces, overlap=overlap), circle_error


def measure(mp, thickness=0.1):
    nodes = [(mp.patch(k).x, mp.patch(k).y) for k in range(4)]
    exact = [manufactured(x, y, thickness) for x, y in nodes]
    fields = [values["u"] for values in exact]
    numerical = {NAMES[o]: mp.partial(fields, o) for o in BASE_EXACT}
    numerical["lap"] = mp.laplacian(fields)
    errors = {}
    for name in numerical:
        reference = [values[name] for values in exact]
        absolute = max(float(np.max(np.abs(a-b))) for a, b in zip(numerical[name], reference))
        scale = max(float(np.max(np.abs(a))) for a in reference)
        errors[name] = {"absolute_max": absolute, "relative_max": absolute/scale}
    return errors, numerical


def plot_result(mp, numerical, rows, path, thickness=0.1):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), constrained_layout=True)
    lap_errors = [np.abs(numerical["lap"][k] - lap_exact(mp.patch(k).x, mp.patch(k).y, thickness)) for k in range(4)]
    vmax = max(float(e.max()) for e in lap_errors)
    for k in range(4):
        g = mp.patch(k)
        step = max(1, g.x.shape[0]//16)
        for i in range(0, g.x.shape[0], step):
            axes[0].plot(g.x[i, :], g.y[i, :], color=f"C{k}", lw=.5)
            axes[0].plot(g.x[:, i], g.y[:, i], color=f"C{k}", lw=.5)
        im = axes[1].pcolormesh(g.x, g.y, lap_errors[k], shading="gouraud", vmin=0, vmax=vmax)
        layer = -np.expm1(-(np.hypot(g.x, g.y)-0.5)/thickness)
        layer_im = axes[3].pcolormesh(g.x, g.y, layer, shading="gouraud", vmin=0, vmax=1)
    fig.colorbar(im, ax=axes[1])
    axes[0].set_title("Cylinder in rectangle: BSPF patches")
    axes[1].set_title("Absolute Laplacian error")
    fig.colorbar(layer_im, ax=axes[3])
    axes[3].set_title(f"Layer factor 1-exp(-(r-R)/δ), δ={thickness:g}")
    for ax in (axes[0], axes[1], axes[3]):
        ax.set(aspect="equal", xlabel="x", ylabel="y")
    for name in rows[0]["errors"]:
        axes[2].semilogy([r["n"] for r in rows], [r["errors"][name]["relative_max"] for r in rows], "o-", label=name)
    axes[2].set(xlabel="Nodes per patch direction", ylabel="Relative maximum error", title="MMS refinement (including walls)")
    axes[2].legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[33, 65, 129])
    parser.add_argument("--layer-thickness", type=float, default=0.1, help="positive exponential layer thickness (default: 0.1)")
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--plot", help="save a mesh/error/convergence figure")
    parser.add_argument("--json", help="save error metrics")
    args = parser.parse_args()
    if any(n < 17 for n in args.sizes) or args.overlap < 0:
        parser.error("sizes must be >= 17 and overlap >= 0")
    if not np.isfinite(args.layer_thickness) or args.layer_thickness <= 0:
        parser.error("layer thickness must be finite and positive")
    rows = []
    print("MMS: u = (1-exp(-(r-0.5)/delta)) * [sin(2x+0.5) cos(3y-0.2) + 0.5 exp(xy)]")
    print(f"Boundary-layer thickness delta={args.layer_thickness:g}; unchanged mesh")
    print("Domain: [-2,2] x [-1.5,1.5] minus disk r<0.5; four n x n patches")
    print("Relative maximum errors over all nodes (including physical boundaries):")
    for n in sorted(set(args.sizes)):
        mp, geometry_error = cylinder_rectangle(n, args.overlap)
        errors, numerical = measure(mp, args.layer_thickness)
        rows.append(dict(n=n, distinct_nodes=mp.node_count, circle_radius_error=geometry_error, errors=errors))
        print(f"n={n:3d} nodes={mp.node_count:6d}  " + "  ".join(f"{name}={e['relative_max']:.3e}" for name, e in errors.items()))
    print(f"Circle radius max error (2001 samples/arc): {geometry_error:.3e}")
    assert geometry_error < 1e-12
    assert all(np.isfinite(e["absolute_max"]) for r in rows for e in r["errors"].values())
    if args.json:
        Path(args.json).write_text(json.dumps(dict(overlap=args.overlap, layer_thickness=args.layer_thickness, results=rows), indent=2) + "\n")
    if args.plot:
        plot_result(mp, numerical, rows, args.plot, args.layer_thickness)


if __name__ == "__main__":
    main()
