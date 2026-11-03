# SPDX-License-Identifier: GPL-3.0-or-later
"""3D vector calculus on a shifted Taylor-Green vortex (non-periodic box).

Gradient (full Jacobian), curl, divergence and Laplacian of a velocity field
on [0, 3 pi / 2]^3 with Chebyshev endpoint jets. Velocity components are a
leading axis for curl/divergence, or a trailing batch axis for gradient and
Laplacian. Same field and settings as pybspf's examples/operation/differentiate_3d.

    python examples/basic/diff_3d.py [--plot out.png]
"""
import argparse

import numpy as np

import bspf_cpp as bspf
from benchmarks import LENGTH_3D, OPTIONS_3D, odd_sizes, relative_l2, vortex


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", metavar="PNG", help="save a figure")
    args = parser.parse_args()

    x, y, z = (np.linspace(0, LENGTH_3D, n) for n in (33, 41, 49))
    grid = bspf.Grid.uniform((x, y, z), **OPTIONS_3D)
    velocity, exact_jacobian, exact_vorticity = vortex(x, y, z)
    field = np.ascontiguousarray(np.moveaxis(velocity, 0, -1))  # (nx, ny, nz, 3)
    gaps = [float(np.max(np.abs(np.take(field, 0, axis=i) - np.take(field, -1, axis=i)))) for i in range(3)]
    print(f"grid {x.size} x {y.size} x {z.size}; opposite-face mismatch {np.round(gaps, 2)} (not periodic)")

    jacobian = grid.gradient(field)          # (derivative axis, nx, ny, nz, component)
    vorticity = grid.curl(velocity)
    divergence = grid.divergence(velocity)
    laplacian = grid.laplacian(field)
    print(f"  Jacobian   relative L2 {relative_l2(jacobian, exact_jacobian):.3e}")
    print(f"  curl       relative L2 {relative_l2(vorticity, exact_vorticity):.3e}")
    print(f"  divergence max |div u| {np.max(np.abs(divergence)):.3e}")
    print(f"  Laplacian  relative L2 {relative_l2(laplacian, -3 * field):.3e}")

    print("refinement (Jacobian on N^3):")
    sizes, errors = odd_sizes(17, 65, 5), []
    for n in sizes:
        c = np.linspace(0, LENGTH_3D, n)
        un, reference, _ = vortex(c, c, c)
        gn = bspf.Grid.uniform((c, c, c), **OPTIONS_3D)
        errors.append(relative_l2(gn.gradient(np.ascontiguousarray(np.moveaxis(un, 0, -1))), reference))
        print(f"  N = {n:2d}: {errors[-1]:.3e}")
    assert errors[-1] < errors[0] / 100

    if args.plot:
        import matplotlib.pyplot as plt
        mid = z.size // 2
        error = np.sqrt(np.sum((jacobian - exact_jacobian) ** 2, axis=(0, 4)))
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
        for ax, values, title in zip(axes[:2], (exact_jacobian[0, :, :, mid, 0], error[:, :, mid]),
                                     ("exact du/dx", "Jacobian error (Frobenius)")):
            image = ax.pcolormesh(x, y, values.T, shading="auto")
            ax.set(title=f"{title}, z = {z[mid]:.2f}", xlabel="x", ylabel="y", aspect="equal")
            fig.colorbar(image, ax=ax)
        axes[2].loglog(sizes, errors, "o-")
        axes[2].set(title="convergence", xlabel="points per axis", ylabel="Jacobian relative L2")
        axes[2].set_xticks(sizes, [str(n) for n in sizes])
        axes[2].minorticks_off()
        fig.savefig(args.plot, dpi=120)
        print("saved", args.plot)


if __name__ == "__main__":
    main()
