# SPDX-License-Identifier: GPL-3.0-or-later
"""2D differentiation of a smooth background plus a circular tanh front.

A Grid of two Lines differentiates along either axis of an (nx, ny) array,
with or without the Fourier correction, and gives the gradient. Same field
and settings as pybspf's examples/operation/differentiate_2d (the random
mode amplitudes come from NumPy rather than JAX).

    python examples/basic/diff_2d.py [--plot out.png]
"""
import argparse

import numpy as np

import bspf_cpp as bspf
from benchmarks import DOMAIN_2D, OPTIONS_2D, front, front_dx, front_dy, odd_sizes, relative_l2


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", metavar="PNG", help="save a figure")
    args = parser.parse_args()

    x, y = np.linspace(*DOMAIN_2D, 97), np.linspace(*DOMAIN_2D, 81)
    grid = bspf.Grid.uniform((x, y), **OPTIONS_2D)
    u = front(x, y)

    ux = grid.partial(u, (1, 0))
    print(f"grid {x.size} x {y.size}")
    print(f"  d/dx relative L2 error {relative_l2(ux, front_dx(x, y)):.3e}")
    ux_spline = grid.line(0).derivative(1, correction=False)(u, axis=0)
    print(f"  d/dx spline only       {relative_l2(ux_spline, front_dx(x, y)):.3e}")
    gradient = grid.gradient(u)  # (2, nx, ny)
    print(f"  gradient relative L2   {relative_l2(gradient, np.stack((front_dx(x, y), front_dy(x, y)))):.3e}")

    print("refinement (d/dx on N x N):")
    sizes, errors = odd_sizes(33, 513, 9), []
    for n in sizes:
        xn = np.linspace(*DOMAIN_2D, n)
        gn = bspf.Grid.uniform((xn, xn), **OPTIONS_2D)
        errors.append(relative_l2(gn.partial(front(xn, xn), (1, 0)), front_dx(xn, xn)))
        print(f"  N = {n:3d}: {errors[-1]:.3e}")
    assert errors[-1] < errors[0]

    if args.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.4), constrained_layout=True)
        X, Y = np.meshgrid(x, y, indexing="ij")
        for ax, values, title in zip(axes, (u, front_dx(x, y), np.abs(ux - front_dx(x, y))),
                                     ("field", "exact d/dx", "absolute error")):
            image = ax.pcolormesh(X, Y, values, shading="auto")
            ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
            fig.colorbar(image, ax=ax)
        axes[3].loglog(sizes, errors, "o-")
        axes[3].set(title="convergence", xlabel="points per axis", ylabel="relative L2 error")
        axes[3].set_xticks([33, 65, 129, 257, 513], ["33", "65", "129", "257", "513"])
        axes[3].minorticks_off()
        fig.savefig(args.plot, dpi=120)
        print("saved", args.plot)


if __name__ == "__main__":
    main()
