# SPDX-License-Identifier: GPL-3.0-or-later
"""1D integration: recover f(x) = sin(x / (beta + cos x)) from samples of f'.

A quadrature op gives definite integrals over any [a, b] inside the grid; a
primitive op gives the antiderivative on the grid, fixed to zero at x_0 (add
the integration constant). Same field and settings as pybspf's
examples/operation/integrate_1d.

    python examples/basic/integrate_1d.py [--plot out.png]
"""
import argparse

import numpy as np

import bspf_cpp as bspf
from benchmarks import DOMAIN_1D, OPTIONS_1D, phase, phase_derivative, relative_max


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=513, help="grid points")
    parser.add_argument("--plot", metavar="PNG", help="save a figure")
    args = parser.parse_args()

    x = np.linspace(*DOMAIN_1D, args.n)
    line = bspf.Line(x, **OPTIONS_1D)
    f, df = phase(x), phase_derivative(x)

    # Definite integrals of f' are differences of f.
    whole = line.quadrature()(df)[0]
    print(f"N = {args.n}")
    print(f"  integral over [x0, x1]     error {abs(whole - (f[-1] - f[0])):.3e}")
    a, b = 1.0, 4.5
    part = line.quadrature(a, b)(df)[0]
    print(f"  integral over [{a}, {b}]     error {abs(part - (phase(b) - phase(a))):.3e}")
    print(f"  reversed bounds give the negative: {line.quadrature(b, a)(df)[0] == -part}")

    # Antiderivative on the grid, plus the integration constant f(x0).
    recovered = line.primitive(1)(df) + f[0]
    print(f"  antiderivative relative max error {relative_max(recovered, f):.3e}")

    print("refinement (antiderivative):")
    sizes, errors = (129, 257, 513, 1025), []
    for n in sizes:
        xn = np.linspace(*DOMAIN_1D, n)
        fn = phase(xn)
        errors.append(relative_max(bspf.Line(xn, **OPTIONS_1D).primitive(1)(phase_derivative(xn)) + fn[0], fn))
        print(f"  N = {n:5d}: {errors[-1]:.3e}")
    assert errors[-1] < errors[0]

    if args.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), constrained_layout=True)
        axes[0].plot(x, df)
        axes[0].set(title="samples of f'(x)", xlabel="x")
        axes[1].plot(x, f, label="exact f")
        axes[1].plot(x, recovered, "--", label="primitive")
        axes[1].set(title="recovered f(x)", xlabel="x")
        axes[1].legend()
        axes[2].loglog(sizes, errors, "o-")
        axes[2].set(title="convergence", xlabel="grid points", ylabel="relative max error")
        axes[2].set_xticks(sizes, [str(n) for n in sizes])
        axes[2].minorticks_off()
        fig.savefig(args.plot, dpi=120)
        print("saved", args.plot)


if __name__ == "__main__":
    main()
