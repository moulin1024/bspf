# SPDX-License-Identifier: GPL-3.0-or-later
"""1D interpolation of f(x) = sin(x / (beta + cos x)) between grid points.

An evaluation op is compiled for fixed query points (here 4x denser than the
grid) and can return values or derivatives of the same spline + Fourier
interpolant. Points must lie inside the grid; outside points raise.

    python examples/basic/interpolate_1d.py [--plot out.png]
"""
import argparse

import numpy as np

import bspf_cpp as bspf
from benchmarks import DOMAIN_1D, OPTIONS_1D, phase, phase_derivative, relative_max


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=257, help="grid points")
    parser.add_argument("--plot", metavar="PNG", help="save a figure")
    args = parser.parse_args()

    x = np.linspace(*DOMAIN_1D, args.n)
    line = bspf.Line(x, **OPTIONS_1D)
    f = phase(x)
    t = np.linspace(*DOMAIN_1D, 4 * args.n - 3)  # includes points between samples

    values = line.evaluate(t)(f)
    slopes = line.evaluate(t, derivative=1)(f)
    print(f"N = {args.n} samples, {t.size} query points")
    print(f"  values       relative max error {relative_max(values, phase(t)):.3e}")
    print(f"  derivatives  relative max error {relative_max(slopes, phase_derivative(t)):.3e}")
    linear = np.interp(t, x, f)
    print(f"  (linear interpolation: {relative_max(linear, phase(t)):.3e})")
    try:
        line.evaluate(np.array([x[-1] + 0.1]))
    except ValueError as error:
        print(f"  outside the grid: ValueError({error})")

    if args.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
        window = (t > 2.2) & (t < 4.2)
        axes[0].plot(t[window], phase(t)[window], label="exact")
        axes[0].plot(t[window], values[window], "--", label="BSPF")
        axes[0].plot(x[(x > 2.2) & (x < 4.2)], f[(x > 2.2) & (x < 4.2)], ".", label="samples")
        axes[0].set(title="values (zoom)", xlabel="x")
        axes[0].legend()
        axes[1].semilogy(t, np.abs(values - phase(t)) + 1e-17, label="BSPF")
        axes[1].semilogy(t, np.abs(linear - phase(t)) + 1e-17, label="linear")
        axes[1].set(title="absolute error", xlabel="x")
        axes[1].legend()
        fig.savefig(args.plot, dpi=120)
        print("saved", args.plot)


if __name__ == "__main__":
    main()
