# SPDX-License-Identifier: GPL-3.0-or-later
"""1D differentiation of f(x) = sin(x / (beta + cos x)) on [0, 2 pi].

Compile the derivative once from a Line, apply it to samples, compare with
the exact derivative and with the spline-only derivative (no Fourier
correction), then refine the grid. Same field and settings as pybspf's
examples/operation/differentiate_1d.

    python examples/basic/diff_1d.py [--plot out.png]
"""
import argparse
import time

import numpy as np

import bspf_cpp as bspf
from benchmarks import DOMAIN_1D, OPTIONS_1D, phase, phase_derivative, relative_max


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=513, help="grid points")
    parser.add_argument("--plot", metavar="PNG", help="save a figure")
    args = parser.parse_args()

    x = np.linspace(*DOMAIN_1D, args.n)
    line = bspf.Line(x, **OPTIONS_1D)  # geometry, basis and constrained fit
    d1 = line.derivative(1)            # compiled operator, reused below
    f, exact = phase(x), phase_derivative(x)

    df = d1(f)
    spline_only = line.derivative(1, correction=False)(f)
    print(f"N = {args.n}, degree {line.degree}, {line.n_basis} B-splines")
    print(f"  BSPF        relative max error {relative_max(df, exact):.3e}")
    print(f"  spline only relative max error {relative_max(spline_only, exact):.3e}")

    # Steady-state cost: one call on a batch of 1000 fields (columns).
    batch = np.outer(f, np.linspace(0.5, 1.5, 1000))
    d1(batch)
    t0 = time.perf_counter()
    d1(batch)
    print(f"  {batch.shape[1]} fields in {1e3 * (time.perf_counter() - t0):.2f} ms")

    print("refinement:")
    sizes, errors = (129, 257, 513, 1025), []
    for n in sizes:
        xn = np.linspace(*DOMAIN_1D, n)
        errors.append(relative_max(bspf.Line(xn, **OPTIONS_1D).derivative(1)(phase(xn)), phase_derivative(xn)))
        print(f"  N = {n:5d}: {errors[-1]:.3e}")
    assert errors[-1] < errors[0]

    if args.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), constrained_layout=True)
        axes[0].plot(x, f)
        axes[0].set(title="f(x)", xlabel="x")
        axes[1].plot(x, exact, label="exact")
        axes[1].plot(x, df, "--", label="BSPF")
        axes[1].set(title="f'(x)", xlabel="x")
        axes[1].legend()
        axes[2].loglog(sizes, errors, "o-")
        axes[2].set(title="convergence", xlabel="grid points", ylabel="relative max error")
        axes[2].set_xticks(sizes, [str(n) for n in sizes])
        axes[2].minorticks_off()
        fig.savefig(args.plot, dpi=120)
        print("saved", args.plot)


if __name__ == "__main__":
    main()
