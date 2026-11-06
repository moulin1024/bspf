# SPDX-License-Identifier: GPL-3.0-or-later
"""Linear viscous stiffness diagnostic, not a nonlinear shock-tube simulation.

Compare homogeneous-Dirichlet Laplacians on [0,1]: BSPF weak Galerkin,
Chebyshev-Lobatto collocation through FFTs, and Chebyshev weak Galerkin.
Dense matrices are assembled only to inspect the spectra, not as a proposed
time-stepping implementation. RK4 bounds are eigenvalue bounds for diffusion;
they are not full compressible Navier-Stokes stability guarantees.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import bspf_cpp as bspf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "basic"))
from compare_cylinder_chebyshev import cheb_diff

OPTIONS = dict(degree=9, n_basis=18, lam=1e-6, endpoint_method="chebyshev",
               boundary_points=16, chebyshev_modes=12, chebyshev_alpha=1e-12,
               constraint_order=4)
RK4_NEGATIVE_LIMIT = 2.785293563405282
METHODS = ("BSPF Galerkin", "Chebyshev FFT", "Chebyshev Galerkin")


def cheb_galerkin(n):
    """Exact polynomial mass/stiffness via n-point Gauss-Legendre quadrature."""
    identity = np.eye(n)
    coeff = np.fft.rfft(np.concatenate((identity, identity[-2:0:-1])), axis=0).real/(n-1)
    coeff[[0, -1]] *= .5
    z, weights = np.polynomial.legendre.leggauss(n)
    # z=1-2*x; x integration has weight weights/2.
    phi = np.polynomial.chebyshev.chebval(z, coeff).T
    dphi = np.polynomial.chebyshev.chebval(z, np.polynomial.chebyshev.chebder(coeff, scl=-2, axis=0)).T
    mass = phi.T@((weights/2)[:, None]*phi)
    stiffness = dphi.T@((weights/2)[:, None]*dphi)
    return -np.linalg.solve(mass[1:-1, 1:-1], stiffness[1:-1])


def run(sizes, nu):
    rows = []
    for n in sizes:
        identity = np.eye(n)
        x = np.linspace(0, 1, n)
        weak = bspf.GalerkinLine(bspf.Line(x, **OPTIONS))
        bspf_full = weak.second_derivative(identity)
        xc = (1-np.cos(np.pi*np.arange(n)/(n-1)))/2
        cheb_full = cheb_diff(cheb_diff(identity, 0), 0)[1:-1]
        for name, nodes, full in ((METHODS[0], x, bspf_full), (METHODS[1], xc, cheb_full),
                                 (METHODS[2], xc, cheb_galerkin(n))):
            eigenvalues = np.linalg.eigvals(full[:, 1:-1])
            rho = float(np.max(np.abs(eigenvalues)))
            assert eigenvalues.real.max() < 0
            assert np.max(np.abs(eigenvalues.imag)) < 1e-9*rho
            # First Dirichlet eigenmode independently checks sign and scaling.
            low = np.sort(eigenvalues.real)[-1]
            assert abs(low+np.pi**2)/np.pi**2 < 1e-5
            mode = np.sin(np.pi*nodes)
            mode_error = float(np.max(np.abs(full@mode+np.pi**2*mode[1:-1]))/np.pi**2)
            rows.append(dict(n=n, method=name, diffusivity=nu, spectral_radius=rho,
                             rk4_diffusion_dt=RK4_NEGATIVE_LIMIT/(nu*rho),
                             minimum_spacing=float(np.min(np.diff(nodes))),
                             sine_second_derivative_relative_error=mode_error))
        b, c, g = rows[-3:]
        print(f"n={n}: BSPF dt={b['rk4_diffusion_dt']:.6g}, "
              f"Chebyshev FFT dt={c['rk4_diffusion_dt']:.6g}, "
              f"Chebyshev Galerkin dt={g['rk4_diffusion_dt']:.6g}, "
              f"ratios={b['rk4_diffusion_dt']/c['rk4_diffusion_dt']:.2f} / "
              f"{b['rk4_diffusion_dt']/g['rk4_diffusion_dt']:.2f}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[33, 65, 129, 257])
    parser.add_argument("--nu", type=float, default=.01, help="constant diffusion coefficient, not dynamic viscosity")
    parser.add_argument("--json")
    parser.add_argument("--plot")
    args = parser.parse_args()
    if min(args.sizes) < 25 or not np.isfinite(args.nu) or args.nu <= 0:
        parser.error("sizes >=25 and finite positive nu required")
    rows = run(sorted(set(args.sizes)), args.nu)
    if args.json:
        Path(args.json).write_text(json.dumps(dict(bspf_options=OPTIONS,
            description="Dirichlet diffusion spectrum only; not a shock-tube solution or wall-clock benchmark",
            results=rows), indent=2)+"\n")
    if args.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for name in METHODS:
            group = [r for r in rows if r["method"] == name]
            sizes = [r["n"] for r in group]
            axes[0].loglog(sizes, [r["spectral_radius"] for r in group], "o-", label=name)
            axes[1].loglog(sizes, [r["rk4_diffusion_dt"] for r in group], "o-", label=name)
        axes[0].set(xlabel="Nodes", ylabel="Laplacian spectral radius", title="Measured diffusion stiffness")
        axes[1].set(xlabel="Nodes", ylabel="RK4 eigenvalue time-step limit", title=f"Diffusivity ν={args.nu:g}")
        for ax in axes:
            ax.legend()
            ax.grid(alpha=.2)
        fig.savefig(args.plot, dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
