# SPDX-License-Identifier: GPL-3.0-or-later
"""Manufactured-solution check of the SBP-SAT BSPF Navier--Stokes solver on the Laval-nozzle block.

A smooth state (rho, u, v, p)(x, y) is forced into the steady compressible
Navier--Stokes equations by the source S = div(F - F_v), derived symbolically
(sympy) with the solver's own constitutive laws (mu = T^0.76 / Re, Pr = 0.72).
All four faces of the curved nozzle block impose the manufactured state weakly,
through the same Rusanov numerical flux and gradient lifting that the nozzle's
inlet and walls use (the sextic nozzle block, L = 4). Two errors per grid:

  * truncation: max |rhs(U_exact)| (the spatial operator applied to the exact
    state, metrics, SAT terms and spectral vanishing viscosity included);
  * solution: max |U - U_exact| after marching to steady state.

Numerics in C++ (bspf.CompressibleSolver2D); faces are Exterior boundaries
holding the manufactured state at their nodes and Gauss points.

    python examples/pde/laval_nozzle_mms.py [--re 100] [--sizes 33x13 49x17 65x23 97x33]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp

import bspf_cpp as bspf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from laval_nozzle_2d import GAMMA, Nozzle, conservative  # noqa: E402

PRANDTL = 0.72


def manufactured(reynolds):
    x, y = sp.symbols("x y")
    rho = 1 + sp.Rational(1, 5) * sp.sin(x / 2 + sp.Rational(3, 10)) * sp.cos(7 * y / 10)
    u = sp.Rational(9, 10) + sp.Rational(1, 5) * sp.cos(2 * x / 5) * sp.sin(3 * y / 5 + sp.Rational(1, 5))
    v = sp.Rational(3, 20) * sp.sin(3 * x / 5 + sp.Rational(1, 10)) * sp.cos(y / 2)
    p = sp.Rational(7, 10) + sp.Rational(1, 10) * sp.cos(3 * x / 10 + 2 * y / 5)
    g = sp.Float(GAMMA)
    E = p / (g - 1) + rho * (u**2 + v**2) / 2
    F = [rho * u, rho * u**2 + p, rho * u * v, u * (E + p)]
    G = [rho * v, rho * u * v, rho * v**2 + p, v * (E + p)]
    if reynolds > 0:
        T = g * p / rho
        mu = T**sp.Float(0.76) / reynolds
        kappa = mu / ((g - 1) * PRANDTL)
        ux, uy, vx, vy = sp.diff(u, x), sp.diff(u, y), sp.diff(v, x), sp.diff(v, y)
        div = ux + vy
        txx, tyy, txy = mu * (2 * ux - 2 * div / 3), mu * (2 * vy - 2 * div / 3), mu * (uy + vx)
        Fv = [0, txx, txy, u * txx + v * txy + kappa * sp.diff(T, x)]
        Gv = [0, txy, tyy, u * txy + v * tyy + kappa * sp.diff(T, y)]
        F = [a - b for a, b in zip(F, Fv)]
        G = [a - b for a, b in zip(G, Gv)]
    S = [sp.diff(a, x) + sp.diff(b, y) for a, b in zip(F, G)]
    state = sp.lambdify((x, y), [rho, u, v, p], "numpy")
    source = sp.lambdify((x, y), S, "numpy")
    return (lambda X, Y: [np.broadcast_to(q, X.shape) for q in state(X, Y)],
            lambda X, Y: np.stack([np.broadcast_to(q, X.shape) for q in source(X, Y)], axis=-1))


def check(n_xi, n_eta, reynolds, steps, tolerance, anderson=8, **options):
    state, source = manufactured(reynolds)
    nozzle = Nozzle(4.0, shape="sextic")
    grid = nozzle.grid(n_xi, n_eta)
    x, y = grid.x, grid.y
    exact = conservative(*state(x, y))
    face_nodes = [(x[0], y[0]), (x[-1], y[-1]), (x[:, 0], y[:, 0]), (x[:, -1], y[:, -1])]
    # all faces impose the manufactured state weakly; the Gauss-point states come from the solver's face points
    probe = bspf.CompressibleSolver2D(grid, [bspf.Boundary(bspf.BoundaryKind.slip_wall)] * 4, reynolds=reynolds,
                                      quadrature=options.get("quadrature", 5))
    boundaries = []
    for k in range(4):
        pts = probe.face_points(k)
        boundaries.append(bspf.Boundary(bspf.BoundaryKind.exterior, nodes=conservative(*state(*face_nodes[k])),
                                        points=conservative(*state(pts[:, 0], pts[:, 1]))))
    options.setdefault("cfl", 0.45 if reynolds > 0 else 0.6)
    solver = bspf.CompressibleSolver2D(grid, boundaries, reynolds=reynolds, gamma=GAMMA, **options)
    solver.set_source(np.ascontiguousarray(source(x, y)))
    truncation = float(np.abs(solver.rhs(exact)).max())
    U, info = solver.solve(exact.copy(), max_steps=steps, tolerance=tolerance, anderson=anderson)
    return dict(n=[n_xi, n_eta], reynolds=reynolds, truncation=truncation,
                solution=float(np.abs(U - exact).max() / np.abs(exact).max()),
                steps=info["steps"], residual=info["residual"], seconds=info["seconds"], **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--re", type=float, default=100.0, help="Reynolds number (0: Euler)")
    parser.add_argument("--sizes", nargs="+", default=["33x13", "49x17", "65x23", "97x33"])
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--tol", type=float, default=1e-11)
    parser.add_argument("--quadrature", type=int, default=5, help="Gauss points per cell")
    parser.add_argument("--json", help="write the results")
    args = parser.parse_args()
    rows = []
    for size in args.sizes:
        n_xi, n_eta = (int(v) for v in size.split("x"))
        row = check(n_xi, n_eta, args.re, args.steps, args.tol, quadrature=args.quadrature)
        rows.append(row)
        print(f"{n_xi:4d} x {n_eta:3d}  truncation {row['truncation']:.2e}  solution {row['solution']:.2e}"
              f"  ({row['steps']} steps, {row['seconds']:.1f} s, residual {row['residual']:.1e})", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
