# SPDX-License-Identifier: GPL-3.0-or-later
"""Compressible Euler / Navier--Stokes in a planar Laval nozzle on one mapped BSPF block.

Python only orchestrates: geometry (MappedGrid2D from four spline curves),
boundary conditions, the initial state, grid sequencing and diagnostics. The
numerics are bspf.CompressibleSolver2D (C++, include/bspf/compressible.hpp):
Galerkin BSPF weak form with fluxes at Gauss points, weak boundary fluxes,
Legendre spectral vanishing viscosity, local-time-step RK4 with Anderson
acceleration.

BSPF lines: degree 9 splines, Chebyshev endpoint jets, n_basis ~ N/3 per axis.

Geometry (lengths scaled by the throat half-height): walls y = +-h(x), x in [-L, L],
t = x / L, area ratio A at inlet and exit (h(+-L) = A, h(0) = 1, h' = 0 at the
throat and both ends):
    quartic  h = 1 + (A - 1)(2 t^2 - t^4)          throat curvature radius L^2 / (2(A - 1))
    sextic   h = 1 + (A - 1)(1 - (1 - t^2)^3)      also h'' = 0 at both ends
The walls are single Bezier curves, so MappedGrid2D's transfinite map is analytic.
The quartic wall turns the flow back sharply at the exit, where its compression
waves steepen; the sextic wall (L = 4) is the clean benchmark.

Units: stagnation density, temperature and sound speed are 1, so p0 = 1/gamma,
R = 1/gamma, T = c^2. Re = rho0 c0 h_throat / mu0, mu = mu0 T^0.76, Pr = 0.72.

Checks for inviscid shock-free flow: total enthalpy H = (E + p)/rho and entropy
p / rho^gamma are constant; the mass flow is the same through every section.
laval_nozzle_mms.py verifies the full Navier--Stokes operator on this grid.

    python examples/pde/laval_nozzle_2d.py [--n 97 33] [--sequence 49 17] [--multigrid 0 [--newton] --device cuda]
        [--anderson 8] [--re 0 | --re 2000 --cluster 8 --cfl 0.45] [--plot out.png]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import bspf_cpp as bspf

GAMMA = 1.4
P0 = 1.0 / GAMMA
# Chebyshev endpoint jets (derivative errors near rounding on smooth fields), no fit penalty
# (exact constant reproduction, needed for conservation). n_basis grows with the grid (see
# Nozzle.grid) so the fit amplification max|F| stays O(10-100), as FP32 on the GPU needs.
OPTIONS = dict(degree=9, lam=0.0, endpoint_method="chebyshev", boundary_points=16, chebyshev_modes=12,
               chebyshev_alpha=1e-12, constraint_order=4)


# --------------------------------------------------------------------------- geometry
def bezier_polynomial(fx, fy, degree):
    """Bezier control points of the polynomial curve s -> (fx(s), fy(s)), s in [0, 1]."""
    from math import comb
    s = 0.5 - 0.5 * np.cos(np.pi * np.arange(degree + 1) / degree)
    B = np.array([[comb(degree, k) * si**k * (1 - si)**(degree - k) for k in range(degree + 1)] for si in s])
    return np.linalg.solve(B, np.column_stack((fx(s), fy(s))))


class Nozzle:
    """shape "quartic": h = 1 + (A - 1)(2 t^2 - t^4), throat curvature radius L^2 / (2(A - 1));
    shape "sextic": h = 1 + (A - 1)(1 - (1 - t^2)^3), also h'' = 0 at both ends (no wall curvature
    at the exit), throat curvature radius L^2 / (6(A - 1))."""

    def __init__(self, length=3.0, area=1.5, shape="quartic"):
        self.L, self.A, self.shape = length, area, shape
        if shape not in ("quartic", "sextic"):
            raise ValueError("shape must be quartic or sextic")

    def h(self, x):
        t = x / self.L
        if self.shape == "sextic":
            return 1 + (self.A - 1) * (1 - (1 - t**2)**3)
        return 1 + (self.A - 1) * (2 * t**2 - t**4)

    def dh(self, x):
        t = x / self.L
        if self.shape == "sextic":
            return (self.A - 1) * 6 * t * (1 - t**2)**2 / self.L
        return (self.A - 1) * (4 * t - 4 * t**3) / self.L

    @property
    def degree(self):
        return 6 if self.shape == "sextic" else 4

    def curves(self):
        L, A = self.L, self.A
        wall = lambda sign: bspf.SplineCurve(
            bezier_polynomial(lambda s: -L + 2 * L * s, lambda s: sign * self.h(-L + 2 * L * s), self.degree), self.degree)
        straight = lambda p, q: bspf.SplineCurve(np.array([p, q], float), 1)
        return wall(-1), straight((L, -A), (L, A)), wall(1), straight((-L, -A), (-L, A))

    def grid(self, n_xi, n_eta, cluster=1.0, order="high"):
        """cluster >= 1: wall-to-centre node density ratio across the channel (1: uniform)."""
        # n_basis ~ N/3 per axis keeps the fit amplification max|F| near 10 (FP32 on the GPU needs it;
        # N/4 or a fixed 18 lets it grow like (N / n_basis)^3 into the hundreds)
        short = min(n_xi, n_eta)
        if order == "high":
            basis = lambda n: min(max(24, n // 3), n)
            options = dict(OPTIONS, n_basis=basis(n_xi), n_basis_eta=basis(n_eta),
                           boundary_points=min(OPTIONS["boundary_points"], short),
                           chebyshev_modes=min(OPTIONS["chebyshev_modes"], short - 1))
        else:  # lower-order trial spaces for coarse multigrid levels (p-coarsening)
            options = LOW_ORDER[order](n_xi, n_eta)
        if cluster > 1:
            options["dist_eta"] = wall_distribution(cluster)
        return bspf.MappedGrid2D(*self.curves(), (n_xi, n_eta), **options)


def wall_distribution(ratio, m=2):
    """Bernstein coefficients of psi with psi'(s) proportional to 1 + (ratio - 1)(2s - 1)^(2m).

    psi(s) is the fraction of the nodes before curve parameter s, so the node density at both
    walls is `ratio` times the density at the centre. psi is a polynomial of degree 2m + 1, and the
    Bernstein form is exact (interpolation at degree + 1 points).
    """
    from math import comb
    beta, d = ratio - 1.0, 2 * m + 1
    psi = lambda s: (s + beta * ((2 * s - 1)**d + 1) / (2 * d)) / (1 + beta / d)
    s = 0.5 - 0.5 * np.cos(np.pi * np.arange(d + 1) / d)
    B = np.array([[comb(d, k) * si**k * (1 - si)**(d - k) for k in range(d + 1)] for si in s])
    c = np.linalg.solve(B, psi(s))
    c[0], c[-1] = 0.0, 1.0
    return c


# Coarse multigrid levels (p-coarsening): splines of degree 5 or 3 with finite-difference endpoint
# jets, valid on much smaller grids than the degree-9 Chebyshev lines.
LOW_ORDER = {
    "medium": lambda a, b: dict(degree=5, lam=0.0, constraint_order=3, n_basis=min(max(12, a // 3), a),
                                n_basis_eta=min(max(12, b // 3), b), boundary_points=min(8, a, b)),
    "low": lambda a, b: dict(degree=3, lam=0.0, constraint_order=2, n_basis=min(max(6, a // 3), a),
                             n_basis_eta=min(max(6, b // 3), b), boundary_points=min(4, a, b)),
}


# --------------------------------------------------------------------------- quasi-1D theory
def area_ratio(M, g=GAMMA):
    return (1 / M) * ((2 / (g + 1)) * (1 + 0.5 * (g - 1) * M * M))**((g + 1) / (2 * (g - 1)))


def mach_from_area(a, supersonic, g=GAMMA):
    if a <= 1 + 1e-14:
        return 1.0
    return brentq(lambda M: area_ratio(M, g) - a, 1 + 1e-12, 50) if supersonic else \
        brentq(lambda M: area_ratio(M, g) - a, 1e-8, 1 - 1e-12)


def quasi_1d(nozzle, x):
    M = np.array([mach_from_area(nozzle.h(xi), xi > 0) for xi in np.ravel(x)]).reshape(np.shape(x))
    T = 1 / (1 + 0.5 * (GAMMA - 1) * M**2)
    return M, T, T**(1 / (GAMMA - 1)), M * np.sqrt(T)  # Mach, T, rho, u


def critical_mass_flow(g=GAMMA):
    """rho* c* A* for the planar nozzle (A* = 2 throat half-heights)."""
    return 2 * (2 / (g + 1))**(1 / (g - 1)) * np.sqrt(2 / (g + 1))


# --------------------------------------------------------------------------- orchestration
def make_solver(nozzle, n_xi, n_eta, reynolds=0.0, cluster=1.0, order="high", **options):
    """C++ solver on the nozzle block: inlet (total conditions), supersonic outlet, walls."""
    grid = nozzle.grid(n_xi, n_eta, cluster, order)
    wall = bspf.BoundaryKind.no_slip_wall if reynolds > 0 else bspf.BoundaryKind.slip_wall
    boundaries = [bspf.Boundary(bspf.BoundaryKind.inlet, total_pressure=P0, total_temperature=1.0),
                  bspf.Boundary(bspf.BoundaryKind.outlet), bspf.Boundary(wall), bspf.Boundary(wall)]
    return bspf.CompressibleSolver2D(grid, boundaries, reynolds=reynolds, gamma=GAMMA, **options)


def conservative(rho, u, v, p):
    return np.stack((rho, rho * u, rho * v, p / (GAMMA - 1) + 0.5 * rho * (u * u + v * v)), axis=-1)


def primitives(U):
    rho = U[..., 0]
    u, v = U[..., 1] / rho, U[..., 2] / rho
    return rho, u, v, (GAMMA - 1) * (U[..., 3] - 0.5 * rho * (u * u + v * v))


def initial_state(nozzle, solver, viscous):
    """Quasi-1D isentropic flow, streamlines following the wall slope (no-slip layer if viscous)."""
    g = solver.grid
    x, y = g.x, g.y
    _, T, rho, u = quasi_1d(nozzle, x[:, 0])
    rho, T, u = (np.repeat(q[:, None], x.shape[1], axis=1) for q in (rho, T, u))
    v = u * y * nozzle.dh(x) / nozzle.h(x)
    if viscous:
        eta = np.linspace(0, 1, x.shape[1])[None, :]
        layer = np.tanh(eta / 0.1) * np.tanh((1 - eta) / 0.1)
        u, v = u * layer, v * layer
    return conservative(rho, u, v, rho * T / GAMMA)


def invariants(U):
    rho, u, v, p = primitives(U)
    return np.abs((U[..., 3] + p) / rho * (GAMMA - 1) - 1), np.abs(p / rho**GAMMA * GAMMA - 1)


def mach(U):
    rho, u, v, p = primitives(U)
    return np.hypot(u, v) / np.sqrt(GAMMA * p / rho)


def run(nozzle, n, reynolds, cluster, sequence=None, steps=200000, tolerance=1e-10, anderson=8, verbose=True,
        **options):
    """Steady state on grid n, optionally started from a converged coarser grid (grid sequencing)."""
    stages = [tuple(sequence), tuple(n)] if sequence else [tuple(n)]
    U, coarse, log = None, None, []
    for shape in stages:
        solver = make_solver(nozzle, *shape, reynolds=reynolds, cluster=cluster, **options)
        U = initial_state(nozzle, solver, reynolds > 0) if coarse is None else solver.prolong(coarse, U)
        report = (lambda step, r: print(f"  {shape[0]}x{shape[1]} step {step:6d}  residual {r:.3e}", flush=True)) \
            if verbose else None
        U, info = solver.solve(U, max_steps=steps, tolerance=tolerance, anderson=anderson,
                               check=50, progress=(lambda s, r, f=report: f(s, r) if s % 1000 == 0 else None)
                               if verbose else None)
        info["n"] = list(shape)
        log.append(info)
        coarse = solver
    return solver, U, log


ORDERS = ("high", "medium", "low")


def order_valid(shape, order, implicit=True):
    """Where each trial space works on the nozzle (measured: smaller grids fail to construct or
    diverge). Line-implicit smoothing: degree-9 Chebyshev needs 17 points per axis, degree 5
    needs 9 and 25 along the longer axis, degree 3 needs 5. Explicit smoothing: degree 5 also
    needs 17 per axis, degree 3 needs 13 along the longer axis."""
    a, b = min(shape), max(shape)
    if implicit:
        return {"high": a >= 17, "medium": a >= 9 and b >= 25, "low": a >= 5}[order]
    return {"high": a >= 17, "medium": a >= 17, "low": a >= 5 and b >= 13}[order]


def hierarchy(n, depth=0, min_points=5, implicit=True):
    """Multigrid levels (shape, order), finest first, mixing h- and p-coarsening: halve the grid
    (n -> (n + 1) / 2 per axis, an axis below min_points is kept: semi-coarsening) while the
    current order stays valid, otherwise lower the order on the same nodes. depth 0: as deep as
    possible; otherwise the first `depth` levels."""
    levels = [(tuple(n), "high")]
    while depth == 0 or len(levels) < depth:
        shape, order = levels[-1]
        half = tuple((v + 1) // 2 if (v + 1) // 2 >= min_points else v for v in shape)
        if half != shape and order_valid(half, order, implicit):
            levels.append((half, order))
        elif order != ORDERS[-1]:
            levels.append((shape, ORDERS[ORDERS.index(order) + 1]))  # p-coarsening
        else:
            break
    return levels


def run_multigrid(nozzle, n, reynolds, cluster, levels, device="cpu", tolerance=1e-10, verbose=True, cycle=None,
                  **options):
    """FAS multigrid (bspf.multigrid_solve, C++), finest grid n, `levels` levels of hierarchy()
    (0: as deep as possible); device cpu or cuda."""
    plan = hierarchy(n, levels, implicit=options.get("line_implicit", "full") != "none")
    shapes = [s for s, _ in plan]
    cpus = [make_solver(nozzle, *s, reynolds=reynolds, cluster=cluster, order=o, **options) for s, o in plan]
    U = initial_state(nozzle, cpus[0], reynolds > 0)
    report = (lambda c, r: print(f"  cycle {c:4d}  residual {r:.3e}", flush=True) if c % 10 == 0 else None) \
        if verbose else None
    cycle = dict(cycle or {})
    newton = cycle.pop("newton", False)
    if newton:  # Newton-Krylov, FAS-preconditioned
        cycle.pop("max_cycles", None)
        report = (lambda k, r, its: print(f"  newton {k:3d}  residual {r:.3e}  ({its} krylov)", flush=True)) \
            if verbose else None
    precision = cycle.pop("precision", "double")
    module = bspf.cuda if device == "cuda" else bspf
    if newton and device == "cuda":
        # mixed precision: Newton, FGMRES and the Jacobian products on an FP64 device solver, the FAS
        # preconditioner on FP32 device levels
        levels_ = [bspf.cuda.CompressibleSolver2D(c, refresh=10, precision="float") for c in cpus]
        residual = bspf.cuda.CompressibleSolver2D(cpus[0], precision="double")
        U, info = module.newton_krylov_solve(levels_, U, tolerance=tolerance, progress=report, residual=residual,
                                             **cycle)
    elif newton:
        U, info = module.newton_krylov_solve(cpus, U, tolerance=tolerance, progress=report, **cycle)
    else:
        levels_ = [bspf.cuda.CompressibleSolver2D(c, refresh=10, precision=precision) for c in cpus] \
            if device == "cuda" else cpus
        U, info = module.multigrid_solve(levels_, U, tolerance=tolerance, progress=report, **cycle)
    info["n"] = list(n)
    info["levels"] = [f"{s[0]}x{s[1]} {o}" for s, o in plan]
    return cpus[0], U, [info]


def summary(nozzle, solver, U, log, reynolds):
    m = solver.mass_flow(U)
    out = dict(n=list(solver.shape[:2]), reynolds=reynolds, stages=[
        {k: v for k, v in info.items() if k != "history"} for info in log],
        seconds=sum(info["seconds"] for info in log), mass_flow=float(m.mean()),
        discharge_coefficient=float(m.mean() / critical_mass_flow()),
        mass_flow_variation=float((m.max() - m.min()) / m.mean()))
    if out["reynolds"] == 0:
        dH, ds = invariants(U)
        out.update(enthalpy_error=float(dH.max()), entropy_error=float(ds.max()))
    M = mach(U)
    out.update(exit_mach_center=float(M[-1, U.shape[1] // 2]),
               exit_mach_quasi1d=float(mach_from_area(nozzle.A, True)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, nargs=2, default=(97, 33), help="nodes along xi and eta")
    parser.add_argument("--sequence", type=int, nargs=2, help="coarse grid solved first (grid sequencing)")
    parser.add_argument("--re", type=float, default=0.0, help="Reynolds number (0: Euler)")
    parser.add_argument("--shape", default="sextic", choices=("quartic", "sextic"), help="wall polynomial")
    parser.add_argument("--length", type=float, default=4.0, help="nozzle half-length L")
    parser.add_argument("--cluster", type=float, default=1.0,
                        help="wall-to-centre node density ratio across the channel (1: uniform)")
    parser.add_argument("--cfl", type=float, default=0.6)
    parser.add_argument("--line-implicit", default=None, choices=("none", "viscous", "full"),
                        help="wall-normal line-implicit preconditioning (default: full on a single grid, none as "
                             "the multigrid smoother: explicit smoothing gives grid-independent cycles on the "
                             "Euler nozzle, line-implicit smoothing diverges from 193 x 129 on)")
    parser.add_argument("--svv", type=float, default=0.2, help="spectral vanishing viscosity strength")
    parser.add_argument("--cutoff", type=float, default=0.5, help="SVV cutoff mode fraction")
    parser.add_argument("--quadrature", type=int, default=5, help="Gauss points per cell")
    parser.add_argument("--anderson", type=int, default=8, help="Anderson depth (0: plain RK4)")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--multigrid", type=int, default=1,
                        help="FAS multigrid levels (1: single grid, 0: as deep as possible, h and p coarsening)")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="multigrid on the CPU or the GPU (FP32)")
    parser.add_argument("--mg-smooth", type=int, nargs=3, default=(10, 10, 60), metavar=("PRE", "POST", "COARSE"))
    parser.add_argument("--mg-cycle", type=int, default=1, help="1: V-cycle, 2: W-cycle")
    parser.add_argument("--mg-damping", type=float, default=0.5, help="coarse-correction damping omega")
    parser.add_argument("--mg-max", type=int, default=200, help="maximum cycles")
    parser.add_argument("--precision", default="double", choices=("double", "float"),
                        help="device precision of plain multigrid levels (Newton: FP64 residual, FP32 preconditioner)")
    parser.add_argument("--newton", action="store_true",
                        help="Newton-Krylov (FGMRES) preconditioned by the multigrid cycle (needs --multigrid)")
    parser.add_argument("--json", help="write a summary")
    parser.add_argument("--save", help="save x, y and the state U to an .npz file")
    parser.add_argument("--plot", help="save a figure")
    args = parser.parse_args()
    nozzle = Nozzle(args.length, shape=args.shape)
    if args.line_implicit is None:
        args.line_implicit = "full" if args.multigrid == 1 else "none"
    options = dict(cfl=args.cfl, svv=args.svv, svv_cutoff=args.cutoff, quadrature=args.quadrature,
                   line_implicit=args.line_implicit)
    if args.multigrid != 1:
        cycle = dict(pre=args.mg_smooth[0], post=args.mg_smooth[1], coarse=args.mg_smooth[2],
                     cycle_index=args.mg_cycle, damping=args.mg_damping, max_cycles=args.mg_max, newton=args.newton,
                     precision=args.precision)
        solver, U, log = run_multigrid(nozzle, args.n, args.re, args.cluster, args.multigrid, args.device, args.tol,
                                       cycle=cycle, **options)
    else:
        if args.device == "cuda":
            raise SystemExit("--device cuda is used with --multigrid (the single-grid GPU path: bspf.cuda.CompressibleSolver2D)")
        solver, U, log = run(nozzle, args.n, args.re, args.cluster, args.sequence, args.steps, args.tol,
                             args.anderson, **options)
    result = summary(nozzle, solver, U, log, args.re)
    print(json.dumps(result, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(dict(result, history=[info["history"] for info in log]), indent=1))
    if args.save:
        np.savez(args.save, x=solver.grid.x, y=solver.grid.y, U=U)
    if args.plot:
        plot(nozzle, solver, U, log[-1]["history"], args.re > 0, args.plot)


def plot(nozzle, solver, U, history, viscous, path):
    import matplotlib.pyplot as plt
    g = solver.grid
    x, y = g.x, g.y
    fig, ax = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    M = mach(U)
    im = ax[0, 0].pcolormesh(x, y, M, shading="gouraud", cmap="viridis")
    ax[0, 0].contour(x, y, M, levels=[1.0], colors="w", linewidths=1)
    ax[0, 0].set(aspect="equal", title="Mach number (white: sonic line)")
    fig.colorbar(im, ax=ax[0, 0])
    xs = np.linspace(-nozzle.L, nozzle.L, 400)
    j = x.shape[1] // 2
    ax[0, 1].plot(xs, quasi_1d(nozzle, xs)[0], "k-", lw=1, label="quasi-1D")
    ax[0, 1].plot(x[:, j], M[:, j], "o", ms=3, label="centerline")
    ax[0, 1].plot(x[:, 0], M[:, 0], "s", ms=3, label="wall")
    ax[0, 1].set(xlabel="x", ylabel="Mach", title="Mach along the nozzle")
    ax[0, 1].legend()
    it, r = np.array(history).T
    ax[1, 0].semilogy(it, r)
    ax[1, 0].set(xlabel="step", ylabel="residual (rms, all components)", title="convergence to steady state")
    if not viscous:
        _, ds = invariants(U)
        im = ax[1, 1].pcolormesh(x, y, np.log10(np.maximum(ds, 1e-16)), shading="gouraud", cmap="magma")
        ax[1, 1].set(aspect="equal", title="log10 |entropy error|")
        fig.colorbar(im, ax=ax[1, 1])
    else:
        rho, u, v, p = primitives(U)
        i = int(np.argmin(np.abs(x[:, 0] - 2.0)))
        ax[1, 1].plot(u[i], y[i], ".-")
        ax[1, 1].set(xlabel="u", ylabel="y", title=f"velocity profile at x = {x[i, 0]:.2f}")
    fig.savefig(path, dpi=120)
    print("saved", path)


if __name__ == "__main__":
    main()
