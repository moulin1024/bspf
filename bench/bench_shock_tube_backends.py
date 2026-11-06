# SPDX-License-Identifier: GPL-3.0-or-later
"""Matched FFT/dense BSPF Navier--Stokes operators and optional full evolutions.

Set BLAS thread counts externally. Timings include Python and physical fluxes.
Dense mode is the reference, not a Chebyshev or finite-volume solver.
"""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"examples"/"pde"))
from shock_tube_1d import BSPFNavierStokes, initial_state, primitives


def compare(n, repeats, final_time, mu):
    solvers = {name: BSPFNavierStokes(n, mu=mu, backend=name) for name in ("fft", "dense")}
    state = initial_state(solvers["fft"].x)
    values = {name: solver.rhs(state) for name, solver in solvers.items()}
    times = {name: [] for name in solvers}
    for repetition in range(repeats):
        # Alternate order so each backend is first equally often.
        for name in (("fft", "dense") if repetition % 2 == 0 else ("dense", "fft")):
            start = perf_counter()
            for _ in range(5):
                solvers[name].rhs(state)
            times[name].append((perf_counter()-start)/5)
    median = {name: float(np.median(values)) for name, values in times.items()}
    row = dict(n=n, rhs_seconds=median, rhs_speedup=median["dense"]/median["fft"],
               rhs_max_absolute_difference=float(np.max(np.abs(values["fft"]-values["dense"]))),
               diffusion_radius_relative_difference=abs(solvers["fft"].diffusion_radius/solvers["dense"].diffusion_radius-1),
               mass_rank=solvers["fft"].galerkin.mass_rank,
               setup_seconds={name: s.setup_seconds for name, s in solvers.items()})
    if final_time:
        results = {name: solver.solve(final_time, progress=True) for name, solver in solvers.items()}
        difference = np.abs(np.column_stack(primitives(results["fft"]["output"]))
                            - np.column_stack(primitives(results["dense"]["output"])))
        row["evolution"] = {name: result["summary"] for name, result in results.items()}
        row["primitive_max_absolute_difference"] = dict(zip(("rho", "u", "p", "T"), difference.max(axis=0).tolist()))
        row["evolution_speedup"] = results["dense"]["summary"]["solve_seconds"]/results["fft"]["summary"]["solve_seconds"]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[65, 129, 257, 513])
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--final-time", type=float, default=0., help="0: RHS timing only; positive: also evolve both backends")
    parser.add_argument("--mu", type=float, default=.001)
    parser.add_argument("--json")
    args = parser.parse_args()
    if args.repeats < 1 or not np.isfinite(args.final_time) or args.final_time < 0:
        parser.error("positive repeats and finite nonnegative final time required")
    rows = []
    for n in sorted(set(args.sizes)):
        row = compare(n, args.repeats, args.final_time, args.mu)
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(dict(repeats=args.repeats, results=rows), indent=2)+"\n")


if __name__ == "__main__":
    main()
