# SPDX-License-Identifier: GPL-3.0-or-later
"""Matched end-to-end and RHS cost of 65-node BSPF and local enrichment.

Set OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 externally. Excludes plotting,
file output and imports; includes setup, evolution, diagnostics and final
probe evaluation. Enriched mode count grows during each measured evolution.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples'/'pde'))
from shock_tube_1d import BSPFNavierStokes, initial_state
from shock_tube_enrichment import run_enriched
from shock_tube_spline_enrichment import LocalRefinement


class TimedRefinement(LocalRefinement):
    def __init__(self):
        super().__init__(refinement=8, halfwidth=2.)
        self.timing = dict(rhs=0., initial_space=0., rebuilds=0.)
        self.calls = 0

    def __call__(self, base, centers, widths):
        start = perf_counter()
        first = self.previous is None
        space = super().__call__(base, centers, widths)
        self.timing['initial_space' if first else 'rebuilds'] += perf_counter()-start
        self.calls += 1
        action = space.rhs
        space.unprofiled_rhs = action
        def timed(state):
            start = perf_counter()
            value = action(state)
            self.timing['rhs'] += perf_counter()-start
            return value
        space.rhs = timed
        return space


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--rhs-repeats', type=int, default=21)
    parser.add_argument('--json', default='benchmarks/shock_tube/enrichment_cost.json')
    args = parser.parse_args()
    if args.repeats < 1 or args.rhs_repeats < 1:
        parser.error('positive repetition counts required')
    names = ('base_q12', 'base_q64', 'enriched_q64')
    rows = {name:[] for name in names}
    base_solvers = {}
    # Warm up native kernels before measuring full evolutions.
    warm = BSPFNavierStokes(65, mu=.0001, quadrature=12)
    warm.rhs(initial_state(warm.x, .01))
    for repetition in range(args.repeats):
        order = names if repetition % 2 == 0 else names[::-1]
        for name in order:
            gc.collect()
            if name == 'enriched_q64':
                factory = TimedRefinement()
                start = perf_counter()
                result = run_enriched(n=65, mu=.0001, final_time=.1, width=.01,
                    interval=.005, quadrature=64, cfl=.35, space_factory=factory)
                elapsed = perf_counter()-start
                phases = dict(factory.timing)
                phases['other'] = elapsed-sum(phases.values())
                modes = 65+result['summary']['enrichment_count']
                space = factory.previous
            else:
                start = perf_counter()
                solver = BSPFNavierStokes(65, mu=.0001, quadrature=12 if name == 'base_q12' else 64)
                build = perf_counter()-start
                result = solver.solve(.1, .01, .35)
                elapsed = perf_counter()-start
                phases = dict(setup=build, solve=result['summary']['solve_seconds'])
                phases['other'] = elapsed-sum(phases.values())
                modes = 65
                base_solvers[name] = solver
            row = dict(seconds=elapsed, phases=phases, steps=result['summary']['steps'],
                final_modes=modes, rejected_stages=result['summary']['rejected_stages'])
            rows[name].append(row)
            print(f'{repetition+1}/{args.repeats} {name}: {elapsed:.6f}s, {row["steps"]} steps',flush=True)
    # Cost at the final 65+172 space. Same analytic smooth input for all cases.
    actions, states = {}, {}
    for name, solver in base_solvers.items():
        actions[name], states[name] = solver.rhs, initial_state(solver.x, .01)
    actions['enriched_q64'] = space.unprofiled_rhs
    states['enriched_q64'] = space.fit(initial_state(space.base.q, .01))
    for name in names:
        actions[name](states[name])
    rhs_times = {name:[] for name in names}
    for repetition in range(args.rhs_repeats):
        for name in (names if repetition % 2 == 0 else names[::-1]):
            start = perf_counter()
            for _ in range(10):
                actions[name](states[name])
            rhs_times[name].append((perf_counter()-start)/10)
    summary = {}
    for name in names:
        summary[name] = dict(median_seconds=float(np.median([r['seconds'] for r in rows[name]])),
            min_seconds=min(r['seconds'] for r in rows[name]), max_seconds=max(r['seconds'] for r in rows[name]),
            median_rhs_seconds=float(np.median(rhs_times[name])),
            median_phases={key:float(np.median([r['phases'][key] for r in rows[name]])) for key in rows[name][0]['phases']},
            steps=rows[name][-1]['steps'], final_modes=rows[name][-1]['final_modes'])
    reference = summary['base_q12']
    for row in summary.values():
        row['total_ratio_to_base_q12'] = row['median_seconds']/reference['median_seconds']
        row['rhs_ratio_to_base_q12'] = row['median_rhs_seconds']/reference['median_rhs_seconds']
    report = dict(parameters=dict(n=65,mu=.0001,final_time=.1,width=.01,cfl=.35,
        refinement=8,patch_halfwidth=2.,update_interval=.005,repeats=args.repeats,rhs_repeats=args.rhs_repeats),
        environment=dict(platform=platform.platform(),python=sys.version,threads={key:os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')}),
        notes=['End-to-end includes setup and final probes, excludes plotting, imports and file output.',
            'Enriched space grows; RHS microbenchmark uses its final size. Timing wrappers add two clock reads per RHS.',
            'Rebuild phase includes basis construction, mass factors and spectral-radius estimation; other includes remapping, timestep selection and diagnostics.'],
        quadrature_samples={name:len(s.q) for name,s in base_solvers.items()},
        summary=summary, runs=rows, rhs_samples=rhs_times)
    path=Path(args.json)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
