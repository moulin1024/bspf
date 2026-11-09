# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare ordinary BSPF resolution against enriched shock/plateau errors."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples/pde'))
from shock_tube_1d import BSPFNavierStokes, primitives
from shock_tube_enrichment import run_enriched
from shock_tube_spline_enrichment import LocalRefinement, SparseSplineSpace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes', type=int, nargs='+', default=[237, 321, 385, 449, 513])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--prefix', default='benchmarks/shock_tube/matched_error')
    parser.add_argument('--quadrature', type=int, default=12)
    parser.add_argument('--cfl', type=float, default=.7)
    args = parser.parse_args()
    ref = np.load(ROOT/'benchmarks/shock_tube/enrichment_n65_mu0001_fine.npz')
    x, reference = ref['x'], primitives(ref['reference'])
    # Fixed bands on either side of the steep viscous front exclude its core.
    plateau = ((x >= .64) & (x <= .67)) | ((x >= .69) & (x <= .73))
    shock = (x >= .64) & (x <= .73)
    prefix = ROOT/args.prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    runs, summary, outputs = {}, {}, {}
    configs = [('enriched', 65)] + [(f'global{n}', n) for n in args.modes]
    for repetition in range(args.repeats):
        for name, n in configs if repetition % 2 == 0 else configs[::-1]:
            start = perf_counter()
            if name == 'enriched':
                result = run_enriched(interval=.005, quadrature=32, cfl=args.cfl,
                    space_factory=LocalRefinement(8, 2., space_type=SparseSplineSpace))
            else:
                result = BSPFNavierStokes(n, mu=.0001, quadrature=args.quadrature).solve(.1, .01, args.cfl)
            seconds = perf_counter()-start
            np.testing.assert_array_equal(result['probes'], x)
            outputs[name] = result['output']
            metrics = {}
            for field, values, target in zip(('rho','u','p','T'), primitives(result['output']), reference):
                error = abs(values-target)
                metrics[field] = dict(l1=float(np.trapezoid(error,x)), linf=float(error.max()),
                    shock_linf=float(error[shock].max()), plateau_linf=float(error[plateau].max()))
            runs.setdefault(name, []).append(dict(seconds=seconds, summary=result['summary']))
            summary[name] = dict(median_seconds=float(np.median([r['seconds'] for r in runs[name]])),
                final_modes=n+(result['summary']['enrichment_count'] if name=='enriched' else 0),
                steps=result['summary']['steps'], metrics=metrics)
            target = summary.get('enriched')
            if target:
                for row in summary.values():
                    row['error_ratios'] = {metric:max(row['metrics'][field][metric]/target['metrics'][field][metric]
                        for field in metrics) for metric in ('linf','shock_linf','plateau_linf')}
            report = dict(parameters=vars(args), summary=summary, runs=runs,
                note='Timing includes setup, integration and final probes; excludes imports and file IO. '
                     'Plateau bands [0.64,0.67] and [0.69,0.73] exclude the front core. '
                     'All errors use a 1025-node numerical reference, not an exact solution. '
                     'Ordinary BSPF interpolates the initial condition; enrichment uses an L2 projection.')
            prefix.with_suffix('.json').write_text(json.dumps(report, indent=2))
            np.savez(prefix.with_suffix('.npz'), x=x, reference=ref['reference'], **outputs)
            print(repetition+1, name, round(seconds,4), summary[name]['error_ratios'], flush=True)


if __name__ == '__main__':
    main()
