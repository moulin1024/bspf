# SPDX-License-Identifier: GPL-3.0-or-later
"""Sensitivity study using an existing shock_tube_enrichment.py reference NPZ."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples'/'pde'))
from shock_tube_enrichment import run_enriched, primitives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default='benchmarks/shock_tube/enrichment_n65_mu0001_fine.npz')
    parser.add_argument('--prefix', default='benchmarks/shock_tube/enrichment_study')
    args = parser.parse_args()
    reference_path = Path(args.reference)
    data = np.load(reference_path)
    metadata = json.loads(reference_path.with_suffix('.json').read_text())['parameters']
    parameters = dict(n=metadata['n'], mu=metadata['mu'], final_time=metadata['time'],
                      width=metadata['width'], interval=metadata['interval'], quadrature=metadata['quadrature'])
    interval, quadrature = parameters['interval'], parameters['quadrature']
    variants = dict(adaptive={}, quadrature_double=dict(quadrature=2*quadrature),
        half_dt=dict(cfl=.175), half_interval=dict(interval=interval/2),
        quarter_interval=dict(interval=interval/4), static=dict(interval=metadata['time']+1),
        multiscale=dict(scales=(.25,.5,1.)))
    results = {}
    for name, changes in variants.items():
        results[name] = run_enriched(**(parameters | changes))
        print(name, results[name]['summary'], flush=True)
    x, reference = data['x'], primitives(data['reference'])
    outputs = {'base':data['base'], **{k:v['output'] for k,v in results.items()}}
    metrics = {}
    for name, output in outputs.items():
        metrics[name] = {field:dict(l1=float(np.trapezoid(abs(v-r), x)), linf=float(abs(v-r).max()),
            total_variation=float(np.sum(abs(np.diff(v)))))
            for field,v,r in zip(('rho','u','p','T'),primitives(output),reference)}
    report = dict(parameters=parameters, reference=str(reference_path), metrics=metrics,
        runs={k:{s:v[s] for s in ('summary','history','updates')} for k,v in results.items()})
    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2))
    np.savez(prefix.with_suffix('.npz'), x=x, reference=data['reference'], **outputs)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for key, label in [('base','BSPF'),('adaptive',f'Adaptive, update {interval:g}'),
                       ('half_interval',f'Adaptive, update {interval/2:g}'),('reference','Reference')]:
        output = data['reference'] if key == 'reference' else outputs[key]
        for ax, field, v in zip(axes.flat, ('Density','Velocity','Pressure','Temperature'),primitives(output)):
            ax.plot(x,v,label=label,lw=1.1)
            ax.set(title=field, xlabel='x', xlim=(.35,.85))
            ax.legend(fontsize=8)
    fig.suptitle(f"Moving tanh enrichment: n={parameters['n']}, mu={parameters['mu']:g}, t={parameters['final_time']:g}")
    fig.savefig(prefix.with_suffix('.png'), dpi=160)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
