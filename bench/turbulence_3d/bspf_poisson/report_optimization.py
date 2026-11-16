"""Regenerate the measured CUDA optimization comparison (no GPU required)."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'curved_poisson/benchmarks/internal_soliton/cpp'))
from bundle import read_bundle
EVIDENCE = HERE.parent / 'evidence'
OUT = EVIDENCE / 'bspf_cuda3d_optimization'

def read(path):
    return json.loads(path.read_text())

def solution(path):
    return read_bundle(path)['solution']

def median(data):
    return float(np.median([r['cg_s'] for r in data['runs'][1:]]))

rows = []
for n, q in [(65, 98), (97, 146), (129, 194)]:
    baseline = read(OUT / f'baseline_n{n}.json')
    original = read(EVIDENCE / f'bspf_cuda3d_rtx3070/n{n}_q{2*n}/result.json')
    reduced_path = OUT / f'quadrature/n{n}_q{q}'
    reduced = read(reduced_path / 'result.json')
    u = solution(OUT / f'baseline_n{n}.bin')
    v = solution(reduced_path / 'solution.bin')
    delta = float(np.linalg.norm(u-v)/np.linalg.norm(u))
    assert delta < 1e-11, (n, delta)
    assert reduced['relative_l2'] < 1e-11
    assert all(r['relative_residual'] <= 1e-12 for r in reduced['runs'])
    row = dict(n=n, dofs=n**3, baseline_q=2*n, reduced_q=q,
               baseline_cg_s=median(baseline), reduced_cg_s=median(reduced),
               speedup=median(baseline)/median(reduced),
               baseline_device_bytes=baseline['device_bytes'],
               reduced_device_bytes=reduced['device_bytes'],
               baseline_relative_l2=original['relative_l2'],
               reduced_relative_l2=reduced['relative_l2'],
               relative_nodal_change=delta,
               iterations=reduced['runs'][-1]['iterations'])
    direct_file = OUT / f'direct_n{n}.json'
    if direct_file.exists():
        direct = read(direct_file)
        row['direct_cg_s'] = median(direct)
        row['direct_relative_nodal_change'] = float(np.linalg.norm(u-solution(direct_file.with_suffix('.bin')))/np.linalg.norm(u))
        assert row['direct_relative_nodal_change'] < 1e-11
    rows.append(row)

(OUT / 'comparison.json').write_text(json.dumps(rows, indent=2)+'\n')
keys = list(dict.fromkeys(k for row in rows for k in row))
with (OUT / 'comparison.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader(); writer.writerows(rows)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
n = [r['n'] for r in rows]
for key, label in [('baseline', 'Original quadrature'), ('reduced', 'Reduced quadrature')]:
    axes[0].plot(n, [r[f'{key}_cg_s'] for r in rows], 'o-', label=label)
    axes[1].plot(n, [r[f'{key}_device_bytes']/1e9 for r in rows], 'o-', label=label)
    axes[2].loglog([r[f'{key}_cg_s'] for r in rows], [r[f'{key}_relative_l2'] for r in rows], 'o-', label=label)
for ax in axes:
    ax.grid(alpha=.25); ax.legend(fontsize=8)
for ax in axes[:2]:
    ax.set_xlabel('Nodes per axis'); ax.set_xticks(n)
axes[0].set_ylabel('Warm CG median (s)')
axes[1].set_ylabel('Solver device buffers (GB)')
axes[2].set_xlabel('Warm CG median (s)'); axes[2].set_ylabel('Relative L2 error')
fig.suptitle('RTX 3070: curved nonperiodic BSPF MMS, FP64 CG / FP32 preconditioner')
fig.savefig(OUT / 'optimization.png', dpi=180)
fig.savefig(OUT / 'optimization.pdf')
print(json.dumps(rows, indent=2))
