"""Compare measured default-quadrature BSPF with matching-DOF MFEM and NekRS."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/bspf-matplotlib')
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / 'evidence'
OUT = ROOT / 'bspf_cuda3d_optimization'
rows, sources = [], []
for n in [17, 33, 65, 97, 129]:
    q = max(96, (3*n+1)//2)
    bp = (ROOT / 'bspf_cuda3d_rtx3070' if n < 65 else OUT / 'quadrature') / f'n{n}_q{q}/result.json'
    parity = ROOT / f'bspf_cuda3d_parity/n{n}_q{q}/result.json'
    if parity.exists():
        bp = parity
    shared = ROOT / f'bspf_cuda3d_shared/n{n}_q{q}/result.json'
    if shared.exists():
        bp = shared
    e = (n-1)//8
    mp = ROOT / f'mfem_lor_amg_fp32_rtx3070/e{e}_p8/result.json'
    kp = ROOT / f'nekrs_rtx3070/e{e}_p8/result.json'
    b, m, k = [json.loads(p.read_text()) for p in (bp, mp, kp)]
    assert b['q'] == q
    assert m['preconditioner_precision'] == 'float32'
    assert m['operator_precision'] == m['cg_precision'] == 'float64'
    assert b['dofs'] == m['dofs'] == k['unique_dofs'] == n**3
    sources.append(dict(n=n, bspf=str(bp.relative_to(ROOT)), mfem=str(mp.relative_to(ROOT)), nekrs=str(kp.relative_to(ROOT))))
    rows.append(dict(n=n, dofs=n**3, bspf_q=q,
                     bspf_error=b['relative_l2'], bspf_cg_s=b['warm_cg_s'], bspf_wall_s=b['warm_wall_s'],
                     mfem_error=m['relative_l2'], mfem_cg_s=float(np.median(m['linear_samples_seconds'])), mfem_wall_s=m['warm_median_seconds'],
                     nekrs_error=k['relative_l2'], nekrs_cg_s=k['warm_solve_s'], nekrs_wall_s=k['warm_wall_s']))

with (OUT / 'optimized_solver_comparison.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
(OUT / 'optimized_solver_comparison.json').write_text(json.dumps(dict(
    gpu='RTX 3070', sources=sources, results=rows,
    notes=['BSPF q=max(96,ceil(1.5*n)); prefers shared, then reflection-split results when available.',
           'Matching DOFs, different trial spaces and integration rules.',
           'Warm medians: six solves for shared BSPF, three otherwise; excludes setup and MMS preparation/audits.',
           'Full warm timing includes RHS/boundary work; MFEM p8 includes host RHS assembly.',
           'All preconditioners FP32; operators and outer Krylov solves FP64.']), indent=2)+'\n')
fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
solvers = [('bspf', 'BSPF degree 8, optimized quadrature', 'o', '#1764ab'),
           ('nekrs', 'NekRS, p-MG + GPU AMG', 's', '#d16b17'),
           ('mfem', 'MFEM/libCEED, LOR + FP32 GPU AMG', '^', '#27814a')]
for solver, label, marker, color in solvers:
    for ax, field in zip(axes, ['dofs', f'{solver}_cg_s', f'{solver}_wall_s']):
        ax.loglog([r[field] for r in rows], [r[f'{solver}_error'] for r in rows],
                  marker+'-', color=color, label=label, linewidth=1.8, markersize=5)
        if field == 'dofs' and solver != 'bspf':
            continue
        for r in rows:
            offset = (4, 7) if solver != 'mfem' else (4, -12)
            ax.annotate(str(r['n']), (r[field], r[f'{solver}_error']),
                        xytext=offset, textcoords='offset points', fontsize=7, color=color)
for ax, title, xlabel in zip(axes,
        ['Accuracy vs DOFs', 'Accuracy vs warm linear solve', 'Accuracy vs full warm solve'],
        ['Global DOFs', 'Linear solve time (s)', 'RHS + boundary + solve time (s)']):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel); ax.set_ylabel('Relative physical L2 error')
    ax.grid(True, which='major', alpha=.25); ax.margins(x=.2, y=.1)
fig.suptitle('RTX 3070 · nonperiodic curved-domain turbulence MMS', fontsize=14)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .105), ncol=3, fontsize=9)
fig.text(.5, .025,
         'Labels: nodes per axis. BSPF degree 8: degree-8 polynomial part + global Fourier enrichment; q = max(96, ceil(1.5n)).\n'
         'All preconditioners FP32. Warm medians exclude setup and MMS preparation/audits; full MFEM time includes host RHS assembly.',
         ha='center', fontsize=9)
fig.tight_layout(rect=[0, .18, 1, .93])
for suffix in ['png', 'pdf']:
    fig.savefig(OUT / f'optimized_solver_comparison.{suffix}', dpi=180)
print(json.dumps(rows, indent=2))
