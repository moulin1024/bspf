# SPDX-License-Identifier: GPL-3.0-or-later
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'benchmarks/traveling_shock/dg'
dg=json.loads((out/'dg_p7_ne576.json').read_text())
checks={key:json.loads((out/f'check_p7_ne576_{key}.json').read_text()) for key in ('time','quadrature')}
previous=json.loads((out.parent/'wenoz/serial_timing_1e6.json').read_text())['selected']
rows=[dict(method=r['method'],modes=r['modes'],worst_error=r['worst_error'],wall_seconds=r['wall_seconds']) for r in previous]
rows.append(dict(method='Modal DG-BR1, p=7',modes=dg['modes'],worst_error=dg['worst_error'],wall_seconds=dg['wall_seconds']))
assert all(r['worst_error']<=1e-6 for r in rows)
assert all(r['worst_error']<=1e-6 for r in checks.values())
notes=['Same exact viscous shock, mu=0.001, Pr=0.75, final time=0.05.',
       'DG checks 64001 points and both traces of each interface; threshold uses their maximum.',
       'DG p=7, 576 uniform elements, 17 Gauss points, HLLC, central BR1, explicit RK4.',
       'No claim of entropy stability for this conventional modal implementation.',
       'Single serial warm baseline timings; reference timings are from the preceding serial benchmark.',
       'Full sensitivity checks ran concurrently; their wall times are not used for performance comparison.',
       'End-to-end time includes setup, initialization, integration and error checks; JIT and result saving excluded.',
       'Configurations meet a common tolerance, not exactly equal errors; no globally optimal resolution claim.']
summary=dict(threshold=1e-6,comparison=rows,dg=dg,checks=checks,notes=notes)
(out/'threshold_comparison.json').write_text(json.dumps(summary,indent=2))
lines=['DG comparison at 1e-6 worst primitive-field error threshold','',
       'Method | Coefficients per field | Worst error | Wall seconds']
for r in rows:lines.append(f"{r['method']} | {r['modes']} | {r['worst_error']:.8e} | {r['wall_seconds']:.3f}")
lines+=['','Full DG sensitivity checks:']
for key,r in checks.items():
    delta=max(max(r['profile_difference'].values()),max(r['trace_difference'].values()))
    lines.append(f"{key}: error {r['worst_error']:.8e}; maximum change including traces {delta:.8e}")
lines+=['']+notes
(out/'threshold_comparison.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
