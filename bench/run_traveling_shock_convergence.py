# SPDX-License-Identifier: GPL-3.0-or-later
"""Resume the degree-13 resolution sweep and full-evolution sensitivity runs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'benchmarks/traveling_shock/high_resolution'
p = argparse.ArgumentParser()
p.add_argument('group', choices=['sweep', 'validation', 'ordinary'])
a = p.parse_args()
if a.group == 'ordinary':
    configs = [(n,12,.7) for n in (193,257,385,513,641,769,1025,1537)]
elif a.group == 'sweep':
    configs = [(n,64,.7) for n in (193,257,385,513,641,769)]
else:
    configs = [(n,q,cfl) for n in (513,769) for q,cfl in ((64,.35),(128,.7))]
for n,q,cfl in configs:
    suffix = '' if a.group == 'ordinary' or (q,cfl)==(64,.7) else f'_q{q}_cfl{int(round(100*cfl))}'
    prefix = OUT / (f'ordinary{n}_conditioned' if a.group == 'ordinary' else f'enriched{n}_p13_reconditioned{suffix}')
    if not prefix.with_suffix('.npz').exists():
        cmd = [sys.executable, str(ROOT/'bench/bench_traveling_shock_high.py'),
               '--n',str(n),'--basis',str((n-1)//16+9),
               '--quadrature',str(q),'--cfl',str(cfl),'--prefix',str(prefix)]
        if a.group != 'ordinary':
            cmd += ['--enriched','--recondition']
        with Path(str(prefix)+'.log').open('w') as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    report=json.loads(prefix.with_suffix('.json').read_text())
    assert report['n'] == n and report['quadrature'] == q and report['cfl'] == cfl
    assert report['line_options']['n_basis'] == (n-1)//16+9
    assert report['enriched'] == (a.group != 'ordinary')
    if a.group != 'ordinary':
        assert report['spline_degree'] == 13 and report['recondition']
        assert report['refinement'] == 4 and report['halfwidth'] == 12.
    print(prefix.name,report['final_modes'],{k:v['linf'] for k,v in report['metrics'].items()},flush=True)
