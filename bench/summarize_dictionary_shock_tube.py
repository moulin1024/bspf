"""Collect tube errors on a common grid and preserve the timing scope."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/pde'))
from shock_tube_1d import primitives


def main():
    root=ROOT/'benchmarks/shock_tube';folder=root/'dictionary'
    ref=np.load(root/'fc_gram/fc_6119.npz');fields=np.asarray(primitives(ref['output']))
    rows=[]
    entries=[(folder/tag,'BSPF dictionary') for tag in ('n65_tol6','n129_b4_tol7')]
    entries += [(root/'fc_gram'/f'fc_{n}_dictionary','FC-Gram') for n in (999,2023,4071)]
    for stem,method in entries:
        s=json.loads(stem.with_suffix('.json').read_text());data=np.load(stem.with_suffix('.npz'))
        assert np.array_equal(data['x'],ref['x'])
        errors=np.max(abs(np.asarray(primitives(data['output']))-fields),axis=1)
        row=dict(tag=stem.name,method=method,errors=dict(zip(('rho','u','p','T'),map(float,errors))),
                 max_error=float(errors.max()),seconds=s['end_to_end_seconds'],
                 timing_note=s.get('timing_note','Single serial warm run; setup, evolution and dense output included; imports/JIT/files/error comparison excluded.'),
                 summary=s)
        if 'updates' in s:
            updates=s['updates'];row['max_remap_error']=max(u['remap_error'] for u in updates)
            row['remaps']=sum(u['remap_error']>0 for u in updates)
            row['min_modes']=min(u['modes'] for u in updates)
            row['max_quadrature_points']=max(u['quadrature_points'] for u in updates)
        rows.append(row)
    result=dict(reference='FC-Gram 6119; numerical reference, not certified exact solution',
                metric='Maximum absolute primitive discrepancy on 64001 common points',rows=rows)
    (folder/'summary.json').write_text(json.dumps(result,indent=2))
    for r in rows:print(r['tag'],r['max_error'],r['seconds'],r.get('max_remap_error'),r.get('max_quadrature_points'))


if __name__=='__main__':main()
