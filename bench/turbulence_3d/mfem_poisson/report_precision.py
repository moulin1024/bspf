"""Compare measured FP32/FP64 GPU LOR-AMG with the same FP64 operator."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'evidence'
OUT=ROOT/'mfem_lor_amg_fp32_rtx3070'
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'curved_poisson/benchmarks/internal_soliton/cpp'))
from bundle import read_bundle
rows=[]
for e in [2,4,8,12,16]:
    old=ROOT/f'mfem_lor_amg_rtx3070/e{e}_p8'
    new=OUT/f'e{e}_p8'
    a,b=[json.loads((p/'result.json').read_text()) for p in [old,new]]
    assert b['preconditioner_precision']=='float32' and b['operator_precision']==b['cg_precision']=='float64'
    assert a['dofs']==b['dofs'] and a['quadrature']==b['quadrature']
    # Same element-local GLL ordering and boundary data in both exports.
    u,v=[read_bundle(p/'solution_local.bin')['coefficients'] for p in [old,new]]
    delta=float(np.linalg.norm(u-v)/np.linalg.norm(u))
    assert delta<1e-9, (e,delta)
    assert b['relative_residual']<2e-11
    ta,tb=[float(np.median(r['linear_samples_seconds'])) for r in [a,b]]
    rows.append(dict(n=8*e+1,dofs=b['dofs'],fp64_cg_s=ta,fp32_cg_s=tb,speedup=ta/tb,
                     fp64_iterations=a['iterations'],fp32_iterations=b['iterations'],
                     fp64_l2=a['relative_l2'],fp32_l2=b['relative_l2'],relative_solution_change=delta,
                     fp32_true_residual=b['relative_residual'],fp32_full_warm_s=b['warm_median_seconds']))
(OUT/'precision_comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
with (OUT/'precision_comparison.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2))
