"""Collect completed native results with validation and comparison provenance."""
from pathlib import Path
import json, re
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'evidence/nekrs_rtx3070'
results=[]
for folder in sorted(ROOT.glob('e*_p8'),key=lambda p:int(p.name.split('_')[0][1:])):
    path=folder/'result.json'
    if not path.exists():continue
    d=json.loads(path.read_text())
    text=(folder/'solve.log').read_text()
    d['native_initialization_s']=float(re.search(r'initialization took ([\d.e+-]+) s',text).group(1))
    d['gpu_coarse_solver_confirmed']='setup SEMFEM solver (lambdaAvg=1) FP32 on device' in text
    if not d['gpu_coarse_solver_confirmed']:raise RuntimeError(folder)
    d['case']=folder.name
    results.append(d)
orders=[]
for a,b in zip(results,results[1:]):
    orders.append(dict(from_elements=a['elements_per_axis'],to_elements=b['elements_per_axis'],
                       order=float(np.log(a['relative_l2']/b['relative_l2'])/np.log(b['elements_per_axis']/a['elements_per_axis']))))
report=dict(status='complete_to_16_cubed_elements' if [r['elements_per_axis'] for r in results]==[2,4,8,12,16] else 'partial',
            problem='-Laplacian(u_x)=f; exact Dirichlet data; nonperiodic Q2 curved box',
            dataset='../nonperiodic_curved_shell32/modes.npz',
            build=json.loads((ROOT/'build_provenance.json').read_text()),
            polynomial_guard=json.loads((ROOT/'e2_p8_polynomial/result.json').read_text()),
            minimum_case_limitation='e=1: stock NekRS ogsSetup fails on empty masked gather (log: e1_p8_polynomial/solve.log)',
            degree=8,operator_quadrature='9-point collocated GLL per axis',
            comparison_operator_quadrature='MFEM: 11-point Gauss per axis',
            preconditioner='p-MG 8→5→1, RAS/optimized fourth-kind Chebyshev, FP32 GPU BoomerAMG on coarse SEMFEM',
            timing='One first solve plus three reset warm solves; median GPU-synchronized wall times. Sampling, I/O, audit and setup excluded.',
            observed_h_orders=orders,results=results)
(ROOT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['status'])
