"""Plot physical-grid counts for all solvers, excluding FC continuation nodes."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/bspf-matplotlib')
import csv
import hashlib
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'evidence'
OUT=ROOT/'fc_pytorch_rtx3070'
parser=argparse.ArgumentParser()
parser.add_argument('--suite-root',type=Path,help='Mixed MMS suite directory; original evidence is the default')
parser.add_argument('--validated-quadrature',action='store_true',help='Use the optional quadrature-check selections instead of the main campaign')
parser.add_argument('--max-n',type=int,default=145,help='Largest physical nodes per axis included in the comparison')
parser.add_argument('--mfem-dir',type=Path,help='Override the MFEM evidence directory within a suite')
args=parser.parse_args()
if args.suite_root:
    ROOT=args.suite_root.resolve();OUT=ROOT
    suite=json.loads((ROOT/'suite_status.json').read_text())
    expected_hash=suite['modes_sha256']
    mixed_label='waves + Gaussians + tanh'
    if 'modes' in suite:
        modes_path=Path(suite['modes'])
        if not modes_path.exists():
            modes_path=ROOT.parent/'nonperiodic_curved_gaussian/modes.npz'
        assert hashlib.sha256(modes_path.read_bytes()).hexdigest()==expected_hash
        with np.load(modes_path) as data:
            if data['tanh_amplitudes'].size==0:mixed_label='waves + Gaussians'
    validated=json.loads((ROOT/'quadrature_validation.json').read_text())['selected'] if args.validated_quadrature else {}
    mfem_dir=args.mfem_dir.resolve() if args.mfem_dir else ROOT/('mfem_gpu_rhs' if (ROOT/'mfem_gpu_rhs/report.json').exists() else 'mfem')
rows=[]
for n in [17,33,65,97,129,145]:
    if n>args.max_n:continue
    q=max(96,(3*n+1)//2);e=(n-1)//8
    paths={
        'FC-Gram PyTorch':OUT/f'turbulence_n{n}/result.json',
        'BSPF CUDA':(ROOT/'bspf_cuda3d_rtx3070' if n<65 else ROOT/'bspf_cuda3d_optimization/quadrature')/f'n{n}_q{q}/result.json',
        'MFEM p8, LOR-AMG':ROOT/f'mfem_lor_amg_fp32_rtx3070/e{e}_p8/result.json',
        'NekRS p8':ROOT/f'nekrs_rtx3070/e{e}_p8/result.json',
        'Global Chebyshev':ROOT/f'chebyshev_cuda3d_rtx3070/n{n}_q{q}/result.json'}
    if n==145:
        paths['MFEM p8, LOR-AMG']=ROOT/f'mfem_lor_amg_fp32_rtx3070_145/e{e}_p8/result.json'
    optimized=ROOT/'bspf_cuda3d_parity'/f'n{n}_q{q}/result.json'
    if optimized.exists():paths['BSPF CUDA']=optimized
    shared=ROOT/'bspf_cuda3d_shared'/f'n{n}_q{q}/result.json'
    if shared.exists():paths['BSPF CUDA']=shared
    if args.suite_root:
        paths={
            'FC-Gram PyTorch':ROOT/f'fc/turbulence_n{n}/result.json',
            'BSPF CUDA':ROOT/validated.get(f'bspf_{n}',f'bspf/n{n}_q{q}/result.json'),
            'MFEM p8, LOR-AMG':mfem_dir/f'e{e}_p8/result.json',
            'NekRS p8':ROOT/f'nekrs/e{e}_p8/result.json',
            'Global Chebyshev':ROOT/validated.get(f'chebyshev_{n}',f'chebyshev/n{n}_q{q}/result.json')}
    for name,path in paths.items():
        d=json.loads(path.read_text())
        if args.suite_root:
            provenance=json.loads((mfem_dir/'report.json').read_text()) if name.startswith('MFEM') else d
            assert provenance.get('modes_sha256',provenance.get('data_sha256'))==expected_hash, path
        seconds=(d['warm_cg_s'] if name in ('BSPF CUDA','Global Chebyshev') else float(np.median(d['linear_samples_seconds'])) if name.startswith('MFEM') else d['warm_solve_s'])
        timing_path=path
        if name=='FC-Gram PyTorch' and 'warm_wall_s' not in d:
            timing_path=OUT/f'full_timing/n{n}/result.json'
            timing=json.loads(timing_path.read_text())
            assert timing['data_sha256']==d['data_sha256'] and timing['degree']==d['degree']
            previous=np.load(path.parent/'solution.npz')['u']
            repeated=np.load(timing_path.parent/'solution.npz')['u']
            assert np.linalg.norm(previous-repeated)/np.linalg.norm(previous)<1e-12
            wall=timing['warm_wall_s']
        elif name.startswith('MFEM'):wall=d['warm_median_seconds']
        else:wall=d['warm_wall_s']
        plot_n=n
        rows.append(dict(solver=name,n=n,plot_n=plot_n,physical_dofs=n**3,dofs=plot_n**3,
            count_kind='physical_grid',
            warm_solve_s=seconds,warm_wall_s=wall,
            relative_l2=d['relative_l2'],source=str(path.relative_to(ROOT)),wall_timing_source=str(timing_path.relative_to(ROOT))))
with (OUT/'comparison.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0],lineterminator="\n");w.writeheader();w.writerows(rows)
fig,axes=plt.subplots(1,2,figsize=(12.5,6.3))
for name,marker in zip(paths,['o','s','^','D','P']):
    values=[r for r in rows if r['solver']==name]
    for ax,key in zip(axes,['dofs','warm_wall_s']):
        ax.loglog([r[key] for r in values],[r['relative_l2'] for r in values],marker+('--' if name=='Global Chebyshev' else '-'),label=('BSPF degree 8' if name=='BSPF CUDA' else name))
        if key=='warm_wall_s':
            for r in values:ax.annotate(str(r['plot_n']),(r[key],r['relative_l2']),xytext=((-16,9) if r['n']<=65 else (3,-14)) if name=='Global Chebyshev' else (3,5),textcoords='offset points',fontsize=7)
for ax,label in zip(axes,['Physical grid points','Repeated full solve wall time (seconds)']):
    ax.set_xlabel(label);ax.set_ylabel('Relative physical L2 error');ax.grid(True,which='major',alpha=.25);ax.margins(x=.07,y=.09)
axes[0].legend(fontsize=9)
fig.suptitle('RTX 3070: curved, nonperiodic 3D '+(f'mixed MMS ({mixed_label})' if args.suite_root else 'turbulence Poisson MMS'))
mfem_note='MFEM includes host RHS assembly.'
if args.suite_root:
    mfem_data=json.loads((mfem_dir/'e2_p8/result.json').read_text())
    if mfem_data['rhs_assembly']=='mfem_device_cached_geometry':mfem_note='MFEM uses GPU RHS assembly with cached fixed-mesh geometry.'
fig.text(.5,.025,'FP64 operators / outer Krylov; FP32 preconditioners. FC: FGMRES, tolerance 1e−10; others use CG.\n'
         'Full solve: RHS + boundary lifting + solve + recovery; excludes setup, analytic sampling and error audits.\n'
         'All counts use physical nodal DOFs n³, including boundary nodes; labels are physical nodes/axis.\n'
         'FC continuation nodes are excluded from the count; their computational cost remains in the timing.\n'
         'Chebyshev degree n−1. BSPF includes Fourier enrichment. '+mfem_note,ha='center',fontsize=8.5)
fig.tight_layout(rect=[0,.19,1,.94])
for ext in ['png','pdf']:fig.savefig(OUT/f'comparison.{ext}',dpi=180)
print(OUT/'comparison.png')
