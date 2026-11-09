"""Summarize completed local prior-art comparisons without hiding controls."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'benchmarks/traveling_shock/prior_art'
rows=[]
for path in sorted(folder.glob('*.json')):
    if path.stem=='comparison':continue
    d=json.loads(path.read_text())
    if 'metrics' not in d:continue
    rows.append(dict(case=path.stem,method=d['method'],n=d['n'],
                     coefficients=d.get('final_modes',d['n']),fft_points=d.get('fft_points'),
                     error=max(v['linf'] for v in d['metrics'].values()),
                     seconds=d['seconds'],steps=d['steps']))
controls={}
for name in ('fc_2049_half_cfl','fc_2049_unfiltered','fc_1793_half_cfl','esg_129_half_cfl','esg_129_double_quadrature'):
    p=folder/f'{name}.npz';ref=folder/('_'.join(name.split('_')[:2])+'.npz')
    if p.exists() and ref.exists():
        a=np.load(p)['output'];b=np.load(ref)['output']
        def prim(U):
            rho=U[:,0];u=U[:,1]/rho;p=.4*(U[:,2]-.5*rho*u*u)
            return np.column_stack((rho,u,p,p/rho))
        controls[name]=float(np.max(abs(prim(a)-prim(b))))
(folder/'comparison.json').write_text(json.dumps(dict(rows=rows,controls=controls),indent=2))
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for method,label in [('bspf','Enriched BSPF'),('esg','Legendre ESG-I adaptation'),('fc','FC-Gram')]:
    subset=sorted([r for r in rows if r['method']==method and r['case'].count('_')==1],key=lambda r:r['n'])
    if subset:
        axes[0].loglog([r['coefficients'] for r in subset],[r['error'] for r in subset],'o-',label=label)
        axes[1].loglog([r['seconds'] for r in subset],[r['error'] for r in subset],'o-',label=label)
for ax in axes:
    ax.axhline(1e-6,color='gray',ls='--');ax.grid(alpha=.25);ax.set_ylabel('Worst primitive peak error');ax.legend(fontsize=8)
axes[0].set_xlabel('Coefficients / physical samples per field')
axes[1].set_xlabel('End-to-end seconds (single serial run)')
fig.savefig(folder/'comparison.png',dpi=180);fig.savefig(folder/'comparison.pdf')
print(json.dumps(dict(rows=rows,controls=controls),indent=2))
