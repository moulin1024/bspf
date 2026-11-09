"""Summarize FC shock-tube convergence against the finest numerical run."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'benchmarks/shock_tube/fc_gram'
sizes=(487,999,2023,4071,6119)
data={n:np.load(folder/f'fc_{n}.npz') for n in sizes}

def primitive(U):
    r=U[:,0];u=U[:,1]/r;p=.4*(U[:,2]-.5*U[:,1]*u)
    return np.column_stack((r,u,p,p/r))

fine=primitive(data[6119]['output']);x=data[6119]['x'];rows=[];controls={}
for n in sizes:
    d=json.loads((folder/f'fc_{n}.json').read_text())
    times=[d['end_to_end_seconds']]
    repeat=folder/f'fc_{n}_repeat.json'
    if repeat.exists():times.append(json.loads(repeat.read_text())['end_to_end_seconds'])
    error=None if n==6119 else float(np.max(abs(primitive(data[n]['output'])-fine)))
    rows.append(dict(n=n,fft_points=d['fft_points'],seconds=times,
                     difference_to_finest=error,stored_reference_discrepancy=max(d['stored_reference_discrepancy'].values()),
                     mass_drift=d['relative_mass_drift'],energy_drift=d['relative_energy_drift']))
    half=folder/f'fc_{n}_half_cfl.npz'
    if half.exists():controls[f'half_cfl_{n}']=float(np.max(abs(primitive(np.load(half)['output'])-primitive(data[n]['output']))))
summary=dict(finest_n=6119,rows=rows,controls=controls,
             note='Differences from a finer numerical FC solution, not certified exact errors. Timings are individual serial warm runs.')
(folder/'summary.json').write_text(json.dumps(summary,indent=2))
fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for n in sizes:
    v=primitive(data[n]['output'])
    ax[0,0].plot(x,v[:,0],label=f'{n} points',lw=1)
    ax[0,1].plot(x,v[:,1],label=f'{n} points',lw=1)
    if n!=6119:
        ax[1,0].semilogy(x,np.maximum(np.max(abs(v-fine),axis=1),1e-16),label=str(n))
ax[0,0].set(xlim=(.3,.75),xlabel='x',ylabel='Density',title='Full viscous shock-tube wave pattern')
ax[0,1].set(xlim=(.665,.695),xlabel='x',ylabel='Velocity',title='Shock and nearby ringing')
ax[1,0].set(xlim=(.3,.75),ylim=(1e-12,.1),xlabel='x',ylabel='Maximum primitive discrepancy',title='Compared with the 6119-point numerical solution')
ax[1,0].axhline(1e-8,color='gray',ls='--')
ax[1,1].loglog([r['n'] for r in rows[:-1]],[r['difference_to_finest'] for r in rows[:-1]],'o-')
ax[1,1].axhline(1e-8,color='gray',ls='--')
ax[1,1].set(xlabel='Physical FC samples',ylabel='Peak difference from 6119-point result',title='Spatial / temporal convergence combined')
for a in ax.flat:a.grid(alpha=.25)
for a in (ax[0,0],ax[0,1],ax[1,0]):a.legend(fontsize=8)
fig.savefig(folder/'comparison.png',dpi=180);fig.savefig(folder/'comparison.pdf')
print(json.dumps(summary,indent=2))
