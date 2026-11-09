"""Plot the matched-threshold dictionary BSPF / FC comparison."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'benchmarks/traveling_shock/dictionary_fc_final'
summary=json.loads((folder/'summary.json').read_text())
bspf=json.loads((folder/'bspf_0.json').read_text())
data={name:np.load(folder/f'{name}_0.npz') for name in ('bspf','fc')}
def primitive(U):
    rho=U[:,0];u=U[:,1]/rho;p=.4*(U[:,2]-.5*U[:,1]*u)
    return np.column_stack((rho,u,p,p/rho))
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
x=data['bspf']['x'];exact=primitive(data['bspf']['exact'])
axes[0,0].plot(x,exact[:,0],'k',lw=2,label='Analytical shock')
for name,label,color in [('bspf','Dictionary BSPF','C0'),('fc','FC-Gram','C1')]:
    values=primitive(data[name]['output'])
    axes[0,0].plot(x,values[:,0],color=color,ls='--',lw=1,label=label)
    axes[0,1].semilogy(x,np.maximum(np.max(abs(values-exact),axis=1),1e-16),color=color,label=label)
axes[0,0].set(xlim=(.405,.445),xlabel='x',ylabel='Density',title='Full NS evolution to t = 0.05')
axes[0,1].set(xlim=(.39,.46),ylim=(1e-13,1e-7),xlabel='x',ylabel='Maximum primitive error',title='Error against analytical solution')
axes[0,1].axhline(1e-8,color='gray',ls=':',label='1e-8 threshold')
axes[1,0].boxplot([summary['timing'][name]['runs'] for name in ('bspf','fc')],
    tick_labels=['Dictionary BSPF\n70 coefficients','FC-Gram\n2120 physical points'],showfliers=True)
axes[1,0].set(ylabel='End-to-end seconds',title=f"Seven alternating pairs; {summary['speedup_fc_over_bspf']:.2f}x median speedup")
updates=bspf['updates'];times=np.r_[[u['time'] for u in updates],bspf['time']]
centers=np.r_[[u['block'][0] for u in updates],updates[-1]['block'][0]]
widths=np.r_[[u['block'][1] for u in updates],updates[-1]['block'][1]]
axes[1,1].step(times,centers,where='post',label='Selected fixed dictionary block')
axes[1,1].fill_between(times,centers-.5*widths,centers+.5*widths,step='post',alpha=.15,label='Block fine core')
axes[1,1].plot(times,bspf['center']+bspf['speed']*times,'k--',label='Exact location: diagnostic only')
axes[1,1].set(xlabel='Time',ylabel='x',title='Discrete block selection; no trajectory input')
for ax in axes.flat:ax.grid(alpha=.25)
for ax in (axes[0,0],axes[0,1],axes[1,1]):ax.legend(fontsize=8)
fig.savefig(folder/'comparison.png',dpi=180);fig.savefig(folder/'comparison.pdf')
print(folder/'comparison.png')
