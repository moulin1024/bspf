"""Plot the matched-threshold timing and physical error, including knot layout."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'benchmarks/traveling_shock/graded_spline_fc_1e8'
summary=json.loads((folder/'summary.json').read_text())
b=np.load(folder/'bspf_0.npz');f=np.load(folder/'fc_0.npz')

def primitive(U):
    rho=U[:,0];u=U[:,1]/rho;p=.4*(U[:,2]-.5*rho*u*u)
    return np.column_stack((rho,u,p,p/rho))

fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
x=b['x'];exact=primitive(b['exact']);pb=primitive(b['output']);pf=primitive(f['output'])
ax[0,0].plot(x,exact[:,0],'k-',lw=2,label='Exact')
ax[0,0].plot(x,pb[:,0],'--',label='BSPF + graded splines')
ax[0,0].plot(x,pf[:,0],':',label='FC-Gram')
ax[0,0].set(xlim=(.39,.455),xlabel='x',ylabel='Density at t = 0.05')
ax[0,0].legend(fontsize=9)
for values,label in ((pb,'BSPF + graded splines'),(pf,'FC-Gram')):
    error=np.max(abs(values-exact),axis=1)
    ax[0,1].semilogy(x,np.maximum(error,1e-16),label=label)
ax[0,1].axhline(1e-8,color='gray',ls='--',label='1e-8 threshold')
ax[0,1].set(xlim=(.35,.49),ylim=(1e-13,3e-8),xlabel='x',ylabel='Maximum primitive absolute error')
ax[0,1].legend(fontsize=9)
timing=summary['timing'];names=['bspf','fc'];med=[timing[n]['median'] for n in names]
errors=np.array([[timing[n]['median']-timing[n]['minimum'] for n in names],
                 [timing[n]['maximum']-timing[n]['median'] for n in names]])
ax[1,0].bar(['BSPF + splines','FC-Gram'],med,yerr=errors,capsize=5,color=['C0','C1'])
for i,t in enumerate(med):ax[1,0].text(i,t+.13,f'{t:.3f} s',ha='center')
ax[1,0].set(ylabel='End-to-end seconds',ylim=(0,max(med)*1.35),title='Median and range of five alternating runs')
knots=np.unique(b['enrichment_knots'])
ax[1,1].plot(np.linspace(0,1,25),np.ones(25),'o',ms=4,label='25 fixed BSPF base points')
ax[1,1].plot(knots,np.zeros_like(knots),'|',ms=15,label='Adaptive enrichment knots')
ax[1,1].set(xlim=(.30,.55),ylim=(-.5,1.5),yticks=[0,1],yticklabels=['Spline knots','Base grid'],xlabel='x',title='Basis knots adapt; the base grid stays fixed')
ax[1,1].legend(loc='upper right',fontsize=8)
for a in ax.flat:a.grid(alpha=.2)
fig.savefig(folder/'comparison.png',dpi=180)
fig.savefig(folder/'comparison.pdf')
print(folder/'comparison.png')
