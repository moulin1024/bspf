# SPDX-License-Identifier: GPL-3.0-or-later
"""Plot saved previous/optimized enrichment profiles; no simulations are rerun."""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'/'pde'))
from shock_tube_1d import primitives

root=Path(__file__).resolve().parents[1]
data=np.load(root/'benchmarks/shock_tube/enrichment_v2_benchmark.npz')
x=data['x']
fields={key:primitives(data[key]) for key in ('reference','previous_q32_cfl035','updated_q32_cfl070')}
styles=[('reference','Reference: 1025 nodes','black','-',2.0),
        ('previous_q32_cfl035','Previous: 2.95 s, CFL 0.35','#0072B2','-',1.6),
        ('updated_q32_cfl070','Updated: 1.88 s, CFL 0.70','#D55E00','--',1.5)]
fig,axes=plt.subplots(2,3,figsize=(15,8.5),layout='constrained')
for index,title in enumerate(('Density','Velocity','Pressure','Temperature')):
    ax=axes.flat[index]
    for key,label,color,ls,lw in styles:
        ax.plot(x,fields[key][index],label=label,color=color,ls=ls,lw=lw)
    ax.set(title=title,xlabel='x',xlim=(.30,.85))
    ax.grid(alpha=.18)
for ax,index,title in ((axes.flat[4],1,'Velocity: shock close-up'),(axes.flat[5],2,'Pressure error relative to reference')):
    for key,label,color,ls,lw in styles:
        if index==2 and key=='reference':
            ax.axhline(0,color='black',lw=.8);continue
        y=fields[key][index] if index==1 else fields[key][index]-fields['reference'][index]
        ax.plot(x,y,label=label,color=color,ls=ls,lw=lw)
    ax.set(title=title,xlabel='x',xlim=(.655,.705))
    ax.grid(alpha=.18)
handles,labels=axes.flat[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=10)
fig.suptitle('BSPF enrichment: 65 base + 167 used enrichment modes per field\n'
             'Physical viscosity μ = 10⁻⁴, t = 0.1 — no added damping; previous/updated curves nearly overlap',fontsize=14)
output=root/'benchmarks/shock_tube/enrichment_v2_results'
fig.savefig(output.with_suffix('.png'),dpi=180)
fig.savefig(output.with_suffix('.pdf'))
print(output.with_suffix('.png'))
