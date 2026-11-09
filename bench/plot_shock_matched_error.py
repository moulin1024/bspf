# SPDX-License-Identifier: GPL-3.0-or-later
"""Plot the measured ordinary/enriched BSPF comparison."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/pde'))
from shock_tube_1d import primitives
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prefix', default='benchmarks/shock_tube/matched_error_timing')
args = parser.parse_args()
prefix = ROOT/args.prefix
data = np.load(prefix.with_suffix('.npz'))
report = json.loads(prefix.with_suffix('.json').read_text())
x = data['x']; reference = primitives(data['reference'])
styles = [('enriched','Enriched: 65 + 167','#0072B2','-'),
          ('global237','Ordinary: 237','#999999',':'),
          ('global450','Ordinary: 450','#D55E00','--'),
          ('global480','Ordinary: 480','#009E73','-.')]
fig,axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
axes[0,0].plot(x,reference[2],color='black',lw=2,label='Reference: 1025')
for key,label,color,ls in styles:
    if key == 'global237':
        continue
    fields=primitives(data[key])
    axes[0,0].plot(x,fields[2],color=color,ls=ls,lw=1.4,label=label)
    for ax,index in ((axes[0,1],1),(axes[1,0],2)):
        ax.plot(x,fields[index]-reference[index],color=color,ls=ls,lw=1.3)
for ax,title in ((axes[0,0],'Pressure near the shock'),(axes[0,1],'Velocity error'),(axes[1,0],'Pressure error')):
    ax.set(xlim=(.64,.73),xlabel='x',title=title)
    ax.grid(alpha=.2)
for ax in (axes[0,1],axes[1,0]):
    ax.axhline(0,color='black',lw=.6)
    for lo,hi in ((.64,.67),(.69,.73)):
        ax.axvspan(lo,hi,color='#eeeecc',alpha=.4,zorder=0)
axes[0,0].set_ylim(.08,.34)
axes[0,0].legend(fontsize=8)
ax=axes[1,1]
values=[report['summary'][key]['median_seconds'] for key,*_ in styles]
ax.barh([label for _,label,*_ in styles],values,color=[color for _,_,color,_ in styles])
ax.invert_yaxis();ax.set(xlabel='Seconds, median of 3 runs',title='Setup + evolution + final probes',xlim=(0,max(values)*1.22))
for i,value in enumerate(values): ax.text(value+.04,i,f'{value:.2f} s',va='center')
fig.suptitle('Ordinary vs enriched BSPF: same physical viscosity and CFL\n'
             'μ = 10⁻⁴, t = 0.1; shaded bands measure error beside the shock',fontsize=13)
fig.savefig(prefix.with_suffix('.png'),dpi=180)
fig.savefig(prefix.with_suffix('.pdf'))
