"""Plot the fixed-footprint hard-MMS refinement campaign."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(directory, fixed=None):
    fig,axes=plt.subplots(1,3,figsize=(12.6,4.2),layout='constrained',sharex=True)
    metrics=[('flux_pct','Flux'),('field_pct','Whole-domain poloidal field'),
             ('local_field_pct','Near-wall poloidal field')]
    for f in sorted(directory.glob('large_refinement_q*.json'),
                    key=lambda p:int(p.stem.split('_q')[-1])):
        run=json.loads(f.read_text())
        rows=run['rows']
        q=run['assembly_order']
        for ax,(metric,title) in zip(axes,metrics):
            ax.loglog([r['dofs'] for r in rows],[r[metric] for r in rows],'-o',
                      markersize=4,label=f'BSPF {q} × {q}')
            ax.set(xlabel='Active coefficients',title=title)
    if fixed is not None:
        rows=json.loads(fixed.read_text())['rows']
        for ax,(metric,_) in zip(axes,metrics):
            ax.loglog([r['dofs'] for r in rows],[r[metric] for r in rows],
                      '-D',color='#642a81',lw=2.2,markersize=4,label='BSPF · composite')
    native=[r for r in json.loads((directory/'hard_mms_convergence.json').read_text())['rows']
            if r['method']=='JOREK']
    for ax,key in zip(axes,['flux_relative_L2','Bp_relative_L2','local_Bp_relative_L2']):
        ax.loglog([r['free_dofs'] for r in native],[100*r[key] for r in native],
                  '--s',color='0.3',label='JOREK 8 × 8',markersize=4)
        ax.grid(True,which='major',alpha=.3)
    axes[0].set_ylabel('Relative L₂ error (%)')
    axes[0].legend(fontsize=8)
    fig.savefig(directory/('fixed_refinement.png' if fixed else 'large_refinement.png'),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--fixed',type=Path)
    args=parser.parse_args()
    plot(args.directory,args.fixed)
