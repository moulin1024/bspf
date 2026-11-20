"""Plot accuracy at fixed knot spacing against degree, size, and solve cost."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(source,output):
    report=json.loads(source.read_text())
    rows=report['rows']
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained',sharey=True)
    for ax,key,label in zip(axes,['degree','dofs','refine_s'],
                            ['Correction degree p','Total coefficients','Assembly + solve (s)']):
        for y,legend,mark in [('field_pct','Poloidal field','o'),('flux_pct','Flux','s'),
                              ('local_field_pct','Near-wall field','^')]:
            ax.semilogy([r[key] for r in rows],[r[y] for r in rows],marker=mark,label=legend)
        ax.set_xlabel(label)
        ax.grid(True,alpha=.25)
    axes[0].set_xticks([r['degree'] for r in rows])
    axes[0].set_ylabel('Relative L₂ error (%)')
    axes[0].legend()
    fig.suptitle(f"Hard MAST-U MMS · fixed knot spacing {1000*report['spacing_m']:.2f} mm")
    fig.savefig(output,dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    plot(args.source,args.output)
