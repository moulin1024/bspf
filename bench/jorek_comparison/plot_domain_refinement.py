"""Compare the original plateau with domain-wide defect correction."""
import json
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(directory):
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for file,label,marker in [('fixed_refinement.json','Fixed region','o'),
                              ('domain_refinement.json','Domain-wide correction','s')]:
        rows=json.loads((directory/file).read_text())['rows']
        for ax,key,title in zip(axes,('flux_pct','field_pct'),('Flux','Poloidal field')):
            ax.loglog([r['dofs'] for r in rows],[r[key] for r in rows],marker=marker,label=label)
            ax.set(title=title,xlabel='Retained coefficients',ylabel='Whole-domain relative L₂ error (%)')
            ax.grid(True,which='major',alpha=.3)
    axes[0].legend()
    fig.savefig(directory/'domain_refinement.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    plot(parser.parse_args().directory)
