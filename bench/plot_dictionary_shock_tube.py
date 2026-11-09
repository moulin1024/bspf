"""Plot fixed-dictionary shock-tube runs against the fine FC numerical reference."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/pde'))
from shock_tube_1d import primitives


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tags',nargs='+')
    args=parser.parse_args()
    folder=ROOT/'benchmarks/shock_tube/dictionary'
    reference=np.load(ROOT/'benchmarks/shock_tube/fc_gram/fc_6119.npz')
    x=reference['x'];exact=primitives(reference['output'])
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    names=('Density','Velocity','Pressure')
    for i,name in enumerate(names):
        axes[0,i].plot(x,exact[i],color='black',lw=1.8,label='FC-Gram 6119 reference')
        axes[0,i].set(xlabel='x',ylabel=name,xlim=(.25,.75))
    for tag in args.tags:
        data=np.load(folder/f'{tag}.npz');s=json.loads((folder/f'{tag}.json').read_text())
        fields=primitives(data['output']);label=f"BSPF {s['n']} + blocks (peak {s['peak_modes']})"
        for i in range(3):axes[0,i].plot(x,fields[i],ls='--',lw=1.2,label=label)
        error=np.max(np.abs(np.asarray(fields)-np.asarray(exact)),axis=0)
        axes[1,0].semilogy(x,np.maximum(error,1e-16),lw=1,label=label)
        updates=s['updates'];t=[u['time'] for u in updates]
        axes[1,1].step(t,[u['modes'] for u in updates],where='post',label=label)
        axes[1,2].semilogy(t,np.maximum([u['error'] for u in updates],1e-16),label=label)
        axes[1,2].axhline(s['tolerance'],color='gray',ls=':',lw=.8)
    axes[0,0].legend(fontsize=8)
    axes[1,0].set(xlabel='x',ylabel='Max primitive discrepancy vs FC 6119',ylim=(1e-11,1e-1))
    axes[1,0].axhline(1e-8,color='gray',ls=':',label='1e-8 target')
    axes[1,1].set(xlabel='Time',ylabel='Active coefficients per field')
    axes[1,2].set(xlabel='Time',ylabel='State / scaled-rate projection residual')
    for ax in axes.ravel():ax.grid(alpha=.2)
    fig.suptitle('Viscous Sod shock tube: automatic fixed spline-block dictionary\n'
                 r'$\mu=10^{-4}$, Pr = 0.72, initial smoothing = 0.01, t = 0.1; no added damping')
    fig.savefig(folder/'comparison.png',dpi=180)
    fig.savefig(folder/'comparison.pdf')
    print(folder/'comparison.png')


if __name__=='__main__':main()
