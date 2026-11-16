"""Reproducible 3D schematic of the saved Gaussian and tanh MMS features."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/bspf-bubbles-mpl')
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mms import CurvedBox

HERE=Path(__file__).resolve().parent
POS='#e68432';NEG='#3377b5';TANH='#13958a';FRAME='#8a96a3'


def surface(center,radii):
    theta,phi=np.meshgrid(np.linspace(0,2*np.pi,65),np.linspace(0,np.pi,33))
    return (center[0]+radii[0]*np.sin(phi)*np.cos(theta),
            center[1]+radii[1]*np.sin(phi)*np.sin(theta),
            center[2]+radii[2]*np.cos(phi))


def domain(ax):
    box=CurvedBox();t=np.linspace(0,1,51)
    # Mapped grid lines on every face reveal the curved domain, without opaque walls.
    for fixed in range(3):
        axes=[a for a in range(3) if a!=fixed]
        for side in (0,1):
            for varying in axes:
                other=next(a for a in axes if a!=varying)
                for value in (0,.25,.5,.75,1):
                    s=np.zeros((len(t),3));s[:,fixed]=side;s[:,varying]=t;s[:,other]=value
                    X=box.map(s)[0]
                    ax.plot(*X.T,color=FRAME,lw=.6,alpha=.27,zorder=0)
    ax.set(xlim=(0,1),ylim=(0,1),zlim=(0,1),xlabel='x',ylabel='y',zlabel='z')
    ax.set_box_aspect((1,1,1));ax.view_init(elev=23,azim=-58)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.set_ticks([0,.5,1]);axis.pane.fill=False
        axis._axinfo['grid']['color']=(.7,.75,.8,.12)
    ax.tick_params(labelsize=8,pad=0)


def gaussians(ax,p,alpha=.58):
    for center,sigma,amplitude in zip(p['gaussian_centers'],p['gaussian_sigmas'],p['gaussian_amplitudes']):
        color=POS if amplitude>0 else NEG
        # Surfaces mark half the absolute amplitude of each individual component.
        xyz=surface(center,np.sqrt(2*np.log(2))*sigma)
        ax.plot_surface(*xyz,color=color,alpha=alpha,linewidth=0,shade=True,rcount=33,ccount=65)
        ax.scatter(*center,color=color,s=8,depthshade=False)


def tanh(ax,p,combined=False):
    for center,R,delta in zip(p['tanh_centers'],p['tanh_radii'],p['tanh_widths']):
        ax.plot_surface(*surface(center,np.full(3,R)),color=TANH,alpha=.20 if combined else .35,
                        linewidth=0,shade=True,rcount=33,ccount=65)
        # Exact radii for 90% and 10% of the plateau value of the squared-radius tanh.
        radii=np.sqrt(R*R+np.array([-1,1])*2*R*delta*np.arctanh(.8))
        theta=np.linspace(0,2*np.pi,201)
        for radius in radii:
            for plane in range(3):
                xyz=np.tile(center[:,None],(1,len(theta)));a,b=[i for i in range(3) if i!=plane]
                xyz[a]+=radius*np.cos(theta);xyz[b]+=radius*np.sin(theta)
                ax.plot(*xyz,color=TANH,alpha=.8,linestyle='--',lw=.8)
        if not combined:
            end=center+np.array([R,0,0]);ax.plot(*np.stack([center,end]).T,color='#075f58',lw=1.8)
            ax.scatter(*center,color='#075f58',s=18,depthshade=False)
            ax.text(*(center+np.array([R*.4,-.05,.025])),f'R = {R:.2f}',fontsize=9,color='#075f58')
            ax.text2D(.02,.88,f'Interface width δ = {delta:.3f}\nDashed rings: 10% / 90% levels',
                      transform=ax.transAxes,fontsize=9,color='#075f58')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',type=Path,default=HERE/'evidence/nonperiodic_curved_mixed')
    args=parser.parse_args();meta=json.loads((args.dataset/'manifest.json').read_text())
    p={k:np.asarray(v) for k,v in meta['parameters'].items()}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelpad':1})
    fig=plt.figure(figsize=(16,6.1),facecolor='white')
    axes=[fig.add_subplot(1,3,i+1,projection='3d',computed_zorder=False) for i in range(3)]
    for ax in axes:domain(ax)
    gaussians(axes[0],p);tanh(axes[1],p);gaussians(axes[2],p,alpha=.48);tanh(axes[2],p,combined=True)
    for ax,title in zip(axes,['(a) Eight Gaussian bubbles','(b) Sharp spherical tanh bubble','(c) Combined feature geometry']):
        ax.set_title(title,pad=10,fontsize=13,fontweight='medium')
    fig.suptitle('Localized structures in the curved-domain Poisson MMS',fontsize=18,y=.97)
    handles=[Patch(facecolor=POS,label='Positive Gaussian'),Patch(facecolor=NEG,label='Negative Gaussian'),
             Patch(facecolor=TANH,alpha=.55,label='Tanh interface'),Line2D([],[],color=FRAME,lw=1,label='Curved domain')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.105),ncol=4,frameon=False,fontsize=11)
    fig.text(.5,.055,'Gaussian ellipsoids: individual half-amplitude surfaces. Tanh sphere: half-amplitude interface. Positions and sizes are taken from the saved dataset.',
             ha='center',fontsize=10,color='#374151')
    fig.text(.5,.022,f'Schematic geometry only; the underlying plane-wave field is omitted. Reproducible placement seed: {meta["seed"]}.',
             ha='center',fontsize=9,color='#596575')
    fig.subplots_adjust(left=.015,right=.97,bottom=.20,top=.84,wspace=.025)
    for ext in ('png','pdf','svg'):fig.savefig(args.dataset/f'bubbles_3d.{ext}',dpi=220,facecolor='white')
    print(args.dataset/'bubbles_3d.png')
if __name__=='__main__':main()
