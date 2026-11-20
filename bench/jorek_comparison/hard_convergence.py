"""Plot JOREK bicubic and global BSPF flux and spatial convergence on the MAST-U wall."""
import argparse
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np

from campaign import Restart,rule
from challenge import LocalField
from global_bspf import geometry
from mms import compile_kernel,native,SCORE_ORDER
from refined_global import RefinedGlobalBSPF
from subdivide import subdivide
from tilted_mms import manufactured_tilted


def validate_edges(parent,children,n):
    # All inherited cubic edges, including the wall, must reproduce the parent
    # trace. Four points determine each cubic polynomial on a child interval.
    sample=np.linspace(0,1,4)
    old=parent.coeff[:2].reshape(2,16,-1).transpose(2,0,1)
    new=children.coeff[:2].reshape(2,16,-1).transpose(2,0,1).reshape(-1,n,n,2,16)
    maximum=0.
    for k in range(n):
        for end in (0,1):
            for along_s in (False,True):
                def tab(q):
                    return (rule(q,[end]) if along_s else rule([end],q))[0]
                child=new[:,k,end*(n-1)] if along_s else new[:,end*(n-1),k]
                maximum=max(maximum,np.max(abs(child@tab(sample)-old@tab((k+sample)/n))))
    if maximum>1e-9:
        raise ValueError(f'changed parent edge by {maximum} m')
    return maximum


def sample_solution(ref,coeff,solver,exact):
    q=np.linspace(0,1,3)
    tab=rule(q)[0]
    coords=(ref.coeff[:2].reshape(2,16,-1).transpose(2,0,1)@tab).transpose(0,2,1)
    xy,unique,ids=np.unique(coords.reshape(-1,2).round(12),axis=0,
                            return_index=True,return_inverse=True)
    grid=ids.reshape(-1,3,3)
    a,b,c,d=(grid[:,:-1,:-1],grid[:,1:,:-1],grid[:,1:,1:],grid[:,:-1,1:])
    triangles=np.concatenate([np.stack(t,axis=-1).reshape(-1,3) for t in ((a,b,c),(a,c,d))])
    u,v,w=xy[triangles].transpose(1,0,2)
    det=(v[:,0]-u[:,0])*(w[:,1]-u[:,1])-(v[:,1]-u[:,1])*(w[:,0]-u[:,0])
    triangles=triangles[det>1e-15]
    psi=(coeff.reshape(-1,16)@tab).ravel()[unique]
    bspf=np.concatenate([solver.evaluate(*batch.T)[0] for batch in np.array_split(xy,32)])
    return dict(R=xy[:,0],Z=xy[:,1],triangles=triangles,exact=exact(*xy.T)[0],JOREK=psi,BSPF=bspf)


def figures(out,rows,data):
    tri=Triangulation(data['R'],data['Z'],data['triangles'])
    levels=np.linspace(min(data[k].min() for k in ('exact','JOREK','BSPF')),
                       max(data[k].max() for k in ('exact','JOREK','BSPF')),31)
    fig,axes=plt.subplots(1,3,figsize=(10,6),layout='constrained',sharey=True)
    for ax,key,title in zip(axes,('exact','JOREK','BSPF'),('Exact flux','JOREK bicubic','Refined global BSPF')):
        art=ax.tricontourf(tri,data[key],levels=levels,cmap='RdBu_r')
        ax.tricontour(tri,data[key],levels=levels[2:-1:3],colors='k',linewidths=.5)
        ax.set(aspect='equal',xlabel='R (m)',title=title)
    axes[0].set_ylabel('Z (m)')
    fig.colorbar(art,ax=axes,label='Manufactured flux ψ',shrink=.8)
    fig.savefig(out/'hard_mms_solutions.png',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained',sharex=True)
    for ax,key,title in zip(axes,['flux_relative_L2','Bp_relative_L2','local_Bp_relative_L2'],
                            ['Flux · whole domain','Poloidal field · whole domain','Poloidal field · near-wall ellipse']):
        for name,marker,color in [('JOREK','s','#bd4936'),('BSPF','o','#156591')]:
            series=[r for r in rows if r['method']==name]
            ax.loglog([r['free_dofs'] for r in series],[100*r[key] for r in series],
                      marker=marker,color=color,label=name,lw=1.8)
        ax.set(xlabel='Free coefficients',title=title)
        ax.grid(True,which='major',alpha=.3)
    axes[0].set_ylabel('Relative L₂ error (%)')
    axes[0].legend()
    fig.savefig(out/'hard_mms_convergence.png',dpi=180)
    plt.close(fig)


def main(path,out):
    out.mkdir(parents=True,exist_ok=True)
    ref=Restart(path)
    exact=manufactured_tilted(3.)
    order=8
    rows=[]
    solver=RefinedGlobalBSPF(ref,order=order)
    solver.pilot(lambda r,z:exact(r,z)[3],lambda r,z:exact(r,z)[0])
    for div in (0,12,16,20,24,32):
        if div:
            solver.refine(min(solver.width)/div,support_padding=2*solver.bin_width+5*min(solver.width)/32)
        local=LocalField()
        flux,bp=solver.score(ref,exact,SCORE_ORDER,observe=local.collect)
        rows.append(dict(method='BSPF',spacing_divisor=div,free_dofs=solver.active_dofs if div else solver.coef.size,
                         flux_relative_L2=flux,Bp_relative_L2=bp,local_Bp_relative_L2=np.sqrt(local.error/local.energy)))
    with tempfile.TemporaryDirectory() as tmp:
        kernel=compile_kernel(Path(tmp),order)
        for n in (1,2,3):
            mesh=path
            if n>1:
                mesh=Path(tmp)/'subdivided.h5'
                subdivide(path,mesh,n)
            level=Restart(mesh)
            edge_error=validate_edges(ref,level,n)
            q,w=np.polynomial.legendre.leggauss(5)
            area=geometry(level,(q+1)/2,w/2)[2].sum()
            local=LocalField()
            captured=[]
            row=native(mesh,level,3.,kernel,tilted=True,observe=local.collect,
                       capture=captured.append if n==3 else None)
            rows.append(dict(method='JOREK',subdivision=n,elements=level.count,domain_area_m2=area,
                             maximum_edge_difference_m=edge_error,
                             local_Bp_relative_L2=np.sqrt(local.error/local.energy),**row))
        data=sample_solution(level,captured[0],solver,exact)
    np.savez_compressed(out/'hard_mms_fields.npz',**data)
    (out/'hard_mms_convergence.json').write_text(json.dumps(dict(assembly_order=order,score_order=SCORE_ORDER,rows=rows),
                                                            indent=2,allow_nan=False)+'\n')
    figures(out,rows,data)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('restart',type=Path)
    parser.add_argument('output_directory',type=Path)
    args=parser.parse_args()
    main(args.restart,args.output_directory)
