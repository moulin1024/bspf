"""Add reproducible Gaussian/tanh scalar features and sample the mixed MMS on CUDA."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/bspf-mixed-mpl')
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import cupy as cp
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mms import CurvedBox,PlaneWaveMMS
from poisson_features import PoissonFeatures
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('mixed_mfem_sampler',HERE/'mfem_poisson/run.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=20261117)
    p.add_argument('--gaussians',type=int,default=8);p.add_argument('--n',type=int,default=129)
    p.add_argument('--tanh-width',type=float,default=.015)
    p.add_argument('--no-tanh',action='store_true',help='Keep the same Gaussian mixture and omit the tanh sphere')
    p.add_argument('--base',type=Path,default=HERE/'evidence/nonperiodic_curved_shell32/modes.npz')
    p.add_argument('--out',type=Path,default=HERE/'evidence/nonperiodic_curved_mixed');args=p.parse_args()
    if args.gaussians<1 or args.n<3 or args.tanh_width<=0:p.error('Positive count/width and n>=3 required')
    rng=np.random.default_rng(args.seed);box=CurvedBox()
    parameters=dict(gaussian_centers=box.map(rng.uniform(.18,.82,(args.gaussians,3)))[0],
        gaussian_sigmas=rng.uniform(.045,.10,(args.gaussians,3)),
        gaussian_amplitudes=rng.choice([-1.,1.],args.gaussians)*rng.uniform(.5,.9,args.gaussians),
        tanh_centers=box.map(rng.uniform(.35,.65,(1,3)))[0],tanh_radii=np.array([.22]),
        tanh_widths=np.array([args.tanh_width]),tanh_amplitudes=np.array([.8]))
    if args.no_tanh:
        for key in ('tanh_centers','tanh_radii','tanh_widths','tanh_amplitudes'):
            parameters[key]=parameters[key][:0]
    PoissonFeatures(parameters) # Validate all parameters before writing.
    args.out.mkdir(parents=True,exist_ok=True)
    with np.load(args.base) as d:original={k:d[k] for k in d.files}
    path=args.out/'modes.npz'
    np.savez(path,**original,**parameters,feature_seed=args.seed,feature_schema='scalar_poisson_gaussian_tanh_v1')
    sampler=shared.Sampler(path);features=sampler.features
    # Independent CPU sampler versus the production CUDA sampler.
    probe=box.map(rng.uniform(0,1,(97,3)))[0]
    reference=PlaneWaveMMS.load(args.base).evaluate(probe)
    extra,extra_force=PoissonFeatures(parameters).evaluate(probe)
    expected=np.stack((reference['velocity'][:,0]+extra,reference['vector_poisson_forcing'][:,0]+extra_force),axis=-1)
    actual=cp.asnumpy(sampler(probe));errors=np.linalg.norm(actual-expected,axis=0)/np.linalg.norm(expected,axis=0)
    if errors.max()>1e-12:raise RuntimeError(f'CPU/CUDA sampler mismatch: {errors}')
    nodes=np.linspace(0,1,args.n);z,y,x=cp.meshgrid(cp.asarray(nodes),cp.asarray(nodes),cp.asarray(nodes),indexing='ij')
    physical,_,det=box.map(cp.stack((x,y,z),axis=-1),cp)
    sampled=sampler(physical);cp.cuda.Stream.null.synchronize()
    with h5py.File(args.out/'samples.h5','w') as h:
        h.create_dataset('reference_nodes',data=nodes)
        h.create_dataset('phi',data=cp.asnumpy(sampled[...,0]),compression='gzip',compression_opts=1)
        h.create_dataset('forcing',data=cp.asnumpy(sampled[...,1]),compression='gzip',compression_opts=1)
        h.attrs['storage']='z,y,x; coordinates obtained from CurvedBox(reference_nodes)'
        h.attrs['equation']='-Delta_X phi=f; phi=original_u_x + scalar Gaussian mixture'+('' if args.no_tanh else ' + spherical tanh')
    del sampled,physical,det,x,y,z;cp.get_default_memory_pool().free_all_blocks()
    # A physical z slice through the sphere center exposes the circular transition.
    zslice=.5 if args.no_tanh else float(parameters['tanh_centers'][0,2]);axis=np.linspace(0,1,321)
    yy,xx=np.meshgrid(axis,axis,indexing='ij');points=np.stack((xx,yy,np.full_like(xx,zslice)),axis=-1)
    gp=cp.asarray(points);total=cp.asnumpy(sampler(gp));parts={k:cp.asnumpy(v) for k,v in features.evaluate(gp,components=True).items()}
    wave=total[...,0]-parts['gaussian']-parts['tanh']
    fields=[wave,parts['gaussian'],parts['tanh'],total[...,0],total[...,1]]
    titles=['Plane-wave component','Random Gaussian mixture','Spherical tanh interface','Composite scalar phi','Analytic forcing −Δphi']
    if args.no_tanh:
        fields.pop(2);titles.pop(2)
    fig,axes=plt.subplots(1,len(fields),figsize=(3.6*len(fields),4.0))
    for ax,values,title in zip(axes,fields,titles):
        im=ax.imshow(values,origin='lower',extent=[0,1,0,1],cmap='viridis',aspect='equal')
        ax.set(title=title,xlabel='Physical x',ylabel='Physical y');fig.colorbar(im,ax=ax,shrink=.75)
    feature_label='Gaussian mixture only' if args.no_tanh else f'tanh width={args.tanh_width}'
    fig.suptitle(f'Mixed Poisson MMS: physical z={zslice:.3f}; seed={args.seed}; {feature_label}')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(args.out/f'preview.{ext}',dpi=170)
    np.savez_compressed(args.out/'slice.npz',x=axis,y=axis,z=zslice,plane_wave=wave,phi=total[...,0],forcing=total[...,1],**parts)
    metadata=dict(seed=args.seed,base_sha256=hashlib.sha256(args.base.read_bytes()).hexdigest(),
        modes_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),n=args.n,
        equation='phi=u_x_original + sum A exp(-0.5 sum((X-c)/sigma)^2) + A_t/2*(1-tanh((r^2-R^2)/(2*R*delta))); f=-Delta_X(phi)',
        velocity_note='Only the scalar Poisson MMS is augmented; original velocity coefficients remain divergence-free and unchanged.',
        parameters={k:v.tolist() for k,v in parameters.items()},cpu_gpu_relative_errors=errors.tolist(),
        gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
    metadata['includes_tanh']=not args.no_tanh
    if args.no_tanh:
        metadata['equation']='phi=u_x_original + sum A exp(-0.5 sum((X-c)/sigma)^2); f=-Delta_X(phi)'
    (args.out/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
