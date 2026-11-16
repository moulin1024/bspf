"""Prepare the existing BSPF trial space; run/audit the native 3D CUDA solver."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import cupy as cp
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT/'bench/curved_poisson/src'),str(HERE.parent/'mfem_poisson')]
from optimized_bvp import HybridAxis
from nodal_forcing import spline_matrix
from nodal_mass import NodalMass
# Distinct module name avoids collisions with other benchmark run.py files.
import importlib.util
spec=importlib.util.spec_from_file_location('mfem_mms_sampler',HERE.parent/'mfem_poisson/run.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
Sampler,CurvedBox,write_bundle,read_bundle=shared.Sampler,shared.CurvedBox,shared.write_bundle,shared.read_bundle
DATA=HERE.parent/'evidence/nonperiodic_curved_shell32/modes.npz'
EXE=Path(os.environ.get('BSPF_POISSON_BINARY','/tmp/bspf-poisson3d-cuda-build/bspf_poisson3d'))

def tensor(u,ax,ay,az,xp=np):
    return xp.einsum('ai,bj,ck,kji->cba',ax,ay,az,u,optimize=True)

def grid(nodes):
    z,y,x=np.meshgrid(nodes,nodes,nodes,indexing='ij');return np.stack([x,y,z],axis=-1)

def prepare(n,q,polynomial,folder,basis='bspf',modes=DATA,forcing_mode='nodal-bspf',spline_degree=5,rhs_assembly='auto'):
    start=time.perf_counter()
    if basis=='bspf':
        if n<11 or n%2!=1:raise ValueError('Use odd n >= 11: degree 8 + paired Fourier modes')
        axis=HybridAxis((n-9)//2,degree=8,nodes='symmetric_qr')
    elif basis=='chebyshev':
        sys.path.insert(0,str(HERE.parent))
        from chebyshev_poisson.axis import ChebyshevAxis
        axis=ChebyshevAxis(n)
    else:raise ValueError(f'Unknown trial basis: {basis}')
    z,w=np.polynomial.legendre.leggauss(q);z=(z+1)/2;w=w/2
    B,D=axis.evaluate(z,1);mass=B.T@(w[:,None]*B);K=D.T@(w[:,None]*D)
    lam,V=la.eigh(K[1:-1,1:-1],mass[1:-1,1:-1])
    if lam.min()<=0:raise RuntimeError('nonpositive axis spectrum')
    eigdef=float(np.linalg.norm(K[1:-1,1:-1]@V-mass[1:-1,1:-1]@(V*lam))/np.linalg.norm(K[1:-1,1:-1]@V))
    sampler=Sampler(modes);shape=(q,q,q)
    if rhs_assembly=='auto':rhs_assembly='nodal-mass' if forcing_mode=='nodal-bspf' else 'quadrature'
    if rhs_assembly not in ('nodal-mass','quadrature') or (rhs_assembly=='nodal-mass' and forcing_mode!='nodal-bspf'):
        raise ValueError('nodal-mass assembly requires full nodal-bspf forcing')
    forcing=None if rhs_assembly=='nodal-mass' else np.empty(shape)
    assembled_load=None;mass_setup_s=mass_apply_s=0.;mass_bytes=0
    measure=np.empty(shape);metrics=np.empty((6,)+shape)
    if forcing_mode not in ('nodal-bspf','nodal-spline','quadrature'):
        raise ValueError('Unknown forcing sampling mode')
    if forcing_mode=='nodal-bspf' and basis!='bspf':
        raise ValueError('nodal-bspf forcing requires the BSPF trial basis')
    forcing_xy=None;interpolation=None
    if forcing_mode!='quadrature':
        nodes=cp.asarray(axis.x);node_y,node_x=cp.meshgrid(nodes,nodes,indexing='ij')
        nodal=cp.empty((n,n,n),dtype=cp.float64)
        for k,zk in enumerate(axis.x):
            points=cp.stack([node_x,node_y,cp.full_like(node_x,zk)],axis=-1)
            physical,_,_=CurvedBox().map(points,cp)
            nodal[k]=sampler(physical,polynomial)[...,1]
        if rhs_assembly=='nodal-mass':
            cp.cuda.get_current_stream().synchronize();stamp=time.perf_counter()
            mass_operator=NodalMass(B,z,w,CurvedBox(),xp=cp)
            cp.cuda.get_current_stream().synchronize();mass_setup_s=time.perf_counter()-stamp
            stamp=time.perf_counter();device_load=mass_operator(nodal)
            cp.cuda.get_current_stream().synchronize();mass_apply_s=time.perf_counter()-stamp
            assembled_load=cp.asnumpy(device_load);mass_bytes=mass_operator.bytes
            del device_load,mass_operator
        else:
            interpolation=cp.asarray(B if forcing_mode=='nodal-bspf' else spline_matrix(axis.x,z,spline_degree))
            # Share two contractions, then interpolate one z slab at a time.
            forcing_xy=cp.einsum('ai,bj,kji->kba',interpolation,interpolation,nodal,optimize=True)
        del nodal
    # Generate the source and geometry in CUDA slabs; bound preparation memory.
    yz=cp.asarray(z);wx=cp.asarray(w)
    yy,xx=cp.meshgrid(yz,yz,indexing='ij')
    for k,zk in enumerate(z):
        points=cp.stack([xx,yy,cp.full_like(xx,zk)],axis=-1)
        physical,J,det=CurvedBox().map(points,cp)
        inv=cp.linalg.inv(J);G=inv@inv.swapaxes(-1,-2)
        wt=w[k]*wx[:,None]*wx[None,:]*det
        if forcing is not None:
            source=(sampler(physical,polynomial)[...,1] if forcing_mode=='quadrature' else
                    cp.einsum('k,kji->ji',interpolation[k],forcing_xy,optimize=True))
            forcing[k]=cp.asnumpy(source)
        measure[k]=cp.asnumpy(wt)
        for c,(i,j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):metrics[c,k]=cp.asnumpy(wt*G[...,i,j])
    nodepoints=grid(axis.x);physical,_,_=CurvedBox().map(nodepoints)
    wall=np.ones((n,n,n),bool);wall[1:-1,1:-1,1:-1]=False
    boundary=np.zeros((n,n,n));boundary[wall]=cp.asnumpy(sampler(physical[wall],polynomial)[:,0])
    rng=np.random.default_rng(20261116);probe=rng.normal(size=(n,n,n));probe[wall]=0
    arrays=dict(B=B,D=D,V=V,**{'lambda':lam},measure=measure.reshape(q*q,q),boundary=boundary.reshape(n*n,n),probe=probe.reshape(n*n,n))
    if assembled_load is not None:arrays['load']=assembled_load.reshape(n*n,n)
    else:arrays['forcing']=forcing.reshape(q*q,q)
    for c,key in enumerate(['xx','xy','xz','yy','yz','zz']):arrays[key]=metrics[c].reshape(q*q,q)
    write_bundle(folder/'input.bin',arrays)
    np.savez(folder/'axis.npz',x=axis.x,coefficients=axis.coefficients)
    metadata=dict(axis=axis.metadata,eigenpair_relative_defect=eigdef,preparation_s=time.perf_counter()-start,modes_sha256=hashlib.sha256(Path(modes).read_bytes()).hexdigest(),
                  forcing_mode=forcing_mode,forcing_spline_degree=spline_degree if forcing_mode=='nodal-spline' else None,
                  volume_forcing_samples=q**3 if forcing_mode=='quadrature' else n**3,
                  rhs_assembly=rhs_assembly,rhs_mass_setup_s=mass_setup_s,rhs_mass_apply_s=mass_apply_s,rhs_mass_bytes=mass_bytes)
    (folder/'preparation.json').write_text(json.dumps(metadata,indent=2)+'\n')
    cp.get_default_memory_pool().free_all_blocks()
    return axis,arrays,sampler,metadata

def audit(axis,u,sampler,q,polynomial):
    z,w=np.polynomial.legendre.leggauss(q);z=(z+1)/2;w=w/2
    B=cp.asarray(axis.evaluate(z,0)[0]);U=cp.asarray(u)
    # First x,y contractions, then stream z quadrature slices.
    xy=cp.einsum('ai,bj,kji->kba',B,B,U,optimize=True)
    yy,xx=cp.meshgrid(cp.asarray(z),cp.asarray(z),indexing='ij');ww=cp.asarray(w[:,None]*w[None,:])
    sums=cp.zeros(3)
    for k,zk in enumerate(z):
        values=cp.einsum('k,kji->ji',B[k],xy)
        physical,_,det=CurvedBox().map(cp.stack([xx,yy,cp.full_like(xx,zk)],axis=-1),cp)
        exact=sampler(physical,polynomial)[...,0];weight=ww*w[k]*det
        sums+=cp.stack([cp.sum(weight*(values-exact)**2),cp.sum(weight*exact**2),cp.sum(weight)])
    error,norm,volume=cp.asnumpy(sums)
    return dict(q=q,relative_l2=float(np.sqrt(error/norm)),volume=float(volume))

def check_operator(arrays,folder,n):
    u=arrays['probe'].reshape(n,n,n);B,D,V=arrays['B'],arrays['D'],arrays['V'];q=len(B)
    g=[arrays[key].reshape(q,q,q) for key in ['xx','xy','xz','yy','yz','zz']]
    dx,dy,dz=tensor(u,D,B,B),tensor(u,B,D,B),tensor(u,B,B,D)
    exact=tensor(g[0]*dx+g[1]*dy+g[2]*dz,D.T,B.T,B.T)+tensor(g[1]*dx+g[3]*dy+g[4]*dz,B.T,D.T,B.T)+tensor(g[2]*dx+g[4]*dy+g[5]*dz,B.T,B.T,D.T)
    native=read_bundle(folder/'solution_check.bin')
    op_error=float(la.norm(native['apply'].reshape(n,n,n)-exact)/la.norm(exact))
    lam=arrays['lambda'];av=[g[i].sum() for i in [0,3,5]]
    denom=av[0]*lam[None,None,:]+av[1]*lam[None,:,None]+av[2]*lam[:,None,None]
    z=tensor(tensor(u[1:-1,1:-1,1:-1],V.T,V.T,V.T)/denom,V,V,V)
    zn=native['precondition'].reshape(n,n,n)
    pc_error=float(la.norm(zn[1:-1,1:-1,1:-1]-z)/la.norm(z))
    energy=float(np.sum(u*native['apply'].reshape(n,n,n)))
    positive=float(np.sum(u*zn))
    if op_error>2e-12 or pc_error>3e-6 or energy<=0 or positive<=0:raise RuntimeError((op_error,pc_error,energy,positive))
    return dict(operator_relative_error=op_error,preconditioner_relative_error=pc_error,energy=energy,preconditioned_energy=positive)

def case(args,n):
    q=args.q or max((3*n+1)//2,48 if args.polynomial else 96);folder=args.output/(f'n{n}_q{q}'+('_polynomial' if args.polynomial else ''));folder.mkdir(parents=True,exist_ok=True)
    modes=Path(getattr(args,'modes',DATA))
    forcing_mode=getattr(args,'forcing_mode','nodal-bspf' if getattr(args,'basis','bspf')=='bspf' else 'quadrature')
    spline_degree=getattr(args,'forcing_spline_degree',5)
    rhs_assembly=getattr(args,'rhs_assembly','auto')
    if rhs_assembly=='auto':rhs_assembly='nodal-mass' if forcing_mode=='nodal-bspf' else 'quadrature'
    if (folder/'result.json').exists():
        previous=json.loads((folder/'result.json').read_text())
        if previous['modes_sha256']!=hashlib.sha256(modes.read_bytes()).hexdigest():
            raise ValueError('Different MMS in existing results; choose a separate --output directory')
        if (previous.get('forcing_mode','quadrature')!=forcing_mode or
            (forcing_mode=='nodal-spline' and previous.get('forcing_spline_degree')!=spline_degree)):
            raise ValueError('Different forcing interpolation in existing results; choose a separate --output directory')
        if previous.get('rhs_assembly','quadrature')!=rhs_assembly:
            raise ValueError('Different RHS assembly in existing results; choose a separate --output directory')
    axis,arrays,sampler,meta=prepare(n,q,args.polynomial,folder,basis=getattr(args,'basis','bspf'),modes=modes,forcing_mode=forcing_mode,spline_degree=spline_degree,rhs_assembly=rhs_assembly)
    env=os.environ.copy();sdk=os.environ.get('BSPF_HPC_SDK','/home/moulin/Workspace/nvidia/hpc_sdk/Linux_x86_64/26.5')
    env['LD_LIBRARY_PATH']=os.environ.get('BSPF_NATIVE_LIBRARY_PATH', f'{sdk}/math_libs/lib64:{sdk}/cuda/13.2/lib64')
    cmd=[str(EXE),str(folder/'input.bin'),str(folder/'solution')]+(['check'] if args.check else [])
    binary_hash=hashlib.sha256(EXE.read_bytes()).hexdigest()
    with (folder/'native.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((folder/'solution.json').read_text());u=read_bundle(folder/'solution.bin')['solution'].reshape(n,n,n)
    if args.check:result['cpu_reference_check']=check_operator(arrays,folder,n)
    wall=np.ones(u.shape,bool);wall[1:-1,1:-1,1:-1]=False
    result['boundary_error']=float(np.max(abs(u[wall]-arrays['boundary'].reshape(u.shape)[wall])))
    if result['boundary_error']!=0:raise RuntimeError('Dirichlet boundary changed')
    qa=max(96,q+8)
    result['audits']=[audit(axis,u,sampler,k,args.polynomial) for k in [qa,qa+8]]
    print(json.dumps(dict(n=n,quadrature=q,audits=result['audits'])),flush=True)
    # Localized interfaces can need finer error integration than the trial space.
    while (not args.polynomial and sampler.features.active and
           abs(result['audits'][-2]['relative_l2']/result['audits'][-1]['relative_l2']-1)>1e-4):
        qa=result['audits'][-1]['q']+max(16,result['audits'][-1]['q']//4)
        if qa>512:raise RuntimeError(f'Error quadrature unresolved: {result["audits"]}')
        result['audits'].append(audit(axis,u,sampler,qa,args.polynomial))
        print(json.dumps(dict(n=n,audit=result['audits'][-1])),flush=True)
    result['relative_l2']=result['audits'][-1]['relative_l2']
    if args.polynomial and result['relative_l2']>1e-9:raise RuntimeError(result['audits'])
    if not args.polynomial and abs(result['audits'][-2]['relative_l2']/result['relative_l2']-1)>1e-4:raise RuntimeError('Error quadrature unresolved')
    result.update(meta);result['warm_cg_s']=float(np.median([r['cg_s'] for r in result['runs'][1:]]));result['warm_wall_s']=float(np.median([r['wall_s'] for r in result['runs'][1:]]))
    result['binary_sha256']=binary_hash
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    cp.get_default_memory_pool().free_all_blocks()

def main():
    p=argparse.ArgumentParser();p.add_argument('--modes',type=Path,default=DATA);p.add_argument('--n',type=int,nargs='+',default=[17,33,65]);p.add_argument('--q',type=int,help='Gauss points per axis (default: ceil(1.5*n), minimum 96 for turbulence or 48 for polynomial MMS)');p.add_argument('--polynomial',action='store_true');p.add_argument('--check',action='store_true');p.add_argument('--output',type=Path,default=HERE.parent/'evidence/bspf_cuda3d_rtx3070')
    p.add_argument('--forcing-mode',choices=['nodal-bspf','nodal-spline','quadrature'],default='nodal-bspf')
    p.add_argument('--rhs-assembly',choices=['auto','nodal-mass','quadrature'],default='auto',help='auto uses precomputed mass factors for full BSPF nodal forcing')
    p.add_argument('--forcing-spline-degree',type=int,default=5,help='Odd interpolating B-spline degree (default: 5)')
    args=p.parse_args()
    if args.q and args.q<max(args.n):p.error('q must be >= n')
    if args.forcing_mode=='nodal-spline' and (args.forcing_spline_degree<1 or args.forcing_spline_degree%2!=1 or args.forcing_spline_degree>=min(args.n)):
        p.error('Forcing spline degree must be positive, odd, and smaller than every n')
    for n in args.n:case(args,n)
if __name__=='__main__':main()
