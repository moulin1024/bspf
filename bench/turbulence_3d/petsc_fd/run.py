"""Prepare/audit the FD8 curved-domain MMS; CUDA PETSc outer Krylov solve."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/bspf-matplotlib')
import argparse,hashlib,importlib.util,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import cupy as cp
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parents[1]/'curved_poisson/benchmarks/internal_soliton/cpp'))
from mms import CurvedBox
from bundle import read_bundle,write_bundle
from discretization import stencils,geometry,cpu_apply
from auxiliary import assemble
spec=importlib.util.spec_from_file_location('fd_shared_sampler',HERE.parent/'mfem_poisson/run.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
DATA=HERE.parent/'evidence/nonperiodic_curved_shell32/modes.npz'
EXE=Path('/tmp/bspf-petsc-fd8-build/petsc_fd8')
SDK=os.environ.get('BSPF_HPC_SDK','/home/moulin/Workspace/nvidia/hpc_sdk/Linux_x86_64/26.5')

def field(points,kind,sampler):
    if kind=='turbulence':return sampler(points,False)
    if kind=='polynomial':return cp.stack([1+cp.sum(points**2,axis=-1),cp.full(points.shape[:-1],-6.)],axis=-1)
    x,y,z=[points[...,i] for i in range(3)]
    s=cp.sin(2*cp.pi*x)*cp.sin(2*cp.pi*y)*cp.sin(2*cp.pi*z)
    c=.2*cp.cos(cp.pi*(3*x+2*y-z))
    return cp.stack([s+c,cp.pi**2*(12*s+14*c)],axis=-1)

def prepare(n,kind,flat,folder):
    start=time.perf_counter();box=CurvedBox((0,0,0)) if flat else CurvedBox()
    nodes=np.linspace(0,1,n);starts,weights=stencils(n);sampler=shared.Sampler(DATA) if kind=='turbulence' else None
    u=np.empty((n,n,n));force=np.empty_like(u);determinant=np.empty_like(u);G=np.empty(u.shape+(3,3));drift=np.empty(u.shape+(3,))
    yy,xx=cp.meshgrid(cp.asarray(nodes),cp.asarray(nodes),indexing='ij')
    for k,z in enumerate(nodes):
        points=cp.stack([xx,yy,cp.full_like(xx,z)],axis=-1)
        physical,g,b,det=geometry(points,box,cp);values=field(physical,kind,sampler)
        u[k]=cp.asnumpy(values[...,0]);force[k]=cp.asnumpy(values[...,1]);determinant[k]=cp.asnumpy(det)
        G[k]=cp.asnumpy(g);drift[k]=cp.asnumpy(b)
    boundary=u.copy();boundary[1:-1,1:-1,1:-1]=0
    interior=(slice(1,-1),)*3
    metric=np.stack([G[...,i,j][interior] for i,j in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]])
    b=np.moveaxis(drift[interior],-1,0)
    rng=np.random.default_rng(20261116);probe=rng.normal(size=u.shape)
    m=n-2
    arrays=dict(n=np.array([[n]]),starts=starts,weights=weights.reshape(2*n,10),metric=metric.reshape(6*m*m,m),
                drift=b.reshape(3*m*m,m),boundary=boundary.reshape(n*n,n),forcing=force[interior].reshape(m*m,m),probe=probe.reshape(n*n,n))
    assembly_start=time.perf_counter()
    row,col,values,mass=assemble(n,box)
    def packed(a):
        out=np.zeros(((len(a)+1023)//1024)*1024);out[:len(a)]=a
        return out.reshape(-1,1024)
    arrays.update(pc_nnz=np.array([[len(col)]]),pc_rows=packed(row),pc_cols=packed(col),pc_values=packed(values))
    (folder/'auxiliary.json').write_text(json.dumps(dict(type='mass-lumped full-metric Q1',assembly='CUDA',nnz=len(col),assembly_s=time.perf_counter()-assembly_start,min_lumped_mass=float(mass.min())))+'\n')
    write_bundle(folder/'input.bin',arrays)
    np.savez(folder/'reference.npz',exact=u,det=determinant)
    cp.get_default_memory_pool().free_all_blocks()
    return u,determinant,G,drift,probe,starts,weights,sampler,box,time.perf_counter()-start

def interpolation(nodes,points):
    n=len(nodes);B=np.zeros((len(points),n))
    for k,x in enumerate(points):
        start=min(max(int(np.floor(x*(n-1)))-4,0),n-9);ids=np.arange(start,start+9)
        for i in ids:
            others=ids[ids!=i];B[k,i]=np.prod((x-nodes[others])/(nodes[i]-nodes[others]))
    return B

def audit(u,kind,sampler,box,q):
    n=len(u);x,w=np.polynomial.legendre.leggauss(q);x=(x+1)/2;w=w/2
    B=cp.asarray(interpolation(np.linspace(0,1,n),x));U=cp.asarray(u)
    xy=cp.einsum('ai,bj,kji->kba',B,B,U,optimize=True)
    yy,xx=cp.meshgrid(cp.asarray(x),cp.asarray(x),indexing='ij');ww=cp.asarray(w[:,None]*w[None,:]);sums=cp.zeros(2)
    for k,z in enumerate(x):
        values=cp.einsum('k,kji->ji',B[k],xy)
        physical,_,det=box.map(cp.stack([xx,yy,cp.full_like(xx,z)],axis=-1),cp)
        exact=field(physical,kind,sampler)[...,0];wt=ww*w[k]*det
        sums+=cp.stack([cp.sum(wt*(values-exact)**2),cp.sum(wt*exact**2)])
    a,b=cp.asnumpy(sums);return dict(q=q,relative_l2=float(np.sqrt(a/b)))

def case(args,n):
    name=f'{args.field}_{"flat" if args.flat else "curved"}_n{n}_{args.pc}'
    folder=args.output/name;folder.mkdir(parents=True,exist_ok=True)
    u,det,G,drift,probe,starts,weights,sampler,box,prep=prepare(n,args.field,args.flat,folder)
    env=os.environ.copy();env['LD_LIBRARY_PATH']=f'/tmp/petsc-3.26.0/arch-bspf-cuda/lib:/tmp/bspf-mfem-poisson3d-fp32:{SDK}/math_libs/lib64:{SDK}/cuda/13.2/lib64:{SDK}/comm_libs/13.2/hpcx/hpcx-2.50/ompi5/lib'
    env['OMPI_MCA_pml']='ob1';env['OMPI_MCA_btl']='self';env['OPENBLAS_NUM_THREADS']='1'
    command=[str(EXE),'-input',str(folder/'input.bin'),'-output',str(folder/'solution'),'-pc_kind',args.pc,'-use_gpu_aware_mpi','0']
    with (folder/'native.log').open('w') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    native=read_bundle(folder/'solution.bin');v=u.copy();v[1:-1,1:-1,1:-1]=native['interior'].reshape((n-2,)*3)
    result=json.loads((folder/'solution.json').read_text());w=det.copy()
    for axis in range(3):
        for i in [0,-1]:
            sl=[slice(None)]*3;sl[axis]=i;w[tuple(sl)]*=.5
    result.update(field=args.field,flat=args.flat,relative_nodal_l2=float(np.sqrt(np.sum(w*(v-u)**2)/np.sum(w*u*u))),preparation_s=prep,
                  warm_solve_s=float(np.median(result['solve_s'][1:])),binary_sha256=hashlib.sha256(EXE.read_bytes()).hexdigest(),
                  modes_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest() if sampler else None)
    if args.check:
        ref=cpu_apply(probe,starts,weights,G,drift)
        result['cpu_operator_relative_error']=float(np.linalg.norm(ref-native['probe_apply'].reshape(ref.shape))/np.linalg.norm(ref))
        if result['cpu_operator_relative_error']>2e-12:raise RuntimeError('CPU/GPU operator mismatch')
    if args.audit:
        q=max(96,(3*n+1)//2)
        result['audits']=[audit(v,args.field,sampler,box,k) for k in [q,q+8]]
        result['relative_l2']=result['audits'][-1]['relative_l2']
        result['audit_relative_change']=abs(result['audits'][0]['relative_l2']/result['relative_l2']-1)
        if abs(result['audits'][0]['relative_l2']-result['relative_l2'])>max(1e-10,.005*result['relative_l2']):raise RuntimeError('Error integration not resolved')
    if args.field=='polynomial' and result['relative_nodal_l2']>1e-9:raise RuntimeError('Polynomial guard failed')
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    cp.get_default_memory_pool().free_all_blocks()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,nargs='+',default=[17,33,65]);p.add_argument('--field',choices=['polynomial','smooth','turbulence'],default='turbulence');p.add_argument('--flat',action='store_true');p.add_argument('--check',action='store_true');p.add_argument('--audit',action='store_true');p.add_argument('--pc',choices=['fp32','hypre'],default='fp32');p.add_argument('--output',type=Path,default=HERE.parent/'evidence/petsc_fd8_rtx3070')
    args=p.parse_args();args.output=args.output.resolve();results=[case(args,n) for n in args.n]
    if len(results)>1:
        orders=[float(np.log(a['relative_nodal_l2']/b['relative_nodal_l2'])/np.log((b['n']-1)/(a['n']-1))) for a,b in zip(results,results[1:])]
        print('Observed nodal L2 orders:',orders,flush=True)
if __name__=='__main__':main()
