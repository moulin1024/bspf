"""NekRS/CUDA scalar Poisson benchmark, same DOFs/domain/MMS as MFEM."""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import cupy as cp
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'mfem_poisson'))
from run import Sampler, audit, CurvedBox, write_bundle
from mesh import write_mesh

def native(folder, export=False):
    env=os.environ.copy()
    sdk=os.environ.get('BSPF_HPC_SDK','/home/moulin/Workspace/nvidia/hpc_sdk/Linux_x86_64/26.5')
    prefix='/tmp/bspf-nekrs-install'
    env.update(NEKRS_HOME=prefix,NEKRS_JITC_NTHREADS='8',OMPI_CC='gcc',OMPI_CXX='g++',OMPI_FC='gfortran',
               OMPI_MCA_pml='ob1',OMPI_MCA_btl='self',
               OCCA_CACHE_DIR='/tmp/bspf-nekrs-occa-cache',
               NEKRS_CACHE_DIR='/tmp/bspf-nekrs-case-cache/'+folder.name,
               PATH=f'{prefix}/bin:{sdk}/comm_libs/mpi/bin:{sdk}/cuda/13.2/bin:'+env['PATH'],
               LD_LIBRARY_PATH=f'{prefix}/lib:{prefix}/occa/lib:{sdk}/math_libs/lib64:{sdk}/cuda/13.2/lib64:{sdk}/comm_libs/13.2/hpcx/hpcx-2.50/ompi5/lib')
    Path(env['NEKRS_CACHE_DIR']).mkdir(parents=True,exist_ok=True)
    Path(env['OCCA_CACHE_DIR']).mkdir(parents=True,exist_ok=True)
    if export:env['BSPF_EXPORT']='1'
    else:env.pop('BSPF_EXPORT',None)
    with (folder/('export.log' if export else 'solve.log')).open('w') as log:
        subprocess.run([prefix+'/bin/nekrs','--setup','poisson','--backend','CUDA','--device-id','0'],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

def case(output,e,sampler,polynomial=False,audit_only=False):
    folder=output/(f'e{e}_p8'+('_polynomial' if polynomial else ''));folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():
        previous=json.loads((folder/'result.json').read_text())
        original_hash=hashlib.sha256((HERE.parent/'evidence/nonperiodic_curved_shell32/modes.npz').read_bytes()).hexdigest()
        if previous.get('modes_sha256',original_hash)!=sampler.modes_sha256:
            raise ValueError('Different MMS in existing results; choose a separate --output directory')
    if not audit_only:
        for name in ['poisson.udf','poisson.par']:shutil.copyfile(HERE/name,folder/name)
        write_mesh(folder/'poisson.re2',e)
        native(folder,True)
    mesh=json.loads((folder/'mesh.json').read_text())
    assert mesh['unique_dofs']==(8*e+1)**3,mesh
    assert mesh['free_dofs']==(8*e-1)**3,mesh
    ref=np.fromfile(folder/'reference.bin').reshape(-1,3)
    xyz=np.fromfile(folder/'coordinates.bin').reshape(-1,3)
    geometry_error=float(np.max(np.abs(CurvedBox().map(ref)[0]-xyz)))
    assert geometry_error<1e-12,geometry_error
    d=9; refs=ref.reshape(-1,d,d,d,3);origins=refs[:,0,0,0,:]
    nodes=(refs[0,0,0,:,0]-origins[0,0])*e
    zz,yy,xx=np.meshgrid(nodes,nodes,nodes,indexing='ij')
    expected=origins[:,None,None,None,:]+np.stack([xx,yy,zz],axis=-1)/e
    assert np.max(np.abs(refs-expected))<1e-12,'Unexpected element/node orientation'
    sample_time=None
    if not audit_only:
        start=time.perf_counter();values=cp.asnumpy(sampler(xyz,polynomial))
        values[:,0].tofile(folder/'exact.bin');values[:,1].tofile(folder/'force.bin')
        sample_time=time.perf_counter()-start
        del values;cp.get_default_memory_pool().free_all_blocks()
        native(folder)
    coeff=np.fromfile(folder/'solution.bin').reshape(-1,d**3)
    write_bundle(folder/'solution_local.bin',dict(coefficients=coeff,nodes=nodes[None,:],origins=origins,probes=np.fromfile(folder/'probes.bin').reshape(-1,5)))
    q=max(12,int(np.ceil(1.8*32/e))+8)
    checks=[audit(folder,sampler,e,qq,polynomial) for qq in (q,q+4)]
    while (not polynomial and sampler.features.active and
           abs(checks[-2]['relative_l2']/checks[-1]['relative_l2']-1)>3e-5):
        q=checks[-1]['q']+max(4,checks[-1]['q']//4)
        if q>160:raise RuntimeError(f'Error quadrature not converged: {checks}')
        checks.append(audit(folder,sampler,e,q,polynomial))
    if not polynomial and abs(checks[-2]['relative_l2']/checks[-1]['relative_l2']-1)>3e-5:
        raise RuntimeError(f'Error quadrature not converged: {checks}')
    if abs(checks[-1]['volume']-.92)>1e-12:raise RuntimeError(checks)
    timing=json.loads((folder/'timing.json').read_text());residual=json.loads((folder/'residual.json').read_text())
    if residual['relative_residual']>1e-9 or residual['boundary_max_error']>1e-12:raise RuntimeError(residual)
    if polynomial and checks[-1]['relative_l2']>1e-9:raise RuntimeError(checks)
    result=dict(elements_per_axis=e,modes_sha256=sampler.modes_sha256,**mesh,geometry_error=geometry_error,**timing,**residual,
                audit=checks,relative_l2=checks[-1]['relative_l2'],gpu_sampling_s=sample_time,
                warm_wall_s=float(np.median([v['wall_s'] for v in timing['runs'][1:]])),
                warm_solve_s=float(np.median([v['solve_s'] for v in timing['runs'][1:]])))
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    cp.get_default_memory_pool().free_all_blocks()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--elements',nargs='+',type=int,default=[2,4,8,12,16,18]);p.add_argument('--polynomial',action='store_true');p.add_argument('--audit-only',action='store_true')
    p.add_argument('--modes',type=Path,default=HERE.parent/'evidence/nonperiodic_curved_shell32/modes.npz')
    p.add_argument('--output',type=Path,default=HERE.parent/'evidence/nekrs_rtx3070');args=p.parse_args()
    sampler=Sampler(args.modes)
    args.output.mkdir(parents=True,exist_ok=True)
    for e in args.elements:
        if not 1<=e<=18:raise ValueError('Current benchmark cap: 18 elements per axis (145 global nodes)')
        case(args.output,e,sampler,args.polynomial,args.audit_only)
if __name__=='__main__':main()
