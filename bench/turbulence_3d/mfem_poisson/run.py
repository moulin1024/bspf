"""GPU scalar Poisson convergence campaign for the curved nonperiodic MMS."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import cupy as cp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent/'curved_poisson/benchmarks/internal_soliton/cpp'))
from mms import CurvedBox, PlaneWaveMMS
from poisson_features import PoissonFeatures
from bundle import read_bundle, write_bundle


def basis(nodes, points):
    weights = np.array([1/np.prod(v-np.delete(nodes,i)) for i,v in enumerate(nodes)])
    delta = np.asarray(points)[:,None]-nodes
    hit = delta == 0
    safe = np.where(hit, 1, delta)
    out = weights/safe
    out /= out.sum(axis=1)[:,None]
    for row in np.where(hit.any(axis=1))[0]:
        out[row] = hit[row]
    return out


class Sampler:
    def __init__(self, path):
        self.modes_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.m = PlaneWaveMMS.load(path, xp=cp)
        self.features=PoissonFeatures.load(path,xp=cp)
        k2=cp.sum(self.m.k**2,axis=1)
        self.c=cp.stack((self.m.a[:,0], k2*self.m.a[:,0]),axis=1)
        self.s=cp.stack((self.m.b[:,0], k2*self.m.b[:,0]),axis=1)

    def __call__(self, points, polynomial=False):
        points=cp.asarray(points)
        if polynomial:
            return cp.stack((1+cp.sum(points**2,axis=-1),cp.full(points.shape[:-1],-6.)),axis=-1)
        shape=points.shape[:-1]; points=points.reshape(-1,3)
        out=cp.empty((len(points),2))
        for start in range(0,len(points),2048):
            sl=slice(start,start+2048)
            phase=(points[sl]-self.m.center)@self.m.k.T
            out[sl]=cp.cos(phase)@self.c+cp.sin(phase)@self.s
        out[:,0]+=self.m.u_offset[0]
        if self.features.active:
            extra,forcing=self.features.evaluate(points)
            out[:,0]+=extra;out[:,1]+=forcing
        return out.reshape(shape+(2,))


def audit(folder, sampler, e, q, polynomial=False):
    data=read_bundle(folder/'solution_local.bin'); nodes=data['nodes'].ravel(); d=len(nodes)
    coeff=data['coefficients'].reshape(-1,d,d,d)
    probe_error=0.
    for el,x,y,z,value in data['probes']:
        v=np.einsum('k,j,i,kji->',basis(nodes,[z])[0],basis(nodes,[y])[0],basis(nodes,[x])[0],coeff[int(el)])
        probe_error=max(probe_error,abs(v-value))
    if probe_error>1e-10: raise RuntimeError(f'Native probe mismatch {probe_error}')
    g,w=np.polynomial.legendre.leggauss(q); g=(g+1)/2; w=w/2
    zz,yy,xx=np.meshgrid(g,g,g,indexing='ij')
    local=cp.asarray(np.stack((xx,yy,zz),axis=-1))
    weights=cp.asarray(w[:,None,None]*w[None,:,None]*w[None,None,:]/e**3)
    B=cp.asarray(basis(nodes,g)); sums=cp.zeros(3)
    batch=max(1,32768//q**3)
    for start in range(0,len(coeff),batch):
        c=cp.asarray(coeff[start:start+batch]); origins=cp.asarray(data['origins'][start:start+batch])
        values=cp.matmul(c,B.T)
        values=cp.einsum('by,ezyx->ezbx',B,values)
        values=cp.einsum('az,ezyx->eayx',B,values)
        points,_,det=CurvedBox().map(origins[:,None,None,None,:]+local/e,cp)
        exact=sampler(points,polynomial)[...,0]
        wt=weights*det
        sums+=cp.stack((cp.sum(wt*(values-exact)**2),cp.sum(wt*exact**2),cp.sum(wt)))
    err,norm,volume=cp.asnumpy(sums)
    return dict(q=q,relative_l2=float(np.sqrt(err/norm)),absolute_l2=float(np.sqrt(err)),volume=float(volume),native_probe_error=probe_error)


def native(args, folder, e,p,q,action):
    env=os.environ.copy()
    base=os.environ.get('BSPF_HPC_SDK','/home/moulin/Workspace/nvidia/hpc_sdk/Linux_x86_64/26.5')
    env['LD_LIBRARY_PATH']=f'/tmp/bspf-libceed/lib:{base}/math_libs/lib64:{base}/cuda/13.2/lib64:{base}/comm_libs/13.2/hpcx/hpcx-2.50/ompi5/lib'
    command=[str(args.binary),str(e),str(p),str(q),action,'ceed-cuda:/gpu/cuda/gen',str(folder),str(p+4),'1',str(args.repeats)]
    if args.preconditioner!='jacobi': command.append(args.preconditioner)
    env['OMPI_MCA_pml']='ob1';env['OMPI_MCA_btl']='self'
    with (folder/f'{action}.log').open('w') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


def case(args,sampler,e,p,q,label,polynomial=False):
    folder=args.output/label
    if args.resume and (folder/'result.json').exists():
        cached=json.loads((folder/'result.json').read_text())
        if (cached['elements_per_axis'],cached['degree'],cached['quadrature'],cached['repeats']) != (e,p,q,args.repeats):
            raise RuntimeError('Cached case configuration differs')
        expected={'lor-amg':'LORSolver<HypreBoomerAMG>', 'lor-amg-fp32':'LORSolver<FloatHypreBoomerAMG>', 'jacobi':'OperatorJacobiSmoother'}[args.preconditioner]
        if cached['preconditioner']!=expected: raise RuntimeError('Cached preconditioner differs')
        return cached
    config=dict(elements_per_axis=e,degree=p,quadrature=q,repeats=args.repeats,
                polynomial=polynomial,preconditioner=args.preconditioner,modes_sha256=hashlib.sha256(args.modes.read_bytes()).hexdigest())
    config_path=folder/'case_config.json'
    recover=(args.resume and config_path.exists() and
             json.loads(config_path.read_text())==config and
             (folder/'timing.json').exists() and (folder/'solution_local.bin').exists())
    if recover:
        print(f'{label}: recovering completed native solve',flush=True)
        sampling_seconds=json.loads((folder/'sampling.json').read_text())['sampling_seconds']
        geometry_error=json.loads((folder/'sampling.json').read_text())['geometry_error']
    else:
        if folder.exists():
            folder.rename(folder.with_name(label+f'_failed_{time.time_ns()}'))
        folder.mkdir()
        config_path.write_text(json.dumps(config,indent=2)+'\n')
        print(f'{label}: export/sample/solve',flush=True)
        native(args,folder,e,p,q,'export')
        t=time.perf_counter(); data=read_bundle(folder/'coordinates.bin')
        n=int(data['count'].item()); nd=int(data['ndofs'].item())
        points=np.stack([data[k].ravel()[:n] for k in ('x','y','z')],axis=1)
        ref=np.stack([data[k].ravel()[:n] for k in ('s','t','r')],axis=1)
        mapped,_,_=CurvedBox().map(ref)
        geometry_error=float(np.max(abs(points-mapped)))
        if geometry_error>2e-13: raise RuntimeError('Geometry mismatch')
        data['forcing']=np.zeros_like(data['x']);data['forcing'].ravel()[:n]=cp.asnumpy(sampler(points,polynomial)[:,1])
        points=np.stack([data[k].ravel()[:nd] for k in ('bx','by','bz')],axis=1)
        data['boundary']=np.zeros_like(data['bx']); data['boundary'].ravel()[:nd]=cp.asnumpy(sampler(points,polynomial)[:,0])
        write_bundle(folder/'samples.bin',data)
        sampling_seconds=time.perf_counter()-t
        del data,points,ref,mapped
        cp.get_default_memory_pool().free_all_blocks()
        (folder/'sampling.json').write_text(json.dumps(dict(sampling_seconds=sampling_seconds,geometry_error=geometry_error))+'\n')
        native(args,folder,e,p,q,'solve')
    if geometry_error is None and (folder/'coordinates.bin').exists():
        data=read_bundle(folder/'coordinates.bin'); n=int(data['count'].item())
        geometry_error=0.
        for start in range(0,n,65536):
            sl=slice(start,min(start+65536,n))
            reference=np.stack([data[k].ravel()[sl] for k in ('s','t','r')],axis=1)
            physical=np.stack([data[k].ravel()[sl] for k in ('x','y','z')],axis=1)
            mapped,_,_=CurvedBox().map(reference)
            geometry_error=max(geometry_error,float(np.max(abs(mapped-physical))))
        del data
        if geometry_error>2e-13: raise RuntimeError('Recovered geometry mismatch')
    result=json.loads((folder/'timing.json').read_text())
    qa=max(p+4,int(np.ceil(1.8*32/e))+8) if not polynomial else p+4
    print(f'{label}: independent error integration q={qa},{qa+4}',flush=True)
    checks=[audit(folder,sampler,e,qa,polynomial),audit(folder,sampler,e,qa+4,polynomial)]
    while (not polynomial and sampler.features.active and
           abs(checks[-2]['relative_l2']-checks[-1]['relative_l2'])>max(1e-10,checks[-1]['relative_l2']*1e-3)):
        qa=checks[-1]['q']+max(4,checks[-1]['q']//4)
        if qa>160:raise RuntimeError(f'Error quadrature not converged: {checks}')
        checks.append(audit(folder,sampler,e,qa,polynomial))
    if abs(checks[-2]['relative_l2']-checks[-1]['relative_l2'])>max(1e-10,checks[-1]['relative_l2']*1e-3):
        raise RuntimeError(f'Error quadrature not converged: {checks}')
    result.update(label=label,elements_per_axis=e,degree=p,quadrature=q,audits=checks,
                  relative_l2=checks[-1]['relative_l2'],sampling_seconds=sampling_seconds,
                  geometry_error=geometry_error,warm_median_seconds=float(np.median(result['samples_seconds'])))
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    for name in ('coordinates.bin','samples.bin'): (folder/name).unlink(missing_ok=True)
    print(f"{label}: dofs={result['dofs']} error={result['relative_l2']:.4e} warm={result['warm_median_seconds']:.4f}s",flush=True)
    return result


def plot(results,output):
    for r in results:
        r['warm_linear_median_seconds']=float(np.median(r['linear_samples_seconds']))
    keys=['label','elements_per_axis','degree','quadrature','dofs','relative_l2','warm_median_seconds','warm_linear_median_seconds','first_solve_seconds','setup_seconds','preconditioner_setup_seconds','amg_setup_seconds','iterations','rhs_assembly']
    with (output/'results.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore')
        writer.writeheader();writer.writerows(results)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),sharey=True,layout='constrained')
    for ax,key,title in zip(axes,('warm_median_seconds','warm_linear_median_seconds','first_solve_seconds'),('Warm solve: assembly + CG + recovery','Warm GPU CG only','First solve: includes lazy compilation')):
        for p in sorted({r['degree'] for r in results}):
            rows=sorted([r for r in results if r['degree']==p and not r['label'].endswith('qcheck')],key=lambda r:r['elements_per_axis'])
            ax.loglog([r[key] for r in rows],[r['relative_l2'] for r in rows],'o-',label=f'p = {p}')
            for r in rows: ax.annotate(f"{r['elements_per_axis']}³",(r[key],r['relative_l2']),xytext=(4,-13 if p==8 and r['elements_per_axis']==4 else 5),textcoords='offset points',fontsize=8)
        ax.set(xlabel='Wall time [s]',title=title); ax.grid(True,which='both',alpha=.25);ax.legend();ax.margins(x=.15,y=.1)
    axes[0].set_ylabel('Relative physical-domain L² error in $u_x$')
    fig.suptitle(f"3D nonperiodic curved-domain Poisson · RTX 3070 · MFEM/libCEED FP64\nCG + {results[0]['preconditioner']}; labels = number of elements; exact analytic turbulence MMS",fontsize=11)
    fig.savefig(output/'accuracy_walltime.png',dpi=200);fig.savefig(output/'accuracy_walltime.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--binary',type=Path,default=Path('/tmp/bspf-mfem-poisson3d/mfem_poisson3d'))
    parser.add_argument('--modes',type=Path,default=ROOT/'evidence/nonperiodic_curved_shell32/modes.npz')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',default='4:4,8:4,12:4,4:8,8:8,12:8,16:8')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--preconditioner',choices=['jacobi','lor-amg','lor-amg-fp32'],default='jacobi')
    parser.add_argument('--quadrature-check-case',help='e:p case to overintegrate; defaults to the last case')
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=args.resume)
    if args.resume:
        old=json.loads((args.output/'report.json').read_text())
        if old['modes_sha256'] != hashlib.sha256(args.modes.read_bytes()).hexdigest():
            raise RuntimeError('Cannot resume with different modes')
    sampler=Sampler(args.modes)
    guard=case(args,sampler,1,8,11,'polynomial_guard',True) if args.preconditioner.startswith('lor-amg') else case(args,sampler,2,4,8,'polynomial_guard',True)
    if guard['relative_l2']>1e-9: raise RuntimeError('Polynomial reproduction failed')
    results=[]
    for spec in args.cases.split(','):
        e,p=map(int,spec.split(':'));q=p+3
        results.append(case(args,sampler,e,p,q,f'e{e}_p{p}'))
        report=dict(equation='-laplacian(phi)=f; phi=u_x + optional localized scalar features; exact Dirichlet on all six curved faces',
                    modes_sha256=hashlib.sha256(args.modes.read_bytes()).hexdigest(),guard=guard,results=results,
                    timing_scope='Synchronized RHS assembly, boundary elimination, CG, recovery. Excludes setup, analytic sampling and error audit.')
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');plot(results,args.output)
    e,p=map(int,(args.quadrature_check_case or args.cases.split(',')[-1]).split(':'))
    check=case(args,sampler,e,p,p+5,f'e{e}_p{p}_qcheck')
    report['operator_quadrature_check']=check
    baseline=next(r for r in results if r['elements_per_axis']==e and r['degree']==p)
    report['operator_quadrature_relative_error_change']=abs(check['relative_l2']/baseline['relative_l2']-1)
    if report['operator_quadrature_relative_error_change']>0.01:
        raise RuntimeError('Operator quadrature changes final error by more than 1%')
    report['observed_h_orders']={}
    for degree in sorted({r['degree'] for r in results}):
        rows=sorted([r for r in results if r['degree']==degree],key=lambda r:r['elements_per_axis'])
        report['observed_h_orders'][str(degree)]=[float(np.log(a['relative_l2']/b['relative_l2'])/np.log(b['elements_per_axis']/a['elements_per_axis'])) for a,b in zip(rows,rows[1:])]
    plot(results,args.output)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
