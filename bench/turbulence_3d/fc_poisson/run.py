"""Curved, nonperiodic MMS benchmark for the CUDA FC-Gram Poisson adaptation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mms import CurvedBox
from poisson_features import PoissonFeatures
from petsc_fd.discretization import geometry
from solver import FCOperator, fgmres

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'evidence/nonperiodic_curved_shell32/modes.npz'

def gpu(x):return torch.as_tensor(x,dtype=torch.float64,device='cuda')

def reference_grid(nodes):
    z,y,x=np.meshgrid(nodes,nodes,nodes,indexing='ij')
    return np.stack((x,y,z),axis=-1)

class Sampler:
    def __init__(self,field,path=DATA):
        self.field=field
        self.features=PoissonFeatures.load(path,xp=torch,device='cuda')
        with np.load(path) as d:
            self.k=gpu(d['k']);self.a=gpu(d['a'][:,0]);self.b=gpu(d['b'][:,0])
            self.center=gpu(d['center']);self.offset=float(d['u_offset'][0])
        self.k2=(self.k**2).sum(dim=1)
    def __call__(self,X):
        shape=X.shape[:-1];X=gpu(X).reshape(-1,3)
        if self.field=='polynomial':
            return (1+(X**2).sum(dim=1)).reshape(shape),torch.full(shape,-6.,device='cuda',dtype=torch.float64)
        if self.field=='smooth':
            s=torch.sin(2*torch.pi*X).prod(dim=1)
            c=.2*torch.cos(torch.pi*(3*X[:,0]+2*X[:,1]-X[:,2]))
            return (s+c).reshape(shape),(torch.pi**2*(12*s+14*c)).reshape(shape)
        u=torch.empty(len(X),device='cuda',dtype=torch.float64);f=torch.empty_like(u)
        for start in range(0,len(X),2048):
            sl=slice(start,start+2048);phase=(X[sl]-self.center)@self.k.T
            c=phase.cos();s=phase.sin()
            u[sl]=c@self.a+s@self.b+self.offset
            f[sl]=c@(self.k2*self.a)+s@(self.k2*self.b)
        if self.features.active:
            extra,forcing=self.features.evaluate(X);u+=extra;f+=forcing
        return u.reshape(shape),f.reshape(shape)

def audit(A,u,sampler,box,q):
    """Integrate the FC interpolant on independent tensor Gauss nodes."""
    t,w=np.polynomial.legendre.leggauss(q);t=(t+1)/2;w=w/2
    E=A.E.cpu().numpy();freq=A.freq.cpu().numpy();L=len(E)
    B=(np.exp(1j*(t[:,None]+25/(A.n-1))*freq[None,:])@np.fft.fft(E,axis=0)/L).real
    B=gpu(B)
    # Interpolate x,y first; integrate z in slabs to bound sampling memory.
    xy=u@B.T
    xy=torch.movedim(torch.movedim(xy,1,-1)@B.T,-1,1)
    num=0.;den=0.
    for start in range(0,q,4):
        stop=min(start+4,q)
        values=torch.einsum('az,zyx->ayx',B[start:stop],xy)
        z,y,x=np.meshgrid(t[start:stop],t,t,indexing='ij')
        X,_,det=box.map(np.stack((x,y,z),axis=-1))
        exact,_=sampler(X)
        weight=gpu(w[start:stop,None,None]*w[None,:,None]*w[None,None,:]*det)
        num+=((values-exact)**2*weight).sum().item();den+=(exact**2*weight).sum().item()
    return (num/den)**.5

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--modes',type=Path,default=DATA);p.add_argument('--n',type=int,required=True)
    p.add_argument('--field',choices=['turbulence','polynomial','smooth'],default='turbulence')
    p.add_argument('--degree',type=int,default=8);p.add_argument('--repeat',type=int,default=3)
    p.add_argument('--rtol',type=float,default=1e-10);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--audit',action='store_true');args=p.parse_args()
    if (args.out/'result.json').exists():
        previous=json.loads((args.out/'result.json').read_text())
        if previous['data_sha256']!=hashlib.sha256(args.modes.read_bytes()).hexdigest():
            p.error('Different MMS in existing results; choose a separate --out directory')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(1);torch.cuda.init();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();box=CurvedBox();sampler=Sampler(args.field,args.modes)
    X,G,drift,det=geometry(reference_grid(np.linspace(0,1,args.n)),box)
    metric=np.stack([G[...,0,0],G[...,1,1],G[...,2,2],G[...,0,1],G[...,0,2],G[...,1,2]],axis=-1)
    sl=(slice(1,-1),)*3;A=FCOperator(args.n,metric[sl],drift[sl],args.degree)
    exact,forcing=sampler(X);boundary=exact.clone();boundary[sl]=0
    rhs=forcing[sl].reshape(-1)-A.full(boundary)
    torch.cuda.synchronize();setup=time.perf_counter()-start
    checks={}
    if args.n<=33:
        rng=np.random.default_rng(128);probe=rng.normal(size=(args.n,)*3)
        ref=A.cpu_full(probe);actual=A.full(gpu(probe)).cpu().numpy()
        checks['cpu_gpu_operator_relative_error']=float(np.linalg.norm(ref-actual)/np.linalg.norm(ref))
        print(checks,flush=True)
        assert checks['cpu_gpu_operator_relative_error']<1e-10
        # Verify cached extension against the actual upstream public API.
        v=gpu(probe[0]);upstream=A.fc.extend(v,dim=1)
        checks['upstream_extension_max_error']=(upstream-v@A.E.T).abs().max().item()
        assert checks['upstream_extension_max_error']<1e-12
        # The preconditioner should invert the separable mean-metric FC operator.
        r=gpu(probe[1:-1,1:-1,1:-1]).reshape(-1);z=A.precondition(r)
        full=torch.zeros_like(exact);full[sl]=z.reshape((args.n-2,)*3)
        mean=A.metric.mean(dim=(0,1,2));refinv=sum(-mean[i]*A.derivative(full,2-i,2)[sl] for i in range(3)).reshape(-1)
        checks['fp32_reference_inverse_residual']=(torch.linalg.vector_norm(refinv-r)/torch.linalg.vector_norm(r)).item()
        assert checks['fp32_reference_inverse_residual']<1e-4
    runs=[]
    for run in range(args.repeat+1):
        torch.cuda.synchronize();wall_start=time.perf_counter()
        boundary=exact.clone();boundary[sl]=0
        rhs=forcing[sl].reshape(-1)-A.full(boundary)
        torch.cuda.synchronize();start=time.perf_counter()
        interior,info=fgmres(A,rhs,rtol=args.rtol)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        solution=boundary.clone();solution[sl]=interior.reshape((args.n-2,)*3)
        torch.cuda.synchronize();wall_elapsed=time.perf_counter()-wall_start
        runs.append(dict(solve_s=elapsed,wall_s=wall_elapsed,**info));print(json.dumps(dict(run=run,**runs[-1])),flush=True)
    mask=torch.ones_like(solution,dtype=torch.bool);mask[sl]=False
    checks['boundary_max_error']=(solution[mask]-exact[mask]).abs().max().item()
    assert checks['boundary_max_error']==0
    nodal=(torch.linalg.vector_norm(solution-exact)/torch.linalg.vector_norm(exact)).item()
    result=dict(n=args.n,unique_dofs=args.n**3,interior_dofs=(args.n-2)**3,field=args.field,degree=args.degree,
        padding_per_side=25,setup_s=setup,warm_solve_s=float(np.median([r['solve_s'] for r in runs[1:]])),
        warm_wall_s=float(np.median([r['wall_s'] for r in runs[1:]])),
        nodal_relative_l2=nodal,runs=runs,checks=checks,eigenvector_condition=A.eigenvector_condition,
        eigenpair_defect=A.eigenpair_defect,device=torch.cuda.get_device_name(),torch_version=torch.__version__,
        outer_precision='float64',preconditioner_precision='complex64 (FP32 real/imag)',krylov='FGMRES(40)',
        data_sha256=hashlib.sha256(args.modes.read_bytes()).hexdigest())
    if args.audit:
        q=max(96,int(np.ceil(1.5*args.n)));errors=[]
        for order in (q,q+8):
            error=audit(A,solution,sampler,box,order);errors.append(error)
            print(json.dumps(dict(audit_q=order,relative_l2=error)),flush=True)
        orders=[q,q+8]
        while (sampler.features.active and args.field=='turbulence' and
               abs(errors[-1]-errors[-2])>1e-10+1e-3*errors[-1]):
            order=orders[-1]+max(16,orders[-1]//4)
            if order>512:raise RuntimeError(f'Independent error quadrature unresolved: {errors}')
            error=audit(A,solution,sampler,box,order);errors.append(error);orders.append(order)
            print(json.dumps(dict(audit_q=order,relative_l2=error)),flush=True)
        result.update(relative_l2=errors[-1],audit_orders=orders,audit_errors=errors,
                      audit_relative_change=abs(errors[-1]-errors[-2])/max(errors[-1],1e-30))
        result['audit_absolute_change']=abs(errors[-1]-errors[-2])
        if abs(errors[-1]-errors[-2])>1e-10+1e-3*errors[-1]:raise RuntimeError('Independent error quadrature unresolved')
    result['peak_torch_allocated_bytes']=torch.cuda.max_memory_allocated()
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(args.out/'solution.npz',u=solution.cpu().numpy())
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
