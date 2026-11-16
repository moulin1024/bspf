"""One native FD4 solve per grid; independent off-grid physical L2 audit."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['MPLCONFIGDIR']='/private/tmp/fd4-mpl'
import sys,importlib.util,json,time,subprocess,hashlib,argparse
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('opt_run',ROOT/'benchmarks/nonconformal_bspf_opt/run.py');opt=importlib.util.module_from_spec(spec);spec.loader.exec_module(opt);opt.configure_hard();r=opt.r
import numpy as np
from scipy.sparse import csr_matrix
OUT=ROOT/'evidence/finite_difference_poisson/v1';OUT.mkdir(parents=True,exist_ok=True)
def truth(p,kind):
 x,y=p[...,0],p[...,1]
 if kind=='guard':return 1+x*x+y*y,np.full_like(x,-4.)
 if kind=='smooth':
  u=np.sin(3*x+.3)*np.cos(4*y-.2)+.2*x+.1*y
  return u,25*np.sin(3*x+.3)*np.cos(4*y-.2)
 v=r.truth(p);return v[0].reshape(x.shape),v[3].reshape(x.shape)
def interpolation(n,z,degree=7):
 count=degree+1;start=np.clip(np.floor(z*(n-1)).astype(int)-degree//2,0,n-count);ids=start[:,None]+np.arange(count);xx=z[:,None]*(n-1)-ids
 weights=np.ones_like(xx)
 for k in range(count):
  for j in range(count):
   if j!=k:weights[:,k]*=xx[:,j]/(k-j)
 return csr_matrix((weights.ravel(),(np.repeat(np.arange(len(z)),count),ids.ravel())),shape=(len(z),n))
def run(n,kind):
 folder=OUT/f'{kind}_n{n}';folder.mkdir(exist_ok=True);path=folder/'result.json'
 if path.exists():print(path.read_text(),flush=True);return
 row=dict(n=n,kind=kind,id=folder.name,status='preparing',method='native_fd4');r.dump(path,row)
 begin=time.perf_counter();z=np.linspace(0,1,n);p,xs,xt,J,xss,xst,xtt=r.PATCH.jet(z[:,None],z[None,:]);A=np.sum(xt*xt,axis=-1)/J**2;C=np.sum(xs*xs,axis=-1)/J**2;B=-np.sum(xs*xt,axis=-1)/J**2;rv=A[...,None]*xss+2*B[...,None]*xst+C[...,None]*xtt;G=(-xt[...,1]*rv[...,0]+xt[...,0]*rv[...,1])/J;H=(xs[...,1]*rv[...,0]-xs[...,0]*rv[...,1])/J
 row['geometry_setup_seconds']=time.perf_counter()-begin;begin=time.perf_counter();u,f=truth(p,kind);g=np.zeros_like(u);g[0]=u[0];g[-1]=u[-1];g[:,0]=u[:,0];g[:,-1]=u[:,-1];row['sampling_seconds']=time.perf_counter()-begin;row['forcing_samples']=(n-2)**2
 r.write_bundle(folder/'input.bin',dict(A=A,B=B,C=C,G=G,H=H,forcing=f,boundary=g));exe=HERE/'fd4';row['binary_sha256']=hashlib.sha256(exe.read_bytes()).hexdigest();row['source_sha256']=hashlib.sha256((HERE/'solver.cpp').read_bytes()).hexdigest()
 proc=subprocess.run([str(exe),str(folder/'input.bin'),str(folder/'solution')],capture_output=True,text=True);(folder/'native.log').write_text(proc.stdout+proc.stderr)
 if proc.returncode:row.update(status='failed',failure=proc.stderr);r.dump(path,row);print(row,flush=True);return
 row.update(json.loads((folder/'solution.json').read_text()));sol=r.read_bundle(folder/'solution.bin')['solution'];row['nodal_max_error']=float(np.max(abs(sol-u)))
 errors=[];orders=(640,768) if kind=='hard' else (100,140)
 for q in orders:
  zz,w=r.roots_legendre(q);zz=(zz+1)/2;w=w/2
  if kind=='hard':
   cachepath=opt.OUT/f'exact_global_q{q}.npz'
   if not cachepath.exists():
    pp,_,_,jj,*_=r.PATCH.jet(zz[:,None],zz[None,:]);np.savez(cachepath,u=truth(pp,kind)[0],j=jj)
   cache=np.load(cachepath);exact=cache['u'];jj=cache['j']
  else:pp,_,_,jj,*_=r.PATCH.jet(zz[:,None],zz[None,:]);exact=truth(pp,kind)[0]
  I=interpolation(n,zz);interp=(I@(I@sol).T).T;ww=np.outer(w,w)*jj;errors.append(float(np.sqrt(np.sum(ww*(interp-exact)**2)/np.sum(ww*exact**2))))
 row.update(l2_relative=errors[-1],quadrature_errors=errors,quadrature_orders=orders,quadrature_stable=abs(errors[0]-errors[1])<.01*errors[1]+1e-11)
 row['status']='validated' if row['converged'] and row['relative_residual']<2e-12 and row['quadrature_stable'] else 'audit_failed';r.dump(path,row);print(json.dumps(row),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['smooth','guard','hard'],required=True);ap.add_argument('--n',nargs='+',type=int,required=True);args=ap.parse_args()
 for n in args.n:run(n,args.kind)
